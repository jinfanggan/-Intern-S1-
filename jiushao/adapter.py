"""官方评测契约适配层。

把官方 client（同步 `client.chat(messages, temperature, max_tokens) -> str`）
包装成本引擎求解链所需的接口，并把内部事件流转成官方要求的 trace 格式。

提交时由 user_agent.py 的 ReasoningAgent 调用本模块——内部完整复用
solver（TIR 多链）/ aggregate（等价归一投票）/ sandbox（代码执行），
正式评测零改动。
"""
import asyncio
import hashlib

from .aggregate import cluster_and_vote
from .oracle import construct_oracle, score_candidate
from .solver import solve_tir


class OfficialClientLLM:
    """鸭子类型兼容 jiushao.llm.LLM 的 chat 接口，但底层走官方同步 client。

    - 同步 → 异步：用 asyncio.to_thread 包官方 client.chat
    - 缓存：仅内存（提交环境不写磁盘，避免只读文件系统/污染）
    - usage：官方 client 不回 token 数，置 0（评测不需要成本统计）
    """

    def __init__(self, client, *, default_max_tokens: int = 4096,
                 default_temperature: float = 0.6):
        self.client = client
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self._cache: dict[str, dict] = {}
        self.usage = {"in": 0, "out": 0, "calls": 0, "cache_hits": 0}

    async def chat(self, messages, *, temperature=None, max_tokens=None,
                   cache_tag: str = "", retries: int = 3) -> dict:
        key = hashlib.sha256(
            f"{cache_tag}|{temperature}|{messages}".encode()).hexdigest()[:24]
        if key in self._cache:
            self.usage["cache_hits"] += 1
            return {**self._cache[key], "cached": True}
        temp = self.default_temperature if temperature is None else temperature
        mt = max_tokens or self.default_max_tokens
        # 官方 client 自带 retry；这里直接调用，异常向上抛由 solve 兜底
        content = await asyncio.to_thread(self.client.chat, messages, temp, mt)
        self.usage["calls"] += 1
        data = {"content": content or "", "usage_in": 0, "usage_out": 0}
        self._cache[key] = data
        return {**data, "cached": False}

    def cost_usd(self) -> float:
        return 0.0


def _events_to_trace(events: list[dict]) -> list[dict]:
    """内部事件流 → 官方 trace 格式 [{step, content}]，全部可 JSON 序列化。"""
    trace = []
    for e in events:
        step = f"{e['stage']}:{e['type']}"
        content = {k: v for k, v in e.items()
                   if k not in ("stage", "type", "ts")}
        # 保证可序列化（值已是 str/num/bool/list/dict）
        trace.append({"step": step, "content": content})
    return trace


def oracle_filter(ok_chains: list[dict], oracle: dict) -> list[dict] | None:
    """用数值裁判过滤候选（创新点一的 best-of-N 选择形态）。

    - 对每个候选解析解打分（在裁判采样点上的数值吻合度）
    - 剔除被裁判明确否决（score==0）的候选，保留 score>0 者
    - 适用性自检：若所有候选 score 全为 0，视为裁判不适用于本题
      （符号/证明题，或裁判构造失真）→ 返回 None，调用方退回纯投票
    """
    scored = [(c, score_candidate(c["answer"] or "", oracle)["score"])
              for c in ok_chains]
    if not any(s > 0 for _, s in scored):
        return None                       # 裁判不可靠，不参与决策
    return [c for c, s in scored if s > 0]


async def solve_problem(client, problem: str, *, n_chains: int = 3,
                        max_rounds: int = 4, temperature: float = 0.6,
                        use_oracle: bool = True) -> dict:
    """多候选 TIR + 数值裁判过滤 + 等价归一投票。

    相对官方 baseline 的差异化：baseline 用 LLM 自验（VERDICT: A/B，弱），
    我们用程序构造的数值裁判过滤候选（强、接地）。裁判不适用的题
    （符号/证明）自动退回纯投票，零副作用。

    返回 {final_response, trace}，final_response 为选中簇的完整解答
    （含 \\boxed{} 终答，交官方 judger 提取判分）。
    """
    llm = OfficialClientLLM(client, default_temperature=temperature)
    events: list[dict] = []

    def emit(stage, etype, **data):
        events.append({"stage": stage, "type": etype, **data})

    emit("input", "problem", problem=problem, n_chains=n_chains)

    # N 条 TIR 链并行（单链异常降级为 None，不拖垮整题）
    chains = await asyncio.gather(*[
        solve_tir(llm, problem, chain_id=i, max_rounds=max_rounds,
                  temperature=temperature, emit=emit)
        for i in range(n_chains)
    ], return_exceptions=True)
    ok = [c for c in chains if isinstance(c, dict)]
    for c in chains:
        if isinstance(c, Exception):
            emit("solve", "chain_error", error=repr(c)[:300])

    if not ok:
        return {"final_response": "", "trace": _events_to_trace(events)}

    # 数值裁判过滤（可用时）：剔除被裁判数值否决的候选
    pool = ok
    if use_oracle and len(ok) > 1:
        oracle = await construct_oracle(llm, problem, emit=emit)
        if oracle:
            survivors = oracle_filter(ok, oracle)
            if survivors is None:
                emit("oracle", "skip", reason="裁判不适用（全部候选未通过，疑似符号/证明题）")
            elif len(survivors) < len(ok):
                emit("oracle", "filter", kept=len(survivors), total=len(ok))
                pool = survivors
            else:
                emit("oracle", "all_pass", total=len(ok))

    # 等价归一投票，选最大簇；final_response 用该簇代表链的完整解答
    agg = cluster_and_vote([c["answer"] for c in pool])
    emit("aggregate", "vote", clusters=agg["clusters"], final=agg["final"])

    final_chain = next(
        (c for c in pool if c["answer"] == agg["final"]), pool[0])
    final_response = final_chain.get("transcript") or (agg["final"] or "")
    emit("select", "final", answer=agg["final"])

    return {"final_response": final_response,
            "trace": _events_to_trace(events)}
