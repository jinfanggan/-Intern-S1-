"""创新点二：歧义分叉求解（Interpretation Forking）。

把 Self-Consistency 从「路径层」提升到「解读层」。

实证动机（来自本项目调试日志）：
  "divide 8 elements into 5 non-empty ordered subsets" 被模型理解为
  「5 个带标号的桶」（算得满射数 126000，计算无误），而题意是
  「内部有序的子集」（Lah 数 L(8,5)=11760）。计算全对、审题错了——
  且这种错误采样再多链也救不回（所有链共享同一误读，误差完全相关）。

机制：先派审题智能体枚举合法解读 → 各解读独立分叉求解投票 →
      仲裁智能体结合答案选最符合题意的解读。
"""
import json
import re

from .aggregate import cluster_and_vote
from .solver import solve_tir

ENUM_PROMPT = (
    "你是数学题审题专家。下面的数学题可能存在多种合理解读"
    "（常见于术语歧义、记号歧义、中英文翻译歧义，如 ordered/unordered、"
    "集合/多重集、开区间/闭区间）。\n"
    "请列出所有【合理且互不等价】的解读；若题意明确无歧义，只列 1 种。\n"
    "最多列 3 种。只输出 JSON，格式：\n"
    '{"interpretations": [{"reading": "对题意的一句话明确表述", '
    '"note": "歧义点说明"}]}'
)

ARBITRATE_PROMPT = (
    "你是数学题仲裁专家。同一道题在不同解读下得到了不同答案。"
    "请判断哪一种解读最符合该题的【标准数学含义与惯例】。\n"
    "只输出 JSON：{\"best\": 解读编号(从0开始), \"reason\": \"一句话理由\"}"
)


def _parse_json(text: str) -> dict | None:
    """从模型输出中抽第一个 JSON 对象。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def enumerate_interpretations(llm, problem: str, *, emit=None) -> list[dict]:
    """枚举题面的合法解读。返回 [{reading, note}, ...]，至少 1 种。"""
    resp = await llm.chat(
        [{"role": "system", "content": ENUM_PROMPT},
         {"role": "user", "content": problem}],
        temperature=0.0, cache_tag="enum-interp")
    data = _parse_json(resp["content"])
    interps = (data or {}).get("interpretations") or []
    interps = [i for i in interps if isinstance(i, dict) and i.get("reading")][:3]
    if emit:
        emit("fork", "enumerate", count=len(interps),
             readings=[i["reading"][:80] for i in interps])
    return interps or [{"reading": "（题意明确，按字面理解）", "note": ""}]


async def arbitrate(llm, problem: str, candidates: list[dict], *, emit=None) -> int:
    """在各解读的答案间仲裁，返回最佳解读下标。"""
    listing = "\n".join(
        f"解读 {i}：{c['reading']}\n  → 该解读下的答案：{c['answer']}"
        for i, c in enumerate(candidates))
    resp = await llm.chat(
        [{"role": "system", "content": ARBITRATE_PROMPT},
         {"role": "user", "content": f"原题：\n{problem}\n\n{listing}"}],
        temperature=0.0, cache_tag="arbitrate")
    data = _parse_json(resp["content"])
    best = (data or {}).get("best", 0)
    try:
        best = int(best)
    except (TypeError, ValueError):
        best = 0
    best = best if 0 <= best < len(candidates) else 0
    if emit:
        emit("fork", "arbitrate", best=best,
             reason=str((data or {}).get("reason", ""))[:120])
    return best


async def solve_with_forking(llm, problem: str, *, n_chains: int = 3,
                             max_rounds: int = 4, temperature: float = 0.6,
                             emit=None) -> dict:
    """歧义分叉求解。返回 {answer, transcript, interpretation, n_interp}。

    - 枚举解读；单解读 → 退化为普通多链 TIR + 投票
    - 多解读 → 每解读独立分叉求解投票 → 仲裁选最佳解读的答案
    """
    import asyncio

    interps = await enumerate_interpretations(llm, problem, emit=emit)

    async def solve_one(interp_id: int, reading: str) -> dict:
        # 单解读时不加解读前缀，避免干扰；多解读时显式注入该解读
        q = problem if len(interps) == 1 else (
            f"{problem}\n\n【请严格按以下解读求解】{reading}")
        chains = await asyncio.gather(*[
            solve_tir(llm, q, chain_id=interp_id * 100 + i,
                      max_rounds=max_rounds, temperature=temperature, emit=emit)
            for i in range(n_chains)
        ], return_exceptions=True)
        ok = [c for c in chains if isinstance(c, dict)]
        agg = cluster_and_vote([c["answer"] for c in ok]) if ok else {"final": None, "clusters": []}
        rep = next((c for c in ok if c["answer"] == agg["final"]), ok[0] if ok else None)
        return {"reading": reading, "answer": agg["final"],
                "clusters": agg["clusters"],
                "transcript": rep["transcript"] if rep else ""}

    cand = await asyncio.gather(*[
        solve_one(i, itp["reading"]) for i, itp in enumerate(interps)])

    if len(cand) == 1:
        c = cand[0]
        return {"answer": c["answer"], "transcript": c["transcript"],
                "interpretation": c["reading"], "n_interp": 1}

    best = await arbitrate(llm, problem, cand, emit=emit)
    c = cand[best]
    if emit:
        emit("fork", "select", interpretation=c["reading"][:80], answer=c["answer"])
    return {"answer": c["answer"], "transcript": c["transcript"],
            "interpretation": c["reading"], "n_interp": len(cand)}
