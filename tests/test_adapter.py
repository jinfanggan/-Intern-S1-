"""适配层单元测试：用假官方 client（同步 .chat）验证契约，不打网络。"""
import asyncio

import pytest

from jiushao.adapter import OfficialClientLLM, _events_to_trace, solve_problem


class FakeOfficialClient:
    """模拟官方 client：同步 chat(messages, temperature, max_tokens) -> str。"""

    def __init__(self, scripts):
        self.scripts = scripts
        self.n = 0
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({"messages": messages, "temperature": temperature})
        out = self.scripts[min(self.n, len(self.scripts) - 1)]
        self.n += 1
        return out


@pytest.mark.asyncio
async def test_official_llm_wraps_sync_client():
    c = FakeOfficialClient([r"答案 \boxed{42}"])
    llm = OfficialClientLLM(c)
    r = await llm.chat([{"role": "user", "content": "q"}], temperature=0.6)
    assert r["content"] == r"答案 \boxed{42}" and not r["cached"]
    assert c.calls[0]["temperature"] == 0.6


@pytest.mark.asyncio
async def test_official_llm_memory_cache():
    c = FakeOfficialClient([r"\boxed{1}"])
    llm = OfficialClientLLM(c)
    msgs = [{"role": "user", "content": "q"}]
    await llm.chat(msgs, cache_tag="t")
    r2 = await llm.chat(msgs, cache_tag="t")
    assert r2["cached"] and c.n == 1  # 第二次命中内存缓存，不再调 client


def test_events_to_trace_serializable():
    import json
    events = [{"stage": "solve", "type": "chain_done", "ts": 1.0,
               "chain": 0, "answer": "42"}]
    trace = _events_to_trace(events)
    assert trace[0]["step"] == "solve:chain_done"
    assert "ts" not in trace[0]["content"]
    json.dumps(trace)  # 必须可序列化


@pytest.mark.asyncio
async def test_solve_problem_returns_contract():
    # 单轮直接给 boxed 答案，无代码块 → 每链一次调用
    c = FakeOfficialClient([r"推理过程……所以 \boxed{72}"])
    r = await solve_problem(c, "求解 X", n_chains=3, max_rounds=4)
    assert isinstance(r["final_response"], str) and r["final_response"]
    assert r"\boxed{72}" in r["final_response"]
    assert isinstance(r["trace"], list) and r["trace"]
    import json
    json.dumps(r)  # 整体可 JSON 序列化（官方要求）


@pytest.mark.asyncio
async def test_solve_problem_voting_picks_majority():
    # 3 链：两链 42、一链 99 → 投票选 42
    c = FakeOfficialClient([r"\boxed{42}", r"\boxed{42}", r"\boxed{99}"])
    # FakeClient 按调用顺序返回，3 条链各 1 次调用
    r = await solve_problem(c, "q", n_chains=3, max_rounds=2)
    assert r"\boxed{42}" in r["final_response"]


@pytest.mark.asyncio
async def test_solve_problem_all_chains_fail():
    class BrokenClient:
        def chat(self, *a, **k):
            raise RuntimeError("API down")
    r = await solve_problem(BrokenClient(), "q", n_chains=2)
    assert r["final_response"] == "" and isinstance(r["trace"], list)


def test_user_agent_contract():
    """直接验证提交入口 ReasoningAgent 的契约。"""
    import importlib.util
    import pathlib
    ua_path = pathlib.Path(__file__).parent.parent / "submission" / "user_agent.py"
    spec = importlib.util.spec_from_file_location("user_agent", ua_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    c = FakeOfficialClient([r"\boxed{5}"])
    agent = mod.ReasoningAgent(client=c, n_chains=2, max_rounds=2)
    out = agent.solve("1+4=?", {"idx": 0})
    assert isinstance(out, dict)
    assert isinstance(out["final_response"], str) and out["final_response"]
    import json
    json.dumps(out)


class TestOracleFilter:
    """数值裁判过滤（创新点一接入 solve_problem）。"""

    ORACLE = {"points": [{"param": 0.001, "value": 195.7}]}

    def test_filters_rejected_candidates(self):
        from jiushao.adapter import oracle_filter
        ok = [{"answer": r"\frac{0.67}{\epsilon^{0.75}}"},  # 过点
              {"answer": r"\epsilon^{2}"}]                    # 不过点
        survivors = oracle_filter(ok, self.ORACLE)
        assert survivors is not None and len(survivors) == 1
        assert survivors[0]["answer"] == r"\frac{0.67}{\epsilon^{0.75}}"

    def test_unreliable_oracle_returns_none(self):
        from jiushao.adapter import oracle_filter
        # 两个候选都不过点 → 裁判判全 0 → 视为不适用
        ok = [{"answer": r"\epsilon^{2}"}, {"answer": r"\epsilon^{3}"}]
        assert oracle_filter(ok, self.ORACLE) is None

    def test_all_pass_keeps_all(self):
        from jiushao.adapter import oracle_filter
        ok = [{"answer": r"\frac{0.67}{\epsilon^{0.75}}"},
              {"answer": r"\frac{1.0}{\epsilon^{0.75}}"}]  # 都在因子窗口内
        survivors = oracle_filter(ok, self.ORACLE)
        assert survivors is not None and len(survivors) == 2

    @pytest.mark.asyncio
    async def test_use_oracle_false_skips(self):
        # use_oracle=False → 不构造裁判，纯投票
        c = FakeOfficialClient([r"\boxed{42}"])
        r = await solve_problem(c, "q", n_chains=3, max_rounds=2, use_oracle=False)
        steps = [t["step"] for t in r["trace"]]
        assert not any(s.startswith("oracle") for s in steps)
