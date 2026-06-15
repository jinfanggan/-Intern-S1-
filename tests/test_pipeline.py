"""pipeline 单元测试：单题全流程（FakeLLM，不打网络）。"""
import pytest

from jiushao.dsets import Problem
from jiushao.pipeline import run_problem


def _problem():
    return Problem(id="t-1", question="1+1=?", answer="2", subject="算术")


@pytest.mark.asyncio
async def test_b0_correct(fake_llm_factory, collect_events):
    emit, events = collect_events
    llm = fake_llm_factory([r"显然 \boxed{2}"])
    r = await run_problem(llm, _problem(), "b0", emit)
    assert r["correct"] is True and r["pred"] == "2"
    stages = {e["stage"] for e in events}
    assert {"input", "solve", "aggregate", "verdict"} <= stages


@pytest.mark.asyncio
async def test_b0_wrong(fake_llm_factory, collect_events):
    emit, _ = collect_events
    llm = fake_llm_factory([r"\boxed{3}"])
    r = await run_problem(llm, _problem(), "b0", emit)
    assert r["correct"] is False


@pytest.mark.asyncio
async def test_b2_vote_event(fake_llm_factory, collect_events):
    emit, events = collect_events
    llm = fake_llm_factory([r"\boxed{2}"])  # 所有链同答案 → 1 簇 8 票
    r = await run_problem(llm, _problem(), "b2", emit)
    assert r["correct"] is True
    vote = [e for e in events if e["type"] == "vote"]
    assert len(vote) == 1 and vote[0]["clusters"][0][1] == 8


@pytest.mark.asyncio
async def test_chain_exception_tolerated(fake_llm_factory, collect_events):
    """单链异常不应拖垮整题（其余链照常投票）。"""
    emit, events = collect_events
    llm = fake_llm_factory([r"\boxed{2}"])
    fail_first = {"n": 0}
    orig = llm.chat

    async def flaky(messages, **kw):
        fail_first["n"] += 1
        if fail_first["n"] == 1:
            raise RuntimeError("模拟 API 故障")
        return await orig(messages, **kw)

    llm.chat = flaky
    r = await run_problem(llm, _problem(), "b2", emit)
    assert r["correct"] is True
    assert r["n_chains_ok"] == 7  # 8 链坏 1
    assert any(e["type"] == "chain_error" for e in events)


class TestSolverRegistry:
    """工程化：求解器注册表 + profile 驱动的消融。"""

    def test_registry_has_builtin_solvers(self):
        from jiushao.solver import SOLVERS
        import jiushao.oracle_loop  # noqa: F401  触发注册
        assert {"cot", "tir", "oracle-tir"} <= set(SOLVERS)

    def test_register_custom_solver(self):
        from jiushao.solver import SOLVERS, register_solver

        async def dummy(llm, q, *, chain_id=0, temperature=None, emit=None):
            return {"answer": "42"}

        register_solver("dummy", dummy)
        assert SOLVERS["dummy"] is dummy

    @pytest.mark.asyncio
    async def test_profile_drives_solver_choice(self, fake_llm_factory, collect_events):
        """新增对比实验只需 PROFILES 加一行——这里临时注入一个 profile 验证。"""
        from jiushao.config import PROFILES
        from jiushao.solver import register_solver

        async def const(llm, q, *, chain_id=0, temperature=None, emit=None):
            return {"answer": "2"}

        register_solver("const", const)
        PROFILES["_test_const"] = {"solver": "const", "n_chains": 1, "solver_kwargs": {}}
        emit, _ = collect_events
        r = await run_problem(fake_llm_factory(["x"]), _problem(), "_test_const", emit)
        assert r["pred"] == "2" and r["correct"]
        del PROFILES["_test_const"]
