"""solver 单元测试：用 FakeLLM 验证 CoT 与 TIR 状态机的关键路径。"""
import pytest

from jiushao.solver import solve_cot, solve_tir


@pytest.mark.asyncio
async def test_cot_extracts_boxed(fake_llm_factory, collect_events):
    emit, events = collect_events
    llm = fake_llm_factory([r"思路……所以 \boxed{42}"])
    out = await solve_cot(llm, "q", emit=emit)
    assert out["answer"] == "42" and out["rounds"] == 1
    assert any(e["type"] == "chain_done" for e in events)


@pytest.mark.asyncio
async def test_tir_stops_on_boxed_without_code(fake_llm_factory):
    llm = fake_llm_factory([r"无需计算，\boxed{7}"])
    out = await solve_tir(llm, "q", max_rounds=4)
    assert out["answer"] == "7" and out["rounds"] == 1 and llm.n_calls == 1


@pytest.mark.asyncio
async def test_tir_code_loop_then_answer(fake_llm_factory, collect_events):
    emit, events = collect_events
    llm = fake_llm_factory([
        "先算积分：\n```python\nimport sympy as sp\nx = sp.Symbol('x')\n"
        "print(sp.integrate(x**2, (x, 0, 1)))\n```",
        r"执行结果是 1/3，所以 \boxed{\frac{1}{3}}",
    ])
    out = await solve_tir(llm, "q", max_rounds=4, emit=emit)
    assert out["answer"] == r"\frac{1}{3}"
    assert llm.n_calls == 2
    sandbox_events = [e for e in events if e["stage"] == "sandbox"]
    assert len(sandbox_events) == 1 and sandbox_events[0]["ok"]


@pytest.mark.asyncio
async def test_tir_recovers_from_code_error(fake_llm_factory, collect_events):
    emit, events = collect_events
    llm = fake_llm_factory([
        "```python\nprint(undefined_var)\n```",
        r"修正后口算得 \boxed{9}",
    ])
    out = await solve_tir(llm, "q", max_rounds=4, emit=emit)
    assert out["answer"] == "9"
    err_events = [e for e in events if e["stage"] == "sandbox"]
    assert len(err_events) == 1 and not err_events[0]["ok"]


@pytest.mark.asyncio
async def test_tir_rounds_exhausted_forces_final(fake_llm_factory):
    # 永远只给代码不给答案 → 轮数耗尽后强制催终答
    llm = fake_llm_factory([
        "```python\nprint(1)\n```",
        "```python\nprint(2)\n```",
        r"\boxed{3}",
    ])
    out = await solve_tir(llm, "q", max_rounds=2)
    assert out["answer"] == "3"
    assert llm.n_calls == 3  # 2 轮 + 1 次强制终答


@pytest.mark.asyncio
async def test_chain_angle_varies_prompt(fake_llm_factory):
    llm = fake_llm_factory([r"\boxed{1}"])
    captured = []
    orig = llm.chat

    async def spy(messages, **kw):
        captured.append(messages[0]["content"])
        return await orig(messages, **kw)

    llm.chat = spy
    await solve_cot(llm, "q", chain_id=0)
    await solve_cot(llm, "q", chain_id=3)
    assert captured[0] != captured[1]  # 不同链注入不同切入角
