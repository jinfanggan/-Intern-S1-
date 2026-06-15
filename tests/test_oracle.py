"""数值神谕单元测试（FakeLLM，不打网络）。"""
import pytest

from jiushao.oracle import _parse_oracle_output, construct_oracle, score_candidate


class TestParseOracleOutput:
    def test_valid_json_last_line(self):
        out = _parse_oracle_output('调试信息\n{"points": [{"param": 0.001, "value": 195.7}]}')
        assert out["points"][0]["value"] == 195.7

    def test_garbage_returns_none(self):
        assert _parse_oracle_output("没有 json") is None


@pytest.mark.asyncio
async def test_construct_oracle_end_to_end(fake_llm_factory):
    """FakeLLM 给出数值积分代码 → 沙箱真实执行 → 神谕真值。"""
    llm = fake_llm_factory(["""写代码数值求解：
```python
from scipy.integrate import quad
import json
pts = []
for eps in [0.001, 1e6]:
    val, _ = quad(lambda x: 1.0/(eps + x**4), 0, 98)
    pts.append({"param": eps, "value": val})
print(json.dumps({"points": pts}))
```"""])
    oracle = await construct_oracle(llm, "积分题")
    assert oracle is not None and len(oracle["points"]) == 2
    assert all(p["value"] > 0 for p in oracle["points"])


@pytest.mark.asyncio
async def test_construct_oracle_impossible(fake_llm_factory):
    llm = fake_llm_factory(['```python\nimport json\nprint(json.dumps({"impossible": "证明题"}))\n```'])
    assert await construct_oracle(llm, "证明题") is None


@pytest.mark.asyncio
async def test_construct_oracle_nan_filtered(fake_llm_factory):
    llm = fake_llm_factory(['```python\nimport json\nprint(json.dumps({"points": [{"param": 1, "value": float("nan")}]}))\n```'])
    assert await construct_oracle(llm, "x") is None


class TestScoreCandidate:
    ORACLE = {"points": [{"param": 0.001, "value": 195.7},
                         {"param": 739072203.35, "value": 4.45e-09}]}

    def test_gold_regime_list_passes(self):
        # HARDMath 真题 gold：两 regime 渐近式，各点应由对应元素命中
        s = score_candidate(r"[\frac{1.0}{\epsilon^{0.93}}, \frac{0.67}{\epsilon^{0.75}}]",
                            self.ORACLE)
        assert s["score"] == 1.0

    def test_wrong_candidate_fails(self):
        s = score_candidate(r"\epsilon^{2}", self.ORACLE)
        assert s["score"] == 0.0

    def test_partial_credit(self):
        # 只对 small regime 正确的候选 → 0.5
        s = score_candidate(r"\frac{0.67}{\epsilon^{0.75}}", self.ORACLE)
        assert s["score"] == 0.5

    def test_scalar_no_param(self):
        s = score_candidate("42", {"points": [{"param": None, "value": 42.0}]})
        assert s["score"] == 1.0

    def test_empty_candidate(self):
        s = score_candidate("", self.ORACLE)
        assert s["score"] == 0.0


class TestOracleLoop:
    """神谕反馈闭环（FakeLLM）。"""

    ORACLE = {"points": [{"param": 0.001, "value": 195.7}]}

    @pytest.mark.asyncio
    async def test_passes_first_try(self, fake_llm_factory):
        from jiushao.oracle_loop import solve_with_oracle
        llm = fake_llm_factory([r"\boxed{\frac{0.67}{\epsilon^{0.75}}}"])
        r = await solve_with_oracle(llm, "q", oracle=self.ORACLE)
        assert r["verified"] and r["revisions"] == 0

    @pytest.mark.asyncio
    async def test_revises_after_feedback(self, fake_llm_factory):
        from jiushao.oracle_loop import solve_with_oracle
        llm = fake_llm_factory([
            r"\boxed{\epsilon^{2}}",                      # 第一次：错误数量级
            r"\boxed{\frac{0.67}{\epsilon^{0.75}}}",      # 收到数值反馈后修正
        ])
        r = await solve_with_oracle(llm, "q", oracle=self.ORACLE)
        assert r["verified"] and r["revisions"] == 1
        assert llm.n_calls == 2

    @pytest.mark.asyncio
    async def test_gives_best_after_exhaustion(self, fake_llm_factory):
        from jiushao.oracle_loop import solve_with_oracle
        llm = fake_llm_factory([r"\boxed{\epsilon^{2}}"])  # 永远错
        r = await solve_with_oracle(llm, "q", oracle=self.ORACLE, max_revisions=1)
        assert not r["verified"] and r["score"] == 0.0 and r["revisions"] == 1

    @pytest.mark.asyncio
    async def test_no_oracle_degrades_to_tir(self, fake_llm_factory):
        from jiushao.oracle_loop import solve_with_oracle
        # construct_oracle 拿到的回复没有代码块 → 构造失败 → 退化普通 TIR
        llm = fake_llm_factory([r"无法构造", r"\boxed{42}"])
        r = await solve_with_oracle(llm, "q", oracle=None)
        assert not r["oracle_ok"] and r["answer"] == "42"

    def test_feedback_message_contains_numbers(self):
        from jiushao.oracle_loop import _build_feedback
        s = score_candidate(r"\epsilon^{2}", self.ORACLE)
        msg = _build_feedback(r"\epsilon^{2}", s, revealed=[0])
        assert "195.7" in msg and "真值" in msg


class TestAntiGaming:
    """防验证器博弈：留出点 + 常数溯源。"""

    ORACLE4 = {"points": [
        {"param": 0.001, "value": 100.0}, {"param": 0.005, "value": 50.0},
        {"param": 1e6, "value": 1e-6}, {"param": 5e6, "value": 2e-7}]}

    def test_split_alternates(self):
        from jiushao.oracle_loop import split_points
        revealed, held = split_points(self.ORACLE4)
        assert len(revealed) == 2 and len(held) == 2
        assert set(revealed) | set(held) == {0, 1, 2, 3}

    def test_single_point_no_split(self):
        from jiushao.oracle_loop import split_points
        revealed, held = split_points({"points": [{"param": 1, "value": 2.0}]})
        assert revealed == [0] and held == []

    def test_feedback_hides_heldout_truth(self):
        from jiushao.oracle_loop import _build_feedback, split_points
        s = score_candidate(r"\epsilon", self.ORACLE4)
        revealed, held = split_points(self.ORACLE4)
        msg = _build_feedback("x", s, revealed)
        for i in held:
            truth = f'{self.ORACLE4["points"][i]["value"]:.6g}'
            assert truth not in msg, f"留出点真值 {truth} 泄漏进反馈"
        assert "未公开的校验点" in msg

    def test_provenance_traceable(self):
        from jiushao.oracle_loop import check_provenance
        transcript = "[assistant r0]\n算\n[sandbox r0]\n执行成功，输出：\n0.0118703"
        r = check_provenance("I = 0.0118703 e^{x}", transcript, "题面")
        assert r["ok"]

    def test_provenance_untraceable(self):
        from jiushao.oracle_loop import check_provenance
        transcript = "[assistant r0]\n口算\n[sandbox r0]\n执行成功，输出：\n2.71828"
        r = check_provenance("I = 0.0118703 e^{x}", transcript, "题面")
        assert not r["ok"] and "0.0118703" in r["missing"]

    def test_provenance_simple_numbers_exempt(self):
        from jiushao.oracle_loop import check_provenance
        r = check_provenance(r"\frac{1}{2} x^2 + 3", "无沙箱", "题面")
        assert r["ok"]  # 1/2、3 这类平凡数不在审查范围


class TestErrorLedger:
    """出错账本：回溯重推时携带已否决尝试，防重蹈覆辙。"""

    ORACLE = {"points": [{"param": 0.001, "value": 195.7}]}

    @pytest.mark.asyncio
    async def test_ledger_accumulates(self, fake_llm_factory):
        from jiushao.oracle_loop import solve_with_oracle
        llm = fake_llm_factory([r"\boxed{\epsilon}", r"\boxed{\epsilon^{2}}"])
        r = await solve_with_oracle(llm, "q", oracle=self.ORACLE, max_revisions=1)
        assert len(r["ledger"]) == 2
        assert all("reason" in rec and "answer" in rec for rec in r["ledger"])

    @pytest.mark.asyncio
    async def test_feedback_carries_previous_failures(self, fake_llm_factory):
        from jiushao.oracle_loop import solve_with_oracle
        llm = fake_llm_factory([r"\boxed{\epsilon}", r"\boxed{\epsilon^{2}}", r"\boxed{\epsilon^{3}}"])
        prompts = []
        orig = llm.chat

        async def spy(messages, **kw):
            prompts.append(messages[-1]["content"])
            return await orig(messages, **kw)

        llm.chat = spy
        await solve_with_oracle(llm, "q", oracle=self.ORACLE, max_revisions=2)
        # 第 3 次求解的题面里应带有第 1 次被否决的答案
        third = [p for p in prompts if "已被机器验证否决" in p]
        assert third and r"\epsilon" in third[-1]

    @pytest.mark.asyncio
    async def test_verified_returns_ledger(self, fake_llm_factory):
        from jiushao.oracle_loop import solve_with_oracle
        llm = fake_llm_factory([r"\boxed{\frac{0.67}{\epsilon^{0.75}}}"])
        r = await solve_with_oracle(llm, "q", oracle=self.ORACLE)
        assert r["verified"] and r["ledger"] == []
