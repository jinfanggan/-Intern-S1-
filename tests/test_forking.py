"""创新点二歧义分叉单元测试（FakeLLM，不打网络）。"""
import json

import pytest

from jiushao.forking import (_parse_json, arbitrate, enumerate_interpretations,
                             solve_with_forking)


class SeqClient:
    """按"匹配关键词"返回脚本的 FakeLLM，兼容 jiushao LLM.chat 接口。

    rules: list[(关键词或None, 返回内容)]，按顺序首个匹配命中。
    """

    def __init__(self, rules):
        self.rules = rules
        self.usage = {"in": 0, "out": 0, "calls": 0, "cache_hits": 0}

    async def chat(self, messages, *, temperature=None, max_tokens=None,
                   cache_tag="", retries=5):
        self.usage["calls"] += 1
        sysc = messages[0]["content"]
        for kw, out in self.rules:
            if kw is None or kw in sysc or kw in cache_tag:
                return {"content": out, "usage_in": 1, "usage_out": 1, "cached": False}
        return {"content": r"\boxed{0}", "usage_in": 1, "usage_out": 1, "cached": False}


def test_parse_json():
    assert _parse_json('前缀 {"a": 1} 后缀')["a"] == 1
    assert _parse_json("无 json") is None


@pytest.mark.asyncio
async def test_enumerate_single_interpretation():
    c = SeqClient([("审题", '{"interpretations": [{"reading": "唯一解读", "note": ""}]}')])
    interps = await enumerate_interpretations(c, "q")
    assert len(interps) == 1 and interps[0]["reading"] == "唯一解读"


@pytest.mark.asyncio
async def test_enumerate_caps_at_three():
    many = {"interpretations": [{"reading": f"r{i}", "note": ""} for i in range(5)]}
    c = SeqClient([("审题", json.dumps(many))])
    interps = await enumerate_interpretations(c, "q")
    assert len(interps) == 3


@pytest.mark.asyncio
async def test_enumerate_fallback_on_garbage():
    c = SeqClient([("审题", "不是 json")])
    interps = await enumerate_interpretations(c, "q")
    assert len(interps) == 1  # 退化为"按字面理解"


@pytest.mark.asyncio
async def test_arbitrate_picks_index():
    c = SeqClient([("仲裁", '{"best": 1, "reason": "符合惯例"}')])
    cand = [{"reading": "A", "answer": "1"}, {"reading": "B", "answer": "2"}]
    assert await arbitrate(c, "q", cand) == 1


@pytest.mark.asyncio
async def test_arbitrate_clamps_out_of_range():
    c = SeqClient([("仲裁", '{"best": 9}')])
    cand = [{"reading": "A", "answer": "1"}]
    assert await arbitrate(c, "q", cand) == 0


@pytest.mark.asyncio
async def test_forking_single_interp_degrades():
    # 单解读 → 直接多链 TIR 投票，无仲裁
    c = SeqClient([
        ("审题", '{"interpretations": [{"reading": "唯一", "note": ""}]}'),
        (None, r"\boxed{42}"),
    ])
    r = await solve_with_forking(c, "q", n_chains=2, max_rounds=2)
    assert r["n_interp"] == 1 and r"\boxed{42}" in r["transcript"]


@pytest.mark.asyncio
async def test_forking_multi_interp_arbitrates():
    # 两解读：解读0→126000(误读)，解读1→11760(正解)，仲裁选 1
    c = SeqClient([
        ("审题", '{"interpretations": [{"reading": "带标号桶", "note": "误读"}, '
                 '{"reading": "内部有序子集", "note": "正解"}]}'),
        ("仲裁", '{"best": 1, "reason": "ordered 指内部有序"}'),
        ("Python", r"\boxed{11760}"),  # solve_tir 的 system 含 Python
    ])
    r = await solve_with_forking(c, "q", n_chains=1, max_rounds=1)
    assert r["n_interp"] == 2 and r["interpretation"] == "内部有序子集"


@pytest.mark.asyncio
async def test_forking_adapter_contract():
    from jiushao.adapter import solve_with_forking_adapter
    c_rules = [
        ("审题", '{"interpretations": [{"reading": "唯一", "note": ""}]}'),
        (None, r"\boxed{7}"),
    ]

    class SyncClient:
        def chat(self, messages, temperature=0.2, max_tokens=4096):
            sysc = messages[0]["content"]
            for kw, out in c_rules:
                if kw is None or kw in sysc:
                    return out
            return r"\boxed{7}"

    r = await solve_with_forking_adapter(SyncClient(), "q", n_chains=1, max_rounds=1)
    assert isinstance(r["final_response"], str) and r["final_response"]
    json.dumps(r)  # trace 可序列化
