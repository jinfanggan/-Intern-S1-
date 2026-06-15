"""测试公共设施：FakeLLM（不打网络的 LLM 替身）。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jiushao.config import MODEL_PROFILES  # noqa: E402


class FakeLLM:
    """按脚本回放回复的 LLM 替身，兼容 jiushao.llm.LLM 的 chat 接口。

    scripts: list[str]，第 i 次调用返回 scripts[min(i, len-1)]。
    """

    def __init__(self, scripts: list[str]):
        self.scripts = scripts
        self.n_calls = 0
        self.usage = {"in": 0, "out": 0, "calls": 0, "cache_hits": 0}
        self.profile = {"price_in": 0, "price_out": 0}

    async def chat(self, messages, *, temperature=None, max_tokens=None,
                   cache_tag="", retries=5):
        content = self.scripts[min(self.n_calls, len(self.scripts) - 1)]
        self.n_calls += 1
        self.usage["calls"] += 1
        return {"content": content, "usage_in": 10, "usage_out": 20, "cached": False}

    def cost_usd(self):
        return 0.0


@pytest.fixture
def fake_llm_factory():
    return FakeLLM


@pytest.fixture
def collect_events():
    """事件收集器：返回 (emit, events)，emit 兼容 runlog 的事件接口。"""
    events: list[dict] = []

    def emit(stage, etype, **data):
        events.append({"stage": stage, "type": etype, **data})

    emit._close = lambda: None
    return emit, events
