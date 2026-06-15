"""基础设施单元测试：runlog 事件流/断点续跑、llm 缓存键、dsets 加载、令牌桶。"""
import asyncio
import json
import time

import pytest

from jiushao import dsets
from jiushao.llm import TokenBucket
from jiushao.runlog import RunLogger


class TestRunLogger:
    def test_event_stream_and_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr("jiushao.runlog.RUNS_ROOT", tmp_path)
        lg = RunLogger("t-run", meta={"model": "fake"})
        emit = lg.event_writer("p/1")          # 含特殊字符的 id
        emit("solve", "chain_round", chain=0, content="x")
        emit("verdict", "judge", correct=True)
        emit._close()
        files = list((tmp_path / "t-run" / "events").glob("*.jsonl"))
        assert len(files) == 1
        events = [json.loads(l) for l in files[0].read_text().splitlines()]
        assert [e["stage"] for e in events] == ["solve", "verdict"]
        assert all("ts" in e for e in events)

        lg.write_result({"id": "p/1", "correct": True})
        assert lg.done_ids() == {"p/1"}        # 断点续跑接口
        assert (tmp_path / "t-run" / "meta.json").exists()

    def test_resume_skips_done(self, tmp_path, monkeypatch):
        monkeypatch.setattr("jiushao.runlog.RUNS_ROOT", tmp_path)
        lg = RunLogger("t-run2", meta={})
        lg.write_result({"id": "a", "correct": False})
        lg2 = RunLogger("t-run2")              # 重新打开同一 run
        assert lg2.done_ids() == {"a"}


class TestTokenBucket:
    def test_rate_limiting(self):
        async def go():
            bucket = TokenBucket(rpm=600)      # 0.1s/次
            t0 = time.monotonic()
            for _ in range(3):
                await bucket.acquire()
            return time.monotonic() - t0

        elapsed = asyncio.run(go())
        assert elapsed >= 0.15                 # 3 次至少跨 2 个间隔


class TestDatasets:
    def test_math500_schema(self):
        ps = dsets.load("math500", limit=3)
        assert len(ps) == 3
        p = ps[0]
        assert p.id and p.question and p.answer and p.subject

    def test_all_loaders_nonempty(self):
        for name in dsets.LOADERS:
            ps = dsets.load(name, limit=2)
            assert ps and ps[0].question, name

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            dsets.load("不存在")


class TestDomainClassify:
    """答案形态大类归一。"""

    def _p(self, subject="", **meta):
        from jiushao.dsets import Problem
        return Problem(id="x", question="q", answer="a", subject=subject, meta=meta)

    def test_subject_keyword_priority(self):
        from jiushao.dsets import classify_domain
        assert classify_domain(self._p("integral")) == "numeric"
        assert classify_domain(self._p("拓扑")) == "proof"
        assert classify_domain(self._p("抽象代数")) == "symbolic"
        assert classify_domain(self._p("复分析")) == "symbolic"

    def test_answer_type_fallback(self):
        from jiushao.dsets import classify_domain
        # subject 不明（theoremqa）→ 走 answer_type
        assert classify_domain(self._p("theoremqa", answer_type="float")) == "numeric"
        assert classify_domain(self._p("theoremqa", answer_type="bool")) == "decision"
        assert classify_domain(self._p("theoremqa", answer_type="integer")) == "symbolic"

    def test_unknown_is_other(self):
        from jiushao.dsets import classify_domain
        assert classify_domain(self._p("", )) == "other"

    def test_all_loaders_classify_without_error(self):
        from jiushao import dsets
        for name in dsets.LOADERS:
            for p in dsets.load(name, limit=5):
                assert dsets.classify_domain(p) in dsets.DOMAIN_LABELS
