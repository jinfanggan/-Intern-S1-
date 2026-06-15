"""结构化运行日志。

设计目标：
1. debug —— 每题完整事件流（每条链每轮、沙箱执行、聚簇、判分）可单独回看；
2. 前端展示 —— 事件流即推理树回放数据源，字段稳定、自描述；
3. 合规留痕 —— 满足赛规「日志可复核」要求。

目录结构：
    runs/<run_name>/
    ├── meta.json            运行配置（模型、profile、数据集、时间）
    ├── results.jsonl        每题一行摘要（断点续跑据此跳过已完成题）
    └── events/<pid>.jsonl   每题事件流，事件 = {ts, stage, type, **data}
"""
import json
import time
from pathlib import Path

from .config import RUNS_ROOT


def _sanitize(pid: str) -> str:
    return pid.replace("/", "_").replace("\\", "_")


class RunLogger:
    def __init__(self, run_name: str, meta: dict | None = None):
        self.root = RUNS_ROOT / run_name
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.root / "results.jsonl"
        if meta is not None:
            (self.root / "meta.json").write_text(
                json.dumps({**meta, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                           ensure_ascii=False, indent=2))

    # -- 事件流（前端回放数据源） ------------------------------------------
    def event_writer(self, pid: str):
        """返回该题的 emit(stage, type, **data) 函数。"""
        path = self.events_dir / (_sanitize(pid) + ".jsonl")
        f = path.open("a", encoding="utf-8")

        def emit(stage: str, etype: str, **data):
            f.write(json.dumps(
                {"ts": round(time.time(), 3), "stage": stage, "type": etype, **data},
                ensure_ascii=False) + "\n")
            f.flush()

        emit._close = f.close  # type: ignore[attr-defined]
        return emit

    # -- 摘要与断点续跑 ------------------------------------------------------
    def done_ids(self) -> set[str]:
        if not self.results_path.exists():
            return set()
        return {json.loads(l)["id"] for l in self.results_path.read_text().splitlines() if l.strip()}

    def write_result(self, result: dict):
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def load_results(self) -> list[dict]:
        if not self.results_path.exists():
            return []
        return [json.loads(l) for l in self.results_path.read_text().splitlines() if l.strip()]
