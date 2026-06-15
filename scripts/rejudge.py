#!/usr/bin/env python3
"""离线重判：判分器升级后，对已有 run 的预测重新判分（零 API 成本）。

用法：python scripts/rejudge.py <run_name> ...（缺省重判全部 dbg- run）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jiushao import dsets  # noqa: E402
from jiushao.config import RUNS_ROOT  # noqa: E402
from jiushao.judge import judge_with_meta  # noqa: E402


def build_meta_index():
    idx = {}
    for name in dsets.LOADERS:
        for p in dsets.load(name):
            idx[p.id] = p
    return idx


def main(runs):
    idx = build_meta_index()
    for run in runs:
        path = RUNS_ROOT / run / "results.jsonl"
        if not path.exists():
            print(f"跳过 {run}")
            continue
        results = [json.loads(l) for l in path.read_text().splitlines()]
        old = sum(r["correct"] for r in results)
        flips = []
        for r in results:
            p = idx.get(r["id"])
            new = judge_with_meta(r["pred"], r["gold"], p.meta if p else None)
            if new != r["correct"]:
                flips.append((r["id"], r["correct"], new, r["pred"], r["gold"]))
            r["correct"] = new
        new_correct = sum(r["correct"] for r in results)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
        print(f"{run}: {old}/{len(results)} → {new_correct}/{len(results)}"
              f"  ({old/len(results):.0%} → {new_correct/len(results):.0%})")
        for pid, o, n, pred, gold in flips:
            arrow = "✗→✓" if n else "✓→✗"
            print(f"    {arrow} {pid}: pred={str(pred)[:50]!r} gold={str(gold)[:40]!r}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["dbg-theoremqa-b1", "dbg-hardmath-b1", "dbg-arb-b1", "dbg-math500-b2"])
