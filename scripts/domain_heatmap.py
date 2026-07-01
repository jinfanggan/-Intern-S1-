#!/usr/bin/env python3
"""跨数据集 × 答案形态大类 × profile 热力图。

汇总多个 run 的 results.jsonl，按答案形态大类聚合，对比 b1/b2（或任意 profile）。
用法：python scripts/domain_heatmap.py <run1> <run2> ...
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jiushao.config import RUNS_ROOT
from jiushao import dsets
from jiushao.dsets import DOMAIN_LABELS

DOMAIN_ORDER = ["numeric", "symbolic", "proof", "decision", "other"]


def build_domain_index() -> dict:
    """id → domain，给加 domain 字段之前跑的旧 run 兜底。"""
    idx = {}
    for name in dsets.LOADERS:
        for p in dsets.load(name):
            idx[p.id] = dsets.classify_domain(p)
    return idx


def load(run):
    p = RUNS_ROOT / run / "results.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main(runs):
    didx = build_domain_index()
    # profile -> domain -> [correct, total]
    by_prof = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    prof_total = defaultdict(lambda: [0, 0])
    for run in runs:
        for r in load(run):
            prof = r.get("profile", run)
            d = r.get("domain") or didx.get(r["id"], "other")
            by_prof[prof][d][0] += r["correct"]
            by_prof[prof][d][1] += 1
            prof_total[prof][0] += r["correct"]
            prof_total[prof][1] += 1

    profs = sorted(by_prof)
    print(f"汇总 runs: {runs}\n")
    # 表头
    hdr = f"{'答案形态大类':<14}" + "".join(f"{p:>14}" for p in profs)
    print(hdr)
    print("-" * len(hdr))
    for d in DOMAIN_ORDER:
        if not any(d in by_prof[p] for p in profs):
            continue
        row = f"{DOMAIN_LABELS[d]:<14}"
        for p in profs:
            c, n = by_prof[p].get(d, [0, 0])
            row += f"{(f'{c}/{n}={c/n:.0%}' if n else '-'):>14}"
        print(row)
    print("-" * len(hdr))
    row = f"{'总体':<14}"
    for p in profs:
        c, n = prof_total[p]
        row += f"{(f'{c}/{n}={c/n:.0%}' if n else '-'):>14}"
    print(row)


if __name__ == "__main__":
    main(sys.argv[1:] or [
        "abl-tqa200-b1", "full-hardmath-b1", "full-arb-b1"])
