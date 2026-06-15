#!/usr/bin/env python3
"""失败分诊：把错题按根因分类，区分系统 bug 与模型能力问题。

用法：python scripts/triage.py <run_name> [run_name2 ...]

分类：
  PIPELINE_ERROR   整题异常（代码 bug / API 故障）
  NO_ANSWER        未抽取到答案（抽取器或求解链问题）
  JUDGE_SUSPECT    pred 与 gold 数值接近或形似 → 疑似判分器误判（系统 bug）
  MODEL_WRONG      真实错误（模型能力问题）
"""
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jiushao.config import RUNS_ROOT  # noqa: E402


def _num(s):
    try:
        return float(re.sub(r"[,\s$]", "", str(s)))
    except (ValueError, TypeError):
        return None


def classify(r: dict) -> tuple[str, str]:
    if r.get("error"):
        return "PIPELINE_ERROR", r["error"][:80]
    pred, gold = r.get("pred"), r.get("gold")
    if pred is None or str(pred).strip() == "":
        return "NO_ANSWER", ""
    fp, fg = _num(pred), _num(gold)
    if fp is not None and fg is not None and fg != 0:
        rel = abs(fp - fg) / abs(fg)
        if rel < 0.05:
            return "JUDGE_SUSPECT", f"数值相对差 {rel:.2%}"
    sim = difflib.SequenceMatcher(None, str(pred).lower(), str(gold).lower()).ratio()
    if sim > 0.75:
        return "JUDGE_SUSPECT", f"字符串相似度 {sim:.0%}"
    return "MODEL_WRONG", ""


def main(runs):
    buckets: dict[str, list] = {}
    total = n_correct = 0
    for run in runs:
        path = RUNS_ROOT / run / "results.jsonl"
        if not path.exists():
            print(f"跳过 {run}（无结果）")
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            total += 1
            if r["correct"]:
                n_correct += 1
                continue
            cat, why = classify(r)
            buckets.setdefault(cat, []).append((run, r, why))

    print(f"总计 {total} 题，正确 {n_correct}（{n_correct/total:.0%}），错误 {total-n_correct}\n")
    for cat in ["PIPELINE_ERROR", "NO_ANSWER", "JUDGE_SUSPECT", "MODEL_WRONG"]:
        items = buckets.get(cat, [])
        if not items:
            continue
        print(f"━━━ {cat}（{len(items)} 题）━━━")
        for run, r, why in items:
            print(f"  [{run}] {r['id']}")
            print(f"      pred={str(r['pred'])[:70]!r}")
            print(f"      gold={str(r['gold'])[:70]!r}  {why}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["dbg-theoremqa-b1", "dbg-hardmath-b1", "dbg-arb-b1", "dbg-math500-b2"])
