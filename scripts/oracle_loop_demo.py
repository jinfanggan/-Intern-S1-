#!/usr/bin/env python3
"""数值裁判反馈闭环实验：b1 全错的 7 道 HARDMath 积分题能救回几道？

配对设计：闭环首次求解与 b1 同 prompt 同链号 → 命中缓存复现原错误答案，
之后的修正轮才是新增调用 → 增益完全归因于数值裁判反馈。
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jiushao.config import DATASETS_ROOT
from jiushao.judge import judge_with_meta
from jiushao.llm import LLM
from jiushao.oracle import construct_oracle
from jiushao.oracle_loop import solve_with_oracle
from jiushao.runlog import RunLogger


async def main():
    data = json.loads((DATASETS_ROOT / "HARDMath" / "data" / "HARDMath.json").read_text())
    items = [(k, d) for k, d in list(data.items())[:20] if d["question_type"] == "integral"]
    llm = LLM("gpt-5-nano")
    logger = RunLogger("oracle-loop-integral7", meta={"exp": "数值裁判反馈闭环", "n": len(items)})

    async def one(k, d):
        pid = f"hardmath-{k}"
        emit = logger.event_writer(pid)
        pts = [v for v in (d.get("small_eval_point"), d.get("large_eval_point")) if v is not None]
        oq = (d["question"] + "\n\n（构造数值评估时请使用采样点 "
              + " 与 ".join(f"epsilon={v}" for v in pts) + "）")
        oracle = await construct_oracle(llm, oq, emit=emit)        # 命中 demo 缓存
        r = await solve_with_oracle(llm, d["question"], oracle=oracle, emit=emit)
        meta = {"hardmath": {"small_eval": d.get("small_eval_point"),
                             "small_val": d.get("small_analytical"),
                             "large_eval": d.get("large_eval_point"),
                             "large_val": d.get("large_analytical")}}
        regime_ok = judge_with_meta(r["answer"], str(d.get("answer_val", "")), meta)
        emit._close()
        logger.write_result({"id": pid, "subject": "integral", "pred": r["answer"],
                             "gold": str(d.get("answer_val", ""))[:80],
                             "correct": bool(r["verified"] or regime_ok),
                             "verified": r["verified"], "regime_ok": regime_ok,
                             "score": r["score"], "revisions": r["revisions"]})
        print(f"{'✓' if (r['verified'] or regime_ok) else '✗'} {pid}: "
              f"数值裁判通过={r['verified']} regime判分={regime_ok} "
              f"数值裁判得分={r['score']} 修正轮数={r['revisions']}")
        return r["verified"] or regime_ok

    results = await asyncio.gather(*[one(k, d) for k, d in items])
    print(f"\n===== 数值裁判反馈闭环：{sum(results)}/{len(results)} 救回（b1 基线 0/{len(results)}）=====")
    u = llm.usage
    print(f"API: {u['calls']} 次, 缓存命中 {u['cache_hits']}, 约 ${llm.cost_usd():.4f}")


if __name__ == "__main__":
    asyncio.run(main())
