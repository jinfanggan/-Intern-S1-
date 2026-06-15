#!/usr/bin/env python3
"""数值裁判实战验证（创新点一原型）。

对 dbg-hardmath-b1 中的积分题：
1. live 构造数值裁判（指定与数据集相同的采样点）；
2. 用数据集自带的 small/large_numerical 真值校验数值裁判质量（裁判先自证）；
3. 用数值裁判重判 b1 已存预测，与 regime 判分对照。
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jiushao.config import DATASETS_ROOT, RUNS_ROOT
from jiushao.llm import LLM
from jiushao.oracle import construct_oracle, score_candidate


async def main():
    data = json.loads((DATASETS_ROOT / "HARDMath" / "data" / "HARDMath.json").read_text())
    results = {json.loads(l)["id"]: json.loads(l)
               for l in (RUNS_ROOT / "dbg-hardmath-b1" / "results.jsonl").read_text().splitlines()}

    # 取 dbg 里跑过的积分题
    items = [(k, d) for k, d in list(data.items())[:20] if d["question_type"] == "integral"]
    llm = LLM("gpt-5-nano")
    print(f"积分题 {len(items)} 道，开始构造数值裁判…\n")

    ok_oracle = ok_match = 0
    for k, d in items:
        pid = f"hardmath-{k}"
        pts = {"small": d.get("small_eval_point"), "large": d.get("large_eval_point")}
        pts = {kk: v for kk, v in pts.items() if v is not None}
        if not pts:
            print(f"- {pid}: 数据集无评测点，跳过")
            continue
        q = (d["question"] + "\n\n（构造数值评估时请使用采样点 "
             + " 与 ".join(f"epsilon={v}" for v in pts.values()) + "）")
        oracle = await construct_oracle(llm, q)
        if oracle is None:
            print(f"✗ {pid}: 数值裁判构造失败")
            continue
        ok_oracle += 1
        # —— 校验数值裁判 vs 数据集自带数值真值 ——
        truth = {}
        for kk, v in pts.items():
            tv = d.get(f"{kk}_numerical")
            if tv not in (None, ""):
                truth[float(v)] = float(tv)
        errs = []
        for pt in oracle["points"]:
            if pt["param"] is not None and float(pt["param"]) in truth:
                t = truth[float(pt["param"])]
                errs.append(abs(pt["value"] - t) / abs(t) if t else float("inf"))
        match = bool(errs) and max(errs) < 0.05
        ok_match += match
        # —— 用数值裁判重判 b1 已存预测 ——
        pred = results.get(pid, {}).get("pred")
        s = score_candidate(pred, oracle) if pred else {"score": None}
        old = results.get(pid, {}).get("correct")
        print(f"{'✓' if match else '⚠'} {pid}: 数值裁判点数={len(oracle['points'])} "
              f"vs真值误差={[f'{e:.1%}' for e in errs] or '采样点不匹配'} | "
              f"b1预测数值裁判得分={s['score']} (regime判分={old})")

    print(f"\n数值裁判构造成功率: {ok_oracle}/{len(items)}，与数据集真值吻合(<5%): {ok_match}/{ok_oracle}")
    u = llm.usage
    print(f"API: {u['calls']} 次, 缓存 {u['cache_hits']}, 约 ${llm.cost_usd():.4f}")


if __name__ == "__main__":
    asyncio.run(main())
