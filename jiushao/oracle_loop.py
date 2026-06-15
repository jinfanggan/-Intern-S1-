"""数值裁判反馈求解闭环（创新点一的完整形态）。

求解 → 数值裁判打分 → 未通过则把「你的答案在采样点的值 vs 数值真值」作为反馈重解。
这是 CRITIC 意义上「绑定外部工具反馈」的自修正——反馈是程序算出的数值差异，
不是模型的自我感觉。

防博弈设计（验证器反作弊）：数值裁判采样点拆为「公开点 / 留出点」，
反馈只展示公开点的真值，verified 判定要求全部点（含模型从未见过的留出点）通过——
朝公开真值拟合系数的假表达式会在留出点上露馅。
"""
from .oracle import construct_oracle, score_candidate
from .solver import solve_tir


def split_points(oracle: dict) -> tuple[list[int], list[int]]:
    """按参数排序后交替分配 → (公开点下标, 留出点下标)。单点无法拆分则全公开。"""
    pts = oracle["points"]
    if len(pts) < 2:
        return list(range(len(pts))), []
    order = sorted(range(len(pts)),
                   key=lambda i: (pts[i]["param"] is None, pts[i]["param"] or 0))
    revealed = [i for k, i in enumerate(order) if k % 2 == 0]
    held = [i for k, i in enumerate(order) if k % 2 == 1]
    return revealed, held


def _build_feedback(answer: str | None, score: dict, revealed: list[int],
                    ledger: list[dict] | None = None) -> str:
    lines = []
    # —— 出错账本：此前所有被否决的尝试（防止回溯后重蹈覆辙） ——
    if ledger:
        lines.append("【此前已被机器验证否决的尝试，请勿重复】")
        for i, rec in enumerate(ledger[-4:], 1):       # 最多带最近 4 条，控上下文
            lines.append(f"{i}. 答案 {str(rec['answer'])[:80]!r} —— {rec['reason']}")
        lines.append("")
    lines += [
        f"你上一次的答案是：{answer!r}。",
        "机器验证（直接数值求解原问题）发现它与数值真值不符：",
    ]
    for i in revealed:
        d = score["details"][i]
        loc = f"在参数取 {d['param']} 处" if d["param"] is not None else "该问题"
        if d["candidate"] is None:
            lines.append(f"- {loc}：你的答案无法数值求值；数值真值为 {d['truth']:.6g}。")
        elif d["ok"]:
            lines.append(f"- {loc}：你的答案给出 {d['candidate']:.6g}，与真值 {d['truth']:.6g} 吻合 ✓")
        else:
            lines.append(f"- {loc}：你的答案给出 {d['candidate']:.6g}，"
                         f"但数值真值为 {d['truth']:.6g}（不在 3 倍因子窗口内）。")
    lines.append(
        "注意：除上述公开点外，还有若干【未公开的校验点】会参与最终验证，"
        "凑系数拟合公开点是无效的——请通过严格推导重新求解，"
        "确保解析结果在整个 regime 上成立，最终答案放入 \\boxed{}。")
    return "\n".join(lines)


def _ledger_reason(score: dict, revealed: list[int]) -> str:
    """把一次失败浓缩成账本条目的原因描述（只记事实）。"""
    fails = []
    for i in revealed:
        d = score["details"][i]
        if not d["ok"]:
            if d["candidate"] is None:
                fails.append(f"参数 {d['param']} 处无法求值")
            else:
                fails.append(f"参数 {d['param']} 处给出 {d['candidate']:.3g}"
                             f"（真值 {d['truth']:.3g}）")
    held_fail = sum(1 for i, d in enumerate(score["details"])
                    if i not in revealed and not d["ok"])
    if held_fail:
        fails.append(f"另有 {held_fail} 个未公开校验点未通过")
    return "；".join(fails) or "未通过全部校验点"


import re as _re


def check_provenance(answer: str | None, transcript: str, question: str) -> dict:
    """常数溯源：答案中 ≥4 位有效数字的常数必须出自沙箱输出或题面。

    推导出的常数必然出现在某次代码执行的 stdout 中；凭空冒出（或从反馈
    倒推拟合）的系数没有出处。返回 {ok, missing}。
    """
    if not answer:
        return {"ok": True, "missing": []}
    suspects = [m.group(0) for m in
                _re.finditer(r"\d+\.\d{3,}(?:[eE][-+]?\d+)?", str(answer))]
    if not suspects:
        return {"ok": True, "missing": []}
    # 来源：题面 + 转录中所有沙箱输出段
    sources = question + "\n" + "\n".join(
        seg for seg in transcript.split("[sandbox")
        if "输出" in seg or "stdout" in seg) if "[sandbox" in transcript else question
    src_nums = [float(m.group(0)) for m in
                _re.finditer(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", sources)]
    missing = []
    for s in suspects:
        v = float(s)
        if not any(sn != 0 and abs(v - sn) / abs(sn) < 1e-3
                   or (sn == 0 and abs(v) < 1e-9) for sn in src_nums):
            missing.append(s)
    return {"ok": not missing, "missing": missing}


async def solve_with_oracle(llm, question: str, *, oracle: dict | None = None,
                            max_revisions: int = 2, tir_rounds: int = 4,
                            chain_id: int = 0, emit=None) -> dict:
    """返回 {answer, verified, score, revisions, oracle_ok, provenance_ok}。

    verified 的完整含义：全部采样点（含模型从未见过的留出点）通过
    且答案常数可溯源至实际执行的代码输出。oracle 传 None 时现场构造；
    构造失败则退化为普通 TIR 单链。
    """
    if oracle is None:
        oracle = await construct_oracle(llm, question, emit=emit)
    if oracle is None:
        out = await solve_tir(llm, question, chain_id=chain_id,
                              max_rounds=tir_rounds, emit=emit)
        return {"answer": out["answer"], "verified": False, "score": None,
                "revisions": 0, "oracle_ok": False, "provenance_ok": None}

    revealed, held = split_points(oracle)
    best = {"answer": None, "score": -1.0}
    feedback = ""
    ledger: list[dict] = []                    # 出错账本：本题所有被否决的尝试
    for rev in range(max_revisions + 1):
        q = question if not feedback else question + "\n\n" + feedback
        out = await solve_tir(llm, q, chain_id=chain_id + 100 * rev,
                              max_rounds=tir_rounds, emit=emit)
        s = score_candidate(out["answer"] or "", oracle)
        held_pass = all(s["details"][i]["ok"] for i in held)
        prov = check_provenance(out["answer"], out["transcript"], question)
        if emit:
            emit("oracle", "score", revision=rev, answer=out["answer"],
                 score=s["score"], passed=s["passed"], checked=s["checked"],
                 held_passed=held_pass, provenance_ok=prov["ok"],
                 provenance_missing=prov["missing"])
        if s["score"] > best["score"]:
            best = {"answer": out["answer"], "score": s["score"]}
        if s["score"] >= 1.0 and prov["ok"]:
            return {"answer": out["answer"], "verified": True, "score": 1.0,
                    "revisions": rev, "oracle_ok": True, "provenance_ok": True,
                    "ledger": ledger}
        # —— 记账（只记事实，供重推与跨题剧本蒸馏复用） ——
        reason = (f"常数 {prov['missing']} 无推导出处（疑似拟合）"
                  if s["score"] >= 1.0 else _ledger_reason(s, revealed))
        ledger.append({"answer": out["answer"], "reason": reason,
                       "score": s["score"], "revision": rev})
        if emit:
            emit("oracle", "ledger", revision=rev, reason=reason)
        if s["score"] >= 1.0:
            # 答案过点但常数无出处 → 疑似拟合，要求展示推导
            feedback = (f"你的答案 {out['answer']!r} 数值上通过了校验，但其中的常数 "
                        f"{prov['missing']} 在你的推导与代码执行输出中没有出处。"
                        "请用代码完整推导这些常数的来历（逐步计算并 print 中间量），"
                        "重新给出最终答案，放入 \\boxed{}。")
            continue
        feedback = _build_feedback(out["answer"], s, revealed, ledger[:-1])
    return {"answer": best["answer"], "verified": False, "score": best["score"],
            "revisions": max_revisions, "oracle_ok": True,
            "provenance_ok": None, "ledger": ledger}


# --- 注册为可消融的求解器（--profile b3-oracle） ---
async def _oracle_solver(llm, question, *, chain_id: int = 0,
                         temperature: float | None = None, emit=None,
                         max_revisions: int = 2, tir_rounds: int = 4) -> dict:
    r = await solve_with_oracle(llm, question, max_revisions=max_revisions,
                                tir_rounds=tir_rounds, chain_id=chain_id, emit=emit)
    return {"answer": r["answer"], "transcript": "",
            "verified": r.get("verified"), "score": r.get("score"),
            "ledger": r.get("ledger", [])}


from .solver import register_solver  # noqa: E402
register_solver("oracle-tir", _oracle_solver)
