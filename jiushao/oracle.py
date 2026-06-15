"""数值裁判（创新点一「评估器优先」的核心构件）。

思想：不求解析解，而是让模型写代码**数值求解原问题**（数值积分/数值 ODE/蒙特卡洛），
得到采样点真值作为「可执行裁判」；所有候选解析解必须与真值吻合才能进入投票。
裁判是程序而非模型——绕开 LLM 自我校验不可靠的根本缺陷。
"""
import json
import re

from .judge import _eval_at, _within_factor
from .llm import LLM
from .sandbox import extract_code_blocks, run_code

ORACLE_SYSTEM = (
    "你是数学评估器构造专家。给定一道数学题，你的任务不是求解析解，"
    "而是写一段 Python 代码用数值方法直接计算问题的真值：\n"
    "- 定积分/含参积分 → scipy.integrate.quad\n"
    "- ODE → scipy.integrate.solve_ivp\n"
    "- 概率/计数 → 蒙特卡洛或暴力枚举\n"
    "规则：\n"
    "1. 若问题含小参数（如 epsilon），在能代表不同 regime 的采样点处计算真值，"
    "且【每个 regime 取 2 个不同采样点】（如小 regime 取 1e-3 与 5e-3，"
    "大 regime 取 1e6 与 5e6，按题意选取合理范围）。\n"
    "2. 最后必须 print 一个 JSON（不要输出其他内容到最后一行）：\n"
    '   含参：{\"points\": [{\"param\": 0.001, \"value\": 195.7}, ...]}\n'
    '   无参：{\"points\": [{\"param\": null, \"value\": 42.0}]}\n'
    '3. 若该问题无法数值求解，print {\"impossible\": \"原因\"}。\n'
    "4. 只输出一个 ```python 代码块。"
)


def _parse_oracle_output(stdout: str) -> dict | None:
    """从 stdout 提取最后一个合法 JSON 对象。"""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


async def construct_oracle(llm: LLM, question: str, *, timeout: int = 60,
                           emit=None) -> dict | None:
    """构造数值裁判。返回 {"points": [{"param", "value"}...], "code": ...}，失败返回 None。"""
    resp = await llm.chat(
        [{"role": "system", "content": ORACLE_SYSTEM},
         {"role": "user", "content": question}],
        cache_tag="oracle-v0")
    codes = extract_code_blocks(resp["content"])
    if not codes:
        return None
    result = run_code(codes[-1], timeout=timeout)
    if emit:
        emit("oracle", "construct", code=codes[-1], ok=result["ok"],
             stdout=result["stdout"][-500:], stderr=result["stderr"][-300:])
    if not result["ok"]:
        return None
    data = _parse_oracle_output(result["stdout"])
    if not data or "impossible" in data or not data.get("points"):
        return None
    # 健全性检查：真值必须是有限数
    points = []
    for pt in data["points"]:
        try:
            v = float(pt["value"])
            if v == v and abs(v) != float("inf"):     # 非 NaN/Inf
                points.append({"param": pt.get("param"), "value": v})
        except (TypeError, ValueError, KeyError):
            continue
    return {"points": points, "code": codes[-1]} if points else None


def score_candidate(candidate: str, oracle: dict, *, factor: float = 3.0) -> dict:
    """候选解析解对数值裁判打分。

    返回 {passed, checked, score}：score = 通过点数/检查点数；
    含参题候选可能是 regime 列表 → 每个采样点只要任一元素吻合即通过该点。
    """
    from .judge import _candidates, _split_list
    elements: list[str] = []
    for c in _candidates(candidate or ""):
        elements.append(c)
        elements.extend(_split_list(c) or [])
    if not elements:
        return {"passed": 0, "checked": len(oracle["points"]), "score": 0.0}
    from .judge import _expr_of

    def _value_of(el: str, param) -> complex | None:
        if param is not None:                  # 含参题：在采样点代入
            return _eval_at(el, param)
        e = _expr_of(el)                       # 无参题：候选必须是常量
        if e is None or e.free_symbols:
            return None
        try:
            return complex(e.evalf())
        except Exception:
            return None

    passed, checked, details = 0, 0, []
    for pt in oracle["points"]:
        checked += 1
        target = complex(pt["value"])
        best = None                            # 该点上候选元素的最佳求值
        ok = False
        for el in elements:
            v = _value_of(el, pt["param"])
            if v is None:
                continue
            if best is None or abs(v - target) < abs(best - target):
                best = v
            if _within_factor(v, target, factor):
                ok = True
                break
        passed += ok
        details.append({"param": pt["param"], "truth": pt["value"],
                        "candidate": None if best is None else abs(best),
                        "ok": ok})
    return {"passed": passed, "checked": checked,
            "score": passed / checked if checked else 0.0, "details": details}
