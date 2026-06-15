"""求解链：B0 纯 CoT 单链 / B1+ TIR 工具循环链。

所有阶段通过 emit(stage, type, **data) 写结构化事件日志（见 runlog.py），
便于 debug 与前端推理树回放。emit 为 None 时静默。
"""
from .judge import extract_answer
from .llm import LLM
from .sandbox import extract_code_blocks, run_code

SYSTEM_COT = (
    "你是一位数学专家。仔细分析问题，先简述解题思路，再逐步推理求解。"
    "最终答案务必放在 \\boxed{} 中。"
)

SYSTEM_TIR = (
    "你是一位数学专家，可以使用 Python 帮助计算。\n"
    "规则：\n"
    "1. 先简述解题计划（用什么方法、分几步）。\n"
    "2. 需要计算时写 ```python 代码块（可用 sympy/numpy/scipy），用 print 输出结果；"
    "我会执行并把输出返回给你。\n"
    "3. 拿到执行结果后继续推理；如代码报错请修正重试。\n"
    "4. 得到最终答案后，将其放在 \\boxed{} 中并停止。"
)

# 多链采样的切入角扰动（gpt-5 系不支持温度，多样性靠提示扰动；支持温度的模型两者叠加）
ANGLES = [
    "", "尝试用与直觉第一反应不同的方法求解。", "优先考虑数值/计算式方法。",
    "优先考虑解析/代数方法。", "先考虑特殊情形或极端情形获得直觉，再严格求解。",
    "注意检查边界条件、定义域与特殊点。", "考虑能否用变换（换元、对称性、不变量）简化问题。",
    "用最直接朴素的方法一步步算，不追求技巧。",
]


def _noop(stage, etype, **data):
    pass


async def solve_cot(llm: LLM, question: str, *, chain_id: int = 0,
                    temperature: float | None = None, emit=None) -> dict:
    emit = emit or _noop
    angle = ANGLES[chain_id % len(ANGLES)] if chain_id else ""
    messages = [
        {"role": "system", "content": SYSTEM_COT + ("\n" + angle if angle else "")},
        {"role": "user", "content": question},
    ]
    resp = await llm.chat(messages, temperature=temperature, cache_tag=f"cot-{chain_id}")
    answer = extract_answer(resp["content"])
    emit("solve", "chain_round", chain=chain_id, round=0, content=resp["content"],
         cached=resp["cached"], usage_in=resp["usage_in"], usage_out=resp["usage_out"])
    emit("solve", "chain_done", chain=chain_id, answer=answer, rounds=1)
    return {"answer": answer, "transcript": resp["content"], "rounds": 1}


async def solve_tir(llm: LLM, question: str, *, chain_id: int = 0, max_rounds: int = 4,
                    temperature: float | None = None, emit=None) -> dict:
    emit = emit or _noop
    angle = ANGLES[chain_id % len(ANGLES)] if chain_id else ""
    messages = [
        {"role": "system", "content": SYSTEM_TIR + ("\n" + angle if angle else "")},
        {"role": "user", "content": question},
    ]
    transcript_parts = []
    for rnd in range(max_rounds):
        resp = await llm.chat(messages, temperature=temperature,
                              cache_tag=f"tir-{chain_id}-r{rnd}")
        content = resp["content"]
        transcript_parts.append(f"[assistant r{rnd}]\n{content}")
        emit("solve", "chain_round", chain=chain_id, round=rnd, content=content,
             cached=resp["cached"], usage_in=resp["usage_in"], usage_out=resp["usage_out"])
        codes = extract_code_blocks(content)
        answer = extract_answer(content)
        has_boxed = "\\boxed{" in content
        if has_boxed and not codes:
            emit("solve", "chain_done", chain=chain_id, answer=answer, rounds=rnd + 1)
            return {"answer": answer, "transcript": "\n".join(transcript_parts),
                    "rounds": rnd + 1}
        if not codes:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user",
                             "content": "请继续：若需计算请给出 ```python 代码块，"
                                        "若已得出结论请将最终答案放入 \\boxed{}。"})
            continue
        result = run_code(codes[-1])
        emit("sandbox", "exec", chain=chain_id, round=rnd, code=codes[-1],
             ok=result["ok"], stdout=result["stdout"], stderr=result["stderr"])
        feedback = (f"代码执行成功，输出：\n{result['stdout'] or '(无输出)'}"
                    if result["ok"] else
                    f"代码执行失败：\n{result['stderr']}\n请修正后重试。")
        transcript_parts.append(f"[sandbox r{rnd}]\n{feedback}")
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": feedback})
    messages.append({"role": "user",
                     "content": "轮数已尽，请基于现有信息直接给出最终答案，放入 \\boxed{}。"})
    resp = await llm.chat(messages, temperature=temperature,
                          cache_tag=f"tir-{chain_id}-final")
    transcript_parts.append(f"[assistant final]\n{resp['content']}")
    answer = extract_answer(resp["content"])
    emit("solve", "chain_round", chain=chain_id, round=max_rounds, content=resp["content"],
         cached=resp["cached"], usage_in=resp["usage_in"], usage_out=resp["usage_out"])
    emit("solve", "chain_done", chain=chain_id, answer=answer, rounds=max_rounds + 1)
    return {"answer": answer, "transcript": "\n".join(transcript_parts),
            "rounds": max_rounds + 1}


# ---------------------------------------------------------------------------
# 求解器注册表（工程化：让任何新方法/对比基线注册一行即进消融矩阵）
#
# 统一签名：async solver(llm, question, *, chain_id, temperature, emit, **kw)
#           返回至少含 {"answer": str|None}，可附带 verified/score/ledger 等。
# 新增对比实验三步：① 写 solver 函数 ② register_solver(名, 函数)
#                  ③ config.PROFILES 加一行 → `--profile 名` 即可跑。
# ---------------------------------------------------------------------------
SOLVERS: dict = {}


def register_solver(name: str, fn) -> None:
    SOLVERS[name] = fn


register_solver("cot", solve_cot)
register_solver("tir", solve_tir)
