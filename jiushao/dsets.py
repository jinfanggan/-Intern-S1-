"""数据集加载：统一为 Problem(id, question, answer, subject, meta)。"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATASETS_ROOT


@dataclass
class Problem:
    id: str
    question: str
    answer: str
    subject: str = ""
    meta: dict = field(default_factory=dict)


def load_math500(limit: int | None = None) -> list[Problem]:
    path = DATASETS_ROOT / "MATH-500" / "test.jsonl"
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        if limit and len(out) >= limit:
            break
        d = json.loads(line)
        out.append(Problem(
            id=d.get("unique_id", f"math500-{i}"),
            question=d["problem"], answer=str(d["answer"]),
            subject=d.get("subject", ""), meta={"level": d.get("level")},
        ))
    return out


_TQA_FORMAT_HINT = {
    "bool": "（答案只输出 True 或 False）",
    "option": "（答案只输出选项标号，如 (a)）",
    "integer": "（答案输出一个整数）",
    "float": "（答案输出一个数值）",
    "list of integer": "（答案输出整数列表，如 [1, 2]）",
    "list of float": "（答案输出数值列表，如 [1.0, 2.5]）",
}


def load_theoremqa(limit: int | None = None) -> list[Problem]:
    import pandas as pd
    df = pd.read_parquet(DATASETS_ROOT / "TheoremQA" / "data" / "test-00000-of-00001.parquet")
    out = []
    for i, row in df.iterrows():
        if limit and len(out) >= limit:
            break
        if row.get("Picture") is not None:      # 纯文本赛道，跳过带图题
            continue
        atype = row["Answer_type"]
        hint = _TQA_FORMAT_HINT.get(atype, "")
        out.append(Problem(
            id=f"theoremqa-{i}", question=row["Question"] + hint, answer=str(row["Answer"]),
            subject="theoremqa",
            # 官方标准答案按两位小数给出，数值容差放宽到 1e-2
            meta={"answer_type": atype, "rel_tol": 1e-2},
        ))
    return out


def load_hardmath(limit: int | None = None) -> list[Problem]:
    data = json.loads((DATASETS_ROOT / "HARDMath" / "data" / "HARDMath.json").read_text())
    out = []
    for k, d in data.items():
        if limit and len(out) >= limit:
            break
        out.append(Problem(
            id=f"hardmath-{k}", question=d["question"], answer=str(d.get("answer_val", "")),
            subject=d.get("question_type", "hardmath"),
            meta={
                "answer_type": d.get("answer_type"),
                # regime 数值点判分所需（judge.hardmath_equiv）
                "hardmath": {
                    "small_eval": d.get("small_eval_point"),
                    "small_val": d.get("small_analytical"),
                    "large_eval": d.get("large_eval_point"),
                    "large_val": d.get("large_analytical"),
                },
            },
        ))
    return out


def load_arb_math(limit: int | None = None) -> list[Problem]:
    out = []
    for fname in ["arb_math_numerical.json", "arb_math_symbolic.json"]:
        data = json.loads((DATASETS_ROOT / "ARB" / "data" / fname).read_text())
        for d in data:
            if limit and len(out) >= limit:
                break
            out.append(Problem(
                id=f"arb-{d['_id']}",
                question=d["Problem_Statement"] + "\n" + d.get("Output Format Instructions", ""),
                answer=str(d.get("Final Answer", "")),
                subject=d.get("Topic", "arb"), meta={"type": d.get("Problem Type")},
            ))
    return out[:limit] if limit else out


def load_official_sample(limit: int | None = None) -> list[Problem]:
    """官方 baseline 仓库的样例题（最接近正式评测分布的金标准）。

    路径优先级：reference/Challenge-Cup-2026/sample_data/dev.jsonl
    （clone 的官方 baseline）。带 answer，供本地对齐与判分。
    """
    path = DATASETS_ROOT.parent / "reference" / "Challenge-Cup-2026" / "sample_data" / "dev.jsonl"
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        if limit and len(out) >= limit:
            break
        d = json.loads(line)
        out.append(Problem(
            id=f"official-{d.get('idx', i)}",
            question=d["problem"], answer=str(d.get("answer", "")),
            subject=d.get("subject", "official"),
            meta={"source": d.get("source", "official_sample")},
        ))
    return out


LOADERS = {
    "math500": load_math500,
    "theoremqa": load_theoremqa,
    "hardmath": load_hardmath,
    "arb": load_arb_math,
    "official": load_official_sample,
}


def load(name: str, limit: int | None = None) -> list[Problem]:
    if name not in LOADERS:
        raise ValueError(f"未知数据集 {name}，可选: {list(LOADERS)}")
    return LOADERS[name](limit)
