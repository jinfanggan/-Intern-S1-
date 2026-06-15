#!/usr/bin/env python3
"""对比实验：baseline(纯投票) vs +数值裁判，走完整 adapter 路径。

用法：python scripts/compare_oracle.py [dataset] [n_chains]
默认在官方样例题上跑，用 OpenAI key 模拟官方 client（同为 OpenAI 兼容）。
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from jiushao.adapter import solve_problem
from jiushao.config import load_env
from jiushao.dsets import load
from jiushao.judge import judge_with_meta

load_env()


class MockOfficialClient:
    """模拟官方 InternChatClient：同步 chat(messages, temperature, max_tokens) -> str。"""

    def __init__(self, model="gpt-5-nano"):
        self.c = OpenAI()
        self.model = model

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        r = self.c.chat.completions.create(
            model=self.model, messages=messages,
            max_completion_tokens=max_tokens, reasoning_effort="low")
        return r.choices[0].message.content or ""


async def run(dataset="official", n_chains=3):
    problems = load(dataset)
    client = MockOfficialClient()
    print(f"数据集 {dataset}：{len(problems)} 题，n_chains={n_chains}\n")
    rows = []
    for p in problems:
        # 同一题先关后开裁判；首跑结果进 OfficialClientLLM 内存缓存，
        # 但两次是独立 LLM 实例（缓存不跨调用），故均为真实调用
        base = await solve_problem(client, p.question, n_chains=n_chains, use_oracle=False)
        orac = await solve_problem(client, p.question, n_chains=n_chains, use_oracle=True)
        b_ok = judge_with_meta(_ans(base), p.answer, p.meta)
        o_ok = judge_with_meta(_ans(orac), p.answer, p.meta)
        oracle_acted = any(t["step"].startswith("oracle:filter") for t in orac["trace"])
        rows.append((p.id, p.subject, p.answer, b_ok, o_ok, oracle_acted))
        print(f"{p.id} [{p.subject}] gold={p.answer}")
        print(f"   baseline(纯投票)  {'✓' if b_ok else '✗'}")
        print(f"   +数值裁判         {'✓' if o_ok else '✗'}  (裁判过滤生效={oracle_acted})")
    nb = sum(r[3] for r in rows); no = sum(r[4] for r in rows)
    print(f"\n=== 汇总 {dataset} ({len(rows)}题) ===")
    print(f"baseline 纯投票: {nb}/{len(rows)}")
    print(f"+数值裁判:       {no}/{len(rows)}  (裁判生效 {sum(r[5] for r in rows)} 题)")


def _ans(result):
    """从 final_response（完整解答）提取答案供本地判分。"""
    from jiushao.judge import extract_answer
    return extract_answer(result["final_response"]) or result["final_response"]


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "official"
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    asyncio.run(run(ds, nc))
