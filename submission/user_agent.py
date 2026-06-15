"""提交入口：满足官方接口契约 ReasoningAgent(client).solve(problem, metadata) -> dict。

薄壳——求解逻辑全部复用 jiushao 引擎（见 jiushao/adapter.py）。
提交包结构：
    user_agent.py        ← 本文件（提交根目录必需）
    jiushao/             ← 引擎包（solver/aggregate/sandbox/judge/adapter ...）
    requirements.txt
"""
import asyncio
import os
import sys

# 确保能 import 同级 jiushao 包（正式评测从提交根目录加载）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jiushao.adapter import solve_problem


class ReasoningAgent:
    """官方契约要求：__init__(client, ...) + solve(problem, metadata) -> dict。"""

    def __init__(self, client, *args, **kwargs):
        self.client = client
        # 轻量、确定性的配置（不读隐藏数据、不做耗时外部请求）
        self.n_chains = int(kwargs.get("n_chains", 3))
        self.max_rounds = int(kwargs.get("max_rounds", 4))
        self.temperature = float(kwargs.get("temperature", 0.6))

    def solve(self, problem: str, metadata: dict) -> dict:
        try:
            result = asyncio.run(solve_problem(
                self.client, problem,
                n_chains=self.n_chains, max_rounds=self.max_rounds,
                temperature=self.temperature))
        except Exception as exc:  # 兜底：保证返回合法 dict，不抛给评测器
            return {"final_response": "",
                    "trace": [{"step": "error",
                               "content": f"{type(exc).__name__}: {exc}"}]}
        # 契约：final_response 必须是非空字符串
        if not isinstance(result.get("final_response"), str):
            result["final_response"] = str(result.get("final_response") or "")
        return result
