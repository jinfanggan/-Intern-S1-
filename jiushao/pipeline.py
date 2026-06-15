"""单题流水线：按 profile 组织 求解(N 链) → 聚合投票 → 判分，全程写事件日志。

求解器从 solver.SOLVERS 注册表查取，profile 声明用哪个 solver 及其参数
（solver_kwargs）——新增对比实验无需改本文件。
"""
import asyncio
import time

from .aggregate import cluster_and_vote
from .config import PROFILES
from .dsets import Problem, classify_domain
from .judge import judge_with_meta
from .llm import LLM
from . import oracle_loop  # noqa: F401  触发 oracle-tir 注册
from .solver import SOLVERS


async def run_problem(llm: LLM, problem: Problem, profile_name: str, emit) -> dict:
    profile = PROFILES[profile_name]
    n_chains = profile["n_chains"]
    t0 = time.time()
    emit("input", "problem", question=problem.question, gold=problem.answer,
         subject=problem.subject, profile=profile_name)

    # -- 求解：N 链并行（求解器由注册表查取） ---------------------------------
    solver = SOLVERS[profile["solver"]]
    skw = profile.get("solver_kwargs", {})
    temperature = profile.get("temperature", 0.7 if n_chains > 1 else None)
    chains = await asyncio.gather(*[
        solver(llm, problem.question, chain_id=i, temperature=temperature,
               emit=emit, **skw)
        for i in range(n_chains)
    ], return_exceptions=True)
    ok_chains = [c for c in chains if isinstance(c, dict)]
    for c in chains:
        if isinstance(c, Exception):
            emit("solve", "chain_error", error=repr(c)[:500])

    # -- 聚合投票 -------------------------------------------------------------
    answers = [c["answer"] for c in ok_chains]
    if n_chains == 1:
        final = answers[0] if answers else None
        emit("aggregate", "single", answer=final)
    else:
        agg = cluster_and_vote(answers)
        final = agg["final"]
        emit("aggregate", "vote", clusters=agg["clusters"], final=final)

    # -- 判分 -----------------------------------------------------------------
    correct = judge_with_meta(final, problem.answer, problem.meta)
    emit("verdict", "judge", pred=final, gold=problem.answer, correct=correct,
         elapsed=round(time.time() - t0, 1))

    return {
        "id": problem.id, "subject": problem.subject,
        "domain": classify_domain(problem), "profile": profile_name,
        "pred": final, "gold": problem.answer, "correct": correct,
        "n_chains_ok": len(ok_chains), "n_chains": n_chains,
        "elapsed": round(time.time() - t0, 1),
    }
