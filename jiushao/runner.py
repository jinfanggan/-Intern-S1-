"""批量评测 runner：并发 + 断点续跑 + 实时进度 + 分领域报告。"""
import asyncio
from collections import defaultdict

from .dsets import load
from .llm import LLM
from .pipeline import run_problem
from .runlog import RunLogger


async def run_eval(dataset: str, model: str, profile: str, *,
                   limit: int | None = None, offset: int = 0, concurrency: int = 4,
                   run_name: str | None = None) -> dict:
    problems = load(dataset, None)[offset:offset + limit if limit else None]
    run_name = run_name or f"{dataset}-{model}-{profile}" + (f"-n{limit}" if limit else "")
    logger = RunLogger(run_name, meta={
        "dataset": dataset, "model": model, "profile": profile,
        "limit": limit, "total": len(problems)})
    done = logger.done_ids()
    todo = [p for p in problems if p.id not in done]
    print(f"[{run_name}] 共 {len(problems)} 题，已完成 {len(done)}，待跑 {len(todo)}")

    llm = LLM(model)
    sem = asyncio.Semaphore(concurrency)
    n_done = 0

    async def one(p):
        nonlocal n_done
        async with sem:
            emit = logger.event_writer(p.id)
            try:
                result = await run_problem(llm, p, profile, emit)
            except Exception as e:
                result = {"id": p.id, "subject": p.subject, "profile": profile,
                          "pred": None, "gold": p.answer, "correct": False,
                          "error": repr(e)[:300]}
                emit("verdict", "error", error=repr(e)[:500])
            finally:
                emit._close()
            logger.write_result(result)
            n_done += 1
            mark = "✓" if result["correct"] else "✗"
            print(f"  [{n_done}/{len(todo)}] {mark} {p.id}  "
                  f"pred={str(result['pred'])[:40]!r} gold={str(result['gold'])[:40]!r}")
            return result

    await asyncio.gather(*[one(p) for p in todo])
    return report(logger, llm)


def report(logger: RunLogger, llm: LLM | None = None) -> dict:
    results = logger.load_results()
    if not results:
        return {}
    total = len(results)
    correct = sum(r["correct"] for r in results)
    from .dsets import DOMAIN_LABELS
    by_domain = defaultdict(lambda: [0, 0])
    by_subject = defaultdict(lambda: [0, 0])
    for r in results:
        d = r.get("domain") or "other"
        by_domain[d][0] += r["correct"]
        by_domain[d][1] += 1
        s = r.get("subject") or "unknown"
        by_subject[s][0] += r["correct"]
        by_subject[s][1] += 1
    print(f"\n===== 报告：{logger.root.name} =====")
    print(f"总体: {correct}/{total} = {correct/total:.1%}")
    print("— 按答案形态大类（哪个机制该发力）—")
    for d, (c, n) in sorted(by_domain.items(), key=lambda kv: kv[1][0]/kv[1][1]):
        print(f"  {DOMAIN_LABELS.get(d, d):<12} {c}/{n} = {c/n:.1%}")
    print("— 按原始 subject 细分 —")
    for s, (c, n) in sorted(by_subject.items(), key=lambda kv: kv[1][0]/kv[1][1]):
        print(f"  {s:<28} {c}/{n} = {c/n:.1%}")
    if llm:
        u = llm.usage
        print(f"API: {u['calls']} 次调用, {u['cache_hits']} 次缓存命中, "
              f"{u['in']/1000:.1f}k in / {u['out']/1000:.1f}k out, "
              f"约 ${llm.cost_usd():.4f}")
    return {"accuracy": correct / total, "total": total,
            "by_domain": {k: (v[0], v[1]) for k, v in by_domain.items()},
            "by_subject": {k: (v[0], v[1]) for k, v in by_subject.items()}}
