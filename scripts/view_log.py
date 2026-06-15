#!/usr/bin/env python3
"""按题回放事件日志，终端友好版（前端推理树的雏形）。

用法：
    python scripts/view_log.py                          # 列出所有 run
    python scripts/view_log.py <run_name>               # 列出该 run 的所有题及对错
    python scripts/view_log.py <run_name> <题id>        # 回放该题完整解题过程
    python scripts/view_log.py <run_name> <题id> --full # 不截断模型输出
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jiushao.config import RUNS_ROOT  # noqa: E402

R, G, Y, B, DIM, RST = "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[2m", "\033[0m"


def list_runs():
    runs = sorted(p.name for p in RUNS_ROOT.iterdir()
                  if p.is_dir() and p.name != "cache")
    print("可用 run：")
    for r in runs:
        n = len(list((RUNS_ROOT / r / "events").glob("*.jsonl"))) if (RUNS_ROOT / r / "events").exists() else 0
        print(f"  {r}  ({n} 题)")


def list_problems(run: str):
    path = RUNS_ROOT / run / "results.jsonl"
    for line in path.read_text().splitlines():
        d = json.loads(line)
        mark = f"{G}✓{RST}" if d["correct"] else f"{R}✗{RST}"
        print(f"  {mark} {d['id']:<40} pred={str(d['pred'])[:30]!r:<34} gold={str(d['gold'])[:25]!r}")


def show(text: str, full: bool, limit: int = 1200):
    text = text.strip()
    if not full and len(text) > limit:
        text = text[:limit] + f"\n{DIM}……(截断，--full 查看全文){RST}"
    print("    " + text.replace("\n", "\n    "))


def replay(run: str, pid: str, full: bool):
    safe = pid.replace("/", "_").replace("\\", "_")
    path = RUNS_ROOT / run / "events" / (safe + ".jsonl")
    if not path.exists():
        cands = [p.stem for p in (RUNS_ROOT / run / "events").glob(f"*{safe}*")]
        if len(cands) == 1:
            path = RUNS_ROOT / run / "events" / (cands[0] + ".jsonl")
        else:
            print(f"找不到题 {pid}，候选：{cands[:10]}")
            return
    events = [json.loads(l) for l in path.read_text().splitlines()]
    for e in events:
        st, ty = e["stage"], e["type"]
        if ty == "problem":
            print(f"\n{B}━━━ 题目 [{e.get('subject','')}] profile={e.get('profile')} ━━━{RST}")
            show(e["question"], full)
            print(f"  {DIM}标准答案: {e['gold']}{RST}")
        elif ty == "chain_round":
            cached = " (缓存)" if e.get("cached") else ""
            print(f"\n{Y}── 链{e['chain']} 第{e['round']}轮{cached} "
                  f"[{e.get('usage_in',0)}in/{e.get('usage_out',0)}out] ──{RST}")
            show(e.get("content", ""), full)
        elif ty == "exec":
            ok = f"{G}成功{RST}" if e["ok"] else f"{R}失败{RST}"
            print(f"\n{B}── 沙箱执行（链{e['chain']}）{ok} ──{RST}")
            print(f"    {DIM}代码:{RST}")
            show(e.get("code", ""), full, 600)
            out = e.get("stdout") or e.get("stderr") or "(无输出)"
            print(f"    {DIM}输出:{RST}")
            show(out, full, 400)
        elif ty == "chain_done":
            print(f"  {DIM}链{e['chain']} 结束，答案 = {e.get('answer')!r}（{e.get('rounds')} 轮）{RST}")
        elif ty == "chain_error":
            print(f"  {R}链异常: {e.get('error')}{RST}")
        elif ty in ("vote", "single"):
            if ty == "vote":
                print(f"\n{Y}━━━ 聚合投票 ━━━{RST}")
                for ans, n in e.get("clusters", []):
                    print(f"    {n} 票 ← {ans!r}")
            print(f"  最终答案: {e.get('final', e.get('answer'))!r}")
        elif ty == "judge":
            mark = f"{G}✓ 正确{RST}" if e["correct"] else f"{R}✗ 错误{RST}"
            print(f"\n{B}━━━ 判分 {mark}  pred={e['pred']!r}  gold={e['gold']!r}"
                  f"  耗时 {e.get('elapsed')}s ━━━{RST}")
        elif ty == "error":
            print(f"  {R}整题异常: {e.get('error')}{RST}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--full"]
    full = "--full" in sys.argv
    if not args:
        list_runs()
    elif len(args) == 1:
        list_problems(args[0])
    else:
        replay(args[0], args[1], full)
