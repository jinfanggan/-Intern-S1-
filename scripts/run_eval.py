#!/usr/bin/env python3
"""评测入口。

示例：
    python scripts/run_eval.py --dataset math500 --model gpt-5-nano --profile b0 --limit 10
    python scripts/run_eval.py --dataset theoremqa --model gpt-5-mini --profile b2 --limit 50
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jiushao.config import MODEL_PROFILES, PROFILES
from jiushao.dsets import LOADERS
from jiushao.runner import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--model", default="gpt-5-nano", choices=list(MODEL_PROFILES))
    ap.add_argument("--profile", default="b0", choices=list(PROFILES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    asyncio.run(run_eval(args.dataset, args.model, args.profile,
                         limit=args.limit, offset=args.offset, concurrency=args.concurrency,
                         run_name=args.run_name))


if __name__ == "__main__":
    main()
