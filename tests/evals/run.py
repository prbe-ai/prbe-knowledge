"""Eval runner stub.

Phase 0 scaffold. Full datasets + scoring land in Tier 10 Phase 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "datasets"


async def run(eval_name: str, limit: int) -> None:
    dataset = EVAL_DIR / f"{eval_name}.jsonl"
    if not dataset.exists():
        print(f"dataset missing: {dataset}")
        return

    cases = [json.loads(line) for line in dataset.read_text().splitlines()[:limit]]
    print(f"would run {len(cases)} cases for eval {eval_name!r} (stub)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, choices=["entity_extractor", "query_expansion"])
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(run(args.eval, args.limit))


if __name__ == "__main__":
    main()
