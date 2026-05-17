"""Trace analyzer CLI: build a JSONL digest of yesterday's traces.

Invocation (from the in-cluster K8s Job, see
k8s/jobs/nightly-trace-digest.yaml):

    uv run python -m services.retrieval.agent.trace_analyzer \
        --date 2026-05-17 \
        --out /tmp/digests.jsonl

The job redirects stdout to a file inside the pod and the workflow
streams it back with `kubectl logs`. No new R2 bucket is needed; the
digest is workflow-ephemeral.

One JSONL line per trace. See `digest.summarize_trace` for the shape.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date as date_cls

from services.retrieval.agent.trace_analyzer.digest import summarize_trace
from services.retrieval.agent.trace_analyzer.loader import iter_trace_blobs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="trace_analyzer")
    parser.add_argument(
        "--date",
        required=True,
        help="UTC date to digest, e.g. 2026-05-17",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output JSONL path. '-' (default) writes to stdout.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    try:
        target = date_cls.fromisoformat(args.date)
    except ValueError as exc:
        print(f"error: invalid --date: {exc}", file=sys.stderr)
        return 2

    out_stream = sys.stdout
    file_handle = None
    if args.out != "-":
        # Context manager doesn't fit here because the open is
        # conditional and the close needs to be skipped for stdout.
        file_handle = open(args.out, "w", encoding="utf-8")  # noqa: SIM115
        out_stream = file_handle

    yielded = 0
    try:
        async for blob in iter_trace_blobs(target):
            digest = summarize_trace(blob)
            out_stream.write(json.dumps(digest, default=str) + "\n")
            yielded += 1
    finally:
        if file_handle is not None:
            file_handle.close()

    print(f"trace_analyzer: wrote {yielded} digest line(s) for {target}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
