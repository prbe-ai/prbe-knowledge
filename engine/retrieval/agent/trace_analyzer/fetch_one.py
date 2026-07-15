"""Fetch one trace blob from R2 by (bucket, key) and print the JSON.

The nightly orchestrator's sub-agents need to read the full per-turn
transcript of cited traces. R2 access requires the data-plane R2 creds
which live in the in-cluster `managed-data-plane-secrets`. Rather than
plumbing those creds into the GH Actions runner (more secret surface,
more rotation), sub-agents invoke this module via:

    kubectl exec deploy/managed-retrieval -- python -m \
      services.retrieval.agent.trace_analyzer.fetch_one \
      --bucket prbe-acme \
      --key search-traces/2026-05-17/q-1779062199597.json.gz

The output is pretty-printed JSON (the gunzipped blob), written to
stdout. Errors go to stderr with a non-zero exit code.

Why a dedicated module: the prior approach (sub-agent runs `wrangler
r2 object get`) failed on the first dispatch because wrangler isn't
pre-installed on `ubuntu-latest` and no R2 creds were exposed to the
runner anyway. kubectl-exec uses the cluster's existing creds path —
no new secrets surface.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys

from shared.exceptions import StorageNotFound, StorageUnavailable
from shared.storage import get_store


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trace_analyzer.fetch_one",
        description="Fetch one trace blob from R2 and print its JSON.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="R2 bucket name, e.g. prbe-acme",
    )
    parser.add_argument(
        "--key",
        required=True,
        help="Object key, e.g. search-traces/2026-05-17/q-1779062199597.json.gz",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON (indent=2). Default: compact.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    store = get_store()
    try:
        body = await store.get(args.bucket, args.key)
    except StorageNotFound:
        print(
            f"error: blob not found at {args.bucket}/{args.key}",
            file=sys.stderr,
        )
        return 2
    except StorageUnavailable as exc:
        print(f"error: R2 unavailable: {exc}", file=sys.stderr)
        return 3

    try:
        blob = json.loads(gzip.decompress(body))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not decode blob: {exc}", file=sys.stderr)
        return 4

    indent = 2 if args.pretty else None
    print(json.dumps(blob, default=str, indent=indent))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
