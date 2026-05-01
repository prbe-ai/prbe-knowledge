"""CLI dispatch for the synth tool. Subcommands grow over plans 1-3."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.synth",
        description="Synthetic company corpus generator for prbe-knowledge eval datasets.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser(
        "extract",
        help="Extract WorldModel from repos in a profile (no DB writes).",
    )
    extract.add_argument("--profile", required=True, type=str, help="Path to profile YAML.")
    extract.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write world_model.json (default: eval-datasets/<run-id>/).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "extract":
        # Plan 1 task 21 wires this in. For now, surface a clear stub error.
        print("extract: not yet implemented", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
