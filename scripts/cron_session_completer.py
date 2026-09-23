"""End agent sessions (Claude Code, Codex, pi) whose client never said goodbye.

Appends a finalize.marker key to each idle session's live queue row. The worker,
on next claim, sees an end signal on top of the row and mines the session once
-- which is the ONLY thing that produces the qa / code_change / decision /
file_ref / directive unit docs. A session that is never ended is captured but
never mined.

The tap ends a session itself when it ends cleanly or when Claude Code dies
under it, so the common case needs nothing here. This is the backstop for the
cases that leave no goodbye at all: a laptop that slept and never woke, a
machine that lost the network before its last drain.

Two schedules run it: research hourly with `--idle-minutes 1440` (a CronJob in
research-os), and the managed plane nightly with 360 (session-finalizer-nightly.yml,
opt-in via SESSION_FINALIZER_ENABLED). A session that resumes after
being ended is simply live again (the end signal is no longer the newest key),
and is re-ended and re-mined once when it next goes idle. Sessions whose newest
key is already an end signal are skipped, so re-running is free: the sweep ends
each idle session once, not once per run.
"""
from __future__ import annotations

import argparse
import asyncio

from engine.shared.config import get_settings
from engine.shared.db import init_pool
from kb.session_completer import enqueue_idle_session_finalizers

#: Same floor as session-finalizer-nightly.yml.
MIN_IDLE_MINUTES = 60


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idle-minutes",
        type=int,
        default=None,
        help=(
            "Finalize sessions with no ingest activity for this many minutes. "
            "Omit to use claude_code_session_idle_minutes from settings."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help=(
            "Cap on sessions finalized per source per run. Every one buys a "
            "full multi-segment extraction, so this is a cost bound. Work not "
            "reached this run is reached on the next."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count what would be finalized and write nothing.",
    )
    args = parser.parse_args(argv)
    if args.idle_minutes is not None and args.idle_minutes < 1:
        parser.error("--idle-minutes must be >= 1")
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def resolve_idle_minutes(explicit: int | None) -> int:
    """The window to use: the flag when given, else the configured default.

    Split out so the precedence is testable without a live pool — getting this
    backwards would silently finalize live sessions at the 5-minute default
    during a scheduled run.
    """
    if explicit is not None:
        return explicit
    return get_settings().claude_code_session_idle_minutes


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    await init_pool()
    try:
        idle_minutes = resolve_idle_minutes(args.idle_minutes)
        if idle_minutes < MIN_IDLE_MINUTES:
            # Ending a session mines it. A short window ends sessions people
            # have only paused, and every pause then costs a full re-mine.
            raise SystemExit(
                f"refusing --idle-minutes {idle_minutes}: below {MIN_IDLE_MINUTES}. "
                "The settings default (claude_code_session_idle_minutes) is for "
                "tests; production passes the window explicitly."
            )
        n = await enqueue_idle_session_finalizers(
            idle_minutes, limit=args.limit, dry_run=args.dry_run
        )
        verb = "would enqueue" if args.dry_run else "enqueued"
        print(
            f"{verb} {n} finalize events "
            f"(idle_minutes={idle_minutes} limit={args.limit})"
        )
    finally:
        from engine.shared.db import close_pool

        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
