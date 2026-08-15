"""Finalize agent sessions (Claude Code, Codex) whose client never said goodbye.

Writes a finalize.marker into each idle session's live queue row. The worker,
on next claim, sees the marker and runs the `session_complete=True` path —
which is the ONLY thing that produces the qa / code_change / decision /
file_ref unit docs. A session that is never finalized is captured but never
mined.

TWO SCHEDULES, ONE SCRIPT
-------------------------
The tap sends an explicit finalize when a session ends cleanly, so the common
case needs nothing here. This script is the backstop for the cases that leave
no goodbye at all: a hard-killed terminal, a laptop that slept and never woke,
a crashed daemon, a machine that lost the network before its last drain.

* NIGHTLY SWEEP (`--idle-minutes 360`, knowledge-cron.yml) — the safety net.
  Six hours is deliberately far above any think-time gap: a session a
  researcher is still using between meetings must not be finalized out from
  under them, because a session that re-activates after finalize re-runs the
  extraction LLM on every subsequent batch (the marker key is sticky, so
  `complete` stays true forever after).
* FAST CADENCE (no flag → `claude_code_session_idle_minutes`, default 5) —
  what a per-minute runner would use if one is ever added. Left as the default
  so the flagless invocation keeps its historical meaning.

Idempotent: sessions whose live row already carries a finalize.marker are
filtered out by the finder query, so re-running is free and a nightly sweep
cannot double-charge the extractor.
"""
from __future__ import annotations

import argparse
import asyncio

from engine.shared.config import get_settings
from engine.shared.db import init_pool
from kb.session_completer import enqueue_idle_session_finalizers


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
    args = parser.parse_args(argv)
    if args.idle_minutes is not None and args.idle_minutes < 1:
        parser.error("--idle-minutes must be >= 1")
    return args


def resolve_idle_minutes(explicit: int | None) -> int:
    """The window to use: the flag when given, else the configured default.

    Split out so the precedence is testable without a live pool — getting this
    backwards would silently finalize live sessions at the 5-minute default
    during a nightly run.
    """
    if explicit is not None:
        return explicit
    return get_settings().claude_code_session_idle_minutes


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    await init_pool()
    try:
        idle_minutes = resolve_idle_minutes(args.idle_minutes)
        n = await enqueue_idle_session_finalizers(idle_minutes)
        print(f"enqueued {n} finalize events (idle_minutes={idle_minutes})")
    finally:
        from engine.shared.db import close_pool

        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
