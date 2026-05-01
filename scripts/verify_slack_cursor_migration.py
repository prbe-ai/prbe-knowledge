"""Pre-deploy migration safety check for the Slack backfill round-robin rewrite.

USAGE:
    uv run python scripts/verify_slack_cursor_migration.py < prod_cursors.json

Where `prod_cursors.json` is the output of:

    SELECT json_agg(json_build_object(
        'customer_id', customer_id,
        'status', status,
        'cursor', last_cursor::json,
        'events_enqueued', events_enqueued
    ))
    FROM backfill_state
    WHERE source_system = 'slack'
      AND status IN ('running', 'pending')
      AND last_cursor IS NOT NULL;

The script reads each cursor, runs it through `_decode_slack_cursor` (the new
migration logic), and prints what the round-robin walker would do on resume.
Any cursor that produces an empty `active` map when the source had channels in
flight is a data-loss risk and is flagged.
"""

from __future__ import annotations

import json
import sys

from services.ingestion.handlers.slack import _decode_slack_cursor


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        print("ERROR: no input on stdin (pipe the SQL output in)", file=sys.stderr)
        return 2

    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: input is not JSON: {exc}", file=sys.stderr)
        return 2

    if rows is None:
        print("OK: no in-flight Slack backfills — migration is a no-op")
        return 0

    flagged = 0
    print(f"Inspecting {len(rows)} in-flight Slack backfill(s)\n")

    for row in rows:
        cursor_obj = row.get("cursor")
        cursor_str = json.dumps(cursor_obj) if cursor_obj is not None else None
        cust = row.get("customer_id", "?")
        status = row.get("status", "?")
        enq = row.get("events_enqueued", 0)

        old_channels: set[str] = set()
        if isinstance(cursor_obj, dict):
            old_channels |= {ch for ch in (cursor_obj.get("channels_remaining") or []) if ch}
            cur = cursor_obj.get("current_channel")
            if cur:
                old_channels.add(cur)
            # already-new shape
            if isinstance(cursor_obj.get("active"), dict):
                old_channels |= set(cursor_obj["active"].keys())

        migrated = _decode_slack_cursor(cursor_str)
        new_channels = set(migrated["active"].keys())

        lost = old_channels - new_channels
        kept = old_channels & new_channels

        verdict = "OK"
        if lost:
            verdict = "FLAG"
            flagged += 1
        elif not new_channels and old_channels:
            verdict = "FLAG"
            flagged += 1

        print(f"[{verdict}] customer={cust} status={status} enqueued={enq}")
        print(f"       channels in old cursor: {sorted(old_channels)}")
        print(f"       channels in new cursor: {sorted(new_channels)}")
        if lost:
            print(f"       *** LOST CHANNELS: {sorted(lost)} ***")
        cursor_for_C = {ch: migrated["active"].get(ch) for ch in sorted(kept)}
        print(f"       per-channel page cursors: {cursor_for_C}")
        print()

    if flagged:
        print(f"\n{flagged} cursor(s) flagged — DO NOT DEPLOY until investigated.")
        return 1
    print(f"\nAll {len(rows)} in-flight backfill(s) round-trip cleanly. Safe to deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
