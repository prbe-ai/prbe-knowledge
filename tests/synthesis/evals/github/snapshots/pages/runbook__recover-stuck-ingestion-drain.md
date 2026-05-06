---
title: "Recover from a stuck ingestion drain"
slug: recover-stuck-ingestion-drain
wiki_type: runbook
last_updated: 2026-04-17T09:45:00Z
owners: [person:maison]
---

# Recover from a stuck ingestion drain

When an ingestion drain wedges in the `running` state and the next cron
tick does not pick it back up, follow these steps.

## Steps

1. Check `wiki_synthesis_runs` for rows in `running` state older than
   one hour.
2. Run the reclaim script (`scripts/reclaim_stuck_runs.py`) — it flips
   stuck runs to `failed`.
3. The drain restarts on the next cron tick.
4. If the drain was wedged on a single bad event, mark that queue row
   `synthesis_skipped` with `synthesis_error='manual_skip'`.

## History

[[person:maison]] hit this in production twice and captured the steps
in [issue #19](https://github.com/prbe-ai/prbe-knowledge/issues/19).

## Sources

- [Issue #19: runbook: how to recover from a stuck ingestion drain](https://github.com/prbe-ai/prbe-knowledge/issues/19)
