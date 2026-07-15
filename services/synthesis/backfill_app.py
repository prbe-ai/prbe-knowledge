"""Thin deploy wrapper — canonical module: kb.synthesis.backfill_app.

Kept so `python -m services.synthesis.backfill_app` (wiki backfill app)
keeps working unchanged.
"""

from kb.synthesis.backfill_app import run_backfill_app_forever

__all__ = ["run_backfill_app_forever"]

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(run_backfill_app_forever())
