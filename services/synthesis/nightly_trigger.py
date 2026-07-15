"""Thin deploy wrapper — canonical module: kb.synthesis.nightly_trigger.

Kept so `python -m services.synthesis.nightly_trigger` (docker-compose cron
profile, services/cron Dockerfile, knowledge-cron workflow) keeps working
unchanged.
"""

from kb.synthesis.nightly_trigger import main

__all__ = ["main"]

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(main())
