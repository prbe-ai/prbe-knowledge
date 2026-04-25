"""Entry point: `python -m services.ingestion.poller`.

Kept thin so the actual scheduler logic stays in __init__.py and is testable
without invoking the asyncio bootstrap.
"""

from __future__ import annotations

import asyncio

from services.ingestion.poller import run_poller_forever

if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_poller_forever())
