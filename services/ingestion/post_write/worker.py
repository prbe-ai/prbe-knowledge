"""Thin deploy wrapper — canonical module: engine.ingest.post_write.worker.

Kept so `python -m services.ingestion.post_write.worker` keeps working
unchanged.
"""

from engine.ingest.post_write.worker import run_worker_forever

__all__ = ["run_worker_forever"]

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(run_worker_forever())
