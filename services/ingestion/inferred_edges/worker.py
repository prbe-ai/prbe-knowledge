"""Thin deploy wrapper — canonical module: engine.ingest.inferred_edges.worker.

Kept so `python -m services.ingestion.inferred_edges.worker` (see the
Dockerfile in this directory) keeps working unchanged.
"""

from engine.ingest.inferred_edges.worker import run_worker_forever

__all__ = ["run_worker_forever"]

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(run_worker_forever())
