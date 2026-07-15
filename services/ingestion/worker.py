"""Thin deploy wrapper — the composed worker process now lives in kb.worker.

Kept so `python -m services.ingestion.worker` (docker-compose, Helm worker
deployment, hosted data-plane charts) keeps working unchanged across the
engine/ + kb/ split. The generic queue-drain Worker/ReclaimLoop classes are
in engine.ingest.worker; kb.worker composes them with the integration
pieces (backfills, pollers, Granola listener).
"""

from kb.worker import run_worker_forever

__all__ = ["run_worker_forever"]

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(run_worker_forever())
