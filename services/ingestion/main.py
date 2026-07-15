"""Thin deploy wrapper — the ingestion app now lives in kb.ingestion_app.

Kept so existing entrypoints (`uvicorn services.ingestion.main:app` in
docker-compose, the community Helm chart, and the hosted data-plane charts)
keep working unchanged across the engine/ + kb/ split.
"""

from kb.ingestion_app import app

__all__ = ["app"]

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "services.ingestion.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
