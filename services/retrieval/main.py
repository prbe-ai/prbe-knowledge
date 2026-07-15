"""Thin deploy wrapper — the retrieval app now lives in engine.retrieval.main.

Kept so `uvicorn services.retrieval.main:app` (docker-compose, Helm,
sandbox, hosted data-plane charts) keeps working unchanged.

This wrapper is also the composition root: importing kb.handlers fires the
@register_connector decorators, which populate the engine source registry
(per-source score multiplier / half-life / doc_type prefix) that fusion and
the doc-type resolver read. engine/ itself never imports kb/.
"""

import kb.handlers  # noqa: F401  (source-profile registration side effect)
from engine.retrieval.main import app

__all__ = ["app"]

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "services.retrieval.main:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
    )
