"""Thin deploy wrapper — canonical module: kb.synthesis.triage_app.

Kept so `python -m services.synthesis.triage_app` (wiki-worker app) keeps
working unchanged.
"""

from kb.synthesis.triage_app import run_triage_app_forever

__all__ = ["run_triage_app_forever"]

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(run_triage_app_forever())
