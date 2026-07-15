"""Thin deploy wrapper — canonical module: kb.synthesis.synthesis_app.

Kept so `python -m services.synthesis.synthesis_app` (wiki-synthesis app)
keeps working unchanged.
"""

from kb.synthesis.synthesis_app import run_synthesis_app_forever

__all__ = ["run_synthesis_app_forever"]

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(run_synthesis_app_forever())
