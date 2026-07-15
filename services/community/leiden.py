"""Thin deploy wrapper — canonical module: engine.community.leiden.

Kept so `python -m services.community.leiden` (community/leiden cron)
keeps working unchanged.
"""

from engine.community.leiden import main

__all__ = ["main"]

if __name__ == "__main__":  # pragma: no cover
    main()
