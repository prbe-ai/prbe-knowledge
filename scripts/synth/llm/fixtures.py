"""FixtureStore — file-backed replay/record store for LLM responses.

Layout: {root}/{provider.value}/{key}.json
Each file contains the raw response dict ({"text": "..."} for text,
or the structured output dict for generate_structured calls).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.synth.llm.base import Provider


class FixtureStore:
    """Reads and writes per-provider fixture JSON files."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def path_for(self, provider: Provider, key: str) -> Path:
        """Return the fixture file path for (provider, key)."""
        return self._root / provider.value / f"{key}.json"

    def load(self, provider: Provider, key: str) -> dict | None:
        """Load and return the fixture dict, or None if it does not exist."""
        p = self.path_for(provider, key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def record(self, provider: Provider, key: str, response: dict) -> None:
        """Write *response* to the fixture file, creating parent dirs as needed."""
        p = self.path_for(provider, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(response, indent=2), encoding="utf-8")
