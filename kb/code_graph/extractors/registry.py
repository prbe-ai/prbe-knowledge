"""Extractor registry — picks the right Extractor for a file path.

Each language module calls `register(extractor)` at import. The registry
asserts each registered instance satisfies the `Extractor` Protocol via
`@runtime_checkable` — drift surfaces immediately at boot, not at first
extract() call.
"""

from __future__ import annotations

from kb.code_graph.types import Extractor

_BY_EXTENSION: dict[str, Extractor] = {}
_ALL: list[Extractor] = []


def register(extractor: Extractor) -> None:
    if not isinstance(extractor, Extractor):
        raise TypeError(
            f"object {extractor!r} does not satisfy Extractor Protocol"
        )
    _ALL.append(extractor)
    for ext in extractor.file_extensions:
        existing = _BY_EXTENSION.get(ext)
        if existing is not None and existing is not extractor:
            raise ValueError(
                f"extension {ext!r} double-registered: "
                f"{existing.language!r} and {extractor.language!r}"
            )
        _BY_EXTENSION[ext] = extractor


def get_extractor_for_file(file_path: str) -> Extractor | None:
    """Return the extractor for the file, or None if no language matches."""
    # Match the longest-suffix extension first so '.tsx' wins over '.ts'.
    candidates = sorted(_BY_EXTENSION.keys(), key=len, reverse=True)
    for ext in candidates:
        if file_path.endswith(ext):
            return _BY_EXTENSION[ext]
    return None


def registered_extractors() -> list[Extractor]:
    return list(_ALL)


__all__ = ["get_extractor_for_file", "register", "registered_extractors"]
