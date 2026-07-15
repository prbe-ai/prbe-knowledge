"""Per-language tree-sitter extractors.

Importing this package triggers the registration of every concrete extractor
via `_register`. The package's public API is `get_extractor_for_file()` and
`registered_extractors()`.

Languages in PR-A:
    Python      (deep)
    TypeScript  (smoke)
    JavaScript  (smoke)
    Go          (smoke)
    Java        (smoke)

Swift is deferred — `tree-sitter-swift` on PyPI is a placeholder (v0.0.1)
and not production-ready as of 2026-05. Follow-up with a vendored grammar.
"""

from __future__ import annotations

from services.ingestion.code_graph.extractors import (
    go,  # noqa: F401
    java,  # noqa: F401
    javascript,  # noqa: F401
    python,  # noqa: F401
    typescript,  # noqa: F401
)
from services.ingestion.code_graph.extractors.registry import (
    get_extractor_for_file,
    registered_extractors,
)

__all__ = ["get_extractor_for_file", "registered_extractors"]
