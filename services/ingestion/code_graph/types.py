"""Shared data types for the code-graph extraction pipeline.

The Extractor Protocol is `@runtime_checkable` so the registry can assert
each registered language module satisfies the contract at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from shared.constants import EdgeType, NodeLabel


@dataclass(slots=True)
class Symbol:
    """One extracted symbol (function, method, class, module).

    `def_line` is ALWAYS the line of the `def` / `class` keyword, never
    the decorator line above it. Per-language extractors enforce this so
    that adding/removing decorators doesn't change the symbol's identity.
    See spec §4.3 for the full doc_id format.
    """

    qualified_name: str          # e.g. "module.submodule.Class.method"
    kind: NodeLabel              # FUNCTION | METHOD | CLASS | MODULE
    file_path: str               # repo-relative
    def_line: int                # 1-indexed; the `def`/`class` keyword line
    end_line: int                # 1-indexed; last line of the def block
    source_snippet: str          # body text used as Document.metadata['body']
    signature: str = ""          # e.g. "def normalize(event, hydrated)"
    docstring: str | None = None
    parent_qname: str | None = None  # for Method nodes, the enclosing class qname


@dataclass(slots=True)
class CodeEdge:
    """One extracted relationship between two symbols (or a symbol + module/repo).

    `target_qname` may be unresolved when `ambiguous=True` — in that case
    `target_candidates` lists the qualified-name candidates the resolver
    couldn't disambiguate. PR-B's promoter narrows AMBIGUOUS → INFERRED.
    """

    edge_type: EdgeType          # CALLS | IMPORTS | INHERITS | ...
    from_qname: str
    to_qname: str                # canonical target; may be a candidate alias
    ambiguous: bool = False
    target_candidates: list[str] = field(default_factory=list)
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractResult:
    """Per-file output of an Extractor.

    `errors` carries non-fatal extraction issues (timeout, partial parse)
    so the connector can record them in `code_repo_state` without failing
    the whole batch. Empty by default.
    """

    symbols: list[Symbol] = field(default_factory=list)
    edges: list[CodeEdge] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class Extractor(Protocol):
    """Contract every per-language extractor implements.

    `language` is the canonical id used for `documents.language` and
    `code_repo_state.language`. `extractor_version` is bumped per-language
    whenever the extractor's behavior changes — bumping invalidates that
    language's cache entries on next push.

    Implementations should treat `extract` as pure: same (file_path,
    content) inputs → same outputs. The connector enforces a per-file
    timeout (10s default) above this layer; extractors do not need to
    implement their own timeout.
    """

    language: str
    extractor_version: str
    file_extensions: tuple[str, ...]   # e.g. (".py",) or (".ts", ".tsx")

    def extract(
        self,
        file_path: str,
        content: bytes,
        repo_root: str,
    ) -> ExtractResult: ...


__all__ = [
    "CodeEdge",
    "ExtractResult",
    "Extractor",
    "Symbol",
]
