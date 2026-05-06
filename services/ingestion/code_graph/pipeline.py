"""Pipeline: turns a batch of file contents into a NormalizationResult.

Used by both the initial-backfill and incremental paths in
`handlers/codegraph.py`. Responsibilities:

  1. Per-file SHA-256 hash. Skip if the existing `code_repo_state` row
     matches (cache hit).
  2. Secrets-skip check (regex floor + filename guards). Files that match
     emit no symbols; cache row stamped with `_skipped_secrets`.
  3. Per-file extractor dispatch with a 10-second per-file timeout. A
     pathological file emits a partial ExtractResult on timeout instead
     of blocking the whole batch. Spec §10 critical gap #3.
  4. Cross-file qualifier pass — promotes single-match AMBIGUOUS edges.
  5. Maps Symbols → Documents (one per symbol), graph_nodes, graph_edges,
     and code_repo_state updates. Returns a single NormalizationResult.

ACL: workspace-level READ keyed by the repo's owner login (derived from
the repo's `org/repo` form). Mirrors the GitHub connector convention so
existing ACL filters work the same way for symbol Documents.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import asyncpg

from services.ingestion.chunker import count_tokens
from services.ingestion.code_graph.extractors import get_extractor_for_file
from services.ingestion.code_graph.qualifier import promote_single_match
from services.ingestion.code_graph.secrets import (
    SKIPPED_LANGUAGE_SENTINEL,
    looks_like_secret_dump,
)
from services.ingestion.code_graph.types import ExtractResult, Symbol
from shared.constants import (
    DocClass,
    DocType,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    CodeRepoStateUpdate,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    NormalizationResult,
)

if TYPE_CHECKING:
    from services.ingestion.code_graph.clone import FileEntry  # noqa: F401

log = get_logger(__name__)

# Per-file timeout for extract(). Spec §10 critical gap #3: a pathological
# file (10k locals from a code generator) shouldn't hang the batch.
PER_FILE_EXTRACT_TIMEOUT_SECONDS = 10.0

# Files larger than this skip extraction entirely. Most >256 KB source files
# are machine-generated (vendored bundles, codegen output) and either timeout
# in tree-sitter or produce useless symbol noise. asyncio.wait_for can't
# cancel the parser thread, so a leaked thread holding a giant AST is the
# real cost we're avoiding here, not the wall-clock time.
_MAX_FILE_BYTES_FOR_EXTRACTION = 256 * 1024

# Permission level for code-graph symbol Documents at the workspace level.
# WORKSPACE / READ matches the GitHub connector's repo-level ACL convention.

_OWNER_FALLBACK = "unknown"


@dataclass(slots=True)
class _PreparedFile:
    """A file that passed the cache + secrets checks and is ready to extract."""

    rel_path: str
    content: bytes
    content_hash: str


async def extract_files_to_result(
    customer_id: str,
    repo: str,
    sha: str,
    files: list,  # list[FileEntry]
    *,
    cached_state: dict[str, str] | None = None,
) -> NormalizationResult:
    """Run the full extraction pipeline over `files`.

    `cached_state` is `{file_path: content_hash}` of what we've already
    extracted. Files whose new hash matches are cache hits — skipped.
    None = fetch state from DB inline.

    Returns a NormalizationResult ready for `Normalizer._persist`. Files
    that hit cache, files that match secrets, and files with no extractor
    don't appear in `documents` but DO appear in `code_repo_state_updates`
    so future pushes can short-circuit.
    """
    if cached_state is None:
        cached_state = await _load_cached_state(customer_id, repo)

    prepared, skipped_secrets, skipped_unsupported = _prepare_files(
        files, cached_state
    )

    extractions: list[tuple[str, ExtractResult, str]] = []  # (rel_path, ExtractResult, language)
    for pf in prepared:
        extractor = get_extractor_for_file(pf.rel_path)
        if extractor is None:
            skipped_unsupported.append(pf)
            continue
        result = await _extract_with_timeout(
            extractor, pf.rel_path, pf.content, repo
        )
        extractions.append((pf.rel_path, result, extractor.language))

    # Cross-file qualifier promotion.
    promote_single_match([r for _, r, _ in extractions])

    # qualified_name → NodeLabel for accurate edge endpoint typing.
    # graph_writer.upsert_edges keys node lookup on (label, canonical_id), so a
    # CALLS edge with from_label=FUNCTION whose actual node was written as
    # METHOD silently misses. Build the map once across all files so any
    # in-repo edge endpoint resolves to the correct label.
    qname_to_kind: dict[str, NodeLabel] = {}
    for _, result, _ in extractions:
        for symbol in result.symbols:
            key = symbol.file_path if symbol.kind == NodeLabel.MODULE else symbol.qualified_name
            qname_to_kind[key] = symbol.kind

    documents: list[Document] = []
    nodes: list[GraphNodeSpec] = []
    edges: list[GraphEdgeSpec] = []
    state_updates: list[CodeRepoStateUpdate] = []

    repo_node = GraphNodeSpec(
        label=NodeLabel.REPO,
        canonical_id=repo,
        properties={"name": repo.rsplit("/", 1)[-1]},
    )
    nodes.append(repo_node)
    owner_login = repo.split("/", 1)[0] if "/" in repo else _OWNER_FALLBACK

    now = datetime.now(UTC)

    for rel_path, result, language in extractions:
        # Find the prepared file for this path to get its hash.
        ph = next((p for p in prepared if p.rel_path == rel_path), None)
        if ph is None:
            continue
        symbol_count = len(result.symbols)
        state_updates.append(
            CodeRepoStateUpdate(
                repo=repo,
                file_path=rel_path,
                content_hash=ph.content_hash,
                language=language,
                symbol_count=symbol_count,
                extractor_version=_extractor_version_for(rel_path),
            )
        )
        for symbol in result.symbols:
            documents.append(
                _build_symbol_document(
                    customer_id, repo, sha, owner_login, language, symbol, now
                )
            )
            nodes.append(_symbol_node(repo, symbol))
            # COMPILED_FROM: Document → Symbol-node so LIST queries can
            # walk back from the symbol node to its Document. Spec §4.4.
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.COMPILED_FROM,
                    from_label=NodeLabel.DOCUMENT,
                    from_canonical_id=_doc_id_for_symbol(repo, symbol),
                    to_label=symbol.kind,
                    to_canonical_id=f"{repo}:{symbol.qualified_name}"
                    if symbol.kind != NodeLabel.MODULE
                    else f"{repo}:{symbol.file_path}",
                )
            )
        for edge in result.edges:
            mapped = _map_edge(repo, edge, qname_to_kind)
            if mapped is not None:
                edges.append(mapped)

    # Cache-hit + skipped-secret + unsupported-extension files: stamp their
    # state row so we don't re-attempt them on the next push.
    for pf in skipped_secrets:
        state_updates.append(
            CodeRepoStateUpdate(
                repo=repo,
                file_path=pf.rel_path,
                content_hash=pf.content_hash,
                language=SKIPPED_LANGUAGE_SENTINEL,
                symbol_count=0,
                extractor_version="secrets-v1",
            )
        )
    for pf in skipped_unsupported:
        # Unsupported language — record so next push skips immediately.
        state_updates.append(
            CodeRepoStateUpdate(
                repo=repo,
                file_path=pf.rel_path,
                content_hash=pf.content_hash,
                language="_unsupported",
                symbol_count=0,
                extractor_version="dispatch-v1",
            )
        )

    log.info(
        "code_graph.pipeline.done",
        customer=customer_id,
        repo=repo,
        files_total=len(files),
        files_extracted=len(extractions),
        files_skipped_cache=len(files) - len(prepared) - len(skipped_secrets) - len(skipped_unsupported),
        files_skipped_secrets=len(skipped_secrets),
        files_skipped_unsupported=len(skipped_unsupported),
        symbols=len(documents),
        nodes=len(nodes),
        edges=len(edges),
    )

    return NormalizationResult(
        documents=documents,
        graph_nodes=nodes,
        graph_edges=edges,
        code_repo_state_updates=state_updates,
    )


# ---- helpers --------------------------------------------------------------


def _prepare_files(
    files: list,
    cached_state: dict[str, str],
) -> tuple[list[_PreparedFile], list[_PreparedFile], list[_PreparedFile]]:
    """Hash + cache + secrets check. Returns (to_extract, secrets, _unused_yet)."""
    to_extract: list[_PreparedFile] = []
    skipped_secrets: list[_PreparedFile] = []
    skipped_unsupported: list[_PreparedFile] = []

    for fe in files:
        rel = fe.rel_path
        content = fe.content
        ch = hashlib.sha256(content).hexdigest()
        if cached_state.get(rel) == ch:
            # Cache hit: file unchanged since last extraction. Don't extract.
            # Don't even record a state update (the existing row is current).
            continue
        if len(content) > _MAX_FILE_BYTES_FOR_EXTRACTION:
            # Oversized: most likely generated. Skip extract, stamp the state
            # row downstream so we don't re-attempt next push.
            skipped_unsupported.append(
                _PreparedFile(rel_path=rel, content=content, content_hash=ch)
            )
            continue
        if looks_like_secret_dump(rel, content):
            skipped_secrets.append(
                _PreparedFile(rel_path=rel, content=content, content_hash=ch)
            )
            continue
        to_extract.append(
            _PreparedFile(rel_path=rel, content=content, content_hash=ch)
        )
    # `skipped_unsupported` is filled in by extract_files_to_result after
    # extractor dispatch — files without an extractor are caught there.
    return to_extract, skipped_secrets, skipped_unsupported


async def _extract_with_timeout(
    extractor, rel_path: str, content: bytes, repo: str
) -> ExtractResult:
    """Run extractor.extract with a per-file timeout off the event loop.

    extract() is sync; we run it via to_thread so a slow file doesn't
    block the loop. asyncio.wait_for can't cancel a thread, so the thread
    will finish on its own — but the caller gives up and moves on.
    Acceptable for PR-A: the runaway thread eats CPU briefly until done.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(extractor.extract, rel_path, content, repo),
            timeout=PER_FILE_EXTRACT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        log.warning(
            "code_graph.extract.timeout",
            rel_path=rel_path,
            repo=repo,
            timeout=PER_FILE_EXTRACT_TIMEOUT_SECONDS,
        )
        return ExtractResult(errors=[f"extractor timeout after {PER_FILE_EXTRACT_TIMEOUT_SECONDS}s"])


async def _load_cached_state(customer_id: str, repo: str) -> dict[str, str]:
    """Return `{file_path: content_hash}` for previously-extracted files."""
    async with with_tenant(customer_id) as conn:
        rows: list[asyncpg.Record] = await conn.fetch(
            """
            SELECT file_path, content_hash
            FROM code_repo_state
            WHERE customer_id = $1 AND repo = $2
            """,
            customer_id,
            repo,
        )
    return {r["file_path"]: r["content_hash"] for r in rows}


def _extractor_version_for(rel_path: str) -> str:
    extractor = get_extractor_for_file(rel_path)
    return extractor.extractor_version if extractor else "dispatch-v1"


def _doc_id_for_symbol(repo: str, symbol: Symbol) -> str:
    return f"code_graph:{repo}:{symbol.file_path}:{symbol.qualified_name}:{symbol.def_line}"


def _build_symbol_document(
    customer_id: str,
    repo: str,
    sha: str,
    owner_login: str,
    language: str,
    symbol: Symbol,
    now: datetime,
) -> Document:
    doc_id = _doc_id_for_symbol(repo, symbol)
    body = symbol.source_snippet or symbol.signature or symbol.qualified_name
    body_preview = symbol.docstring.splitlines()[0] if symbol.docstring else symbol.signature
    if body_preview and len(body_preview) > 200:
        body_preview = body_preview[:200]

    content_hash = hashlib.sha256(
        (
            symbol.source_snippet
            + "\x1f"
            + (symbol.signature or "")
            + "\x1f"
            + (symbol.docstring or "")
        ).encode("utf-8", errors="replace")
    ).hexdigest()

    perma = (
        f"https://github.com/{repo}/blob/{sha}/{symbol.file_path}"
        f"#L{symbol.def_line}-L{symbol.end_line}"
    )

    return Document(
        doc_id=doc_id,
        customer_id=customer_id,
        source_system=SourceSystem.CODE_GRAPH,
        source_id=f"symbol:{repo}:{symbol.file_path}:{symbol.qualified_name}:{symbol.def_line}",
        source_url=perma,
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.CODE_SYMBOL,
        content_type="text/plain",
        language=language,
        content_hash=content_hash,
        title=symbol.qualified_name,
        body_preview=body_preview,
        body_size_bytes=len(body.encode("utf-8")),
        body_token_count=count_tokens(body),
        author_id=None,
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.WORKSPACE,
                    principal_id=owner_login,
                    permission=Permission.READ,
                )
            ],
            captured_at=now,
        ),
        # Transient body for the chunker; never persisted into metadata
        # (the storage guard at normalizer.py:191 raises on metadata['body']).
        body=body,
        metadata={
            "symbol": {
                "kind": symbol.kind.value,
                "file_path": symbol.file_path,
                "def_line": symbol.def_line,
                "end_line": symbol.end_line,
                "parent_qname": symbol.parent_qname,
                "signature": symbol.signature,
                "docstring": symbol.docstring,
                "repo": repo,
                "sha": sha,
                "language": language,
            },
        },
    )


def _symbol_node(repo: str, symbol: Symbol) -> GraphNodeSpec:
    if symbol.kind == NodeLabel.MODULE:
        canonical_id = f"{repo}:{symbol.file_path}"
    else:
        canonical_id = f"{repo}:{symbol.qualified_name}"
    return GraphNodeSpec(
        label=symbol.kind,
        canonical_id=canonical_id,
        properties={
            "name": symbol.qualified_name.rsplit(".", 1)[-1],
            "qualified_name": symbol.qualified_name,
            "file_path": symbol.file_path,
            "def_line": symbol.def_line,
        },
    )


def _map_edge(
    repo: str,
    edge,
    qname_to_kind: dict[str, NodeLabel],
) -> GraphEdgeSpec | None:
    """Map a CodeEdge (qualified-name endpoints) to a GraphEdgeSpec.

    Endpoint labels are looked up in qname_to_kind first (built from the
    actual symbols this batch wrote), falling back to per-edge-type
    heuristics for cross-repo / external targets the lookup doesn't know.
    Right label matters: graph_writer.upsert_edges keys node lookup on
    (label, canonical_id), so a label miss silently drops the edge.
    """
    from_canonical = f"{repo}:{edge.from_qname}"
    to_canonical = f"{repo}:{edge.to_qname}"

    confidence = "AMBIGUOUS" if edge.ambiguous else "EXTRACTED"

    properties = dict(edge.properties)
    if edge.ambiguous and edge.target_candidates:
        properties["candidates"] = edge.target_candidates

    # Per-edge-type fallback labels for endpoints not in our extracted set
    # (e.g. external imports, calls into stdlib).
    fallback_from = NodeLabel.FUNCTION
    fallback_to = NodeLabel.FUNCTION
    if edge.edge_type == EdgeType.IMPORTS:
        fallback_from = NodeLabel.MODULE
        fallback_to = NodeLabel.MODULE
    elif edge.edge_type == EdgeType.DEFINED_IN:
        # Module-in-Repo case: to_qname matches the repo.
        if edge.to_qname == repo or edge.to_qname.endswith(repo.rsplit("/", 1)[-1]):
            fallback_from = NodeLabel.MODULE
            fallback_to = NodeLabel.REPO
            to_canonical = repo
        else:
            fallback_from = NodeLabel.FUNCTION
            fallback_to = NodeLabel.MODULE
    elif edge.edge_type in (EdgeType.INHERITS, EdgeType.IMPLEMENTS):
        fallback_from = NodeLabel.CLASS
        fallback_to = NodeLabel.CLASS

    from_label = qname_to_kind.get(edge.from_qname, fallback_from)
    to_label = qname_to_kind.get(edge.to_qname, fallback_to)

    return GraphEdgeSpec(
        edge_type=edge.edge_type,
        from_label=from_label,
        from_canonical_id=from_canonical,
        to_label=to_label,
        to_canonical_id=to_canonical,
        properties=properties,
        confidence=confidence,
    )


__all__ = ["extract_files_to_result"]
