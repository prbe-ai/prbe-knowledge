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

from engine.ingest.chunker import count_tokens
from engine.shared.constants import (
    MAX_SYMBOL_CHUNK_TOKENS,
    CodeSymbolKind,
    DocClass,
    DocType,
    DocumentKind,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from engine.shared.db import with_tenant
from engine.shared.logging import get_logger
from engine.shared.models import (
    METADATA_CHUNK_INDEX,
    ACLPrincipal,
    ACLSnapshot,
    ChunkPiece,
    CodeRepoStateUpdate,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    NormalizationResult,
    PreChunkedDocument,
    make_code_symbol,
    make_document,
)
from kb.code_graph.chunking import split_symbol_body
from kb.code_graph.extractors import get_extractor_for_file
from kb.code_graph.qualifier import promote_single_match
from kb.code_graph.secrets import (
    SKIPPED_LANGUAGE_SENTINEL,
    looks_like_secret_dump,
)
from kb.code_graph.types import ExtractResult, Symbol

if TYPE_CHECKING:
    from kb.code_graph.clone import FileEntry  # noqa: F401

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

    # Set of qualified_names this batch extracted. Used by _map_edge to
    # distinguish in-repo endpoints (write as CODE_SYMBOL) from external
    # endpoints (per-edge-type fallback label). Pre-0091 this dict stored
    # the fine-grained NodeLabel per qname; post-collapse every extracted
    # symbol writes as CODE_SYMBOL so we only need to know whether the qname
    # is in this batch or not.
    extracted_qnames: set[str] = set()
    for _, result, _ in extractions:
        for symbol in result.symbols:
            key = symbol.file_path if symbol.kind == CodeSymbolKind.MODULE else symbol.qualified_name
            extracted_qnames.add(key)

    nodes: list[GraphNodeSpec] = []
    edges: list[GraphEdgeSpec] = []
    state_updates: list[CodeRepoStateUpdate] = []
    pre_chunked_docs: list = []  # PreChunkedDocument list — typed below

    repo_node = make_document(
        canonical_id=repo,
        kind=DocumentKind.REPO,
        properties={"name": repo.rsplit("/", 1)[-1]},
    )
    nodes.append(repo_node)
    owner_login = _owner_for_repo(repo)

    now = datetime.now(UTC)
    total_symbols = 0

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
        # Skip files with no extracted symbols — nothing to surface.
        # (Empty .py files, parse errors that returned an empty result.)
        if not result.symbols:
            for edge in result.edges:
                mapped = _map_edge(repo, edge, extracted_qnames)
                if mapped is not None:
                    edges.append(mapped)
            continue

        # Build the file Document + its pre-chunked pieces. One Document
        # per file means search can return the whole file's symbols when
        # an identity query (repo + file + symbol qname) hits the metadata
        # chunk, even if the user wasn't searching for the symbol's body
        # directly.
        file_doc, file_chunks, file_metadata_chunk = _build_file_document_with_symbol_chunks(
            customer_id=customer_id,
            repo=repo,
            sha=sha,
            owner_login=owner_login,
            language=language,
            file_path=rel_path,
            symbols=result.symbols,
            now=now,
        )
        pre_chunked_docs.append(
            PreChunkedDocument(
                document=file_doc,
                chunks=file_chunks,
                metadata_chunk=file_metadata_chunk,
            )
        )
        total_symbols += len(result.symbols)

        # Per-symbol graph nodes — every code symbol writes as a CODE_SYMBOL
        # node with the fine-grained kind stamped into properties (see
        # _symbol_node). COMPILED_FROM edges go file Document → N CodeSymbol
        # nodes (1:N).
        file_doc_id = file_doc.doc_id
        for symbol in result.symbols:
            nodes.append(_symbol_node(repo, symbol))
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.COMPILED_FROM,
                    from_label=NodeLabel.DOCUMENT,
                    from_canonical_id=file_doc_id,
                    to_label=NodeLabel.CODE_SYMBOL,
                    to_canonical_id=f"{repo}:{symbol.qualified_name}"
                    if symbol.kind != CodeSymbolKind.MODULE
                    else f"{repo}:{symbol.file_path}",
                )
            )
        for edge in result.edges:
            mapped = _map_edge(repo, edge, extracted_qnames)
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
        file_documents=len(pre_chunked_docs),
        symbols=total_symbols,
        nodes=len(nodes),
        edges=len(edges),
    )

    return NormalizationResult(
        documents_with_chunks=pre_chunked_docs,
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


def _doc_id_for_file(repo: str, file_path: str) -> str:
    """code.file Document id. Stable across re-extracts of the same file."""
    return f"code_graph:{repo}:{file_path}"


def _owner_for_repo(repo: str) -> str:
    """Pull the GitHub owner login out of an `<owner>/<repo>` slug.

    Falls back to a sentinel for malformed slugs so ACL construction
    never explodes on bad input — the workspace ACL still applies, just
    to the placeholder principal that no real user holds.
    """
    return repo.split("/", 1)[0] if "/" in repo else _OWNER_FALLBACK


def _file_permalink(repo: str, sha: str, file_path: str) -> str:
    """GitHub blob URL for the whole file (no line range). The dashboard
    uses this as the Document-level "View on GitHub" link. Per-symbol
    deep links (#L42-L58) live on chunks via a future schema addition;
    until that lands, the file-level URL is the best the renderer has.
    """
    return f"https://github.com/{repo}/blob/{sha}/{file_path}"


def _symbol_permalink(repo: str, sha: str, symbol: Symbol) -> str:
    """GitHub blob URL with line range for a single symbol."""
    return (
        f"https://github.com/{repo}/blob/{sha}/{symbol.file_path}"
        f"#L{symbol.def_line}-L{symbol.end_line}"
    )


def _codegraph_acl(owner_login: str, captured_at: datetime) -> ACLSnapshot:
    """Workspace-level READ to the repo owner. Mirrors the GitHub
    connector's repo ACL convention so existing ACL filters apply
    transparently to code-graph Documents.
    """
    return ACLSnapshot(
        principals=[
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=owner_login,
                permission=Permission.READ,
            )
        ],
        captured_at=captured_at,
    )


def _symbol_chunk_id_suffix(symbol: Symbol) -> str:
    """Stable, collision-resistant chunk identity within a file Document.

    Two stub functions can share an identical body (so `content_hash`
    collides), but they always have distinct (qualified_name, def_line)
    pairs. Sanitize qname for chunk_id safety: dots, slashes, and colons
    get squashed so the chunk_id stays parseable as
    `<doc_id>:c_<this_suffix>`.
    """
    safe_qname = symbol.qualified_name.replace(":", "_").replace("/", "_")
    return f"{safe_qname}@{symbol.def_line}"


def _build_symbol_document(
    customer_id: str,
    repo: str,
    sha: str,
    owner_login: str,
    language: str,
    symbol: Symbol,
    now: datetime,
) -> Document:
    """LEGACY per-symbol Document builder (`doc_type='code.symbol'`).

    Retained for the transitional phase only — the file-as-Document
    rewrite (Path 2) emits one `code.file` Document per file with
    chunks per symbol. Migration 0050 hard-deletes data written by
    this function. Slated for removal once the new path is verified
    in prod.
    """
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

    return Document(
        doc_id=doc_id,
        customer_id=customer_id,
        source_system=SourceSystem.CODE_GRAPH,
        source_id=f"symbol:{repo}:{symbol.file_path}:{symbol.qualified_name}:{symbol.def_line}",
        source_url=_symbol_permalink(repo, sha, symbol),
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
        acl=_codegraph_acl(owner_login, now),
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


def _build_file_document_with_symbol_chunks(
    *,
    customer_id: str,
    repo: str,
    sha: str,
    owner_login: str,
    language: str,
    file_path: str,
    symbols: list[Symbol],
    now: datetime,
):
    """Path 2: file becomes the Document, symbols become chunks.

    Returns (Document, content_chunks, metadata_chunk):

      - Document has body=None and doc_type='code.file'. Source URL points
        to the file's GitHub blob (no line range; per-symbol deep-links
        come in a follow-up that adds chunks.source_url).
      - One ChunkPiece per symbol: chunk content is a synthetic header
        (`# repo · file · qname (kind) · L<def>-<end>`) followed by the
        source snippet. The header puts identifying text into the embedded
        body so semantic search ranks repo-qualified queries correctly,
        which is the whole point of this rewrite.
      - One metadata ChunkPiece carrying repo + file path + language +
        a list of every symbol qname in the file. Search hits on this
        chunk surface the file Document for identity queries
        ("IngestionToken in prbe-backend") even when no symbol body
        embedding ranks high.

    chunk identity:
      - content chunks: chunk_id = `<doc_id>:c_<content_hash[:16]>` (built
        downstream in `normalizer.py:_persist_chunks`). Each window has
        distinct content (header differs, body slice differs) so
        content_hash is unique per window — no collision risk.
      - metadata chunk: chunk_index = METADATA_CHUNK_INDEX sentinel
        (set by ChunkPiece consumer in normalizer)

    Per-symbol size cap:
      Each emitted chunk is bounded at MAX_SYMBOL_CHUNK_TOKENS. Symbols
      that fit emit one ChunkPiece (current behavior). Symbols that
      exceed the cap are split into windows by `split_symbol_body`:
      window 0 carries the full header; windows 1+ carry a lighter
      `# {qname} (cont.)` continuation marker (avoids inflating BM25
      firing on identifier tokens — see services/ingestion/code_graph/
      chunking.py docstring for rationale).

    Caller MUST NOT pass an empty symbols list — the orchestrator skips
    files with zero extractions before calling.
    """
    if not symbols:
        raise ValueError("file Document requires at least one symbol")

    doc_id = _doc_id_for_file(repo, file_path)
    title = file_path.rsplit("/", 1)[-1]  # filename, no path
    permalink = _file_permalink(repo, sha, file_path)

    # Per-symbol content chunks. Synthetic header (`# repo · file · qname …`)
    # gets embedded with the body so a search for "IngestionToken in
    # prbe-backend" naturally ranks this chunk above prbe-dashboard's
    # same-named symbol — the header carries the repo name as searchable
    # text that the symbol body itself never would.
    content_chunks: list[ChunkPiece] = []
    for symbol in symbols:
        body = symbol.source_snippet or symbol.signature or symbol.qualified_name
        primary_header = (
            f"# {repo} · {symbol.file_path} · "
            f"{symbol.qualified_name} ({symbol.kind.value}) · "
            f"L{symbol.def_line}-{symbol.end_line}\n"
        )
        # Lighter header for split-symbol continuation windows. Drops
        # repo/file/path tokens to avoid amplifying BM25 firing across
        # N copies of the same identifying text. Window 0 always uses
        # primary_header; windows 1+ use this one.
        continuation_header = f"# {symbol.qualified_name} (cont.)\n"
        for window_content in split_symbol_body(
            body,
            primary_header=primary_header,
            continuation_header=continuation_header,
            max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
        ):
            content_chunks.append(
                ChunkPiece(
                    chunk_index=len(content_chunks),
                    content=window_content,
                    token_count=count_tokens(window_content),
                )
            )

    # Metadata chunk: synthetic key:value text used for identity queries.
    # The search layer hits this chunk for queries like "IngestionToken in
    # prbe-backend" because all the identifying tokens (repo, file, every
    # symbol's qname) live here as plain searchable text.
    metadata_lines = [
        f"Repo: {repo}",
        f"File: {file_path}",
        f"Language: {language}",
        f"Symbols ({len(symbols)}):",
    ]
    for symbol in symbols:
        metadata_lines.append(
            f"  - {symbol.qualified_name} ({symbol.kind.value})"
        )
    metadata_content = "\n".join(metadata_lines)
    metadata_chunk = ChunkPiece(
        chunk_index=METADATA_CHUNK_INDEX,
        content=metadata_content,
        token_count=count_tokens(metadata_content),
    )

    # Document content_hash: sha256 over every symbol's identity tuple.
    # Stable across re-extracts of the same file at the same commit; flips
    # the moment any symbol changes (which is what we want for SCD2).
    hash_input = "\x1f".join(
        f"{s.qualified_name}|{s.def_line}|{s.end_line}|{s.source_snippet}"
        for s in symbols
    )
    content_hash = hashlib.sha256(
        hash_input.encode("utf-8", errors="replace")
    ).hexdigest()

    body_preview = f"{len(symbols)} symbols: " + ", ".join(
        s.qualified_name for s in symbols[:5]
    )
    if len(symbols) > 5:
        body_preview += f", … (+{len(symbols) - 5} more)"
    if len(body_preview) > 200:
        body_preview = body_preview[:200]

    file_doc = Document(
        doc_id=doc_id,
        customer_id=customer_id,
        source_system=SourceSystem.CODE_GRAPH,
        source_id=f"file:{repo}:{file_path}",
        source_url=permalink,
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.CODE_FILE,
        content_type="text/plain",
        language=language,
        content_hash=content_hash,
        title=title,
        body_preview=body_preview,
        # body_size_bytes / body_token_count reflect the metadata chunk —
        # the connector owns chunking, the Document carries no body itself.
        body_size_bytes=len(metadata_content.encode("utf-8")),
        body_token_count=metadata_chunk.token_count,
        author_id=None,
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=_codegraph_acl(owner_login, now),
        # CRITICAL: body=None for pre-chunked Documents. The normalizer's
        # body-guard catches the `body is not None AND pre_chunked is not
        # None` case and raises — keeping that contract clean here.
        body=None,
        metadata={
            "file": {
                "path": file_path,
                "language": language,
                "repo": repo,
                "sha": sha,
                "symbol_count": len(symbols),
                # Per-symbol metadata for the dashboard renderer until the
                # follow-up adds chunks.metadata. Each entry maps qname →
                # (kind, def_line, end_line) so the renderer can compose
                # per-symbol GitHub permalinks on the fly.
                "symbols": {
                    s.qualified_name: {
                        "kind": s.kind.value,
                        "def_line": s.def_line,
                        "end_line": s.end_line,
                    }
                    for s in symbols
                },
            },
        },
    )

    return file_doc, content_chunks, metadata_chunk


def _symbol_node(repo: str, symbol: Symbol) -> GraphNodeSpec:
    if symbol.kind == CodeSymbolKind.MODULE:
        canonical_id = f"{repo}:{symbol.file_path}"
    else:
        canonical_id = f"{repo}:{symbol.qualified_name}"
    return make_code_symbol(
        canonical_id=canonical_id,
        kind=symbol.kind,
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
    extracted_qnames: set[str],
) -> GraphEdgeSpec | None:
    """Map a CodeEdge (qualified-name endpoints) to a GraphEdgeSpec.

    Endpoint labels are CODE_SYMBOL when the qname was extracted in this
    batch (we just wrote it as such), otherwise a per-edge-type fallback
    for cross-repo / external targets. Right label matters: graph_writer.
    upsert_edges keys node lookup on (label, canonical_id), so a label
    miss silently drops the edge.

    Post-0091 the only non-CODE_SYMBOL fallback is DOCUMENT (for the repo
    node, which is now a Document with DocumentKind.REPO).
    """
    from_canonical = f"{repo}:{edge.from_qname}"
    to_canonical = f"{repo}:{edge.to_qname}"

    confidence = "AMBIGUOUS" if edge.ambiguous else "EXTRACTED"

    properties = dict(edge.properties)
    if edge.ambiguous and edge.target_candidates:
        properties["candidates"] = edge.target_candidates

    # Per-edge-type fallback labels for endpoints not in our extracted set
    # (external imports, calls into stdlib, cross-repo references).
    fallback_from = NodeLabel.CODE_SYMBOL
    fallback_to = NodeLabel.CODE_SYMBOL
    # Module-in-Repo case: a DEFINED_IN edge whose to_qname matches the repo
    # points at the Repo node (now a Document post-0091).
    if edge.edge_type == EdgeType.DEFINED_IN and (
        edge.to_qname == repo or edge.to_qname.endswith(repo.rsplit("/", 1)[-1])
    ):
        fallback_to = NodeLabel.DOCUMENT
        to_canonical = repo

    from_label = (
        NodeLabel.CODE_SYMBOL if edge.from_qname in extracted_qnames else fallback_from
    )
    to_label = (
        NodeLabel.CODE_SYMBOL if edge.to_qname in extracted_qnames else fallback_to
    )

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
