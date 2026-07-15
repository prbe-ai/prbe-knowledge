"""Path 2 file-as-Document invariants.

The file Document ditches per-symbol Documents in favor of one Document
per file with N pre-chunked symbol bodies + 1 metadata chunk. These
tests pin down the structural contract — what the connector emits and
what the normalizer is supposed to consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from engine.shared.constants import (
    CodeSymbolKind,
    DocClass,
    DocType,
    EdgeType,
    NodeLabel,
    SourceSystem,
)
from kb.code_graph.pipeline import (
    _build_file_document_with_symbol_chunks,
    extract_files_to_result,
)
from kb.code_graph.types import Symbol


@dataclass
class _FE:
    rel_path: str
    content: bytes


_TWO_SYMBOL_SAMPLE = b'''\
class Greeter:
    """A greeter class."""

    def hello(self, name: str) -> str:
        return self._format(name)

    def _format(self, name: str) -> str:
        return f"hi, {name}"
'''


def _make_symbol(
    *,
    qualified_name: str,
    kind: NodeLabel,
    file_path: str = "src/greeter.py",
    def_line: int = 10,
    end_line: int = 15,
    source_snippet: str = "def foo(): pass",
) -> Symbol:
    return Symbol(
        kind=kind,
        qualified_name=qualified_name,
        file_path=file_path,
        def_line=def_line,
        end_line=end_line,
        signature=None,
        docstring=None,
        source_snippet=source_snippet,
        parent_qname=None,
    )


# ---- _build_file_document_with_symbol_chunks unit tests ------------------


def test_doc_id_format_is_repo_file_only() -> None:
    """Path 2 doc_id: code_graph:<repo>:<file_path>. No symbol or line.
    The Document IS the file; symbols live as chunks.
    """
    from datetime import UTC, datetime

    syms = [_make_symbol(qualified_name="foo", kind=CodeSymbolKind.FUNCTION)]
    doc, _chunks, _metadata = _build_file_document_with_symbol_chunks(
        customer_id="c1",
        repo="acme/api",
        sha="deadbeef",
        owner_login="acme",
        language="python",
        file_path="src/x.py",
        symbols=syms,
        now=datetime.now(UTC),
    )
    assert doc.doc_id == "code_graph:acme/api:src/x.py"
    assert doc.doc_type == DocType.CODE_FILE
    assert doc.source_system == SourceSystem.CODE_GRAPH
    assert doc.doc_class == DocClass.RAW_SOURCE


def test_chunk_count_matches_symbol_count() -> None:
    """N symbols → N content chunks + 1 metadata chunk."""
    from datetime import UTC, datetime

    syms = [
        _make_symbol(qualified_name="Greeter", kind=CodeSymbolKind.CLASS),
        _make_symbol(qualified_name="Greeter.hello", kind=CodeSymbolKind.METHOD),
        _make_symbol(qualified_name="Greeter._format", kind=CodeSymbolKind.METHOD),
    ]
    _doc, chunks, metadata = _build_file_document_with_symbol_chunks(
        customer_id="c1",
        repo="acme/api",
        sha="deadbeef",
        owner_login="acme",
        language="python",
        file_path="src/greeter.py",
        symbols=syms,
        now=datetime.now(UTC),
    )
    assert len(chunks) == 3
    assert metadata is not None
    assert metadata.chunk_index < 0, "metadata chunk uses negative sentinel index"


def test_content_chunks_carry_synthetic_header_with_repo_and_qname() -> None:
    """The repo name appearing in chunk content is the whole Path 2 fix
    for repo-qualified search ranking. Verify it's there.
    """
    from datetime import UTC, datetime

    sym = _make_symbol(
        qualified_name="Greeter.hello",
        kind=CodeSymbolKind.METHOD,
        source_snippet="def hello(self, name): return name",
        def_line=42,
        end_line=43,
    )
    _doc, chunks, _meta = _build_file_document_with_symbol_chunks(
        customer_id="c1",
        repo="prbe-ai/prbe-backend",
        sha="abc",
        owner_login="prbe-ai",
        language="python",
        file_path="app/api_key.py",
        symbols=[sym],
        now=datetime.now(UTC),
    )
    chunk = chunks[0]
    assert "prbe-ai/prbe-backend" in chunk.content
    assert "Greeter.hello" in chunk.content
    assert "Method" in chunk.content
    # The actual source body still appears below the header.
    assert "def hello(self, name):" in chunk.content
    # Line range present so future per-chunk source_url work has data.
    assert "L42-43" in chunk.content


def test_metadata_chunk_lists_every_symbol_with_kind() -> None:
    from datetime import UTC, datetime

    syms = [
        _make_symbol(qualified_name="Greeter", kind=CodeSymbolKind.CLASS),
        _make_symbol(qualified_name="Greeter.hello", kind=CodeSymbolKind.METHOD),
        _make_symbol(qualified_name="format_name", kind=CodeSymbolKind.FUNCTION),
    ]
    _doc, _chunks, metadata = _build_file_document_with_symbol_chunks(
        customer_id="c1",
        repo="acme/api",
        sha="abc",
        owner_login="acme",
        language="python",
        file_path="src/x.py",
        symbols=syms,
        now=datetime.now(UTC),
    )
    assert metadata is not None
    text = metadata.content
    assert "Repo: acme/api" in text
    assert "File: src/x.py" in text
    assert "Language: python" in text
    assert "Greeter (Class)" in text
    assert "Greeter.hello (Method)" in text
    assert "format_name (Function)" in text


def test_empty_symbols_raises() -> None:
    """Caller MUST filter empty symbol lists upstream — emitting a file
    Document with zero chunks would produce a Document the search layer
    can't anchor on for anything except the metadata chunk, which lists
    no symbols. Better to skip the file outright.
    """
    from datetime import UTC, datetime

    with pytest.raises(ValueError, match="at least one symbol"):
        _build_file_document_with_symbol_chunks(
            customer_id="c1",
            repo="acme/api",
            sha="abc",
            owner_login="acme",
            language="python",
            file_path="src/x.py",
            symbols=[],
            now=datetime.now(UTC),
        )


def test_oversized_symbol_splits_into_multiple_chunks() -> None:
    """Symbols whose body exceeds MAX_SYMBOL_CHUNK_TOKENS must split.
    Pre-fix, a 30KB module-scope class landed as one giant chunk and
    blew Probe MCP 25KB tool-result caps. Post-fix, every emitted
    chunk fits the budget, window 0 keeps the full identifying header,
    and the chunk_index is globally contiguous (not per-symbol).
    """
    from datetime import UTC, datetime

    from engine.ingest.chunker import count_tokens
    from engine.shared.constants import MAX_SYMBOL_CHUNK_TOKENS

    method = (
        "    def {name}(self, x):\n"
        "        result = self.process(x)\n"
        "        if result is None:\n"
        "            raise ValueError('no result')\n"
        "        return result\n"
    )
    big_body = "class GiantClass:\n" + "\n".join(
        method.format(name=f"method_{i}") for i in range(80)
    )
    assert count_tokens(big_body) > MAX_SYMBOL_CHUNK_TOKENS

    syms = [
        _make_symbol(
            qualified_name="GiantClass",
            kind=CodeSymbolKind.CLASS,
            file_path="src/giant.py",
            source_snippet=big_body,
            def_line=1,
            end_line=400,
        ),
        _make_symbol(
            qualified_name="small_helper",
            kind=CodeSymbolKind.FUNCTION,
            file_path="src/giant.py",
            source_snippet="def small_helper(): return 1",
            def_line=410,
            end_line=411,
        ),
    ]
    _doc, chunks, _meta = _build_file_document_with_symbol_chunks(
        customer_id="c1",
        repo="acme/api",
        sha="abc",
        owner_login="acme",
        language="python",
        file_path="src/giant.py",
        symbols=syms,
        now=datetime.now(UTC),
    )

    # GiantClass alone produces multiple windows; small_helper is one chunk.
    assert len(chunks) > 2

    # Every chunk fits the budget (small slack for header rounding).
    for c in chunks:
        assert c.token_count <= MAX_SYMBOL_CHUNK_TOKENS + 16

    # chunk_index is globally contiguous starting at 0 — chunks plumbing
    # downstream (`_apply_chunk_plan`) relies on stable per-doc ordering.
    for i, c in enumerate(chunks):
        assert c.chunk_index == i

    # Window 0 of GiantClass keeps the full repo/file/qname header so
    # repo-qualified queries still rank correctly.
    giantclass_chunks = [c for c in chunks if "GiantClass" in c.content]
    assert giantclass_chunks
    assert "acme/api" in giantclass_chunks[0].content
    assert "src/giant.py" in giantclass_chunks[0].content

    # Continuation windows must NOT carry repo/file path tokens — that's
    # the BM25 over-fire mitigation. Find the continuation chunks (the
    # ones starting with the lighter "(cont.)" header).
    cont_chunks = [c for c in chunks if "(cont.)" in c.content[:80]]
    assert cont_chunks, "expected at least one continuation window"
    for c in cont_chunks:
        assert "acme/api" not in c.content, (
            "continuation header must not include repo (BM25 over-fire risk)"
        )
        assert "src/giant.py" not in c.content


def test_metadata_jsonb_carries_per_symbol_lookup() -> None:
    """Until per-chunk source_url lands in a follow-up, the dashboard
    renderer composes per-symbol GitHub permalinks from the file
    Document's source_url + the per-symbol line range stored on
    Document.metadata.file.symbols.
    """
    from datetime import UTC, datetime

    sym = _make_symbol(
        qualified_name="hello",
        kind=CodeSymbolKind.FUNCTION,
        def_line=10,
        end_line=15,
    )
    doc, _chunks, _meta = _build_file_document_with_symbol_chunks(
        customer_id="c1",
        repo="acme/api",
        sha="abc",
        owner_login="acme",
        language="python",
        file_path="src/x.py",
        symbols=[sym],
        now=datetime.now(UTC),
    )
    file_meta = doc.metadata.get("file", {})
    assert file_meta.get("repo") == "acme/api"
    assert file_meta.get("sha") == "abc"
    assert file_meta.get("path") == "src/x.py"
    assert file_meta.get("symbol_count") == 1
    sym_lookup = file_meta.get("symbols", {})
    assert "hello" in sym_lookup
    assert sym_lookup["hello"]["def_line"] == 10
    assert sym_lookup["hello"]["end_line"] == 15
    assert sym_lookup["hello"]["kind"] == "Function"


# ---- end-to-end pipeline tests -------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_emits_one_file_document_with_n_symbols() -> None:
    """For a Python file with one Greeter class containing two methods,
    the pipeline emits exactly one file Document carrying three content
    chunks (Greeter, Greeter.hello, Greeter._format) and one metadata
    chunk. Backstops the orchestrator's switch to documents_with_chunks.
    """
    files = [_FE(rel_path="src/greeter.py", content=_TWO_SYMBOL_SAMPLE)]
    result = await extract_files_to_result(
        customer_id="c1",
        repo="acme/api",
        sha="deadbeef",
        files=files,
        cached_state={},
    )
    assert not result.documents, "Path 2 emits no raw Documents"
    assert len(result.documents_with_chunks) == 1
    pre = result.documents_with_chunks[0]
    assert pre.document.doc_type == DocType.CODE_FILE
    assert len(pre.chunks) >= 2  # at least Greeter.hello + Greeter._format
    assert pre.metadata_chunk is not None


@pytest.mark.asyncio
async def test_compiled_from_edges_are_one_per_symbol_to_file_doc() -> None:
    """COMPILED_FROM is now 1:N — one file Document → many Symbol nodes."""
    files = [_FE(rel_path="src/greeter.py", content=_TWO_SYMBOL_SAMPLE)]
    result = await extract_files_to_result(
        customer_id="c1",
        repo="acme/api",
        sha="deadbeef",
        files=files,
        cached_state={},
    )
    compiled = [e for e in result.graph_edges if e.edge_type == EdgeType.COMPILED_FROM]
    assert compiled, "expected COMPILED_FROM edges"
    # All COMPILED_FROM edges share the same from_canonical_id (the file Doc).
    from_ids = {e.from_canonical_id for e in compiled}
    assert len(from_ids) == 1, (
        f"expected all COMPILED_FROM edges to share the same file Document "
        f"as origin, got {from_ids}"
    )
    assert from_ids == {"code_graph:acme/api:src/greeter.py"}
