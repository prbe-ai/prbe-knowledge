"""The remediation sweep: what it covers, what it costs, and what it must not
report as clean.

The sweep exists because redaction only stops the NEXT leak. On 2026-09-12 a
live AWS access key and its secret sat in five live chunks of one customer's
session, readable by that customer's whole team, ten days after capture.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from engine.ingest import credential_sweep

_ROOT = Path(__file__).resolve().parent.parent
_SRC = (_ROOT / "engine" / "ingest" / "credential_sweep.py").read_text()


def test_one_detection_scan_per_page_not_per_document() -> None:
    """A gitleaks spawn costs ~1 CPU-second of pure startup regardless of
    input size. Per document plus a verify each, a 50-document page was 50-100
    spawns; for the `probe` corpus that is hundreds of CPU-hours, which is a
    sweep nobody runs."""
    page = _SRC[_SRC.index("async def _sweep_page") : _SRC.index("async def _rewrite_chunks")]
    detection = page[: page.index("for doc_id in doc_ids:\n        c_start")]
    assert detection.count("redact_documents_async") == 1, (
        "the detection scan must be one call for the whole page"
    )


def test_the_verify_scan_stays_per_document() -> None:
    """`unresolved` has to name the document it belongs to, not the page it
    was in. Verify runs only for documents that actually changed -- 426 of
    6,124 runs on 2026-09-16 -- so the batching win is kept."""
    page = _SRC[_SRC.index("async def _sweep_page") : _SRC.index("async def _rewrite_chunks")]
    per_doc = page[page.index("for doc_id in doc_ids:\n        c_start") :]
    assert "redact_documents_async" in per_doc
    assert "result.unresolved.append(doc_id)" in per_doc


def test_document_title_and_preview_are_swept() -> None:
    """Grounding runs FTS over `title || ' ' || body_preview`. Until
    2026-09-17 the sweep touched only `chunks`, so a credential in a session's
    opening line stayed lexically searchable while every chunk was clean and
    the document was reported remediated."""
    assert "body_preview" in _SRC
    assert "_rewrite_document_fields" in _SRC
    rewrite = _SRC[_SRC.index("async def _rewrite_document_fields") :]
    assert "UPDATE documents" in rewrite
    assert "SET title = $3, body_preview = $4" in rewrite


def test_the_document_rewrite_leaves_content_hash_alone() -> None:
    """`documents.content_hash` describes the BODY, which is not in this
    table. Moving it would make every consumer comparing it think the document
    changed."""
    rewrite = _SRC[_SRC.index("async def _rewrite_document_fields") :]
    assert "content_hash" not in rewrite.split('"""')[2]


def test_a_redaction_collision_drops_the_duplicate_instead_of_aborting() -> None:
    """`chunks` has UNIQUE (customer_id, doc_id, content_hash). Two versions
    differing ONLY in the credential become identical once it is replaced, so
    the second UPDATE violated the constraint, aborted the transaction, and
    left the rest of the page unswept while the caller was told how many
    chunks had been rewritten."""
    rewrite = _SRC[_SRC.index("async def _rewrite_chunks") : _SRC.index("async def _rewrite_document_fields")]
    assert "NOT EXISTS" in rewrite
    assert "DELETE FROM chunks" in rewrite
    assert "chunks_deduped" in rewrite


def test_a_deduped_chunk_is_not_also_counted_as_rewritten() -> None:
    rewrite = _SRC[_SRC.index("async def _rewrite_chunks") : _SRC.index("async def _rewrite_document_fields")]
    assert "result.chunks_rewritten -= 1" in rewrite


def test_the_sweep_is_resumable_and_bounded() -> None:
    """A full-corpus run is hours of work. A process that cannot be resumed is
    one that gets abandoned half-done and reported as a sweep."""
    params = inspect.signature(credential_sweep.sweep_credentials).parameters
    assert {"since", "after_doc_id", "max_documents"} <= set(params)
    assert "result.last_doc_id = page[-1]" in _SRC


def test_every_document_selection_path_is_ordered_by_doc_id() -> None:
    """The cursor is `doc_id > after_doc_id`, which only resumes correctly if
    every path walks in that order."""
    selection = _SRC[_SRC.index("    if doc_ids:") : _SRC.index("    by_doc:")]
    assert selection.count("ORDER BY") == 2, selection
    assert "ORDER BY d.doc_id" in selection
    assert "ORDER BY doc_id" in selection
    assert "pending.sort()" in selection, "the explicit doc_ids path needs it too"


def test_a_scan_failure_marks_every_document_in_the_page() -> None:
    """Not "clean". The whole point of ScanUnavailable is that a scanner which
    could not look is not a scanner that found nothing."""
    page = _SRC[_SRC.index("async def _sweep_page") : _SRC.index("async def _rewrite_chunks")]
    assert "result.scan_failed.extend(doc_ids)" in page


def test_the_route_refuses_to_answer_cleanly_on_a_failed_scan() -> None:
    route = (_ROOT / "kb" / "purge_routes.py").read_text()
    assert "if result.scan_failed:" in route
    body = route[route.index("if result.scan_failed:") :]
    assert "503" in body[:900]


def test_the_route_bounds_one_call() -> None:
    """It is synchronous by design; a full-corpus sweep must be walked in
    bounded calls rather than held open until something times out."""
    route = (_ROOT / "kb" / "purge_routes.py").read_text()
    m = re.search(r"max_documents: int = Field\(default=(\d+), ge=\d+, le=(\d+)\)", route)
    assert m, "max_documents must be bounded"
    assert int(m.group(1)) <= int(m.group(2))
