"""Remediate credentials already stored in the knowledge base.

Redaction stops the NEXT leak. It does nothing about what is already indexed:
on 2026-09-12 a live AWS access key id and its secret access key sat in five
live chunks of one customer's captured session, readable by that customer's
whole team, ten days after capture. The same pattern matched documents across
7 of 13 tenants.

WHY THIS REWRITES RATHER THAN DELETES
-------------------------------------
Deleting the chunk removes the credential AND the research record — the
transcript of how an experiment was actually run, which is the thing the
customer is paying us to keep. Rewriting removes only the credential. The
session stays searchable and complete apart from a `<redacted:...>` marker
where the value was.

It is also the only option that is safe to run broadly. A sweep that deletes is
a sweep nobody dares point at 13 tenants; a sweep that replaces a substring is
one you can.

WHY THE CONTENT HASH MOVES WITH IT
----------------------------------
`chunks.content_hash` is the reuse key: the normalizer skips re-embedding when
the hash is unchanged. Rewriting content without rewriting the hash would leave
a row whose hash describes text that no longer exists, and the next ingestion of
that document would "reuse" the unredacted chunk straight back in.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
The embedding is left alone. It was computed from the pre-redaction text, so it
still points at a 40-character opaque token that is no longer in the content —
a small semantic drift on a handful of chunks, against the cost and blast radius
of re-embedding. Retrieval reads `content`, so nothing serves the old value.
Re-embedding on the next natural reindex is sufficient and is the cheaper place
to pay for it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import structlog

from engine.ingest.secret_redaction import available, redact_documents_async
from engine.shared.db import with_tenant
from engine.shared.exceptions import ScanUnavailable

log = structlog.get_logger(__name__)

#: How many chunks to hold in memory (and rewrite) at once.

#: Documents per page. The sweep used to SELECT every chunk a tenant owns in
#: one query and hold the contents in memory — for a large tenant that is an
#: OOM and a request timeout, and two gitleaks subprocesses per document on top.
#: Paging by document keeps the working set flat regardless of tenant size.
_DOC_PAGE = 50


@dataclass
class SweepResult:
    """What the sweep touched. Counts and rule names only — never a value."""

    documents_scanned: int = 0
    documents_changed: int = 0
    chunks_scanned: int = 0
    chunks_rewritten: int = 0
    rules: dict[str, int] = field(default_factory=dict)
    #: Documents that still hold a finding after the rewrite. Should always be
    #: empty; a non-empty list means a rule matched text it could not replace,
    #: which is a bug worth failing loudly on rather than reporting "done".
    unresolved: list[str] = field(default_factory=list)
    #: Documents the scanner could not produce a verdict on (ScanUnavailable).
    #: A sweep with entries here has NOT cleared this tenant, and the route
    #: says so rather than reporting the findings it did manage to collect as
    #: if they were the whole answer.
    scan_failed: list[str] = field(default_factory=list)
    #: Chunks dropped because redaction made them byte-identical to another
    #: chunk of the same document. See `_rewrite_chunks`.
    chunks_deduped: int = 0
    #: `documents.title` / `body_preview` rewrites. Counted apart from chunks
    #: because they are a different table and were missed entirely until
    #: 2026-09-17: grounding runs FTS over `title || ' ' || body_preview`, so a
    #: credential there stayed lexically searchable while every chunk of the
    #: document was clean and the sweep reported it remediated.
    documents_rewritten: int = 0
    #: The last doc_id this run completed, for resuming a long sweep.
    last_doc_id: str | None = None
    skipped_no_binary: bool = False

    def as_dict(self) -> dict:
        return {
            "documents_scanned": self.documents_scanned,
            "documents_changed": self.documents_changed,
            "chunks_scanned": self.chunks_scanned,
            "chunks_rewritten": self.chunks_rewritten,
            "rules": self.rules,
            "unresolved": self.unresolved,
            "scan_failed": self.scan_failed,
            "chunks_deduped": self.chunks_deduped,
            "documents_rewritten": self.documents_rewritten,
            "last_doc_id": self.last_doc_id,
            "skipped_no_binary": self.skipped_no_binary,
        }


def _chunk_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def sweep_credentials(
    customer_id: str,
    *,
    doc_ids: list[str] | None = None,
    dry_run: bool = False,
    since: datetime | None = None,
    after_doc_id: str | None = None,
    max_documents: int | None = None,
) -> SweepResult:
    """Find and replace stored credentials for one tenant.

    `doc_ids=None` sweeps every document the tenant has. `dry_run=True` reports
    what would change and writes nothing — the intended first pass, because
    triaging "is this a live key or AWS's documentation example" is a judgement
    a human makes once per finding and a sweep should not make for them.

    `since` narrows to documents ingested at or after a moment — the shape you
    want after fixing a scanner regression, where the exposed window is known
    and the rest of the corpus was cleared by an earlier run.

    `after_doc_id` + `max_documents` make a long sweep resumable: documents are
    walked in `doc_id` order, `result.last_doc_id` is where this run stopped,
    and passing it back as `after_doc_id` continues from there. A full-corpus
    run is hours of work and a process that cannot be resumed is a process
    that gets abandoned half-done and reported as a sweep.
    """
    result = SweepResult()
    if not available():
        # Reporting "0 findings" when the scanner is absent is the worst
        # possible output: it reads as "clean".
        log.error("credential_sweep.no_scanner", customer=customer_id)
        result.skipped_no_binary = True
        return result

    # EVERY version, not just the live one. `chunks` is SCD2: a re-ingest does
    # not overwrite a row, it closes the old one out with `valid_to` and writes
    # a new one. So ordinary re-ingestion MOVES a leaked credential from the
    # live row into a historical row -- and retrieval serves those under
    # TemporalMode.ALL / AS_OF / CHANGED_BETWEEN. A sweep that filtered
    # `valid_to IS NULL` would report a document remediated while the value sat
    # one version back, still searchable.
    if doc_ids:
        pending = list(dict.fromkeys(doc_ids))
        if after_doc_id is not None:
            pending = [d for d in pending if d > after_doc_id]
        pending.sort()
    elif since is not None:
        # Narrowed by INGEST time, read from `documents`: `chunks` carries no
        # timestamp of its own that means "when this arrived".
        async with with_tenant(customer_id) as conn:
            id_rows = await conn.fetch(
                """
                SELECT DISTINCT d.doc_id
                  FROM documents d
                 WHERE d.customer_id = $1
                   AND d.ingested_at >= $2
                   AND ($3::text IS NULL OR d.doc_id > $3)
                 ORDER BY d.doc_id
                """,
                customer_id,
                since,
                after_doc_id,
            )
        pending = [r["doc_id"] for r in id_rows]
    else:
        async with with_tenant(customer_id) as conn:
            id_rows = await conn.fetch(
                """
                SELECT DISTINCT doc_id FROM chunks
                 WHERE customer_id = $1
                   AND ($2::text IS NULL OR doc_id > $2)
                 ORDER BY doc_id
                """,
                customer_id,
                after_doc_id,
            )
        pending = [r["doc_id"] for r in id_rows]

    if max_documents is not None:
        pending = pending[:max_documents]

    by_doc: dict[str, list[tuple[str, str]]] = {}
    for start in range(0, len(pending), _DOC_PAGE):
        page = pending[start:start + _DOC_PAGE]
        async with with_tenant(customer_id) as conn:
            rows = await conn.fetch(
                """
                SELECT chunk_id, doc_id, content
                  FROM chunks
                 WHERE customer_id = $1 AND doc_id = ANY($2::text[])
                 ORDER BY doc_id, chunk_index, valid_from
                """,
                customer_id,
                page,
            )
        for row in rows:
            by_doc.setdefault(row["doc_id"], []).append(
                (row["chunk_id"], row["content"] or "")
            )

        # `documents.title` and `body_preview` are the fields grounding runs
        # FTS over. Until 2026-09-17 the sweep never touched them, so a
        # credential in a session's opening line stayed lexically searchable
        # while every chunk was clean and the document was reported remediated.
        async with with_tenant(customer_id) as conn:
            doc_rows = await conn.fetch(
                """
                SELECT doc_id, version, title, body_preview
                  FROM documents
                 WHERE customer_id = $1 AND doc_id = ANY($2::text[])
                 ORDER BY doc_id, version
                """,
                customer_id,
                page,
            )
        docs: dict[str, list[tuple[int, str, str]]] = {}
        for row in doc_rows:
            docs.setdefault(row["doc_id"], []).append(
                (row["version"], row["title"] or "", row["body_preview"] or "")
            )

        await _sweep_page(customer_id, by_doc, docs, result, dry_run=dry_run)
        # The cursor: where a resumed run picks up. Set AFTER the page is
        # swept, so a crash mid-page resumes at the page rather than past it.
        result.last_doc_id = page[-1]
        by_doc = {}

    return result


async def _sweep_page(
    customer_id: str,
    by_doc: dict[str, list[tuple[str, str]]],
    docs: dict[str, list[tuple[int, str, str]]],
    result: SweepResult,
    *,
    dry_run: bool,
) -> None:
    """One page: ONE detection scan, then per-document verify and rewrite.

    The detection scan used to be per document, plus a verify scan per document
    with findings -- 50 to 100 gitleaks spawns for a page of 50, and a spawn
    costs ~1 CPU-second of pure startup regardless of input size. For the
    `probe` corpus that is hundreds of CPU-hours, which is a sweep nobody runs.

    One scan for the page, because the overwhelming majority of pages are
    clean and that is the case worth making cheap. Attribution stays exact:
    `redact_documents` replaces found values literally in every piece it was
    given, so diffing each document's own pieces says which documents changed.
    The verify scan stays PER DOCUMENT and runs only for documents that
    actually changed (426 of 6,124 runs on 2026-09-16), so `unresolved` still
    names the document it belongs to rather than the page.
    """
    doc_ids = list(by_doc)
    if not doc_ids:
        return

    # One flat list, and the slices to map it back. Document fields ride along
    # in the same scan: they are the same text from the customer's point of
    # view and splitting them would cost a second spawn to say so.
    pieces: list[str] = []
    spans: dict[str, tuple[int, int, int, int]] = {}
    for doc_id in doc_ids:
        chunk_start = len(pieces)
        pieces.extend(c for _, c in by_doc[doc_id])
        field_start = len(pieces)
        for _version, title, preview in docs.get(doc_id, []):
            pieces.extend((title, preview))
        spans[doc_id] = (chunk_start, field_start, field_start, len(pieces))

    result.documents_scanned += len(doc_ids)
    result.chunks_scanned += sum(len(by_doc[d]) for d in doc_ids)

    try:
        redacted, findings = await redact_documents_async(pieces)
    except ScanUnavailable as exc:
        # A page the scanner could not read must not be counted as clean.
        # Every document in it is inconclusive, and the caller reports a sweep
        # with any of these as such rather than as a remediation.
        result.scan_failed.extend(doc_ids)
        log.error(
            "credential_sweep.scan_failed",
            customer=customer_id,
            documents=len(doc_ids),
            error=str(exc),
        )
        return
    if not findings:
        return

    for finding in findings:
        result.rules[finding.rule] = result.rules.get(finding.rule, 0) + 1

    for doc_id in doc_ids:
        c_start, c_end, f_start, f_end = spans[doc_id]
        chunks = by_doc[doc_id]
        new_chunk_text = redacted[c_start:c_end]
        new_fields = redacted[f_start:f_end]
        changed = [
            (chunk_id, new)
            for (chunk_id, old), new in zip(chunks, new_chunk_text, strict=True)
            if new != old
        ]
        versions = docs.get(doc_id, [])
        changed_docs = [
            (version, new_fields[i * 2], new_fields[i * 2 + 1])
            for i, (version, title, preview) in enumerate(versions)
            if new_fields[i * 2] != title or new_fields[i * 2 + 1] != preview
        ]
        if not changed and not changed_docs:
            continue

        result.documents_changed += 1
        result.chunks_rewritten += len(changed)
        result.documents_rewritten += len(changed_docs)

        # Prove it actually went, for this document alone. A rule that matches
        # text it cannot replace would otherwise be reported as a successful
        # remediation.
        try:
            _, still = await redact_documents_async(list(new_chunk_text) + list(new_fields))
        except ScanUnavailable as exc:
            # The rewrite may well have worked, but nothing verified it.
            result.scan_failed.append(doc_id)
            log.error(
                "credential_sweep.verify_failed",
                customer=customer_id,
                doc_id=doc_id,
                error=str(exc),
            )
            still = []
        if still:
            result.unresolved.append(doc_id)
            log.error(
                "credential_sweep.unresolved",
                customer=customer_id,
                doc_id=doc_id,
                rules=sorted({f.rule for f in still}),
            )

        if dry_run:
            continue
        if changed:
            await _rewrite_chunks(customer_id, doc_id, changed, result)
        if changed_docs:
            await _rewrite_document_fields(customer_id, doc_id, changed_docs)


async def _rewrite_chunks(
    customer_id: str,
    doc_id: str,
    changed: list[tuple[str, str]],
    result: SweepResult,
) -> None:
    """Rewrite content + hash, handling the collision redaction can create.

    `chunks` has UNIQUE (customer_id, doc_id, content_hash). Two versions of a
    document that differed ONLY in the credential become byte-identical once it
    is replaced, so the second UPDATE violates the constraint and -- before
    this -- aborted the transaction, leaving the rest of the page unswept while
    the caller was told how many chunks had been rewritten.

    The duplicate is deleted rather than kept: it is now the same text as its
    survivor, its whole reason for existing was the value being removed, and
    leaving it would mean retrieval could serve the identical span twice.

    `token_count` is recomputed and `chunk_id` deliberately is NOT: the id
    encodes the ORIGINAL content hash and is referenced elsewhere, so rewriting
    it would orphan those references to remove a value the content no longer
    holds. The id is an identifier here, not a checksum.
    """
    async with with_tenant(customer_id) as conn:
        for chunk_id, new in changed:
            new_hash = _chunk_hash(new)
            updated = await conn.fetchval(
                """
                UPDATE chunks
                   SET content = $3, content_hash = $4,
                       token_count = length($3) / 4
                 WHERE customer_id = $1 AND chunk_id = $2
                   AND NOT EXISTS (
                       SELECT 1 FROM chunks other
                        WHERE other.customer_id = $1
                          AND other.doc_id = $5
                          AND other.content_hash = $4
                          AND other.chunk_id <> $2
                   )
                RETURNING chunk_id
                """,
                customer_id,
                chunk_id,
                new,
                new_hash,
                doc_id,
            )
            if updated is not None:
                continue
            # The guard fired: some other chunk of this document already holds
            # the redacted text. Drop this one.
            await conn.execute(
                "DELETE FROM chunks WHERE customer_id = $1 AND chunk_id = $2",
                customer_id,
                chunk_id,
            )
            result.chunks_deduped += 1
            result.chunks_rewritten -= 1
            log.info(
                "credential_sweep.chunk_deduped",
                customer=customer_id,
                doc_id=doc_id,
            )


async def _rewrite_document_fields(
    customer_id: str,
    doc_id: str,
    changed: list[tuple[int, str, str]],
) -> None:
    """Rewrite `documents.title` / `body_preview` for the versions that hold a
    credential. `content_hash` is left alone: it describes the BODY, which does
    not live in this table, and moving it would make every consumer that
    compares it think the document changed."""
    async with with_tenant(customer_id) as conn:
        await conn.executemany(
            """
            UPDATE documents
               SET title = $3, body_preview = $4
             WHERE customer_id = $1 AND doc_id = $2 AND version = $5
            """,
            [
                (customer_id, doc_id, title, preview, version)
                for version, title, preview in changed
            ],
        )
