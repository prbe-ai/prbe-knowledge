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

import structlog

from engine.ingest.secret_redaction import available, redact_documents_async
from engine.shared.db import with_tenant
from engine.shared.exceptions import ScanUnavailable

log = structlog.get_logger(__name__)

#: How many chunks to hold in memory (and rewrite) at once.
_BATCH = 200

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
            "skipped_no_binary": self.skipped_no_binary,
        }


def _chunk_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def sweep_credentials(
    customer_id: str,
    *,
    doc_ids: list[str] | None = None,
    dry_run: bool = False,
) -> SweepResult:
    """Find and replace stored credentials for one tenant.

    `doc_ids=None` sweeps every document the tenant has. `dry_run=True` reports
    what would change and writes nothing — the intended first pass, because
    triaging "is this a live key or AWS's documentation example" is a judgement
    a human makes once per finding and a sweep should not make for them.
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
    else:
        async with with_tenant(customer_id) as conn:
            id_rows = await conn.fetch(
                "SELECT DISTINCT doc_id FROM chunks WHERE customer_id = $1 ORDER BY doc_id",
                customer_id,
            )
        pending = [r["doc_id"] for r in id_rows]

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
        await _sweep_page(customer_id, by_doc, result, dry_run=dry_run)
        by_doc = {}

    return result


async def _sweep_page(
    customer_id: str,
    by_doc: dict[str, list[tuple[str, str]]],
    result: SweepResult,
    *,
    dry_run: bool,
) -> None:
    """One page of documents: scan, verify, rewrite."""
    for doc_id, chunks in by_doc.items():
        result.documents_scanned += 1
        result.chunks_scanned += len(chunks)
        contents = [c for _, c in chunks]
        try:
            redacted, findings = await redact_documents_async(contents)
        except ScanUnavailable as exc:
            # One document the scanner could not read must not abort the page
            # AND must not be counted as clean. Record it; the caller reports
            # a sweep with any of these as inconclusive.
            result.scan_failed.append(doc_id)
            log.error(
                "credential_sweep.scan_failed",
                customer=customer_id,
                doc_id=doc_id,
                error=str(exc),
            )
            continue
        if not findings:
            continue
        result.documents_changed += 1
        for finding in findings:
            result.rules[finding.rule] = result.rules.get(finding.rule, 0) + 1

        changed = [
            (chunk_id, new)
            for (chunk_id, old), new in zip(chunks, redacted, strict=True)
            if new != old
        ]
        result.chunks_rewritten += len(changed)

        # Prove it actually went. A rule that matches text it cannot replace
        # would otherwise be reported as a successful remediation.
        try:
            _, still = await redact_documents_async(redacted)
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

        if dry_run or not changed:
            continue

        async with with_tenant(customer_id) as conn:
            for start in range(0, len(changed), _BATCH):
                window = changed[start:start + _BATCH]
                # `token_count` is recomputed and `chunk_id` deliberately is
                # NOT: the id encodes the ORIGINAL content hash and is
                # referenced elsewhere, so rewriting it would orphan those
                # references to remove a value the content no longer holds.
                # The id is an identifier here, not a checksum.
                await conn.executemany(
                    """
                    UPDATE chunks
                       SET content = $3, content_hash = $4,
                           token_count = length($3) / 4
                     WHERE customer_id = $1 AND chunk_id = $2
                    """,
                    [
                        (customer_id, chunk_id, new, _chunk_hash(new))
                        for chunk_id, new in window
                    ],
                )

        log.warning(
            "credential_sweep.document_remediated",
            customer=customer_id,
            doc_id=doc_id,
            chunks=len(changed),
            rules=sorted({f.rule for f in findings}),
        )
