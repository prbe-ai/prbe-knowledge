"""One-off backfill: regenerate `directed_vectors` for every existing
wiki page using the currently-configured `DIRECTED_PHRASES_MODEL`.

Why this exists:

  PR #228 (2026-05-10) swapped directed-phrase generation from Haiku
  4.5 to Gemini 3 Flash. Existing wiki pages still carry Haiku-authored
  phrases until they're organically re-synthesized. This script forces
  an immediate regen for every `wiki:*` document, surface='llm', for
  every customer with wiki pages.

  Engineer-pinned phrases (source='human') are preserved untouched —
  reconciliation only diffs the human-pin set against frontmatter, and
  this backfill doesn't change frontmatter.

How:

  For each customer with wiki pages:
    1. Insert a `wiki_synthesis_runs` row with kind='wake', stage='synthesis'
       to get a `run_id` that satisfies the `ck_dv_run_for_llm` CHECK on
       directed_vectors.
    2. Iterate each wiki page, call `persist_directed_vectors` with the
       new run_id and the configured provider.
    3. Mark the synthesis run 'complete' with `pages_updated` count.

  ~1 LLM call per wiki page (~$0.0005 with Gemini 3 Flash), single-process.
  Idempotent: re-running re-replaces every page's source='llm' rows with
  a fresh set under a new run_id. Old rows are deleted via the existing
  `_reconcile_llm` always-delete pattern in directed_phrases.py.

Run on a fly worker box:

    flyctl ssh console -a prbe-knowledge-worker
    cd /app
    .venv/bin/python -m scripts.backfill_directed_phrases [--customer-id ID] [--dry-run]

Pass `--customer-id` to scope to one tenant. Default is all customers
with at least one wiki page.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from services.synthesis.directed_phrases import persist_directed_vectors
from services.synthesis.providers import get_directed_phrases_provider
from shared.constants import (
    DIRECTED_PHRASES_MODEL,
    WIKI_DOC_TYPE_PREFIX,
    WIKI_INDEX_DOC_TYPE,
)
from shared.db import close_pool, init_pool, raw_conn, with_tenant
from shared.logging import get_logger

log = get_logger(__name__)


async def _all_customers_with_wiki() -> list[str]:
    """Customers with at least one live wiki page. raw_conn bypasses RLS;
    fly's prbe role has BYPASSRLS so the cross-tenant query works.
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT customer_id
            FROM documents
            WHERE doc_type LIKE $1
              AND doc_type <> $2
              AND valid_to IS NULL
            ORDER BY customer_id
            """,
            f"{WIKI_DOC_TYPE_PREFIX}%",
            WIKI_INDEX_DOC_TYPE,
        )
    return [r["customer_id"] for r in rows]


async def _list_wiki_pages(customer_id: str) -> list[dict]:
    """Fetch each live wiki page's title + assembled body + frontmatter
    for a single tenant. The wiki index page is excluded (no directed
    vectors for index pages).
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT
              d.doc_id,
              d.title,
              d.metadata->'frontmatter' AS frontmatter,
              COALESCE(
                string_agg(c.content, E'\n\n' ORDER BY c.chunk_index),
                ''
              ) AS body
            FROM documents d
            LEFT JOIN chunks c
              ON c.doc_id = d.doc_id
             AND c.customer_id = d.customer_id
             AND COALESCE(c.kind, 'content') = 'content'
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE d.customer_id = $1
              AND d.doc_type LIKE $2
              AND d.doc_type <> $3
              AND d.valid_to IS NULL
            GROUP BY d.doc_id, d.title, d.metadata
            ORDER BY d.doc_id
            """,
            customer_id,
            f"{WIKI_DOC_TYPE_PREFIX}%",
            WIKI_INDEX_DOC_TYPE,
        )
    return [dict(r) for r in rows]


async def _open_run(customer_id: str) -> int:
    """Insert a wiki_synthesis_runs row so the backfill has a valid
    run_id for `directed_vectors.synthesis_run_id`. kind='wake' is the
    'manually triggered ad-hoc' kind per the schema's CHECK constraint.
    """
    async with with_tenant(customer_id) as conn:
        run_id = await conn.fetchval(
            """
            INSERT INTO wiki_synthesis_runs
              (customer_id, kind, stage, status)
            VALUES ($1, 'wake', 'synthesis', 'running')
            RETURNING run_id
            """,
            customer_id,
        )
    return int(run_id)


async def _close_run(
    customer_id: str, run_id: int, pages_updated: int, status: str
) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_runs
            SET status = $3,
                finished_at = NOW(),
                pages_updated = $4
            WHERE customer_id = $1 AND run_id = $2
            """,
            customer_id,
            run_id,
            status,
            pages_updated,
        )


async def backfill_customer(
    customer_id: str, dry_run: bool
) -> tuple[int, int]:
    """Returns (pages_seen, pages_succeeded). When dry_run, no LLM
    calls and no run row is opened.
    """
    pages = await _list_wiki_pages(customer_id)
    if not pages:
        return 0, 0

    log.info(
        "backfill.customer_start",
        customer=customer_id,
        pages=len(pages),
        dry_run=dry_run,
        model=DIRECTED_PHRASES_MODEL,
    )

    if dry_run:
        for p in pages:
            log.info(
                "backfill.dry_run_page",
                customer=customer_id,
                doc_id=p["doc_id"],
                title=p["title"],
                body_chars=len(p["body"] or ""),
            )
        return len(pages), 0

    run_id = await _open_run(customer_id)
    # Single provider for the whole customer's batch (lazy-construct the
    # client once, not per-page).
    provider = get_directed_phrases_provider()
    succeeded = 0
    for p in pages:
        title = p["title"] or p["doc_id"]
        body = p["body"] or ""
        try:
            res = await persist_directed_vectors(
                customer_id=customer_id,
                doc_id=p["doc_id"],
                page_title=title,
                page_body=body,
                frontmatter=p["frontmatter"],
                synthesis_run_id=run_id,
                provider=provider,
            )
            log.info(
                "backfill.page_done",
                customer=customer_id,
                doc_id=p["doc_id"],
                run_id=run_id,
                llm_added=res.llm_added,
                llm_removed=res.llm_removed,
                llm_failed=res.llm_failed,
                llm_dropped_internal=res.llm_dropped_internal,
                llm_dropped_vs_human=res.llm_dropped_vs_human,
            )
            if not res.llm_failed:
                succeeded += 1
        except Exception as exc:
            log.error(
                "backfill.page_failed",
                customer=customer_id,
                doc_id=p["doc_id"],
                run_id=run_id,
                error=str(exc),
                error_class=type(exc).__name__,
            )

    final_status = "complete" if succeeded == len(pages) else "partial"
    await _close_run(customer_id, run_id, succeeded, final_status)

    log.info(
        "backfill.customer_complete",
        customer=customer_id,
        run_id=run_id,
        pages=len(pages),
        succeeded=succeeded,
        failed=len(pages) - succeeded,
        status=final_status,
    )
    return len(pages), succeeded


async def main(args: argparse.Namespace) -> int:
    # CLI scripts must own the pool lifecycle — `with_tenant` / `raw_conn`
    # both fail loud if init_pool hasn't been called.
    await init_pool()
    try:
        return await _run(args)
    finally:
        await close_pool()


async def _run(args: argparse.Namespace) -> int:
    if args.customer_id:
        customers = [args.customer_id]
    else:
        customers = await _all_customers_with_wiki()

    log.info(
        "backfill.start",
        customers=len(customers),
        dry_run=args.dry_run,
        model=DIRECTED_PHRASES_MODEL,
    )

    total_pages = 0
    total_succeeded = 0
    for cid in customers:
        n_pages, n_succeeded = await backfill_customer(cid, args.dry_run)
        total_pages += n_pages
        total_succeeded += n_succeeded

    log.info(
        "backfill.done",
        total_pages=total_pages,
        succeeded=total_succeeded,
        failed=0 if args.dry_run else total_pages - total_succeeded,
        dry_run=args.dry_run,
    )
    # Print a one-line summary to stdout for the operator.
    if args.dry_run:
        print(
            f"backfill (dry_run): {total_pages} pages WOULD be regenerated "
            f"with {DIRECTED_PHRASES_MODEL}",
            flush=True,
        )
        return 0
    print(
        f"backfill: {total_succeeded}/{total_pages} pages regenerated "
        f"({DIRECTED_PHRASES_MODEL})",
        flush=True,
    )
    return 0 if total_succeeded == total_pages else 1


def cli() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Backfill directed_vectors for existing wiki pages using the "
            "currently-configured DIRECTED_PHRASES_MODEL."
        )
    )
    ap.add_argument(
        "--customer-id",
        default=None,
        help="Backfill a single customer (default: all customers with wiki pages).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List pages without calling the LLM or writing to directed_vectors.",
    )
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args)))


if __name__ == "__main__":
    cli()
