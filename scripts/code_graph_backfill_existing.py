"""One-shot code-graph backfill for tenants that connected GitHub before PR-A.

The bridge fan-out fires on `installation` and `installation_repositories`
GitHub webhooks — events that already happened (and were ignored) for any
tenant connected before PR-A landed. This script reproduces what those
webhooks would have done:

  1. Find customers with an active `integration_tokens` row for github.
  2. Mint a fresh installation token via prbe-backend.
  3. List the repos the installation can see (GET /installation/repositories).
  4. For each repo, look up the default branch and HEAD SHA.
  5. Call bridge.enqueue_initial_backfill(customer, repo, head_sha, ...).

Idempotent end-to-end:
  - Bridge dedupes on (customer_id, source_event_id) where the event id
    embeds the SHA, so re-running with the same HEAD is a no-op.
  - The pipeline's per-file content_hash cache (code_repo_state) means a
    second run after a few new commits only re-extracts changed files.

Usage:
    # One tenant (recommended for first runs)
    .venv/bin/python -m scripts.code_graph_backfill_existing --customer-id cust-prbe-founders

    # All tenants with active github integration (gated behind --yes)
    .venv/bin/python -m scripts.code_graph_backfill_existing --all-customers --yes

    # See what would happen without enqueueing
    .venv/bin/python -m scripts.code_graph_backfill_existing --customer-id cust-prbe-founders --dry-run

Environment (same as the worker — point at prod via env, secrets from fly):
    DATABASE_URL, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
    R2_BUCKET_PREFIX, BACKEND_INTERNAL_KEY (for fetch_github_installation_token),
    PRBE_BACKEND_URL, TOKEN_ENCRYPTION_KEY.

Suggested run path: `flyctl ssh console -a prbe-knowledge-worker` then
`uv run python -m scripts.code_graph_backfill_existing --customer-id …`.
That keeps prod creds off your laptop.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

import httpx

from services.ingestion.code_graph.bridge import enqueue_initial_backfill
from shared.backend_client import fetch_github_installation_token
from shared.config import get_settings
from shared.constants import GITHUB_INSTALLATION_SCOPE_PREFIX, SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_GITHUB_PAGE_SIZE = 100


@dataclass(slots=True)
class _Tenant:
    customer_id: str
    integration_token_id: int
    installation_id: str  # parsed from `scope = installation:<id>`


@dataclass(slots=True)
class _Repo:
    full_name: str  # "owner/name"
    default_branch: str
    head_sha: str


async def _list_tenants(customer_id: str | None) -> list[_Tenant]:
    """Active github integration_tokens with installation-scoped credentials.

    The installation-id is parsed from `scope='installation:<id>'`. Older
    PAT-based github tokens (not in our prod tenants any more) don't carry
    an installation id and would need a different path; we skip them.
    """
    where = ["source_system = 'github'", "status = 'active'"]
    params: list[object] = []
    if customer_id is not None:
        params.append(customer_id)
        where.append(f"customer_id = ${len(params)}")
    sql = f"""
        SELECT id, customer_id, scope
        FROM integration_tokens
        WHERE {" AND ".join(where)}
    """

    async with raw_conn() as conn:
        rows = await conn.fetch(sql, *params)

    tenants: list[_Tenant] = []
    for r in rows:
        scope = r["scope"] or ""
        if not scope.startswith(GITHUB_INSTALLATION_SCOPE_PREFIX):
            log.warning(
                "code_graph.backfill.skip_non_installation",
                customer=r["customer_id"],
                scope=scope[:32],
            )
            continue
        installation_id = scope[len(GITHUB_INSTALLATION_SCOPE_PREFIX):]
        tenants.append(
            _Tenant(
                customer_id=r["customer_id"],
                integration_token_id=int(r["id"]),
                installation_id=installation_id,
            )
        )
    return tenants


async def _list_installation_repos(
    http: httpx.AsyncClient, bearer: str
) -> list[dict]:
    """All repos the installation token can read. Paginated."""
    repos: list[dict] = []
    page = 1
    while True:
        resp = await http.get(
            f"{_GITHUB_API}/installation/repositories",
            params={"per_page": _GITHUB_PAGE_SIZE, "page": page},
            headers={
                "Authorization": f"Bearer {bearer}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        resp.raise_for_status()
        body = resp.json()
        chunk = body.get("repositories", []) or []
        repos.extend(chunk)
        if len(chunk) < _GITHUB_PAGE_SIZE:
            break
        page += 1
    return repos


async def _resolve_head_sha(
    http: httpx.AsyncClient, bearer: str, full_name: str, default_branch: str
) -> str | None:
    """HEAD SHA on the default branch, or None on permission/404 errors."""
    resp = await http.get(
        f"{_GITHUB_API}/repos/{full_name}/git/ref/heads/{default_branch}",
        headers={
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if resp.status_code == 404:
        log.warning(
            "code_graph.backfill.head_ref_404", repo=full_name, branch=default_branch
        )
        return None
    resp.raise_for_status()
    body = resp.json()
    sha = (body.get("object") or {}).get("sha")
    return sha if isinstance(sha, str) else None


async def _backfill_tenant(
    http: httpx.AsyncClient, tenant: _Tenant, *, dry_run: bool
) -> tuple[int, int]:
    """Returns (enqueued_count, skipped_count)."""
    bearer, _expires = await fetch_github_installation_token(
        http, customer_id=tenant.customer_id
    )

    raw_repos = await _list_installation_repos(http, bearer)
    log.info(
        "code_graph.backfill.repos_visible",
        customer=tenant.customer_id,
        installation=tenant.installation_id,
        repo_count=len(raw_repos),
    )

    enqueued = 0
    skipped = 0
    for r in raw_repos:
        full_name = r.get("full_name")
        default_branch = r.get("default_branch")
        archived = bool(r.get("archived"))
        if not isinstance(full_name, str) or not isinstance(default_branch, str):
            skipped += 1
            continue
        if archived:
            log.info(
                "code_graph.backfill.skip_archived",
                customer=tenant.customer_id,
                repo=full_name,
            )
            skipped += 1
            continue

        sha = await _resolve_head_sha(http, bearer, full_name, default_branch)
        if sha is None:
            skipped += 1
            continue

        if dry_run:
            print(
                f"  [dry-run] would enqueue {tenant.customer_id} "
                f"{full_name}@{sha[:8]} (branch={default_branch})"
            )
            enqueued += 1
            continue

        new = await enqueue_initial_backfill(
            customer_id=tenant.customer_id,
            repo=full_name,
            head_sha=sha,
            integration_token_id=tenant.integration_token_id,
            originating_source=SourceSystem.GITHUB,
        )
        marker = "enqueued" if new else "deduped"
        print(
            f"  {marker} {tenant.customer_id} {full_name}@{sha[:8]} "
            f"(branch={default_branch})"
        )
        enqueued += 1

    return enqueued, skipped


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    try:
        tenants = await _list_tenants(args.customer_id)
        if not tenants:
            print("No matching tenants with active github integration.", file=sys.stderr)
            return 1
        if args.all_customers and not args.yes and len(tenants) > 1:
            print(
                f"--all-customers matched {len(tenants)} tenants; pass --yes to "
                "confirm. Use --customer-id <id> to target one.",
                file=sys.stderr,
            )
            return 1

        async with httpx.AsyncClient(timeout=30.0) as http:
            total_enqueued = 0
            total_skipped = 0
            for tenant in tenants:
                print(
                    f"\n== {tenant.customer_id} (installation={tenant.installation_id}) =="
                )
                enq, skp = await _backfill_tenant(http, tenant, dry_run=args.dry_run)
                total_enqueued += enq
                total_skipped += skp

        print(
            f"\nDone. tenants={len(tenants)} "
            f"{'would-enqueue' if args.dry_run else 'enqueued'}={total_enqueued} "
            f"skipped={total_skipped}"
        )
        return 0
    finally:
        await close_pool()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--customer-id", help="Single tenant (cust-…)")
    g.add_argument(
        "--all-customers",
        action="store_true",
        help="Every tenant with active github integration",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List repos that would be enqueued; don't write R2 / queue rows",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Required with --all-customers when more than one tenant matches",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(_parse_args())))
