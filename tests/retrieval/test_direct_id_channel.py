"""/retrieve/direct's id channel against a real database, as a role FORCE RLS binds.

The id lookup is plain SQL (no pg_search), so this runs on the stock test
Postgres. Documents are stored the way prod stores them (read 2026-10-09 on
the research plane): a run is a custom_ingest document whose source_id is
`experiments:run:<uuid>`; a GitHub PR is `github:<owner/repo>:pr:<N>` with
source_id `owner/repo#N`, and its reviews carry the PR's URL plus a fragment.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio

from engine.retrieval import direct
from engine.retrieval.retrievers import id_lookup
from engine.shared.custom_ingest import custom_ingest_doc_id
from engine.shared.db import raw_conn, with_tenant
from engine.shared.partitions import ensure_tenant_partition

TENANT = "direct-id-tenant"
OTHER = "direct-id-other"
PROJECT = "240f2b75-a2ee-4189-ad05-9883bd5514a5"
RUN_UUID = "61c0db57-56d1-49a4-a0a3-3f29cd7e98eb"
RUN_DOC = custom_ingest_doc_id(TENANT, "experiments", f"run:{RUN_UUID}")
PR_DOC = "github:prbe-ai/research-os:pr:2499"
REVIEW_DOC = "github:prbe-ai/research-os:review:5463735468"
PR_URL = "https://github.com/prbe-ai/research-os/pull/2499"


async def _doc(conn, tenant, doc_id, *, source, source_id, url, doc_type, metadata) -> None:
    now = datetime.now(UTC)
    await conn.execute(
        """INSERT INTO documents (
               doc_id, version, customer_id, source_system, source_id, source_url,
               doc_class, doc_type, content_type, content_hash, title,
               created_at, updated_at, valid_from, acl, metadata
           ) VALUES ($1, 1, $2, $3, $4, $5, 'raw_source', $6, 'text/plain', $1, $1,
                     $7, $7, $7, '{}', $8::jsonb)""",
        doc_id, tenant, source, source_id, url, doc_type, now, json.dumps(metadata),
    )
    for index, (kind, content) in enumerate(
        (("metadata", f"metadata of {doc_id}"), ("content", f"body of {doc_id}"))
    ):
        await conn.execute(
            """INSERT INTO chunks (
                   chunk_id, doc_id, customer_id, chunk_index, content, content_hash,
                   token_count, first_seen_version, last_seen_version, kind, visibility
               ) VALUES ($1, $2, $3, $4, $5, $1, 3, 1, 2147483647, $6, 'approved')""",
            f"{doc_id}:c_{kind}", doc_id, tenant, index, content, kind,
        )


@pytest_asyncio.fixture
async def seeded(live_db, monkeypatch):
    async with raw_conn() as conn:
        for tenant in (TENANT, OTHER):
            await conn.execute(
                "INSERT INTO customers (customer_id, display_name, api_key_hash) "
                "VALUES ($1, $1, $1)",
                tenant,
            )
            await ensure_tenant_partition(conn, tenant)
        run_meta = {"source_key": "experiments", "project_id": PROJECT}
        await _doc(
            conn, TENANT, RUN_DOC, source="custom_ingest", source_id=f"experiments:run:{RUN_UUID}",
            url="https://prbe.ai/dashboard/ingestion", doc_type="custom.experiment.run",
            metadata=run_meta,
        )
        # The same run id in another tenant: must never surface.
        await _doc(
            conn, OTHER, custom_ingest_doc_id(OTHER, "experiments", f"run:{RUN_UUID}"),
            source="custom_ingest", source_id=f"experiments:run:{RUN_UUID}",
            url="https://prbe.ai/dashboard/ingestion", doc_type="custom.experiment.run",
            metadata=run_meta,
        )
        await _doc(
            conn, TENANT, PR_DOC, source="github", source_id="prbe-ai/research-os#2499",
            url=PR_URL, doc_type="github.pull_request", metadata={},
        )
        await _doc(
            conn, TENANT, REVIEW_DOC, source="github",
            source_id="prbe-ai/research-os#review:5463735468",
            url=f"{PR_URL}#pullrequestreview-5463735468", doc_type="github.review", metadata={},
        )

    # Read as a fresh non-superuser role: a superuser bypasses the FORCE RLS
    # the lookup relies on, and a policy failure would hide behind it.
    role = f"direct_id_{uuid4().hex}"
    async with raw_conn() as conn:
        schema = await conn.fetchval("SELECT quote_ident(current_schema())")
        await conn.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS")
        await conn.execute(f"GRANT USAGE ON SCHEMA {schema}, public TO {role}")
        await conn.execute(f"GRANT SELECT ON documents, chunks TO {role}")

    @asynccontextmanager
    async def restricted(tenant):
        async with with_tenant(tenant) as conn:
            await conn.execute(f"SET LOCAL ROLE {role}")
            yield conn

    for module in (direct, id_lookup):
        monkeypatch.setattr(module, "with_tenant", restricted)
    try:
        yield
    finally:
        async with raw_conn() as conn:
            await conn.execute(f"DROP OWNED BY {role}")
            await conn.execute(f"DROP ROLE {role}")


async def _ask(query: str, **scope) -> direct.DirectRetrieveResponse:
    return await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query=query, channels=["id"], **scope), TENANT
    )


@pytest.mark.integration
async def test_a_typed_run_uuid_resolves_to_its_run(seeded) -> None:
    response = await _ask(f"what happened in run {RUN_UUID}?")
    assert not response.lost_channels, response.model_dump()
    assert [r.doc_id for r in response.results] == [RUN_DOC]
    (result,) = response.results
    assert [m.channel for m in result.matched_via] == ["id"]
    # The body chunk, never the metadata chunk.
    assert [c.content for c in result.chunks] == [f"body of {RUN_DOC}"]


@pytest.mark.integration
async def test_a_pasted_github_pr_link_resolves_to_the_pr_not_its_reviews(seeded) -> None:
    response = await _ask(PR_URL)
    assert not response.lost_channels, response.model_dump()
    assert [r.doc_id for r in response.results] == [PR_DOC]


@pytest.mark.integration
async def test_the_request_scope_holds_inside_the_lookup(seeded) -> None:
    in_scope = {"sources": ["custom_ingest"], "source_keys": ["experiments"],
                "doc_types": ["custom.experiment.run"], "scope": {"project_id": PROJECT}}
    assert [r.doc_id for r in (await _ask(RUN_UUID, **in_scope)).results] == [RUN_DOC]
    for out_of_scope in (
        {"sources": ["github"]},
        {"source_keys": ["artifacts"]},
        {"doc_types": ["github.pull_request"]},
        {"scope": {"project_id": "240f2b75-a2ee-4189-ad05-9883bd5514a6"}},
    ):
        response = await _ask(RUN_UUID, **out_of_scope)
        assert response.results == [], out_of_scope
        assert not response.lost_channels
        # The live gate re-checks scope after every channel, so the response
        # alone cannot show the lookup enforced it. Ask the channel itself.
        filters = {"sources": None, "source_keys": None, "doc_types": None, "project_id": None}
        filters.update({k: v for k, v in out_of_scope.items() if k != "scope"})
        filters["project_id"] = out_of_scope.get("scope", {}).get("project_id")
        assert await direct.id_search(TENANT, RUN_UUID, top_k=10, **filters) == [], out_of_scope


@pytest.mark.integration
async def test_a_query_without_an_identifier_reads_nothing(seeded) -> None:
    response = await _ask("loss curve diverged")
    assert response.results == []
    assert not response.lost_channels
