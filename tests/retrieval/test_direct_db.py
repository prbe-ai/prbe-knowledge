"""Real pg_search and pgvector coverage; only the embedding provider is stubbed.

Uses the existing pg_search_db fixture, which loudly skips databases without
the production BM25 extension. No network/model calls or index backfills.
"""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from engine.retrieval import direct
from engine.retrieval.retrievers import bm25, vector
from engine.shared.db import raw_conn, with_tenant


@pytest.mark.integration
async def test_direct_real_retrievers_respect_tenant_scope_and_live_documents(pg_search_db, monkeypatch):
    customer = "direct-recall-test"
    project = "240f2b75-a2ee-4189-ad05-9883bd5514a5"
    other_project = "240f2b75-a2ee-4189-ad05-9883bd5514a6"
    now = datetime.now(UTC)
    embedding = "[" + ",".join(["1"] + ["0"] * 3071) + "]"
    cases = [
        ("semantic", customer, "adaptive policy optimization", embedding, project, False, "approved", "experiments"),
        ("keyword", customer, "latent exact token", None, project, False, "approved", "experiments"),
        ("foreign", customer + "-other", "latent", embedding, project, False, "approved", "experiments"),
        ("wrongproject", customer, "latent", embedding, other_project, False, "approved", "experiments"),
        ("wrongcorpus", customer, "latent", embedding, project, False, "approved", "other"),
        ("deleted", customer, "latent", embedding, project, True, "approved", "experiments"),
        ("draft", customer, "latent", embedding, project, False, "draft", "experiments"),
    ]
    async with raw_conn() as conn:
        for tenant in (customer, customer + "-other"):
            await conn.execute(
                "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,$1,$1)", tenant,
            )
        for suffix, tenant, content, vec, pid, deleted, visibility, key in cases:
            doc_id = f"{tenant}:{suffix}"
            await conn.execute(
                """INSERT INTO documents (
                    doc_id,version,customer_id,source_system,source_id,source_url,
                    doc_class,doc_type,content_type,content_hash,title,
                    created_at,updated_at,valid_from,acl,metadata,deleted_at,visibility
                ) VALUES ($1,1,$2,'custom_ingest',$1,'','raw_source',
                    'custom.experiment.run','text/plain',$1,$3,$4,$4,$4,'{}',
                    $5::jsonb,CASE WHEN $6::bool THEN $4::timestamptz ELSE NULL END,$7)""",
                doc_id, tenant, content, now, json.dumps({"project_id": pid, "source_key": key}),
                deleted, visibility,
            )
            await conn.execute(
                """INSERT INTO chunks (
                    chunk_id,doc_id,customer_id,chunk_index,content,content_hash,token_count,
                    embedding_v2,first_seen_version,last_seen_version,visibility
                ) VALUES ($1,$1,$2,0,$3,$1,5,$4::halfvec,1,1,$5)""",
                doc_id, tenant, content, vec, visibility,
            )

    class Embedder:
        async def embed_query(self, _query):
            return [1.0] + [0.0] * 3071

    monkeypatch.setattr(vector, "get_embedder_v2", lambda: Embedder())
    # The fixture seeds as its DB owner. Execute retrieval with a fresh
    # non-superuser role so a policy failure cannot hide behind superuser bypass.
    role = f"direct_search_{uuid4().hex}"
    async with raw_conn() as conn:
        schema = await conn.fetchval("SELECT quote_ident(current_schema())")
        await conn.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS")
        await conn.execute(f"GRANT USAGE ON SCHEMA {schema}, public, paradedb TO {role}")
        await conn.execute(f"GRANT SELECT ON documents, chunks TO {role}")
        assert await conn.fetchval(
            "SELECT bool_and(relforcerowsecurity) FROM pg_class WHERE oid IN ('documents'::regclass,'chunks'::regclass)"
        )

    @asynccontextmanager
    async def restricted(tenant):
        async with with_tenant(tenant) as conn:
            await conn.execute(f"SET LOCAL ROLE {role}")
            yield conn

    for module in (direct, vector, bm25):
        monkeypatch.setattr(module, "with_tenant", restricted)
    try:
        response = await direct.retrieve_direct(
            direct.DirectRetrieveRequest(
                query="latent", source_keys=["experiments"], sources=["custom_ingest"],
                doc_types=["custom.experiment.run"], scope={"project_id": project},
            ), customer,
        )
    finally:
        async with raw_conn() as conn:
            await conn.execute(f"DROP OWNED BY {role}")
            await conn.execute(f"DROP ROLE {role}")
    assert not response.lost_channels, response.model_dump()
    by_id = {result.doc_id: result for result in response.results}
    assert set(by_id) == {f"{customer}:semantic", f"{customer}:keyword"}
    assert {m.channel for m in by_id[f"{customer}:semantic"].matched_via} == {"vector"}
    assert {m.channel for m in by_id[f"{customer}:keyword"].matched_via} == {"bm25"}
