"""Integration tests for /internal/feature-nodes/upsert.

Exercise the full DB transaction against live Postgres (the live_db
fixture). The endpoint is gated by X-Internal-Knowledge-Key and is
tenant-scoped via X-Prbe-Customer.

Covers:
  * Happy path (minimal + full with pre-seeded evidence).
  * Race-loser: our finalize lands BEFORE the GitHub PR webhook ingest.
    The endpoint stubs the PR Document so the FEATURE→OWNS edge
    lands; the later heavy ingest ON-CONFLICT-merges richer
    properties onto the same node_id.
  * Race-winner: PR Document already exists from a prior ingest. The
    endpoint reuses that node_id (no duplicate row).
  * Evidence is LOOKUP-ONLY — missing evidence docs cause the
    corresponding FEATURE→DOCUMENTS edge to drop silently (logged);
    found evidence docs land an edge to the existing node_id.
  * Evidence dedupe + length cap defences (F5 from adversarial review).
  * URL parser robustness against trailing slash / fragments / garbage
    (F1 from adversarial review).
  * Auth gates: 401 on missing/wrong internal key; 400 on missing
    customer header.
  * Cross-tenant RLS isolation.

Mirrors the httpx.AsyncClient + ASGITransport + lifespan_context
pattern from test_entity_clusters_routes.py — TestClient spins its
own event loop and clashes with the asyncpg pool the live_db fixture
binds to.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import httpx
import orjson
import pytest
import pytest_asyncio
from httpx import ASGITransport

from services.ingestion.graph_writer import upsert_nodes
from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.constants import DocType, NodeLabel, SourceSystem
from shared.db import close_pool, init_pool, raw_conn, with_tenant
from shared.models import GraphNodeSpec

CUSTOMER_ID = "cust-feature-nodes"
INTERNAL_KEY = "test-internal-key"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    get_settings.cache_clear()  # type: ignore[attr-defined]


def _headers(
    *, customer: str | None = CUSTOMER_ID, key: str | None = INTERNAL_KEY
) -> dict[str, str]:
    h: dict[str, str] = {}
    if customer is not None:
        h["X-Prbe-Customer"] = customer
    if key is not None:
        h["X-Internal-Knowledge-Key"] = key
    return h


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client bound to the app's own DB pool (via lifespan)."""
    await close_pool()
    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
        app.router.lifespan_context(app),
    ):
        yield c
    await init_pool(settings)


def _request_body(
    *,
    canonical_id: str = "feature:gh:acme/widgets#42",
    repo_full_name: str = "acme/widgets",
    pr_number: int = 42,
    why: str = "Async webhooks cut queue depth ~40%.",
    title: str = "Add async webhooks",
    merged_at: str = "2026-05-15T12:00:00+00:00",
    merge_sha: str = "deadbeef",
    evidence_doc_ids: list[str] | None = None,
    author_id: str | None = "richardwei6",
    source_pr_url: str | None = None,
) -> dict[str, object]:
    return {
        "canonical_id": canonical_id,
        "title": title,
        "why": why,
        # `is None` not `or` — must let empty-string + other falsy
        # invalid URLs through verbatim so the parser tests exercise them.
        "source_pr_url": (
            f"https://github.com/{repo_full_name}/pull/{pr_number}"
            if source_pr_url is None
            else source_pr_url
        ),
        "merged_at": merged_at,
        "merge_sha": merge_sha,
        "evidence_doc_ids": evidence_doc_ids or [],
        "author_id": author_id,
        "repo_full_name": repo_full_name,
    }


async def _seed_customer(conn: asyncpg.Connection, customer_id: str = CUSTOMER_ID) -> None:
    await conn.execute(
        "INSERT INTO customers(customer_id, display_name, api_key_hash) "
        "VALUES ($1, 'feat-test', $2) ON CONFLICT (customer_id) DO NOTHING",
        customer_id, f"hash-{customer_id}",
    )


async def _seed_doc(
    customer_id: str,
    canonical_id: str,
    *,
    source_system: str = "slack",
) -> int:
    """Pre-seed a Document graph node + provenance row, mirroring what
    a heavy source ingest would produce. Used to set up the lookup-only
    evidence-doc happy path. Acquires its own with_tenant connection
    so callers don't need to be inside one already.
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO graph_nodes(customer_id, label, canonical_id, properties)
            VALUES ($1, 'Document', $2, '{}'::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO UPDATE
                SET updated_at = NOW()
            RETURNING node_id
            """,
            customer_id, canonical_id,
        )
        node_id = row["node_id"]
        await conn.execute(
            """
            INSERT INTO graph_node_provenance(node_id, customer_id, source_system)
            VALUES ($1, $2, $3)
            ON CONFLICT DO NOTHING
            """,
            node_id, customer_id, source_system,
        )
        return node_id


# ---------------------------------------------------------------------------
# Auth gates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_401_when_key_missing(client: httpx.AsyncClient) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers={"X-Prbe-Customer": CUSTOMER_ID},  # no internal key
        json=_request_body(),
    )
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_401_when_key_wrong(client: httpx.AsyncClient) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(key="totally-wrong"),
        json=_request_body(),
    )
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_400_when_customer_header_missing(client: httpx.AsyncClient) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers={"X-Internal-Knowledge-Key": INTERNAL_KEY},  # no customer
        json=_request_body(),
    )
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_minimal(client: httpx.AsyncClient) -> None:
    """Minimal request (no author, no evidence). FEATURE + PR Document stub
    + Repo stub land; OWNS + TOUCHES edges land. No AUTHORED, no DOCUMENTS.
    """
    async with raw_conn() as conn:
        await _seed_customer(conn)

    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(author_id=None),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["canonical_id"] == "feature:gh:acme/widgets#42"
    assert body["node_id"] is not None
    assert body["edges_created"] == 3  # OWNS + TOUCHES + DESCRIBES

    async with with_tenant(CUSTOMER_ID) as conn:
        nodes = await conn.fetch(
            "SELECT label, canonical_id FROM graph_nodes "
            "WHERE customer_id = $1 ORDER BY label, canonical_id",
            CUSTOMER_ID,
        )
        labels = {(r["label"], r["canonical_id"]) for r in nodes}
        assert ("Feature", "feature:gh:acme/widgets#42") in labels
        assert ("Document", "github:acme/widgets:pr:42") in labels
        assert ("Repo", "acme/widgets") in labels
        # No Person — author_id was None.
        assert not any(r["label"] == "Person" for r in nodes)

        edges = await conn.fetch(
            "SELECT edge_type FROM graph_edges WHERE customer_id = $1 "
            "ORDER BY edge_type",
            CUSTOMER_ID,
        )
        assert [r["edge_type"] for r in edges] == ["DESCRIBES", "OWNS", "TOUCHES"]


@pytest.mark.asyncio
async def test_happy_path_with_preseeded_evidence_and_author(
    client: httpx.AsyncClient,
) -> None:
    """Full request: author + pre-seeded evidence docs across four sources.
    All four edge types (OWNS, AUTHORED, TOUCHES, DOCUMENTS) land; the
    DOCUMENTS edges target the pre-seeded node_ids (no duplicates).
    """
    async with raw_conn() as conn:
        await _seed_customer(conn)

    evidence = [
        "slack:T0APAH5J5PX:C0B529ULVJL:1778916184.160259",
        "notion:1234567890abcdef1234567890abcdef",
        "linear:org-acme:issue:abc-123",
        "github:acme/widgets:commit:deadbeef",
    ]
    seeded_ids: dict[str, int] = {}
    for canonical, source in zip(
        evidence,
        ["slack", "notion", "linear", "github"],
        strict=True,
    ):
        seeded_ids[canonical] = await _seed_doc(
            CUSTOMER_ID, canonical, source_system=source
        )

    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(
            author_id="richardwei6",
            evidence_doc_ids=evidence,
        ),
    )
    assert resp.status_code == 200, resp.text
    # OWNS + TOUCHES + AUTHORED + 4 DOCUMENTS + DESCRIBES = 8.
    assert resp.json()["edges_created"] == 8

    async with with_tenant(CUSTOMER_ID) as conn:
        edges = await conn.fetch(
            "SELECT edge_type, to_node_id FROM graph_edges "
            "WHERE customer_id = $1",
            CUSTOMER_ID,
        )
        by_type: dict[str, list[int]] = {}
        for r in edges:
            by_type.setdefault(r["edge_type"], []).append(r["to_node_id"])
        assert set(by_type.keys()) == {
            "OWNS",
            "TOUCHES",
            "AUTHORED",
            "DOCUMENTS",
            "DESCRIBES",
        }
        assert len(by_type["DOCUMENTS"]) == 4
        # Every DOCUMENTS edge targets a pre-seeded node_id (no duplicates).
        seeded_node_ids = set(seeded_ids.values())
        assert set(by_type["DOCUMENTS"]) == seeded_node_ids

        # Each evidence canonical_id still has exactly one Document row.
        rows = await conn.fetch(
            "SELECT canonical_id, COUNT(*) AS n FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Document' "
            "  AND canonical_id = ANY($2::text[]) "
            "GROUP BY canonical_id",
            CUSTOMER_ID, evidence,
        )
        for r in rows:
            assert r["n"] == 1, f"duplicate row for {r['canonical_id']}"


@pytest.mark.asyncio
async def test_evidence_lookup_only_drops_edge_for_missing_doc(
    client: httpx.AsyncClient,
) -> None:
    """Two evidence doc_ids submitted. One is pre-seeded, one isn't. The
    found-doc edge lands; the missing-doc edge is silently dropped. The
    missing canonical_id MUST NOT appear in graph_nodes (no stubbing).
    """
    async with raw_conn() as conn:
        await _seed_customer(conn)

    present_doc = "slack:T0:C0:1.2"
    missing_doc = "slack:T0:C0:9.9"
    await _seed_doc(CUSTOMER_ID, present_doc, source_system="slack")

    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(
            author_id=None,
            evidence_doc_ids=[present_doc, missing_doc],
        ),
    )
    assert resp.status_code == 200, resp.text
    # OWNS + TOUCHES + 1 DOCUMENTS (only the present one) + DESCRIBES = 4.
    assert resp.json()["edges_created"] == 4

    async with with_tenant(CUSTOMER_ID) as conn:
        # Missing doc was NOT created as a phantom.
        present = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM graph_nodes "
            "  WHERE customer_id = $1 AND canonical_id = $2)",
            CUSTOMER_ID, missing_doc,
        )
        assert not present


@pytest.mark.asyncio
async def test_evidence_cross_tenant_canonical_id_is_not_stubbed(
    client: httpx.AsyncClient,
) -> None:
    """Defense against a malicious / buggy apps-plane caller passing an
    evidence doc_id whose canonical_id embeds a *different* tenant's id
    (e.g. ``custom_ingest:OTHER_TENANT:foo:bar``). Lookup-only means the
    canonical_id only matches if a row with that exact string already
    exists IN THIS tenant — and it doesn't, so the edge silently drops
    and no phantom row is created. F3 from adversarial review.
    """
    async with raw_conn() as conn:
        await _seed_customer(conn)
        await _seed_customer(conn, "other-tenant")
    # Seed a doc with that canonical_id under 'other-tenant' so we can
    # also verify it stayed put (no overwrite).
    foreign_doc = "custom_ingest:other-tenant:src:doc-1"
    await _seed_doc(
        customer_id="other-tenant",
        canonical_id=foreign_doc,
        source_system="custom_ingest",
    )

    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),  # X-Prbe-Customer = CUSTOMER_ID
        json=_request_body(
            author_id=None,
            evidence_doc_ids=[foreign_doc],
        ),
    )
    assert resp.status_code == 200, resp.text
    # OWNS + TOUCHES + 0 DOCUMENTS (cross-tenant canonical_id silently dropped)
    # + DESCRIBES = 3.
    assert resp.json()["edges_created"] == 3

    # No row exists UNDER CUSTOMER_ID with the foreign canonical_id.
    # (Scoped by customer_id explicitly — the test DB role bypasses RLS
    # USING, so the seeded row under 'other-tenant' is otherwise visible
    # to a bare SELECT under the CUSTOMER_ID GUC. The route's own
    # `WHERE customer_id = $1` filter is what makes lookup-only safe
    # in this environment.)
    async with raw_conn() as conn:
        leaked = await conn.fetchval(
            "SELECT COUNT(*) FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Document' "
            "  AND canonical_id = $2",
            CUSTOMER_ID, foreign_doc,
        )
        assert leaked == 0, (
            "cross-tenant leak: foreign canonical_id was stubbed under THIS tenant"
        )
        # Original seeded row is still there under other-tenant.
        kept = await conn.fetchval(
            "SELECT COUNT(*) FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Document' "
            "  AND canonical_id = $2",
            "other-tenant", foreign_doc,
        )
        assert kept == 1


@pytest.mark.asyncio
async def test_evidence_doc_ids_deduped_in_feature_properties(
    client: httpx.AsyncClient,
) -> None:
    """Duplicate evidence doc_ids in the request must NOT produce duplicate
    edges OR duplicate entries on the FEATURE node's
    `properties.evidence_doc_ids` JSONB array. F5 from adversarial review.
    """
    async with raw_conn() as conn:
        await _seed_customer(conn)
    doc = "slack:T0:C0:1.2"
    await _seed_doc(CUSTOMER_ID, doc, source_system="slack")

    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(
            author_id=None,
            evidence_doc_ids=[doc, doc, doc],  # cited 3x
        ),
    )
    assert resp.status_code == 200, resp.text
    # OWNS + TOUCHES + 1 DOCUMENTS (deduped) + DESCRIBES = 4.
    assert resp.json()["edges_created"] == 4

    async with with_tenant(CUSTOMER_ID) as conn:
        # FEATURE.properties.evidence_doc_ids is a single-entry list.
        props_row = await conn.fetchrow(
            "SELECT properties FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Feature' "
            "  AND canonical_id = $2",
            CUSTOMER_ID, "feature:gh:acme/widgets#42",
        )
        props = orjson.loads(props_row["properties"])
        assert props["evidence_doc_ids"] == [doc]


@pytest.mark.asyncio
async def test_evidence_doc_ids_length_cap_rejected(
    client: httpx.AsyncClient,
) -> None:
    """Pydantic rejects evidence_doc_ids over 100 entries (F5)."""
    async with raw_conn() as conn:
        await _seed_customer(conn)
    too_many = [f"slack:T0:C0:{i}" for i in range(101)]
    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(author_id=None, evidence_doc_ids=too_many),
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# URL parser robustness (F1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/acme/widgets/pull/42",
        "https://github.com/acme/widgets/pull/42/",
        "https://github.com/acme/widgets/pull/42/files",
        "https://github.com/acme/widgets/pull/42#issuecomment-99",
        "https://github.com/acme/widgets/pull/42?foo=bar",
    ],
)
@pytest.mark.asyncio
async def test_pr_url_parser_accepts_realistic_variants(
    client: httpx.AsyncClient, url: str
) -> None:
    """Trailing slash / fragments / query strings shouldn't break
    pr_number extraction or produce a phantom canonical_id."""
    async with raw_conn() as conn:
        await _seed_customer(conn)
    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(author_id=None, source_pr_url=url),
    )
    assert resp.status_code == 200, f"{url!r}: {resp.text}"

    async with with_tenant(CUSTOMER_ID) as conn:
        owns_target = await conn.fetchrow(
            """
            SELECT gn.canonical_id
            FROM graph_edges ge
            JOIN graph_nodes gn ON gn.node_id = ge.to_node_id
            WHERE ge.customer_id = $1 AND ge.edge_type = 'OWNS'
            """,
            CUSTOMER_ID,
        )
        assert owns_target["canonical_id"] == "github:acme/widgets:pr:42"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://github.com/acme/widgets",  # no /pull/<N>
        "https://example.com/foo",
        "not even a url",
        "https://github.com/acme/widgets/pull/",  # trailing slash, no number
    ],
)
@pytest.mark.asyncio
async def test_pr_url_parser_rejects_invalid(
    client: httpx.AsyncClient, url: str
) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(author_id=None, source_pr_url=url),
    )
    assert resp.status_code == 422, f"{url!r}: {resp.text}"


# ---------------------------------------------------------------------------
# Race conditions with parallel GitHub webhook ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_race_loser_stub_merges_with_later_ingest(client: httpx.AsyncClient) -> None:
    """The endpoint's PR Document stub lands FIRST; the heavy GitHub ingest
    runs after. The stub's node_id must be preserved; the heavy ingest's
    properties must shallow-JSONB-merge in; the FEATURE→OWNS edge must
    still point at the same node.
    """
    async with raw_conn() as conn:
        await _seed_customer(conn)

    # 1. Endpoint call lands first — stubs the PR Document.
    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(author_id=None),
    )
    assert resp.status_code == 200, resp.text

    async with with_tenant(CUSTOMER_ID) as conn:
        stub_row = await conn.fetchrow(
            "SELECT node_id, properties FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Document' "
            "  AND canonical_id = 'github:acme/widgets:pr:42'",
            CUSTOMER_ID,
        )
        assert stub_row is not None
        stub_node_id = stub_row["node_id"]
        stub_props = orjson.loads(stub_row["properties"])
        assert stub_props == {"doc_type": "github.pull_request"}
        owns_target = await conn.fetchval(
            "SELECT to_node_id FROM graph_edges "
            "WHERE customer_id = $1 AND edge_type = 'OWNS'",
            CUSTOMER_ID,
        )
        assert owns_target == stub_node_id

    # 2. Heavy GitHub webhook ingest fires later — same canonical_id,
    #    richer properties.
    async with with_tenant(CUSTOMER_ID) as conn:
        await upsert_nodes(
            conn,
            customer_id=CUSTOMER_ID,
            nodes=[
                GraphNodeSpec(
                    label=NodeLabel.DOCUMENT,
                    canonical_id="github:acme/widgets:pr:42",
                    properties={
                        "doc_type": DocType.GITHUB_PULL_REQUEST.value,
                        "title_at_ingest": "Add async webhooks",
                    },
                ),
            ],
            source_system=SourceSystem.GITHUB.value,
        )

    # 3. Same node_id, properties merged (stub keys preserved, new keys
    #    landed), edge still points at it.
    async with with_tenant(CUSTOMER_ID) as conn:
        merged_row = await conn.fetchrow(
            "SELECT node_id, properties FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Document' "
            "  AND canonical_id = 'github:acme/widgets:pr:42'",
            CUSTOMER_ID,
        )
        assert merged_row["node_id"] == stub_node_id
        merged_props = orjson.loads(merged_row["properties"])
        assert merged_props == {
            "doc_type": "github.pull_request",
            "title_at_ingest": "Add async webhooks",
        }
        owns_target = await conn.fetchval(
            "SELECT to_node_id FROM graph_edges "
            "WHERE customer_id = $1 AND edge_type = 'OWNS'",
            CUSTOMER_ID,
        )
        assert owns_target == stub_node_id

        # No duplicate Document row for the same canonical_id.
        dup = await conn.fetchval(
            "SELECT COUNT(*) FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Document' "
            "  AND canonical_id = 'github:acme/widgets:pr:42'",
            CUSTOMER_ID,
        )
        assert dup == 1


@pytest.mark.asyncio
async def test_race_winner_reuses_existing_node(client: httpx.AsyncClient) -> None:
    """Heavy GitHub ingest writes the PR Document FIRST. The endpoint's
    call reuses that node_id (no duplicate row), the OWNS edge targets
    it, and the heavy ingest's richer properties survive the stub merge
    (F7 from adversarial review).
    """
    async with raw_conn() as conn:
        await _seed_customer(conn)

    # 1. Heavy ingest first.
    async with with_tenant(CUSTOMER_ID) as conn:
        prior = await upsert_nodes(
            conn,
            customer_id=CUSTOMER_ID,
            nodes=[
                GraphNodeSpec(
                    label=NodeLabel.DOCUMENT,
                    canonical_id="github:acme/widgets:pr:42",
                    properties={
                        "doc_type": DocType.GITHUB_PULL_REQUEST.value,
                        "title_at_ingest": "Add async webhooks",
                    },
                ),
            ],
            source_system=SourceSystem.GITHUB.value,
        )
        existing_node_id = prior[("Document", "github:acme/widgets:pr:42")]

    # 2. Endpoint call.
    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(),
        json=_request_body(author_id=None),
    )
    assert resp.status_code == 200, resp.text

    async with with_tenant(CUSTOMER_ID) as conn:
        rows = await conn.fetch(
            "SELECT node_id, properties FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Document' "
            "  AND canonical_id = 'github:acme/widgets:pr:42'",
            CUSTOMER_ID,
        )
        assert len(rows) == 1
        assert rows[0]["node_id"] == existing_node_id
        # Richer properties from the prior ingest were not stripped by
        # our subsequent stub upsert.
        merged_props = orjson.loads(rows[0]["properties"])
        assert merged_props.get("title_at_ingest") == "Add async webhooks"

        owns_target = await conn.fetchval(
            "SELECT to_node_id FROM graph_edges "
            "WHERE customer_id = $1 AND edge_type = 'OWNS'",
            CUSTOMER_ID,
        )
        assert owns_target == existing_node_id


# ---------------------------------------------------------------------------
# Cross-tenant isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_tenant_customer_id_pinned(client: httpx.AsyncClient) -> None:
    """A call as customer A must only INSERT rows with customer_id=A. We can't
    directly test the RLS USING clause here (the test DB role is superuser
    locally — see `test_rls_cross_tenant_denial.py` for the dedicated RLS
    coverage), but we CAN verify the route's writes are correctly scoped to
    the X-Prbe-Customer header. All graph_nodes / graph_edges rows for B
    must remain zero after a call posted as A.
    """
    cust_a = "cust-a-feat"
    cust_b = "cust-b-feat"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust_a)
        await _seed_customer(conn, cust_b)

    resp = await client.post(
        "/internal/feature-nodes/upsert",
        headers=_headers(customer=cust_a),
        json=_request_body(),
    )
    assert resp.status_code == 200, resp.text

    async with raw_conn() as conn:
        a_count = await conn.fetchval(
            "SELECT COUNT(*) FROM graph_nodes WHERE customer_id = $1", cust_a
        )
        assert a_count > 0, "expected A's nodes to land"
        b_count = await conn.fetchval(
            "SELECT COUNT(*) FROM graph_nodes WHERE customer_id = $1", cust_b
        )
        assert b_count == 0, "no rows should have landed under B"
        b_edges = await conn.fetchval(
            "SELECT COUNT(*) FROM graph_edges WHERE customer_id = $1", cust_b
        )
        assert b_edges == 0, "no edges should have landed under B"
