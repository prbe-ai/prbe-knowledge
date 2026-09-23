"""Entity auto-merge on real Postgres, from ingestion to the merge.

    graph_writer.upsert_nodes ──► node_post_write_queue ──► PostWriteWorker
        ──► AutoMergeAnalyzer (real trigram SQL, real conflict filter)
        ──► jev_judge ──► jev.post_choice ──► [fake Jev over MockTransport]
        ──► merge_cluster (real merge txn) | entity_merge_suggestions | defer

Only two things are faked: Jev's HTTP answer (a handler that decides like
Jev would, so no key or network is needed) and the node embedding (no vector
is computed, so candidates come from the trigram leg alone -- but the stub
still takes the node's row lock exactly as the real write does, because that
lock is what once deadlocked the merge against its own caller). Every entity
is synthetic.
"""

from __future__ import annotations

import asyncio
import json
import types
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from engine.ingest.auto_merge import analyzer as az
from engine.ingest.auto_merge import jev_judge
from engine.ingest.entity_clusters_routes import MergeRequest, merge_cluster
from engine.ingest.entity_merge_suggestions_routes import ApproveAllRequest, approve_all_suggestions
from engine.ingest.graph_writer import upsert_edges, upsert_nodes
from engine.ingest.post_write import worker as worker_module
from engine.ingest.post_write.worker import PostWriteWorker
from engine.retrieval.agent import jev
from engine.shared.constants import (
    AUTO_MERGE_JEV_MODEL,
    AUTO_MERGE_MAX_DEFERRALS,
    AUTO_MERGE_RETRY_SECONDS,
    JEV_BREAKER_FAILURES,
    AutoMergeJudge,
    EdgeType,
    NodeLabel,
)
from engine.shared.db import raw_conn, with_tenant
from engine.shared.models import GraphEdgeSpec, GraphNodeSpec, make_document, make_person

CUSTOMER = "acme-automerge-test"

Decide = Callable[[dict], tuple[str, float]]


class FakeJev:
    """Answers the Choice the way Jev would, per a test's decide(state)."""

    def __init__(self, decide: Decide, status: int = 200) -> None:
        self.decide = decide
        self.status = status
        self.requests: list[dict] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": {"error_type": "overloaded"}})
        criteria = body["questions"]["match"]["criteria"]
        choice, p = self.decide(body["state"])
        rest = (1.0 - p) / max(1, len(criteria) - 1)
        probs = {k: (p if k == choice else rest) for k in criteria}
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {"match": {"type": "choice", "choice": choice, "probabilities": probs}},
            "usage": {"input_tokens": 700},
        })


def pick(new_id: str, primary_id: str, p: float) -> Decide:
    """Choose `primary_id` for `new_id` at probability p; none_of_these otherwise."""

    def decide(state: dict) -> tuple[str, float]:
        if state["new_entity"]["canonical_id"] == new_id:
            for key, cand in state["candidates"].items():
                if cand["canonical_id"] == primary_id:
                    return key, p
        return jev_judge.NONE_OF_THESE, 0.97

    return decide


@pytest.fixture
async def world(live_db, monkeypatch):
    fake = FakeJev(lambda state: (jev_judge.NONE_OF_THESE, 0.97))
    monkeypatch.setattr(az, "AUTO_MERGE_JUDGE", AutoMergeJudge.JEV)
    monkeypatch.setattr(az, "get_settings", lambda: types.SimpleNamespace(typesafe_api_key="k-test"))
    breaker = jev.Breaker()
    monkeypatch.setattr(jev_judge, "MERGE_BREAKER", breaker)
    monkeypatch.setattr(az, "MERGE_BREAKER", breaker)
    monkeypatch.setattr(jev, "_shared_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))

    async def lock_like_an_embedding_write(self, conn, node_id):
        # The real _ensure_embedding UPDATEs this row; the lock is the point.
        # The marker lets a test see whether that write survived.
        await conn.execute(
            "UPDATE graph_nodes SET properties = properties || '{\"embedded\": true}'::jsonb WHERE node_id = $1",
            node_id,
        )

    monkeypatch.setattr(PostWriteWorker, "_ensure_embedding", lock_like_an_embedding_write)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, status) "
            "VALUES ($1, $1, 'test-' || $1, 'active')",
            CUSTOMER,
        )
    return types.SimpleNamespace(
        fake=fake, breaker=breaker, worker=PostWriteWorker(concurrency=1, execute_high_confidence=True)
    )


async def ingest(*nodes: GraphNodeSpec, source: str = "github") -> None:
    async with with_tenant(CUSTOMER) as conn:
        await upsert_nodes(conn, CUSTOMER, list(nodes), source)


async def drain(worker: PostWriteWorker) -> int:
    """Process the queue until empty. Bounded: a merge that waits on a lock
    its own caller holds must fail this test in seconds, not hang it."""

    async def _run() -> int:
        n = 0
        while (row := await worker._claim_one()) is not None:
            await worker._process(row)
            n += 1
        return n

    return await asyncio.wait_for(_run(), timeout=30)


async def fetch(sql: str, *args):
    async with raw_conn() as conn:
        return await conn.fetch(sql, *args)


async def test_shared_email_person_is_merged_by_jev(world):
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    assert await drain(world.worker) == 1  # alone in the graph: no candidates, no Jev call
    assert world.fake.requests == []

    world.fake.decide = pick("ada-gh", "ada@example.com", 0.99)
    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"}))
    assert await drain(world.worker) == 1

    (req,) = world.fake.requests
    assert req["model"] == AUTO_MERGE_JEV_MODEL
    assert req["state"]["new_entity"]["canonical_id"] == "ada-gh"
    audit = await fetch("SELECT primary_canonical_id, merged_alias_canonical_ids, reason FROM entity_merge_audit")
    assert [(a["primary_canonical_id"], a["merged_alias_canonical_ids"]) for a in audit] == [
        ("ada@example.com", ["ada-gh"])
    ]
    assert audit[0]["reason"].startswith(
        "auto: model=jev-1.13.0 confidence=high p=0.99 rationale=shared email ada@example.com"
    )
    aliases = await fetch("SELECT alias_canonical_id, primary_canonical_id FROM entity_aliases")
    assert [(a["alias_canonical_id"], a["primary_canonical_id"]) for a in aliases] == [("ada-gh", "ada@example.com")]
    assert await fetch("SELECT 1 FROM node_post_write_queue") == []

    # The alias stays merged: re-ingesting it writes to the primary, so the
    # node queued for judging is the primary's, not a resurrected alias.
    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh"}))
    queued = await fetch(
        "SELECT g.canonical_id FROM node_post_write_queue q JOIN graph_nodes g USING (node_id)"
    )
    assert [r["canonical_id"] for r in queued] == ["ada@example.com"]


async def test_name_only_person_becomes_a_suggestion_not_a_merge(world):
    await ingest(make_person("U123", {"name": "Grace Hopper", "source_system": "slack"}), source="slack")
    await drain(world.worker)

    world.fake.decide = pick("grace-gh", "U123", 0.99)
    await ingest(make_person("grace-gh", {"name": "Grace Hopper", "login": "grace-gh"}))
    await drain(world.worker)

    assert await fetch("SELECT 1 FROM entity_merge_audit") == []
    rows = await fetch(
        "SELECT primary_canonical_id, candidate_canonical_id, confidence, llm_model, rationale "
        "FROM entity_merge_suggestions"
    )
    assert [(r["primary_canonical_id"], r["candidate_canonical_id"], r["confidence"], r["llm_model"]) for r in rows] == [
        ("U123", "grace-gh", "medium", "jev-1.13.0")
    ]
    assert rows[0]["rationale"].startswith('same name "Grace Hopper"')


async def seed_document(doc_id: str) -> None:
    """A real `documents` row whose doc_id is a graph node's canonical_id."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id, source_system, source_id, source_url,
                doc_class, doc_type, content_type, content_hash, title,
                body_size_bytes, body_token_count, created_at, updated_at, valid_from, ingested_at,
                acl, metadata, entities, attachments, doc_references, normalizer_version
            ) VALUES (
                $1, 1, $2, 'github', $1, 'https://example/' || $1,
                'raw_source', 'github.pull_request', 'text/markdown', 'h-' || $1, 'Add widgets',
                0, 0, NOW(), NOW(), NOW(), NOW(),
                '{}'::jsonb, '{}'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, 'v1'
            )
            """,
            doc_id,
            CUSTOMER,
        )


async def test_a_prs_document_node_is_never_folded_into_its_mention(world):
    # The PR's document node (its id is the document's doc_id) and the bare
    # `owner/repo#N` mention name the same PR. Folding the document node away
    # would cut the document out of graph retrieval, so it is never judged --
    # and never the alias: the mention folds into IT (the twin rule).
    await ingest(make_document("acme/widgets#12", properties={"name": "Add widgets"}))
    await drain(world.worker)

    world.fake.decide = pick("github:acme/widgets:pr:12", "acme/widgets#12", 0.99)
    await seed_document("github:acme/widgets:pr:12")
    await ingest(make_document("github:acme/widgets:pr:12", properties={"name": "Add widgets"}))
    await drain(world.worker)

    assert world.fake.requests == []  # skipped before the candidate search and the judge
    audit = await fetch("SELECT primary_canonical_id, merged_alias_canonical_ids FROM entity_merge_audit")
    assert [(a["primary_canonical_id"], a["merged_alias_canonical_ids"]) for a in audit] == [
        ("github:acme/widgets:pr:12", ["acme/widgets#12"])
    ]
    assert await fetch("SELECT 1 FROM entity_merge_suggestions") == []
    assert await fetch("SELECT 1 FROM graph_nodes WHERE canonical_id = 'github:acme/widgets:pr:12'") != []


async def test_a_document_stub_written_before_its_document_is_not_folded(world):
    # A writer (feature_nodes_routes) creates the PR's node, marked with its
    # doc_type, before the document row exists.
    await ingest(make_document("acme/widgets#12", properties={"name": "Add widgets"}))
    await drain(world.worker)
    world.fake.decide = pick("github:acme/widgets:pr:12", "acme/widgets#12", 0.99)
    await ingest(GraphNodeSpec(label=NodeLabel.DOCUMENT, canonical_id="github:acme/widgets:pr:12",
                               properties={"doc_type": "github.pull_request"}))
    await drain(world.worker)
    assert world.fake.requests == []
    audit = await fetch("SELECT primary_canonical_id FROM entity_merge_audit")
    assert [a["primary_canonical_id"] for a in audit] == ["github:acme/widgets:pr:12"]  # the stub is the primary


async def test_approving_an_old_suggestion_never_folds_a_document(world):
    # Suggestions written before the guard can still name a document's node
    # as the alias; approve-all must refuse (and dismiss) them.
    await ingest(make_document("acme/widgets#12", properties={"name": "Add widgets"}),
                 make_document("github:acme/widgets:pr:12", properties={"name": "Add widgets"}))
    await seed_document("github:acme/widgets:pr:12")
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO entity_merge_suggestions (customer_id, label, primary_canonical_id, "
            "candidate_canonical_id, confidence, rationale, llm_model, status) "
            "VALUES ($1, 'Document', 'acme/widgets#12', 'github:acme/widgets:pr:12', 'high', 'x', 'm', 'pending')",
            CUSTOMER,
        )
    out = await approve_all_suggestions(ApproveAllRequest(customer_id=CUSTOMER, confidence="high"))
    assert (out.approved, out.dismissed_already_merged) == (0, 1)
    assert await fetch("SELECT 1 FROM entity_merge_audit") == []
    assert await fetch("SELECT 1 FROM graph_nodes WHERE canonical_id = 'github:acme/widgets:pr:12'") != []


async def test_a_non_document_repo_node_merges_into_its_owner_form(world):
    await ingest(make_document("acme/acme-widgets", properties={"name": "acme-widgets"}))
    await drain(world.worker)

    world.fake.decide = pick("wiki:repo:acme_widgets", "acme/acme-widgets", 0.99)
    await ingest(make_document("wiki:repo:acme_widgets", properties={"name": "acme_widgets"}))
    await drain(world.worker)

    audit = await fetch("SELECT primary_canonical_id, merged_alias_canonical_ids, reason FROM entity_merge_audit")
    assert [(a["primary_canonical_id"], a["merged_alias_canonical_ids"]) for a in audit] == [
        ("acme/acme-widgets", ["wiki:repo:acme_widgets"])
    ]
    assert "rationale=same repo name once owner/wiki prefix" in audit[0]["reason"]


async def status_of(canonical_id: str) -> dict:
    (row,) = await fetch(
        "SELECT q.analyzer_status FROM node_post_write_queue q JOIN graph_nodes g USING (node_id) "
        "WHERE g.canonical_id = $1",
        canonical_id,
    )
    return json.loads(row["analyzer_status"])["auto_merge"]


async def expire_locks() -> None:
    async with raw_conn() as conn:
        await conn.execute("UPDATE node_post_write_queue SET locked_until = NOW() - interval '1 second'")


async def test_jev_outage_keeps_the_node_queued_and_retries_it(world):
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    await drain(world.worker)

    world.fake.status = 503
    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"}))
    await drain(world.worker)

    status = await status_of("ada-gh")
    # Deferred, and the outage did not spend one of the node's 3 attempts.
    assert (status["status"], status["deferrals"], status["attempts"]) == ("deferred", 1, 0)
    (row,) = await fetch("SELECT locked_until > NOW() + make_interval(secs => $1) AS delayed FROM node_post_write_queue",
                         float(AUTO_MERGE_RETRY_SECONDS - 60))
    assert row["delayed"]
    assert await world.worker._claim_one() is None  # not claimable before the delay
    assert await fetch("SELECT 1 FROM entity_merge_audit") == []

    # A second failure doubles the delay; still no attempt spent.
    await expire_locks()
    await drain(world.worker)
    status = await status_of("ada-gh")
    assert (status["deferrals"], status["attempts"]) == (2, 0)
    (row,) = await fetch("SELECT locked_until > NOW() + make_interval(secs => $1) AS delayed FROM node_post_write_queue",
                         float(2 * AUTO_MERGE_RETRY_SECONDS - 60))
    assert row["delayed"]

    # Time passes, Jev recovers: the same row is reclaimed and merged.
    await expire_locks()
    world.fake.status = 200
    world.fake.decide = pick("ada-gh", "ada@example.com", 0.99)
    assert await drain(world.worker) == 1
    assert len(await fetch("SELECT 1 FROM entity_merge_audit")) == 1
    assert await fetch("SELECT 1 FROM node_post_write_queue") == []


async def test_a_judge_that_never_recovers_parks_the_node_after_the_cap(world):
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    await drain(world.worker)
    world.fake.status = 503
    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"}))
    # Fast-forward to the last allowed deferral, then fail once more.
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE node_post_write_queue SET analyzer_status = $1::jsonb",
            json.dumps({"auto_merge": {"status": "deferred", "attempts": 0, "deferrals": AUTO_MERGE_MAX_DEFERRALS,
                                       "failed_calls": AUTO_MERGE_MAX_DEFERRALS}}),
        )
    await drain(world.worker)
    status = await status_of("ada-gh")
    assert (status["status"], status["attempts"]) == ("failed", worker_module._MAX_ATTEMPTS)
    await expire_locks()
    assert await world.worker._claim_one() is None  # parked: visible, not retried


async def test_open_breaker_defers_without_calling_jev_or_spending_an_attempt(world):
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    await drain(world.worker)
    for _ in range(JEV_BREAKER_FAILURES):
        world.breaker.failure()

    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"}))
    await drain(world.worker)

    assert world.fake.requests == []  # nothing was sent
    status = await status_of("ada-gh")
    assert (status["status"], status["attempts"], status["failed_calls"]) == ("deferred", 0, 0)


async def test_a_long_outage_with_the_breaker_open_never_parks_the_node(world):
    # Far past the cap in deferrals, but none of them sent anything.
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    await drain(world.worker)
    for _ in range(JEV_BREAKER_FAILURES):
        world.breaker.failure()
    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"}))
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE node_post_write_queue SET analyzer_status = $1::jsonb",
            json.dumps({"auto_merge": {"status": "deferred", "attempts": 0,
                                       "deferrals": 3 * AUTO_MERGE_MAX_DEFERRALS, "failed_calls": 0}}),
        )
    await drain(world.worker)
    status = await status_of("ada-gh")
    assert (status["status"], status["failed_calls"]) == ("deferred", 0)


async def test_a_failed_pending_edge_drain_keeps_the_embedding_and_still_judges(world, monkeypatch):
    async def failing_drain(conn, *args, **kwargs):
        await conn.execute("SELECT 1 / 0")  # aborts whatever transaction it runs in

    monkeypatch.setattr(worker_module, "drain_pending_edges", failing_drain)
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    await drain(world.worker)

    world.fake.decide = pick("ada-gh", "ada@example.com", 0.99)
    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"}))
    await drain(world.worker)

    marked = await fetch(
        "SELECT canonical_id FROM graph_nodes WHERE customer_id = $1 AND properties ? 'embedded'", CUSTOMER
    )
    # Both nodes' "embedding" writes survived the failed drain...
    assert {r["canonical_id"] for r in marked} >= {"ada@example.com"}
    # ...and the judgment still ran to a merge.
    assert len(await fetch("SELECT 1 FROM entity_merge_audit")) == 1


async def test_a_reupserted_cluster_primary_is_not_folded_into_another_node(world):
    # ada@example.com becomes a cluster primary (alias ada-gh).
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    await drain(world.worker)
    world.fake.decide = pick("ada-gh", "ada@example.com", 0.99)
    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"}))
    await drain(world.worker)
    # A third copy arrives and Jev leaves it alone.
    world.fake.decide = lambda state: (jev_judge.NONE_OF_THESE, 0.97)
    await ingest(make_person("U999", {"name": "Ada Lovelace", "email": "ada@example.com"}), source="slack")
    await drain(world.worker)

    # The primary is re-upserted and judged a duplicate of the third copy.
    # Folding it in would delete its node while ada-gh still routes to it.
    world.fake.decide = pick("ada@example.com", "U999", 0.99)
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    await drain(world.worker)

    audit = await fetch("SELECT merged_alias_canonical_ids FROM entity_merge_audit")
    assert [a["merged_alias_canonical_ids"] for a in audit] == [["ada-gh"]]
    aliases = await fetch("SELECT alias_canonical_id, primary_canonical_id FROM entity_aliases")
    assert [(a["alias_canonical_id"], a["primary_canonical_id"]) for a in aliases] == [("ada-gh", "ada@example.com")]
    assert await fetch("SELECT 1 FROM graph_nodes WHERE canonical_id = 'ada@example.com'") != []
    # No suggestion either: approving one would build the very chain the
    # refusal stops (the approve routes do not set the flag).
    assert await fetch("SELECT 1 FROM entity_merge_suggestions") == []


async def test_twins_merged_into_each_other_at_once_leave_exactly_one_node(world):
    # Two workers can judge a PR's twin ids at the same moment and each pick
    # the other. Without the per-(tenant, label) lock both merges pass their
    # checks and the loser dies mid-merge on a foreign key (seen when the lock
    # was removed); with it the loser finds its node gone and 404s cleanly.
    pairs = [(f"acme/widgets#{n}", f"github:acme/widgets:pr:{n}") for n in range(20, 26)]
    for a, b in pairs:
        await ingest(make_document(a, properties={"name": f"PR {a}"}), make_document(b, properties={"name": f"PR {b}"}))

    def req(primary: str, alias: str) -> MergeRequest:
        return MergeRequest(customer_id=CUSTOMER, performed_by_user_id=az.SYSTEM_USER_ID, label="Document",
                            primary_canonical_id=primary, alias_canonical_ids=[alias], refuse_cluster_primaries=True)

    for a, b in pairs:
        results = await asyncio.gather(merge_cluster(req(a, b)), merge_cluster(req(b, a)), return_exceptions=True)
        ok = [r for r in results if not isinstance(r, BaseException)]
        failed = [r for r in results if isinstance(r, BaseException)]
        assert len(ok) == 1, results
        assert [getattr(f, "status_code", None) for f in failed] == [404], failed
        survivors = await fetch("SELECT canonical_id FROM graph_nodes WHERE canonical_id = ANY($1::text[])", [a, b])
        assert [r["canonical_id"] for r in survivors] == [ok[0].primary_canonical_id]


async def test_a_node_reupserted_mid_processing_stays_queued_for_another_pass(world):
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    row = await world.worker._claim_one()
    # The node changes while this pass is judging it: the upsert re-enqueues it.
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com", "title": "CTO"}))
    await world.worker._process(row)
    # The stale pass neither deleted the fresh row nor stamped it.
    (queued,) = await fetch("SELECT analyzer_status, locked_until FROM node_post_write_queue")
    assert (json.loads(queued["analyzer_status"]), queued["locked_until"]) == ({}, None)
    assert await drain(world.worker) == 1
    assert await fetch("SELECT 1 FROM node_post_write_queue") == []



async def test_a_prs_document_folds_its_mention_into_itself(world):
    # The connector writes the mention and the document together; the
    # document's node absorbs the mention (and every edge pointing at it).
    world.fake.decide = lambda state: (_ for _ in ()).throw(AssertionError("no judge for a twin"))
    await ingest(make_document("acme/widgets#12", properties={"name": "Add widgets"}))
    await drain(world.worker)
    await seed_document("github:acme/widgets:pr:12")
    await ingest(make_document("github:acme/widgets:pr:12", properties={"doc_type": "github.pull_request"}))
    await drain(world.worker)

    assert world.fake.requests == []
    audit = await fetch("SELECT primary_canonical_id, merged_alias_canonical_ids, reason FROM entity_merge_audit")
    assert [(a["primary_canonical_id"], a["merged_alias_canonical_ids"]) for a in audit] == [
        ("github:acme/widgets:pr:12", ["acme/widgets#12"])
    ]
    assert audit[0]["reason"].startswith("auto: rule:document-twin confidence=high")
    # Later writes of the mention now land on the document's node.
    await ingest(make_document("acme/widgets#12", properties={"name": "Add widgets"}))
    assert await fetch("SELECT 1 FROM graph_nodes WHERE canonical_id = 'acme/widgets#12'") == []


async def test_a_mention_written_after_its_document_folds_into_it(world):
    await seed_document("github:acme/widgets:issue:3")
    await ingest(make_document("github:acme/widgets:issue:3", properties={"doc_type": "github.issue"}))
    await drain(world.worker)
    await ingest(make_document("acme/widgets#3", properties={"name": "A bug"}))
    await drain(world.worker)
    audit = await fetch("SELECT primary_canonical_id, merged_alias_canonical_ids FROM entity_merge_audit")
    assert [(a["primary_canonical_id"], a["merged_alias_canonical_ids"]) for a in audit] == [
        ("github:acme/widgets:issue:3", ["acme/widgets#3"])
    ]


async def test_a_mention_that_heads_a_cluster_is_not_folded(world):
    # The old judge folded documents INTO mentions, so a mention can head a
    # cluster; folding it would strand that cluster (the repair script
    # unmerges first).
    await ingest(make_document("acme/widgets#12", properties={"name": "Add widgets"}),
                 make_document("acme/widgets#12-old", properties={"name": "Add widgets"}))
    async with raw_conn() as conn:
        await conn.execute("UPDATE graph_nodes SET properties = properties || '{}'::jsonb")
    await merge_cluster(MergeRequest(customer_id=CUSTOMER, performed_by_user_id=az.SYSTEM_USER_ID, label="Document",
                                     primary_canonical_id="acme/widgets#12", alias_canonical_ids=["acme/widgets#12-old"]))
    await seed_document("github:acme/widgets:pr:12")
    await ingest(make_document("github:acme/widgets:pr:12", properties={"doc_type": "github.pull_request"}))
    await drain(world.worker)
    audit = await fetch("SELECT merged_alias_canonical_ids FROM entity_merge_audit")
    assert [a["merged_alias_canonical_ids"] for a in audit] == [["acme/widgets#12-old"]]


async def test_a_stray_alias_node_is_not_judged(world):
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    await drain(world.worker)
    world.fake.decide = pick("ada-gh", "ada@example.com", 0.99)
    await ingest(make_person("ada-gh", {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"}))
    await drain(world.worker)
    # A copy of the alias appears anyway (e.g. written by an older route).
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO graph_nodes (customer_id, label, canonical_id, properties) VALUES ($1, 'Person', 'ada-gh', '{}')",
            CUSTOMER,
        )
        node_id = await conn.fetchval("SELECT node_id FROM graph_nodes WHERE canonical_id = 'ada-gh'")
        await conn.execute("INSERT INTO node_post_write_queue (customer_id, node_id, analyzer_status) VALUES ($1, $2, '{}')",
                           CUSTOMER, node_id)
    world.fake.requests.clear()
    await drain(world.worker)
    assert world.fake.requests == []
    assert len(await fetch("SELECT 1 FROM entity_merge_audit")) == 1


async def test_edge_linking_only_mode_links_a_parked_edge_and_judges_nothing(world):
    # What the research plane runs: auto-merge and node embeddings off.
    worker = PostWriteWorker(concurrency=1, execute_high_confidence=False,
                             auto_merge_enabled=False, embeddings_enabled=False)
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}))
    async with with_tenant(CUSTOMER) as conn:
        ids = {("Person", "ada@example.com"): await conn.fetchval(
            "SELECT node_id FROM graph_nodes WHERE canonical_id = 'ada@example.com'")}
        await upsert_edges(conn, CUSTOMER, [GraphEdgeSpec(
            edge_type=EdgeType.AUTHORED, from_label=NodeLabel.PERSON, from_canonical_id="ada@example.com",
            to_label=NodeLabel.DOCUMENT, to_canonical_id="github:acme/widgets:pr:12")], ids, "github")
    assert len(await fetch("SELECT 1 FROM pending_edges")) == 1

    await ingest(make_document("github:acme/widgets:pr:12", properties={"doc_type": "github.pull_request"}))
    await drain(worker)
    assert await fetch("SELECT 1 FROM pending_edges") == []
    assert len(await fetch("SELECT 1 FROM graph_edges WHERE edge_type = 'AUTHORED'")) == 1
    assert world.fake.requests == []
    assert await fetch("SELECT 1 FROM graph_nodes WHERE properties ? 'embedded'") == []
    assert await fetch("SELECT 1 FROM node_post_write_queue") == []


async def test_a_claim_takes_fresh_rows_first_and_due_rows_when_none_are_left(world):
    await ingest(make_person("ada@example.com", {"name": "Ada Lovelace"}), make_person("bob@example.com", {"name": "Bob"}))
    async with raw_conn() as conn:
        # ada's row is a deferral whose delay has passed; bob's is fresh.
        await conn.execute(
            "UPDATE node_post_write_queue q SET locked_until = NOW() - interval '1 minute' "
            "FROM graph_nodes g WHERE g.node_id = q.node_id AND g.canonical_id = 'ada@example.com'"
        )
    first = await world.worker._claim_one()
    second = await world.worker._claim_one()
    names = [(await fetch("SELECT canonical_id FROM graph_nodes WHERE node_id = $1", r["node_id"]))[0]["canonical_id"]
             for r in (first, second)]
    assert names == ["bob@example.com", "ada@example.com"]
    assert await world.worker._claim_one() is None


async def test_a_long_fresh_backlog_cannot_starve_a_due_retry(world):
    await ingest(*[make_person(f"p{i}@example.com", {"name": f"P{i}"}) for i in range(worker_module._DUE_EVERY + 2)])
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE node_post_write_queue q SET locked_until = NOW() - interval '1 minute' "
            "FROM graph_nodes g WHERE g.node_id = q.node_id AND g.canonical_id = 'p0@example.com'"
        )
    claimed = []
    for _ in range(worker_module._DUE_EVERY):
        row = await world.worker._claim_one()
        claimed.append((await fetch("SELECT canonical_id FROM graph_nodes WHERE node_id = $1", row["node_id"]))[0]["canonical_id"])
    assert "p0@example.com" in claimed  # reached by the every-Nth due-first claim


async def test_the_pending_edge_sweep_runs_once_per_tenant_per_interval(world, monkeypatch):
    calls = []

    async def counting_reap(conn, customer_id):
        calls.append(customer_id)
        return 0

    monkeypatch.setattr(worker_module, "reap_expired_pending_edges", counting_reap)
    monkeypatch.setattr(worker_module, "_last_reap", {})
    await ingest(*[make_person(f"p{i}@example.com", {"name": f"P{i}"}) for i in range(5)])
    await drain(world.worker)
    assert calls == [CUSTOMER]


async def test_the_sweep_leaves_leased_pending_edges_alone(world):
    from engine.ingest.graph_writer import reap_expired_pending_edges

    async with raw_conn() as conn:
        for leased in (False, True):
            await conn.execute(
                "INSERT INTO pending_edges (customer_id, missing_label, missing_canonical_id, edge_type, from_label, "
                "from_canonical_id, to_label, to_canonical_id, source_system, created_at, locked_until) "
                "VALUES ($1, 'Document', 'x', 'AUTHORED', 'Person', 'a', 'Document', 'x', 'github', "
                "NOW() - interval '30 days', $2)",
                CUSTOMER, datetime.now(UTC) + timedelta(seconds=30) if leased else None,
            )
    async with with_tenant(CUSTOMER) as conn:
        assert await reap_expired_pending_edges(conn, CUSTOMER) == 1
    assert len(await fetch("SELECT 1 FROM pending_edges")) == 1
