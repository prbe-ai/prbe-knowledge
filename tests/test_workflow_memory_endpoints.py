"""The HTTP surface: gating, the two-call declaration flow, and what leaks.

WHAT THIS FILE HAS TO DEFEND:

* THE GATE IS REAL AND FAILS CLOSED. A tenant without the capability cell gets
  nothing from any of the three endpoints, and the modules underneath are never
  reached -- an endpoint that gated the RESPONSE but still ran the LLM and the
  write would be off in the dashboard and on in the bill.
* OFF IS NOT 403. Every response carries `enabled` / `entitled` / `upgrade_url`,
  because an HTTP status cannot spell the difference between "nobody turned this
  on" and "your plan does not include it", and a client that cannot tell them
  apart reports the switch as broken.
* PREVIEW WRITES NOTHING. It is the half of the declaration flow that exists so
  a human can look; if it wrote, the confirmation step would be decoration.
* A CREDENTIAL NEVER COMES BACK OUT. The refusal names the DETECTOR, never the
  match. An error body is the last place a leaked secret should be re-echoed --
  it lands in a response, a log line, and somebody's terminal scrollback.
* "NO SITUATIONS CONFIGURED" IS ITS OWN ANSWER. Otherwise a tenant whose feature
  was switched on but never seeded is indistinguishable on the wire from one
  whose rules simply did not match, and nobody ever finds out.

Run with the isolated wfmem database (these fixtures TRUNCATE):

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:5432/prbe_knowledge_wfmem \
        .venv/bin/pytest tests/test_workflow_memory_endpoints.py -q
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from engine.shared.db import close_pool, raw_conn, with_tenant
from engine.shared.wfmem.capabilities import (
    InputPath,
    OutputSurface,
    input_capability_key,
    output_capability_key,
)
from engine.shared.wfmem.situations import enable_capability
from engine.shared.wfmem.structuring import ClauseDraft

TENANT = "cust-wfmem-http"
ACTOR = "user:alice"
OTHER_ACTOR = "user:bob"

INTERNAL_KEY = "test-internal-key"
HEADERS = {"X-Internal-Knowledge-Key": INTERNAL_KEY, "X-Prbe-Customer": TENANT}

DECLARED_KEY = input_capability_key(InputPath.DECLARED)
RETRIEVAL_KEY = output_capability_key(OutputSurface.RETRIEVAL)

DRAFT = {
    "kind": "step",
    "body": "open a Probe run before the first GPU step",
    "semantic_action": None,
    "binding": {},
    "scope": {},
}


@dataclass
class _Chunk:
    chunk_index: int
    embedding: list[float]


@dataclass
class _EmbedResult:
    embedded: list[_Chunk]
    failed: list[Any]


class _StubEmbedder:
    """Deterministic orthogonal-ish vectors so nothing here depends on a model."""

    model_id = "stub-embedder"

    async def embed_many(self, texts: list[str]) -> Any:
        chunks = [_Chunk(i, [float(len(t) % 7), 1.0]) for i, t in enumerate(texts)]
        return _EmbedResult(embedded=chunks, failed=[])

    async def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0]


@pytest_asyncio.fixture(autouse=True)
async def _internal_key(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Point the auth dependency at a known internal key.

    `get_settings` is cached, so the value is patched on the resolved settings
    object rather than through the environment -- an env var set after the first
    call would be silently ignored and every request would 503.
    """
    from pydantic import SecretStr

    from engine.shared.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "internal_knowledge_api_key", SecretStr(INTERNAL_KEY))
    yield


@pytest_asyncio.fixture
async def tenant(live_db: None) -> AsyncIterator[str]:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'wfmem-http', 'h-wfmem-http')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            TENANT,
        )
    yield TENANT


@pytest_asyncio.fixture
async def enabled(tenant: str) -> AsyncIterator[str]:
    """Both capabilities on. Enabling the declared input also seeds situations."""
    await enable_capability(TENANT, DECLARED_KEY)
    await enable_capability(TENANT, RETRIEVAL_KEY)
    yield TENANT


async def _post(path: str, body: dict[str, Any], *, headers: dict[str, str] | None = None) -> Any:
    from engine.retrieval.main import app

    await close_pool()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        # `is None`, NOT `or`: an EMPTY dict is how a caller says "send no auth
        # headers", and `headers or HEADERS` silently substitutes the good ones
        # for it -- which made the unauthenticated test pass a fully
        # authenticated request and assert 401 against a 200.
        return await client.post(
            path, json=body, headers=HEADERS if headers is None else headers
        )


def _patch_models(monkeypatch: pytest.MonkeyPatch, *, draft: ClauseDraft | None = None) -> None:
    """Replace the two model calls and the embedder.

    conftest blanks the provider keys, so an unpatched test would either hit a
    real provider or fall through to the hash-vector stub -- one is a bill, the
    other is noise dressed as a result.
    """
    import engine.retrieval.procedures as mod

    async def fake_structure(prose: str, context: Any, **kwargs: Any) -> ClauseDraft:
        return draft or ClauseDraft(
            kind="step",
            body="open a Probe run before the first GPU step",
            semantic_action=None,
            binding={},
            scope={},
        )

    monkeypatch.setattr(mod, "structure", fake_structure)
    monkeypatch.setattr(
        "engine.shared.wfmem.declaring.get_embedder_v2", lambda: _StubEmbedder()
    )
    monkeypatch.setattr(
        "engine.shared.wfmem.classifier.get_embedder_v2", lambda: _StubEmbedder()
    )


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/procedures/preview", {"prose": "always log runs"}),
        ("/procedures/declare", {"draft": DRAFT, "actor_ref": ACTOR}),
        ("/procedures/query", {"actor_ref": ACTOR}),
    ],
)
async def test_a_tenant_without_the_capability_gets_nothing(
    tenant: str, path: str, body: dict[str, Any]
) -> None:
    response = await _post(path, body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["capability"]["enabled"] is False
    assert payload.get("clause_id") is None
    assert payload.get("draft") is None
    assert payload.get("clauses", []) == []


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/procedures/preview", {"prose": "always log runs"}),
        ("/procedures/declare", {"draft": DRAFT, "actor_ref": ACTOR}),
        ("/procedures/query", {"actor_ref": ACTOR}),
    ],
)
async def test_off_is_two_hundred_with_an_envelope_not_a_403(
    tenant: str, path: str, body: dict[str, Any]
) -> None:
    """`enabled` alone cannot spell three states, and a status code spells fewer.

    A client that cannot distinguish "nobody turned it on" from "the plan does
    not include it" renders a toggle, flips it, sees nothing happen, and files
    "the feature is broken" rather than "we are not paying for this".
    """
    payload = (await _post(path, body)).json()
    assert payload["capability"] == {"enabled": False, "entitled": True, "upgrade_url": None}


async def test_a_disabled_tenant_never_reaches_the_model(
    tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gating the RESPONSE but not the WORK would be off in the dashboard and on
    in the bill."""
    import engine.retrieval.procedures as mod

    called = False

    async def exploding_structure(*args: Any, **kwargs: Any) -> ClauseDraft:
        nonlocal called
        called = True
        raise AssertionError("the structuring pass ran for a disabled tenant")

    monkeypatch.setattr(mod, "structure", exploding_structure)
    await _post("/procedures/preview", {"prose": "always log runs"})
    assert called is False


async def test_each_capability_gates_its_own_surface(tenant: str) -> None:
    """The input and output cells are independent. A tenant may want to record
    rules without serving them back yet, and the reverse during a migration."""
    await enable_capability(TENANT, DECLARED_KEY)

    write_side = (await _post("/procedures/declare", {"draft": DRAFT, "actor_ref": ACTOR})).json()
    read_side = (await _post("/procedures/query", {"actor_ref": ACTOR})).json()

    assert write_side["capability"]["enabled"] is True
    assert read_side["capability"]["enabled"] is False


# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------


async def test_preview_returns_a_draft_and_writes_nothing(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models(monkeypatch)

    payload = (await _post("/procedures/preview", {"prose": "always open a run"})).json()

    assert payload["draft"]["body"] == "open a Probe run before the first GPU step"
    assert payload["draft"]["kind"] == "step"
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval("SELECT count(*) FROM clauses") == 0
        assert await conn.fetchval("SELECT count(*) FROM clause_evidence") == 0


async def test_preview_surfaces_an_existing_neighbour(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models(monkeypatch)
    await _post(
        "/procedures/declare",
        {"draft": DRAFT, "actor_ref": ACTOR, "source_ref": {"session": "s1"}},
    )

    payload = (await _post("/procedures/preview", {"prose": "always open a run"})).json()
    bodies = [n["body"] for n in payload["neighbours"]]
    assert DRAFT["body"] in bodies


# --------------------------------------------------------------------------
# Declare
# --------------------------------------------------------------------------


async def test_declare_writes_the_confirmed_draft(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models(monkeypatch)
    payload = (
        await _post(
            "/procedures/declare",
            {"draft": DRAFT, "actor_ref": ACTOR, "source_ref": {"session": "s1"}},
        )
    ).json()

    assert payload["created"] is True
    assert payload["clause_id"] is not None
    async with with_tenant(TENANT) as conn:
        body = await conn.fetchval("SELECT body FROM clauses WHERE id = $1", payload["clause_id"])
    assert body == DRAFT["body"]


async def test_an_edited_draft_is_what_gets_stored(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirmation must accept CORRECTIONS, not just approval.

    `clauses.body` is authoritative for the reclassifier precisely because a
    human signed off on it; a flow that could only accept-or-reject would push
    people to approve a near-miss rather than retype the whole rule.
    """
    _patch_models(monkeypatch)
    edited = {**DRAFT, "body": "open a Probe run before the first GPU step, always"}

    payload = (
        await _post("/procedures/declare", {"draft": edited, "actor_ref": ACTOR})
    ).json()

    async with with_tenant(TENANT) as conn:
        body = await conn.fetchval("SELECT body FROM clauses WHERE id = $1", payload["clause_id"])
    assert body == edited["body"]


async def test_a_credential_is_refused_without_echoing_it(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal names the DETECTOR, never the match.

    An error body is the last place a leaked secret should reappear: it lands in
    a response, a log line, and somebody's terminal scrollback, which is three
    more places than the one it was already in.
    """
    _patch_models(monkeypatch)
    secret = "ghp_" + "a" * 36
    bad = {**DRAFT, "body": f"deploy with {secret}"}

    response = await _post("/procedures/declare", {"draft": bad, "actor_ref": ACTOR})

    assert response.status_code == 422
    assert secret not in response.text
    assert "credential" in response.text
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval("SELECT count(*) FROM clauses") == 0


async def test_a_merge_reports_that_it_did_not_create(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`created: false` is the one outcome the author most needs to see."""
    _patch_models(monkeypatch)
    first = (await _post("/procedures/declare", {"draft": DRAFT, "actor_ref": ACTOR})).json()

    merged = (
        await _post(
            "/procedures/declare",
            {
                "draft": DRAFT,
                "actor_ref": OTHER_ACTOR,
                "relation": "merge",
                "related_clause_id": first["clause_id"],
            },
        )
    ).json()

    assert merged["created"] is False
    assert merged["clause_id"] == first["clause_id"]


async def test_a_relation_without_a_counterpart_is_a_422(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models(monkeypatch)
    response = await _post(
        "/procedures/declare",
        {"draft": DRAFT, "actor_ref": ACTOR, "relation": "conflict"},
    )
    assert response.status_code == 422


async def test_a_refusal_is_reported_rather_than_raised(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merge into a clause that does not exist is the author's mistake, not a
    server fault, and the client needs the sentence to show them."""
    _patch_models(monkeypatch)
    response = await _post(
        "/procedures/declare",
        {
            "draft": DRAFT,
            "actor_ref": ACTOR,
            "relation": "merge",
            "related_clause_id": "11111111-1111-1111-1111-111111111111",
        },
    )
    assert response.status_code == 200
    assert "does not exist" in response.json()["refused"]
    assert response.json()["clause_id"] is None


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


async def test_query_serves_a_declared_rule_to_another_actor(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop, end to end over HTTP: declared by one person, served to another.

    Two declarations because the visibility guard holds a single-author rule
    private until a second distinct human appears -- which is the behaviour, not
    an inconvenience.
    """
    _patch_models(monkeypatch)
    first = (await _post("/procedures/declare", {"draft": DRAFT, "actor_ref": ACTOR})).json()
    await _post(
        "/procedures/declare",
        {
            "draft": DRAFT,
            "actor_ref": OTHER_ACTOR,
            "relation": "merge",
            "related_clause_id": first["clause_id"],
        },
    )

    payload = (await _post("/procedures/query", {"actor_ref": "user:carol"})).json()

    assert [c["body"] for c in payload["clauses"]] == [DRAFT["body"]]
    assert payload["serve_id"] is not None


async def test_a_served_response_writes_a_ledger_row(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models(monkeypatch)
    first = (await _post("/procedures/declare", {"draft": DRAFT, "actor_ref": ACTOR})).json()
    await _post(
        "/procedures/declare",
        {
            "draft": DRAFT,
            "actor_ref": OTHER_ACTOR,
            "relation": "merge",
            "related_clause_id": first["clause_id"],
        },
    )

    payload = (
        await _post("/procedures/query", {"actor_ref": "user:carol", "session_id": "sess-7"})
    ).json()

    async with with_tenant(TENANT) as conn:
        row = await conn.fetchrow(
            "SELECT * FROM serve_ledger WHERE id = $1", payload["serve_id"]
        )
    assert row["actor_ref"] == "user:carol"
    assert row["session_id"] == "sess-7"


async def test_an_unmatched_query_serves_nothing(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`unknown` serves zero clauses rather than falling back to everything.

    Serving a broadly-scoped rule into a situation nobody could identify is
    exactly what the escape hatch exists to prevent -- and with a workspace-wide
    scope default, the situation is doing all of the narrowing.
    """
    _patch_models(monkeypatch)
    import engine.retrieval.procedures as mod
    from engine.shared.wfmem.classifier import Classification, Outcome

    async def unknown(*args: Any, **kwargs: Any) -> Classification:
        return Classification(
            outcome=Outcome.UNKNOWN,
            situation_id=None,
            slug=None,
            confidence=0.0,
            method="embedding",
            model="stub",
            prompt_version=None,
            runner_up=None,
        )

    monkeypatch.setattr(mod, "classify", unknown)
    payload = (
        await _post("/procedures/query", {"actor_ref": ACTOR, "query": "something vague"})
    ).json()

    assert payload["clauses"] == []
    assert payload["serve_id"] is None
    assert payload["no_situations_configured"] is False


async def test_no_vocabulary_is_reported_distinctly_from_no_match(tenant: str) -> None:
    """The whole reason the classifier carries three outcomes.

    A tenant switched on by a dashboard PATCH or a hand-run UPDATE never gets
    seeded, classifies everything as unknown, and serves zero cards forever. On
    the wire that is identical to "your rules did not match" unless somebody
    says otherwise, which is how a dead feature goes unreported for weeks.
    """
    # Enabling the RETRIEVAL cell alone skips the seeding that rides on the
    # DECLARED cell -- which is precisely the reachable broken state.
    await enable_capability(TENANT, RETRIEVAL_KEY)
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval("SELECT count(*) FROM situations") == 0

    payload = (
        await _post("/procedures/query", {"actor_ref": ACTOR, "query": "launching a run"})
    ).json()

    assert payload["no_situations_configured"] is True
    assert payload["clauses"] == []
    assert payload["classification"]["outcome"] == "no_vocabulary"


async def test_an_unknown_situation_slug_serves_nothing(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models(monkeypatch)
    payload = (
        await _post(
            "/procedures/query", {"actor_ref": ACTOR, "situation_slug": "not-a-situation"}
        )
    ).json()
    assert payload["clauses"] == []


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


async def test_an_unauthenticated_request_is_refused(tenant: str) -> None:
    response = await _post("/procedures/query", {"actor_ref": ACTOR}, headers={})
    assert response.status_code == 401


async def test_a_wrong_internal_key_is_refused(tenant: str) -> None:
    response = await _post(
        "/procedures/query",
        {"actor_ref": ACTOR},
        headers={"X-Internal-Knowledge-Key": "wrong", "X-Prbe-Customer": TENANT},
    )
    assert response.status_code == 401


async def test_the_customer_header_scopes_the_request(
    enabled: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scope comes from the header, never the body.

    The internal key grants access; the customer header decides whose data. A
    request carrying a body field that could override it would turn one leaked
    internal key into every tenant's store.
    """
    _patch_models(monkeypatch)
    await _post("/procedures/declare", {"draft": DRAFT, "actor_ref": ACTOR})

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ('cust-wfmem-http-other', 'other', 'h-other')
            ON CONFLICT (customer_id) DO NOTHING
            """
        )
    await enable_capability("cust-wfmem-http-other", RETRIEVAL_KEY)

    payload = (
        await _post(
            "/procedures/query",
            {"actor_ref": ACTOR},
            headers={
                "X-Internal-Knowledge-Key": INTERNAL_KEY,
                "X-Prbe-Customer": "cust-wfmem-http-other",
            },
        )
    ).json()
    assert payload["clauses"] == []


def test_the_request_models_do_not_accept_a_customer_override() -> None:
    """A body field named like the tenant must not exist on any request model.

    Belt-and-braces against the most valuable single bug in this surface: pydantic
    ignores unknown fields by default, so this asserts the field is absent from the
    schema rather than trusting that nothing reads it.
    """
    from engine.retrieval.procedures import DeclareRequest, PreviewRequest, QueryRequest

    for model in (PreviewRequest, DeclareRequest, QueryRequest):
        fields = set(model.model_fields)
        assert not fields & {"customer_id", "customer", "tenant", "tenant_id"}, (
            f"{model.__name__} exposes a tenant field: {fields}"
        )


def test_the_declare_response_never_carries_clause_bodies() -> None:
    """The write path echoes ids, not content.

    Nothing forces this today, but a `body` on the response is how a future
    convenience turns the write endpoint into a second serving surface -- one
    that does not apply the visibility predicate and writes no ledger row.
    """
    from engine.retrieval.procedures import DeclareResponse

    assert "body" not in DeclareResponse.model_fields
