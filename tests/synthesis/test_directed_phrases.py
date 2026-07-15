"""Unit + live-DB tests for services/synthesis/directed_phrases.py.

Covers:
  - parse_directed_frontmatter: empty / missing / malformed / valid.
  - persist_directed_vectors:
      * human-pin add: row appears with source='human', NULL run_id.
      * human-pin remove: row deletes when frontmatter no longer lists it.
      * llm regen: rows from older synthesis_run_id are deleted, new
        rows persist with current run_id.
      * dedupe: an LLM phrase matching a human pin (under stub-embedder
        identity) is suppressed.
      * partial state on LLM failure: human pins still reconcile; the
        result.llm_failed flag flips; no exception bubbles up.

The Anthropic LLM is mocked to avoid hitting a real API.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from engine.shared.db import raw_conn, with_tenant
from engine.shared.embeddings import reset_embedder
from kb.synthesis.directed_phrases import (
    parse_directed_frontmatter,
    persist_directed_vectors,
)

_NOW = datetime(2026, 5, 8, tzinfo=UTC)


def _hash(s: str) -> bytes:
    return hashlib.sha256(" ".join(s.lower().split()).encode("utf-8")).digest()


# ---- frontmatter parsing -------------------------------------------------


def test_parse_frontmatter_missing() -> None:
    assert parse_directed_frontmatter(None) == []
    assert parse_directed_frontmatter({}) == []


def test_parse_frontmatter_present() -> None:
    assert parse_directed_frontmatter(
        {"directed": ["phrase one", "phrase two"]}
    ) == ["phrase one", "phrase two"]


def test_parse_frontmatter_strips_whitespace_and_drops_empty() -> None:
    assert parse_directed_frontmatter(
        {"directed": ["  phrase one  ", "", "  ", "phrase two"]}
    ) == ["phrase one", "phrase two"]


def test_parse_frontmatter_malformed_not_a_list_returns_empty() -> None:
    # When `directed:` is misauthored as a string (or dict) we silently
    # drop it; logging makes it discoverable, the page still persists.
    assert parse_directed_frontmatter({"directed": "not a list"}) == []
    assert parse_directed_frontmatter({"directed": {"foo": "bar"}}) == []


def test_parse_frontmatter_drops_non_string_items() -> None:
    assert parse_directed_frontmatter(
        {"directed": ["good", 42, None, "also good"]}
    ) == ["good", "also good"]


# ---- persist orchestrator (live DB) --------------------------------------


@pytest.fixture(autouse=True)
def _reset_embedder_around() -> None:
    reset_embedder()
    yield
    reset_embedder()


async def _seed_customer_and_doc(
    customer_id: str, doc_id: str, title: str = "page"
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'test', 'h-' || $1) ON CONFLICT DO NOTHING",
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                $1, 1, $2,
                'wiki', $1, 'https://wiki.example/' || $1,
                'compiled_wiki', 'wiki.runbook', 'text/markdown',
                'h-' || $1, $3, 100, 0,
                $4, $4, $4, $4, '{}'::jsonb
            )
            ON CONFLICT DO NOTHING
            """,
            doc_id,
            customer_id,
            title,
            _NOW,
        )


class _FakeProvider:
    """Protocol-conforming fake DirectedPhrasesProvider. Defined as a
    real class (not a MagicMock) so attribute typos in the orchestrator
    (e.g. `provider.run(...)` instead of `provider.generate(...)`) raise
    AttributeError loudly instead of auto-creating a passing-but-fake
    coroutine.
    """

    def __init__(
        self, phrases: list[str], *, exc: Exception | None = None
    ) -> None:
        self._phrases = list(phrases)
        self._exc = exc
        self.calls: list[tuple[str, str]] = []

    async def generate(self, *, page_title: str, page_body: str) -> list[str]:
        self.calls.append((page_title, page_body))
        if self._exc is not None:
            raise self._exc
        return list(self._phrases)


def _make_provider(phrases: list[str]) -> _FakeProvider:
    """Build a fake DirectedPhrasesProvider whose `generate()` returns
    the given phrases. Tests inject this where they used to inject
    `anthropic_client=`.
    """
    return _FakeProvider(phrases)


def _make_failing_provider(exc: Exception) -> _FakeProvider:
    """Like `_make_provider`, but raises on `.generate()`. Mirrors the
    'LLM 5xx mid-batch' failure mode the persist orchestrator handles by
    flipping `result.llm_failed=True` and skipping `_reconcile_llm`.
    """
    return _FakeProvider([], exc=exc)


@pytest.mark.asyncio
async def test_persist_adds_human_pins_no_llm(live_db) -> None:
    cust = "cust-dp-1"
    doc_id = "wiki:runbook:dp1"
    await _seed_customer_and_doc(cust, doc_id)
    # synthesis_run_id=None disables the LLM step entirely.
    res = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="Runbook",
        page_body="body text",
        frontmatter={"directed": ["alpha phrase", "beta phrase"]},
        synthesis_run_id=None,
    )
    assert res.human_added == 2
    assert res.human_removed == 0
    assert res.llm_added == 0
    assert res.llm_removed == 0
    assert res.llm_failed is False

    async with with_tenant(cust) as conn:
        rows = await conn.fetch(
            """
            SELECT source, source_text, synthesis_run_id
            FROM directed_vectors
            WHERE customer_id = $1 AND doc_id = $2
            ORDER BY source_text
            """,
            cust,
            doc_id,
        )
    assert [r["source_text"] for r in rows] == ["alpha phrase", "beta phrase"]
    assert all(r["source"] == "human" for r in rows)
    assert all(r["synthesis_run_id"] is None for r in rows)


@pytest.mark.asyncio
async def test_persist_removes_human_pins_when_frontmatter_drops_them(live_db) -> None:
    cust = "cust-dp-2"
    doc_id = "wiki:runbook:dp2"
    await _seed_customer_and_doc(cust, doc_id)
    await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter={"directed": ["alpha", "beta", "gamma"]},
        synthesis_run_id=None,
    )
    # Now re-run with one pin removed.
    res = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter={"directed": ["alpha", "gamma"]},
        synthesis_run_id=None,
    )
    assert res.human_removed == 1
    assert res.human_added == 0  # alpha + gamma already exist (idempotent)
    async with with_tenant(cust) as conn:
        rows = await conn.fetch(
            "SELECT source_text FROM directed_vectors WHERE customer_id=$1 AND doc_id=$2 ORDER BY source_text",
            cust,
            doc_id,
        )
    assert [r["source_text"] for r in rows] == ["alpha", "gamma"]


@pytest.mark.asyncio
async def test_persist_llm_regen_replaces_old_run_rows(live_db) -> None:
    cust = "cust-dp-3"
    doc_id = "wiki:runbook:dp3"
    await _seed_customer_and_doc(cust, doc_id)

    # First run: LLM emits two phrases under run_id 100.
    client_v1 = _make_provider(["llm one", "llm two"])
    res1 = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter={},
        synthesis_run_id=100,
        provider=client_v1,
    )
    assert res1.llm_added == 2
    assert res1.llm_removed == 0

    # Second run: LLM emits a different set under run_id 200 -> old rows
    # for this doc are deleted, new ones inserted.
    client_v2 = _make_provider(["llm three", "llm four"])
    res2 = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter={},
        synthesis_run_id=200,
        provider=client_v2,
    )
    assert res2.llm_removed == 2
    assert res2.llm_added == 2

    async with with_tenant(cust) as conn:
        rows = await conn.fetch(
            "SELECT source_text, synthesis_run_id FROM directed_vectors WHERE customer_id=$1 AND doc_id=$2 ORDER BY source_text",
            cust,
            doc_id,
        )
    assert [r["source_text"] for r in rows] == ["llm four", "llm three"]
    assert all(r["synthesis_run_id"] == 200 for r in rows)


@pytest.mark.asyncio
async def test_persist_llm_failure_skips_llm_keeps_humans(live_db) -> None:
    cust = "cust-dp-4"
    doc_id = "wiki:runbook:dp4"
    await _seed_customer_and_doc(cust, doc_id)

    failing_client = _make_failing_provider(RuntimeError("api down"))

    res = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter={"directed": ["pinned phrase"]},
        synthesis_run_id=42,
        provider=failing_client,
    )
    assert res.llm_failed is True
    assert res.human_added == 1
    assert res.llm_added == 0

    async with with_tenant(cust) as conn:
        rows = await conn.fetch(
            "SELECT source, source_text FROM directed_vectors WHERE customer_id=$1 AND doc_id=$2",
            cust,
            doc_id,
        )
    assert len(rows) == 1
    assert rows[0]["source"] == "human"
    assert rows[0]["source_text"] == "pinned phrase"


@pytest.mark.asyncio
async def test_persist_human_pin_preserved_when_embed_fails(live_db) -> None:
    """REGRESSION for the P1 human-pin embed-failure deletion bug.

    Setup: pin "alpha phrase" via frontmatter on run 1 (succeeds, row in
    DB). On run 2, the embedder partially fails for that pin. Pre-fix:
    `human_payload` is rebuilt only from successfully-embedded phrases,
    `desired_hashes = human_payload.keys()` excludes the failed pin,
    `_reconcile_human` computes `to_delete = existing - desired` which
    INCLUDES the still-valid pin → DELETE wipes the row. The docstring
    promised "authoritative; never overwritten" — a transient embed
    error broke the contract.

    Post-fix: `desired_hashes` is built from ALL human phrases regardless
    of embed success; the failed pin's existing row is preserved.

    The fixture stub-mode embedder doesn't fail naturally — we patch
    `embed_many` for run 2 to simulate a partial-batch failure that
    omits the human pin's index.
    """
    cust = "cust-dp-human-embed-fail"
    doc_id = "wiki:runbook:embed-fail"
    await _seed_customer_and_doc(cust, doc_id)

    # Run 1: pin succeeds and row appears.
    res1 = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter={"directed": ["alpha phrase"]},
        synthesis_run_id=None,  # human-only path, no LLM
    )
    assert res1.human_added == 1
    assert res1.human_removed == 0

    # Run 2: same frontmatter, but the embedder claims partial failure
    # for index 0 (the human pin). Pre-fix this DELETED the row.
    from kb.synthesis import directed_phrases as mod

    class _BrokenEmbedder:
        async def embed_many(self, phrases: list[str]):
            # Return an EmbedResult-shaped object: nothing embedded,
            # everything failed — simulating an OpenAI 5xx mid-batch.
            from engine.shared.embeddings import EmbedResult

            return EmbedResult(
                embedded=[],
                failed=[
                    type(
                        "FailedEmbedding",
                        (),
                        {"chunk_index": 0, "error": "simulated"},
                    )()
                ],
            )

    res2 = await mod.persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter={"directed": ["alpha phrase"]},
        synthesis_run_id=None,
        embedder=_BrokenEmbedder(),  # type: ignore[arg-type]
    )
    assert res2.human_removed == 0, (
        "Human pin must NOT be deleted when its embed fails "
        "(transient error must not violate the never-overwritten contract)."
    )

    async with with_tenant(cust) as conn:
        rows = await conn.fetch(
            "SELECT source_text FROM directed_vectors "
            "WHERE customer_id=$1 AND doc_id=$2 AND source='human'",
            cust,
            doc_id,
        )
    assert [r["source_text"] for r in rows] == ["alpha phrase"], (
        "Run 1's pin must survive run 2's embed failure."
    )


@pytest.mark.asyncio
async def test_persist_llm_failure_preserves_prior_llm_rows(live_db) -> None:
    """REGRESSION for the LLM-purge P2: a transient LLM failure must NOT
    delete previous runs' LLM rows. The pre-fix behavior was: catch the
    LLM exception, set llm_phrases=[], then call _reconcile_llm anyway —
    which DELETEs all rows whose synthesis_run_id != current_run, leaving
    nothing. One bad run permanently wiped the page's LLM-generated
    triggers until the next successful run.

    Setup: run-100 succeeds and writes 2 LLM rows. Run-200 fails the
    LLM call. Expectation: run-100's rows remain visible (last-known-good
    is preserved).
    """
    cust = "cust-dp-llm-purge"
    doc_id = "wiki:runbook:purge"
    await _seed_customer_and_doc(cust, doc_id)

    # Run 100: succeeds, writes 2 llm rows.
    ok_client = _make_provider(["good phrase one", "good phrase two"])
    res1 = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter=None,
        synthesis_run_id=100,
        provider=ok_client,
    )
    assert res1.llm_added == 2
    assert res1.llm_failed is False

    # Run 200: LLM fails. Must NOT delete run 100's rows.
    failing_client = _make_failing_provider(RuntimeError("provider 5xx"))
    res2 = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter=None,
        synthesis_run_id=200,
        provider=failing_client,
    )
    assert res2.llm_failed is True
    assert res2.llm_removed == 0, (
        "LLM-failure path must NOT call _reconcile_llm; if it did, "
        "previous run's rows would be deleted (P2 regression)."
    )

    async with with_tenant(cust) as conn:
        rows = await conn.fetch(
            "SELECT source_text, synthesis_run_id FROM directed_vectors "
            "WHERE customer_id=$1 AND doc_id=$2 AND source='llm' "
            "ORDER BY source_text",
            cust,
            doc_id,
        )
    assert [r["source_text"] for r in rows] == ["good phrase one", "good phrase two"]
    assert all(r["synthesis_run_id"] == 100 for r in rows), (
        "Run 100's rows must remain unchanged after run 200's LLM failure."
    )


@pytest.mark.asyncio
async def test_persist_llm_dup_of_human_pin_is_dropped(live_db) -> None:
    """When the LLM emits a phrase identical to a human pin, the embedder
    stub assigns the SAME vector (deterministic hash). The dedupe pass
    drops it so we don't insert two source-distinct rows for the same
    phrase. Pins always win.
    """
    cust = "cust-dp-5"
    doc_id = "wiki:runbook:dp5"
    await _seed_customer_and_doc(cust, doc_id)

    client = _make_provider(["pinned phrase", "novel llm phrase"])
    res = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter={"directed": ["pinned phrase"]},
        synthesis_run_id=7,
        provider=client,
    )
    # Human pin added, LLM contributes only the non-dup phrase.
    assert res.human_added == 1
    assert res.llm_added == 1

    async with with_tenant(cust) as conn:
        rows = await conn.fetch(
            "SELECT source, source_text FROM directed_vectors WHERE customer_id=$1 AND doc_id=$2 ORDER BY source, source_text",
            cust,
            doc_id,
        )
    sources = sorted({r["source"] for r in rows})
    assert sources == ["human", "llm"]
    texts = sorted(r["source_text"] for r in rows)
    assert texts == ["novel llm phrase", "pinned phrase"]
    # Drop telemetry: the LLM-vs-human dedup pass should have dropped
    # the duplicate ("pinned phrase") and kept the novel one. Pin this
    # so threshold-tuning regressions are visible in test output.
    assert res.llm_dropped_vs_human == 1
    assert res.llm_dropped_internal == 0


@pytest.mark.asyncio
async def test_persist_dedup_telemetry_counts_internal_drops(live_db) -> None:
    """When the LLM emits two phrases that hash to identical embeddings
    (the stub-mode embedder is deterministic per string, so two distinct
    strings can only share a vector if they normalize to the same hash —
    but the dedup pass also runs on cosine-distance < threshold, which
    triggers when DIFFERENT strings happen to produce nearby vectors).

    This test pins the internal-LLM-dedup counter by feeding TWO
    duplicate strings; the embedder gives them identical vectors, so the
    second one trips the internal-dedup branch.
    """
    cust = "cust-dp-dedup-telemetry"
    doc_id = "wiki:runbook:dedup-tel"
    await _seed_customer_and_doc(cust, doc_id)

    # Two duplicates of the same phrase + one novel one.
    client = _make_provider(["alpha phrase", "alpha phrase", "beta phrase"])
    res = await persist_directed_vectors(
        customer_id=cust,
        doc_id=doc_id,
        page_title="t",
        page_body="b",
        frontmatter=None,
        synthesis_run_id=42,
        provider=client,
    )
    assert res.llm_added == 2  # alpha (first occurrence) + beta
    assert res.llm_dropped_vs_human == 0
    # The duplicate "alpha phrase" trips internal-dedup.
    assert res.llm_dropped_internal == 1
