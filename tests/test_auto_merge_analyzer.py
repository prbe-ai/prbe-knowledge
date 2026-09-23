"""Unit tests for the AutoMergeAnalyzer.

The deterministic pieces (path-canonical detection, property-key conflict
filtering, the gpt-oss prompt) plus the analyze() flow with its DB edges and
judge stubbed. tests/test_auto_merge_live.py runs the flow on real Postgres.
"""

from __future__ import annotations

import json
import types
import uuid

import pytest
from fastapi import HTTPException

from engine.ingest.auto_merge import analyzer as az
from engine.ingest.auto_merge.analyzer import (
    Candidate,
    _build_prompt,
    _is_path_canonical,
    _properties_conflict,
)
from engine.ingest.auto_merge.jev_judge import Judgment
from engine.ingest.auto_merge.models import AutoMergeVerdict
from engine.retrieval.agent import jev
from engine.retrieval.agent.jev import (
    JevBreakerOpen,
    JevError,
    JevRequestRejected,
    JevRequestTooLarge,
)
from engine.shared import constants
from engine.shared.constants import (
    AUTO_MERGE_BREAKER_JITTER_SECONDS,
    AUTO_MERGE_RETRY_SECONDS,
    JEV_BREAKER_FAILURES,
    SEARCH_AGENT_INFERENCE_MODEL,
    AutoMergeJudge,
)
from engine.shared.llm import LLMError

# --------------------------------------------------------------------------- #
# _is_path_canonical
# --------------------------------------------------------------------------- #


def test_path_canonical_rejects_pr_canonical_ids() -> None:
    assert _is_path_canonical("PR", "prbe-ai/prbe-knowledge#345")
    assert _is_path_canonical("Issue", "owner/repo#42")


def test_path_canonical_rejects_code_symbol_ids() -> None:
    assert _is_path_canonical(
        "Function",
        "prbe-ai/prbe-knowledge:engine.ingest.graph_writer.upsert_nodes",
    )
    assert _is_path_canonical(
        "Method", "prbe-ai/prbe-knowledge:app.services.foo.MyClass.__init__"
    )


def test_path_canonical_rejects_repo_owner_name() -> None:
    assert _is_path_canonical("Repo", "prbe-ai/prbe-knowledge")


def test_path_canonical_accepts_freeform_labels() -> None:
    assert not _is_path_canonical("Person", "richardwei6")
    assert not _is_path_canonical("Person", "Richard Wei")
    assert not _is_path_canonical("Topic", "litellm-proxy")
    assert not _is_path_canonical("Author", "ashwaryeyadav")
    assert not _is_path_canonical("Feature", "auto-merge")


def test_path_canonical_treats_blank_as_skip() -> None:
    # Blank canonical_id is degenerate — analyzer should skip to avoid noise.
    assert _is_path_canonical("Person", "")


# --------------------------------------------------------------------------- #
# _properties_conflict
# --------------------------------------------------------------------------- #


def test_properties_conflict_on_different_emails() -> None:
    assert _properties_conflict(
        {"email": "a@x.com", "name": "A"},
        {"email": "b@x.com", "name": "B"},
    )


def test_properties_match_on_same_emails() -> None:
    assert not _properties_conflict(
        {"email": "a@x.com", "name": "A"},
        {"email": "a@x.com", "name": "A2"},
    )


def test_properties_no_conflict_when_one_side_missing_email() -> None:
    # If only one side has the stable key, we can't decisively reject.
    # Let the LLM judge.
    assert not _properties_conflict(
        {"name": "A"},
        {"email": "a@x.com", "name": "A2"},
    )


def test_properties_conflict_on_repo_number_mismatch() -> None:
    assert _properties_conflict(
        {"repo": "owner/x", "number": 1},
        {"repo": "owner/y", "number": 1},
    )


def test_properties_conflict_on_owner_name_mismatch() -> None:
    assert _properties_conflict(
        {"owner": "a", "name": "x"},
        {"owner": "b", "name": "x"},
    )


# --------------------------------------------------------------------------- #
# _build_prompt
# --------------------------------------------------------------------------- #


def test_build_prompt_includes_new_entity_and_all_candidates() -> None:
    node = {
        "label": "Person",
        "canonical_id": "richardwei6",
        "properties": {"name": "Richard Wei", "source_system": "github"},
        "degree": 12,
    }
    candidates = [
        Candidate(
            canonical_id="Richard Wei",
            properties={"email": "richard@prbe.ai", "source_system": "slack"},
            degree=8,
            trigram_score=0.45,
            vector_distance=None,
        ),
        Candidate(
            canonical_id="00000000-0000-0000-0000-000000000001",
            properties={"name": "Richard W", "email": "richard@prbe.ai"},
            degree=4,
            trigram_score=None,
            vector_distance=0.12,
        ),
    ]
    prompt = _build_prompt(node, candidates)
    assert "richardwei6" in prompt
    assert "Richard Wei" in prompt
    assert "00000000-0000-0000-0000-000000000001" in prompt
    assert "trigram=0.45" in prompt
    assert "vector_distance=0.120" in prompt
    assert "richard@prbe.ai" in prompt


def test_build_prompt_with_empty_signals() -> None:
    node = {
        "label": "Topic",
        "canonical_id": "auto-merge",
        "properties": {},
        "degree": 0,
    }
    candidates = [
        Candidate(
            canonical_id="entity-dedup",
            properties={},
            degree=0,
            trigram_score=None,
            vector_distance=None,
        ),
    ]
    prompt = _build_prompt(node, candidates)
    assert "signals:      none" in prompt


# --------------------------------------------------------------------------- #
# AutoMergeVerdict schema
# --------------------------------------------------------------------------- #


def test_verdict_unique_with_no_primary() -> None:
    v = AutoMergeVerdict.model_validate_json(
        '{"verdict": "unique", "primary_canonical_id": null, '
        '"confidence": null, "rationale": "no overlap"}'
    )
    assert v.verdict == "unique"
    assert v.primary_canonical_id is None


def test_verdict_duplicate_high_confidence() -> None:
    v = AutoMergeVerdict.model_validate_json(
        '{"verdict": "duplicate", "primary_canonical_id": "Richard Wei", '
        '"confidence": "high", "rationale": "shared email richard@prbe.ai"}'
    )
    assert v.verdict == "duplicate"
    assert v.primary_canonical_id == "Richard Wei"
    assert v.confidence == "high"


def test_verdict_rejects_extra_fields() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AutoMergeVerdict.model_validate_json(
            '{"verdict": "unique", "rationale": "x", "extra_field": 1}'
        )


# --------------------------------------------------------------------------- #
# analyze() flow: judge outcome -> merge / suggestion / nothing / defer
#
# The DB edges (_load_node, _find_candidates, merge_cluster) are stubbed here;
# tests/test_auto_merge_live.py runs the same flow against real Postgres.
# --------------------------------------------------------------------------- #

PR_NODE = {
    "node_id": 7,
    "label": "Document",
    "canonical_id": "github:acme/widgets:pr:12",
    "properties": {"name": "Add widgets"},
    "degree": 1,
    "has_embedding": True,
}
PR_CANDS = [
    Candidate("acme/widgets#12", {"name": "Add widgets"}, 3, 0.78, 0.02),
    Candidate("acme/widgets#13", {"name": "Remove widgets"}, 2, 0.70, 0.09),
]
PERSON_NODE = {
    "node_id": 8,
    "label": "Person",
    "canonical_id": "ada-gh",
    "properties": {"name": "Ada Lovelace", "login": "ada-gh", "source_system": "github"},
    "degree": 1,
    "has_embedding": True,
}

# Both judges, so the verdict-to-action contract is pinned for the rollback too.
JUDGES = [
    pytest.param("jev-1.13.0", 0.99, id="jev"),
    pytest.param(SEARCH_AGENT_INFERENCE_MODEL, None, id="gptoss"),
]


class FakeConn:
    def __init__(self, documents: tuple[str, ...] = ()) -> None:
        self.suggestions: list[tuple] = []
        self.documents = set(documents)

    async def fetchval(self, sql: str, *args):
        assert "FROM documents" in sql, sql
        return args[1] in self.documents

    async def fetchrow(self, sql: str, *args):
        assert "INSERT INTO entity_merge_suggestions" in sql, sql
        self.suggestions.append(args)
        return {"suggestion_id": uuid.uuid4()}


def _verdict(primary: str | None, confidence: str | None) -> AutoMergeVerdict:
    if primary is None:
        return AutoMergeVerdict(verdict="unique", rationale="none of them")
    return AutoMergeVerdict(
        verdict="duplicate", primary_canonical_id=primary, confidence=confidence, rationale="same repo and number"
    )


def _analyzer(monkeypatch, node, cands, *, judgment=None, raises=None, execute=True, merge_raises=None):
    a = az.AutoMergeAnalyzer(execute_high_confidence=execute)

    async def load(conn, node_id):
        return node

    async def find(conn, n):
        return list(cands)

    async def judge(n, c):
        if raises is not None:
            raise raises
        return judgment

    merges: list = []

    async def fake_merge(body):
        merges.append(body)
        if merge_raises is not None:
            raise merge_raises
        return types.SimpleNamespace(merge_id=uuid.uuid4())

    monkeypatch.setattr(a, "_load_node", load)
    monkeypatch.setattr(a, "_find_candidates", find)
    monkeypatch.setattr(a, "_judge", judge)
    monkeypatch.setattr(az, "merge_cluster", fake_merge)
    return a, merges


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_high_merges_and_the_audit_reason_names_the_judge(monkeypatch, model, p):
    j = Judgment(verdict=_verdict("acme/widgets#12", "high"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j)
    conn = FakeConn()
    result = await a.analyze(conn, "acme-test", 7)
    assert result.action == "merged"
    assert (result.judge_model, result.p) == (model, p)
    (body,) = merges
    assert body.primary_canonical_id == "acme/widgets#12"
    assert body.alias_canonical_ids == ["github:acme/widgets:pr:12"]
    expected_p = " p=0.99" if p is not None else ""
    assert body.reason == f"auto: model={model} confidence=high{expected_p} rationale=same repo and number"
    # A re-upserted cluster primary must not be folded in (its aliases would
    # route to a deleted node): merge_cluster enforces it under its lock.
    assert body.refuse_cluster_primaries is True
    assert conn.suggestions == []


async def test_a_failed_merge_is_kept_as_a_high_suggestion(monkeypatch):
    j = Judgment(verdict=_verdict("acme/widgets#12", "high"), model="jev-1.13.0", p=0.99)
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j, merge_raises=RuntimeError("statement timeout"))
    conn = FakeConn()
    result = await a.analyze(conn, "acme-test", 7)
    assert len(merges) == 1
    assert result.action == "suggested"
    (row,) = conn.suggestions
    assert row[2:5] == ("acme/widgets#12", "github:acme/widgets:pr:12", "high")


@pytest.mark.parametrize("status", [404, 409])
async def test_a_merge_refused_or_already_gone_is_an_error_not_a_suggestion(monkeypatch, status):
    # 404: a concurrent merge folded one side first. 409: e.g. the new node is
    # a cluster primary; a suggestion would invite approving the chain.
    j = Judgment(verdict=_verdict("acme/widgets#12", "high"), model="jev-1.13.0", p=0.99)
    a, _ = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j, merge_raises=HTTPException(status_code=status))
    conn = FakeConn()
    result = await a.analyze(conn, "acme-test", 7)
    assert result.action == "error" and conn.suggestions == []


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_a_document_node_is_never_judged_or_merged(monkeypatch, model, p):
    # The new node's id is one of the tenant's documents: folding it into its
    # mention would cut the document out of graph retrieval.
    j = Judgment(verdict=_verdict("acme/widgets#12", "high"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j)
    conn = FakeConn(documents=("github:acme/widgets:pr:12",))
    result = await a.analyze(conn, "acme-test", 7)
    assert (result.action, result.rationale) == ("skipped", "document node")
    assert merges == [] and conn.suggestions == []


async def test_a_document_stub_written_before_its_document_is_skipped_too(monkeypatch):
    # Some writers create a document's node (marked with doc_type) before the
    # documents row exists; the row check alone would miss it.
    j = Judgment(verdict=_verdict("acme/widgets#12", "high"), model="jev-1.13.0", p=0.99)
    node = dict(PR_NODE, properties={"doc_type": "github.pull_request"})
    a, merges = _analyzer(monkeypatch, node, PR_CANDS, judgment=j)
    result = await a.analyze(FakeConn(), "acme-test", 7)
    assert (result.action, result.rationale) == ("skipped", "document node")
    assert merges == []


async def test_the_merge_rechecks_both_refusals_under_the_lock(monkeypatch):
    j = Judgment(verdict=_verdict("acme/widgets#12", "high"), model="jev-1.13.0", p=0.99)
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j)
    await a.analyze(FakeConn(), "acme-test", 7)
    (body,) = merges
    assert (body.refuse_cluster_primaries, body.refuse_document_aliases) == (True, True)


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_high_without_execute_writes_a_suggestion_stamped_with_the_judge(monkeypatch, model, p):
    j = Judgment(verdict=_verdict("acme/widgets#12", "high"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j, execute=False)
    conn = FakeConn()
    result = await a.analyze(conn, "acme-test", 7)
    assert result.action == "suggested"
    assert merges == []
    (row,) = conn.suggestions
    # customer, label, primary, candidate(new node), confidence, rationale, llm_model
    assert row == ("acme-test", "Document", "acme/widgets#12", "github:acme/widgets:pr:12", "high",
                   "same repo and number", model)


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_medium_writes_a_suggestion(monkeypatch, model, p):
    j = Judgment(verdict=_verdict("acme/widgets#12", "medium"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j)
    conn = FakeConn()
    result = await a.analyze(conn, "acme-test", 7)
    assert result.action == "suggested" and merges == []
    assert conn.suggestions[0][4] == "medium"


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_unique_writes_nothing(monkeypatch, model, p):
    j = Judgment(verdict=_verdict(None, None), model=model, p=p)
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j)
    conn = FakeConn()
    result = await a.analyze(conn, "acme-test", 7)
    assert result.action == "unique"
    assert merges == [] and conn.suggestions == []


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_primary_outside_the_candidates_is_an_error(monkeypatch, model, p):
    j = Judgment(verdict=_verdict("acme/widgets#99", "high"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, judgment=j)
    result = await a.analyze(FakeConn(), "acme-test", 7)
    assert result.action == "error" and merges == []


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_person_on_a_name_alone_is_a_suggestion_not_a_merge(monkeypatch, model, p):
    cands = [Candidate("U123", {"name": "Ada Lovelace", "display_name": "Ada Lovelace"}, 4, 1.0, 0.05)]
    j = Judgment(verdict=_verdict("U123", "high"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, PERSON_NODE, cands, judgment=j)
    conn = FakeConn()
    result = await a.analyze(conn, "acme-test", 8)
    assert result.action == "suggested"
    assert merges == []
    assert conn.suggestions[0][4] == "medium"


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_person_with_a_shared_login_merges(monkeypatch, model, p):
    cands = [Candidate("ada@example.com", {"name": "Ada Lovelace", "login": "ADA-GH", "source_system": "github"},
                       4, 0.4, 0.05)]
    j = Judgment(verdict=_verdict("ada@example.com", "high"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, PERSON_NODE, cands, judgment=j)
    result = await a.analyze(FakeConn(), "acme-test", 8)
    assert result.action == "merged"
    assert merges[0].primary_canonical_id == "ada@example.com"


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_a_login_from_another_source_system_is_not_identity(monkeypatch, model, p):
    # A GitHub login can collide with an opaque id from elsewhere (a Slack user
    # id is also a short token): a login only counts within one source system.
    node = dict(PERSON_NODE, canonical_id="u012abcdef",
                properties={"name": "Ada", "login": "u012abcdef", "source_system": "github"})
    cands = [Candidate("U012ABCDEF", {"name": "Ada", "source_system": "slack"}, 4, 1.0, 0.05)]
    j = Judgment(verdict=_verdict("U012ABCDEF", "high"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, node, cands, judgment=j)
    result = await a.analyze(FakeConn(), "acme-test", 8)
    assert result.action == "suggested" and merges == []


@pytest.mark.parametrize(("model", "p"), JUDGES)
async def test_document_without_id_evidence_is_a_suggestion_not_a_merge(monkeypatch, model, p):
    # Two different pages the judge is sure about: no shared id, so no auto-merge.
    node = dict(PR_NODE, canonical_id="notion:page:1111", properties={"title": "Q3 plan"})
    cands = [Candidate("notion:page:2222", {"title": "Q3 plan"}, 2, 0.9, 0.01)]
    j = Judgment(verdict=_verdict("notion:page:2222", "high"), model=model, p=p)
    a, merges = _analyzer(monkeypatch, node, cands, judgment=j)
    conn = FakeConn()
    result = await a.analyze(conn, "acme-test", 7)
    assert result.action == "suggested" and merges == []
    assert conn.suggestions[0][4] == "medium"


async def test_open_breaker_defers_before_the_candidate_search(monkeypatch):
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, raises=AssertionError("judge must not run"))

    async def no_search(conn, n):
        raise AssertionError("an open breaker must not pay for the candidate search")

    monkeypatch.setattr(a, "_find_candidates", no_search)
    monkeypatch.setattr(az, "get_settings", lambda: types.SimpleNamespace(typesafe_api_key="k"))
    monkeypatch.setattr(az, "AUTO_MERGE_JUDGE", AutoMergeJudge.JEV)
    breaker = jev.Breaker()
    for _ in range(JEV_BREAKER_FAILURES):
        breaker.failure()
    monkeypatch.setattr(az, "MERGE_BREAKER", breaker)
    result = await a.analyze(FakeConn(), "acme-test", 7)
    assert result.action == "deferred"
    assert 1 <= result.retry_after_seconds <= breaker.seconds_until_closed() + 1 + AUTO_MERGE_BREAKER_JITTER_SECONDS
    assert merges == []


async def test_breaker_opening_mid_call_defers(monkeypatch):
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, raises=JevBreakerOpen("breaker_open"))
    result = await a.analyze(FakeConn(), "acme-test", 7)
    assert result.action == "deferred"
    assert result.retry_after_seconds >= 1
    assert merges == []


async def test_jev_outage_defers(monkeypatch):
    a, merges = _analyzer(monkeypatch, PR_NODE, PR_CANDS, raises=JevError("ReadTimeout: <empty>"))
    result = await a.analyze(FakeConn(), "acme-test", 7)
    assert result.action == "deferred"
    assert result.retry_after_seconds == AUTO_MERGE_RETRY_SECONDS
    assert merges == []


@pytest.mark.parametrize(
    "exc",
    [JevRequestTooLarge("http_400:max_tokens_exceeded"), JevRequestRejected("http_422:validation_error")],
)
async def test_permanent_request_failures_are_errors_not_retries(monkeypatch, exc):
    a, _ = _analyzer(monkeypatch, PR_NODE, PR_CANDS, raises=exc)
    result = await a.analyze(FakeConn(), "acme-test", 7)
    assert result.action == "error"


async def test_gptoss_llm_error_keeps_todays_behaviour(monkeypatch):
    a, _ = _analyzer(monkeypatch, PR_NODE, PR_CANDS, raises=LLMError("boom", provider="cerebras"))
    result = await a.analyze(FakeConn(), "acme-test", 7)
    assert result.action == "error"


# --------------------------------------------------------------------------- #
# _judge dispatch: Jev by default, gpt-oss on rollback or without a key
# --------------------------------------------------------------------------- #


def _gptoss_response(verdict: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(verdict)}}]}


async def test_rollback_switch_runs_the_gptoss_path_unchanged(monkeypatch):
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _gptoss_response({"verdict": "duplicate", "primary_canonical_id": "acme/widgets#12",
                                 "confidence": "high", "rationale": "same PR"})

    monkeypatch.setattr(az, "AUTO_MERGE_JUDGE", AutoMergeJudge.GPTOSS)
    monkeypatch.setattr(az, "acompletion", fake_acompletion)
    monkeypatch.setattr(az.jev_judge, "judge", lambda *a, **k: pytest.fail("Jev must not be called"))
    judgment = await az.AutoMergeAnalyzer()._judge(PR_NODE, PR_CANDS)
    assert judgment.model == SEARCH_AGENT_INFERENCE_MODEL and judgment.p is None
    assert judgment.verdict.primary_canonical_id == "acme/widgets#12"
    (kw,) = calls
    # Today's request, verbatim.
    assert kw["model"] == SEARCH_AGENT_INFERENCE_MODEL
    assert kw["messages"] == [
        {"role": "system", "content": az._SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(PR_NODE, PR_CANDS)},
    ]
    assert kw["response_format"] is az._VERDICT_RESPONSE_FORMAT
    assert (kw["temperature"], kw["max_tokens"], kw["custom_llm_provider"]) == (0.1, 512, "openai")


async def test_jev_is_the_default_judge(monkeypatch):
    seen: dict = {}

    async def fake_judge(node, candidates, *, api_key):
        seen.update(node=node, n=len(candidates), api_key=api_key)
        return Judgment(verdict=_verdict(None, None), model="jev-1.13.0", p=0.9)

    monkeypatch.setattr(az, "AUTO_MERGE_JUDGE", AutoMergeJudge.JEV)
    monkeypatch.setattr(az, "get_settings", lambda: types.SimpleNamespace(typesafe_api_key="k-test"))
    monkeypatch.setattr(az.jev_judge, "judge", fake_judge)
    monkeypatch.setattr(az, "acompletion", lambda **k: pytest.fail("gpt-oss must not be called"))
    judgment = await az.AutoMergeAnalyzer()._judge(PR_NODE, PR_CANDS)
    assert judgment.model == "jev-1.13.0"
    assert seen == {"node": PR_NODE, "n": 2, "api_key": "k-test"}


async def test_missing_key_falls_back_to_gptoss_and_warns_once(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _gptoss_response({"verdict": "unique", "rationale": "no"})

    warnings: list = []
    monkeypatch.setattr(az, "AUTO_MERGE_JUDGE", AutoMergeJudge.JEV)
    monkeypatch.setattr(az, "get_settings", lambda: types.SimpleNamespace(typesafe_api_key=""))
    monkeypatch.setattr(az, "acompletion", fake_acompletion)
    monkeypatch.setattr(az.log, "warning", lambda event, **kw: warnings.append(event))
    az._warn_jev_unconfigured.cache_clear()
    try:
        for _ in range(3):
            judgment = await az.AutoMergeAnalyzer()._judge(PR_NODE, PR_CANDS)
            assert judgment.model == SEARCH_AGENT_INFERENCE_MODEL
    finally:
        az._warn_jev_unconfigured.cache_clear()
    assert warnings == ["auto_merge.jev_unconfigured"]


def test_the_shipped_judge_is_jev():
    # The other tests set the switch explicitly; this pins what ships.
    assert constants.AUTO_MERGE_JUDGE is AutoMergeJudge.JEV
