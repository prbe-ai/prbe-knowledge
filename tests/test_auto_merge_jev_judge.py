"""Jev as the auto-merge judge: the request it sends, the verdict it maps to,
the identity evidence, and the templated rationale.

All entities here are synthetic. No network: `judge()` runs over an httpx
MockTransport.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import httpx
import pytest

from engine.ingest.auto_merge import jev_judge as jj
from engine.retrieval.agent.jev import ChoiceAnswer
from engine.shared.constants import (
    AUTO_MERGE_JEV_HIGH_AT,
    AUTO_MERGE_JEV_MAX_VALUE_CHARS,
    AUTO_MERGE_JEV_MODEL,
    AUTO_MERGE_JEV_SUGGEST_AT,
)


@dataclass
class Cand:
    canonical_id: str
    properties: dict
    degree: int = 1
    trigram_score: float | None = None
    vector_distance: float | None = None


NODE = {
    "label": "Person",
    "canonical_id": "ada-gh",
    "properties": {"name": "Ada Lovelace", "login": "ada-gh", "email": "ada@example.com"},
    "degree": 3,
}
CANDS = [
    Cand("ada@example.com", {"name": "Ada Lovelace", "email": "ada@example.com"}, 5, 0.41, 0.0412),
    Cand("grace@example.com", {"name": "Grace Hopper", "email": "grace@example.com"}, 2, None, 0.2),
]


def _answer(choice: str, p: float, model: str = "jev-1.13.0") -> ChoiceAnswer:
    probs = {"c0": 0.0, "c1": 0.0, jj.NONE_OF_THESE: 0.0}
    probs[choice] = p
    return ChoiceAnswer(choice=choice, probabilities=probs, model=model, input_tokens=900, elapsed_ms=120.0)


# --------------------------------------------------------------------------- #
# the request -- this IS what the 2026-09-23 replay measured
# --------------------------------------------------------------------------- #


def test_request_shape_matches_the_replay():
    state, question, keys = jj.build_request(NODE, CANDS)
    assert list(keys) == ["c0", "c1"]  # analyzer ranking order
    assert keys["c0"] is CANDS[0]
    assert state == {
        "new_entity": {
            "label": "Person",
            "canonical_id": "ada-gh",
            "properties": NODE["properties"],
            "degree": 3,
        },
        "candidates": {
            "c0": {
                "canonical_id": "ada@example.com",
                "properties": CANDS[0].properties,
                "degree": 5,
                "signals": {"trigram": 0.41, "vector_distance": 0.041},
            },
            "c1": {
                "canonical_id": "grace@example.com",
                "properties": CANDS[1].properties,
                "degree": 2,
                "signals": {"vector_distance": 0.2},
            },
        },
    }
    assert question["type"] == "choice"
    assert question["criteria"] == {
        "c0": "state.candidates.c0 (canonical_id ada@example.com) is the same real-world thing as state.new_entity",
        "c1": "state.candidates.c1 (canonical_id grace@example.com) is the same real-world thing as state.new_entity",
        "none_of_these": "none of the candidates is the same real-world thing as state.new_entity",
    }


def test_instructions_are_the_measured_text():
    # Tripwire: rewording the instructions is a model change. Re-run
    # scripts/jev_automerge/ on the frozen decision set, then update this hash.
    digest = hashlib.sha256(jj._INSTRUCTIONS.encode()).hexdigest()
    assert digest == "b4dc00dc5511c3dc62e1a140ecb53bea2e02ccc71e6886330745b97fca8eb383"


def test_long_string_values_are_trimmed_but_every_key_survives():
    long = "x" * (AUTO_MERGE_JEV_MAX_VALUE_CHARS + 50)
    props = {"email": "ada@example.com", "bio": long, "tags": [long, "ok"], "nested": {"note": long}, "n": 7}
    out = jj.trim_values(props)
    assert set(out) == set(props)
    assert out["email"] == "ada@example.com"
    assert out["n"] == 7
    assert out["bio"] == "x" * AUTO_MERGE_JEV_MAX_VALUE_CHARS + "…"
    assert out["tags"] == ["x" * AUTO_MERGE_JEV_MAX_VALUE_CHARS + "…", "ok"]
    assert out["nested"]["note"].endswith("…")


def test_long_lists_keep_a_bounded_prefix_and_say_how_many_were_cut():
    out = jj.trim_values({"members": [f"u{i}" for i in range(jj.MAX_LIST_ITEMS + 7)]})
    assert out["members"][: jj.MAX_LIST_ITEMS] == [f"u{i}" for i in range(jj.MAX_LIST_ITEMS)]
    assert out["members"][-1] == "… 7 more"
    assert len(out["members"]) == jj.MAX_LIST_ITEMS + 1
    assert jj.trim_values(["a", "b"]) == ["a", "b"]


def test_request_trims_both_sides():
    node = dict(NODE, properties={"email": "ada@example.com", "bio": "y" * 5000})
    cand = Cand("c", {"description": "z" * 5000})
    state, _, _ = jj.build_request(node, [cand])
    assert len(state["new_entity"]["properties"]["bio"]) == AUTO_MERGE_JEV_MAX_VALUE_CHARS + 1
    assert len(state["candidates"]["c0"]["properties"]["description"]) == AUTO_MERGE_JEV_MAX_VALUE_CHARS + 1


# --------------------------------------------------------------------------- #
# answer -> verdict
# --------------------------------------------------------------------------- #

_, _, KEYS = jj.build_request(NODE, CANDS)


def test_none_of_these_is_unique():
    j = jj.verdict_from_answer(_answer(jj.NONE_OF_THESE, 0.91), KEYS, NODE)
    assert j.verdict.verdict == "unique"
    assert j.verdict.primary_canonical_id is None
    assert (j.p, j.model) == (0.91, "jev-1.13.0")


@pytest.mark.parametrize(
    ("p", "verdict", "confidence"),
    [
        (1.0, "duplicate", "high"),
        (AUTO_MERGE_JEV_HIGH_AT, "duplicate", "high"),
        (AUTO_MERGE_JEV_HIGH_AT - 0.0001, "duplicate", "medium"),
        (AUTO_MERGE_JEV_SUGGEST_AT, "duplicate", "medium"),
        (AUTO_MERGE_JEV_SUGGEST_AT - 0.0001, "unique", None),
        (0.0, "unique", None),
    ],
)
def test_probability_bands(p, verdict, confidence):
    j = jj.verdict_from_answer(_answer("c0", p), KEYS, NODE)
    v = j.verdict
    assert v.verdict == verdict
    assert v.confidence == confidence
    assert j.p == p
    if verdict == "duplicate":
        assert v.primary_canonical_id == "ada@example.com"
    assert len(v.rationale) <= 240


def test_an_answer_from_another_model_never_auto_merges():
    # The bands were measured on AUTO_MERGE_JEV_MODEL; a server that answers
    # with a different model gets a suggestion at most.
    j = jj.verdict_from_answer(_answer("c0", 0.99, model="jev-2.0.0"), KEYS, NODE)
    assert (j.verdict.verdict, j.verdict.confidence, j.model) == ("duplicate", "medium", "jev-2.0.0")


def test_bands_are_the_reviewed_values():
    assert (AUTO_MERGE_JEV_HIGH_AT, AUTO_MERGE_JEV_SUGGEST_AT) == (0.95, 0.70)


# --------------------------------------------------------------------------- #
# identity evidence (the analyzer's Person guard uses this too)
# --------------------------------------------------------------------------- #


GH = {"source_system": "github"}
WS = "11111111-2222-4333-8444-555555555555"


@pytest.mark.parametrize(
    ("a", "pa", "b", "pb", "expect"),
    [
        ("x", {"email": "Ada@Example.com"}, "y", {"email": "ada@example.com "}, "shared email Ada@Example.com"),
        ("ada@example.com", {}, "y", {"email": "ada@example.com"}, "email ada@example.com is the other entity's id"),
        ("x", {"login": "ada-gh", **GH}, "y", {"login": "ADA-GH", **GH}, "shared handle ada-gh"),
        ("ada-gh", GH, "y", {"login": "ada-gh", **GH}, "login ada-gh is the other entity's id"),
        # A login is only an identifier within one source system: a GitHub
        # login can equal an opaque id from somewhere else.
        ("U012ABCDEF", {"source_system": "slack"}, "y", {"login": "u012abcdef", **GH}, None),
        ("x", {"login": "ada-gh"}, "y", {"login": "ada-gh"}, None),  # source system unknown
        ("x", {"login": "ada-gh", **GH}, "y", {"login": "ada-gh", "source_system": "slack"}, None),
        ("x", {"name": "Ada Lovelace"}, "y", {"name": "Ada Lovelace"}, None),  # a name is not an identifier
        ("x", {"email": ""}, "y", {"email": ""}, None),
        ("x", {"email": None}, "y", {}, None),
        ("x", None, "y", None, None),
    ],
)
def test_shared_identifier(a, pa, b, pb, expect):
    assert jj.shared_identifier(a, pa, b, pb) == expect


@pytest.mark.parametrize(
    ("label", "a", "pa", "b", "pb", "has_evidence"),
    [
        ("Document", "github:acme/widgets:pr:12", {}, "acme/widgets#12", {}, True),
        ("Document", "github:acme/widgets:pr:12", {}, "acme/widgets#13", {}, False),
        ("Document", "linear:ws:issue:0f8fad5b-d9cb-469f-a165-70867728950e", {},
         "0f8fad5b-d9cb-469f-a165-70867728950e", {}, True),
        ("Document", "acme_widget_tool", {}, "acme-widget-tool", {}, True),
        ("Document", "acme/acme-widgets", {}, "acme-widgets", {}, True),
        ("Document", "wiki:repo:acme_widgets", {}, "acme/acme-widgets", {}, True),
        ("Document", "alice/utils", {}, "bob/utils", {}, False),  # same name, different owners
        ("Document", "alice-x/utils", {}, "alice/x-utils", {}, False),  # same letters, different segments
        ("Document", "notion:page:1", {"title": "Plan"}, "notion:page:2", {"title": "Plan"}, False),
        # Only the LEAF uuid is the entity's own: sibling issues share their
        # workspace uuid.
        ("Document", f"linear:{WS}:issue:0f8fad5b-d9cb-469f-a165-70867728950e", {},
         f"linear:{WS}:issue:2c1b9f4e-7a3d-4e21-9b8a-5d6f7e8a9b0c", {}, False),
        # A UUID counts only as the id's last segment: an upload id ends in the
        # customer's own key, and the tenant uuid before it is shared by all.
        ("Document", f"custom_ingest:{WS}:notes:q3-plan", {}, f"custom_ingest:{WS}:notes:q3-plan-draft", {}, False),
        ("Document", f"custom_ingest:{WS}:notes:{WS}", {}, WS, {}, True),
        # Case and -/_ fold only for repo-style slugs.
        ("Document", "custom_ingest:t:notes:Readme", {}, "custom_ingest:t:notes:README", {}, False),
        # Case and -/_ fold; dots and run-together letters do not.
        ("Document", "Acme-Widgets", {}, "acme_widgets", {}, True),
        ("Document", "model-v1.1", {}, "model-v11", {}, False),
        ("Document", "acme-widget", {}, "acmewidget", {}, False),
        # The repo NAME rules are identity only for repos (Documents); for any
        # other entity they would be "same name".
        ("Project", "trainer", {}, "acme/trainer", {}, False),
        ("Service", "api-gateway", {}, "API_Gateway", {}, False),
        ("Project", "linear:ws:project:0f8fad5b-d9cb-469f-a165-70867728950e", {},
         "0f8fad5b-d9cb-469f-a165-70867728950e", {}, True),
        ("Person", "ada-gh", GH, "ada@example.com", {"login": "ada-gh", **GH}, True),
        ("Person", "U1", {"name": "Ada"}, "U2", {"name": "Ada"}, False),
        # People need a shared email/login: id look-alikes are not enough.
        ("Person", "acme_widget_tool", {}, "acme-widget-tool", {}, False),
    ],
)
def test_execution_evidence(label, a, pa, b, pb, has_evidence):
    assert (jj.execution_evidence(label, a, pa, b, pb) is not None) == has_evidence


# --------------------------------------------------------------------------- #
# templated rationale
# --------------------------------------------------------------------------- #


def _t(a, pa, b, pb, tri=None, vec=None, p=None):
    return jj.template_rationale({"canonical_id": a, "properties": pa}, Cand(b, pb, 1, tri, vec), p)


@pytest.mark.parametrize(
    ("a", "pa", "b", "pb", "starts"),
    [
        ("x", {"email": "ada@example.com"}, "y", {"email": "ada@example.com"}, "shared email ada@example.com"),
        (
            "linear:ws:issue:0f8fad5b-d9cb-469f-a165-70867728950e",
            {},
            "0f8fad5b-d9cb-469f-a165-70867728950e",
            {},
            "shared id 0f8fad5b-d9cb-469f-a165-70867728950e",
        ),
        ("github:acme/widgets:pr:12", {}, "acme/widgets#12", {}, "same repo acme/widgets and number 12"),
        ("github:acme/widgets:issue:3", {}, "acme/widgets#3", {}, "same repo acme/widgets and number 3"),
        ("acme_widget_tool", {}, "acme-widget-tool", {}, "ids equal ignoring case and -/_"),
        ("acme-widgets", {}, "acme/acme-widgets", {}, "same repo name once owner/wiki prefix"),
        ("wiki:repo:acme_widgets", {}, "acme/acme-widgets", {}, "same repo name once owner/wiki prefix"),
        ("U123", {"name": "Ada Lovelace"}, "U456", {"display_name": "Ada  Lovelace"}, 'same name "Ada Lovelace"'),
        ("run:deadbeef01", {}, "job:deadbeef01", {}, "shared id token deadbeef01"),
        ("notion:page:1", {}, "notion:page:2", {}, "no shared identifier"),
    ],
)
def test_template_picks_the_strongest_signal(a, pa, b, pb, starts):
    assert _t(a, pa, b, pb).startswith(starts)


def test_different_pr_numbers_are_not_evidence():
    assert _t("github:acme/widgets:pr:12", {}, "acme/widgets#13", {}).startswith("no shared identifier")


def test_template_appends_the_numbers():
    out = _t("x", {"email": "a@example.com"}, "y", {"email": "a@example.com"}, tri=0.4567, vec=0.07, p=0.981)
    assert out == "shared email a@example.com; name/id trigram 0.46, embedding similarity 0.93, Jev p=0.98"


def test_template_is_capped_at_240():
    long_id = "acme/" + "w" * 400
    assert len(_t(long_id, {}, long_id.replace("acme/", "other/"), {}, tri=1.0, p=0.99)) <= 240


# --------------------------------------------------------------------------- #
# judge(): one call, end to end over a MockTransport
# --------------------------------------------------------------------------- #


async def test_candidates_too_long_to_be_a_primary_are_never_offered():
    too_long = Cand("x" * (jj.MAX_PRIMARY_ID_CHARS + 1), {"email": "ada@example.com"})

    def handler(req: httpx.Request):
        pytest.fail("nothing left to ask about")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        judgment = await jj.judge(NODE, [too_long], api_key="k", client=client)
    assert judgment.verdict.verdict == "unique"


async def test_judge_sends_the_pinned_model_and_returns_the_answering_model():
    seen = {}

    def handler(req: httpx.Request):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {"match": {"choice": "c0", "probabilities": {"c0": 0.99, "c1": 0.005, "none_of_these": 0.005}}},
            "usage": {"input_tokens": 800},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        judgment = await jj.judge(NODE, CANDS, api_key="k", client=client)
    assert seen["model"] == AUTO_MERGE_JEV_MODEL
    assert judgment.model == "jev-1.13.0"
    assert judgment.p == 0.99
    assert judgment.verdict.verdict == "duplicate"
    assert judgment.verdict.confidence == "high"
    assert judgment.verdict.primary_canonical_id == "ada@example.com"
    assert judgment.verdict.rationale == (
        "shared email ada@example.com; name/id trigram 0.41, embedding similarity 0.96, Jev p=0.99"
    )
