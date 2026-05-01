"""Validator Pass 1: name-only WorldModel check."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.validator import validate_name_only
from scripts.synth.world_model import (
    ChannelHint,
    Person,
    RepoSummary,
    Service,
    ServiceKind,
    WorldModel,
)


def _make_world() -> WorldModel:
    now = datetime(2026, 5, 1, tzinfo=UTC)
    person = Person(
        canonical_id="gh:alice",
        gh_username="alice",
        display_name="Alice Smith",
        email_aliases=("alice@example.com",),
        role_hint=None,
        repos_active_in=("github.com/prbe-ai/prbe-knowledge",),
        activity_score=10.0,
    )
    service = Service(
        name="payments-api",
        qualified="payments-api",
        repo_url="github.com/prbe-ai/prbe-knowledge",
        kind=ServiceKind.API,
        description=None,
        owners=(),
        recent_activity=5.0,
        deploy_target=None,
    )
    channel = ChannelHint(name="#standup", suggested_topic=None, related_services=())
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe-knowledge", sha="abc", default_branch="main"),),
        people=(person,),
        services=(service,),
        topic_pool=(),
        channels=(channel,),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={"github.com/prbe-ai/prbe-knowledge": "abc"},
    )


def _make_doc(text: str, source: Source = Source.SLACK) -> SynthDoc:
    return SynthDoc(
        id="doc-test-1",
        source=source,
        source_event_id="doc-test-1",
        text=text,
        occurred_at=datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC),
        channel="#standup",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-test-1",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments-api",),
        priority=100,
    )


def test_clean_doc_produces_no_violations() -> None:
    world = _make_world()
    doc = _make_doc("Yesterday: shipped payments-api. Today: @alice reviews auth. Blockers: none.")
    violations = validate_name_only((doc,), world)
    assert violations == ()


def test_fabricated_service_name_flagged() -> None:
    world = _make_world()
    doc = _make_doc("shipped fake-service yesterday.")
    violations = validate_name_only((doc,), world)
    assert len(violations) == 1
    assert "fake-service" in violations[0].out_of_world


def test_third_party_saas_not_flagged() -> None:
    world = _make_world()
    # stripe and aws are in THIRD_PARTY_ALLOWLIST
    doc = _make_doc("integrated with stripe and aws.")
    violations = validate_name_only((doc,), world)
    assert violations == ()


def test_world_model_channel_name_not_flagged() -> None:
    world = _make_world()
    doc = _make_doc("posted to #standup channel.")
    violations = validate_name_only((doc,), world)
    assert violations == ()


def test_world_model_service_name_not_flagged() -> None:
    world = _make_world()
    doc = _make_doc("deployed payments-api to production.")
    violations = validate_name_only((doc,), world)
    assert violations == ()
