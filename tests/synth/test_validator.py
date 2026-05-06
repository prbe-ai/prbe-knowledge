"""Validator Pass 1: name-only WorldModel check + combined validate()."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)
from scripts.synth.output.base import SynthDoc
from scripts.synth.validator import validate, validate_name_only
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


# ---------------------------------------------------------------------------
# Combined validate() tests
# ---------------------------------------------------------------------------

_NAME_ONLY_ARCHETYPE = Archetype(
    name="STANDUP_UPDATE",
    category=Category.RECURRING,
    cadence=Cadence.DAILY,
    sources_used=(Source.SLACK,),
    cast_size=(1, 1),
    needs_planner_call=False,
    validator_level=ValidatorLevel.NAME_ONLY,
)

_STRICT_ARCHETYPE = Archetype(
    name="INCIDENT",
    category=Category.PLOT,
    cadence=Cadence.AD_HOC,
    sources_used=(Source.SLACK, Source.NOTION),
    cast_size=(2, 4),
    needs_planner_call=True,
    validator_level=ValidatorLevel.STRICT,
)

_FAKE_SPEC = ScenarioSpec(
    id="scn-fake-1",
    archetype_name="INCIDENT",
    instance_ts=datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC),
    cast=("gh:alice",),
    affected_services=("payments-api",),
)


async def test_combined_validate_no_violations_name_only_archetype() -> None:
    """NAME_ONLY archetype + clean doc → should_drop=False, pass2_result=None."""
    world = _make_world()
    doc = _make_doc("deployed payments-api to production.")
    result = await validate(
        (doc,),
        world,
        scenario=None,
        archetype=_NAME_ONLY_ARCHETYPE,
        pass2_client=None,
        pass2_model=None,
    )
    assert result.should_drop is False
    assert result.pass1_violations == ()
    assert result.pass2_result is None
    assert result.failing_doc_ids == ()


async def test_combined_validate_pass1_failure_drops_scenario() -> None:
    """Doc with unknown service → pass1 violation → should_drop=True."""
    world = _make_world()
    doc = _make_doc("shipped unknown-service yesterday.")
    result = await validate(
        (doc,),
        world,
        scenario=None,
        archetype=_NAME_ONLY_ARCHETYPE,
        pass2_client=None,
        pass2_model=None,
    )
    assert result.should_drop is True
    assert len(result.pass1_violations) == 1
    assert "doc-test-1" in result.failing_doc_ids


async def test_combined_validate_strict_archetype_no_client_skips_pass2() -> None:
    """STRICT archetype but pass2_client=None → Pass 2 skipped, only Pass 1 runs."""
    world = _make_world()
    doc = _make_doc("deployed payments-api to production.")
    result = await validate(
        (doc,),
        world,
        scenario=_FAKE_SPEC,
        archetype=_STRICT_ARCHETYPE,
        pass2_client=None,
        pass2_model=None,
    )
    # Pass 2 skipped because no client; clean doc means no pass1 violations either
    assert result.should_drop is False
    assert result.pass2_result is None
    assert result.pass1_violations == ()


async def test_validate_name_only_allows_email_prefix_canonical_id_slugs() -> None:
    """Persons with email:foo@bar canonical_ids contribute 'foo' and '@foo' to allowlist."""
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

    now = datetime(2026, 5, 1, tzinfo=UTC)
    world = WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe", sha="abc", default_branch="main"),),
        people=(Person(
            canonical_id="email:alice@example.com",
            gh_username=None,
            display_name="Alice",
            email_aliases=("alice@example.com",),
            role_hint=None,
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=5.0,
        ),),
        services=(Service(
            name="payments",
            qualified="payments",
            repo_url="github.com/prbe-ai/prbe",
            kind=ServiceKind.API,
            description=None,
            owners=(),
            recent_activity=1.0,
            deploy_target=None,
        ),),
        topic_pool=(),
        channels=(ChannelHint(name="#general", suggested_topic=None, related_services=()),),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={"github.com/prbe-ai/prbe": "abc"},
    )
    doc = SynthDoc(
        id="d1", source=Source.SLACK, source_event_id="d1",
        text="Hi @alice, can you check payments?",
        occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
        channel="#general", page_id=None, thread_parent_id=None,
        scenario_id="s1", archetype="STANDUP_UPDATE",
        personas=("email:alice@example.com",), services_mentioned=("payments",),
        priority=10,
    )
    violations = validate_name_only((doc,), world)
    assert violations == ()


async def test_combined_validate_name_only_archetype_skips_pass2_even_with_client() -> None:
    """A NAME_ONLY archetype must skip Pass 2 even when pass2_client is provided.

    This is the dual of test_combined_validate_strict_archetype_no_client_skips_pass2 —
    together they pin down both required conditions for Pass 2 to fire:
    archetype.validator_level == STRICT AND pass2_client is not None.
    """

    class _UncallableClient:
        async def generate_structured(self, *a: object, **kw: object) -> None:
            raise AssertionError("Pass 2 must not run for NAME_ONLY archetype")

        async def generate(self, *a: object, **kw: object) -> None:
            raise AssertionError("Pass 2 must not run for NAME_ONLY archetype")

        async def close(self) -> None:
            pass

    world = _make_world()
    doc = _make_doc("deployed payments-api to production.")
    result = await validate(
        (doc,),
        world,
        scenario=_FAKE_SPEC,
        archetype=_NAME_ONLY_ARCHETYPE,
        pass2_client=_UncallableClient(),  # type: ignore[arg-type]
        pass2_model="claude-sonnet-4-6",
    )
    # validator_level guard prevents Pass 2 from running despite client being provided
    assert result.pass2_result is None
    assert result.should_drop is False
    assert result.failing_doc_ids == ()


# ---------------------------------------------------------------------------
# pass1_advisory mode — demote Pass 1 violations to logging-only for STRICT
# ---------------------------------------------------------------------------


async def test_pass1_advisory_demotes_strict_archetype_pass1_violation() -> None:
    """STRICT archetype + Pass 1 violation + pass1_advisory=True → should_drop=False.

    The pass1_violations field still carries the violation for callers to log,
    but the doc is excluded from failing_doc_ids and does not trigger drop.
    """
    world = _make_world()
    doc = _make_doc("deploying mystery-service after standup.")
    result = await validate(
        (doc,),
        world,
        scenario=_FAKE_SPEC,
        archetype=_STRICT_ARCHETYPE,
        pass2_client=None,
        pass2_model=None,
        pass1_advisory=True,
    )
    assert result.should_drop is False
    assert len(result.pass1_violations) == 1, "violation still surfaced for logging"
    assert "mystery-service" in result.pass1_violations[0].out_of_world
    assert result.failing_doc_ids == (), "demoted Pass 1 doc must not appear in failing_doc_ids"


async def test_pass1_advisory_does_not_demote_name_only_archetype() -> None:
    """NAME_ONLY (templated) archetype + Pass 1 violation + pass1_advisory=True
    → still drops. The flag only affects STRICT plot archetypes; templated
    archetypes never had Pass 1 false-positive issues and keep their gate."""
    world = _make_world()
    doc = _make_doc("deploying mystery-service after standup.")
    result = await validate(
        (doc,),
        world,
        scenario=None,
        archetype=_NAME_ONLY_ARCHETYPE,
        pass2_client=None,
        pass2_model=None,
        pass1_advisory=True,
    )
    assert result.should_drop is True
    assert "doc-test-1" in result.failing_doc_ids


async def test_pass1_advisory_default_is_strict() -> None:
    """Without pass1_advisory=True, Pass 1 violations on STRICT archetypes
    still trigger drop. Belt-and-suspenders against accidentally flipping
    the default."""
    world = _make_world()
    doc = _make_doc("deploying mystery-service after standup.")
    result = await validate(
        (doc,),
        world,
        scenario=_FAKE_SPEC,
        archetype=_STRICT_ARCHETYPE,
        pass2_client=None,
        pass2_model=None,
        # pass1_advisory not passed — must default to False
    )
    assert result.should_drop is True
    assert "doc-test-1" in result.failing_doc_ids
