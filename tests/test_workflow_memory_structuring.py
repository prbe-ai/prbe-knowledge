"""The structuring pass: declared prose in, one clause DRAFT out.

WHY EVERY TEST HERE DRIVES A FAKE `completion`, and why that is not laziness.

`tests/conftest.py` sets `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` to `""`, and
there is no `GOOGLE_API_KEY` in the test environment either. There is no live
model, so a test that asserted "this prose becomes kind=anti_pattern" would be
asserting on whatever the absent provider does, which is nothing. The injected
`completion` seam exists so the PARSING and VALIDATION -- which is all this
module actually is -- can be tested exactly. The only tests that need a real
model are under the `REAL MODEL` banner and are skipped without a key.

NO DATABASE. `structure()` reads nothing and writes nothing, so there is no
`live_db` fixture here. Adding one for symmetry with the classifier's suite
would make every test in this file depend on a running Postgres to exercise
code that never opens a connection.

WHAT THESE TESTS HAVE TO DEFEND:

* EVERY FIELD IS VALID AGAINST THE DDL. `clauses.kind` has a five-value CHECK
  and `binding`/`scope` are `NOT NULL DEFAULT '{}'::jsonb`. A draft carrying an
  off-list kind or a `None` binding does not fail here, it fails at somebody
  else's INSERT with a constraint violation and no context.
* AN OFF-LIST KIND IS REFUSED, NOT REPAIRED. The draft is echoed to a human who
  approves it; a wrong-but-plausible kind survives that review because the human
  is reading the body, and then it is indistinguishable from a real one forever.
* NOTHING DEGRADES SILENTLY WHERE THERE IS NO HONEST DEFAULT. Unlike the
  classifier there is no "unknown" draft, so failures raise `StructuringFailed`
  for an endpoint to turn into a 502-with-explanation.
* THE PROMPT ACTUALLY CARRIES THE PROSE AND THE CONTEXT. A refactor that drops
  the context changes nothing visible -- the model still answers -- so it needs
  a test that reads the constructed prompt.
* THE FAILURE LOG DOES NOT ECHO THE RULE. The secret scan has not run at this
  point in the flow, so anything logged from here may be a credential the author
  pasted.

Run it (no database needed, but conftest pins one anyway):

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:5432/prbe_knowledge_wfmem \
        .venv/bin/pytest tests/test_workflow_memory_structuring.py -q
"""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import structlog.testing

from engine.shared.config import get_settings
from engine.shared.constants import WFMEM_STRUCTURING_MODEL
from engine.shared.llm import LLMError
from engine.shared.wfmem.structuring import (
    BINDING_KEYS,
    MAX_CONTEXT_CHARS,
    REASON_LLM_UNAVAILABLE,
    REASON_UNUSABLE_RESPONSE,
    STRUCTURING_MODEL,
    STRUCTURING_PROMPT_VERSION,
    VALID_KINDS,
    ClauseDraft,
    StructuringFailed,
    structure,
)

#: A real declared rule, in the register somebody actually types into
#: `/set-rule`. Used everywhere so the prompt assertions stay readable.
PROSE = "always open a Probe run before the first GPU step"

#: The declaring context: whatever the client had. ADVISORY, and untrusted --
#: it is client-supplied JSON, so nothing here reads a key out of it.
CONTEXT: dict[str, Any] = {
    "repo": "prbe-knowledge",
    "cwd": "/Users/mahit/Desktop/prbe/prbe-knowledge-worktrees/wfmem-store",
    "recent_tools": ["Bash", "Edit"],
}

WELL_FORMED = json.dumps(
    {
        "kind": "step",
        "body": "Open a Probe run before the first GPU step.",
        "semantic_action": "record_experiment_telemetry",
        "binding": {"asset_refs": ["probe-research"]},
    }
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeCompletion:
    """Stand-in for `engine.shared.llm.acompletion`, recording every call.

    The recorded call is half the point: "the prompt contained the context" and
    "the model that was billed is the registry's" cannot be read off the
    returned `ClauseDraft`.
    """

    def __init__(self, content: str | None = None, raises: BaseException | None = None) -> None:
        self._content = content
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )

    @property
    def prompt(self) -> str:
        """Everything the model was shown on the most recent call."""
        assert self.calls, "the LLM was never called"
        return "\n".join(str(m.get("content", "")) for m in self.calls[-1]["messages"])


# --------------------------------------------------------------------------
# The contract other tasks are being written against
# --------------------------------------------------------------------------


def test_prompt_version_is_pinned() -> None:
    assert STRUCTURING_PROMPT_VERSION == "1"


def test_the_valid_kinds_are_exactly_the_check_constraints_five() -> None:
    assert {"step", "check", "exception", "anti_pattern", "asset"} == VALID_KINDS


def test_the_valid_kinds_do_not_drift_from_the_migration() -> None:
    """Read out of the DDL, not remembered.

    The CHECK is the source of truth. If somebody adds a sixth kind to a later
    migration and not here, the structurer can never emit it and the new kind is
    dead on arrival; if somebody removes one, every draft carrying it 500s at
    the INSERT. Neither has a symptom in this module without this test.
    """
    migration = (
        Path(__file__).resolve().parents[1]
        / "db/migrations/versions/20260819_0110_workflow_memory_store.py"
    ).read_text()
    check = re.search(r"CHECK \(kind IN \(([^)]*)\)\)", migration)
    assert check is not None, "the kind CHECK constraint moved; this guard needs updating"
    assert set(re.findall(r"'([a-z_]+)'", check.group(1))) == VALID_KINDS


def test_the_draft_is_frozen() -> None:
    """It is what gets echoed to a human for approval.

    A caller that "fixes up" the kind between the draft and the echo makes the
    thing the person approved differ from the thing that gets written, which is
    the one property the confirmation step exists to provide.
    """
    draft = ClauseDraft(
        kind="step", body="x", semantic_action=None, binding={}, scope={}
    )
    with pytest.raises(FrozenInstanceError):
        draft.kind = "asset"  # type: ignore[misc]


def test_the_model_comes_from_the_house_registry() -> None:
    """The alias names the role; `constants.py` is where model spend is audited."""
    assert STRUCTURING_MODEL == WFMEM_STRUCTURING_MODEL


# --------------------------------------------------------------------------
# A well-formed response
# --------------------------------------------------------------------------


async def test_a_well_formed_response_maps_to_a_clause_draft() -> None:
    completion = FakeCompletion(content=WELL_FORMED)
    draft = await structure(PROSE, CONTEXT, completion=completion)

    assert draft == ClauseDraft(
        kind="step",
        body="Open a Probe run before the first GPU step.",
        semantic_action="record_experiment_telemetry",
        binding={"asset_refs": ["probe-research"]},
        scope={},
    )
    assert completion.calls[0]["model"] == STRUCTURING_MODEL


async def test_binding_and_scope_are_empty_dicts_not_none() -> None:
    """`clauses.binding` and `clauses.scope` are `NOT NULL DEFAULT '{}'::jsonb`.

    `None` is not a null-able absence there, it is a constraint violation at
    somebody else's INSERT -- so the absence has to be spelled `{}` here, where
    the DDL is in view.
    """
    completion = FakeCompletion(content='{"kind": "check", "body": "Smoke before eval."}')
    draft = await structure(PROSE, CONTEXT, completion=completion)

    assert draft.binding == {}
    assert draft.scope == {}
    assert draft.binding is not None and draft.scope is not None


async def test_semantic_action_is_optional_and_nulls_cleanly() -> None:
    """`clauses.semantic_action` IS nullable, so "we do not know" is representable."""
    for content in (
        '{"kind": "step", "body": "b"}',
        '{"kind": "step", "body": "b", "semantic_action": null}',
        '{"kind": "step", "body": "b", "semantic_action": "   "}',
        '{"kind": "step", "body": "b", "semantic_action": 7}',
    ):
        draft = await structure(PROSE, CONTEXT, completion=FakeCompletion(content=content))
        assert draft.semantic_action is None, content


async def test_scope_is_empty_even_when_the_model_proposes_one() -> None:
    """v0 scope is workspace-wide and narrowing is the explicit human act.

    A model-proposed narrowing arrives inside a draft a person is about to
    rubber-stamp, and the thing it silently narrows is WHO SEES THE RULE.
    """
    completion = FakeCompletion(
        content='{"kind": "step", "body": "b", "scope": {"repo": "prbe-knowledge"}}'
    )
    draft = await structure(PROSE, CONTEXT, completion=completion)
    assert draft.scope == {}


# --------------------------------------------------------------------------
# Binding: the three known keys, and nothing else
# --------------------------------------------------------------------------


async def test_the_three_binding_keys_survive() -> None:
    completion = FakeCompletion(
        content=json.dumps(
            {
                "kind": "anti_pattern",
                "body": "Never edit the canonical clone; use a worktree.",
                "binding": {
                    "asset_refs": ["prbe-knowledge:scripts/smoke.sh"],
                    "argv_template": "pytest tests/",
                    "cwd_glob": "~/Desktop/prbe/*",
                },
            }
        )
    )
    draft = await structure(PROSE, CONTEXT, completion=completion)
    assert set(draft.binding) == set(BINDING_KEYS)
    assert draft.binding["asset_refs"] == ["prbe-knowledge:scripts/smoke.sh"]
    assert draft.binding["cwd_glob"] == "~/Desktop/prbe/*"


async def test_unknown_binding_keys_are_dropped() -> None:
    """`binding` is unconstrained JSONB, which is exactly why this is filtered.

    `cwd_glob` compiles into a PreToolUse matcher and `asset_refs` into the
    binding-health probe. A key nobody reads is dead weight; a key that LOOKS
    like one of those and is not is a matcher that silently never fires.
    """
    completion = FakeCompletion(
        content=json.dumps(
            {
                "kind": "step",
                "body": "b",
                "binding": {"cwd_globs": "~/x/*", "concrete_text": "...", "asset_refs": ["a"]},
            }
        )
    )
    draft = await structure(PROSE, CONTEXT, completion=completion)
    assert draft.binding == {"asset_refs": ["a"]}


@pytest.mark.parametrize(
    "binding,expected",
    [
        ('"a string"', {}),
        ("[1, 2, 3]", {}),
        ("null", {}),
        ('{"asset_refs": "not-a-list"}', {"asset_refs": ["not-a-list"]}),
        ('{"asset_refs": ["a", 7, "", "b"]}', {"asset_refs": ["a", "b"]}),
        ('{"argv_template": 7}', {}),
        ('{"cwd_glob": "   "}', {}),
        ('{"asset_refs": []}', {}),
    ],
)
async def test_a_malformed_binding_degrades_to_what_survives(binding: str, expected: dict) -> None:
    """Degrades rather than raising, because `{}` IS a representable answer here.

    That is the whole line between this field and `kind`: the DDL default says
    "no binding" out loud, so dropping a junk value loses nothing a human
    reviewing the echoed draft cannot see. `kind` has no such value.
    """
    completion = FakeCompletion(
        content=f'{{"kind": "step", "body": "b", "binding": {binding}}}'
    )
    draft = await structure(PROSE, CONTEXT, completion=completion)
    assert draft.binding == expected


# --------------------------------------------------------------------------
# Packaging: fences and preambles are what models emit, not errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"kind": "step", "body": "Open a Probe run first."}\n```',
        '```\n{"kind": "step", "body": "Open a Probe run first."}\n```',
        'Sure, here is the structured rule: {"kind": "step", "body": "Open a Probe run first."}',
        '  \n\n{"kind": "step", "body": "Open a Probe run first."}\n',
    ],
)
async def test_the_usual_model_packaging_is_tolerated(content: str) -> None:
    draft = await structure(PROSE, CONTEXT, completion=FakeCompletion(content=content))
    assert draft.kind == "step"
    assert draft.body == "Open a Probe run first."


@pytest.mark.parametrize(
    "kind,expected",
    [
        (" step ", "step"),
        ("STEP", "step"),
        ("Anti-Pattern", "anti_pattern"),
        ("anti pattern", "anti_pattern"),
    ],
)
async def test_a_valid_kind_in_different_clothing_is_normalised(kind: str, expected: str) -> None:
    """Case, spacing and hyphen-vs-underscore are PACKAGING, like a code fence.

    This is not the same act as mapping an off-list word onto its nearest
    neighbour -- these are the same token in different clothes, and the test
    below pins the difference.
    """
    completion = FakeCompletion(content=json.dumps({"kind": kind, "body": "b"}))
    draft = await structure(PROSE, CONTEXT, completion=completion)
    assert draft.kind == expected
    assert draft.kind in VALID_KINDS


# --------------------------------------------------------------------------
# Failure: nothing here has an honest default, so it raises
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
        "I think this is a step, but it depends on what you mean by a run.",
        "[1, 2, 3]",
        '"just a string"',
        "{}",
        '{"body": "Open a Probe run first."}',
        '{"kind": "step"}',
        '{"kind": "step", "body": ""}',
        '{"kind": "step", "body": "   "}',
        '{"kind": "step", "body": 42}',
        '{"kind": "step", "body": null}',
        '{"kind": 7, "body": "b"}',
        '{"kind": null, "body": "b"}',
        '{"kind": ["step"], "body": "b"}',
        '{"kind": "step", "body": "b"',  # truncated at max_tokens
    ],
)
async def test_an_unusable_response_raises_rather_than_drafting(content: str | None) -> None:
    with pytest.raises(StructuringFailed) as caught:
        await structure(PROSE, CONTEXT, completion=FakeCompletion(content=content))
    assert caught.value.reason == REASON_UNUSABLE_RESPONSE


@pytest.mark.parametrize("kind", ["guardrail", "rule", "convention", "antipattern", "steps"])
async def test_an_off_list_kind_is_refused_not_defaulted(kind: str) -> None:
    """The alternative -- default to `step`, or map onto the nearest valid kind --
    produces a draft that is wrong in the one field the human reviewing it is
    least likely to be reading, and once approved it is indistinguishable from a
    real answer to every later reader, the reclassifier included."""
    completion = FakeCompletion(content=json.dumps({"kind": kind, "body": "b"}))
    with pytest.raises(StructuringFailed) as caught:
        await structure(PROSE, CONTEXT, completion=completion)
    assert caught.value.reason == REASON_UNUSABLE_RESPONSE


async def test_the_llm_raising_becomes_a_typed_failure() -> None:
    completion = FakeCompletion(raises=LLMError("gateway 503", status_code=503))

    with structlog.testing.capture_logs() as logs, pytest.raises(StructuringFailed) as caught:
        await structure(PROSE, CONTEXT, completion=completion)

    assert caught.value.reason == REASON_LLM_UNAVAILABLE
    assert [e for e in logs if e["event"] == "wfmem_structuring.llm_failed"], (
        f"a failed structuring call must be visible; captured: {logs}"
    )


async def test_a_non_llm_exception_is_also_wrapped() -> None:
    """`StructuringFailed` is the endpoint's whole contract with this module.

    A `TimeoutError` or a library-internal `TypeError` escaping raw turns a
    502-with-explanation into a 500 with a stack trace.
    """
    completion = FakeCompletion(raises=TimeoutError("read timeout"))
    with pytest.raises(StructuringFailed):
        await structure(PROSE, CONTEXT, completion=completion)


@pytest.mark.parametrize("prose", ["", "   ", "\n\t "])
async def test_blank_prose_is_a_caller_error_and_costs_nothing(prose: str) -> None:
    """ValueError, NOT `StructuringFailed`.

    There is no rule to structure, so nothing about the model has failed and the
    endpoint must not answer 502 and blame it. It also must not be billed for
    discovering that the string was empty.
    """
    completion = FakeCompletion(content=WELL_FORMED)
    with pytest.raises(ValueError):
        await structure(prose, CONTEXT, completion=completion)
    assert completion.calls == []


async def test_the_failure_log_does_not_echo_the_rule_or_the_response() -> None:
    """The secret scan runs on the WRITE path, downstream of here.

    So at this point the prose is unscanned text a human pasted, and the model's
    reply quotes it back. Logging either would put a credential in the log
    aggregator, which is a worse place for it than the database this refused to
    reach.
    """
    prose = "deploy with AKIAIOSFODNN7EXAMPLE and the marker PROSE-MARKER"
    reply = '{"kind": "guardrail", "body": "REPLY-MARKER AKIAIOSFODNN7EXAMPLE"}'

    with structlog.testing.capture_logs() as logs, pytest.raises(StructuringFailed):
        await structure(prose, CONTEXT, completion=FakeCompletion(content=reply))

    rendered = json.dumps(logs, default=str)
    assert "PROSE-MARKER" not in rendered
    assert "REPLY-MARKER" not in rendered
    assert "AKIAIOSFODNN7EXAMPLE" not in rendered
    # The off-list token itself IS logged: it is the one fact that explains the
    # refusal, and a bare enum word is not a credential.
    assert "guardrail" in rendered


async def test_the_llm_failure_log_does_not_echo_the_provider_message() -> None:
    """A provider's error text is not guaranteed to be free of the request.

    A 400 that quotes the offending message back is routine, and the message
    here is unscanned prose. So the log carries the error CLASS and the model,
    not `str(exc)` -- which is a deliberate break from the classifier, whose
    input is a search string rather than a paste.
    """
    completion = FakeCompletion(
        raises=LLMError("invalid request: messages.0.content: PROSE-MARKER AKIAIOSFODNN7EXAMPLE")
    )

    with structlog.testing.capture_logs() as logs, pytest.raises(StructuringFailed):
        await structure(PROSE, CONTEXT, completion=completion)

    rendered = json.dumps(logs, default=str)
    assert "PROSE-MARKER" not in rendered
    assert "AKIAIOSFODNN7EXAMPLE" not in rendered
    assert "LLMError" in rendered


async def test_the_failure_message_is_useful_to_a_human() -> None:
    """The endpoint renders this. "structuring failed" alone helps nobody."""
    with pytest.raises(StructuringFailed) as caught:
        await structure(PROSE, CONTEXT, completion=FakeCompletion(content="not json"))
    assert REASON_UNUSABLE_RESPONSE in str(caught.value)


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------


async def test_the_prompt_carries_the_prose_and_the_context() -> None:
    """Dropping the context breaks nothing visible -- the model still answers.

    Which is exactly why it needs a test that reads the constructed prompt
    rather than the returned draft.
    """
    completion = FakeCompletion(content=WELL_FORMED)
    await structure(PROSE, CONTEXT, completion=completion)

    prompt = completion.prompt
    assert PROSE in prompt
    assert "prbe-knowledge" in prompt
    assert "recent_tools" in prompt
    assert "wfmem-store" in prompt


async def test_the_prompt_lists_every_valid_kind() -> None:
    """A kind the model is never shown is a kind it can never return."""
    completion = FakeCompletion(content=WELL_FORMED)
    await structure(PROSE, CONTEXT, completion=completion)
    for kind in VALID_KINDS:
        assert kind in completion.prompt


async def test_the_prompt_names_the_binding_keys() -> None:
    """Anything not in this list gets dropped, so asking for it is the only way
    the model can produce a binding that survives."""
    completion = FakeCompletion(content=WELL_FORMED)
    await structure(PROSE, CONTEXT, completion=completion)
    for key in BINDING_KEYS:
        assert key in completion.prompt


async def test_the_prompt_tells_the_model_not_to_improve_the_rule() -> None:
    """The conservative-rewrite instruction is load-bearing, not decoration.

    `body` becomes authoritative only because a human recognises their own rule
    in the echo and approves it. A polished rewrite of something they did not
    say gets approved anyway -- people approve what looks right -- and the
    confirmation step becomes theatre.
    """
    completion = FakeCompletion(content=WELL_FORMED)
    await structure(PROSE, CONTEXT, completion=completion)
    prompt = completion.prompt.lower()
    assert "do not" in prompt
    assert "recognise" in prompt or "recognize" in prompt


async def test_the_context_is_capped_but_the_prose_is_not() -> None:
    """The context is machine-assembled and can be arbitrarily large without
    anybody intending it. Truncating background degrades gracefully; truncating
    the rule itself produces a draft of half a rule, which is worse than a big
    prompt."""
    long_prose = PROSE + " " + "and also " * 4_000
    completion = FakeCompletion(content=WELL_FORMED)
    await structure(long_prose, {"junk": "x" * (MAX_CONTEXT_CHARS * 4)}, completion=completion)

    prompt = completion.prompt
    assert long_prose in prompt
    assert prompt.count("x") <= MAX_CONTEXT_CHARS


async def test_a_context_that_will_not_serialise_does_not_take_the_call_down() -> None:
    """It is client-supplied and advisory. It must never be the thing that fails."""
    cyclic: dict[str, Any] = {"repo": "prbe-knowledge"}
    cyclic["self"] = cyclic

    completion = FakeCompletion(content=WELL_FORMED)
    draft = await structure(PROSE, cyclic, completion=completion)

    assert draft.kind == "step"
    assert PROSE in completion.prompt


@pytest.mark.parametrize("context", [None, "a string", 7, ["a", "list"]])
async def test_a_context_that_is_not_a_dict_is_tolerated(context: Any) -> None:
    """Advisory input, not trusted structured data -- so it is never destructured
    and a wrong-shaped one cannot fail the call."""
    completion = FakeCompletion(content=WELL_FORMED)
    draft = await structure(PROSE, context, completion=completion)
    assert draft.kind == "step"


# --------------------------------------------------------------------------
# REAL MODEL -- opt-in, and the only thing here that measures the prompt
# --------------------------------------------------------------------------


def _real_model_configured() -> bool:
    """True only when a live model is reachable. See this file's docstring."""
    try:
        settings = get_settings()
    except Exception:  # pragma: no cover -- config problems are not this test's job
        return False
    secret = settings.anthropic_api_key
    key = secret.get_secret_value() if secret is not None else ""
    return bool(key.strip() or settings.llm_gateway_url.strip())


#: The calibration table. When the prompt is retuned, THIS is the evidence --
#: so keep the prose in the register somebody actually types, and keep the
#: expected kind to the ones the wording genuinely forces.
REAL_RULES: tuple[tuple[str, str], ...] = (
    ("always open a Probe run before the first GPU step", "step"),
    ("never edit the canonical clone directly, always use a worktree", "anti_pattern"),
    ("run the live uvicorn smoke against Docker Postgres before claiming done", "check"),
)


@pytest.mark.skipif(
    not _real_model_configured(),
    reason="needs a real model (ANTHROPIC_API_KEY or LLM_GATEWAY_URL); conftest "
    "blanks the keys, so there is nothing to structure prose with",
)
@pytest.mark.parametrize("prose,expected", REAL_RULES, ids=[k for _, k in REAL_RULES])
async def test_real_model_structures_a_declared_rule(prose: str, expected: str) -> None:
    draft = await structure(prose, CONTEXT)
    assert draft.kind == expected, f"{prose!r} -> {draft.kind!r} ({draft.body!r})"
    assert draft.body.strip()
    assert draft.scope == {}
