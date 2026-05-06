# Validator Regen Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deferred TODO at `scripts/synth/scenarios.py:202-211` with a unified per-doc regen loop so plot archetypes (`incident`, `launch`, `big_refactor`) survive the strict validator in real-LLM mode instead of getting silently dropped.

**Architecture:** A new `scripts/synth/regen.py` module owns regen orchestration: failure-context formatting, surgical doc-level splicing, and the per-scenario round loop. The existing `LLMWriter` gets a new `regenerate()` entry point (full-scenario context, not persona-scoped). `run_scenarios` swaps the `continue` for `regen_loop()`, emits per-round + terminal structured logs, and falls through to drop only after exhaustion. The loop treats Pass 1 token violations and Pass 2 issue strings symmetrically — both feed into a single combined failure-context block in one regen prompt template.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio, `unittest.mock.AsyncMock`, `structlog` via `shared.logging`, the existing `LlmClientProtocol` / `MockLlmClient`. No new dependencies.

**Reads:**
- Handoff: `docs/superpowers/specs/2026-05-05-validator-regen-loop-design.md` (decisions Q1–Q6 are now locked)
- Plan 3 spec: `docs/superpowers/specs/2026-05-02-synthetic-narrative-layer-design.md`
- Drop site: `scripts/synth/scenarios.py:202-211`
- Validator: `scripts/synth/validator.py::validate`, `scripts/synth/llm/validator_pass2.py::validate_pass2`
- Writer: `scripts/synth/llm/writer.py::LLMWriter.write`

**Locked decisions (diverges from handoff in three places):**
- Q1 per-doc regen ✓ (handoff default)
- Q2 full prior scenario + explicit fix instruction ✓ (handoff default)
- **Q3 round budget = 3, profile-configurable** (handoff said 2)
- Q4 termination = drop scenario + structured log ✓ (handoff default), **with both per-round and terminal events including survival info**
- **Q5 always-on with `--no-regen` opt-out flag** (handoff said no flag)
- Q6 Pass 2 retry deferred ✓ — subsumed by unified loop
- **Pass 1 / Pass 2 strategy:** unified — one regen prompt handles both classes of violation

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `scripts/synth/regen.py` | **new** | `format_failure_context`, `splice_regenerated`, `regen_loop`, `RegenResult`, `RoundReport` |
| `scripts/synth/llm/writer.py` | modify | Add `LLMWriter.regenerate()` method (full-scenario context) |
| `scripts/synth/llm/prompts/writer_regen.txt` | **new** | Single regen template, source-agnostic, takes `{source}` placeholder |
| `scripts/synth/profile.py` | modify | Parse + validate `regen.max_rounds` from raw YAML |
| `scripts/synth/scenarios.py` | modify | Replace lines 202-211 with `regen_loop()` invocation; add `regen_enabled` parameter |
| `scripts/synth/cli.py` | modify | Add `--no-regen` flag; thread through to `run_scenarios` |
| `tests/synth/test_regen_format.py` | **new** | Unit tests for `format_failure_context` |
| `tests/synth/test_regen_splice.py` | **new** | Unit tests for `splice_regenerated` |
| `tests/synth/test_regen_loop.py` | **new** | Unit tests for `regen_loop` (mock writer + mock validator) |
| `tests/synth/test_llm_writer.py` | extend | Add tests for `LLMWriter.regenerate()` |
| `tests/synth/test_profile.py` | extend or new | Test `regen.max_rounds` parsing + defaults |
| `tests/synth/test_scenarios.py` | extend | Test that scenarios.py wires regen on Pass 1 / Pass 2 failure |
| `tests/synth/test_regen_observability.py` | **new** | Pin per-round + terminal log shapes |
| `docs/superpowers/specs/2026-05-05-validator-regen-loop-design.md` | modify | Add "Status update 2026-05-05" header noting decisions locked + plan adopted |
| `scripts/synth/dev/seed_to_customer.sh` | **new** | Operator script: source `.env`, generate temp profile, `synth init` + `synth run --integrate` against a chosen customer id |

---

## Out of Scope (separate work items, do not bundle)

- Re-recording `scripts/synth/canonical/v1/raw/` with plot archetypes — separate "Plan 4 V1.5" effort, requires manual LLM keys + budget. Mentioned in handoff "What this unblocks."
- New plot archetypes (`PERF_REGRESSION`, `DEPENDENCY_BUMP`, `CUSTOMER_ESCALATION`).
- Pass 2 retry as a separate axis (Q6 deferred).
- Cost-ceiling enforcement (Q5 deferred).
- Per-tenant `WorldModel` parameterization.

---

## Task 1: Profile schema for `regen.max_rounds`

**Files:**
- Modify: `scripts/synth/profile.py:30-39` (Profile dataclass), end of `load_profile`
- Test: `tests/synth/test_profile.py` (extend if exists, else create)

- [ ] **Step 1: Write the failing test**

If `tests/synth/test_profile.py` does not exist, create it with:

```python
"""Tests for Profile YAML loader, including regen config."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.synth.profile import Profile, ProfileError, load_profile


def _write_profile(tmp_path: Path, extra: str = "") -> Path:
    body = (
        "customer_id: cust-eval-test\n"
        "preset: small\n"
        "seed: 1\n"
        "repos:\n"
        "  - https://github.com/acme/repo\n"
        f"{extra}"
    )
    p = tmp_path / "profile.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_profile_default_regen_max_rounds(tmp_path: Path) -> None:
    profile = load_profile(_write_profile(tmp_path))
    assert profile.regen_max_rounds == 3


def test_load_profile_explicit_regen_max_rounds(tmp_path: Path) -> None:
    profile = load_profile(_write_profile(tmp_path, "regen:\n  max_rounds: 5\n"))
    assert profile.regen_max_rounds == 5


def test_load_profile_regen_max_rounds_must_be_int(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="regen.max_rounds"):
        load_profile(_write_profile(tmp_path, "regen:\n  max_rounds: 'three'\n"))


def test_load_profile_regen_max_rounds_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="regen.max_rounds"):
        load_profile(_write_profile(tmp_path, "regen:\n  max_rounds: 0\n"))
```

If the file exists, append the four tests above (skipping the imports/helper if duplicates would occur — reuse existing fixtures).

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/synth/test_profile.py -v
```

Expected: FAIL with `AttributeError: 'Profile' object has no attribute 'regen_max_rounds'`.

- [ ] **Step 3: Add `regen_max_rounds` to Profile dataclass**

Modify `scripts/synth/profile.py` — replace the `Profile` dataclass (currently lines 30-39):

```python
@dataclass(frozen=True)
class Profile:
    customer_id: str
    repos: tuple[RepoSpec, ...]
    preset: str
    seed: int
    archetypes: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    company_context_path: Path | None = None
    regen_max_rounds: int = 3
    raw: dict = field(default_factory=dict)  # full YAML for plan 3 to consume
```

- [ ] **Step 4: Parse + validate `regen.max_rounds` in `load_profile`**

In `scripts/synth/profile.py`, after the `seed` validation (currently around line 100) and before `cc = raw.get("company_context")`, insert:

```python
    regen_raw = raw.get("regen") or {}
    if not isinstance(regen_raw, dict):
        raise ProfileError(
            f"regen must be a YAML mapping, got {type(regen_raw).__name__}"
        )
    regen_max_rounds = regen_raw.get("max_rounds", 3)
    if isinstance(regen_max_rounds, bool) or not isinstance(regen_max_rounds, int):
        raise ProfileError(
            f"regen.max_rounds must be an integer, got {type(regen_max_rounds).__name__}: {regen_max_rounds!r}"
        )
    if regen_max_rounds < 1:
        raise ProfileError(
            f"regen.max_rounds must be >= 1, got {regen_max_rounds}"
        )
```

Then update the `return Profile(...)` call to pass `regen_max_rounds=regen_max_rounds`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/synth/test_profile.py -v
```

Expected: all four new tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/synth/profile.py tests/synth/test_profile.py
git commit -m "feat(synth): add regen.max_rounds to Profile (default 3)"
```

---

## Task 2: `format_failure_context` utility

**Files:**
- Create: `scripts/synth/regen.py`
- Test: `tests/synth/test_regen_format.py`

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_regen_format.py`:

```python
"""Tests for format_failure_context — converts validator violations into a
single human-readable block to inject into the regen prompt."""

from __future__ import annotations

from scripts.synth.llm.validator_pass2 import Pass2Result, Pass2Violation
from scripts.synth.regen import format_failure_context
from scripts.synth.validator import Violation


def test_format_pass1_only() -> None:
    pass1 = (
        Violation(doc_id="d1", out_of_world=("auto-scaling", "rate-limited")),
    )
    text = format_failure_context(
        pass1_violations=pass1,
        pass2_result=None,
        target_doc_id="d1",
    )
    assert "out-of-world tokens" in text
    assert "auto-scaling" in text
    assert "rate-limited" in text


def test_format_pass2_only() -> None:
    pass2 = Pass2Result(
        passed=False,
        violations=(Pass2Violation(doc_id="d1", issue="root_cause contradicts d0"),),
    )
    text = format_failure_context(
        pass1_violations=(),
        pass2_result=pass2,
        target_doc_id="d1",
    )
    assert "consistency issue" in text
    assert "root_cause contradicts d0" in text


def test_format_combined_pass1_and_pass2() -> None:
    pass1 = (Violation(doc_id="d1", out_of_world=("kubelet",)),)
    pass2 = Pass2Result(
        passed=False,
        violations=(Pass2Violation(doc_id="d1", issue="contradicts d0"),),
    )
    text = format_failure_context(
        pass1_violations=pass1,
        pass2_result=pass2,
        target_doc_id="d1",
    )
    assert "kubelet" in text
    assert "contradicts d0" in text


def test_format_filters_to_target_doc() -> None:
    pass1 = (
        Violation(doc_id="d0", out_of_world=("foo",)),
        Violation(doc_id="d1", out_of_world=("bar",)),
    )
    text = format_failure_context(
        pass1_violations=pass1,
        pass2_result=None,
        target_doc_id="d1",
    )
    assert "bar" in text
    assert "foo" not in text


def test_format_empty_when_no_violations() -> None:
    text = format_failure_context(
        pass1_violations=(),
        pass2_result=None,
        target_doc_id="d1",
    )
    assert text == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/synth/test_regen_format.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.synth.regen'`.

- [ ] **Step 3: Create the regen module with `format_failure_context`**

Create `scripts/synth/regen.py`:

```python
"""Validator regen loop — orchestrates per-doc regeneration when a plot
scenario fails strict validation.

Public surface:
- format_failure_context: build the failure block for a regen prompt
- splice_regenerated: merge regenerated text back into a docs tuple
- regen_loop: per-scenario async orchestrator (max N rounds)
- RegenResult, RoundReport: result shapes for callers + observability

Decisions (locked from 2026-05-05 handoff):
- Per-doc regen, not per-scenario.
- Pass 1 + Pass 2 violations feed a single unified prompt.
- Default round budget 3 (configurable via Profile.regen_max_rounds).
- On exhaustion: drop scenario, terminal log includes survival info.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.synth.llm.validator_pass2 import Pass2Result
from scripts.synth.validator import Violation


def format_failure_context(
    *,
    pass1_violations: tuple[Violation, ...],
    pass2_result: Pass2Result | None,
    target_doc_id: str,
) -> str:
    """Render Pass 1 + Pass 2 violations for `target_doc_id` as a prompt block.

    Returns "" when neither pass flagged the target. The string is meant to
    be interpolated into writer_regen.txt as the `{failure_context}` field.
    """
    lines: list[str] = []

    pass1_for_doc = [v for v in pass1_violations if v.doc_id == target_doc_id]
    if pass1_for_doc:
        tokens = sorted({t for v in pass1_for_doc for t in v.out_of_world})
        lines.append(
            "Pass 1 (out-of-world tokens): "
            + ", ".join(tokens)
            + " — these names are NOT in the WorldModel allowlist. "
            "Replace them or rephrase to avoid them."
        )

    if pass2_result is not None:
        pass2_for_doc = [v for v in pass2_result.violations if v.doc_id == target_doc_id]
        for v in pass2_for_doc:
            lines.append(f"Pass 2 (consistency issue): {v.issue}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/synth/test_regen_format.py -v
```

Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/regen.py tests/synth/test_regen_format.py
git commit -m "feat(synth): add format_failure_context utility for regen prompt assembly"
```

---

## Task 3: Regen prompt template

**Files:**
- Create: `scripts/synth/llm/prompts/writer_regen.txt`

- [ ] **Step 1: Create the prompt template**

Create `scripts/synth/llm/prompts/writer_regen.txt`:

```
A document in this scenario failed strict validation and must be regenerated. Replace the failing document while keeping every other document in the scenario UNCHANGED. Do not invent names not in the allowlists.

Scenario: {scenario_summary}

Full prior scenario (every doc, in chronological order):
{full_scenario_view}

Document to regenerate:
- Doc id: {target_doc_id}
- Source: {target_source}
- Channel: {target_channel}
- Personas: {target_personas}
- Original text (for reference; this is what failed):
{original_text}

Validation feedback you MUST address:
{failure_context}

Allowed services: {allowed_services}
Allowed people: {allowed_people}
Allowed channels: {allowed_channels}
Instance timestamp: {instance_ts}

Generate a replacement {target_source} document. Preserve the same scenario thread (cast, services, root cause, decisions) so the doc still fits beside the others. Output ONLY the document body — no preface, no commentary, no JSON wrapping.
```

- [ ] **Step 2: Verify the file is in place**

```bash
test -f scripts/synth/llm/prompts/writer_regen.txt && echo OK
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/synth/llm/prompts/writer_regen.txt
git commit -m "feat(synth): add writer_regen.txt — unified regen prompt template"
```

---

## Task 4: `LLMWriter.regenerate()` entry point

**Files:**
- Modify: `scripts/synth/llm/writer.py` (extend `LLMWriter` class)
- Test: `tests/synth/test_llm_writer.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/synth/test_llm_writer.py`:

```python
@pytest.mark.asyncio
async def test_regenerate_returns_text_and_uses_regen_template(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "writer_regen.txt").write_text(
        "REGEN | scenario={scenario_summary} | full={full_scenario_view} | "
        "target_doc_id={target_doc_id} | target_source={target_source} | "
        "target_channel={target_channel} | target_personas={target_personas} | "
        "original={original_text} | failure_context={failure_context} | "
        "services={allowed_services} | people={allowed_people} | "
        "channels={allowed_channels} | ts={instance_ts}"
    )

    captured: dict[str, str] = {}

    async def _generate(req):
        captured["prompt"] = req.prompt
        return MagicMock(text="REGENERATED SLACK BODY")

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=_generate)

    world = _make_world()
    company_ctx = _make_company_ctx()
    spec = _make_spec()
    writer = LLMWriter(client=mock_client, model="claude-sonnet-4-6", prompts_dir=prompts_dir)

    target = SynthDoc(
        id="scn-incident-slack-0",
        source=Source.SLACK,
        source_event_id="scn-incident-slack-0",
        text="ORIGINAL slack body that mentioned auto-scaling",
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents",
        page_id=None,
        thread_parent_id=None,
        scenario_id=spec.id,
        archetype="INCIDENT",
        personas=("gh:bob",),
        services_mentioned=("payments",),
        priority=20,
    )
    other = _make_prior_doc(
        "scn-incident-notion-0",
        Source.NOTION,
        datetime(2026, 4, 12, 14, 30, 0, tzinfo=UTC),
        "gh:alice",
    )

    result = await writer.regenerate(
        spec=spec,
        target_doc=target,
        prior_docs_full=(target, other),
        failure_context="Pass 1 (out-of-world tokens): auto-scaling — replace.",
        world=world,
        company_ctx=company_ctx,
    )

    assert result == "REGENERATED SLACK BODY"
    prompt = captured["prompt"]
    assert "scn-incident-slack-0" in prompt
    assert "slack" in prompt
    assert "auto-scaling" in prompt
    assert "ORIGINAL slack body" in prompt
    assert "scn-incident-notion-0" in prompt  # full_scenario_view includes the other doc


@pytest.mark.asyncio
async def test_regenerate_full_scenario_view_is_not_persona_filtered(tmp_path: Path) -> None:
    """Regen sees ALL docs in the scenario regardless of persona/timestamp.

    This is intentionally different from write(), which persona-filters
    prior_emitted_docs. Regen needs the full scenario to fix cross-doc
    references.
    """
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "writer_regen.txt").write_text(
        "{full_scenario_view}|{target_doc_id}|{target_source}|{target_channel}|"
        "{target_personas}|{original_text}|{failure_context}|{scenario_summary}|"
        "{allowed_services}|{allowed_people}|{allowed_channels}|{instance_ts}"
    )

    captured: dict[str, str] = {}

    async def _generate(req):
        captured["prompt"] = req.prompt
        return MagicMock(text="ok")

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=_generate)

    spec = _make_spec()
    target = SynthDoc(
        id="scn-incident-slack-0",
        source=Source.SLACK,
        source_event_id="scn-incident-slack-0",
        text="orig",
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents",
        page_id=None,
        thread_parent_id=None,
        scenario_id=spec.id,
        archetype="INCIDENT",
        personas=("gh:bob",),
        services_mentioned=("payments",),
        priority=20,
    )
    # A doc whose timestamp is AFTER the target — write() would filter it
    # out of the persona view; regenerate() must keep it.
    later = _make_prior_doc(
        "scn-incident-notion-0",
        Source.NOTION,
        datetime(2026, 4, 12, 15, 0, 0, tzinfo=UTC),
        "gh:alice",
    )
    writer = LLMWriter(
        client=mock_client, model="claude-sonnet-4-6", prompts_dir=prompts_dir
    )

    await writer.regenerate(
        spec=spec,
        target_doc=target,
        prior_docs_full=(target, later),
        failure_context="anything",
        world=_make_world(),
        company_ctx=_make_company_ctx(),
    )

    assert "scn-incident-notion-0" in captured["prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/synth/test_llm_writer.py::test_regenerate_returns_text_and_uses_regen_template tests/synth/test_llm_writer.py::test_regenerate_full_scenario_view_is_not_persona_filtered -v
```

Expected: FAIL with `AttributeError: 'LLMWriter' object has no attribute 'regenerate'`.

- [ ] **Step 3: Add `regenerate()` to LLMWriter**

In `scripts/synth/llm/writer.py`, append the following method to the `LLMWriter` class (after `write`, before end-of-class):

```python
    async def regenerate(
        self,
        spec: ScenarioSpec,
        target_doc: SynthDoc,
        prior_docs_full: tuple[SynthDoc, ...],
        failure_context: str,
        world: WorldModel,
        company_ctx: CompanyContext,
    ) -> str:
        """Regenerate a single failing doc body, given the FULL scenario as context.

        Distinct from `write()`:
        - Uses `writer_regen.txt` (not source-specific templates).
        - prior_docs_full is NOT persona-filtered; regen sees everything in
          the scenario so it can fix cross-doc references.
        - The failing doc itself is included in prior_docs_full at its
          existing position so the LLM can see what it's replacing in
          context. The `original_text` field surfaces it explicitly.
        """
        template_path = self._prompts_dir / "writer_regen.txt"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Regen prompt template not found: {template_path}"
            )
        template = template_path.read_text(encoding="utf-8")

        full_view = "\n---\n".join(
            f"[{d.id} | {d.source.value if hasattr(d.source, 'value') else str(d.source)} "
            f"| {d.occurred_at.isoformat()}]\n{d.text}"
            for d in prior_docs_full
        ) or "(empty scenario)"

        allowed_services = ", ".join(
            sorted({s.qualified for s in world.services} | {s.name for s in world.services})
        )
        allowed_people = ", ".join(
            sorted(
                {p.display_name for p in world.people if p.display_name}
                | {p.gh_username for p in world.people if p.gh_username}
            )
        )
        allowed_channels = ", ".join(sorted(ch.name for ch in world.channels))

        target_source_val = (
            target_doc.source.value
            if hasattr(target_doc.source, "value")
            else str(target_doc.source)
        )

        scenario_summary = (
            f"Scenario: {getattr(spec, 'title', spec.id)}\n"
            f"Summary: {getattr(spec, 'summary', '')}\n"
            f"Cast: {', '.join(spec.cast)}\n"
            f"Services: {', '.join(spec.affected_services)}"
        )

        prompt = template.format(
            scenario_summary=scenario_summary,
            full_scenario_view=full_view,
            target_doc_id=target_doc.id,
            target_source=target_source_val,
            target_channel=target_doc.channel or "N/A",
            target_personas=", ".join(target_doc.personas),
            original_text=target_doc.text,
            failure_context=failure_context,
            allowed_services=allowed_services,
            allowed_people=allowed_people,
            allowed_channels=allowed_channels,
            instance_ts=spec.instance_ts.isoformat(),
        )

        req = LlmRequest(
            model=self._model,
            system=(
                "You are a synthetic document regenerator. The original document "
                "failed strict validation. Output ONLY the replacement document body."
            ),
            prompt=prompt,
            max_tokens=2048,
            temperature=0.0,
        )

        log.info(
            "llm_writer.regenerate",
            scenario_id=spec.id,
            target_doc_id=target_doc.id,
            target_source=target_source_val,
            model=self._model,
        )

        response = await self._client.generate(req)
        return response.text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/synth/test_llm_writer.py -v
```

Expected: all writer tests PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/llm/writer.py tests/synth/test_llm_writer.py
git commit -m "feat(synth): add LLMWriter.regenerate() with full-scenario context"
```

---

## Task 5: `splice_regenerated` helper

**Files:**
- Modify: `scripts/synth/regen.py` (extend)
- Test: `tests/synth/test_regen_splice.py`

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_regen_splice.py`:

```python
"""Tests for splice_regenerated — surgical doc-level replacement."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.regen import splice_regenerated


def _doc(doc_id: str, source: Source, text: str) -> SynthDoc:
    return SynthDoc(
        id=doc_id,
        source=source,
        source_event_id=doc_id,
        text=text,
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents" if source == Source.SLACK else None,
        page_id=doc_id if source == Source.NOTION else None,
        thread_parent_id=None,
        scenario_id="scn-1",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=10,
    )


def test_splice_replaces_only_named_doc_text() -> None:
    docs = (
        _doc("d0", Source.SENTRY, "sentry-orig"),
        _doc("d1", Source.SLACK, "slack-orig"),
        _doc("d2", Source.NOTION, "notion-orig"),
    )
    spliced = splice_regenerated(docs, regenerated_text_by_doc_id={"d1": "slack-NEW"})
    assert spliced[0].text == "sentry-orig"
    assert spliced[1].text == "slack-NEW"
    assert spliced[2].text == "notion-orig"


def test_splice_preserves_doc_count_and_order() -> None:
    docs = tuple(_doc(f"d{i}", Source.SLACK, f"orig-{i}") for i in range(5))
    spliced = splice_regenerated(
        docs,
        regenerated_text_by_doc_id={"d2": "fixed-2", "d4": "fixed-4"},
    )
    assert len(spliced) == 5
    assert tuple(d.id for d in spliced) == ("d0", "d1", "d2", "d3", "d4")
    assert spliced[2].text == "fixed-2"
    assert spliced[4].text == "fixed-4"


def test_splice_preserves_thread_parent_and_other_fields() -> None:
    docs = (
        _doc("d0", Source.SLACK, "parent-orig"),
        SynthDoc(
            id="d1",
            source=Source.SLACK,
            source_event_id="d1",
            text="reply-orig",
            occurred_at=datetime(2026, 4, 12, 14, 5, 0, tzinfo=UTC),
            channel="#incidents",
            page_id=None,
            thread_parent_id="d0",
            scenario_id="scn-1",
            archetype="INCIDENT",
            personas=("gh:bob",),
            services_mentioned=("payments",),
            priority=11,
        ),
    )
    spliced = splice_regenerated(docs, regenerated_text_by_doc_id={"d1": "reply-NEW"})
    assert spliced[1].thread_parent_id == "d0"
    assert spliced[1].source_event_id == "d1"
    assert spliced[1].channel == "#incidents"
    assert spliced[1].priority == 11
    assert spliced[1].text == "reply-NEW"


def test_splice_unknown_doc_id_raises() -> None:
    docs = (_doc("d0", Source.SLACK, "orig"),)
    try:
        splice_regenerated(docs, regenerated_text_by_doc_id={"unknown": "x"})
    except ValueError as e:
        assert "unknown" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_splice_empty_replacement_returns_original_tuple() -> None:
    docs = (_doc("d0", Source.SLACK, "orig"),)
    spliced = splice_regenerated(docs, regenerated_text_by_doc_id={})
    assert spliced == docs
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/synth/test_regen_splice.py -v
```

Expected: FAIL with `ImportError: cannot import name 'splice_regenerated' from 'scripts.synth.regen'`.

- [ ] **Step 3: Implement `splice_regenerated` in `scripts/synth/regen.py`**

Append to `scripts/synth/regen.py`:

```python
from scripts.synth.output.base import SynthDoc
from dataclasses import replace


def splice_regenerated(
    original_docs: tuple[SynthDoc, ...],
    *,
    regenerated_text_by_doc_id: dict[str, str],
) -> tuple[SynthDoc, ...]:
    """Return a new tuple where named docs have new `text`, everything else is identical.

    Preserves doc order, count, and every SynthDoc field (id, source,
    source_event_id, occurred_at, channel, page_id, thread_parent_id,
    scenario_id, archetype, personas, services_mentioned, priority) — only
    `text` changes for entries listed in `regenerated_text_by_doc_id`.

    Raises:
        ValueError: if `regenerated_text_by_doc_id` references a doc id
            that is not present in `original_docs`.
    """
    if not regenerated_text_by_doc_id:
        return original_docs

    known_ids = {d.id for d in original_docs}
    unknown = set(regenerated_text_by_doc_id.keys()) - known_ids
    if unknown:
        raise ValueError(
            f"splice_regenerated: doc id(s) not in original scenario: {sorted(unknown)}"
        )

    out: list[SynthDoc] = []
    for d in original_docs:
        new_text = regenerated_text_by_doc_id.get(d.id)
        if new_text is None:
            out.append(d)
        else:
            out.append(replace(d, text=new_text))
    return tuple(out)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/synth/test_regen_splice.py -v
```

Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/regen.py tests/synth/test_regen_splice.py
git commit -m "feat(synth): add splice_regenerated helper for surgical doc replacement"
```

---

## Task 6: `regen_loop` orchestrator

**Files:**
- Modify: `scripts/synth/regen.py` (extend)
- Test: `tests/synth/test_regen_loop.py`

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_regen_loop.py`:

```python
"""Tests for regen_loop — per-scenario async orchestrator with budget."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    DocSpec,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)
from scripts.synth.llm.validator_pass2 import Pass2Result, Pass2Violation
from scripts.synth.output.base import SynthDoc
from scripts.synth.regen import RegenResult, RoundReport, regen_loop
from scripts.synth.validator import CombinedValidatorResult, Violation


def _archetype() -> Archetype:
    return Archetype(
        name="INCIDENT",
        category=Category.PLOT,
        cadence=Cadence.RARE,
        validator_level=ValidatorLevel.STRICT,
        needs_planner_call=True,
        prompt_template_path=None,
    )


def _spec() -> ScenarioSpec:
    ts = datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC)
    return ScenarioSpec(
        id="scn-1",
        archetype_name="INCIDENT",
        instance_ts=ts,
        cast=("gh:alice",),
        affected_services=("payments",),
        doc_specs=(
            DocSpec(
                id="d0",
                source=Source.SLACK,
                occurred_at=ts,
                channel="#incidents",
                page_section=None,
                text="",
                thread_parent_id=None,
                personas=("gh:alice",),
                services_mentioned=("payments",),
            ),
        ),
        title="x",
        summary="y",
        root_cause="z",
        eval_questions=(),
    )


def _doc(doc_id: str, text: str) -> SynthDoc:
    return SynthDoc(
        id=doc_id,
        source=Source.SLACK,
        source_event_id=doc_id,
        text=text,
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-1",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=10,
    )


@pytest.mark.asyncio
async def test_regen_loop_succeeds_on_round_1_when_initial_passes() -> None:
    """If validator passes on the very first call, regen_loop returns
    succeeded=True with rounds=[].

    NOTE: Callers (run_scenarios) only invoke regen_loop AFTER an initial
    failure, so this case is defensive — but the contract is clear.
    """
    docs = (_doc("d0", "ok"),)

    async def validator(_docs):
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=(),
            should_drop=False,
        )

    writer = MagicMock()
    writer.regenerate = AsyncMock(return_value="should not be called")

    result = await regen_loop(
        spec=_spec(),
        archetype=_archetype(),
        docs=docs,
        max_rounds=3,
        writer=writer,
        validate_fn=validator,
        world=MagicMock(),
        company_ctx=MagicMock(),
    )
    assert result.succeeded is True
    assert result.rounds == []
    assert writer.regenerate.await_count == 0


@pytest.mark.asyncio
async def test_regen_loop_succeeds_after_one_round() -> None:
    """Initial state: d0 fails Pass 1. Round 1: writer regenerates, validator passes."""
    initial_docs = (_doc("d0", "auto-scaling went bad"),)

    call_count = {"n": 0}

    async def validator(_docs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: failing
            return CombinedValidatorResult(
                pass1_violations=(Violation(doc_id="d0", out_of_world=("auto-scaling",)),),
                pass2_result=None,
                failing_doc_ids=("d0",),
                should_drop=True,
            )
        # Subsequent: passing
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=(),
            should_drop=False,
        )

    writer = MagicMock()
    writer.regenerate = AsyncMock(return_value="payments service spiked errors")

    result = await regen_loop(
        spec=_spec(),
        archetype=_archetype(),
        docs=initial_docs,
        max_rounds=3,
        writer=writer,
        validate_fn=validator,
        world=MagicMock(),
        company_ctx=MagicMock(),
    )
    assert result.succeeded is True
    assert len(result.rounds) == 1
    assert result.rounds[0].round_num == 1
    assert result.rounds[0].failing_doc_ids == ("d0",)
    assert result.final_docs[0].text == "payments service spiked errors"
    assert writer.regenerate.await_count == 1


@pytest.mark.asyncio
async def test_regen_loop_exhausts_budget_and_fails() -> None:
    """Validator never passes. Loop tries 3 rounds, then returns succeeded=False."""
    docs = (_doc("d0", "auto-scaling"),)

    async def validator(_docs):
        return CombinedValidatorResult(
            pass1_violations=(Violation(doc_id="d0", out_of_world=("auto-scaling",)),),
            pass2_result=None,
            failing_doc_ids=("d0",),
            should_drop=True,
        )

    writer = MagicMock()
    writer.regenerate = AsyncMock(return_value="still has auto-scaling")

    result = await regen_loop(
        spec=_spec(),
        archetype=_archetype(),
        docs=docs,
        max_rounds=3,
        writer=writer,
        validate_fn=validator,
        world=MagicMock(),
        company_ctx=MagicMock(),
    )
    assert result.succeeded is False
    assert len(result.rounds) == 3
    assert writer.regenerate.await_count == 3
    # never_converged tracking
    assert "d0" in result.never_converged_doc_ids


@pytest.mark.asyncio
async def test_regen_loop_tracks_per_round_passing_and_failing_docs() -> None:
    """Two-doc scenario: d0 always passes; d1 fails round 1, passes round 2.

    Asserts the per-round trajectory captures both passing and failing
    doc ids per the user's 'what went wrong AND what went right' decision.
    """
    docs = (_doc("d0", "ok"), _doc("d1", "auto-scaling"))

    call_count = {"n": 0}

    async def validator(_docs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return CombinedValidatorResult(
                pass1_violations=(Violation(doc_id="d1", out_of_world=("auto-scaling",)),),
                pass2_result=None,
                failing_doc_ids=("d1",),
                should_drop=True,
            )
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=(),
            should_drop=False,
        )

    writer = MagicMock()
    writer.regenerate = AsyncMock(return_value="payments errors spiked")

    result = await regen_loop(
        spec=_spec(),
        archetype=_archetype(),
        docs=docs,
        max_rounds=3,
        writer=writer,
        validate_fn=validator,
        world=MagicMock(),
        company_ctx=MagicMock(),
    )
    assert result.succeeded is True
    assert result.rounds[0].failing_doc_ids == ("d1",)
    assert "d0" in result.rounds[0].passing_doc_ids
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/synth/test_regen_loop.py -v
```

Expected: FAIL with `ImportError: cannot import name 'regen_loop' from 'scripts.synth.regen'`.

- [ ] **Step 3: Implement `regen_loop`, `RegenResult`, `RoundReport` in `scripts/synth/regen.py`**

Append to `scripts/synth/regen.py`:

```python
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.synth.archetypes.base import Archetype, ScenarioSpec
    from scripts.synth.company_context import CompanyContext
    from scripts.synth.llm.writer import LLMWriter
    from scripts.synth.validator import CombinedValidatorResult
    from scripts.synth.world_model import WorldModel


@dataclass(frozen=True)
class RoundReport:
    """One regen round's outcome — fed into structured logs."""
    round_num: int
    failing_doc_ids: tuple[str, ...]
    passing_doc_ids: tuple[str, ...]
    violation_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RegenResult:
    """Outcome of regen_loop for one scenario."""
    succeeded: bool
    final_docs: tuple[SynthDoc, ...]
    rounds: list[RoundReport]
    survived_doc_ids: tuple[str, ...]   # Docs that passed validation in the final state
    never_converged_doc_ids: tuple[str, ...]  # Docs failing in every round attempted


async def regen_loop(
    *,
    spec: ScenarioSpec,
    archetype: Archetype,
    docs: tuple[SynthDoc, ...],
    max_rounds: int,
    writer: LLMWriter,
    validate_fn: Callable[[tuple[SynthDoc, ...]], Awaitable[CombinedValidatorResult]],
    world: WorldModel,
    company_ctx: CompanyContext,
) -> RegenResult:
    """Run up to `max_rounds` regen rounds. Returns RegenResult.

    Each round:
      1. Call validate_fn(current_docs).
      2. If should_drop is False → return RegenResult(succeeded=True, ...).
      3. Else build failure_context per failing doc, call writer.regenerate
         once per failing doc, splice, increment round.
    On budget exhaustion → return RegenResult(succeeded=False, ...).

    The caller (run_scenarios) is expected to handle structured logging
    based on the returned RoundReports + RegenResult fields. This keeps the
    loop pure and easier to unit-test.

    `validate_fn` is injected (not imported) so tests can substitute mocks
    without monkeypatching the validator module.
    """
    current = docs
    rounds: list[RoundReport] = []
    ever_failed: set[str] = set()
    last_failing: tuple[str, ...] = ()

    for round_num in range(1, max_rounds + 1):
        result = await validate_fn(current)

        all_doc_ids = tuple(d.id for d in current)
        failing = result.failing_doc_ids
        passing = tuple(d_id for d_id in all_doc_ids if d_id not in set(failing))

        if not result.should_drop:
            # Success — early exit. Don't append a round report for the
            # passing call (callers can infer success from rounds list len
            # vs final state).
            return RegenResult(
                succeeded=True,
                final_docs=current,
                rounds=rounds,
                survived_doc_ids=all_doc_ids,
                never_converged_doc_ids=(),
            )

        # Failure: record this round, regenerate, splice
        ever_failed.update(failing)
        last_failing = failing
        violation_reasons = _collect_violation_reasons(result)
        rounds.append(
            RoundReport(
                round_num=round_num,
                failing_doc_ids=failing,
                passing_doc_ids=passing,
                violation_reasons=violation_reasons,
            )
        )

        # Regenerate each failing doc
        replacements: dict[str, str] = {}
        for failing_id in failing:
            target = next((d for d in current if d.id == failing_id), None)
            if target is None:
                continue
            failure_context = format_failure_context(
                pass1_violations=result.pass1_violations,
                pass2_result=result.pass2_result,
                target_doc_id=failing_id,
            )
            new_text = await writer.regenerate(
                spec=spec,
                target_doc=target,
                prior_docs_full=current,
                failure_context=failure_context,
                world=world,
                company_ctx=company_ctx,
            )
            replacements[failing_id] = new_text

        current = splice_regenerated(current, regenerated_text_by_doc_id=replacements)

    # Budget exhausted: report what survived and what didn't
    final_passing = tuple(d.id for d in current if d.id not in set(last_failing))
    return RegenResult(
        succeeded=False,
        final_docs=current,
        rounds=rounds,
        survived_doc_ids=final_passing,
        never_converged_doc_ids=tuple(sorted(ever_failed & set(last_failing))),
    )


def _collect_violation_reasons(result: CombinedValidatorResult) -> tuple[str, ...]:
    reasons: list[str] = []
    for v in result.pass1_violations:
        reasons.append(f"{v.doc_id}: out_of_world={list(v.out_of_world)}")
    if result.pass2_result is not None:
        for v in result.pass2_result.violations:
            reasons.append(f"{v.doc_id}: {v.issue}")
    return tuple(reasons)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/synth/test_regen_loop.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/regen.py tests/synth/test_regen_loop.py
git commit -m "feat(synth): add regen_loop orchestrator with per-round trajectory tracking"
```

---

## Task 7: Wire regen into `run_scenarios`

**Files:**
- Modify: `scripts/synth/scenarios.py` (replace lines 202-211; extend signature)
- Test: `tests/synth/test_scenarios.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/synth/test_scenarios.py`:

```python
@pytest.mark.asyncio
async def test_run_scenarios_regen_recovers_failing_plot_doc(monkeypatch) -> None:
    """End-to-end: a plot scenario fails Pass 1 in round 1, succeeds round 2 via regen.

    Asserts the scenario is yielded (not dropped) and that the final doc
    text reflects the regenerated content.
    """
    from scripts.synth.archetypes.base import (
        Archetype,
        Cadence,
        Category,
        DocSpec,
        ScenarioSpec,
        Source,
        ValidatorLevel,
    )
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.scenarios import run_scenarios
    from scripts.synth.validator import CombinedValidatorResult, Violation

    # Build a one-doc plot scenario whose initial text fails Pass 1.
    spec = ScenarioSpec(
        id="scn-incident-1",
        archetype_name="INCIDENT",
        instance_ts=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        cast=("gh:alice",),
        affected_services=("payments",),
        doc_specs=(
            DocSpec(
                id="d0",
                source=Source.SLACK,
                occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
                channel="#incidents",
                page_section=None,
                text="",
                thread_parent_id=None,
                personas=("gh:alice",),
                services_mentioned=("payments",),
            ),
        ),
        title="x", summary="y", root_cause="z", eval_questions=(),
    )

    failing_doc = SynthDoc(
        id="d0",
        source=Source.SLACK,
        source_event_id="d0",
        text="auto-scaling broke",
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-incident-1",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=10,
    )

    incident_archetype = Archetype(
        name="INCIDENT",
        category=Category.PLOT,
        cadence=Cadence.RARE,
        validator_level=ValidatorLevel.STRICT,
        needs_planner_call=True,
        prompt_template_path=None,
    )

    async def fake_plot_builder(**kwargs):
        yield spec, [failing_doc]

    monkeypatch.setattr(
        "scripts.synth.archetypes.library.PLOT_BUILDERS",
        {"INCIDENT": fake_plot_builder},
    )
    monkeypatch.setattr(
        "scripts.synth.archetypes.library.get_active",
        lambda profile, archetype_filter=None: {"INCIDENT": incident_archetype},
    )

    # Mock validator: fail first call, pass second
    call_count = {"n": 0}

    async def fake_validate(docs, world, *, scenario, archetype, pass2_client, pass2_model):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return CombinedValidatorResult(
                pass1_violations=(Violation(doc_id="d0", out_of_world=("auto-scaling",)),),
                pass2_result=None,
                failing_doc_ids=("d0",),
                should_drop=True,
            )
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=(),
            should_drop=False,
        )

    monkeypatch.setattr("scripts.synth.validator.validate", fake_validate)

    # Mock writer with .regenerate that returns clean text
    mock_writer = MagicMock()
    mock_writer.regenerate = AsyncMock(return_value="payments service had errors")
    mock_planner = MagicMock()

    yielded: list[tuple] = []
    profile = _profile({"archetypes": {"INCIDENT": {"count": 1}}})
    company_ctx = MagicMock()
    world = _build_test_world()
    ownership = _ownership_full()
    time_window = TimeWindow(end=datetime(2026, 4, 13, tzinfo=UTC), days=7)

    async for s, doc in run_scenarios(
        world=world,
        ownership=ownership,
        profile=profile,
        time_window=time_window,
        company_ctx=company_ctx,
        planner=mock_planner,
        writer=mock_writer,
        validator_pass2_client=None,
        validator_pass2_model=None,
    ):
        yielded.append((s, doc))

    assert len(yielded) == 1
    assert yielded[0][1].text == "payments service had errors"
    assert mock_writer.regenerate.await_count == 1


@pytest.mark.asyncio
async def test_run_scenarios_regen_disabled_drops_immediately(monkeypatch) -> None:
    """When regen_enabled=False, a failing plot scenario drops on round 1
    without calling writer.regenerate."""
    from scripts.synth.archetypes.base import (
        Archetype, Cadence, Category, DocSpec, ScenarioSpec, Source, ValidatorLevel,
    )
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.scenarios import run_scenarios
    from scripts.synth.validator import CombinedValidatorResult, Violation

    spec = ScenarioSpec(
        id="scn-1", archetype_name="INCIDENT",
        instance_ts=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        cast=("gh:alice",), affected_services=("payments",),
        doc_specs=(DocSpec(
            id="d0", source=Source.SLACK,
            occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
            channel="#incidents", page_section=None, text="",
            thread_parent_id=None, personas=("gh:alice",),
            services_mentioned=("payments",),
        ),),
        title="x", summary="y", root_cause="z", eval_questions=(),
    )
    doc = SynthDoc(
        id="d0", source=Source.SLACK, source_event_id="d0",
        text="auto-scaling", occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents", page_id=None, thread_parent_id=None,
        scenario_id="scn-1", archetype="INCIDENT", personas=("gh:alice",),
        services_mentioned=("payments",), priority=10,
    )
    arch = Archetype(
        name="INCIDENT", category=Category.PLOT, cadence=Cadence.RARE,
        validator_level=ValidatorLevel.STRICT, needs_planner_call=True,
        prompt_template_path=None,
    )

    async def fake_plot_builder(**_kwargs):
        yield spec, [doc]

    monkeypatch.setattr(
        "scripts.synth.archetypes.library.PLOT_BUILDERS",
        {"INCIDENT": fake_plot_builder},
    )
    monkeypatch.setattr(
        "scripts.synth.archetypes.library.get_active",
        lambda profile, archetype_filter=None: {"INCIDENT": arch},
    )

    async def fake_validate(*_a, **_kw):
        return CombinedValidatorResult(
            pass1_violations=(Violation(doc_id="d0", out_of_world=("auto-scaling",)),),
            pass2_result=None, failing_doc_ids=("d0",), should_drop=True,
        )

    monkeypatch.setattr("scripts.synth.validator.validate", fake_validate)

    mock_writer = MagicMock()
    mock_writer.regenerate = AsyncMock(return_value="should not be called")

    yielded: list[tuple] = []
    profile = _profile({"archetypes": {"INCIDENT": {"count": 1}}})
    async for _ in run_scenarios(
        world=_build_test_world(),
        ownership=_ownership_full(),
        profile=profile,
        time_window=TimeWindow(end=datetime(2026, 4, 13, tzinfo=UTC), days=7),
        company_ctx=MagicMock(),
        planner=MagicMock(),
        writer=mock_writer,
        validator_pass2_client=None,
        validator_pass2_model=None,
        regen_enabled=False,
    ):
        yielded.append(_)

    assert yielded == []
    assert mock_writer.regenerate.await_count == 0
```

If `tests/synth/test_scenarios.py` imports `MagicMock` / `AsyncMock` already, reuse them. If not, add `from unittest.mock import AsyncMock, MagicMock` to the imports at the top.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/synth/test_scenarios.py::test_run_scenarios_regen_recovers_failing_plot_doc tests/synth/test_scenarios.py::test_run_scenarios_regen_disabled_drops_immediately -v
```

Expected: FAIL — `regen_enabled` is not yet a parameter; the existing drop site doesn't call `regen_loop`.

- [ ] **Step 3: Update `run_scenarios` to call `regen_loop`**

Modify `scripts/synth/scenarios.py`:

(a) Add `regen_enabled: bool = True` to the signature (after `validator_pass2_model`):

```python
async def run_scenarios(
    world: WorldModel,
    ownership: OwnershipIndex,
    profile: Profile,
    time_window: TimeWindow,
    *,
    archetype_filter: tuple[str, ...] | None = None,
    scenario_limit: int | None = None,
    company_ctx: CompanyContext | None = None,
    planner: LLMPlanner | None = None,
    writer: LLMWriter | None = None,
    validator_pass2_client: LlmClientProtocol | None = None,
    validator_pass2_model: str | None = None,
    regen_enabled: bool = True,
) -> AsyncGenerator[tuple[ScenarioSpec, SynthDoc], None]:
```

(b) Replace the plot drop block (current lines 202-211) with a regen call. The current block ends the loop iteration with `continue`; the replacement either yields the recovered docs, or logs the terminal event and continues:

```python
                    if result.should_drop:
                        if not regen_enabled or writer is None or company_ctx is None:
                            log.warning(
                                "plot_scenario_dropped",
                                scenario_id=spec.id,
                                archetype=name,
                                failing=result.failing_doc_ids,
                                regen_enabled=regen_enabled,
                            )
                            continue

                        from scripts.synth.regen import regen_loop

                        async def _validate_again(
                            d: tuple[SynthDoc, ...],
                            *,
                            _world=world,
                            _spec=spec,
                            _arch=archetype,
                            _pass2_client=validator_pass2_client,
                            _pass2_model=validator_pass2_model,
                        ):
                            return await combined_validate(
                                d,
                                _world,
                                scenario=_spec,
                                archetype=_arch,
                                pass2_client=_pass2_client,
                                pass2_model=_pass2_model,
                            )

                        regen_result = await regen_loop(
                            spec=spec,
                            archetype=archetype,
                            docs=docs,
                            max_rounds=profile.regen_max_rounds,
                            writer=writer,
                            validate_fn=_validate_again,
                            world=world,
                            company_ctx=company_ctx,
                        )

                        for round_report in regen_result.rounds:
                            log.info(
                                "plot_scenario_regen_round",
                                scenario_id=spec.id,
                                archetype=name,
                                round=round_report.round_num,
                                failing_doc_ids=list(round_report.failing_doc_ids),
                                passing_doc_ids=list(round_report.passing_doc_ids),
                                violation_reasons=list(round_report.violation_reasons),
                            )

                        if not regen_result.succeeded:
                            log.warning(
                                "plot_scenario_dropped_after_regen",
                                scenario_id=spec.id,
                                archetype=name,
                                rounds_attempted=len(regen_result.rounds),
                                survived_doc_ids=list(regen_result.survived_doc_ids),
                                never_converged_doc_ids=list(regen_result.never_converged_doc_ids),
                            )
                            continue

                        docs = regen_result.final_docs

                    for doc in docs:
                        yield spec, doc
```

Note: this replaces both the `if result.should_drop:` block AND the original `for doc in docs: yield spec, doc` block at lines 212-213, since the success path now lives inside the regen success branch (and the original-pass branch unchanged docs flow through). For clarity, the rewrite collapses both branches: validation runs, then either regen runs (which mutates `docs` on success) or skips, then the single yield-loop runs once at the bottom.

The simplest concrete diff: starting from the current line 202, the body inside `try:` becomes one continuous block where:
- On `result.should_drop` + regen disabled → log + `continue`
- On `result.should_drop` + regen enabled → run loop; on failure log terminal + `continue`; on success replace `docs` with `regen_result.final_docs`
- Fall through to the existing `for doc in docs: yield spec, doc`

Apply this as a clean rewrite of lines 202-213.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/synth/test_scenarios.py -v
```

Expected: existing tests still pass; both new tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/scenarios.py tests/synth/test_scenarios.py
git commit -m "feat(synth): wire regen_loop into run_scenarios with regen_enabled flag"
```

---

## Task 8: CLI `--no-regen` flag

**Files:**
- Modify: `scripts/synth/cli.py` (`run` subcommand args + call site)

- [ ] **Step 1: Add `--no-regen` flag to the `run` subcommand**

In `scripts/synth/cli.py`, after the `--record-llm` argument (current line 305), insert:

```python
    run.add_argument(
        "--no-regen",
        action="store_true",
        default=False,
        help="Disable validator regen loop. Plot scenarios that fail strict "
        "validation will drop immediately (pre-regen behavior).",
    )
```

- [ ] **Step 2: Pass `regen_enabled` through to `run_scenarios`**

In the same file, locate the `run_scenarios(...)` call (around line 763) and add `regen_enabled=not args.no_regen` as a keyword argument:

```python
        async for spec, doc in run_scenarios(
            world=world,
            ownership=ownership,
            profile=profile,
            time_window=time_window,
            archetype_filter=archetype_filter,
            scenario_limit=scenario_limit,
            company_ctx=company_ctx,
            planner=planner,
            writer=writer,
            validator_pass2_client=validator_pass2_client,
            validator_pass2_model=validator_pass2_model,
            regen_enabled=not args.no_regen,
        ):
```

(Match the existing argument list — only add the new line, don't reformat unchanged lines.)

- [ ] **Step 3: Smoke-test the CLI parses the new flag**

```bash
uv run python -m scripts.synth.cli run --help 2>&1 | grep -A 1 "no-regen"
```

Expected output contains `--no-regen` and the help text.

- [ ] **Step 4: Run the full synth test suite**

```bash
uv run pytest tests/synth/ -v
```

Expected: all tests pass (no regressions).

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/cli.py
git commit -m "feat(synth): add --no-regen CLI flag for opt-out of regen loop"
```

---

## Task 9: Observability shape contract test

**Files:**
- Test: `tests/synth/test_regen_observability.py`

- [ ] **Step 1: Write the test**

Create `tests/synth/test_regen_observability.py`:

```python
"""Pin the structured-log field shapes for regen observability.

These events feed dashboards and operator runbooks; renaming a field is a
breaking change. This test asserts on the exact field set + types so a
careless rename will fail CI.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from scripts.synth.archetypes.base import (
    Archetype, Cadence, Category, DocSpec, ScenarioSpec, Source, ValidatorLevel,
)
from scripts.synth.output.base import SynthDoc
from scripts.synth.scenarios import TimeWindow, run_scenarios
from scripts.synth.validator import CombinedValidatorResult, Violation


@pytest.mark.asyncio
async def test_regen_round_log_shape(monkeypatch, caplog) -> None:
    """plot_scenario_regen_round must include round, failing_doc_ids,
    passing_doc_ids, violation_reasons."""
    # Capture structlog events
    captured: list[dict] = []
    structlog.configure(
        processors=[
            lambda _logger, _meth, ev: captured.append(ev) or ev,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    # ... build minimal scenario that fails round 1, passes round 2 ...
    # (reuse the harness pattern from test_run_scenarios_regen_recovers_failing_plot_doc)

    # Assertions:
    round_events = [e for e in captured if e.get("event") == "plot_scenario_regen_round"]
    assert len(round_events) == 1
    e = round_events[0]
    assert "scenario_id" in e
    assert "archetype" in e
    assert "round" in e and isinstance(e["round"], int)
    assert "failing_doc_ids" in e and isinstance(e["failing_doc_ids"], list)
    assert "passing_doc_ids" in e and isinstance(e["passing_doc_ids"], list)
    assert "violation_reasons" in e and isinstance(e["violation_reasons"], list)


@pytest.mark.asyncio
async def test_regen_terminal_drop_log_shape(monkeypatch) -> None:
    """plot_scenario_dropped_after_regen must include rounds_attempted,
    survived_doc_ids, never_converged_doc_ids."""
    # Build scenario where validator never passes; assert terminal event fields.
    # (mirror harness from test_regen_loop_exhausts_budget_and_fails)

    # Assertions on terminal event:
    # assert "rounds_attempted" in event and isinstance(event["rounds_attempted"], int)
    # assert "survived_doc_ids" in event and isinstance(event["survived_doc_ids"], list)
    # assert "never_converged_doc_ids" in event
```

The harness for both tests is the same monkeypatched fixture used in Task 7 — copy the setup verbatim, build the test scenario, drive `run_scenarios` to completion, then assert on the captured events. Use `caplog` if `shared.logging` is configured to use stdlib `logging`; if structlog is fully native, use a custom processor that appends to a list as shown.

**Verify which path applies:**

```bash
grep -n "structlog\|get_logger" shared/logging.py | head -10
```

Adapt the captured-events fixture to whichever logging backend `shared.logging.get_logger` returns.

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/synth/test_regen_observability.py -v
```

Expected: PASS once the assertions are filled in. If it fails on a missing field, add the field in `scripts/synth/scenarios.py` (the log calls in Task 7).

- [ ] **Step 3: Commit**

```bash
git add tests/synth/test_regen_observability.py
git commit -m "test(synth): pin regen log shapes for plot_scenario_regen_round + dropped_after_regen"
```

---

## Task 10: Update handoff doc with status

**Files:**
- Modify: `docs/superpowers/specs/2026-05-05-validator-regen-loop-design.md`

- [ ] **Step 1: Add a status update section**

At the very top of `docs/superpowers/specs/2026-05-05-validator-regen-loop-design.md`, insert (above the existing `# Validator Regen Loop (handoff)` header):

```markdown
> **Status update 2026-05-05:** Decisions Q1–Q6 locked. Implementation plan
> at `docs/superpowers/specs/2026-05-05-validator-regen-loop-plan.md`.
> Diverges from this handoff in three places: round budget = 3 (not 2),
> `--no-regen` opt-out flag added, terminal log includes "what went right"
> (survived doc ids). Pass 1 / Pass 2 are handled by a single unified
> regen loop, not split.

```

- [ ] **Step 2: Verify both docs are in sync**

```bash
grep -l "max_rounds" docs/superpowers/specs/2026-05-05-validator-regen-loop-*.md
```

Expected: both files match.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-05-validator-regen-loop-design.md
git commit -m "docs(synth): note Q1-Q6 locked, link regen-loop implementation plan"
```

---

## Task 11: Operator script — seed synth docs into a target customer id

**Files:**
- Create: `scripts/synth/dev/seed_to_customer.sh`

**Goal:** A self-contained bash script the operator can paste into a fresh shell. It sources a `.env` file (path supplied), generates a one-off profile YAML for the chosen customer id + preset, runs `synth init` to provision the tenant, then `synth run --integrate` to generate fresh docs (regen loop active by default) and write them through to R2 + the ingestion queue.

**End-to-end flow this script enables:**

```
.env  → source → init customer → run --integrate → docs in R2 + ingestion_queue → worker picks up → retrieval API queryable
```

The script is the smoke harness for the regen loop: real Anthropic + OpenAI keys, plot archetypes survive via regen, output lands in a real customer playground.

- [ ] **Step 1: Create the script directory**

```bash
mkdir -p scripts/synth/dev
```

- [ ] **Step 2: Write the script**

Create `scripts/synth/dev/seed_to_customer.sh`:

```bash
#!/usr/bin/env bash
#
# seed_to_customer.sh — paste-friendly synth-to-customer pipeline.
#
# Sources a .env file, materializes a one-off profile YAML, runs
# `synth init` + `synth run --integrate` so plot archetypes (post-regen-
# loop) land in a target customer's R2 bucket + ingestion queue.
#
# Usage:
#   bash scripts/synth/dev/seed_to_customer.sh \
#     --env ~/path/to/.env.local \
#     --customer cust-eval-mahit-demo \
#     [--preset tiny_test] \
#     [--repo github.com/prbe-ai/prbe-knowledge] \
#     [--repo-local ~/Desktop/prbe/prbe-knowledge] \
#     [--seed 7] \
#     [--time-window 30d] \
#     [--mock-llm | --record-llm] \
#     [--no-regen]
#
# Required env vars in the sourced .env (or already exported):
#   DATABASE_URL, DATABASE_URL_SYNC,
#   R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_REGION,
#   ANTHROPIC_API_KEY  (skip with --mock-llm)
#   OPENAI_API_KEY     (skip with --mock-llm)
#
# Fails loudly on missing env vars and on bad customer_id prefixes.

set -euo pipefail

# --- defaults ----------------------------------------------------------------
ENV_FILE=""
CUSTOMER_ID=""
PRESET="tiny_test"
REPO_URL="github.com/prbe-ai/prbe-knowledge"
REPO_LOCAL=""
SEED="7"
TIME_WINDOW="30d"
MOCK_LLM=""
RECORD_LLM=""
NO_REGEN=""

# --- arg parsing -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)         ENV_FILE="$2"; shift 2 ;;
    --customer)    CUSTOMER_ID="$2"; shift 2 ;;
    --preset)      PRESET="$2"; shift 2 ;;
    --repo)        REPO_URL="$2"; shift 2 ;;
    --repo-local)  REPO_LOCAL="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --time-window) TIME_WINDOW="$2"; shift 2 ;;
    --mock-llm)    MOCK_LLM="--mock-llm"; shift 1 ;;
    --record-llm)  RECORD_LLM="--record-llm"; shift 1 ;;
    --no-regen)    NO_REGEN="--no-regen"; shift 1 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    *)
      echo "error: unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- validation --------------------------------------------------------------
if [[ -z "$ENV_FILE" ]]; then
  echo "error: --env <path> is required" >&2
  exit 2
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: env file not found: $ENV_FILE" >&2
  exit 2
fi
if [[ -z "$CUSTOMER_ID" ]]; then
  echo "error: --customer <id> is required" >&2
  exit 2
fi
case "$CUSTOMER_ID" in
  cust-eval-*|cust-synth-*) ;;
  *)
    echo "error: customer_id must start with 'cust-eval-' or 'cust-synth-' (got: $CUSTOMER_ID)" >&2
    echo "       The synth CLI refuses to operate on production-shaped tenants." >&2
    exit 2
    ;;
esac
if [[ -n "$MOCK_LLM" && -n "$RECORD_LLM" ]]; then
  echo "error: --mock-llm and --record-llm are mutually exclusive" >&2
  exit 2
fi

# --- source .env (any var defined in the file becomes exported) -------------
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# --- env-var sanity check ---------------------------------------------------
need_keys=("DATABASE_URL" "DATABASE_URL_SYNC" "R2_ENDPOINT_URL" "R2_ACCESS_KEY_ID" "R2_SECRET_ACCESS_KEY" "R2_REGION")
if [[ -z "$MOCK_LLM" ]]; then
  need_keys+=("ANTHROPIC_API_KEY" "OPENAI_API_KEY")
fi
missing=()
for k in "${need_keys[@]}"; do
  if [[ -z "${!k:-}" ]]; then
    missing+=("$k")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "error: missing required env vars (sourced from $ENV_FILE): ${missing[*]}" >&2
  exit 3
fi

# --- repo-extraction sanity (init reads the repo to build WorldModel) -------
if [[ -z "$REPO_LOCAL" ]]; then
  echo "warn: --repo-local not set; init will clone $REPO_URL via GITHUB_TOKEN if needed." >&2
fi

# --- compose stack health (best-effort) -------------------------------------
if command -v docker >/dev/null 2>&1; then
  if ! docker compose ps 2>/dev/null | grep -q "Up"; then
    echo "warn: docker compose stack does not appear to be up. Run 'docker compose up -d' first." >&2
  fi
fi

# --- materialize a temp profile --------------------------------------------
PROFILE_DIR="$(mktemp -d -t synth-profile.XXXXXX)"
PROFILE_PATH="$PROFILE_DIR/profile.yaml"

{
  echo "# Generated by scripts/synth/dev/seed_to_customer.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "customer_id: $CUSTOMER_ID"
  echo "preset: $PRESET"
  echo "seed: $SEED"
  echo ""
  echo "repos:"
  if [[ -n "$REPO_LOCAL" ]]; then
    echo "  - url: $REPO_URL"
    echo "    local_path: $REPO_LOCAL"
  else
    echo "  - $REPO_URL"
  fi
  echo ""
  echo "world_model:"
  echo "  min_commits_per_persona: 2"
  echo "  topic_pool_lookback_days: 30"
  echo ""
  echo "time_window:"
  echo "  days: ${TIME_WINDOW%d}"
  echo ""
  echo "regen:"
  echo "  max_rounds: 3"
} >"$PROFILE_PATH"

echo ">> profile written: $PROFILE_PATH"
echo ">> customer:        $CUSTOMER_ID"
echo ">> preset:          $PRESET"
echo ">> regen flag:      ${NO_REGEN:-(enabled)}"
echo ">> llm mode:        ${MOCK_LLM:-${RECORD_LLM:-real}}"
echo ""

# --- run the pipeline -------------------------------------------------------
echo ">> [1/2] synth init"
uv run python -m scripts.synth.cli init --profile "$PROFILE_PATH"

echo ""
echo ">> [2/2] synth run --integrate"
# shellcheck disable=SC2086
uv run python -m scripts.synth.cli run \
  --profile "$PROFILE_PATH" \
  --integrate \
  --time-window "$TIME_WINDOW" \
  $MOCK_LLM $RECORD_LLM $NO_REGEN

echo ""
echo ">> done. Synthetic docs are in R2 + ingestion_queue for $CUSTOMER_ID."
echo ">> Next steps:"
echo "   - Tail the worker:    docker compose logs -f worker"
echo "   - Verify retrieval:   curl your retrieval API with this customer_id"
echo "   - Tear down later:    uv run python -m scripts.synth.cli clean --customer $CUSTOMER_ID"
```

- [ ] **Step 3: Make the script executable**

```bash
chmod +x scripts/synth/dev/seed_to_customer.sh
```

- [ ] **Step 4: Smoke-test arg parsing (no real run)**

```bash
bash scripts/synth/dev/seed_to_customer.sh --help
```

Expected: prints the usage block from the script header (lines 2-40), exit 0.

```bash
bash scripts/synth/dev/seed_to_customer.sh --env /nonexistent --customer prod-bad
```

Expected: exits 2 with "env file not found" (validation runs before customer prefix check), or "customer_id must start with cust-eval- or cust-synth-" if the env file exists.

- [ ] **Step 5: Real end-to-end smoke (operator-only, requires real LLM keys)**

This step is the operator's manual verification — do NOT run during automated CI. Document it for the runbook.

```bash
# Pre-req: docker compose up -d   (postgres + minio healthy)
# Pre-req: a .env file at /path/to/.env with all the required vars
bash scripts/synth/dev/seed_to_customer.sh \
  --env /path/to/.env \
  --customer cust-eval-regen-smoke \
  --preset tiny_test \
  --repo-local ~/Desktop/prbe/prbe-knowledge
```

Expected outcome:
- `synth init` provisions the customer row + R2 bucket.
- `synth run --integrate` generates docs; structured logs include `plot_scenario_regen_round` events on plot scenarios that initially fail validation.
- Plot archetype survival rate > 0% (was 0% pre-regen).
- The ingestion_queue table contains rows tagged with `cust-eval-regen-smoke`.

If plot survival is still 0%, check the `plot_scenario_dropped_after_regen` log — `never_converged_doc_ids` will name which docs the writer couldn't fix.

- [ ] **Step 6: Commit**

```bash
git add scripts/synth/dev/seed_to_customer.sh
git commit -m "feat(synth): add seed_to_customer.sh — env+init+run pipeline for ad-hoc seeding"
```

---

## Final verification

- [ ] **Run the full synth test suite + integration:**

```bash
uv run pytest tests/synth/ -v
```

Expected: green. No regressions in templated archetype tests; the four new test files (`test_regen_format.py`, `test_regen_splice.py`, `test_regen_loop.py`, `test_regen_observability.py`) plus the extensions to `test_llm_writer.py`, `test_scenarios.py`, `test_profile.py` all pass.

- [ ] **Type check (if the project uses one — check for `mypy` / `pyright` config):**

```bash
grep -l "mypy\|pyright\|ruff" pyproject.toml
uv run mypy scripts/synth/regen.py scripts/synth/llm/writer.py scripts/synth/scenarios.py scripts/synth/profile.py scripts/synth/cli.py 2>&1 | tail -10
```

Address any introduced type errors.

- [ ] **Lint:**

```bash
uv run ruff check scripts/synth/regen.py scripts/synth/llm/writer.py scripts/synth/scenarios.py scripts/synth/profile.py scripts/synth/cli.py tests/synth/test_regen_*.py
```

Address any warnings.

- [ ] **Push branch + open PR:**

```bash
git push -u origin feat/synth-validator-regen-loop
gh pr create --title "feat(synth): validator regen loop for plot archetypes" --body "$(cat <<'EOF'
## Summary

Replaces the deferred TODO at `scripts/synth/scenarios.py:202-211` with a per-doc regen loop. Plot archetypes (`incident`, `launch`, `big_refactor`) now survive Pass 1 + Pass 2 strictness in real-LLM mode instead of being silently dropped.

- Per-scenario round budget (default 3, profile-configurable via `regen.max_rounds`).
- Unified prompt handles both Pass 1 (out-of-world tokens) and Pass 2 (consistency issues).
- `--no-regen` opt-out for cost-sensitive runs.
- Per-round + terminal structured logs for observability (`plot_scenario_regen_round`, `plot_scenario_dropped_after_regen`).

Plan: `docs/superpowers/specs/2026-05-05-validator-regen-loop-plan.md`
Handoff (decisions): `docs/superpowers/specs/2026-05-05-validator-regen-loop-design.md`

## Test plan

- [ ] `uv run pytest tests/synth/ -v` green
- [ ] Manual real-LLM smoke against `tiny_test` profile — plot survival > 0% (was 0%)
- [ ] Cost check: per-scenario LLM-call delta within ~50% worst case

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

(Defer push to user confirmation — see Plan 4 V1 process.)

---

## Self-review notes

**Spec coverage:** All eight handoff tasks are covered:
- (1) writer regen entry point → Task 4
- (2) regen prompt file → Task 3
- (3) refactor scenarios.py drop site → Task 7
- (4) splice helper → Task 5
- (5) profile field → Task 1
- (6) tests (unit, mock-LLM, real-LLM) → Tasks 1, 2, 5, 6, 7, 9 (real-LLM is in Final verification, manual)
- (7) observability → Task 9 + Task 7 inline log calls
- (8) re-record canonical → explicitly out of scope (Plan 4 V1.5 follow-up)

**Locked-decision coverage:** Q3 (round budget 3 + configurable) → Task 1; Q5 (`--no-regen` opt-out) → Task 8; Q4 "what-went-right" → `RoundReport.passing_doc_ids` + `RegenResult.survived_doc_ids` (Tasks 6, 7); unified Pass 1 / Pass 2 → `format_failure_context` (Task 2) + single `writer_regen.txt` (Task 3); operator end-to-end smoke harness → Task 11 (`seed_to_customer.sh`).

**Risks called out in handoff and how this plan addresses them:**
- Cost regression — `--no-regen` for ad-hoc opt-out; observability surfaces actual round counts.
- Regen prompt convergence — per-round trajectory log (`failing_doc_ids` + `passing_doc_ids` per round) makes it visible when the writer fixes one doc but breaks another.
- Pass 2 nondeterminism — separate concern, deferred (Q6).
- Cross-doc reference bugs from splicing — `splice_regenerated` only mutates `text`; all IDs and threading fields are preserved (Task 5 has an explicit test for `thread_parent_id`).
- Writer prompt drift — single regen template, source-agnostic, minimizes prompt-engineering surface.
