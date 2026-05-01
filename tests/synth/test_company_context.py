"""Tests for scripts/synth/company_context.py — Task 19."""

from __future__ import annotations

import pytest

from scripts.synth.company_context import (
    _INFERENCE_PROMPT_KEY,
    CompanyContext,
    Customer,
    NonEngPerson,
    infer_company_context,
    load_company_context,
)
from scripts.synth.llm_client import StaticLlmClient

# ---------------------------------------------------------------------------
# Test 1: load minimal YAML (name / stage / headcount only)
# ---------------------------------------------------------------------------


def test_load_minimal_yaml() -> None:
    yaml_text = """\
name: Acme Corp
stage: series-a
headcount: 42
"""
    ctx = load_company_context(yaml_text)

    assert isinstance(ctx, CompanyContext)
    assert ctx.name == "Acme Corp"
    assert ctx.stage == "series-a"
    assert ctx.headcount == 42
    assert ctx.customers == ()
    assert ctx.non_eng_people == ()
    assert ctx.cadence == {}
    assert ctx.description is None
    assert ctx.tech_stack == ()


# ---------------------------------------------------------------------------
# Test 2: load full YAML (all fields, including Customer + NonEngPerson)
# ---------------------------------------------------------------------------


def test_load_full_yaml() -> None:
    yaml_text = """\
name: BetaCo
stage: growth
headcount: 250
description: "B2B SaaS for fintech teams"
tech_stack:
  - Python
  - TypeScript
  - PostgreSQL
customers:
  - name: Enterprise
    description: "Fortune 500 banks"
  - name: SMB
    description: "Regional credit unions"
non_eng_people:
  - name: Dana Sales
    role: Account Executive
    slack_handle: "@dana"
  - name: Morgan Ops
    role: Operations Lead
cadence:
  sprint_days: 14
  standup: daily
"""
    ctx = load_company_context(yaml_text)

    assert ctx.name == "BetaCo"
    assert ctx.stage == "growth"
    assert ctx.headcount == 250
    assert ctx.description == "B2B SaaS for fintech teams"
    assert ctx.tech_stack == ("Python", "TypeScript", "PostgreSQL")

    assert len(ctx.customers) == 2
    assert ctx.customers[0] == Customer(name="Enterprise", description="Fortune 500 banks")
    assert ctx.customers[1] == Customer(name="SMB", description="Regional credit unions")

    assert len(ctx.non_eng_people) == 2
    assert ctx.non_eng_people[0] == NonEngPerson(
        name="Dana Sales", role="Account Executive", slack_handle="@dana"
    )
    assert ctx.non_eng_people[1] == NonEngPerson(
        name="Morgan Ops", role="Operations Lead", slack_handle=None
    )

    assert ctx.cadence == {"sprint_days": 14, "standup": "daily"}


# ---------------------------------------------------------------------------
# Test 3: infer_company_context — uses StaticLlmClient with canned response
#
# Design note: _render_inference_prompt returns _INFERENCE_PROMPT_KEY when
# both readme_blob and repo_descriptions are empty (v1 behaviour). We pass
# empty inputs so the prompt deterministically equals the key, which is what
# the StaticLlmClient mapping is keyed by.  (See plan bug-fix documentation
# in the task spec and the comment in _render_inference_prompt.)
# ---------------------------------------------------------------------------

_CANNED_YAML = """\
name: Inferred Inc
stage: seed
headcount: 10
description: "Auto-inferred startup"
"""


@pytest.mark.asyncio
async def test_infer_company_context_uses_llm_and_returns_yaml() -> None:
    static_llm = StaticLlmClient({_INFERENCE_PROMPT_KEY: _CANNED_YAML})

    # Pass empty inputs so _render_inference_prompt returns _INFERENCE_PROMPT_KEY.
    # With non-empty inputs the v1 implementation still returns the key (by
    # design), but empty inputs make the intent explicit and guard against any
    # future change that embeds real content in the prompt.
    cc, raw_yaml = await infer_company_context(
        readme_blob="",
        repo_descriptions=[],
        llm_client=static_llm,
        model="claude-opus",
    )

    assert raw_yaml == _CANNED_YAML
    assert isinstance(cc, CompanyContext)
    assert cc.name == "Inferred Inc"
    assert cc.stage == "seed"
    assert cc.headcount == 10
    assert cc.description == "Auto-inferred startup"
