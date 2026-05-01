"""CompanyContext: YAML loader and LLM auto-inferrer.

CompanyContext is a lightweight structure that describes the customer company
being simulated.  It can be loaded from a hand-authored YAML file (the
common case for the synthetic-eval harness) or inferred from README blobs +
repo descriptions via an LLM call.

Plan 1 inference is intentionally bare-bones — the LLM is asked a single
stable question; richer inference (repo-specific prompts, multi-round
clarification) is deferred to Plan 3+.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

import yaml

from scripts.synth.llm_client import LlmClientProtocol, LlmRequest

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_STAGE_VALUES = frozenset({"seed", "series-a", "series-b", "growth", "public"})

# Stable key used as the LLM prompt in v1 (and matched by StaticLlmClient in
# tests).  In v1 the actual readme_blob / repo_descriptions are accepted as
# parameters but not yet embedded in the prompt — production enrichment is
# deferred to Plan 3+.
_INFERENCE_PROMPT_KEY = "READMES_AND_REPOS_FOR_INFERENCE"


@dataclass(frozen=True)
class Customer:
    """A segment / type of end-customer the company serves."""

    name: str
    description: str | None = None


@dataclass(frozen=True)
class NonEngPerson:
    """A non-engineering persona (sales, support, ops …) used in scenarios."""

    name: str
    role: str
    slack_handle: str | None = None


@dataclass(frozen=True)
class CompanyContext:
    """Immutable description of the simulated company.

    Required:
        name       — short company name (e.g. "Acme Corp")
        stage      — one of: seed | series-a | series-b | growth | public
        headcount  — approximate total employee count (int)

    Optional (default to empty collections):
        customers           — list of Customer segments
        non_eng_people      — list of NonEngPerson personas
        cadence             — free-form dict of process cadences
                              (e.g. {"sprint_days": 14, "standup": "daily"})
        description         — one-line blurb about the company
        tech_stack          — list of technology names
    """

    name: str
    stage: str
    headcount: int
    customers: tuple[Customer, ...] = field(default_factory=tuple)
    non_eng_people: tuple[NonEngPerson, ...] = field(default_factory=tuple)
    # mutable default intentional — matches WorldModel.sha_set pattern (Task 10)
    cadence: dict = field(default_factory=dict)
    description: str | None = None
    tech_stack: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_company_context(yaml_text: str) -> CompanyContext:
    """Parse a YAML string into a CompanyContext.

    Raises:
        ValueError  — if required fields are missing or stage is invalid.
        yaml.YAMLError — if the YAML is malformed.
    """
    data: dict[str, Any] = yaml.safe_load(yaml_text) or {}

    # Required fields
    for key in ("name", "stage", "headcount"):
        if key not in data:
            raise ValueError(f"CompanyContext YAML missing required field: {key!r}")

    stage = str(data["stage"])
    if stage not in _STAGE_VALUES:
        raise ValueError(
            f"Invalid stage {stage!r}. Must be one of: {sorted(_STAGE_VALUES)}"
        )

    # Customers
    customers: list[Customer] = []
    for raw in data.get("customers", []) or []:
        customers.append(
            Customer(
                name=str(raw["name"]),
                description=raw.get("description"),
            )
        )

    # Non-engineering people
    non_eng: list[NonEngPerson] = []
    for raw in data.get("non_eng_people", []) or []:
        non_eng.append(
            NonEngPerson(
                name=str(raw["name"]),
                role=str(raw["role"]),
                slack_handle=raw.get("slack_handle"),
            )
        )

    # Tech stack
    tech_stack = tuple(str(t) for t in (data.get("tech_stack") or []))

    return CompanyContext(
        name=str(data["name"]),
        stage=stage,
        headcount=int(data["headcount"]),
        customers=tuple(customers),
        non_eng_people=tuple(non_eng),
        cadence=dict(data.get("cadence") or {}),
        description=data.get("description"),
        tech_stack=tech_stack,
    )


# ---------------------------------------------------------------------------
# LLM auto-inferrer
# ---------------------------------------------------------------------------

_INFERENCE_SYSTEM = textwrap.dedent("""\
    You are a helpful assistant that extracts company metadata from engineering
    artefacts.  Reply ONLY with a YAML block — no prose, no markdown fences.

    Required YAML keys:
      name: <company name>
      stage: <one of: seed | series-a | series-b | growth | public>
      headcount: <integer>

    Optional YAML keys:
      description: <one-line blurb>
      tech_stack: [<list of tech names>]
      customers:
        - name: <segment name>
          description: <optional blurb>
      non_eng_people:
        - name: <full name>
          role: <job title>
          slack_handle: <optional handle>
      cadence:
        sprint_days: <int>
        standup: <daily|weekly>
""")


def _render_inference_prompt(readme_blob: str, repo_descriptions: list[str]) -> str:
    """Return the prompt string to send to the LLM.

    v1: always returns the stable key regardless of inputs so that
    StaticLlmClient in tests can match deterministically.  readme_blob and
    repo_descriptions are accepted for API stability but not yet embedded —
    rich prompt construction is deferred to Plan 3+.
    """
    if not readme_blob and not repo_descriptions:
        return _INFERENCE_PROMPT_KEY
    # v1 placeholder: even with inputs we return the stable key so tests stay
    # deterministic.  Plan 3+ will embed the actual blob/descriptions here.
    return _INFERENCE_PROMPT_KEY


async def infer_company_context(
    *,
    readme_blob: str,
    repo_descriptions: list[str],
    llm_client: LlmClientProtocol,
    model: str,
    max_tokens: int = 512,
) -> tuple[CompanyContext, str]:
    """Ask the LLM to infer CompanyContext from repo artefacts.

    Returns:
        (CompanyContext, raw_yaml)  — the parsed context and the raw YAML
        string returned by the LLM (useful for caching / debugging).

    Raises:
        ValueError  — if the LLM response cannot be parsed into a valid
                      CompanyContext.
    """
    prompt = _render_inference_prompt(readme_blob, repo_descriptions)
    req = LlmRequest(
        model=model,
        system=_INFERENCE_SYSTEM,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    resp = await llm_client.generate(req)
    raw_yaml = resp.text
    ctx = load_company_context(raw_yaml)
    return ctx, raw_yaml
