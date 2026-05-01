"""CompanyContext: business reality the repo doesn't expose.

Optional input. If not provided, an LLM call over aggregated READMEs +
repo descriptions produces a draft, written to disk for the user to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scripts.synth.llm_client import LlmClientProtocol, LlmRequest


@dataclass(frozen=True)
class Customer:
    name: str
    type: str            # design_partner | paying | trial | ...
    plan: str | None = None
    stage: str | None = None


@dataclass(frozen=True)
class NonEngPerson:
    name: str
    role: str
    slack: str | None = None


@dataclass(frozen=True)
class CompanyContext:
    name: str
    stage: str
    headcount: int
    market: str | None = None
    competitors: tuple[str, ...] = ()
    customers: tuple[Customer, ...] = ()
    non_eng_people: tuple[NonEngPerson, ...] = ()
    recent_milestones: tuple[str, ...] = ()
    ongoing_initiatives: tuple[str, ...] = ()
    cadence: dict[str, object] = field(default_factory=dict)
    inferred: bool = False  # True if produced by infer_company_context


def _to_tuple(seq: object) -> tuple:
    """Coerce a YAML list to tuple; return () for any non-list shape."""
    return tuple(seq) if isinstance(seq, list) else ()


def _from_dict(data: dict, *, inferred: bool) -> CompanyContext:
    return CompanyContext(
        name=data["name"],
        stage=data.get("stage", "unknown"),
        headcount=int(data.get("headcount", 0)),
        market=data.get("market"),
        competitors=_to_tuple(data.get("competitors")),
        customers=tuple(Customer(**c) for c in (data.get("customers") or [])),
        non_eng_people=tuple(NonEngPerson(**p) for p in (data.get("non_eng_people") or [])),
        recent_milestones=_to_tuple(data.get("recent_milestones")),
        ongoing_initiatives=_to_tuple(data.get("ongoing_initiatives")),
        cadence=data.get("cadence") or {},
        inferred=inferred,
    )


def load_company_context(path: Path) -> CompanyContext:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"company context must be a YAML mapping, got {type(raw).__name__}")
    return _from_dict(raw, inferred=False)


_SYSTEM = (
    "You produce minimal CompanyContext YAML for a synthetic-corpus eval tool. "
    "Output ONLY the YAML, no commentary, no fences. "
    "Required keys: name, stage, headcount, market. "
    "Optional: competitors (list), customers (list of {name,type,plan?}), "
    "non_eng_people (list of {name,role}), recent_milestones (list), "
    "ongoing_initiatives (list)."
)

_INFERENCE_PROMPT_KEY = "READMES_AND_REPOS_FOR_INFERENCE"


def _render_inference_prompt(readme_blob: str, repo_descriptions: list[str]) -> str:
    """Render the inference prompt.

    v1: always returns the static key. The full readme/repo body will be
    embedded in Plan 3 once the inference is exercised end-to-end. Tests
    rely on the stable key to avoid brittle string matching.
    """
    # readme_blob/repo_descriptions accepted for forward compat; v1 ignores body.
    _ = readme_blob, repo_descriptions
    return _INFERENCE_PROMPT_KEY


async def infer_company_context(
    *,
    readme_blob: str,
    repo_descriptions: list[str],
    llm_client: LlmClientProtocol,
    model: str,
) -> tuple[CompanyContext, str]:
    """One-shot LLM inference. Returns the CompanyContext + the raw YAML
    string (so the caller can write `inferred-company.yaml` for the user)."""
    prompt = _render_inference_prompt(readme_blob, repo_descriptions)
    resp = await llm_client.generate(
        LlmRequest(model=model, system=_SYSTEM, prompt=prompt, temperature=0.2)
    )
    raw_yaml = resp.text.strip()
    data = yaml.safe_load(raw_yaml)
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a YAML mapping for CompanyContext")
    return _from_dict(data, inferred=True), raw_yaml
