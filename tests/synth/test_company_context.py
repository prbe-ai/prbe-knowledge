"""CompanyContext: load from YAML, OR infer once from READMEs via LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.synth.company_context import (
    Customer,
    NonEngPerson,
    infer_company_context,
    load_company_context,
)
from scripts.synth.llm_client import StaticLlmClient


def test_load_minimal_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cc.yaml"
    p.write_text(
        """
name: prbe.ai
stage: seed
headcount: 8
""".strip()
    )
    cc = load_company_context(p)
    assert cc.name == "prbe.ai"
    assert cc.stage == "seed"
    assert cc.headcount == 8
    assert cc.customers == ()
    assert cc.non_eng_people == ()


def test_load_full_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cc.yaml"
    p.write_text(
        """
name: acme
stage: series-a
headcount: 25
market: payment infra
competitors: [Stripe, Adyen]
customers:
  - {name: Globex, type: paying, plan: team}
non_eng_people:
  - {name: Sam Park, role: founding GTM}
recent_milestones: [Closed Series A]
ongoing_initiatives: [SOC2 Type 2]
""".strip()
    )
    cc = load_company_context(p)
    assert cc.competitors == ("Stripe", "Adyen")
    [cust] = cc.customers
    assert cust == Customer(name="Globex", type="paying", plan="team")
    [neng] = cc.non_eng_people
    assert neng == NonEngPerson(name="Sam Park", role="founding GTM")
    assert "SOC2 Type 2" in cc.ongoing_initiatives


@pytest.mark.asyncio
async def test_infer_company_context_uses_llm_and_returns_yaml(tmp_path: Path) -> None:
    canned = {
        "READMES_AND_REPOS_FOR_INFERENCE": """
name: inferred-co
stage: seed
headcount: 5
market: dev tools
""".strip()
    }
    static_llm = StaticLlmClient(canned)

    cc, raw_yaml = await infer_company_context(
        readme_blob="x",
        repo_descriptions=["y"],
        llm_client=static_llm,
        model="claude-opus",
    )
    # Inference returns both the dataclass and the raw YAML for inspection
    assert cc.name == "inferred-co"
    assert "inferred-co" in raw_yaml
    assert cc.inferred is True
