"""Validator Pass 1 — name-only WorldModel check.

Extracts proper-noun-shaped tokens from synthetic doc text and verifies
each one appears in the WorldModel's entity set or the third-party allowlist.

This pass is intentionally narrow:
- _TOKEN_RE captures Slack channels (#foo-bar), person mentions (@foo-bar),
  and kebab-cased service names (payments-api).
- False positives (common words that happen to match kebab) are rare in
  templated output; the allowlist handles known SaaS names.
- False negatives (camelCase service names, etc.) are accepted in v1.
  Plan 3's Pass 2 (cheap LLM consistency check) handles the rest.

The validator does NOT raise on violations — it returns them so the caller
(CLI / IngestionWriter) can log and decide whether to abort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.synth.output.base import SynthDoc
from scripts.synth.world_model import WorldModel

if TYPE_CHECKING:
    from scripts.synth.archetypes.base import Archetype, ScenarioSpec
    from scripts.synth.llm.base import LlmClientProtocol
    from scripts.synth.llm.validator_pass2 import Pass2Result

# Matches three token shapes likely to be company-internal references:
#   #channel-name   @person-handle   kebab-service-name (at least two segments)
_TOKEN_RE = re.compile(
    r"#[\w-]+"
    r"|@[\w-]+"
    r"|\b[a-z][a-z0-9-]*-[a-z][a-z0-9-]*\b"
)

# Common third-party SaaS names that are obviously not internal services.
# Lowercase, sorted alphabetically.
THIRD_PARTY_ALLOWLIST: frozenset[str] = frozenset({
    "anthropic",
    "aws",
    "datadog",
    "github",
    "granola",
    "linear",
    "notion",
    "openai",
    "sentry",
    "slack",
    "stripe",
})


@dataclass(frozen=True)
class Violation:
    doc_id: str
    out_of_world: tuple[str, ...]


def _extract_proper_nouns(text: str) -> set[str]:
    """Extract tokens matching _TOKEN_RE, lowercased."""
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def validate_name_only(
    docs: tuple[SynthDoc, ...],
    world: WorldModel,
) -> tuple[Violation, ...]:
    """Check that all proper-noun tokens in docs map to known world entities.

    Allowed token set = WorldModel services + people + channels + third-party.
    Returns a tuple of Violation (one per doc with out-of-world tokens).
    """
    allowed: set[str] = set()

    # Services: both bare name and qualified name
    for svc in world.services:
        allowed.add(svc.name.lower())
        allowed.add(svc.qualified.lower())

    # People: display_name, gh_username, channel-mention forms, and
    # canonical_id slug (e.g. "gh:alice" -> "alice", "@alice";
    # "email:alice@example.com" -> "alice", "@alice").
    # The slug covers locally-extracted repos where gh_username may be None.
    for person in world.people:
        if person.display_name:
            allowed.add(person.display_name.lower())
        if person.gh_username:
            allowed.add(person.gh_username.lower())
            allowed.add(f"@{person.gh_username.lower()}")
        if ":" in person.canonical_id:
            rest = person.canonical_id.split(":", 1)[1].lower()
            # For email-based IDs like "alice@example.com", use only the local part.
            slug = rest.split("@")[0]
            if slug:
                allowed.add(slug)
                allowed.add(f"@{slug}")

    # Channels: name as-is (already has # prefix)
    for ch in world.channels:
        allowed.add(ch.name.lower())

    # Third-party SaaS
    for name in THIRD_PARTY_ALLOWLIST:
        allowed.add(name)

    violations: list[Violation] = []
    for doc in docs:
        mentioned = _extract_proper_nouns(doc.text)
        out_of_world = mentioned - allowed
        if out_of_world:
            violations.append(
                Violation(doc_id=doc.id, out_of_world=tuple(sorted(out_of_world)))
            )

    return tuple(violations)


# ---------------------------------------------------------------------------
# Combined validator (Pass 1 + optional Pass 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombinedValidatorResult:
    pass1_violations: tuple[Violation, ...]
    pass2_result: Pass2Result | None
    failing_doc_ids: tuple[str, ...]
    should_drop: bool


async def validate(
    docs: tuple[SynthDoc, ...],
    world: WorldModel,
    *,
    scenario: ScenarioSpec | None,
    archetype: Archetype,
    pass2_client: LlmClientProtocol | None,
    pass2_model: str | None,
) -> CombinedValidatorResult:
    """Pass 1 (always) + Pass 2 (only when archetype.validator_level==STRICT and a pass2 client is
    provided).

    should_drop is True when:
      - Any Pass 1 violation (existential — Pass 1 violation means unknown names)
      - Pass 2 ran and `passed` is False (validate_pass2 already enforces 30% threshold internally)

    failing_doc_ids = union of Pass 1 and Pass 2 violating doc ids.
    """
    from scripts.synth.archetypes.base import ValidatorLevel
    from scripts.synth.llm.validator_pass2 import validate_pass2

    pass1 = validate_name_only(docs, world)
    pass1_failing = tuple(v.doc_id for v in pass1)
    pass1_drop = len(pass1_failing) > 0

    pass2_result: Pass2Result | None = None
    pass2_failing: tuple[str, ...] = ()
    pass2_drop = False

    if (
        archetype.validator_level == ValidatorLevel.STRICT
        and pass2_client is not None
        and pass2_model is not None
        and scenario is not None
        and len(docs) > 0
    ):
        pass2_result = await validate_pass2(
            scenario=scenario,
            docs=docs,
            world=world,
            client=pass2_client,
            model=pass2_model,
        )
        pass2_failing = tuple(v.doc_id for v in pass2_result.violations)
        pass2_drop = not pass2_result.passed

    all_failing = tuple(sorted(set(pass1_failing) | set(pass2_failing)))
    return CombinedValidatorResult(
        pass1_violations=pass1,
        pass2_result=pass2_result,
        should_drop=pass1_drop or pass2_drop,
        failing_doc_ids=all_failing,
    )
