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

from scripts.synth.output.base import SynthDoc
from scripts.synth.world_model import WorldModel

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

    # People: display_name, gh_username, channel-mention forms
    for person in world.people:
        if person.display_name:
            allowed.add(person.display_name.lower())
        if person.gh_username:
            allowed.add(person.gh_username.lower())
            allowed.add(f"@{person.gh_username.lower()}")

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
