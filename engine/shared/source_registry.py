"""Source registry — the seam between generic engine code and per-source knowledge.

Generic consumers (queue-priority at enqueue, retrieval fusion decay, the
doc-type resolver) used to read per-source metadata from hardcoded dicts in
``shared/constants.py`` and ``services/retrieval/doc_type_resolver.py``. That
meant generic code had to know about every specific source. This module
inverts the dependency: each source registers a :class:`SourceProfile` where
the source is defined (the connector module, via ``@register_connector``
reading the connector's class attributes), and generic code resolves metadata
through the getters below.

Unregistered keys resolve to safe defaults, so dynamic custom-ingest source
keys (e.g. ``workspace:<uuid>``) behave sanely without any registration:
``custom.document`` doc-type prefix, default queue priority, neutral score
multiplier, baseline recency decay.

Population happens at composition roots: the ingestion service and worker
already import the handlers package at boot (which fires the
``@register_connector`` decorators); the retrieval service imports it for the
same reason so fusion decay and the doc-type resolver see the registered
profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.shared.constants import DEFAULT_INGESTION_PRIORITY

# Prefix of DocType.CUSTOM_DOCUMENT ("custom.document") — the doc_type family
# that unregistered/dynamic sources fall back to.
DEFAULT_DOC_TYPE_PREFIX = "custom."

# Neutral post-RRF score multiplier: no demotion/promotion.
DEFAULT_SCORE_MULTIPLIER = 1.0


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Per-source metadata read by generic engine code.

    Fields:
      source_key: canonical key, matches ``documents.source_system`` /
        ``ingestion_queue.source_system`` (a ``SourceSystem.value`` for
        built-in sources; free-form for dynamic custom-ingest keys).
      doc_type_prefix: dotted prefix of the source's DocType family
        (``"slack."``, ``"github."``, ...). Used by the retrieval doc-type
        resolver to narrow unqualified tokens by a sources filter.
      ingestion_priority: queue priority at enqueue time. Worker claims order
        by priority DESC. Tiers (see connector modules for the rationale):
        100 interactive webhooks, 75 bursty agent/custom batches, 50 backfill.
      score_multiplier: post-RRF doc-score multiplier applied by fusion.
        Values < 1.0 demote a source's docs at equal vector relevance.
      half_life_days: per-source recency half-life override for fusion decay.
        None means: use the caller-supplied baseline, else
        DEFAULT_RECENCY_HALF_LIFE_DAYS.
    """

    source_key: str
    doc_type_prefix: str = DEFAULT_DOC_TYPE_PREFIX
    ingestion_priority: int = DEFAULT_INGESTION_PRIORITY
    score_multiplier: float = DEFAULT_SCORE_MULTIPLIER
    half_life_days: float | None = None


_registry: dict[str, SourceProfile] = {}


def register_source(profile: SourceProfile) -> None:
    """Register a source's profile. Idempotent for identical re-registration.

    Raises ValueError on a conflicting re-registration — two modules
    disagreeing about a source's metadata is a bug worth failing loudly on.
    """
    existing = _registry.get(profile.source_key)
    if existing is not None and existing != profile:
        raise ValueError(
            f"conflicting SourceProfile registration for {profile.source_key!r}: "
            f"{existing} vs {profile}"
        )
    _registry[profile.source_key] = profile


def get_source_profile(source_key: str) -> SourceProfile:
    """Resolve a source key to its profile; unregistered keys get defaults."""
    profile = _registry.get(source_key)
    if profile is not None:
        return profile
    return SourceProfile(source_key=source_key)


def registered_source_keys() -> list[str]:
    return sorted(_registry)


def is_registered(source_key: str) -> bool:
    return source_key in _registry


# ---- convenience getters (the shapes consumers actually read) ---------------


def doc_type_prefix_for(source_key: str) -> str:
    return get_source_profile(source_key).doc_type_prefix


def ingestion_priority_for(source_key: str) -> int:
    return get_source_profile(source_key).ingestion_priority


def score_multiplier_for(source_key: str) -> float:
    return get_source_profile(source_key).score_multiplier


def half_life_days_for(source_key: str, baseline: float) -> float:
    """Effective recency half-life: per-source override wins, else baseline.

    ``baseline`` is the caller-resolved fallback (caller-supplied
    half_life_days if set, else DEFAULT_RECENCY_HALF_LIFE_DAYS) — same
    resolution order the old SOURCE_HALF_LIFE_DAYS.get(source, baseline)
    call sites used.
    """
    override = get_source_profile(source_key).half_life_days
    return override if override is not None else baseline


__all__ = [
    "DEFAULT_DOC_TYPE_PREFIX",
    "DEFAULT_SCORE_MULTIPLIER",
    "SourceProfile",
    "doc_type_prefix_for",
    "get_source_profile",
    "half_life_days_for",
    "ingestion_priority_for",
    "is_registered",
    "register_source",
    "registered_source_keys",
    "score_multiplier_for",
]
