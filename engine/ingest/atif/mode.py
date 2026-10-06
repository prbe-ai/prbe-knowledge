"""Which source an agent session's indexed text comes from, per tenant."""

from __future__ import annotations

from enum import StrEnum

from engine.shared.config import Settings, get_settings


class RenderMode(StrEnum):
    #: The uploaded events, rendered directly. The trajectory is still built
    #: and stored on a completing pass; it just isn't compared.
    LEGACY = "legacy"
    #: Both, compared on every pass; legacy is served.
    SHADOW = "shadow"
    #: Both, compared on every pass; the trajectory's text is served when it
    #: matches and nothing was unparsed, legacy otherwise.
    ATIF = "atif"


def _ids(value: str) -> frozenset[str]:
    return frozenset(c.strip() for c in value.split(",") if c.strip())


def render_mode(customer_id: str, settings: Settings | None = None) -> RenderMode:
    s = settings or get_settings()
    if customer_id in _ids(s.session_render_atif_customers):
        return RenderMode.ATIF
    if customer_id in _ids(s.session_render_shadow_customers):
        return RenderMode.SHADOW
    try:
        return RenderMode(s.session_render_default.strip().lower())
    except ValueError:
        # A typo in the deploy must not take ingestion down; it means legacy.
        return RenderMode.LEGACY
