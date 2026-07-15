"""Importing this package triggers @register_connector decorators for every
source module listed below. The ingestion service + worker both import this
at boot, so the registry is fully populated before any webhook is served.
The retrieval service imports it too: the same decorators populate
shared.source_registry (per-source doc_type prefix / priority / decay
profiles), which fusion and the doc-type resolver read.

Adding a new connector: create `handlers/<source>.py`, decorate the class with
`@register_connector`, and append its module import here.
"""

from services.ingestion.handlers import (
    claude_code,  # noqa: F401
    codegraph,  # noqa: F401
    custom_ingest,  # noqa: F401
    github,  # noqa: F401
    granola,  # noqa: F401
    incident_sources,  # noqa: F401  (profile-only: no Connector class yet)
    linear,  # noqa: F401
    manual_upload,  # noqa: F401
    notion,  # noqa: F401
    sentry,  # noqa: F401
    slack,  # noqa: F401
    wiki,  # noqa: F401
)

__all__: list[str] = []
