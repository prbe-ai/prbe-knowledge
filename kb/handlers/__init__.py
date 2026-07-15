"""Importing this package triggers @register_connector decorators for every
source connector — the kb integration connectors below plus the engine-door
connectors (custom_ingest, manual_upload) that live in engine.ingest.handlers.
The ingestion service + worker both import this at boot, so the connector
registry is fully populated before any webhook is served. The retrieval
service wrapper (services/retrieval/main.py) imports it too: the same
decorators populate engine.shared.source_registry (per-source doc_type
prefix / priority / decay profiles), which fusion and the doc-type resolver
read.

Adding a new connector: create `kb/handlers/<source>.py`, decorate the class
with `@register_connector`, and append its module import here.
"""

from engine.ingest.handlers import (
    custom_ingest,  # noqa: F401
    manual_upload,  # noqa: F401
)
from kb.handlers import (
    claude_code,  # noqa: F401
    codegraph,  # noqa: F401
    github,  # noqa: F401
    granola,  # noqa: F401
    incident_sources,  # noqa: F401  (profile-only: no Connector class yet)
    linear,  # noqa: F401
    notion,  # noqa: F401
    sentry,  # noqa: F401
    slack,  # noqa: F401
    wiki,  # noqa: F401
)

__all__: list[str] = []
