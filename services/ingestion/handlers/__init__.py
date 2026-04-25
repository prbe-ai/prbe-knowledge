"""Importing this package triggers @register_connector decorators for every
source module listed below. The ingestion service + worker both import this
at boot, so the registry is fully populated before any webhook is served.

Adding a new connector: create `handlers/<source>.py`, decorate the class with
`@register_connector`, and append its module import here.
"""

from services.ingestion.handlers import (
    github,  # noqa: F401
    granola,  # noqa: F401
    linear,  # noqa: F401
    notion,  # noqa: F401
    sentry,  # noqa: F401
    slack,  # noqa: F401
)

__all__: list[str] = []
