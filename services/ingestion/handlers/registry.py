"""Connector registry.

Connector classes register via decorator at module import. The ingestion
service imports `services.ingestion.handlers` (the package) once at startup,
which triggers all per-source modules to register themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from services.ingestion.handlers.base import Connector, ConnectorContext
from shared.constants import SourceSystem
from shared.exceptions import HandlerNotFound

_registry: dict[SourceSystem, type[Connector]] = {}

C = TypeVar("C", bound=Connector)


def register_connector(source: SourceSystem) -> Callable[[type[C]], type[C]]:
    """Decorator: registers a Connector class under its SourceSystem.

        @register_connector(SourceSystem.SLACK)
        class SlackConnector(Connector):
            source_system = SourceSystem.SLACK
            ...
    """

    def decorator(cls: type[C]) -> type[C]:
        if cls.source_system != source:
            raise ValueError(
                f"register_connector({source}) but class declares {cls.source_system}"
            )
        _registry[source] = cls
        return cls

    return decorator


def get_connector_class(source: SourceSystem) -> type[Connector]:
    try:
        return _registry[source]
    except KeyError as exc:
        raise HandlerNotFound(
            f"no connector registered for {source.value}",
            registered=list(_registry),
        ) from exc


def list_registered() -> list[SourceSystem]:
    return sorted(_registry, key=lambda s: s.value)


def build_connector(source: SourceSystem, ctx: ConnectorContext) -> Connector:
    cls = get_connector_class(source)
    return cls(ctx)


__all__ = [
    "build_connector",
    "get_connector_class",
    "list_registered",
    "register_connector",
]
