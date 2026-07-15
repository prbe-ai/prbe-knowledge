"""OpenTelemetry boot stub.

Phase 0 ships logging-only. This module initializes OTel when the
OTEL_EXPORTER_OTLP_ENDPOINT env var is set — otherwise no-op. Drop the
opentelemetry-sdk/exporter packages into pyproject.toml to activate.
"""

from __future__ import annotations

import os

from shared.logging import get_logger

log = get_logger(__name__)


def maybe_init_otel(service_name: str) -> None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("otel.packages_missing", endpoint=endpoint)
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(provider)
    log.info("otel.ready", endpoint=endpoint, service=service_name)
