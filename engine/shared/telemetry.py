"""OpenTelemetry boot stub.

Phase 0 ships logging-only. This module initializes OTel when the
OTEL_EXPORTER_OTLP_ENDPOINT env var is set — otherwise no-op. Drop the
opentelemetry-sdk/exporter packages into pyproject.toml to activate.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from engine.shared.logging import get_logger

log = get_logger(__name__)


def new_trace_id() -> str:
    """Mint a collision-free trace id for one retrieval request.

    The millisecond prefix is kept because it sorts chronologically and
    reads well in logs, but it is NOT unique on its own: research-os fans
    one /v1/search out into several concurrent /retrieve calls, and those
    land in the same millisecond routinely. Two live requests then share a
    trace id and their interleaved log lines are unattributable -- observed
    in production, where two distinct retrievals (different source_keys)
    both logged as `q-1784923345223`.

    The uuid4 suffix is what actually guarantees uniqueness; keep both.
    """
    return f"q-{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:8]}"


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
