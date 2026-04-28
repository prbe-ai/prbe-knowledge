"""UsageLoggingMiddleware — record one usage_events row per retrieval call.

Where this fits in the pipeline:

    request → UsageLoggingMiddleware.dispatch
              ↓ capture start time, caller_kind, request_id
              ↓ call_next(request)  ← runs the route handler
              ↓ either:
                 - handler returned a Response → status='ok'
                 - handler raised → status='error', re-raise after scheduling
              ↓ schedule write_usage_event() as response.background
    response → user

The DB write happens *after* the response body is sent (Starlette's
`Response.background` is awaited post-send), so a slow INSERT can never
add latency to the user's perceived response time. write_usage_event()
also swallows all exceptions — a DB outage degrades to "no usage row"
rather than 500.

The handler stashes two values on `request.state` that we read here:

  * customer_id  — set by main.py after `authenticate_query` resolves
  * result_count — set by main.py after retrieval runs (chunk count or
                   chunk_count for /sources)

If `customer_id` was never stashed (auth failed → handler never ran),
we drop the event rather than write it under a bogus tenant. The 4xx
itself is observable via stdout request logs; we don't need a row.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from starlette.background import BackgroundTask, BackgroundTasks
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from services.retrieval.usage import (
    CALLER_KIND_UNKNOWN,
    STATUS_ERROR,
    STATUS_OK,
    UsageEvent,
    event_type_for,
    write_usage_event,
)
from shared.logging import get_logger

log = get_logger(__name__)


# Paths the middleware logs. Anything else passes through silently.
# /usage/* and /health are explicitly excluded — logging reads of the
# usage table itself would create a self-amplifying audit loop, and the
# health endpoint is hit every few seconds by the load balancer.
_LOGGED_PREFIXES: tuple[str, ...] = ("/retrieve", "/query", "/sources")
_SKIPPED_PREFIXES: tuple[str, ...] = ("/health", "/usage")


def _should_log(path: str) -> bool:
    """True iff this path is a retrieval call we should record.

    /usage/* is excluded explicitly so reads of the audit table never
    feed back into the table itself (otherwise opening the dashboard
    page would produce N rows per second).
    """
    if any(path.startswith(p) for p in _SKIPPED_PREFIXES):
        return False
    return any(path.startswith(p) for p in _LOGGED_PREFIXES)


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    """Record one usage_events row per /retrieve, /query, /sources call.

    Inherits from Starlette's BaseHTTPMiddleware. We use `dispatch()`
    so we can:
      * read request headers/state before the handler runs
      * observe whether the handler raised
      * attach a BackgroundTask to the outgoing Response so the DB write
        completes after the bytes are flushed to the client
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        if not _should_log(path):
            return await call_next(request)

        # Caller-kind: free-form header, default to 'unknown'. We never
        # reject for a missing or unknown value — the dashboard will
        # surface 'unknown' as its own bucket.
        caller_kind = request.headers.get("x-caller-kind", CALLER_KIND_UNKNOWN)
        caller_subject = request.headers.get("x-caller-subject")  # optional

        # Request id: prefer a caller-supplied X-Request-Id, otherwise
        # generate. Stash on request.state so handlers can include it in
        # their own structured logs (single id across logs + audit row).
        raw_request_id = request.headers.get("x-request-id")
        try:
            request_id = str(uuid.UUID(raw_request_id)) if raw_request_id else str(uuid.uuid4())
        except ValueError:
            # Caller sent a non-UUID; ignore it and mint our own.
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        status = STATUS_OK
        error_class: str | None = None
        response: Response | None = None
        try:
            response = await call_next(request)
            # Treat 5xx as error even when the handler didn't raise — this
            # captures FastAPI's own 500 path (validation errors with
            # broken responses, etc.) so the dashboard sees them.
            if response.status_code >= 500:
                status = STATUS_ERROR
                error_class = f"http_{response.status_code}"
        except Exception as exc:
            status = STATUS_ERROR
            error_class = type(exc).__name__
            # We still want to record the event, but the original
            # exception MUST propagate so FastAPI's normal error handler
            # produces the user-facing 500. Schedule the write on a
            # detached task — there's no Response to attach to.
            latency_ms = int((time.perf_counter() - start) * 1000)
            event = self._build_event(
                request,
                path=path,
                caller_kind=caller_kind,
                caller_subject=caller_subject,
                request_id=request_id,
                status=status,
                error_class=error_class,
                latency_ms=latency_ms,
            )
            if event is not None:
                # Fire-and-forget. write_usage_event() swallows its own
                # exceptions, so this task can never raise unhandled.
                asyncio.create_task(write_usage_event(event))  # noqa: RUF006
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        event = self._build_event(
            request,
            path=path,
            caller_kind=caller_kind,
            caller_subject=caller_subject,
            request_id=request_id,
            status=status,
            error_class=error_class,
            latency_ms=latency_ms,
        )
        if event is not None:
            # Chain to any BackgroundTask the handler already set, so we
            # don't clobber existing post-response work.
            existing = response.background
            new_task = BackgroundTask(write_usage_event, event)
            if existing is None:
                response.background = new_task
            else:
                # Starlette doesn't expose a public "and then" combinator.
                # The pragmatic option is to wrap into a tasks list via
                # BackgroundTasks; both will run in order.
                tasks = BackgroundTasks()
                tasks.add_task(_run_existing, existing)
                tasks.add_task(write_usage_event, event)
                response.background = tasks
        return response

    def _build_event(
        self,
        request: Request,
        *,
        path: str,
        caller_kind: str,
        caller_subject: str | None,
        request_id: str,
        status: str,
        error_class: str | None,
        latency_ms: int,
    ) -> UsageEvent | None:
        """Assemble the UsageEvent from request state. Returns None if
        we lack enough context to record the event safely."""
        customer_id: str | None = getattr(request.state, "customer_id", None)
        if not customer_id:
            # Auth failed (401/400) before the handler ran. Drop the row
            # rather than attribute the call to a guessed tenant.
            return None

        event_type = event_type_for(path)

        # Summary: handler may have stashed the request body / doc_id on
        # request.state for us to pick up. Falls through to the path tail
        # for /sources GET routes that don't carry a body.
        summary: str | None = getattr(request.state, "usage_summary", None)
        if summary is None and path.startswith("/sources"):
            # /sources/{doc_id:path} — strip the leading prefix.
            tail = path[len("/sources/") :] if path.startswith("/sources/") else None
            summary = tail or None

        result_count: int | None = getattr(request.state, "result_count", None)

        return UsageEvent(
            customer_id=customer_id,
            caller_kind=caller_kind,
            caller_subject=caller_subject,
            event_type=event_type,
            endpoint=path,
            status=status,
            error_class=error_class,
            request_id=request_id,
            summary=summary,
            latency_ms=latency_ms,
            result_count=result_count,
        )


async def _run_existing(task: BackgroundTask) -> None:
    """Adapter so we can wrap a plain BackgroundTask inside BackgroundTasks."""
    await task()


__all__ = [
    "UsageLoggingMiddleware",
]
