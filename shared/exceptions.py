"""Named exception registry.

Every failure mode gets a class. Handlers, retrievers, and ops code should
never raise bare `Exception`. The classes here also carry a `transient` flag
that the worker uses to decide retry vs DLQ.
"""

from __future__ import annotations

from typing import Any


class PrbeError(Exception):
    """Base for all PRBE-defined errors."""

    transient: bool = False

    def __init__(self, message: str = "", /, **context: Any) -> None:
        super().__init__(message)
        self.context = context

    def __str__(self) -> str:
        base = super().__str__()
        if not self.context:
            return base
        ctx = " ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{base} ({ctx})" if base else ctx


# --- Config -----------------------------------------------------------------

class ConfigError(PrbeError): ...


class MissingSecret(ConfigError): ...


# --- Auth / webhook signing -------------------------------------------------

class AuthError(PrbeError): ...


class InvalidSignature(AuthError): ...


class TokenMissing(AuthError): ...


class TokenExpired(AuthError):
    transient = True


class TokenRefreshFailed(AuthError):
    transient = True


class GitHubAuthError(AuthError):
    """Raised when a GitHub App installation-token mint fails."""

    transient = True


# --- Storage ----------------------------------------------------------------

class StorageError(PrbeError): ...


class StorageUnavailable(StorageError):
    transient = True


class StorageNotFound(StorageError): ...


# --- Database ---------------------------------------------------------------

class DatabaseError(PrbeError): ...


class DatabaseUnavailable(DatabaseError):
    transient = True


class TenantIsolationError(DatabaseError):
    """Raised if a query executes without a bound customer_id GUC."""


# --- Ingestion --------------------------------------------------------------

class IngestionError(PrbeError): ...


class HandlerNotFound(IngestionError): ...


class InvalidWebhookPayload(IngestionError): ...


class DuplicateEventIgnored(IngestionError):
    """Not an error per se — signals idempotent no-op. Worker treats as success."""


class UnsupportedEventType(IngestionError):
    """Event type we don't care about (e.g. Slack user_typing). Skip, not fail."""


class NormalizationError(IngestionError): ...


class SourceAPIError(IngestionError):
    transient = True


class RateLimited(SourceAPIError): ...


class TransientSourceError(SourceAPIError):
    transient = True


class PermanentSourceError(IngestionError):
    """4xx from source that no retry can fix (bad scopes, deleted resource)."""


class SourceAlreadyConnectedError(IngestionError):
    """Tried to connect a (source, external_id) already owned by another customer.

    Raised by `record_mapping` (and the OAuth exchange route) when an install
    would overwrite an existing customer→workspace mapping with a different
    customer_id. Holds enough context for the caller to render a useful 409
    without leaking the existing customer_id externally.
    """

    def __init__(
        self,
        *,
        source_system: str,
        external_id: str,
        existing_customer_id: str,
        attempted_customer_id: str,
        external_name: str | None = None,
    ) -> None:
        super().__init__(
            f"{source_system} workspace {external_id!r} is already connected to a different customer",
            source_system=source_system,
            external_id=external_id,
            existing_customer_id=existing_customer_id,
            attempted_customer_id=attempted_customer_id,
            external_name=external_name,
        )
        self.source_system = source_system
        self.external_id = external_id
        self.existing_customer_id = existing_customer_id
        self.attempted_customer_id = attempted_customer_id
        self.external_name = external_name


# --- Embeddings -------------------------------------------------------------

class EmbeddingError(PrbeError): ...


class EmbeddingBatchRejected(EmbeddingError): ...


class EmbeddingContextLengthExceeded(EmbeddingError): ...


class EmbeddingRateLimited(EmbeddingError):
    transient = True


class EmbeddingProviderUnavailable(EmbeddingError):
    """OpenAI 5xx / connection errors. Distinct from EmbeddingBatchRejected
    (which is per-batch input-shaped) so the worker retries the whole queue
    row instead of silently routing chunks to failed_chunks during an outage.
    """

    transient = True


# --- Retrieval --------------------------------------------------------------

class RetrievalError(PrbeError): ...


class RouterTimeout(RetrievalError):
    transient = True


class RouterParseError(RetrievalError): ...


# --- Wiki agent loop --------------------------------------------------------


class AgentHaltError(PrbeError):
    """Wiki agent loop halted before tool_done was called.

    Raised by the agent harness on any unrecoverable mid-drain failure:
    turn cap reached, stall (3 turns no consequential tool call), update
    cap exceeded, persistent Gemini API error, or compactor crash. The
    synthesis worker catches this and parks all 'synthesizing' rows in
    DLQ with `dlq_reason='agent.{halt_reason}'`. Admin reset is the
    recovery path.
    """

    def __init__(self, reason: str, /, **context: Any) -> None:
        super().__init__(reason, **context)
        self.reason = reason


class ToolValidationError(PrbeError):
    """Agent tool input failed Pydantic validation.

    The harness captures this from a tool dispatch and returns a typed
    tool_result error to the model — the agent re-decides on the next
    turn. Distinct from AgentHaltError because the loop continues.
    """


class AgentCompactionError(PrbeError):
    """Compactor (Flash Lite summarizer) failed.

    Raised by `agent_compactor.call_summarizer` when the summary call
    errors or produces unparseable output. The harness re-raises as
    AgentHaltError('agent.compaction_failed') so the drain DLQs cleanly
    and the operator can see in the dashboard that compaction was the
    root cause.
    """


# --- Queue ------------------------------------------------------------------

class QueueError(PrbeError): ...


class QueueRowStuck(QueueError): ...


class QueueInsertFailed(QueueError):
    transient = True


# --- Chunking / graph -------------------------------------------------------

class ChunkError(PrbeError): ...


class ChunkTooLarge(ChunkError): ...


class GraphError(PrbeError): ...


class GraphNodeConflict(GraphError): ...


# --- Backfill ---------------------------------------------------------------

class BackfillError(PrbeError): ...


class BackfillCursorCorrupt(BackfillError): ...


class NotSupportedByConnector(IngestionError):
    """Connector does not implement an optional capability (e.g. backfill)."""


__all__ = [
    "AgentCompactionError",
    "AgentHaltError",
    "AuthError",
    "BackfillCursorCorrupt",
    "BackfillError",
    "ChunkError",
    "ChunkTooLarge",
    "ConfigError",
    "DatabaseError",
    "DatabaseUnavailable",
    "DuplicateEventIgnored",
    "EmbeddingBatchRejected",
    "EmbeddingContextLengthExceeded",
    "EmbeddingError",
    "EmbeddingProviderUnavailable",
    "EmbeddingRateLimited",
    "GitHubAuthError",
    "GraphError",
    "GraphNodeConflict",
    "HandlerNotFound",
    "IngestionError",
    "InvalidSignature",
    "InvalidWebhookPayload",
    "MissingSecret",
    "NormalizationError",
    "NotSupportedByConnector",
    "PermanentSourceError",
    "PrbeError",
    "QueueError",
    "QueueInsertFailed",
    "QueueRowStuck",
    "RateLimited",
    "RetrievalError",
    "RouterParseError",
    "RouterTimeout",
    "SourceAPIError",
    "StorageError",
    "StorageNotFound",
    "StorageUnavailable",
    "TenantIsolationError",
    "TokenExpired",
    "TokenMissing",
    "TokenRefreshFailed",
    "ToolValidationError",
    "TransientSourceError",
    "UnsupportedEventType",
]
