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
    "TransientSourceError",
    "UnsupportedEventType",
]
