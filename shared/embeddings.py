"""Embedding clients with:

- Configurable batch size
- Native retries via tenacity on transient failures
- Recursive half-split on partial batch rejections so one poison chunk
  doesn't kill a whole batch -- the bad chunk is isolated and written to
  `failed_chunks` for later inspection
- Async (wraps the sync provider SDKs in a thread)

Two concrete providers live here:

- `GeminiEmbedder`: gemini-embedding-2. The production embedder for both
  ingestion (writes `chunks.embedding_v2`) and retrieval (embeds queries).
- `OpenAIEmbedder`: text-embedding-3-large. EVAL-HARNESS ONLY post-2026-05-14
  cutover. Kept so `scripts/eval_data/fixtures.py` can regenerate v1
  baselines for apples-to-apples retrieval comparisons. No production code
  path reads from it.

The recursive half-split + batching machinery lives on `_BaseEmbedder` and
is shared between providers. Providers override `_embed_once` (raw bytes
in, raw vectors out) and may override `embed_documents` / `embed_query`
when the provider needs asymmetric input formatting (Gemini does;
OpenAI does not).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from shared.config import Settings, get_settings
from shared.constants import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_V2_DIM,
    EMBEDDING_V2_MODEL,
)
from shared.exceptions import (
    EmbeddingBatchRejected,
    EmbeddingContextLengthExceeded,
    EmbeddingProviderUnavailable,
    EmbeddingRateLimited,
)
from shared.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class EmbeddedChunk:
    chunk_index: int
    embedding: list[float]


@dataclass(slots=True)
class FailedChunk:
    chunk_index: int
    content_preview: str
    error: str


@dataclass(slots=True)
class EmbedResult:
    embedded: list[EmbeddedChunk]
    failed: list[FailedChunk]


@dataclass(slots=True)
class DocItem:
    """A document chunk + its parent doc title.

    Gemini-2 retrieval quality is meaningfully better when document inputs
    are formatted as `title: {title} | text: {content}`. OpenAI ignores the
    title but accepting the same shape from callers means ingest can drive
    both providers off one input list.
    """

    content: str
    title: str | None = None


class _BaseEmbedder:
    """Shared batch driver with recursive half-split poison isolation.

    Subclasses override `_embed_once` (raw provider call) and -- when the
    provider needs asymmetric formatting -- `embed_documents` / `embed_query`.
    """

    model_id: str
    dim: int

    def __init__(
        self,
        *,
        model_id: str,
        dim: int,
        batch_size: int,
    ) -> None:
        self.model_id = model_id
        self.dim = dim
        self._batch_size = batch_size

    # ---- public surface -----------------------------------------------------

    async def embed_many(self, texts: list[str]) -> EmbedResult:
        """Embed plain texts. Default behavior; works for any provider that
        doesn't need asymmetric prefixing."""
        embedded: list[EmbeddedChunk] = []
        failed: list[FailedChunk] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            indices = list(range(start, start + len(batch)))
            await self._embed_with_split(batch, indices, embedded, failed)
        return EmbedResult(embedded=embedded, failed=failed)

    async def embed_documents(self, items: list[DocItem]) -> EmbedResult:
        """Embed document items (content + optional title). Default impl
        ignores title and forwards content to `embed_many`. Providers that
        benefit from title-aware formatting (Gemini) override this.
        """
        return await self.embed_many([item.content for item in items])

    async def embed_query(self, text: str) -> list[float]:
        """Single-text helper for the retrieval path. Providers override
        this to apply query-side prefixing (Gemini) or leave it as a passthrough
        (OpenAI)."""
        vectors = await self._embed_once([text])
        return vectors[0]

    # ---- subclass hook ------------------------------------------------------

    async def _embed_once(self, batch: list[str]) -> list[list[float]]:
        raise NotImplementedError

    # ---- internals ----------------------------------------------------------

    async def _embed_with_split(
        self,
        batch: list[str],
        indices: list[int],
        embedded: list[EmbeddedChunk],
        failed: list[FailedChunk],
    ) -> None:
        try:
            vectors = await self._embed_once(batch)
        except EmbeddingContextLengthExceeded as exc:
            await self._split_or_record(batch, indices, embedded, failed, str(exc))
            return
        except EmbeddingBatchRejected as exc:
            await self._split_or_record(batch, indices, embedded, failed, str(exc))
            return

        if len(vectors) != len(batch):
            await self._split_or_record(
                batch, indices, embedded, failed, "vector count mismatch"
            )
            return

        for idx, vec in zip(indices, vectors, strict=True):
            embedded.append(EmbeddedChunk(chunk_index=idx, embedding=vec))

    async def _split_or_record(
        self,
        batch: list[str],
        indices: list[int],
        embedded: list[EmbeddedChunk],
        failed: list[FailedChunk],
        error: str,
    ) -> None:
        if len(batch) == 1:
            failed.append(
                FailedChunk(
                    chunk_index=indices[0],
                    content_preview=batch[0][:200],
                    error=error,
                )
            )
            log.warning(
                "embed.chunk_rejected",
                idx=indices[0],
                error=error,
                model=self.model_id,
            )
            return
        mid = len(batch) // 2
        await self._embed_with_split(batch[:mid], indices[:mid], embedded, failed)
        await self._embed_with_split(batch[mid:], indices[mid:], embedded, failed)


class OpenAIEmbedder(_BaseEmbedder):
    """text-embedding-3-large via the AsyncOpenAI SDK.

    Gateway-aware (managed-shared / self-host, plan D1): when
    ``llm_gateway_url`` is set we point AsyncOpenAI at the central LiteLLM
    proxy with ``llm_gateway_key`` as the bearer. LiteLLM exposes
    OpenAI-compatible ``/v1/embeddings``, so the SDK + all of its
    retry / timeout / error machinery work unchanged through the gateway —
    we don't pay a wrapper hop on the hot embedding path. Without the
    gateway var we keep the direct-provider path (``openai_api_key``) for
    dev, self-host-with-own-keys, and the eval harness.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model: str = EMBEDDING_MODEL,
    ) -> None:
        settings = settings or get_settings()
        super().__init__(
            model_id=model,
            dim=EMBEDDING_DIM,
            batch_size=settings.embedding_batch_size,
        )
        gw_url = settings.llm_gateway_url.strip()
        if gw_url:
            # `api_key=""` makes the SDK omit the Authorization header
            # entirely, which a no-auth proxy would reject; pass a literal
            # space so the header is always present (LiteLLM ignores the
            # value when `master_key` isn't enforced).
            gw_key = settings.llm_gateway_key.get_secret_value() or " "
            self._client = AsyncOpenAI(api_key=gw_key, base_url=gw_url)
        else:
            key = settings.openai_api_key.get_secret_value()
            self._client = AsyncOpenAI(api_key=key) if key else None
        # Strip `openai/` prefix from the canonical model constant for the SDK call.
        # Through the LiteLLM gateway the raw id is fine — the proxy's
        # `*` catch-all glob routes it to OpenAI.
        self._sdk_model = model.split("/", 1)[-1] if "/" in model else model

    async def _embed_once(self, batch: list[str]) -> list[list[float]]:
        if self._client is None:
            # Stub mode for tests / local dev without an OpenAI key.
            return [_hash_vector(t, self.dim) for t in batch]

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await self._client.embeddings.create(
                    model=self._sdk_model,
                    input=batch,
                )
                return [d.embedding for d in resp.data]
            except RateLimitError as exc:
                if attempt >= 5:
                    raise EmbeddingRateLimited(str(exc)) from exc
                await asyncio.sleep(min(2**attempt, 30))
            except APITimeoutError as exc:
                if attempt >= 3:
                    raise EmbeddingProviderUnavailable(f"timeout: {exc}") from exc
                await asyncio.sleep(1 * attempt)
            except APIConnectionError as exc:
                if attempt >= 3:
                    raise EmbeddingProviderUnavailable(
                        f"connection error: {exc}"
                    ) from exc
                await asyncio.sleep(1 * attempt)
            except APIError as exc:
                msg = str(exc).lower()
                if "maximum context length" in msg or ("token" in msg and "exceed" in msg):
                    raise EmbeddingContextLengthExceeded(str(exc)) from exc
                status = getattr(exc, "status_code", None)
                if isinstance(exc, APIStatusError) and isinstance(status, int) and status >= 500:
                    if attempt >= 2:
                        raise EmbeddingProviderUnavailable(
                            f"openai {status}: {exc}"
                        ) from exc
                    await asyncio.sleep(1 * attempt)
                    continue
                if attempt >= 2:
                    raise EmbeddingBatchRejected(str(exc)) from exc
                await asyncio.sleep(1)




# ---- Gemini -------------------------------------------------------------

# gemini-embedding-2 does NOT expose a `task_type` field (unlike
# gemini-embedding-001). Asymmetric retrieval is implemented by prefixing
# the input string itself. Format is from
# https://ai.google.dev/gemini-api/docs/embeddings.
_GEMINI_DOC_PREFIX_WITH_TITLE = "title: {title} | text: {content}"
_GEMINI_DOC_PREFIX_NO_TITLE = "text: {content}"
_GEMINI_QUERY_PREFIX = "task: search result | query: {query}"


class GeminiEmbedder(_BaseEmbedder):
    """gemini-embedding-2, two transports.

    Asymmetric retrieval: documents and queries are formatted differently
    before being sent to the model so the same vector space encodes
    "this is a chunk to retrieve" vs "this is a question to match".

    Transport (plan D1, gateway routing):

      - **Gateway mode** (``llm_gateway_url`` set): call via
        ``shared.llm.aembedding`` so the LiteLLM proxy holds the Google
        credentials. The google-genai SDK has no ``base_url`` knob (unlike
        AsyncOpenAI), so we can't point the SDK at the proxy directly —
        the wrapper is the only way through. LiteLLM's ``gemini-*`` model
        route covers ``gemini-embedding-2`` as long as the proxy
        config lists it on the embeddings side (a managed-tenant
        precondition).
      - **Direct mode** (no gateway): the original google-genai async path
        (concurrent sub-batches, native error translation). Dev,
        self-host-with-own-keys, eval harness.

    Both transports share the recursive half-split poison isolation +
    asymmetric prefixing in ``_BaseEmbedder`` / ``embed_documents`` /
    ``embed_query`` above.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model: str = EMBEDDING_V2_MODEL,
    ) -> None:
        settings = settings or get_settings()
        super().__init__(
            model_id=model,
            dim=EMBEDDING_V2_DIM,
            batch_size=settings.embedding_batch_size,
        )
        secret = settings.google_api_key
        api_key = secret.get_secret_value() if secret is not None else ""
        self._api_key = api_key
        self._client: Any | None = None
        # Gateway transport: set when llm_gateway_url is configured. The
        # gateway env vars are read fresh in `shared.llm` per call, so we
        # just need to know which branch to enter on the embedding call.
        self._gateway_url = settings.llm_gateway_url.strip()
        # Strip `google/` prefix from the canonical constant for the SDK call.
        # In gateway mode the wrapper takes the bare id and prepends
        # `gemini/` so LiteLLM's glob routes it to the Gemini provider.
        self._sdk_model = model.split("/", 1)[-1] if "/" in model else model

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            return None
        try:
            from google import genai
        except ImportError as exc:
            raise EmbeddingProviderUnavailable(
                f"google-genai not installed: {exc}"
            ) from exc
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def embed_documents(self, items: list[DocItem]) -> EmbedResult:
        formatted = [_format_gemini_document(item) for item in items]
        return await self.embed_many(formatted)

    async def embed_query(self, text: str) -> list[float]:
        prefixed = _GEMINI_QUERY_PREFIX.format(query=text)
        vectors = await self._embed_once([prefixed])
        return vectors[0]

    # Wall-clock optimization: the google-genai SDK's
    # `embed_content(contents=[N items])` call serializes internally into
    # smaller per-call requests, processed sequentially. For batches of 256
    # this turns into ~30 sequential HTTP round trips that bottleneck on
    # network latency, not API throughput. Splitting our batch into smaller
    # sub-batches and asyncio.gather'ing them gives the SDK no choice but
    # to issue them concurrently.
    #
    # GROUP_SIZE = 4 keeps each sub-call close to one HTTP round trip
    # (the SDK splits larger inputs internally). MAX_PARALLEL = 64 lets a
    # single process saturate per-host network I/O at ~250ms per round
    # trip = ~250 RPM per process. Across 4 partitioned workers that's
    # ~1000 RPM total, well under the 20k-RPM Tier 3 ceiling for
    # Gemini Embedding 2. Earlier values (8/16) left the project at ~1.3k
    # RPM (6% utilization) and finished a 91k-chunk backfill in ~7-8h
    # instead of <1h.
    _SUBBATCH_GROUP_SIZE = 4
    _SUBBATCH_MAX_PARALLEL = 64

    async def _embed_once(self, batch: list[str]) -> list[list[float]]:
        # Gateway transport: route through shared.llm.aembedding (LiteLLM
        # proxy holds the Google credentials). The same sub-group concurrency
        # pattern applies — the proxy fans these out to Google for us — so
        # we keep the grouping for I/O parallelism even on the gateway path.
        if self._gateway_url:
            return await self._embed_once_gateway(batch)

        client = self._ensure_client()
        if client is None:
            # Stub mode -- match OpenAI's behavior so dev/local without
            # GOOGLE_API_KEY doesn't crash the dual-write path.
            return [_hash_vector(t, self.dim) for t in batch]

        # Split the batch into bounded sub-groups and run them concurrently.
        # Reassemble in original order before returning so callers see one
        # vector per input item.
        groups: list[list[str]] = [
            batch[i : i + self._SUBBATCH_GROUP_SIZE]
            for i in range(0, len(batch), self._SUBBATCH_GROUP_SIZE)
        ]
        if not groups:
            return []

        sem = asyncio.Semaphore(self._SUBBATCH_MAX_PARALLEL)

        async def call_one(group: list[str]) -> list[list[float]]:
            async with sem:
                return await self._embed_subbatch(client, group)

        results = await asyncio.gather(
            *[call_one(g) for g in groups], return_exceptions=True
        )

        vectors: list[list[float]] = []
        for r in results:
            if isinstance(r, BaseException):
                # First sub-group failure aborts the whole batch -- the
                # _BaseEmbedder recursive half-split above will then halve
                # OUR input batch and retry. The bad input lands in
                # `failed_chunks` once isolated.
                raise r
            vectors.extend(r)
        return vectors

    async def _embed_subbatch(
        self, client: Any, batch: list[str]
    ) -> list[list[float]]:
        """One concurrent unit of work: a single Gemini call (with retries)
        for a small sub-batch. Errors are translated into our embedding
        taxonomy so the upstream recursive half-split treats them the same
        way it treats OpenAI failures.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await client.aio.models.embed_content(
                    model=self._sdk_model,
                    contents=batch,
                )
            except Exception as exc:
                translated = _translate_gemini_error(exc)
                if isinstance(translated, EmbeddingProviderUnavailable):
                    if attempt >= 3:
                        raise translated from exc
                    await asyncio.sleep(1 * attempt)
                    continue
                if isinstance(translated, EmbeddingRateLimited):
                    if attempt >= 5:
                        raise translated from exc
                    await asyncio.sleep(min(2**attempt, 30))
                    continue
                # ContextLengthExceeded / BatchRejected: bubble up so the
                # recursive splitter isolates the bad chunk.
                raise translated from exc

            embeddings = getattr(resp, "embeddings", None) or []
            vectors: list[list[float]] = []
            for emb in embeddings:
                vec = getattr(emb, "values", None)
                if vec is None and isinstance(emb, dict):
                    vec = emb.get("values")
                if vec is None:
                    raise EmbeddingBatchRejected(
                        "gemini response missing embedding values"
                    )
                vectors.append(list(vec))
            return vectors

    # ---- gateway transport (LiteLLM proxy) ---------------------------------

    async def _embed_once_gateway(self, batch: list[str]) -> list[list[float]]:
        """Embed via ``shared.llm.aembedding`` (the LiteLLM proxy).

        Same sub-group fan-out as the direct path — even though the proxy
        could serialize internally, splitting maximizes I/O concurrency
        (and matches the throughput numbers the direct path was tuned to).
        Errors from the wrapper come back as ``LLMError`` and are
        re-translated into the embedding taxonomy so the upstream recursive
        half-split treats them identically to the direct-SDK path.
        """
        groups: list[list[str]] = [
            batch[i : i + self._SUBBATCH_GROUP_SIZE]
            for i in range(0, len(batch), self._SUBBATCH_GROUP_SIZE)
        ]
        if not groups:
            return []

        sem = asyncio.Semaphore(self._SUBBATCH_MAX_PARALLEL)

        async def call_one(group: list[str]) -> list[list[float]]:
            async with sem:
                return await self._embed_subbatch_gateway(group)

        results = await asyncio.gather(
            *[call_one(g) for g in groups], return_exceptions=True
        )

        vectors: list[list[float]] = []
        for r in results:
            if isinstance(r, BaseException):
                raise r
            vectors.extend(r)
        return vectors

    async def _embed_subbatch_gateway(
        self, batch: list[str]
    ) -> list[list[float]]:
        # Lazy import: keeps the module-load cost down for callers that
        # never touch the gateway path (eval scripts, etc).
        from shared import llm as shared_llm

        # LiteLLM glob-routes `gemini-*` to the Gemini provider; pass the
        # explicit prefix so the proxy config doesn't have to alias the
        # bare id (and to keep parity with shared/llm.py's convention).
        model_id = (
            self._sdk_model
            if "/" in self._sdk_model
            else f"gemini/{self._sdk_model}"
        )
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await shared_llm.aembedding(model=model_id, input=batch)
            except shared_llm.LLMError as exc:
                translated = _translate_gateway_embedding_error(exc)
                if isinstance(translated, EmbeddingProviderUnavailable):
                    if attempt >= 3:
                        raise translated from exc
                    await asyncio.sleep(1 * attempt)
                    continue
                if isinstance(translated, EmbeddingRateLimited):
                    if attempt >= 5:
                        raise translated from exc
                    await asyncio.sleep(min(2**attempt, 30))
                    continue
                raise translated from exc

            # LiteLLM normalizes embedding responses into OpenAI's shape:
            # `.data[i].embedding` is the vector (dict-style on some
            # provider/version combos, hence the dual lookup below).
            data = getattr(resp, "data", None)
            if data is None and isinstance(resp, dict):
                data = resp.get("data")
            data = data or []
            vectors: list[list[float]] = []
            for row in data:
                vec = getattr(row, "embedding", None)
                if vec is None and isinstance(row, dict):
                    vec = row.get("embedding")
                if vec is None:
                    raise EmbeddingBatchRejected(
                        "gateway response missing embedding values"
                    )
                vectors.append(list(vec))
            return vectors


def _translate_gateway_embedding_error(exc: Any) -> Exception:
    """Map ``shared.llm.LLMError`` (which wraps any LiteLLM exception)
    into our embedding error taxonomy, mirroring the direct-SDK paths
    (``_translate_gemini_error`` + the OpenAI inline branches above)."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if isinstance(status, int):
        if status == 429:
            return EmbeddingRateLimited(str(exc))
        if status >= 500:
            return EmbeddingProviderUnavailable(f"gateway {status}: {exc}")
    if "rate" in msg and "limit" in msg:
        return EmbeddingRateLimited(str(exc))
    if "timeout" in msg or "deadline" in msg or "unavailable" in msg:
        return EmbeddingProviderUnavailable(str(exc))
    if (
        "context" in msg
        or "input is too long" in msg
        or ("exceed" in msg and "token" in msg)
        or "maximum context length" in msg
    ):
        return EmbeddingContextLengthExceeded(str(exc))
    return EmbeddingBatchRejected(str(exc))


def _format_gemini_document(item: DocItem) -> str:
    # Cap title length so a long title can't push the prefixed input past
    # Gemini's per-request token ceiling (~2048). The chunker already caps
    # raw content; the title is the only other variable in the prefix.
    # 200 chars ~= 50 tokens worst case, leaving headroom.
    title = (item.title or "").strip()[:200]
    # Strip the prefix's own separator so a title containing "| text:" can't
    # shift content boundaries when the formatted string is later parsed by
    # any downstream tooling.
    title = title.replace("| text:", "").strip()
    if title:
        return _GEMINI_DOC_PREFIX_WITH_TITLE.format(title=title, content=item.content)
    return _GEMINI_DOC_PREFIX_NO_TITLE.format(content=item.content)


def _translate_gemini_error(exc: BaseException) -> Exception:
    """Map google-genai errors into our embedding error taxonomy."""
    msg = str(exc).lower()
    name = type(exc).__name__
    # Status code if the SDK exposes one. Different google-genai versions use
    # different attribute names; check both.
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(exc, "status", None)
    )
    if isinstance(status, str) and status.isdigit():
        status = int(status)

    if ("rate" in msg and "limit" in msg) or status == 429 or "ResourceExhausted" in name:
        return EmbeddingRateLimited(str(exc))
    if "deadline" in msg or "timeout" in msg or "DeadlineExceeded" in name:
        return EmbeddingProviderUnavailable(f"timeout: {exc}")
    if isinstance(status, int) and status >= 500:
        return EmbeddingProviderUnavailable(f"gemini {status}: {exc}")
    if "ServiceUnavailable" in name or "Unavailable" in name:
        return EmbeddingProviderUnavailable(str(exc))
    if (
        "context" in msg
        or "input is too long" in msg
        or ("exceed" in msg and "token" in msg)
    ):
        return EmbeddingContextLengthExceeded(str(exc))
    return EmbeddingBatchRejected(str(exc))


# ---- module-level singletons -------------------------------------------

_embedder: OpenAIEmbedder | None = None
_embedder_v2: GeminiEmbedder | None = None


def get_embedder() -> OpenAIEmbedder:
    """OpenAI embedder singleton. EVAL-HARNESS ONLY — see module docstring.
    Production callers (normalizer, retrievers, synthesis) use
    :func:`get_embedder_v2`."""
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbedder()
    return _embedder


def get_embedder_v2() -> GeminiEmbedder:
    """Gemini embedder singleton. Production embedder for ingestion and
    retrieval (cutover 2026-05-14)."""
    global _embedder_v2
    if _embedder_v2 is None:
        _embedder_v2 = GeminiEmbedder()
    return _embedder_v2


def reset_embedder() -> None:
    global _embedder, _embedder_v2
    _embedder = None
    _embedder_v2 = None


__all__ = [
    "DocItem",
    "EmbedResult",
    "EmbeddedChunk",
    "FailedChunk",
    "GeminiEmbedder",
    "OpenAIEmbedder",
    "get_embedder",
    "get_embedder_v2",
    "reset_embedder",
]


def _hash_vector(text: str, dim: int) -> list[float]:
    """Deterministic unit vector derived from SHA-256 of the text.

    Stub-mode embedding when no provider key is configured. Good enough for
    integration tests -- similar strings map to similar vectors because we
    seed the first bytes from the hash and fill the rest with a derived PRNG.
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    x = int.from_bytes(seed[:8], "big", signed=False) or 1
    vec: list[float] = []
    for _ in range(dim):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        vec.append(((x >> 32) / (1 << 31)) - 1.0)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
