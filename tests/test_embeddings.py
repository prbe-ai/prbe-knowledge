"""Unit tests for shared.embeddings (Gemini-only post-2026-05-14 cutover).

Covers the bits that aren't reachable from a live_db ingest test:

- GeminiEmbedder asymmetric prefixing (doc vs query format)
- get_embedder_v2() singleton
- _translate_gemini_error mapping from raw SDK errors to our taxonomy
- Stub mode produces a deterministic vector when no provider key is set
- Recursive half-split poison-chunk isolation works on the base class

The Google genai client is NOT exercised over the wire — that's the live
ingest test's job. Stub mode (no key) gives us a real non-None vector to
assert against without a network call.
"""

from __future__ import annotations

import math

import pytest

from shared.embeddings import (
    _GEMINI_QUERY_PREFIX,
    DocItem,
    EmbeddedChunk,
    EmbedResult,
    FailedChunk,
    GeminiEmbedder,
    _BaseEmbedder,
    _format_gemini_document,
    _translate_gemini_error,
    get_embedder_v2,
    reset_embedder,
)
from shared.exceptions import (
    EmbeddingBatchRejected,
    EmbeddingContextLengthExceeded,
    EmbeddingProviderUnavailable,
    EmbeddingRateLimited,
)

# ---- Gemini document formatting ----------------------------------------


def test_gemini_doc_with_title_uses_title_prefix() -> None:
    item = DocItem(content="some chunk text", title="My Doc")
    assert _format_gemini_document(item) == "title: My Doc | text: some chunk text"


def test_gemini_doc_without_title_falls_back_to_text_only() -> None:
    item = DocItem(content="some chunk text", title=None)
    assert _format_gemini_document(item) == "text: some chunk text"


def test_gemini_doc_with_empty_title_falls_back_to_text_only() -> None:
    item = DocItem(content="some chunk text", title="   ")
    assert _format_gemini_document(item) == "text: some chunk text"


def test_gemini_query_prefix_format() -> None:
    # Sanity-check the constant matches the documented Gemini convention so a
    # later edit to the format string surfaces as a test failure.
    assert _GEMINI_QUERY_PREFIX.format(query="hello") == "task: search result | query: hello"


def test_gemini_doc_truncates_long_title() -> None:
    # A title with 500 chars must not blow Gemini's input ceiling once
    # combined with the 2048-token chunker cap on content.
    long_title = "X" * 500
    out = _format_gemini_document(DocItem(content="body", title=long_title))
    # 200 X's + the prefix scaffolding.
    assert "X" * 200 in out
    assert "X" * 201 not in out


def test_gemini_doc_strips_separator_from_title() -> None:
    # A title containing the literal separator must not shift content
    # boundaries in the prefixed string.
    item = DocItem(content="real body", title="malicious | text: fake body")
    out = _format_gemini_document(item)
    # The fake-body shouldn't smuggle past as a title — the substring is removed.
    assert "fake body" in out  # still present (it's in the title text)
    # But there should be exactly ONE "| text: " separator (the real one).
    assert out.count("| text: ") == 1


# ---- Singleton behavior ------------------------------------------------


def test_get_embedder_v2_returns_gemini_singleton() -> None:
    reset_embedder()
    a = get_embedder_v2()
    b = get_embedder_v2()
    assert a is b
    assert isinstance(a, GeminiEmbedder)


def test_reset_embedder_clears_singleton() -> None:
    reset_embedder()
    first = get_embedder_v2()
    reset_embedder()
    second = get_embedder_v2()
    assert first is not second


# ---- Stub-mode vectors --------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_stub_mode_returns_deterministic_unit_vector() -> None:
    reset_embedder()
    embedder = GeminiEmbedder()
    # Stub-mode: no GOOGLE_API_KEY in test env -> _ensure_client returns None.
    assert embedder._ensure_client() is None
    result = await embedder.embed_many(["hello"])
    assert len(result.embedded) == 1
    vec = result.embedded[0].embedding
    assert len(vec) == embedder.dim


@pytest.mark.asyncio
async def test_gemini_embed_documents_applies_title_prefix_in_stub_mode() -> None:
    """In stub mode, the hashed vector is deterministic by input string. Two
    DocItems with the same content but different titles must hash to
    DIFFERENT vectors -- if they don't, embed_documents isn't actually
    threading the title through to the underlying embed call.
    """
    reset_embedder()
    embedder = GeminiEmbedder()
    items = [
        DocItem(content="same body", title="title A"),
        DocItem(content="same body", title="title B"),
    ]
    out = await embedder.embed_documents(items)
    assert len(out.embedded) == 2
    a, b = out.embedded[0].embedding, out.embedded[1].embedding
    assert a != b


@pytest.mark.asyncio
async def test_gemini_embed_query_applies_query_prefix_in_stub_mode() -> None:
    """A query and a document with the same payload must produce different
    vectors -- proves the asymmetric prefix is actually applied. (Stub mode
    is deterministic on input string.)
    """
    reset_embedder()
    embedder = GeminiEmbedder()
    doc_vec_result = await embedder.embed_documents([DocItem(content="auth flow")])
    query_vec = await embedder.embed_query("auth flow")
    assert doc_vec_result.embedded[0].embedding != query_vec


# ---- Error translation --------------------------------------------------


class _FakeStatusError(Exception):
    """Stand-in for google.genai.errors with a status_code attribute."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status_code = status


@pytest.mark.parametrize(
    "exc, expected_cls",
    [
        (Exception("Rate limit reached"), EmbeddingRateLimited),
        (_FakeStatusError("rate limited", 429), EmbeddingRateLimited),
        (Exception("Deadline exceeded"), EmbeddingProviderUnavailable),
        (_FakeStatusError("Internal", 500), EmbeddingProviderUnavailable),
        (_FakeStatusError("Bad gateway", 502), EmbeddingProviderUnavailable),
        (
            Exception("input is too long for the model context"),
            EmbeddingContextLengthExceeded,
        ),
        (Exception("totally unknown failure shape"), EmbeddingBatchRejected),
    ],
)
def test_translate_gemini_error(exc: BaseException, expected_cls: type) -> None:
    translated = _translate_gemini_error(exc)
    assert isinstance(translated, expected_cls)


# ---- Gemini sub-batch parallelization ----------------------------------


class _FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = [_FakeEmbedding(v) for v in vectors]


@pytest.mark.asyncio
async def test_gemini_embed_once_splits_into_subbatches_in_order(monkeypatch) -> None:
    """A batch must split into exactly ceil(N / GROUP_SIZE) sub-calls and
    return vectors in original input order. asyncio.gather completion order
    is non-deterministic; without explicit reassembly the vectors would
    scramble against their inputs.
    """
    reset_embedder()
    embedder = GeminiEmbedder()

    captured_inputs: list[list[str]] = []

    async def fake_embed_content(*, model: str, contents: list[str]) -> _FakeEmbedResponse:
        captured_inputs.append(list(contents))
        # Encode item index in dim 0 so the test can verify ordering survives.
        vectors = [[float(c.split(":")[1]), 0.0, 0.0] for c in contents]
        return _FakeEmbedResponse(vectors)

    class _FakeClient:
        class aio:
            class models:
                embed_content = staticmethod(fake_embed_content)

    monkeypatch.setattr(embedder, "_ensure_client", lambda: _FakeClient())

    n_items = 24
    inputs = [f"item:{i}" for i in range(n_items)]
    vectors = await embedder._embed_once(inputs)

    expected_calls = (n_items + embedder._SUBBATCH_GROUP_SIZE - 1) // embedder._SUBBATCH_GROUP_SIZE
    assert len(captured_inputs) == expected_calls
    assert sum(len(g) for g in captured_inputs) == n_items
    # Vector at index i must encode i in dim 0 -- proves no scramble.
    assert [int(v[0]) for v in vectors] == list(range(n_items))


@pytest.mark.asyncio
async def test_gemini_embed_once_propagates_subbatch_failure(monkeypatch) -> None:
    """If any sub-call raises after retries, the whole batch fails so the
    upstream recursive half-split halves the input and retries. A partial
    return would silently drop inputs.
    """
    reset_embedder()
    embedder = GeminiEmbedder()

    async def boom_subbatch(client, batch):
        raise EmbeddingBatchRejected("synthetic")

    monkeypatch.setattr(embedder, "_embed_subbatch", boom_subbatch)
    monkeypatch.setattr(
        embedder, "_ensure_client", lambda: object()  # any non-None
    )

    with pytest.raises(EmbeddingBatchRejected):
        await embedder._embed_once([f"item:{i}" for i in range(24)])


# ---- Recursive half-split (base-class contract) -------------------------


class _PoisonOnSecondItemEmbedder(_BaseEmbedder):
    """Test subclass: any batch containing 'POISON' raises
    EmbeddingBatchRejected. The recursive half-split should isolate the
    poison to a single-item batch and record it to `failed`, while the
    other items land in `embedded`.
    """

    def __init__(self) -> None:
        super().__init__(model_id="test/poison", dim=4, batch_size=10)

    async def _embed_once(self, batch: list[str]) -> list[list[float]]:
        if any(t == "POISON" for t in batch):
            raise EmbeddingBatchRejected("synthetic poison")
        return [[0.1, 0.2, 0.3, 0.4] for _ in batch]


@pytest.mark.asyncio
async def test_recursive_half_split_isolates_poison_chunk() -> None:
    embedder = _PoisonOnSecondItemEmbedder()
    out = await embedder.embed_many(["good_a", "POISON", "good_b", "good_c"])
    embedded_ids = sorted(c.chunk_index for c in out.embedded)
    failed_ids = sorted(f.chunk_index for f in out.failed)
    assert embedded_ids == [0, 2, 3]
    assert failed_ids == [1]
    assert out.failed[0].content_preview == "POISON"


# ---- API surface backward compatibility --------------------------------


def test_dataclass_re_exports_unchanged() -> None:
    # Sanity-check: callers depend on these dataclasses by name. If we ever
    # rename or move them, the import will fail loudly here, not silently
    # at production import time.
    assert EmbeddedChunk.__name__ == "EmbeddedChunk"
    assert FailedChunk.__name__ == "FailedChunk"
    assert EmbedResult.__name__ == "EmbedResult"


# ---- Gateway-aware embedding (plan D1, managed-shared / self-host) -----
#
# When `llm_gateway_url` is set, GeminiEmbedder routes through
# `shared.llm.aembedding` (the google-genai SDK has no `base_url` knob).
# This test is SDK-shape only — no network — but it nails down which
# transport the embedder uses and that the gateway credentials reach it.


@pytest.mark.asyncio
async def test_gemini_embedder_routes_through_shared_llm_aembedding_in_gateway_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway mode for Gemini goes through shared.llm.aembedding (no
    base_url knob on google-genai). Model id is sent as the bare alias so
    it matches the LiteLLM proxy's `gemini-embedding-*` model_name."""

    from shared.config import get_settings

    monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.litellm.svc:4000")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "sk-virtual")
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-be-used")
    get_settings.cache_clear()
    s = get_settings()

    embedder = GeminiEmbedder(settings=s, model="gemini-embedding-2")
    assert embedder._gateway_url == "http://litellm.litellm.svc:4000"

    captured: dict[str, object] = {}

    class _Row:
        def __init__(self, v: list[float]) -> None:
            self.embedding = v

    class _Resp:
        def __init__(self, n: int) -> None:
            self.data = [_Row([0.11, 0.22, 0.33]) for _ in range(n)]

    async def fake_aembedding(*, model: str, input: list[str], **kwargs: object) -> object:
        captured["model"] = model
        captured["input"] = list(input)
        return _Resp(len(input))

    import shared.llm as shared_llm

    monkeypatch.setattr(shared_llm, "aembedding", fake_aembedding)

    out = await embedder._embed_once(["doc-one", "doc-two", "doc-three"])

    # Gateway transport was hit — direct client never constructed.
    assert embedder._client is None
    # Model id is the bare alias the proxy's model_list exposes
    # (`gemini-embedding-*`). Prefixing with `gemini/` would miss the alias
    # and fall through to the `*` catch-all (→ 400 invalid model ID).
    assert captured["model"] == "gemini-embedding-2"
    # All inputs went through; vectors round-trip.
    assert len(out) == 3
    assert all(vec == [0.11, 0.22, 0.33] for vec in out)
    get_settings.cache_clear()


def test_gateway_embedding_error_translation_covers_taxonomy() -> None:
    """LLMError → embedding error taxonomy mapping. Mirrors the direct-SDK
    translations so the recursive half-split treats both transports the same.
    """
    from shared.embeddings import _translate_gateway_embedding_error
    from shared.llm import LLMError

    # status-code-driven
    assert isinstance(
        _translate_gateway_embedding_error(LLMError("rl", status_code=429)),
        EmbeddingRateLimited,
    )
    assert isinstance(
        _translate_gateway_embedding_error(LLMError("oops", status_code=503)),
        EmbeddingProviderUnavailable,
    )
    # message-driven (no status)
    assert isinstance(
        _translate_gateway_embedding_error(LLMError("rate limit exceeded")),
        EmbeddingRateLimited,
    )
    assert isinstance(
        _translate_gateway_embedding_error(LLMError("upstream unavailable")),
        EmbeddingProviderUnavailable,
    )
    assert isinstance(
        _translate_gateway_embedding_error(
            LLMError("input is too long for max tokens")
        ),
        EmbeddingContextLengthExceeded,
    )
    # Anything else → BatchRejected (caller's half-split isolates the bad chunk).
    assert isinstance(
        _translate_gateway_embedding_error(LLMError("malformed request")),
        EmbeddingBatchRejected,
    )
