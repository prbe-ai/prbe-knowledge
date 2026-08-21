"""The situation classifier: free text in, one of the tenant's situations out.

WHY MOST OF THIS FILE USES A FAKE EMBEDDER, and why that is not laziness.

`tests/conftest.py` sets `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` to `""`, and
there is no `GOOGLE_API_KEY` in the test environment either. With no provider
key `engine.shared.embeddings.GeminiEmbedder._embed_once` falls back to
`_hash_vector`, a deterministic SHA-256-derived unit vector. Its docstring
claims "similar strings map to similar vectors"; THEY DO NOT. The whole vector
is a PRNG stream seeded from the first eight bytes of the digest, so one changed
character reseeds it completely. Measured at dim=768: "launching a training run"
vs. "Launching a training run" (one character) scores 0.020, while
"completely unrelated text about lunch" vs. that same string scores 0.020 as
well. Identical, because both numbers are just noise around zero.

So a test that asserts "this phrase classifies as `launch-run`" would, under the
stub, pass or fail by coincidence and would teach nobody anything. Worse, it
would go red for a change that IMPROVED accuracy. The decision logic and the
accuracy are therefore tested separately:

* Everything below the `REAL EMBEDDINGS` banner uses a fake embedder with
  hand-authored vectors, so the cosines are exact and the branch under test is
  the branch that runs. These always run and are the regression suite.
* The accuracy check under that banner is skipped unless a real embedder is
  configured. It is a table of (phrase, expected slug) precisely so it can be
  read as the calibration evidence for FLOOR and MARGIN when somebody retunes
  them.

WHAT THE FAKE-EMBEDDER TESTS HAVE TO DEFEND:

* `NO_VOCABULARY` IS NOT A FLAVOUR OF `UNKNOWN`. It is the last layer that can
  still tell "this tenant was never seeded" from "nothing matched"; downstream
  both are an empty response. It must also cost nothing: no embedding call, no
  LLM call.
* THE LLM IS NOT ON THE HOT PATH. An input that nothing comes close to must
  reach `UNKNOWN` WITHOUT an LLM call. That is a cost and latency guarantee, so
  it is asserted on the fake (was it called?), not inferred from the result.
* A TIE ESCALATES WITH ONLY THE TIED CANDIDATES. Handing the model all twelve
  would make the shortlist pointless and the prompt unreadable.
* PARTIAL EMBEDDER FAILURE MUST NOT SHIFT VECTORS. `EmbedResult` carries
  `embedded` and `failed` as two lists, so the i-th returned vector is NOT the
  i-th input. A reader that zips them by position silently attributes one
  situation's vector to another, which is an accuracy bug with a green suite.
* NOTHING RAISES INTO A SERVING PATH. Embedder down, LLM down, LLM babbling:
  all degrade to `UNKNOWN`.

Run with the isolated wfmem database (conftest enforces a localhost host):

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:5432/prbe_knowledge_wfmem \
        .venv/bin/pytest tests/test_workflow_memory_classifier.py -q
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
import structlog.testing

from engine.shared.config import get_settings
from engine.shared.db import raw_conn, with_tenant
from engine.shared.embeddings import EmbeddedChunk, EmbedResult, FailedChunk
from engine.shared.llm import LLMError
from engine.shared.wfmem.classifier import (
    _TIEBREAK_MAX_TOKENS,
    _parse_tiebreak_response,
    CLASSIFIER_PROMPT_VERSION,
    FLOOR,
    MARGIN,
    MAX_CACHED_VOCABULARIES,
    METHOD_EMBEDDING,
    METHOD_LLM,
    METHOD_NONE,
    TIEBREAK_MODEL,
    Classification,
    Outcome,
    _vocabulary_cache_key,
    _vocabulary_cache_put,
    classify,
    reset_vocabulary_cache,
    situation_embedding_text,
)
from engine.shared.wfmem.situations import seed_situations

TENANT_A = "cust-wfmem-clf-a"
TENANT_B = "cust-wfmem-clf-b"

#: The input every fake-embedder test classifies. Its content is irrelevant --
#: the fake maps it to a vector by exact string -- but a realistic one keeps the
#: prompt assertions readable.
QUERY_TEXT = "merging the reviewed branch and rolling it out to prod"

#: Cosine assigned to every situation a test does not name. Far enough below
#: FLOOR that it can never accidentally participate in a tie.
BACKGROUND_COS = 0.05

#: Expressed RELATIVE TO the constants under test, on purpose. FLOOR and MARGIN
#: are documented as provisional; hard-coding 0.80/0.60 here would turn a
#: deliberate retune into a suite of red boundary tests that say nothing about
#: whether the retune was right.
WELL_ABOVE = min(0.95, FLOOR + 0.20)
WELL_BELOW = max(0.0, FLOOR - 0.25)


# --------------------------------------------------------------------------
# Fixtures and fakes
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_vocabulary_cache() -> AsyncIterator[None]:
    """The cache is a module global and outlives a test.

    Its key contains `max(updated_at)`, which does change between tests
    (`live_db` truncates and each test re-seeds), so leakage is unlikely rather
    than impossible. "Unlikely" is not a property worth debugging at 2am.
    """
    reset_vocabulary_cache()
    yield
    reset_vocabulary_cache()


@pytest_asyncio.fixture
async def tenants(live_db: None) -> AsyncIterator[tuple[str, str]]:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'wfmem-clf-a', 'h-wfmem-clf-a'),
                   ($2, 'wfmem-clf-b', 'h-wfmem-clf-b')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            TENANT_A,
            TENANT_B,
        )
    yield TENANT_A, TENANT_B


@pytest_asyncio.fixture
async def seeded(tenants: tuple[str, str]) -> AsyncIterator[str]:
    """Tenant A with the stock twelve-situation vocabulary."""
    tenant, _ = tenants
    async with with_tenant(tenant) as conn:
        await seed_situations(conn, tenant)
    yield tenant


async def _read_situations(customer_id: str) -> list[dict[str, Any]]:
    """Slug order, matching what the classifier loads.

    Explicit tenant predicate rather than leaning on the GUC: the dev role is a
    SUPERUSER and bypasses RLS, so a bare SELECT under `with_tenant` would
    return every tenant's rows and quietly corrupt the vector mapping.

    `classifiable` (0118) mirrors the classifier's own filter. Without it this
    helper returns the `misc` bucket too, and every test that maps a vector per
    row would be building a mapping one longer than the one under test -- which
    misaligns every vector after `misc` in slug order and would show up as
    inexplicable accuracy failures rather than as an off-by-one.
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT id, slug, label, description FROM situations "
            "WHERE customer_id = $1 AND classifiable ORDER BY slug",
            customer_id,
        )
    return [dict(r) for r in rows]


def _vec(cosine: float, slot: int, dim: int) -> list[float]:
    """A unit vector whose cosine against the query vector is exactly `cosine`.

    Dimension 0 carries the whole projection onto the query; `slot` (>= 1)
    carries the orthogonal remainder, one private dimension per situation. That
    makes every vector distinct (so a mix-up is visible) while keeping each
    cosine an exact, hand-chosen number rather than something to be discovered
    by running the test.
    """
    v = [0.0] * dim
    v[0] = cosine
    v[slot] = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return v


class FakeEmbedder:
    """Maps exact text -> hand-authored vector, and records what it was asked.

    `embed_many` is the only method the classifier uses. That is itself part of
    the contract under test: it is the only entry point on `_BaseEmbedder` that
    reports a partial failure as data (`EmbedResult.failed`) instead of raising,
    which is what a serving path needs.
    """

    def __init__(
        self,
        vectors: dict[str, list[float]],
        *,
        model_id: str = "fake-embedder-v1",
        fail_texts: frozenset[str] = frozenset(),
        raises: BaseException | None = None,
        reverse_order: bool = False,
    ) -> None:
        self.model_id = model_id
        self._vectors = vectors
        self._fail = fail_texts
        self._raises = raises
        self._reverse = reverse_order
        #: One entry per `embed_many` call, in order.
        self.batches: list[list[str]] = []

    async def embed_many(self, texts: list[str]) -> EmbedResult:
        self.batches.append(list(texts))
        if self._raises is not None:
            raise self._raises
        embedded: list[EmbeddedChunk] = []
        failed: list[FailedChunk] = []
        for index, text in enumerate(texts):
            if text in self._fail:
                failed.append(
                    FailedChunk(chunk_index=index, content_preview=text[:200], error="stub failure")
                )
                continue
            assert text in self._vectors, f"FakeEmbedder has no vector for {text!r}"
            embedded.append(EmbeddedChunk(chunk_index=index, embedding=self._vectors[text]))
        if self._reverse:
            # A real embedder is under no obligation to return in input order --
            # the half-split recursion appends as it resolves. Reversing here is
            # how the suite proves the reader keys on `chunk_index`.
            embedded.reverse()
        return EmbedResult(embedded=embedded, failed=failed)

    @property
    def situation_batches(self) -> list[list[str]]:
        """Batches that were the vocabulary, not the one-item query batch."""
        return [b for b in self.batches if len(b) > 1 or (b and b[0] != QUERY_TEXT)]


async def _embedder_for(
    customer_id: str,
    cosines: dict[str, float],
    *,
    query_text: str = QUERY_TEXT,
    **kwargs: Any,
) -> FakeEmbedder:
    """A fake whose vectors give each named slug exactly the cosine asked for."""
    rows = await _read_situations(customer_id)
    dim = len(rows) + 1
    vectors: dict[str, list[float]] = {query_text: _vec(1.0, 1, dim)}
    for slot, row in enumerate(rows, start=1):
        text = situation_embedding_text(row["label"], row["description"])
        vectors[text] = _vec(cosines.get(row["slug"], BACKGROUND_COS), slot, dim)
    return FakeEmbedder(vectors, **kwargs)


class FakeCompletion:
    """Stand-in for `engine.shared.llm.acompletion`, counting every call.

    The call COUNT is the point in half these tests: "did not reach the LLM" is
    a cost guarantee and cannot be read off the returned `Classification`.
    """

    def __init__(self, content: str | None = None, raises: BaseException | None = None) -> None:
        self._content = content
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )

    @property
    def prompt(self) -> str:
        """Everything the model was shown on the most recent call."""
        assert self.calls, "the LLM was never called"
        return "\n".join(str(m.get("content", "")) for m in self.calls[-1]["messages"])


# --------------------------------------------------------------------------
# The contract other tasks are being written against
# --------------------------------------------------------------------------


def test_prompt_version_is_pinned() -> None:
    assert CLASSIFIER_PROMPT_VERSION == "1"


def test_outcome_is_a_three_valued_enum() -> None:
    """Three states, not two. See this module's docstring and situations.py."""
    assert {o.value for o in Outcome} == {"matched", "unknown", "no_vocabulary"}
    assert Outcome.NO_VOCABULARY is not Outcome.UNKNOWN


def test_classification_is_frozen() -> None:
    """It is a record of a decision already made; nothing may edit it in flight."""
    result = Classification(
        outcome=Outcome.UNKNOWN,
        situation_id=None,
        slug=None,
        confidence=0.0,
        method=METHOD_NONE,
        model=None,
        prompt_version=None,
        runner_up=None,
    )
    with pytest.raises(FrozenInstanceError):
        result.slug = "deploy"  # type: ignore[misc]


def test_thresholds_are_named_constants_in_range() -> None:
    assert 0.0 < FLOOR < 1.0
    assert 0.0 < MARGIN < 1.0


# --------------------------------------------------------------------------
# The shared text helper
# --------------------------------------------------------------------------


def test_situation_embedding_text_is_label_newline_description() -> None:
    assert situation_embedding_text("Deploying", "Putting it where others depend") == (
        "Deploying\nPutting it where others depend"
    )


async def test_the_classifier_embeds_exactly_the_shared_helpers_text(seeded: str) -> None:
    """Two call sites formatting this differently is an accuracy bug with no
    failing test of its own -- so the format is asserted where it is USED."""
    rows = await _read_situations(seeded)
    embedder = await _embedder_for(seeded, {})
    await classify(seeded, QUERY_TEXT, embedder=embedder, completion=FakeCompletion())

    assert len(embedder.situation_batches) == 1
    sent = embedder.situation_batches[0]
    assert sent == [situation_embedding_text(r["label"], r["description"]) for r in rows]


# --------------------------------------------------------------------------
# NO_VOCABULARY: the whole reason this module returns an enum
# --------------------------------------------------------------------------


async def test_no_vocabulary_is_distinct_from_unknown(tenants: tuple[str, str]) -> None:
    tenant, _ = tenants  # deliberately NOT seeded
    embedder = FakeEmbedder({})
    completion = FakeCompletion()

    result = await classify(tenant, "anything at all", embedder=embedder, completion=completion)

    assert result.outcome is Outcome.NO_VOCABULARY
    assert result.situation_id is None
    assert result.slug is None
    assert result.confidence == 0.0
    assert result.method == METHOD_NONE
    assert result.model is None
    assert result.prompt_version is None
    assert result.runner_up is None


async def test_no_vocabulary_costs_nothing(tenants: tuple[str, str]) -> None:
    """No embedding call and no LLM call. There is nothing to compare against."""
    tenant, _ = tenants
    embedder = FakeEmbedder({})
    completion = FakeCompletion()

    await classify(tenant, "anything at all", embedder=embedder, completion=completion)

    assert embedder.batches == []
    assert completion.calls == []


async def test_an_unknown_customer_reports_no_vocabulary(live_db: None) -> None:
    """A tenant that does not exist has no vocabulary, which is the honest answer.

    It is also the one that shows up in the audit query -- unlike an `UNKNOWN`,
    which would look like a normal miss forever.
    """
    embedder = FakeEmbedder({})
    result = await classify("cust-does-not-exist", "hello", embedder=embedder)
    assert result.outcome is Outcome.NO_VOCABULARY
    assert embedder.batches == []


async def test_a_seeded_tenant_never_reports_no_vocabulary(seeded: str) -> None:
    """The contrast that makes the enum worth having: same empty-handed answer,
    different reason."""
    embedder = await _embedder_for(seeded, {})  # everything at BACKGROUND_COS
    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=FakeCompletion())
    assert result.outcome is Outcome.UNKNOWN


# --------------------------------------------------------------------------
# The floor: below it, nothing is close, and we do not pay a model to agree
# --------------------------------------------------------------------------


async def test_below_the_floor_is_unknown_without_an_llm_call(seeded: str) -> None:
    """THE COST GUARANTEE. Asserted on the fake, because the returned
    `Classification` looks the same whether or not a model was billed."""
    embedder = await _embedder_for(seeded, {"deploy": WELL_BELOW})
    completion = FakeCompletion(content='{"slug": "deploy", "confidence": 0.99}')

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert completion.calls == [], "an input nothing came close to reached the LLM"
    assert result.outcome is Outcome.UNKNOWN
    assert result.slug is None
    assert result.situation_id is None
    assert result.confidence == 0.0
    assert result.method == METHOD_EMBEDDING
    assert result.model == embedder.model_id
    assert result.prompt_version is None


async def test_two_near_misses_below_the_floor_do_not_escalate(seeded: str) -> None:
    """A tie BELOW the floor is still nothing. The margin rule only applies once
    something has cleared the bar; otherwise every ambiguous nothing pays for a
    model call."""
    embedder = await _embedder_for(
        seeded, {"deploy": FLOOR - 0.01, "incident-recovery": FLOOR - 0.02}
    )
    completion = FakeCompletion(content='{"slug": "deploy", "confidence": 0.9}')

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert completion.calls == []
    assert result.outcome is Outcome.UNKNOWN
    assert result.method == METHOD_EMBEDDING


async def test_exactly_at_the_floor_with_margin_matches(seeded: str) -> None:
    """The bar is `>=`, both halves. Written out because an off-by-one here is
    invisible: it just makes the classifier very slightly stingier forever."""
    embedder = await _embedder_for(seeded, {"deploy": FLOOR, "incident-recovery": FLOOR - MARGIN})
    completion = FakeCompletion()

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert completion.calls == []
    assert result.outcome is Outcome.MATCHED
    assert result.slug == "deploy"
    assert result.confidence == pytest.approx(FLOOR)


# --------------------------------------------------------------------------
# The margin: a clear winner needs no help
# --------------------------------------------------------------------------


async def test_a_clear_winner_matches_on_embeddings_alone(seeded: str) -> None:
    embedder = await _embedder_for(
        seeded,
        {"deploy": WELL_ABOVE, "incident-recovery": WELL_ABOVE - MARGIN * 3},
    )
    completion = FakeCompletion()
    rows = {r["slug"]: r["id"] for r in await _read_situations(seeded)}

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert completion.calls == [], "a clear winner paid for an LLM call"
    assert result.outcome is Outcome.MATCHED
    assert result.slug == "deploy"
    assert result.situation_id == rows["deploy"]
    assert result.confidence == pytest.approx(WELL_ABOVE)
    assert result.method == METHOD_EMBEDDING
    assert result.model == embedder.model_id
    assert result.prompt_version is None
    assert result.runner_up == "incident-recovery"


async def test_returned_vectors_are_keyed_by_chunk_index_not_position(seeded: str) -> None:
    """`EmbedResult.embedded` is not guaranteed to be in input order.

    The half-split recursion in `_BaseEmbedder` appends sub-batches as they
    resolve. A reader that zips `embedded` against its own input list attributes
    one situation's vector to another -- and every threshold test above still
    passes, because the numbers are all still there, just on the wrong rows.
    """
    embedder = await _embedder_for(
        seeded,
        {"deploy": WELL_ABOVE, "incident-recovery": WELL_ABOVE - MARGIN * 3},
        reverse_order=True,
    )

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=FakeCompletion())

    assert result.outcome is Outcome.MATCHED
    assert result.slug == "deploy", "vectors were matched to situations by list position"


# --------------------------------------------------------------------------
# The tie-break: the only path that costs an LLM call
# --------------------------------------------------------------------------


async def test_a_tie_escalates_with_only_the_tied_candidates(seeded: str) -> None:
    embedder = await _embedder_for(
        seeded,
        {
            "deploy": WELL_ABOVE,
            "incident-recovery": WELL_ABOVE - MARGIN / 2,
            # Above the floor, but three margins back: separable, so it has no
            # business in the shortlist.
            "open-pr": max(FLOOR, WELL_ABOVE - MARGIN * 3),
        },
    )
    completion = FakeCompletion(content='{"slug": "incident-recovery", "confidence": 0.82}')
    rows = {r["slug"]: r["id"] for r in await _read_situations(seeded)}

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert len(completion.calls) == 1
    prompt = completion.prompt
    assert "deploy" in prompt
    assert "incident-recovery" in prompt
    for excluded in ("open-pr", "launch-run", "run-eval", "process-dataset"):
        assert excluded not in prompt, f"{excluded} was not tied but reached the prompt"
    assert QUERY_TEXT in prompt

    assert result.outcome is Outcome.MATCHED
    assert result.slug == "incident-recovery"
    assert result.situation_id == rows["incident-recovery"]
    assert result.confidence == pytest.approx(0.82)
    assert result.method == METHOD_LLM
    assert result.model == TIEBREAK_MODEL
    assert result.prompt_version == CLASSIFIER_PROMPT_VERSION
    # The one it nearly picked -- which is the top-cosine candidate it passed
    # over, not the second-cosine one.
    assert result.runner_up == "deploy"


async def test_a_three_way_tie_sends_all_three(seeded: str) -> None:
    embedder = await _embedder_for(
        seeded,
        {
            "deploy": WELL_ABOVE,
            "incident-recovery": WELL_ABOVE - MARGIN / 3,
            "debug-failing-run": WELL_ABOVE - MARGIN / 2,
        },
    )
    completion = FakeCompletion(content='{"slug": "deploy", "confidence": 0.7}')

    await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    prompt = completion.prompt
    for tied in ("deploy", "incident-recovery", "debug-failing-run"):
        assert tied in prompt
    assert "provision-infra" not in prompt


async def test_the_llm_answering_none_yields_unknown(seeded: str) -> None:
    """ "None of these" is a correct answer and must survive as one.

    A tie-break that could only ever return one of its candidates would turn
    every close pair into a forced choice, which is how a classifier starts
    serving confident nonsense.
    """
    embedder = await _embedder_for(
        seeded, {"deploy": WELL_ABOVE, "incident-recovery": WELL_ABOVE - MARGIN / 2}
    )
    completion = FakeCompletion(content='{"slug": null, "confidence": 0.0}')

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert len(completion.calls) == 1
    assert result.outcome is Outcome.UNKNOWN
    assert result.slug is None
    assert result.situation_id is None
    assert result.confidence == 0.0
    assert result.method == METHOD_LLM
    assert result.model == TIEBREAK_MODEL
    assert result.prompt_version == CLASSIFIER_PROMPT_VERSION


async def test_the_llm_may_report_its_own_confidence(seeded: str) -> None:
    embedder = await _embedder_for(
        seeded, {"deploy": WELL_ABOVE, "incident-recovery": WELL_ABOVE - MARGIN / 2}
    )
    completion = FakeCompletion(content='{"slug": "deploy", "confidence": 0.41}')

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert result.confidence == pytest.approx(0.41)


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"slug": "deploy", "confidence": 0.9}\n```',
        'Sure! {"slug": "deploy", "confidence": 0.9}',
        '{"slug": "deploy"}',
    ],
)
async def test_the_tie_break_tolerates_the_usual_model_packaging(seeded: str, content: str) -> None:
    """Fences and a preamble are what models actually emit; neither is an error.

    A missing `confidence` is not either -- it means the model answered the
    question it was asked and skipped the one it was not obliged to.
    """
    embedder = await _embedder_for(
        seeded, {"deploy": WELL_ABOVE, "incident-recovery": WELL_ABOVE - MARGIN / 2}
    )
    result = await classify(
        seeded, QUERY_TEXT, embedder=embedder, completion=FakeCompletion(content=content)
    )
    assert result.outcome is Outcome.MATCHED
    assert result.slug == "deploy"
    assert 0.0 < result.confidence <= 1.0


# --------------------------------------------------------------------------
# Failure paths: nothing raises into a serving path
# --------------------------------------------------------------------------


async def test_an_llm_error_degrades_to_unknown(seeded: str) -> None:
    embedder = await _embedder_for(
        seeded, {"deploy": WELL_ABOVE, "incident-recovery": WELL_ABOVE - MARGIN / 2}
    )
    completion = FakeCompletion(raises=LLMError("gateway 503", status_code=503))

    with structlog.testing.capture_logs() as logs:
        result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert result.outcome is Outcome.UNKNOWN
    assert result.slug is None
    # `none`, not `llm`: no stage produced this answer, we defaulted. That is
    # what separates a lost escalation from a model that genuinely said "none".
    assert result.method == METHOD_NONE
    assert [e for e in logs if e["event"] == "wfmem_classifier.tiebreak_failed"], (
        f"a swallowed LLM failure must be visible; captured: {logs}"
    )


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "I think it is probably the deploy one",
        "{}",
        '{"slug": "run-eval", "confidence": 0.9}',  # a real slug, not a candidate
        '{"slug": "not-a-situation"}',
        "[1, 2, 3]",
    ],
)
async def test_an_unusable_tie_break_response_degrades_to_unknown(
    seeded: str, content: str | None
) -> None:
    """Including an off-menu slug. The shortlist is the permitted answer set --
    a model that returns something else has not broken the tie, and honouring it
    would let the LLM overrule the floor it was never shown."""
    embedder = await _embedder_for(
        seeded, {"deploy": WELL_ABOVE, "incident-recovery": WELL_ABOVE - MARGIN / 2}
    )
    result = await classify(
        seeded, QUERY_TEXT, embedder=embedder, completion=FakeCompletion(content=content)
    )
    assert result.outcome is Outcome.UNKNOWN
    assert result.slug is None
    assert result.method == METHOD_NONE


async def test_a_partial_embedder_failure_degrades_to_unknown(seeded: str) -> None:
    """One `FailedChunk` in the vocabulary batch and the whole thing is void.

    The alternative -- classify against the eleven that worked -- is worse than
    it sounds: the missing one is silently unreachable for as long as the
    failure persists, and nothing downstream can tell that from "no rule for
    this situation".
    """
    rows = await _read_situations(seeded)
    poisoned = situation_embedding_text(rows[0]["label"], rows[0]["description"])
    embedder = await _embedder_for(
        seeded,
        {"deploy": WELL_ABOVE, "incident-recovery": WELL_BELOW},
        fail_texts=frozenset({poisoned}),
    )
    completion = FakeCompletion(content='{"slug": "deploy"}')

    with structlog.testing.capture_logs() as logs:
        result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert result.outcome is Outcome.UNKNOWN
    assert result.method == METHOD_NONE
    assert result.model is None
    assert completion.calls == []
    assert [e for e in logs if e["event"] == "wfmem_classifier.vocabulary_embedding_incomplete"], (
        f"a partial embedding failure must be visible; captured: {logs}"
    )


async def test_a_partial_failure_is_not_cached(seeded: str) -> None:
    """A transient provider hiccup must not pin the tenant to `UNKNOWN` for the
    life of the process. The cache holds answers, not failures."""
    rows = await _read_situations(seeded)
    poisoned = situation_embedding_text(rows[0]["label"], rows[0]["description"])
    broken = await _embedder_for(seeded, {"deploy": WELL_ABOVE}, fail_texts=frozenset({poisoned}))
    assert (
        await classify(seeded, QUERY_TEXT, embedder=broken, completion=FakeCompletion())
    ).outcome is Outcome.UNKNOWN

    healthy = await _embedder_for(seeded, {"deploy": WELL_ABOVE}, model_id=broken.model_id)
    result = await classify(seeded, QUERY_TEXT, embedder=healthy, completion=FakeCompletion())

    assert result.outcome is Outcome.MATCHED
    assert healthy.situation_batches, "the failed attempt poisoned the cache"


async def test_an_embedder_that_raises_degrades_to_unknown(seeded: str) -> None:
    embedder = await _embedder_for(seeded, {}, raises=RuntimeError("provider is down"))
    completion = FakeCompletion(content='{"slug": "deploy"}')

    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=completion)

    assert result.outcome is Outcome.UNKNOWN
    assert result.method == METHOD_NONE
    assert result.model is None
    assert completion.calls == []


async def test_a_failing_query_embedding_degrades_to_unknown(seeded: str) -> None:
    """The vocabulary embedded fine; the input did not. Same answer."""
    embedder = await _embedder_for(
        seeded, {"deploy": WELL_ABOVE}, fail_texts=frozenset({QUERY_TEXT})
    )
    result = await classify(seeded, QUERY_TEXT, embedder=embedder, completion=FakeCompletion())

    assert result.outcome is Outcome.UNKNOWN
    assert result.method == METHOD_NONE


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
async def test_blank_text_is_unknown_without_any_model_call(seeded: str, text: str) -> None:
    """Not `NO_VOCABULARY` -- the vocabulary is there, the input is not.

    Checked AFTER the vocabulary load on purpose: a blank input from an
    unseeded tenant must still report the seeding problem, which is the more
    actionable of the two.
    """
    embedder = await _embedder_for(seeded, {})
    completion = FakeCompletion()

    result = await classify(seeded, text, embedder=embedder, completion=completion)

    assert result.outcome is Outcome.UNKNOWN
    assert result.method == METHOD_NONE
    assert embedder.batches == []
    assert completion.calls == []


# --------------------------------------------------------------------------
# The embedding cache
# --------------------------------------------------------------------------


async def test_the_vocabulary_is_embedded_once_per_process(seeded: str) -> None:
    embedder = await _embedder_for(seeded, {"deploy": WELL_ABOVE})

    for _ in range(3):
        await classify(seeded, QUERY_TEXT, embedder=embedder, completion=FakeCompletion())

    assert len(embedder.situation_batches) == 1, "the vocabulary was re-embedded"
    # The query is not cacheable -- it is different every time by definition.
    assert len(embedder.batches) == 4


async def test_editing_a_description_recomputes_the_vectors(seeded: str) -> None:
    """The reason there is no stored vector column.

    `situations` is per-tenant and editable. A stored embedding goes stale the
    moment somebody rewords a description and nothing tells it to; a cache keyed
    on `max(updated_at)` notices by itself. That only holds if the trigger
    `wfmem_touch_updated_at` actually fires on this table -- so this test is
    also the guard on that trigger, from the consumer's side.
    """
    first = await _embedder_for(seeded, {"deploy": WELL_ABOVE})
    await classify(seeded, QUERY_TEXT, embedder=first, completion=FakeCompletion())
    assert len(first.situation_batches) == 1

    edited = "Our team's wording for shipping a reviewed change to everyone else."
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE situations SET description = $2 WHERE customer_id = $1 AND slug = 'deploy'",
            seeded,
            edited,
        )

    second = await _embedder_for(seeded, {"deploy": WELL_ABOVE}, model_id=first.model_id)
    await classify(seeded, QUERY_TEXT, embedder=second, completion=FakeCompletion())

    assert len(second.situation_batches) == 1, "an edited description served a stale vector"
    assert any(edited in text for text in second.situation_batches[0])


async def test_deleting_a_situation_recomputes_the_vectors(seeded: str) -> None:
    """A delete moves no surviving row's `updated_at`.

    So `max(updated_at)` alone cannot see it, and a cache keyed on that alone
    keeps serving the deleted situation as a candidate -- a tenant who removed a
    situation on purpose would keep being classified into it.
    """
    first = await _embedder_for(seeded, {"deploy": WELL_ABOVE})
    await classify(seeded, QUERY_TEXT, embedder=first, completion=FakeCompletion())

    async with raw_conn() as conn:
        await conn.execute(
            "DELETE FROM situations WHERE customer_id = $1 AND slug = 'deploy'", seeded
        )

    second = await _embedder_for(seeded, {}, model_id=first.model_id)
    result = await classify(seeded, QUERY_TEXT, embedder=second, completion=FakeCompletion())

    assert len(second.situation_batches) == 1, "a deleted situation kept its cached vector"
    assert len(second.situation_batches[0]) == 11
    assert result.outcome is Outcome.UNKNOWN
    assert result.runner_up != "deploy"


async def test_a_different_embedder_model_recomputes_the_vectors(seeded: str) -> None:
    """Vectors from two models are not comparable, and a cosine between them is
    a number with no meaning rather than an error."""
    first = await _embedder_for(seeded, {"deploy": WELL_ABOVE}, model_id="model-one")
    await classify(seeded, QUERY_TEXT, embedder=first, completion=FakeCompletion())

    second = await _embedder_for(seeded, {"deploy": WELL_ABOVE}, model_id="model-two")
    await classify(seeded, QUERY_TEXT, embedder=second, completion=FakeCompletion())

    assert len(second.situation_batches) == 1


def test_the_cache_key_separates_two_tenants_with_identical_vocabularies() -> None:
    """Twelve seed situations written in the same transaction give two tenants
    the same size AND the same `max(updated_at)`. Without `customer_id` in the
    key they would share vectors -- which is only harmless until one of them
    edits a description."""
    stamp = datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC)
    a = _vocabulary_cache_key("cust-a", "model-1", 12, stamp)
    b = _vocabulary_cache_key("cust-b", "model-1", 12, stamp)
    assert a != b

    assert _vocabulary_cache_key("cust-a", "model-2", 12, stamp) != a
    assert _vocabulary_cache_key("cust-a", "model-1", 11, stamp) != a
    assert (
        _vocabulary_cache_key("cust-a", "model-1", 12, datetime(2026, 8, 19, 12, 0, 1, tzinfo=UTC))
        != a
    )
    assert _vocabulary_cache_key("cust-a", "model-1", 12, stamp) == a


async def test_one_tenants_edits_do_not_reach_another(tenants: tuple[str, str]) -> None:
    tenant_a, tenant_b = tenants
    for tenant in (tenant_a, tenant_b):
        async with with_tenant(tenant) as conn:
            await seed_situations(conn, tenant)
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE situations SET description = $2 WHERE customer_id = $1 AND slug = 'deploy'",
            tenant_b,
            "B's own wording, which is about something else entirely.",
        )

    embedder_a = await _embedder_for(tenant_a, {"deploy": WELL_ABOVE})
    result_a = await classify(
        tenant_a, QUERY_TEXT, embedder=embedder_a, completion=FakeCompletion()
    )

    # B's `deploy` text is different, so it lands on BACKGROUND_COS.
    embedder_b = await _embedder_for(tenant_b, {}, model_id=embedder_a.model_id)
    result_b = await classify(
        tenant_b, QUERY_TEXT, embedder=embedder_b, completion=FakeCompletion()
    )

    assert result_a.outcome is Outcome.MATCHED
    assert result_a.slug == "deploy"
    assert result_b.outcome is Outcome.UNKNOWN, "tenant A's vectors were served to tenant B"
    assert len(embedder_b.situation_batches) == 1


def test_the_cache_is_bounded() -> None:
    """Many tenants x many vocabulary versions is unbounded key space.

    An in-process dict keyed on that grows until the pod is OOM-killed, and it
    does it slowly enough that nobody connects the restart to this module.
    """
    reset_vocabulary_cache()
    stamp = datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC)
    seen = []
    for i in range(MAX_CACHED_VOCABULARIES * 4):
        key = _vocabulary_cache_key(f"cust-{i}", "model-1", 12, stamp)
        seen.append(key)
        _vocabulary_cache_put(key, [[1.0]])

    from engine.shared.wfmem.classifier import _VOCABULARY_CACHE

    assert len(_VOCABULARY_CACHE) == MAX_CACHED_VOCABULARIES
    # Least-recently-used goes first: the tail survived, the head did not.
    assert seen[-1] in _VOCABULARY_CACHE
    assert seen[0] not in _VOCABULARY_CACHE


# --------------------------------------------------------------------------
# REAL EMBEDDINGS -- opt-in, and the only thing here that measures accuracy
# --------------------------------------------------------------------------


def _real_embedder_configured() -> bool:
    """True only when a live Gemini embedder is reachable.

    Everything above runs against a fake precisely because the default test
    environment has no key, and `_hash_vector` (the stub the embedder falls back
    to) has NO semantic structure at all. Running the table below against it
    would produce a suite that goes green or red at random.
    """
    try:
        settings = get_settings()
    except Exception:  # pragma: no cover -- config problems are not this test's job
        return False
    secret = settings.google_api_key
    key = secret.get_secret_value() if secret is not None else ""
    return bool(key.strip() or settings.llm_gateway_url.strip())


#: The calibration table. When FLOOR/MARGIN are retuned, THIS is the evidence --
#: so keep it a flat list of (what somebody would actually type, expected slug)
#: and keep the phrases in the register a real transcript would be in, not in
#: the register the seed descriptions are written in. Paraphrasing the
#: description back at the classifier measures nothing.
REAL_PHRASES: tuple[tuple[str, str], ...] = (
    ("kicking off a full fine-tune of the 8b checkpoint on four H100s", "launch-run"),
    ("merging this to main and rolling it out to everyone", "deploy"),
    ("prod is down, paging through alerts and rolling back the last release", "incident-recovery"),
    ("writing the PR description and picking reviewers for the branch", "open-pr"),
    ("this test has been failing since yesterday, reading the stack trace", "debug-failing-run"),
    ("dedup and tokenize the raw crawl before we train on it", "process-dataset"),
    ("scoring the model against the held-out benchmark set", "run-eval"),
    ("leaving comments on someone else's diff and requesting changes", "review-code"),
    ("spinning up a new GPU node pool and a bucket for the artifacts", "provision-infra"),
    ("closing the ticket and telling the team it is done", "claim-done"),
    ("refactoring this module and moving the helper into shared", "edit-repo"),
    ("pulling up run 8812's config to see if I can hit the same number", "reproduce-experiment"),
)


@pytest.mark.skipif(
    not _real_embedder_configured(),
    reason="needs a real embedder (GOOGLE_API_KEY or LLM_GATEWAY_URL); the stub "
    "hash embedder has no semantic similarity, so this would assert on noise",
)
@pytest.mark.parametrize("phrase,expected", REAL_PHRASES, ids=[s for _, s in REAL_PHRASES])
async def test_real_embeddings_pick_the_expected_situation(
    seeded: str, phrase: str, expected: str
) -> None:
    result = await classify(seeded, phrase)
    assert result.outcome is Outcome.MATCHED, (
        f"{phrase!r} did not clear FLOOR={FLOOR}/MARGIN={MARGIN}; closest was {result.runner_up!r}"
    )
    assert result.slug == expected, (
        f"{phrase!r} -> {result.slug!r} (runner-up {result.runner_up!r})"
    )


@pytest.mark.skipif(
    not _real_embedder_configured(),
    reason="needs a real embedder; see test_real_embeddings_pick_the_expected_situation",
)
async def test_real_embeddings_reject_something_off_topic(seeded: str) -> None:
    """The floor has to actually refuse. A classifier that labels everything is
    worse than one that labels nothing: it serves the wrong team rules."""
    result = await classify(seeded, "booking a table for four on Friday evening")
    assert result.outcome is Outcome.UNKNOWN


# --------------------------------------------------------------------------
# The tie-break token budget
# --------------------------------------------------------------------------


def test_the_tiebreak_budget_leaves_room_for_a_thinking_model() -> None:
    """THE TIE-BREAK NEVER ONCE WORKED IN PRODUCTION, and this is why.

    `_TIEBREAK_MAX_TOKENS` was 256 -- generous for a reply that is ~15 tokens of
    JSON, and nowhere near enough for a THINKING model, whose reasoning is billed
    against the same budget before it emits a single visible character. Measured
    against the live model:

        max_tokens=256   completion_tokens=252   '```json\\n{\\n  "slug": null'
        max_tokens=2048  completion_tokens=533   '{"slug": null, "confidence": 0.0}'

    Every ambiguous classification therefore truncated, failed to parse, degraded
    to `unknown`, declined to attach a situation, and left the clause unreachable
    -- which is the bug the `misc` bucket exists to catch, arriving here from
    upstream.

    A pin rather than a proof: no unit test can measure another model's
    reasoning. What it CAN do is stop somebody trimming this back toward the
    cliff as an "obvious" saving, which is exactly how it was set in the first
    place.
    """
    assert _TIEBREAK_MAX_TOKENS >= 1024, (
        "a thinking model spends this budget on reasoning before answering; "
        "256 truncated every reply and the failure was silent"
    )


def test_a_truncated_reply_degrades_to_unknown_and_says_which_stage_failed() -> None:
    """The observed production failure, replayed byte for byte.

    `method` is the load-bearing half: `none` means no stage decided this and we
    defaulted, which is how a broken tie-break is told apart from a model that
    looked and honestly answered "none of these" (`llm`). Collapse the two and a
    permanently dead call reads as a model exercising judgement.
    """
    truncated = '```json\n{\n  "slug": null'

    assert _parse_tiebreak_response(truncated, allowed={"open-pr", "review-code"}) is None


def test_a_complete_reply_parses_whether_or_not_the_model_fences_it() -> None:
    """The model fences its JSON sometimes and not others -- both were observed
    in the same session. Fencing was never the bug, and this pins that so the
    next person debugging a tie-break does not go looking there."""
    allowed = {"open-pr", "review-code"}
    bare = '{"slug": "open-pr", "confidence": 0.8}'
    fenced = '```json\n{"slug": "open-pr", "confidence": 0.8}\n```'

    assert _parse_tiebreak_response(bare, allowed=allowed) == ("open-pr", 0.8)
    assert _parse_tiebreak_response(fenced, allowed=allowed) == ("open-pr", 0.8)
    # An honest abstention is a PARSE SUCCESS carrying no slug, not a failure.
    assert _parse_tiebreak_response(
        '{"slug": null, "confidence": 0.0}', allowed=allowed
    ) == (None, 0.0)
