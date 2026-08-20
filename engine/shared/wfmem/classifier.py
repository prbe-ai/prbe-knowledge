"""Free text -> one of the tenant's situations. Embedding-first, LLM to break ties.

WHY THE ANSWER IS AN ENUM AND NOT AN `Optional[str]`
----------------------------------------------------
There are THREE ways this can come back empty-handed and only two of them mean
the same thing to a reader:

* `MATCHED`       -- a situation cleared the bar.
* `UNKNOWN`       -- the tenant has a vocabulary; nothing in it matched.
* `NO_VOCABULARY` -- the tenant has no situations at all.

The third is the one this module exists to preserve. A tenant whose capability
got switched on without seeding classifies every input as unknown and serves
zero cards, and that failure is INVISIBLE: zero cards because there is no
vocabulary looks exactly like zero cards because no rule matched. Nothing
errors, nothing logs, the feature is quietly dead, and the customer reports "the
thing I turned on does nothing" weeks later. `engine.shared.wfmem.situations`
documents the same hazard at length -- it is why `enable_capability` seeds in
the same transaction and why `enabled_tenants_missing_situations` exists as the
backstop for the routes that bypass it.

The classifier is the LAST LAYER that can still tell the two apart. Downstream
of here both are an empty response and the distinction is unrecoverable. So it
is made here and carried in the return value, and a caller that collapses it
back into "no cards" is throwing away the only diagnostic anybody gets.

THE DECISION RULE, AND WHY THE LLM IS NOT ON THE HOT PATH
---------------------------------------------------------
The serving path has a sub-second target and an LLM call on every
classification would blow it. So:

1. Load the tenant's situations. Empty -> `NO_VOCABULARY`, no embedding call,
   no LLM call. There is nothing to compare against, and paying to discover
   that is pure waste.
2. Embed the input; cosine against each situation's embedding.
3. `MATCHED` on embeddings alone iff `top1 >= FLOOR` AND
   `(top1 - top2) >= MARGIN`. Both halves: a high score that two situations
   share is not a decision, it is a coin flip with a confident number attached.
4. `top1 < FLOOR` -> `UNKNOWN` WITHOUT an LLM call. Nothing is close; paying a
   model to confirm that is waste, and it is waste on the most common input
   (most of what a person types is not one of twelve named situations).
5. `top1 >= FLOOR` but the margin is short -> the LLM breaks the tie between
   ONLY the tied candidates, and may still answer "none".

BOTH SIDES ARE EMBEDDED SYMMETRICALLY, via `embed_many` rather than
`embed_query`/`embed_documents`. `GeminiEmbedder` implements asymmetric
retrieval by prefixing (`task: search result | query: ...` on one side,
`title: ... | text: ...` on the other), which is right when a short question is
being matched against a long document. That is not the shape here: both sides
are one short prose statement of an activity, compared to each other, and
pushing one of them through the query prefix would put the two halves in
deliberately different regions of a space whose FLOOR is a single number.
`embed_many` is also the only entry point that reports a partial failure as
DATA (`EmbedResult.failed`) instead of raising, which is what a serving path
needs.

FAILURE POSTURE: EVERYTHING DEGRADES TO `UNKNOWN`, NOTHING RAISES
------------------------------------------------------------------
Embedder down, embedder rejecting one chunk, LLM down, LLM babbling: all of
them return `UNKNOWN` and log. This is a read path feeding an agent's context,
and the cost of the two failure modes is wildly asymmetric -- a missing card is
a card the person writes themselves, an exception is a 500 on somebody's
search. `UNKNOWN` is the honest answer in every one of those cases: we do not
know what situation this is.

`method` names THE STAGE WHOSE JUDGEMENT IS IN THE RESULT -- `embedding`,
`llm`, or `none` for "no stage decided this, we defaulted". So a degraded
answer is distinguishable from a real one: an LLM that says "none of these" is
`UNKNOWN`/`llm`, while an LLM that timed out is `UNKNOWN`/`none`.

NO STORED VECTOR COLUMN, DELIBERATELY
-------------------------------------
`situations` is per-tenant and EDITABLE. A vector stored next to the row goes
silently stale the moment somebody rewords a description -- silently, because a
stale embedding still produces a plausible cosine, so the classifier just gets
quietly worse at that one situation and nothing says so. An in-process cache
keyed on `max(updated_at)` recomputes on its own. Twelve short texts is one
batched call per process per vocabulary version, which is cheap enough that the
column would be an optimization of something that was never the problem.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID

from engine.shared.constants import WFMEM_CLASSIFIER_TIEBREAK_MODEL
from engine.shared.db import with_tenant
from engine.shared.embeddings import get_embedder_v2
from engine.shared.logging import get_logger
from engine.shared.wfmem.llm_json import loads_forgiving, response_text

log = get_logger(__name__)

#: Bumped whenever the tie-break prompt changes in a way that could change its
#: answers. It is written into `clause_situation_edges.classification` so a
#: later reclassification pass can tell which edges came from which prompt --
#: the only reclassification input not derivable from the clause itself.
CLASSIFIER_PROMPT_VERSION = "1"

# PROVISIONAL. Both numbers are first guesses, not measurements: there is no
# labelled set to fit them to yet. The evidence that will replace them is the
# opt-in table in tests/test_workflow_memory_classifier.py
# (`REAL_PHRASES`), which runs only when a real embedder is configured -- under
# the stub hash embedder these thresholds are meaningless because there is no
# semantic structure to threshold. Retune there, with the accuracy numbers in
# hand, and expect FLOOR to move more than MARGIN.
#
#: Below this, nothing is close enough to be worth naming (or worth an LLM
#: call). Too low and the classifier labels off-topic text with a real
#: situation, which serves somebody another team's rules; too high and it goes
#: silent and the feature looks dead.
FLOOR = 0.55
#: How far top-1 must be clear of top-2 to count as a decision rather than a
#: coin flip. Several seed situations are deliberately adjacent (editing a repo
#: vs. debugging a failing test; deploying vs. recovering from what the deploy
#: broke), so near-ties are expected rather than exceptional.
MARGIN = 0.06

#: The tie-break model, re-exported from the house registry rather than spelled
#: here. Every other model choice in this service (`WIKI_TRIAGE_MODEL`,
#: `DIRECTED_PHRASES_MODEL`, `INFERRED_EDGES_MODEL`, `HAIKU_MODEL`) lives in
#: `engine.shared.constants`, and that file is where somebody auditing what we
#: call and what it costs will grep. A model id defined in a feature module is
#: invisible to that search.
#:
#: The alias stays because the tests and the tie-break path read better naming
#: the ROLE than the registry entry, and because it keeps the swap a one-line
#: change in constants.py.
TIEBREAK_MODEL = WFMEM_CLASSIFIER_TIEBREAK_MODEL

#: A tiny JSON object is the entire expected response. The ceiling exists so a
#: model that decides to explain itself gets cut off rather than billed.
_TIEBREAK_MAX_TOKENS = 256

#: Used when the tie-break model picks a candidate but reports no usable
#: confidence of its own. NOT a measurement and deliberately not a
#: precise-looking number: it means "a model said yes and told us nothing about
#: how sure it was". `method == "llm"` is what tells a reader the scale is not
#: the cosine scale.
LLM_UNREPORTED_CONFIDENCE = 0.5

#: `method` values. Named because they are written into a JSONB column that
#: outlives this module, and a typo in a string literal there is unfindable.
METHOD_EMBEDDING = "embedding"
METHOD_LLM = "llm"
#: No stage produced this answer -- we defaulted. `NO_VOCABULARY`, blank input,
#: and every degraded failure path.
METHOD_NONE = "none"

#: The cache holds one entry per (tenant, vocabulary version, embedder model).
#: That key space is unbounded across tenants and across edits, so the dict has
#: to evict: a process serving many tenants would otherwise grow until it is
#: OOM-killed, slowly enough that nobody connects the restart to this module.
#: 32 is a guess sized for "the tenants one pod is actually serving", not a
#: tuned number.
MAX_CACHED_VOCABULARIES = 32


class Outcome(StrEnum):
    """The three answers. See this module's docstring for why there are three."""

    MATCHED = "matched"
    #: The tenant HAS situations and none of them matched.
    UNKNOWN = "unknown"
    #: The tenant has no situations at all -- almost certainly a seeding failure
    #: rather than a classification result. Do not collapse this into UNKNOWN.
    NO_VOCABULARY = "no_vocabulary"


@dataclass(frozen=True)
class Classification:
    """One classification, with enough provenance to argue with it later.

    Frozen: it records a decision that has already been made, and the fields
    travel together into `clause_situation_edges.classification`. A mutable
    version invites a caller to "fix up" the confidence or the slug between the
    decision and the write, which is how a stored provenance record stops
    describing what actually happened.
    """

    outcome: Outcome
    #: Set only when MATCHED. `UNKNOWN` carrying a slug would be an invitation
    #: to serve it anyway.
    situation_id: UUID | None
    slug: str | None
    #: Top-1 cosine when `method == "embedding"`; the model's self-reported
    #: number when `method == "llm"`. TWO DIFFERENT SCALES, which is why
    #: `method` is not optional -- comparing a 0.7 from one against a 0.7 from
    #: the other is meaningless. 0.0 whenever the outcome is not MATCHED.
    confidence: float
    #: 'embedding' | 'llm' | 'none'. Names the stage whose judgement this is.
    method: str
    #: Embedder or LLM model id ACTUALLY USED -- so a later reader can tell
    #: whether a stored edge predates a model swap. None when nothing ran.
    model: str | None
    #: Set only when `method == "llm"`.
    prompt_version: str | None
    #: The highest-cosine situation OTHER THAN the one returned, for debugging a
    #: near call. On a MATCHED-by-embedding it is the second-place slug; on a
    #: MATCHED-by-LLM it is the candidate the model passed over (which may have
    #: been ahead on cosine); on an UNKNOWN there is no winner, so it is the
    #: closest thing that did not make it -- which is exactly the fact you want
    #: when asking why nothing matched.
    runner_up: str | None


class _Embedder(Protocol):
    """What `classify` needs from an embedder. `GeminiEmbedder` satisfies it."""

    model_id: str

    async def embed_many(self, texts: list[str]) -> Any: ...


@dataclass(frozen=True, slots=True)
class _Situation:
    id: UUID
    slug: str
    label: str
    description: str
    #: Carried on the row rather than fetched by a second `max(updated_at)`
    #: query. Two reads would not just cost a round trip on a sub-second path,
    #: they would RACE: a tenant editing a description between them makes the
    #: process cache the OLD texts' vectors under the NEW version's key, which
    #: then looks fresh forever -- exactly the stale-embedding failure the cache
    #: exists to prevent, reintroduced by the freshness check itself.
    updated_at: datetime


def situation_embedding_text(label: str, description: str) -> str:
    """The exact text embedded for one situation.

    ONE HELPER, USED EVERYWHERE. The label carries the plain name and the
    description carries the discriminating detail (the seed descriptions are
    written for this classifier rather than as UI tooltips -- see
    `SeedSituation.description`), and both are needed. Two call sites formatting
    this differently -- one joining with a newline, one with ": " -- would embed
    into slightly different places and quietly cost accuracy, with no failing
    test anywhere to say so. Anything that ever needs this string calls here.
    """
    return f"{label}\n{description}"


async def classify(
    customer_id: str,
    text: str,
    *,
    embedder: _Embedder | None = None,
    completion: Any | None = None,
) -> Classification:
    """Map `text` onto one of `customer_id`'s situations.

    Never raises. See the module docstring for the decision rule, the three
    outcomes, and why every failure degrades to `UNKNOWN`.

    `embedder` and `completion` are injection seams for tests, not
    configuration: the defaults (`get_embedder_v2()` and
    `engine.shared.llm.acompletion`) are the only thing production uses. They
    exist because the decision logic has to be testable with exact cosines --
    with no provider key the real embedder falls back to a SHA-256 hash vector
    with no semantic structure at all, so a test against it would assert on
    noise.
    """
    try:
        situations = await _load_situations(customer_id)
    except _VocabularyUnavailable:
        # An unreadable database is NOT an empty vocabulary. Reporting it as
        # NO_VOCABULARY would put a healthy tenant on the "somebody forgot to
        # seed this" list, which is the one report that has to stay
        # trustworthy.
        return _unknown(method=METHOD_NONE, model=None, runner_up=None)

    if not situations:
        # Checked FIRST, and before anything is spent. This is the diagnostic
        # the whole enum exists for; a blank input from an unseeded tenant must
        # still report the seeding problem, which is the more actionable of the
        # two facts.
        return _no_vocabulary()

    if not text or not text.strip():
        # Not NO_VOCABULARY: the vocabulary is fine, the input is empty. Nothing
        # to embed, so nothing is spent either.
        return _unknown(method=METHOD_NONE, model=None, runner_up=None)

    embedder = embedder or get_embedder_v2()

    vectors = await _vocabulary_vectors(customer_id, situations, embedder)
    if vectors is None:
        return _unknown(method=METHOD_NONE, model=None, runner_up=None)

    query_vector = await _embed_query(embedder, text)
    if query_vector is None:
        return _unknown(method=METHOD_NONE, model=None, runner_up=None)

    # Deterministic order on a cosine tie: without the slug tiebreak, two
    # situations with identical scores would swap places between processes and
    # the same input would classify differently on different pods.
    scored = sorted(
        zip(situations, [_cosine(query_vector, v) for v in vectors], strict=True),
        key=lambda pair: (-pair[1], pair[0].slug),
    )

    top, top_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else None

    if top_score < FLOOR:
        # Rule 4. No LLM call: nothing is close, and this is the common case.
        return _unknown(
            method=METHOD_EMBEDDING,
            model=embedder.model_id,
            runner_up=_runner_up(scored, None),
        )

    if second_score is None or (top_score - second_score) >= MARGIN:
        # A single-situation vocabulary has no top-2 and so cannot be ambiguous;
        # the floor alone decides it.
        return Classification(
            outcome=Outcome.MATCHED,
            situation_id=top.id,
            slug=top.slug,
            confidence=top_score,
            method=METHOD_EMBEDDING,
            model=embedder.model_id,
            prompt_version=None,
            runner_up=scored[1][0].slug if len(scored) > 1 else None,
        )

    # Rule 5. Everything within MARGIN of the top is tied -- including anything
    # that happens to sit below FLOOR, because the point is that the embedder
    # CANNOT separate these, and dropping one on a threshold it is within noise
    # of would hand the model a rigged single-option prompt.
    tied = [(s, score) for s, score in scored if (top_score - score) < MARGIN]
    return await _break_tie(
        text=text,
        tied=tied,
        scored=scored,
        completion=completion,
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class _VocabularyUnavailable(Exception):
    """The vocabulary could not be READ. NOT the same as an empty vocabulary."""


async def _load_situations(customer_id: str) -> list[_Situation]:
    """The tenant's vocabulary in slug order. An empty list is a real answer.

    Explicit `customer_id = $1` predicate as well as `with_tenant`, per the
    house rule: the dev/CI role is a SUPERUSER and bypasses RLS, so a query
    relying on the GUC alone would return every tenant's rows there and pass
    every test while being catastrophically wrong. Slug order is what pins the
    cached vectors to their situations, so it is not cosmetic.

    A database failure is NOT reported as an empty vocabulary. That would send
    a healthy tenant into `NO_VOCABULARY` and straight onto the
    "somebody forgot to seed this tenant" audit list, which is the one report
    that has to stay trustworthy. It degrades to `UNKNOWN` instead.
    """
    if not customer_id:
        return []
    try:
        async with with_tenant(customer_id) as conn:
            rows = await conn.fetch(
                """
                SELECT id, slug, label, description, updated_at
                  FROM situations
                 WHERE customer_id = $1
                 ORDER BY slug
                """,
                customer_id,
            )
    except Exception as exc:
        log.warning(
            "wfmem_classifier.situation_load_failed",
            customer=customer_id,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        raise _VocabularyUnavailable from exc
    return [
        _Situation(
            id=r["id"],
            slug=r["slug"],
            label=r["label"],
            description=r["description"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# The embedding cache
# --------------------------------------------------------------------------

#: LRU by insertion/access order. See `MAX_CACHED_VOCABULARIES`.
_VOCABULARY_CACHE: OrderedDict[tuple[str, str, int, str], list[list[float]]] = OrderedDict()


def _vocabulary_cache_key(
    customer_id: str, model_id: str, count: int, stamp: datetime
) -> tuple[str, str, int, str]:
    """Every axis along which the cached vectors can go stale.

    `customer_id` -- vocabularies are per-tenant and editable, and two tenants
    seeded in the same transaction have the SAME size and the SAME
    `max(updated_at)`. Without the tenant in the key they would share vectors,
    which is harmless right up until one of them edits a description.

    `model_id` -- vectors from two embedders are not comparable, and a cosine
    between them is a plausible-looking number rather than an error.

    `stamp` = `max(updated_at)` -- the edit detector. It only works because
    `wfmem_touch_updated_at` fires BEFORE UPDATE on `situations` (migration
    0110); if that trigger is ever dropped, `updated_at` freezes at insert time
    and every tenant edit is invisible to this cache forever.

    `count` -- the hole `max(updated_at)` alone leaves. A DELETE moves no
    surviving row's `updated_at`, so deleting a situation would not change the
    key and the cache would keep offering the deleted situation as a candidate.
    Deleting one and inserting another in the same transaction keeps the count
    but moves the stamp, so the pair covers both.

    Not covered, and accepted: a hand-written INSERT that supplies an
    `updated_at` older than the current max (the trigger is BEFORE UPDATE, so
    it does not fire on INSERT). No application path does that. The airtight
    alternative is to key on a digest of the embedded texts -- which we already
    have in hand, since the rows are loaded on every call -- and that is the
    change to make if this key ever proves too weak.
    """
    return (customer_id, model_id, count, stamp.isoformat())


def _vocabulary_cache_put(key: tuple[str, str, int, str], vectors: list[list[float]]) -> None:
    """Insert, evicting least-recently-used past the bound."""
    _VOCABULARY_CACHE[key] = vectors
    _VOCABULARY_CACHE.move_to_end(key)
    while len(_VOCABULARY_CACHE) > MAX_CACHED_VOCABULARIES:
        _VOCABULARY_CACHE.popitem(last=False)


def reset_vocabulary_cache() -> None:
    """Drop everything cached. For tests, mirroring `embeddings.reset_embedder`."""
    _VOCABULARY_CACHE.clear()


async def _vocabulary_vectors(
    customer_id: str, situations: list[_Situation], embedder: _Embedder
) -> list[list[float]] | None:
    """One vector per situation, in the same order. None if anything failed.

    Deliberately all-or-nothing. Classifying against the eleven that embedded
    successfully is worse than it sounds: the twelfth becomes silently
    unreachable for as long as the failure persists, and nothing downstream can
    tell that from "there is no rule for this situation".
    """
    # Both halves of the version come off the rows already in hand -- see
    # `_Situation.updated_at` for why this must not be a second query.
    key = _vocabulary_cache_key(
        customer_id,
        embedder.model_id,
        len(situations),
        max(s.updated_at for s in situations),
    )
    cached = _VOCABULARY_CACHE.get(key)
    if cached is not None:
        _VOCABULARY_CACHE.move_to_end(key)
        return cached

    texts = [situation_embedding_text(s.label, s.description) for s in situations]
    try:
        result = await embedder.embed_many(texts)
    except Exception as exc:
        log.warning(
            "wfmem_classifier.vocabulary_embedding_failed",
            customer=customer_id,
            model=embedder.model_id,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        return None

    vectors = _vectors_by_index(result, len(texts))
    if vectors is None:
        log.warning(
            "wfmem_classifier.vocabulary_embedding_incomplete",
            customer=customer_id,
            model=embedder.model_id,
            expected=len(texts),
            failed=len(getattr(result, "failed", []) or []),
        )
        # NOT cached. A transient provider hiccup must not pin this tenant to
        # UNKNOWN for the life of the process; the cache holds answers, not
        # failures.
        return None

    _vocabulary_cache_put(key, vectors)
    return vectors


async def _embed_query(embedder: _Embedder, text: str) -> list[float] | None:
    """Embed the input text. None on failure.

    Through `embed_many`, the same entry point the vocabulary goes through --
    NOT `embed_query`, which on `GeminiEmbedder` applies the asymmetric
    retrieval prefix. See the module docstring: both sides have to live in the
    same space for a single FLOOR to mean anything.
    """
    try:
        result = await embedder.embed_many([text])
    except Exception as exc:
        log.warning(
            "wfmem_classifier.query_embedding_failed",
            model=embedder.model_id,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        return None
    vectors = _vectors_by_index(result, 1)
    if vectors is None:
        log.warning(
            "wfmem_classifier.query_embedding_failed",
            model=embedder.model_id,
            error="the embedder rejected the input",
        )
        return None
    return vectors[0]


def _vectors_by_index(result: Any, expected: int) -> list[list[float]] | None:
    """Reorder an `EmbedResult` into input order, or None if any input is missing.

    KEYED ON `chunk_index`, NEVER ON LIST POSITION. `EmbedResult` carries
    `embedded` and `failed` as two separate lists and `_BaseEmbedder`'s
    recursive half-split appends sub-batches as they resolve, so the i-th entry
    of `embedded` is NOT the i-th input. Zipping them positionally attributes
    one situation's vector to another -- every threshold still looks right, the
    numbers are all still there, they are just on the wrong rows, and the only
    symptom is that the classifier is mysteriously bad.
    """
    by_index: dict[int, list[float]] = {}
    for chunk in getattr(result, "embedded", None) or []:
        by_index[chunk.chunk_index] = chunk.embedding
    if len(by_index) != expected or any(i not in by_index for i in range(expected)):
        return None
    return [by_index[i] for i in range(expected)]


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine. 0.0 rather than an exception on a zero or mismatched vector.

    A length mismatch cannot happen while `model_id` is in the cache key (that
    is what pins the dimension), so 0.0 here is an unreachable-branch guard, not
    a silent answer to a real question. It is 0.0 rather than a raise because
    everything on this path degrades to UNKNOWN and 0.0 lands there.

    Pure Python, no numpy: twelve dot products over a ~3k-dim vector is a couple
    of milliseconds against a sub-second budget, and the module is not worth a
    dependency it would otherwise not have.
    """
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


# --------------------------------------------------------------------------
# The tie-break
# --------------------------------------------------------------------------

_TIEBREAK_SYSTEM_PROMPT = """\
You label what a person is doing right now with one of a short list of work \
situations.

The candidates below were shortlisted because a semantic similarity model could \
not separate them, so they are genuinely close. Read the descriptions rather \
than the labels: the descriptions are what distinguishes them.

Reply with a single JSON object and nothing else:

  {"slug": "<one of the candidate slugs>", "confidence": <number between 0 and 1>}

If none of the candidates describes what the person is doing, reply:

  {"slug": null, "confidence": 0.0}

Answering null is expected and correct whenever the text is about something \
else. Do not pick the least-bad candidate."""


def _build_tiebreak_messages(text: str, tied: list[_Situation]) -> list[dict[str, Any]]:
    """The prompt. ONLY the tied candidates appear in it.

    Handing the model all twelve would make the shortlist pointless, blow the
    latency budget it was shortlisted to protect, and reintroduce exactly the
    confusion between distant situations that the cosine already resolved.

    `text` is not truncated here, and does not need to be: it only reaches this
    function after the embedder accepted it, and the embedder's per-request
    ceiling is far below anything this prompt would struggle with.
    """
    candidates = "\n\n".join(f"- {s.slug}: {s.label}\n  {s.description}" for s in tied)
    return [
        {"role": "system", "content": _TIEBREAK_SYSTEM_PROMPT},
        {"role": "user", "content": f"CANDIDATES\n\n{candidates}\n\nTEXT\n\n{text}"},
    ]


async def _break_tie(
    *,
    text: str,
    tied: list[tuple[_Situation, float]],
    scored: list[tuple[_Situation, float]],
    completion: Any | None,
) -> Classification:
    """Ask the model to separate the tied candidates. May still answer `UNKNOWN`."""
    candidates = [s for s, _ in tied]
    by_slug = {s.slug: s for s in candidates}
    completion = completion or _default_completion()

    try:
        response = await completion(
            model=TIEBREAK_MODEL,
            messages=_build_tiebreak_messages(text, candidates),
            max_tokens=_TIEBREAK_MAX_TOKENS,
        )
    except Exception as exc:
        # `engine.shared.llm.LLMError` is the expected case and is caught by
        # this clause. It is not named: importing it drags `litellm` into every
        # importer of this module (it is a module-scope import over there), and
        # in a serving path the correct handling for a transport error and for
        # a library-internal one is identical anyway -- degrade, log, do not
        # raise into somebody's search request.
        # `status_code` and `provider` ride along for the same reason they were
        # added to the structuring pass: a 404 from the gateway (no such model
        # deployment) and a 500 from the provider degrade identically here and
        # are completely different jobs to fix. Cheap, structural, and the thing
        # that turns "the classifier is flaky" into an actionable line.
        log.warning(
            "wfmem_classifier.tiebreak_failed",
            model=TIEBREAK_MODEL,
            error=str(exc),
            error_class=type(exc).__name__,
            status_code=getattr(exc, "status_code", None),
            provider=getattr(exc, "provider", None),
        )
        return _unknown(method=METHOD_NONE, model=None, runner_up=_runner_up(scored, None))

    parsed = _parse_tiebreak_response(response_text(response), allowed=set(by_slug))
    if parsed is None:
        # Unusable: unparseable, wrong shape, or a slug that was not on the
        # menu. An off-menu slug is refused rather than honoured -- the
        # shortlist IS the permitted answer set, and accepting anything else
        # would let the model overrule a floor it was never shown.
        log.warning(
            "wfmem_classifier.tiebreak_unusable_response",
            model=TIEBREAK_MODEL,
            candidates=sorted(by_slug),
        )
        return _unknown(method=METHOD_NONE, model=None, runner_up=_runner_up(scored, None))

    slug, confidence = parsed
    if slug is None:
        return _unknown(method=METHOD_LLM, model=TIEBREAK_MODEL, runner_up=_runner_up(scored, None))

    chosen = by_slug[slug]
    return Classification(
        outcome=Outcome.MATCHED,
        situation_id=chosen.id,
        slug=chosen.slug,
        confidence=confidence,
        method=METHOD_LLM,
        model=TIEBREAK_MODEL,
        prompt_version=CLASSIFIER_PROMPT_VERSION,
        runner_up=_runner_up(scored, chosen.slug),
    )


def _default_completion() -> Any:
    """Imported lazily: `engine.shared.llm` imports `litellm` at module scope.

    Most classifications never reach the tie-break, so paying that import at
    module load would tax every importer of this file for a path they will
    usually not take.
    """
    from engine.shared.llm import acompletion

    return acompletion


def _parse_tiebreak_response(
    raw: str | None, *, allowed: set[str]
) -> tuple[str | None, float] | None:
    """`(slug | None, confidence)`, or None when the response is unusable.

    THE TWO EMPTY ANSWERS ARE OPPOSITES and the caller treats them differently:
    `(None, 0.0)` is the model saying "none of these", which is a real
    classification (`method == "llm"`); `None` is the model failing to answer,
    which is not (`method == "none"`).

    Tolerant about packaging -- code fences and a chatty preamble are what
    models actually emit and neither is an error -- and strict about content: a
    slug that was not on the menu is refused rather than repaired.
    """
    if not raw or not raw.strip():
        return None

    payload = loads_forgiving(raw)
    if not isinstance(payload, dict) or "slug" not in payload:
        return None

    slug = payload["slug"]
    if slug is None or (isinstance(slug, str) and slug.strip().lower() in ("", "none", "null")):
        return None, 0.0
    if not isinstance(slug, str) or slug.strip() not in allowed:
        return None

    return slug.strip(), _coerce_confidence(payload.get("confidence"))


def _coerce_confidence(value: Any) -> float:
    """A number in [0, 1], or `LLM_UNREPORTED_CONFIDENCE`.

    A missing or out-of-range confidence is NOT treated as a failed answer: the
    model was asked which candidate fits, and it answered that. Inventing a
    precise-looking number for the part it skipped would be worse than admitting
    we do not have one, which is what the named constant does.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return LLM_UNREPORTED_CONFIDENCE
    if not 0.0 <= float(value) <= 1.0:
        return LLM_UNREPORTED_CONFIDENCE
    return float(value)


# --------------------------------------------------------------------------
# Result constructors
# --------------------------------------------------------------------------


def _runner_up(scored: list[tuple[_Situation, float]], winner_slug: str | None) -> str | None:
    """Highest-cosine situation that is NOT the returned match.

    ONE RULE FOR EVERY PATH, which is why the "no winner" case passes None
    rather than each caller picking its own answer. With a winner it is the
    thing that came second to it; with no winner it is the thing that came
    closest to being one -- which is the question somebody staring at an UNKNOWN
    is actually asking. See `Classification.runner_up`.
    """
    for situation, _ in scored:
        if situation.slug != winner_slug:
            return situation.slug
    return None


def _no_vocabulary() -> Classification:
    return Classification(
        outcome=Outcome.NO_VOCABULARY,
        situation_id=None,
        slug=None,
        confidence=0.0,
        method=METHOD_NONE,
        model=None,
        prompt_version=None,
        runner_up=None,
    )


def _unknown(*, method: str, model: str | None, runner_up: str | None) -> Classification:
    """`prompt_version` follows `method`, not `outcome`.

    A model that read the shortlist and answered "none of these" produced that
    UNKNOWN under a specific prompt, and a later reclassification pass has to be
    able to tell v1's refusals from v2's -- a prompt change is exactly the sort
    of thing that turns a refusal into a match. Blanking it on UNKNOWN would
    make every negative answer unattributable.
    """
    return Classification(
        outcome=Outcome.UNKNOWN,
        situation_id=None,
        slug=None,
        confidence=0.0,
        method=method,
        model=model,
        prompt_version=CLASSIFIER_PROMPT_VERSION if method == METHOD_LLM else None,
        runner_up=runner_up,
    )


__all__ = [
    "CLASSIFIER_PROMPT_VERSION",
    "FLOOR",
    "LLM_UNREPORTED_CONFIDENCE",
    "MARGIN",
    "MAX_CACHED_VOCABULARIES",
    "METHOD_EMBEDDING",
    "METHOD_LLM",
    "METHOD_NONE",
    "TIEBREAK_MODEL",
    "Classification",
    "Outcome",
    "classify",
    "reset_vocabulary_cache",
    "situation_embedding_text",
]
