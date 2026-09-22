"""Phase 0 replay: score a stored search's pool with Jev and compare arms.

WHY THIS IS A REPLAY AND NOT A SHADOW HOOK. Every search already persists its
whole pre-fan-out pool -- content included -- plus what was delivered, to R2
(`trace_blob.py`, sample rate 1.0). So the comparison runs offline on real
queries, with no request-path code and no key on a pod, and it can be re-run.

WHAT IS BEING COMPARED. Three arms, each cut to the SAME budget, because
production delivers a set of documents and that is the only thing a reader
experiences:

    A  today      the gatherer's own picks, topped up by the recall floor
    B  floor only the recall floor with NO model at all
    C  jev        chunks scoring >= theta, topped up by the recall floor

Arm B exists because the gatherer emitted a median of ZERO chunks across the
sampled traces: if B matches A, the expensive model call is not earning its
keep and the cheapest cutover is to delete it, with Jev a separate question.

UNITS ARE DOCUMENTS. The recall floor counts DISTINCT DOCS (`_RECALL_FLOOR_DOCS`)
and Jev scores CHUNKS, so comparing them chunk-wise would let one long document
occupy ten slots on one side and one on the other. A document's score is the max
over its chunks; chunk-level numbers are kept only for span/why_relevant
coverage.

TIME IS FROZEN PER TRACE. `_fuse_prefanout_docs` decays each hit by its age
against `datetime.now(UTC)`, and half-lives differ per source, so replaying a
three-week-old trace today reorders it -- not uniformly, which is worse than a
constant offset. `frozen_now` pins the clock to the trace's own timestamp for
the duration of the call. Blobs that predate the v3 capture do not record the
request's `recency_half_life_days` or its rendered doc ids; those rows are
flagged `approx=True` rather than quietly presented as exact.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from engine.retrieval.agent import loop as _loop
from engine.retrieval.agent.loop import (
    _RECALL_FLOOR_DOCS,
    _backfill_recall_floor,
    _fuse_prefanout_docs,
)
from engine.retrieval.agent.models import GathererNotes, GathererOutput

#: Tenants whose traffic is not a real question from a real person.
#: `new-workspace` is the big one -- an EMPTY tenant where every channel
#: returns zero hits, and the single largest source of rows in the logs.
EXCLUDED_TENANTS = frozenset(
    {"new-workspace", "test-prod", "testing", "probe-demo",
     "oneleet-test-2", "oneleet-pentest-team"}
)

#: Statuses where no selection happened, so there is nothing to compare.
#: Kept OUT of the selection arms and IN the extraction report -- a wrong
#: `doc_types` filter is one way to produce zero recall, and dropping those
#: rows would hide exactly the failure extraction is suspected of.
EXCLUDED_STATUSES = frozenset({"zero_recall_short_circuit", "id_lookup_short_circuit"})

#: Measured, not published: see docs/jev-contract.md. The cap is ~32k on the
#: WHOLE request and each Noul costs ~35 tokens on top of the state, which a
#: chars/4 estimate never sees. 24k is 0.75 of the cap.
JEV_TOKEN_BUDGET = 24_000
JEV_CHARS_PER_TOKEN = 3.77
JEV_TOKENS_PER_QUESTION = 35

#: What a reader gets. Same number for every arm, or the comparison is a
#: comparison of budgets.
DELIVERY_BUDGET_DOCS = _RECALL_FLOOR_DOCS  # 10


# --------------------------------------------------------------------------
# blobs


def load_blob(path: str | Path) -> dict[str, Any]:
    """Read one gzipped trace blob. Mirrors `trace_analyzer/fetch_one.py`."""
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def is_replayable(blob: dict[str, Any]) -> tuple[bool, str]:
    """Should this trace take part in the SELECTION comparison?

    Returns `(ok, reason)`; the reason is recorded for every skip so the
    report can say what it did not look at. A skip that is only visible as a
    smaller N is how a filtered-down corpus starts flattering itself.
    """
    if (blob.get("customer_id") or "") in EXCLUDED_TENANTS:
        return False, "excluded_tenant"
    if (blob.get("status") or "") in EXCLUDED_STATUSES:
        return False, "excluded_status"
    # Test the POOL, not `sub_queries`. A search whose four channels all came
    # back empty still records a sub_query envelope, so the obvious check
    # ("did the pre-fan-out run?") passes on exactly the traces that have
    # nothing to compare -- and the report would carry them as rows where
    # every arm agreed perfectly on the empty set.
    if not pool_chunks(blob.get("prefanout")):
        return False, "empty_pool"
    return True, "ok"


@contextmanager
def frozen_now(blob: dict[str, Any]) -> Iterator[datetime]:
    """Pin `loop`'s clock to the trace's own timestamp.

    The recency decay in `_source_weight` is per-source, so a replay run today
    does not shift every hit by the same factor -- it reorders them. Freezing
    is what makes "recompute the floor and compare it to the floor the blob
    recorded" a fair self-test rather than a test of how old the blob is.
    """
    stamp = blob.get("timestamp_utc")
    try:
        ref = datetime.fromisoformat(stamp) if stamp else datetime.now(UTC)
    except (TypeError, ValueError):
        ref = datetime.now(UTC)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=UTC)

    real = _loop.datetime

    class _Frozen(real):  # type: ignore[misc,valid-type]
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001, ANN206
            return ref if tz is None else ref.astimezone(tz)

    with patch.object(_loop, "datetime", _Frozen):
        yield ref


# --------------------------------------------------------------------------
# the pool


def pool_chunks(prefanout: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Every distinct chunk in the pool, keyed by `chunk_id`.

    First occurrence wins: the same chunk arrives from several channels and
    sub-queries, and the first one carries the same content as the rest. Hits
    with no chunk_id or no content body are skipped -- they cannot be scored
    (nothing to read) and they cannot be delivered.
    """
    out: dict[str, dict[str, Any]] = {}
    for sq in (prefanout or {}).get("sub_queries") or []:
        if not isinstance(sq, dict):
            continue
        for channel in ("vector", "bm25", "graph", "inferred_edge"):
            for hit in sq.get(channel) or []:
                if not isinstance(hit, dict):
                    continue
                cid = hit.get("chunk_id")
                content = hit.get("content")
                if not cid or not (isinstance(content, str) and content.strip()):
                    continue
                out.setdefault(cid, hit)
    return out


def doc_of(chunk_id: str, hit: dict[str, Any]) -> str:
    """The document a chunk belongs to; `doc_id` on the hit is authoritative."""
    return hit.get("doc_id") or chunk_id


def has_provenance(blob: dict[str, Any]) -> bool:
    """Does this trace record WHO chose each delivered chunk?

    `GatheredChunk.harness_appended` did not exist before 2026-09-12. Its
    absence is not "the model picked this" -- it is "nobody wrote it down", and
    conflating the two is a live trap: defaulting missing-to-model turns every
    older trace into one where the gatherer supplied 100% of the answer, and
    produces a clean, false cliff on the day the field shipped.

    The A-vs-B arms do NOT need this (arm A is the delivered set, arm B is a
    reconstruction), so a trace without provenance still takes part in the
    comparison. Only the model-versus-floor attribution is withheld.
    """
    for ch in (blob.get("gathered") or {}).get("chunks") or []:
        if isinstance(ch, dict) and "harness_appended" in ch:
            return True
    return False


def delivered_docs(
    blob: dict[str, Any],
) -> tuple[list[str], set[str], set[str], bool]:
    """What the user got: `(ordered, model_picked, appended, provenance_known)`.

    When `provenance_known` is False the two sets are empty -- not because
    nothing was delivered, but because the blob never recorded which side chose
    it. Callers must branch on the flag rather than reading an empty set as a
    finding. Order is delivery order.
    """
    known = has_provenance(blob)
    ordered: list[str] = []
    model: set[str] = set()
    appended: set[str] = set()
    for ch in (blob.get("gathered") or {}).get("chunks") or []:
        if not isinstance(ch, dict):
            continue
        doc = ch.get("doc_id")
        if not doc:
            continue
        if doc not in ordered:
            ordered.append(doc)
        if known:
            (appended if ch.get("harness_appended") else model).add(doc)
    return ordered, model, appended, known


# --------------------------------------------------------------------------
# the three arms


def arm_today(blob: dict[str, Any], budget: int = DELIVERY_BUDGET_DOCS) -> list[str]:
    """Arm A -- exactly what this search delivered, cut to the budget."""
    ordered, _, _, _ = delivered_docs(blob)
    return ordered[:budget]


def arm_floor_only(
    blob: dict[str, Any], budget: int = DELIVERY_BUDGET_DOCS
) -> list[str]:
    """Arm B -- the recall floor with NO model output to top up.

    Reconstructed by running the REAL `_backfill_recall_floor` against an empty
    `GathererOutput`, not by re-implementing "top 10 by RRF". The floor is not
    simply that: it fills the slots a model left, counts distinct documents,
    weights by source and recency, and can decline to fire. Re-implementing it
    would compare Jev against a paraphrase of the baseline instead of the
    baseline. 25 of 36 sampled traces already ran in exactly this state.
    """
    empty = GathererOutput(chunks=[], gatherer_notes=GathererNotes())
    with frozen_now(blob):
        _backfill_recall_floor(
            empty,
            blob.get("prefanout"),
            half_life_days=blob.get("request_recency_half_life_days"),
            mode="always",
        )
    out: list[str] = []
    for ch in empty.chunks:
        if ch.doc_id and ch.doc_id not in out:
            out.append(ch.doc_id)
    return out[:budget]


def arm_jev(
    blob: dict[str, Any],
    chunk_scores: dict[str, float],
    *,
    theta: float,
    budget: int = DELIVERY_BUDGET_DOCS,
) -> tuple[list[str], int]:
    """Arm C -- documents scoring >= theta, topped up by the floor.

    Returns `(doc_ids, n_from_jev)`. The top-up is deliberate: production would
    keep the floor, so judging bare `J_theta` would judge a policy we are not
    proposing to ship. A document scores as the MAX over its chunks (see the
    module docstring on units).
    """
    pool = pool_chunks(blob.get("prefanout"))
    best: dict[str, float] = {}
    for cid, hit in pool.items():
        s = chunk_scores.get(cid)
        if s is None:
            continue
        doc = doc_of(cid, hit)
        if s > best.get(doc, -1.0):
            best[doc] = s
    picked = [d for d, s in sorted(best.items(), key=lambda kv: -kv[1]) if s >= theta]
    out = picked[:budget]
    n_jev = len(out)
    if len(out) < budget:
        for doc in arm_floor_only(blob, budget=budget * 3):
            if doc not in out:
                out.append(doc)
            if len(out) >= budget:
                break
    return out[:budget], n_jev


def doc_scores(
    blob: dict[str, Any], chunk_scores: dict[str, float]
) -> dict[str, float]:
    """Document-level scores (max over chunks), for the distribution report."""
    pool = pool_chunks(blob.get("prefanout"))
    best: dict[str, float] = {}
    for cid, hit in pool.items():
        s = chunk_scores.get(cid)
        if s is None:
            continue
        doc = doc_of(cid, hit)
        if s > best.get(doc, -1.0):
            best[doc] = s
    return best


# --------------------------------------------------------------------------
# batching for Jev


def estimate_tokens(state_chars: int, n_questions: int) -> int:
    """Measured size model -- see docs/jev-contract.md.

    The question term is the half a chars/4 estimate misses: at 100 chunks it
    is ~3,500 tokens, more than a tenth of the ~32k request cap.
    """
    return int(state_chars / JEV_CHARS_PER_TOKEN) + JEV_TOKENS_PER_QUESTION * n_questions


@dataclass(slots=True)
class JevBatch:
    """One request's worth of pool: the chunks in it and its estimated size."""

    chunk_ids: list[str] = field(default_factory=list)
    chars: int = 0

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.chars, len(self.chunk_ids))


def batch_pool(
    pool: dict[str, dict[str, Any]],
    *,
    query: str,
    token_budget: int = JEV_TOKEN_BUDGET,
) -> list[JevBatch]:
    """Split a pool into requests that fit under the cap.

    Split by MEASURED size, never by chunk count: chunk lengths span two orders
    of magnitude (a one-line Slack message against a whole config file), so a
    fixed count is over budget on one search and wasteful on the next. Nouls are
    independent, so a batch boundary changes no answer -- unlike a Choice, whose
    probabilities must sum to one across the whole option set.

    A single chunk larger than the budget still gets its own batch rather than
    being dropped: the caller truncates it and says so. Silently discarding the
    biggest document in a pool would be a recall bug wearing a size limit's
    clothes.
    """
    overhead = len(query) + 64  # the state envelope around the chunks
    batches: list[JevBatch] = [JevBatch(chars=overhead)]
    for cid, hit in pool.items():
        cost = len(hit.get("content") or "") + len(cid) + 8
        cur = batches[-1]
        if cur.chunk_ids and estimate_tokens(cur.chars + cost, len(cur.chunk_ids) + 1) > token_budget:
            batches.append(JevBatch(chars=overhead))
            cur = batches[-1]
        cur.chunk_ids.append(cid)
        cur.chars += cost
    return [b for b in batches if b.chunk_ids]


# --------------------------------------------------------------------------
# comparison


@dataclass(slots=True)
class ArmComparison:
    """The counts one trace contributes to the report."""

    trace_id: str
    day: str
    customer_id: str
    status: str
    pool_chunks: int
    pool_docs: int
    pool_tokens: int
    batches: int
    a_docs: list[str]
    b_docs: list[str]
    n_model_picked: int
    n_appended: int
    provenance_known: bool
    a_only: int
    b_only: int
    shared: int
    approx: bool

    def to_row(self) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in self.__slots__  # type: ignore[attr-defined]
        }


def compare_a_b(blob: dict[str, Any]) -> ArmComparison:
    """Arm A against arm B for one trace -- needs no Jev key and no deploy.

    This is the cheapest question in the plan: if A and B deliver the same
    documents, the gatherer's LLM call changed nothing that reached the user.
    """
    pool = pool_chunks(blob.get("prefanout"))
    docs = {doc_of(c, h) for c, h in pool.items()}
    chars = sum(len(h.get("content") or "") for h in pool.values())
    a = arm_today(blob)
    b = arm_floor_only(blob)
    _, model, appended, known = delivered_docs(blob)
    sa, sb = set(a), set(b)
    return ArmComparison(
        trace_id=blob.get("trace_id") or "",
        day=(blob.get("timestamp_utc") or "")[:10],
        customer_id=blob.get("customer_id") or "",
        status=blob.get("status") or "",
        pool_chunks=len(pool),
        pool_docs=len(docs),
        pool_tokens=estimate_tokens(chars, len(pool)),
        batches=len(batch_pool(pool, query=blob.get("query") or "")),
        a_docs=a,
        b_docs=b,
        n_model_picked=len(model),
        n_appended=len(appended),
        provenance_known=known,
        a_only=len(sa - sb),
        b_only=len(sb - sa),
        shared=len(sa & sb),
        approx="rendered_doc_ids" not in blob,
    )


def fused_order(blob: dict[str, Any]) -> list[str]:
    """The floor's own ranking of the pool, clock frozen. For reporting."""
    with frozen_now(blob):
        return [
            e["doc_id"]
            for e in _fuse_prefanout_docs(
                blob.get("prefanout"),
                half_life_days=blob.get("request_recency_half_life_days"),
            )
        ]
