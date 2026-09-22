"""Jev (TypeSafe) as the gatherer's selector.

WHAT THIS REPLACES. The gatherer's result-picking LLM turn: gpt-oss-120b reads
the top ~35 pre-fan-out docs that fit a 19,100-token render budget and names
the chunks it wants by RE-TYPING them. Phase 0 (3,048 replayed searches, two
graders from different model families) found that turn no better than the
recall floor alone, and found ranking the pool by Jev's per-chunk probability
2-3x better than either -- at ~$0.0016 and ~0.3s a search instead of ~$0.024
and ~1.9s. Full write-up: docs/plans/jev-phase0-report.md.

WHAT JEV IS. A typed-decision model: it answers yes/no ("Noul") questions
about a state with a probability, and cannot emit text. So it can never
fumble a citation the way the LLM could -- the harness keeps the ids.

THE MEASURED CONTRACT (docs/jev-contract.md -- the published one is looser):

  * ~32k-token cap on the WHOLE request, and every Noul costs ~35 tokens on
    top of the state. A p90 pool does not fit one request.
  * No question cap at 300 per request (the documented 255 is options per
    Choice).
  * NOT deterministic: identical requests drift up to 0.06 per probability.
    That, plus scores bunching near zero (58% of docs <= 0.1), is why the
    selector RANKS rather than thresholds: at theta=0.7 nothing cleared in
    75% of searches, while rank order barely moves under the drift.

    request                              response
    -------                              --------
    POST /v1/systemone                   {"model": "jev-1.13.0",
    {"model": "jev-1.13.0",               "answers": {"<chunk_id>":
     "state": {"query": q,                  {"type": "noul", "noul": 0.84}},
               "chunks": {id: text}},     "usage": {"input_tokens": 29057}}
     "questions": {id: {"type": "noul",
                        "instructions": ...}}}

Everything here is pure or async-over-httpx; the caller owns timeouts and the
fallback to the recall floor.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from engine.shared.constants import (
    JEV_BASE_URL,
    JEV_CHARS_PER_TOKEN,
    JEV_MAX_CHUNK_CHARS,
    JEV_MODEL,
    JEV_REQUEST_TIMEOUT_SECONDS,
    JEV_TOKEN_BUDGET,
    JEV_TOKENS_PER_QUESTION,
)

#: Channels whose hits are content passages. Order matters for provenance
#: only: it is the order a chunk's `matched_via` lists them in.
_CONTENT_CHANNELS = ("vector", "bm25", "graph", "inferred_edge")

#: Deliberately blunt, and identical to what Phase 0 measured. Rewording it is
#: a model change: re-run the replay before shipping a new phrasing.
_NOUL_INSTRUCTIONS = (
    "The passage at key {cid!r} in state.chunks answers, or directly supports "
    "an answer to, state.query."
)


class JevError(RuntimeError):
    """Jev did not produce scores. The caller falls back to the recall floor."""


# --------------------------------------------------------------------------
# the pool


def pool_chunks(prefanout: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Every distinct scoreable chunk in the pre-fan-out, keyed by chunk_id.

    First occurrence wins (the same chunk arrives from several channels and
    sub-queries with the same body). A hit with no chunk_id or no content
    body is skipped: there is nothing to score and nothing to deliver.
    """
    out: dict[str, dict[str, Any]] = {}
    for sq in (prefanout or {}).get("sub_queries") or []:
        if not isinstance(sq, dict):
            continue
        for channel in _CONTENT_CHANNELS:
            for hit in sq.get(channel) or []:
                if not isinstance(hit, dict):
                    continue
                cid = hit.get("chunk_id")
                content = hit.get("content")
                if not cid or not (isinstance(content, str) and content.strip()):
                    continue
                out.setdefault(cid, hit)
    return out


def channels_by_chunk(prefanout: dict[str, Any] | None) -> dict[str, list[str]]:
    """Which channels surfaced each chunk -- its real provenance.

    A Jev-selected chunk reports the channels that RETRIEVED it, not a
    synthetic "jev" channel: Jev chose among what retrieval found, it did not
    find anything itself. (Contrast `recall_floor`, which is its own label
    precisely because no model looked at those chunks.)
    """
    out: dict[str, list[str]] = {}
    for sq in (prefanout or {}).get("sub_queries") or []:
        if not isinstance(sq, dict):
            continue
        for channel in _CONTENT_CHANNELS:
            for hit in sq.get(channel) or []:
                if isinstance(hit, dict) and hit.get("chunk_id"):
                    seen = out.setdefault(hit["chunk_id"], [])
                    if channel not in seen:
                        seen.append(channel)
    return out


# --------------------------------------------------------------------------
# sizing and batching


def estimate_tokens(state_chars: int, n_questions: int) -> int:
    """Measured, not published: see docs/jev-contract.md.

    The question term is the half a chars/4 estimate misses entirely -- at 100
    chunks it is ~3,500 tokens, more than a tenth of the request cap.
    """
    return int(state_chars / JEV_CHARS_PER_TOKEN) + JEV_TOKENS_PER_QUESTION * n_questions


def _body(hit: dict[str, Any]) -> str:
    """The text Jev reads for one chunk, capped so no single chunk can blow a
    request on its own. Truncated, never dropped: silently discarding the
    biggest document in a pool would be a recall bug wearing a size limit's
    clothes.
    """
    text = hit.get("content") or ""
    return text[:JEV_MAX_CHUNK_CHARS]


def batch_pool(
    pool: dict[str, dict[str, Any]],
    *,
    query: str,
    token_budget: int = JEV_TOKEN_BUDGET,
) -> list[list[str]]:
    """Split a pool into requests that fit under the cap, by MEASURED size.

    Never by chunk count: chunk lengths span two orders of magnitude, so a
    fixed count is over budget on one search and wasteful on the next. Nouls
    are independent, so a batch boundary changes no answer (a Choice would --
    its probabilities sum to one across the whole option set).
    """
    overhead = len(query) + 64
    batches: list[list[str]] = []
    cur: list[str] = []
    chars = overhead
    for cid, hit in pool.items():
        cost = len(_body(hit)) + len(cid) + 8
        if cur and estimate_tokens(chars + cost, len(cur) + 1) > token_budget:
            batches.append(cur)
            cur, chars = [], overhead
        cur.append(cid)
        chars += cost
    if cur:
        batches.append(cur)
    return batches


# --------------------------------------------------------------------------
# the call


@dataclass(slots=True)
class ScoreResult:
    """What one pool's scoring produced, and what it cost."""

    scores: dict[str, float] = field(default_factory=dict)
    requests: int = 0
    input_tokens: int = 0
    splits: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


#: One pooled client per running event loop. A client per search paid a fresh
#: TLS handshake to api.typesafe.ai on every request -- on the order of the
#: ~0.2s the scoring call itself takes. Keyed by loop because an AsyncClient is
#: bound to the loop that created it, and test runners start a new loop per
#: test.
_CLIENTS: dict[int, httpx.AsyncClient] = {}


def _shared_client() -> httpx.AsyncClient:
    loop_id = id(asyncio.get_running_loop())
    client = _CLIENTS.get(loop_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
        _CLIENTS[loop_id] = client
    return client


async def _post(
    client: httpx.AsyncClient,
    api_key: str,
    state: dict[str, Any],
    questions: dict[str, Any],
) -> httpx.Response:
    return await client.post(
        f"{JEV_BASE_URL}/v1/systemone",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": JEV_MODEL, "state": state, "questions": questions},
        timeout=JEV_REQUEST_TIMEOUT_SECONDS,
    )


async def _score_batch(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    pool: dict[str, dict[str, Any]],
    cids: list[str],
    out: ScoreResult,
    depth: int = 0,
) -> None:
    """Score one batch; on `max_tokens_exceeded`, halve it and try again.

    The size estimate is calibrated on prose (3.77 chars/token). Config files
    and stack traces tokenize denser and overran a budget that looked safe on
    67 of 3,048 replayed searches. Reacting to the server's own verdict is the
    only way to be right about a tokenizer we do not have; shrinking the
    budget further would tax every normal request to cover the rare dense one.
    """
    if not cids:
        return
    state = {"query": query, "chunks": {cid: _body(pool[cid]) for cid in cids}}
    questions = {
        cid: {"type": "noul", "instructions": _NOUL_INSTRUCTIONS.format(cid=cid)}
        for cid in cids
    }
    try:
        resp = await _post(client, api_key, state, questions)
    except httpx.HTTPError as exc:
        # Class name AND message: httpx timeouts stringify EMPTY, and a blank
        # reason in the log is how an outage reads as nothing.
        out.errors.append(f"{type(exc).__name__}: {str(exc)[:160] or '<empty>'}")
        return
    overflow = resp.status_code == 400 and "max_tokens_exceeded" in resp.text
    if overflow and len(cids) > 1 and depth < 4:
        out.splits += 1
        mid = len(cids) // 2
        await asyncio.gather(
            _score_batch(client, api_key, query, pool, cids[:mid], out, depth + 1),
            _score_batch(client, api_key, query, pool, cids[mid:], out, depth + 1),
        )
        return
    if resp.status_code != 200:
        out.errors.append(f"http_{resp.status_code}: {resp.text[:160]}")
        return
    try:
        body = resp.json()
    except ValueError:
        out.errors.append("malformed_json")
        return
    out.requests += 1
    out.input_tokens += int(((body.get("usage") or {}).get("input_tokens")) or 0)
    for cid, ans in (body.get("answers") or {}).items():
        val = ans.get("noul") if isinstance(ans, dict) else None
        # An answer missing for a chunk is ABSENT, not zero: reading "the
        # server did not answer" as "irrelevant" silently deletes a document.
        if isinstance(val, (int, float)) and cid in pool:
            out.scores[cid] = float(val)


async def score_pool(
    query: str,
    pool: dict[str, dict[str, Any]],
    *,
    api_key: str,
    client: httpx.AsyncClient | None = None,
) -> ScoreResult:
    """Score every chunk in `pool`; batches run concurrently.

    Raises `JevError` only when NOTHING came back. A partial result (one batch
    failed) is returned as-is: the unscored chunks rank below every scored one,
    and the recall floor tops the response up -- which is the same answer the
    request would have got without Jev at all.
    """
    if not api_key:
        raise JevError("TYPESAFE_API_KEY is not configured")
    out = ScoreResult()
    if not pool:
        return out
    t0 = time.perf_counter()
    client = client or _shared_client()
    await asyncio.gather(
        *(
            _score_batch(client, api_key, query, pool, cids, out)
            for cids in batch_pool(pool, query=query)
        )
    )
    out.elapsed_ms = (time.perf_counter() - t0) * 1000
    if not out.scores:
        raise JevError("; ".join(out.errors[:3]) or "no scores returned")
    return out


# --------------------------------------------------------------------------
# selection


@dataclass(slots=True)
class RankedDoc:
    doc_id: str
    chunk_id: str
    score: float


def rank_documents(
    pool: dict[str, dict[str, Any]],
    scores: dict[str, float],
    *,
    limit: int,
) -> list[RankedDoc]:
    """The top `limit` DOCUMENTS by their best-scoring chunk.

    Units are documents because that is what the recall floor counts and what
    a reader receives -- ranking chunks would let one long document take every
    slot. A document scores as its best chunk: a long doc whose fourth chunk is
    the answer must not be ranked on its first.

    Ties break on pool order, which is retrieval order -- deterministic, and a
    sensible prior when Jev cannot tell two chunks apart.
    """
    best: dict[str, RankedDoc] = {}
    order: dict[str, int] = {}
    for i, (cid, hit) in enumerate(pool.items()):
        s = scores.get(cid)
        if s is None:
            continue
        doc = hit.get("doc_id") or cid
        order.setdefault(doc, i)
        if doc not in best or s > best[doc].score:
            best[doc] = RankedDoc(doc_id=doc, chunk_id=cid, score=s)
    ranked = sorted(best.values(), key=lambda r: (-r.score, order[r.doc_id]))
    return ranked[:limit]


def confidence_for(best_score: float | None) -> str:
    """Map the best document's probability onto the response's confidence.

    Cut from Phase 0's answerability data (gpt-4.1-mini grader, 368 traces):
    with the best doc below 0.4 the delivered set answered the query 0.8% of
    the time, against 37.7% above it.
    """
    if best_score is None or best_score < 0.4:
        return "low"
    return "high" if best_score >= 0.7 else "medium"


def near_copy(a: str, b: str, *, threshold: float = 0.8) -> bool:
    """True when two queries share almost all their words.

    Used to skip a retry whose rewrite is the same question: the same query
    returns the same pool, so the second search would cost ~2s and change
    nothing.
    """
    ta = {w for w in a.lower().split() if len(w) > 2}
    tb = {w for w in b.lower().split() if len(w) > 2}
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / len(ta | tb) >= threshold


# --------------------------------------------------------------------------
# extraction: the two search options a Choice can decide
# --------------------------------------------------------------------------
#
# The gpt-oss extractor emits entities, sub-queries, a sort, and a doc-type
# filter. Jev cannot write text, so it cannot produce the first two. The last
# two ARE closed choices and change the whole pool: `sort=recency` reorders
# every channel, and a `doc_types` filter hard-restricts it -- a wrong class
# zeroes the pool. The class list mirrors the extractor prompt's own map
# (extractor.py, "search_options.doc_types"), so both are choosing from the
# same menu.

DOC_CLASSES: dict[str, list[str] | None] = {
    "no_class": None,
    "pull_requests": ["github.pull_request"],
    "github_issues": ["github.issue"],
    "commits": ["github.commit"],
    "code_reviews": ["github.review"],
    "releases": ["github.release"],
    "tickets": ["linear.issue"],
    "ticket_comments": ["linear.comment"],
    "slack_messages": ["slack.message", "slack.thread"],
    "notion_pages": ["notion.page", "notion.database"],
    "sentry_errors": ["sentry.issue", "sentry.event"],
    "meetings": ["granola.meeting"],
    "claude_code_sessions": ["claude_code.session"],
}

_SORT_QUESTION = {
    "type": "choice",
    "instructions": (
        "How should search results for state.query be ordered? `recency` "
        "only when the query asks for the most recent / latest / last / "
        "newest activity; otherwise `relevance`."
    ),
    "criteria": {
        "relevance": "a topical or conceptual question; order by relevance",
        "recency": "asks for the latest / most recent / newest item(s)",
    },
}

_CLASS_QUESTION = {
    "type": "choice",
    "instructions": (
        "Is state.query asking about a whole CLASS of item (e.g. 'the latest "
        "PRs', 'tickets in progress'), and if so which one? Choose `no_class` "
        "when it asks about a specific thing or a topic -- a named PR, an "
        "error message, a concept. When unsure, choose `no_class`: a wrong "
        "class hides every other result."
    ),
    "criteria": {
        "no_class": "a specific item or a topic, not a class of items",
        "pull_requests": "GitHub pull requests / PRs",
        "github_issues": "GitHub issues",
        "commits": "git commits",
        "code_reviews": "code reviews / PR reviews",
        "releases": "releases",
        "tickets": "tickets (Linear)",
        "ticket_comments": "comments on tickets",
        "slack_messages": "Slack messages",
        "notion_pages": "Notion pages",
        "sentry_errors": "Sentry issues, errors or incidents",
        "meetings": "meetings (Granola notes)",
        "claude_code_sessions": "Claude Code sessions",
    },
}


@dataclass(slots=True)
class ExtractionChoice:
    sort: str
    sort_confidence: float
    doc_class: str
    doc_types: list[str] | None
    class_confidence: float
    elapsed_ms: float


async def extract_options(
    query: str,
    *,
    api_key: str,
    client: httpx.AsyncClient | None = None,
) -> ExtractionChoice:
    """Jev's answer for `sort` and `doc_types`. Raises `JevError` on failure."""
    if not api_key:
        raise JevError("TYPESAFE_API_KEY is not configured")
    t0 = time.perf_counter()
    client = client or _shared_client()
    try:
        resp = await _post(
            client,
            api_key,
            {"query": query},
            {"sort": _SORT_QUESTION, "doc_class": _CLASS_QUESTION},
        )
    except httpx.HTTPError as exc:
        raise JevError(f"{type(exc).__name__}: {str(exc)[:160] or '<empty>'}") from exc
    if resp.status_code != 200:
        raise JevError(f"http_{resp.status_code}: {resp.text[:160]}")
    try:
        answers = resp.json().get("answers") or {}
        sort = answers["sort"]
        cls = answers["doc_class"]
    except (ValueError, KeyError, AttributeError) as exc:
        raise JevError(f"malformed answer: {type(exc).__name__}") from exc
    sort_choice = sort.get("choice") if sort.get("choice") in ("relevance", "recency") else "relevance"
    class_choice = cls.get("choice") if cls.get("choice") in DOC_CLASSES else "no_class"
    return ExtractionChoice(
        sort=sort_choice,
        sort_confidence=float(sort.get("confidence") or 0.0),
        doc_class=class_choice,
        doc_types=DOC_CLASSES[class_choice],
        class_confidence=float(cls.get("confidence") or 0.0),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )
