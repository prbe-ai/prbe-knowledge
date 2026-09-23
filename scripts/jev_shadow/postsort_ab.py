"""Post-sort A/B -- does research-os's transcript demotion help or hurt, given Jev's order?

research-os multiplies every claude_code/codex/pi hit's score by 0.6 before its
top_k cut (`app/search/service.py` `_SOURCE_SCORE_MULTIPLIERS`). The engine's
score is a RANK (1.0, 0.99, ...) and the engine returns at most 10 documents, so
the multiplier is a hard partition: non-transcripts first, transcripts last,
Jev's order kept inside each group. Two arms, both pure functions of the same
10-document list the engine returned:

    A  today     partition(engine order) -> _dedupe_by_session -> cut k
    B  proposed  engine order            -> _dedupe_by_session -> cut k

Both arms run research-os's real post-processing, ported verbatim below (the
session dedupe swaps a transcript INTO its digest's slot whatever the scores
say). The DB-backed lenses (workspace, project) are membership filters that are
identical across arms and absent on unscoped searches; session self-exclusion
is not recorded in a trace and is skipped -- both noted in the report.

Subcommands:
    parity    replay arm A on captured (trace, /v1/search) pairs; must match
    build     assemble the corpus: live Jev-era blobs + Phase 0's arm D rows
    judge     per-document YES/NO on Opus (position-blind), set-answerability,
              optional gpt-4.1-mini cross-grader, optional repeat pass
    report    metrics, paired stats, decision rule, cost -> markdown
    calibrate write the 40-query hand-grading slice; --human scores it
    tiers     size the engine-side tier-penalty follow-up (arm C) on the same labels

Judge prompt is Phase 0's, verbatim: changing it is a grader change.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.retrieval.agent import jev_shadow as js  # noqa: E402

# research-os `AGENT_SOURCE_IDS` (app/ingestion/agent_sources.py)
TRANSCRIPT_SOURCES = ("claude_code", "codex", "pi")
CUSTOM_INGEST = "custom_ingest"
SOURCE_KEY_COLON_ESCAPE = "%3A"
KS = (4, 5, 8, 10)
SET_KS = (5, 8)
BODY_LIMIT = 1500
SET_BODY_LIMIT = 700

# Phase 0's judge prompts, verbatim (scripts/jev_shadow/judge.py).
PROMPT = """You are grading a search engine's results.

QUERY:
{query}

DOCUMENT ({doc_id}):
{body}

Answer with exactly one word.
Would a person who asked that query consider this document a useful result --
does it contain, or directly point to, part of the answer?
Answer YES or NO."""

SET_PROMPT = """You are grading a search engine's results.

QUERY:
{query}

THE RESULTS IT RETURNED:
{bodies}

Answer with exactly one word.
Taken together, do these results contain enough to answer that query?
Answer YES or NO."""

# Anthropic first-party rates, $/MTok (claude-api skill table, 2026-06-24).
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "gpt-4.1-mini": (0.40, 1.60),  # approximate, OpenAI list price
}


# --------------------------------------------------------------------------
# research-os post-processing, ported verbatim from app/search/service.py


def transcript_identity(doc_id: str, customer_id: str) -> tuple[str, str] | None:
    source, separator, _tail = doc_id.partition(":")
    if not separator or source not in TRANSCRIPT_SOURCES:
        return None
    parts = doc_id.split(":", 2)
    if len(parts) != 3 or parts[1] != customer_id:
        return None
    return (source, parts[2]) if parts[2] else None


def digest_identity(doc_id: str, customer_id: str) -> tuple[str, str] | None:
    parts = doc_id.split(":", 3)
    if len(parts) != 4 or parts[0] != CUSTOM_INGEST or parts[1] != customer_id:
        return None
    tail = parts[3].split(":", 2)
    if len(tail) != 3 or tail[0] != "session_digest" or tail[1] not in TRANSCRIPT_SOURCES:
        return None
    return (tail[1], tail[2]) if tail[2] else None


def session_behind(doc_id: str, customer_id: str) -> tuple[str, str] | None:
    return transcript_identity(doc_id, customer_id) or digest_identity(doc_id, customer_id)


def dedupe_by_session(doc_ids: list[str], customer_id: str) -> list[str]:
    """`_dedupe_by_session`: one hit per session, transcript beats its digest IN PLACE."""
    best: dict[tuple[str, str], int] = {}
    kept: list[str] = []
    for doc in doc_ids:
        session = session_behind(doc, customer_id)
        if session is None:
            kept.append(doc)
            continue
        is_transcript = transcript_identity(doc, customer_id) is not None
        seen_at = best.get(session)
        if seen_at is None:
            best[session] = len(kept)
            kept.append(doc)
        elif is_transcript and transcript_identity(kept[seen_at], customer_id) is None:
            kept[seen_at] = doc
    return kept


def source_of(doc_id: str, hit: dict[str, Any] | None = None) -> str:
    """`_demoted_score`'s source: the hit's source_system, else the doc_id prefix."""
    src = (hit or {}).get("source_system")
    if src:
        return str(src)
    return doc_id.split(":", 1)[0] if ":" in doc_id else ""


def is_transcript(doc_id: str, hit: dict[str, Any] | None = None) -> bool:
    return source_of(doc_id, hit) in TRANSCRIPT_SOURCES


def partition(engine_docs: list[str], sources: dict[str, str]) -> list[str]:
    """What `hits.sort(key=_demoted_score, reverse=True)` does to a <=10-doc ordinal list:
    every transcript (<=0.60) sorts under every non-transcript (>=0.91), order kept inside."""
    return sorted(engine_docs, key=lambda d: 1 if sources.get(d) in TRANSCRIPT_SOURCES else 0)


def arm_a(engine_docs: list[str], sources: dict[str, str], customer_id: str, k: int) -> list[str]:
    return dedupe_by_session(partition(engine_docs, sources), customer_id)[:k]


def arm_b(engine_docs: list[str], customer_id: str, k: int) -> list[str]:
    return dedupe_by_session(list(engine_docs), customer_id)[:k]


# --------------------------------------------------------------------------
# blobs


def engine_list(blob: dict[str, Any]) -> list[str]:
    """The engine's delivered document order for a live trace (<=10 docs)."""
    return js.arm_today(blob)


def jev_ranked(blob: dict[str, Any]) -> list[str]:
    return [d for d, _ in ((blob.get("selection") or {}).get("ranked") or [])]


def doc_bodies(blob: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cid, hit in js.pool_chunks(blob.get("prefanout")).items():
        out.setdefault(js.doc_of(cid, hit), hit)
    return out


def body_of(hit: dict[str, Any], limit: int = BODY_LIMIT) -> str:
    t = (hit.get("title") or "").strip()
    c = (hit.get("content") or "").strip()[:limit]
    return f"{t}\n{c}" if t else c


def blob_index(root: str) -> dict[str, str]:
    """trace_id -> path for every *.json.gz under root."""
    out: dict[str, str] = {}
    for p in glob.glob(os.path.join(root, "**", "*.json.gz"), recursive=True):
        out[os.path.basename(p)[: -len(".json.gz")]] = p
    return out


def make_record(
    *,
    trace_id: str,
    origin: str,
    customer_id: str,
    day: str | None,
    query: str,
    engine_docs: list[str],
    bodies: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    engine_docs = [d for d in engine_docs if d in bodies][: js.DELIVERY_BUDGET_DOCS]
    if not engine_docs:
        return None
    sources = {d: source_of(d, bodies[d]) for d in engine_docs}
    arms = {
        "A": {str(k): arm_a(engine_docs, sources, customer_id, k) for k in KS},
        "B": {str(k): arm_b(engine_docs, customer_id, k) for k in KS},
    }
    n_tr = sum(1 for d in engine_docs if sources[d] in TRANSCRIPT_SOURCES)
    return {
        "trace_id": trace_id,
        "origin": origin,
        "customer_id": customer_id,
        "day": day,
        "query": query,
        "engine_docs": engine_docs,
        "sources": sources,
        "n_transcripts": n_tr,
        "affected": arms["A"]["10"] != arms["B"]["10"],
        "arms": arms,
        "bodies": {d: body_of(bodies[d]) for d in engine_docs},
        "bodies_short": {d: body_of(bodies[d], SET_BODY_LIMIT) for d in engine_docs},
    }


# --------------------------------------------------------------------------
# parity


def cmd_parity(args: argparse.Namespace) -> int:
    pairs = json.load(open(args.pairs))
    index = blob_index(args.blobs)
    ok = mism = missing = 0
    lines: list[str] = []
    for pair in pairs:
        tids = pair.get("trace_ids") or []
        if len(tids) != 1:
            lines.append(f"SKIP multi/none trace: {pair['query'][:50]!r} traces={tids}")
            continue
        path = index.get(tids[0])
        if not path:
            missing += 1
            lines.append(f"MISSING blob {tids[0]} for {pair['query'][:50]!r}")
            continue
        blob = js.load_blob(path)
        cust = blob.get("customer_id") or ""
        bodies = doc_bodies(blob)
        eng = engine_list(blob)
        ranked = jev_ranked(blob)
        sources = {d: source_of(d, bodies.get(d)) for d in eng}
        replay = arm_a(eng, sources, cust, pair["top_k"])
        delivered = [h["doc_id"] for h in pair["delivered"]]
        if ranked and ranked != eng[: len(ranked)]:
            lines.append(f"NOTE selection.ranked != delivered prefix on {tids[0]}")
        if replay == delivered:
            ok += 1
            lines.append(f"OK   {tids[0]} k={pair['top_k']} n={len(delivered)} selector={blob.get('selector')}")
        else:
            mism += 1
            lines.append(
                f"MISMATCH {tids[0]} k={pair['top_k']} selector={blob.get('selector')}\n"
                f"   engine   {eng}\n   replay   {replay}\n   delivered {delivered}"
            )
    print("\n".join(lines))
    print(f"\nPARITY: {ok} match, {mism} mismatch, {missing} missing blob, {len(pairs)} pairs")
    return 0 if mism == 0 and ok >= args.min_ok else 1


# --------------------------------------------------------------------------
# build


def cmd_build(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    census: collections.Counter = collections.Counter()

    # live, Jev-era
    for path in sorted(glob.glob(os.path.join(args.live, "**", "*.json.gz"), recursive=True)):
        try:
            blob = js.load_blob(path)
        except Exception as exc:  # noqa: BLE001
            census[("live", "unreadable", type(exc).__name__)] += 1
            continue
        cust = blob.get("customer_id") or ""
        if cust in js.EXCLUDED_TENANTS:
            census[("live", cust, "excluded_tenant")] += 1
            continue
        if blob.get("selector") != "jev" or blob.get("status") != "ok":
            census[("live", cust, f"skip:{blob.get('selector')}/{blob.get('status')}")] += 1
            continue
        rec = make_record(
            trace_id=blob.get("trace_id") or os.path.basename(path)[:-8],
            origin="live",
            customer_id=cust,
            day=(blob.get("timestamp_utc") or "")[:10],
            query=blob.get("query") or "",
            engine_docs=engine_list(blob),
            bodies=doc_bodies(blob),
        )
        if rec is None:
            census[("live", cust, "empty")] += 1
            continue
        census[("live", cust, "affected" if rec["affected"] else "unaffected")] += 1
        records.append(rec)

    # Phase 0 arm D rows + their blobs
    if args.phase0:
        index = blob_index(args.phase0_blobs)
        rows = [json.loads(l) for l in gzip.open(args.phase0, "rt")]
        for row in rows:
            cust = row.get("customer_id") or ""
            if cust in js.EXCLUDED_TENANTS or row.get("status") != "ok" or not row.get("d_docs"):
                census[("phase0", cust, "skip")] += 1
                continue
            path = index.get(row["trace_id"])
            if not path:
                census[("phase0", cust, "missing_blob")] += 1
                continue
            blob = js.load_blob(path)
            rec = make_record(
                trace_id=row["trace_id"],
                origin="phase0",
                customer_id=cust,
                day=row.get("day"),
                query=blob.get("query") or "",
                engine_docs=list(row["d_docs"]),
                bodies=doc_bodies(blob),
            )
            if rec is None:
                census[("phase0", cust, "empty")] += 1
                continue
            census[("phase0", cust, "affected" if rec["affected"] else "unaffected")] += 1
            records.append(rec)

    affected = [r for r in records if r["affected"]]
    live_aff = [r for r in affected if r["origin"] == "live"]
    p0_aff = [r for r in affected if r["origin"] == "phase0"]
    rng.shuffle(p0_aff)
    take = live_aff + p0_aff[: max(0, args.n - len(live_aff))]
    with open(args.out, "w") as fh:
        for r in take:
            fh.write(json.dumps(r) + "\n")
    with open(args.out + ".census.json", "w") as fh:
        json.dump(
            {
                "census": {"|".join(map(str, k)): v for k, v in sorted(census.items())},
                "records_total": len(records),
                "affected_total": len(affected),
                "affected_live": len(live_aff),
                "affected_phase0": len(p0_aff),
                "sampled": len(take),
                "seed": args.seed,
            },
            fh,
            indent=1,
        )
    print(f"records {len(records)}; affected {len(affected)} (live {len(live_aff)}, phase0 {len(p0_aff)}); sampled {len(take)} -> {args.out}")
    for k, v in sorted(census.items()):
        print(f"  {v:5d}  {' | '.join(map(str, k))}")
    return 0


# --------------------------------------------------------------------------
# judge


class Usage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tokens: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])

    def add(self, model: str, inp: int, out: int) -> None:
        with self.lock:
            t = self.tokens[model]
            t[0] += inp
            t[1] += out
            t[2] += 1

    def dollars(self) -> dict[str, float]:
        return {
            m: (t[0] * PRICES.get(m, (0, 0))[0] + t[1] * PRICES.get(m, (0, 0))[1]) / 1e6
            for m, t in self.tokens.items()
        }


_YES_NO = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def _parse_yes_no(txt: str) -> bool | None:
    """The verdict is the LAST standalone YES/NO in the text.

    Phase 0's Haiku judge ran with max_tokens=4, so the first token was the
    answer. Opus 5.5 reasons first and sometimes writes a sentence of prose
    before the word ("The query asks whether ... NO"); a start-of-text match
    read a quarter of those as unanswered. The final token is the verdict in
    both shapes.
    """
    found = _YES_NO.findall(txt or "")
    if not found:
        return None
    return found[-1].upper() == "YES"


#: On a `refusal` stop, the same prompt is re-asked on these, in order. Opus 5.5's
#: `bio` classifier refused ~20% of judge calls in the first run and 85% of those
#: were transcripts (biology tenants' sessions), so excluding refused documents
#: would bias exactly the transcript-vs-record comparison this study makes.
#: Which model answered is recorded on every row (`answered_by`).
REFUSAL_FALLBACKS = ("claude-opus-5",)


def ask_anthropic(
    client: Any, model: str, effort: str, prompt: str, usage: Usage
) -> tuple[bool | None, str, str | None, str]:
    """(verdict, raw text tail, stop_reason, answered_by)."""
    import anthropic

    raw = ""
    stop = None
    answered_by = model
    for candidate in (model, *REFUSAL_FALLBACKS):
        max_tokens = 1024
        refused = False
        for attempt in range(3):
            try:
                r = client.messages.create(
                    model=candidate,
                    max_tokens=max_tokens,
                    output_config={"effort": effort},
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.APIStatusError as exc:
                if exc.status_code in (429, 500, 502, 503, 529):
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return None, f"http {exc.status_code}", None, candidate
            except anthropic.APIConnectionError:
                time.sleep(2.0 * (attempt + 1))
                continue
            usage.add(candidate, r.usage.input_tokens, r.usage.output_tokens)
            stop = r.stop_reason
            answered_by = candidate
            if r.stop_reason == "max_tokens":
                max_tokens = 4096
                continue
            if r.stop_reason == "refusal":
                sd = getattr(r, "stop_details", None)
                raw = f"refusal:{getattr(sd, 'category', None)}"
                refused = True
                break
            txt = "".join(b.text for b in r.content if b.type == "text")
            raw = txt[-160:]
            v = _parse_yes_no(txt)
            if v is not None:
                return v, raw, stop, candidate
            # No verdict token at all: one more try before giving up on this document.
        if not refused:
            break
    return None, raw, stop, answered_by


def ask_openai(client: Any, key: str, model: str, prompt: str, usage: Usage) -> tuple[bool | None, str, str | None]:
    for attempt in range(3):
        try:
            r = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None, f"http {r.status_code}", None
            body = r.json()
            u = body.get("usage") or {}
            usage.add(model, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))
            txt = body["choices"][0]["message"]["content"] or ""
            return _parse_yes_no(txt), txt[-160:], None
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return None, "", None


def _done_keys(path: str) -> set[str]:
    """Keys already ANSWERED. A row whose verdict is None is re-asked on restart."""
    keys: set[str] = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if row.get("verdict") is not None:
                keys.add(row["key"])
    return keys


def cmd_judge(args: argparse.Namespace) -> int:
    import httpx

    records = [json.loads(l) for l in open(args.arms)]
    rng = random.Random(args.seed)
    usage = Usage()
    done = _done_keys(args.out)
    out_lock = threading.Lock()
    out_fh = open(args.out, "a")

    anthropic_client = None
    if args.model:
        import anthropic

        key = os.environ.get("JUDGE_KEY") or ""
        if not key:
            print("JUDGE_KEY missing", file=sys.stderr)
            return 2
        anthropic_client = anthropic.Anthropic(api_key=key, max_retries=5, timeout=120.0)
    openai_key = os.environ.get("OPENAI_JUDGE_KEY") or ""
    http = httpx.Client()

    tasks: list[dict[str, Any]] = []
    for rec in records:
        tid = rec["trace_id"]
        repeat = rng.random() < args.repeat_frac
        for doc in rec["engine_docs"]:
            prompt = PROMPT.format(query=rec["query"], doc_id=doc, body=rec["bodies"][doc])
            if args.model:
                tasks.append({"key": f"doc|{tid}|{doc}|{args.model}|1", "kind": "doc", "prompt": prompt, "model": args.model, "trace_id": tid, "doc": doc, "pass": 1})
                if repeat:
                    tasks.append({"key": f"doc|{tid}|{doc}|{args.model}|2", "kind": "doc", "prompt": prompt, "model": args.model, "trace_id": tid, "doc": doc, "pass": 2})
            if args.openai_model and openai_key:
                tasks.append({"key": f"doc|{tid}|{doc}|{args.openai_model}|1", "kind": "doc", "prompt": prompt, "model": args.openai_model, "trace_id": tid, "doc": doc, "pass": 1})
        if args.model and not args.no_sets:
            for k in SET_KS:
                sets = {arm: rec["arms"][arm][str(k)] for arm in ("A", "B")}
                same = set(sets["A"]) == set(sets["B"])
                for arm, docs in sets.items():
                    if same and arm == "B":
                        continue  # identical set: A's verdict is copied at report time
                    joined = "\n\n---\n\n".join(f"[{d}] {rec['bodies_short'][d]}" for d in docs)
                    if not joined:
                        continue
                    prompt = SET_PROMPT.format(query=rec["query"], bodies=joined)
                    tasks.append({"key": f"set|{tid}|{arm}|{k}|{args.model}", "kind": "set", "prompt": prompt, "model": args.model, "trace_id": tid, "arm": arm, "k": k, "same_set": same})
    tasks = [t for t in tasks if t["key"] not in done]
    print(f"{len(tasks)} judge calls to make ({len(done)} already done)", file=sys.stderr)

    def work(t: dict[str, Any]) -> dict[str, Any]:
        if t["model"] == args.openai_model:
            v, raw, stop = ask_openai(http, openai_key, t["model"], t["prompt"], usage)
            answered_by = t["model"]
        else:
            v, raw, stop, answered_by = ask_anthropic(anthropic_client, t["model"], args.effort, t["prompt"], usage)
        row = {k: v_ for k, v_ in t.items() if k != "prompt"}
        row["verdict"] = v
        row["raw"] = raw
        row["stop"] = stop
        row["answered_by"] = answered_by
        return row

    n = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for f in as_completed(futs):
            row = f.result()
            with out_lock:
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n}/{len(tasks)} done, {time.time() - t0:.0f}s, spend so far {usage.dollars()}", file=sys.stderr)
    out_fh.close()
    with open(args.out + ".usage.json", "a") as fh:
        fh.write(json.dumps({"ts": time.time(), "tokens": usage.tokens, "dollars": usage.dollars(), "calls": n}) + "\n")
    print(f"done {n} calls; tokens {dict(usage.tokens)}; dollars {usage.dollars()}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# metrics


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at(order: list[str], rel: dict[str, int], k: int, pool: list[str]) -> float | None:
    ideal = sorted((rel[d] for d in pool), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return None
    return dcg([rel[d] for d in order[:k]]) / idcg


def prec_at(order: list[str], rel: dict[str, int], k: int) -> float:
    top = order[:k]
    return sum(rel[d] for d in top) / k if top else 0.0


def mrr(order: list[str], rel: dict[str, int]) -> float:
    for i, d in enumerate(order):
        if rel[d]:
            return 1.0 / (i + 1)
    return 0.0


def bootstrap_ci(diffs: list[float], rng: random.Random, n: int = 4000) -> tuple[float, float]:
    if not diffs:
        return (float("nan"), float("nan"))
    means = []
    m = len(diffs)
    for _ in range(n):
        s = 0.0
        for _ in range(m):
            s += diffs[rng.randrange(m)]
        means.append(s / m)
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def sign_test(diffs: list[float]) -> tuple[int, int, int, float, float]:
    """(wins, losses, ties, p_one_sided B>A, p_two_sided) by exact binomial."""
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties = len(diffs) - wins - losses
    n = wins + losses
    if n == 0:
        return wins, losses, ties, 1.0, 1.0
    p_ge = sum(math.comb(n, i) for i in range(wins, n + 1)) / 2**n
    p_le = sum(math.comb(n, i) for i in range(0, wins + 1)) / 2**n
    return wins, losses, ties, p_ge, min(1.0, 2 * min(p_ge, p_le))


def kappa(pairs: list[tuple[bool, bool]]) -> tuple[float, float]:
    """(raw agreement, Cohen's kappa)."""
    if not pairs:
        return float("nan"), float("nan")
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return agree, (agree - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_report(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    records = {r["trace_id"]: r for r in (json.loads(l) for l in open(args.arms))}
    verdicts: dict[tuple[str, str, str, int], bool | None] = {}
    sets: dict[tuple[str, str, int], bool | None] = {}
    same_set: dict[tuple[str, int], bool] = {}
    answered_by: collections.Counter = collections.Counter()
    refused_by_source: collections.Counter = collections.Counter()
    for line in open(args.verdicts):
        row = json.loads(line)
        # A re-asked row follows its unanswered predecessor; never let a None overwrite an answer.
        if row["kind"] == "doc":
            key = (row["trace_id"], row["doc"], row["model"], row["pass"])
            if row["verdict"] is not None or key not in verdicts:
                verdicts[key] = row["verdict"]
            if row["verdict"] is not None and row["model"] == args.model and row["pass"] == 1:
                answered_by[row.get("answered_by") or row["model"]] += 1
            if str(row.get("raw", "")).startswith("refusal") or row.get("stop") == "refusal":
                rec_ = records.get(row["trace_id"])
                if rec_:
                    refused_by_source[rec_["sources"].get(row["doc"], "?")] += 1
        else:
            key2 = (row["trace_id"], row["arm"], row["k"])
            if row["verdict"] is not None or key2 not in sets:
                sets[key2] = row["verdict"]
            same_set[(row["trace_id"], row["k"])] = row.get("same_set", False)
    model = args.model
    usage_lines = []
    if os.path.exists(args.verdicts + ".usage.json"):
        usage_lines = [json.loads(l) for l in open(args.verdicts + ".usage.json")]

    per_trace: list[dict[str, Any]] = []
    unjudged_traces = 0
    for tid, rec in records.items():
        rel: dict[str, int] = {}
        missing = False
        for d in rec["engine_docs"]:
            v = verdicts.get((tid, d, model, 1))
            if v is None:
                missing = True
                break
            rel[d] = 1 if v else 0
        if missing:
            unjudged_traces += 1
            continue
        row: dict[str, Any] = {"trace_id": tid, "origin": rec["origin"], "customer_id": rec["customer_id"], "rel": rel}
        for arm in ("A", "B"):
            for k in KS:
                order = rec["arms"][arm][str(k)]
                row[f"P@{k}_{arm}"] = prec_at(order, rel, k)
                row[f"NDCG@{k}_{arm}"] = ndcg_at(order, rel, k, rec["engine_docs"])
            row[f"MRR_{arm}"] = mrr(rec["arms"][arm]["10"], rel)
        per_trace.append(row)

    out: list[str] = []
    out.append("# Post-sort A/B report: research-os transcript demotion vs Jev order\n")
    out.append(f"Judge: `{model}` (effort {args.effort}), Phase 0 prompt verbatim, per document, position-blind. "
               f"Traces judged: {len(per_trace)} (unjudged/incomplete: {unjudged_traces}).\n")
    by_origin = collections.Counter(r["origin"] for r in per_trace)
    by_cust = collections.Counter(r["customer_id"] for r in per_trace)
    out.append(f"Origin: {dict(by_origin)}. Tenants: {dict(by_cust)}.\n")

    out.append("## Primary and guardrail endpoints (paired per trace, B − A)\n")
    out.append("| metric | A (today) | B (proposed) | mean diff | 95% CI | wins/losses/ties | p one-sided B>A | p two-sided |")
    out.append("|---|---|---|---|---|---|---|---|")
    results: dict[str, dict[str, Any]] = {}
    metrics = [f"NDCG@{k}" for k in (10, 8, 5, 4)] + [f"P@{k}" for k in (4, 5, 8, 10)] + ["MRR"]
    for m in metrics:
        pairs = [(r[f"{m}_A"], r[f"{m}_B"]) for r in per_trace if r[f"{m}_A"] is not None and r[f"{m}_B"] is not None]
        if not pairs:
            continue
        diffs = [b - a for a, b in pairs]
        ma = sum(a for a, _ in pairs) / len(pairs)
        mb = sum(b for _, b in pairs) / len(pairs)
        lo, hi = bootstrap_ci(diffs, rng)
        w, l, t, p1, p2 = sign_test(diffs)
        results[m] = {"A": ma, "B": mb, "diff": mb - ma, "ci": (lo, hi), "wins": w, "losses": l, "ties": t, "p1": p1, "p2": p2, "n": len(pairs)}
        out.append(f"| {m} (n={len(pairs)}) | {ma:.3f} | {mb:.3f} | {mb - ma:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {w}/{l}/{t} | {p1:.3g} | {p2:.3g} |")

    out.append("\n## Set answerability (judge on the delivered SET, per arm)\n")
    out.append("| k | A answerable | B answerable | traces (sets differ) | traces (sets same) |")
    out.append("|---|---|---|---|---|")
    for k in SET_KS:
        a_yes = b_yes = n_diff = n_same = 0
        a_n = b_n = 0
        for r in per_trace:
            tid = r["trace_id"]
            va = sets.get((tid, "A", k))
            same = same_set.get((tid, k), False)
            vb = va if same else sets.get((tid, "B", k))
            if same:
                n_same += 1
            else:
                n_diff += 1
            if va is not None:
                a_n += 1
                a_yes += int(va)
            if vb is not None:
                b_n += 1
                b_yes += int(vb)
        out.append(f"| {k} | {100 * a_yes / max(1, a_n):.1f}% ({a_yes}/{a_n}) | {100 * b_yes / max(1, b_n):.1f}% ({b_yes}/{b_n}) | {n_diff} | {n_same} |")

    out.append("\n## Q3: are the documents the multiplier promotes better than the ones it demotes?\n")
    moved_up = moved_down = 0
    up_rel = down_rel = 0
    tr_rel = tr_n = ntr_rel = ntr_n = 0
    for r in per_trace:
        rec = records[r["trace_id"]]
        a10, b10 = rec["arms"]["A"]["10"], rec["arms"]["B"]["10"]
        pos_a = {d: i for i, d in enumerate(a10)}
        pos_b = {d: i for i, d in enumerate(b10)}
        for d in b10:
            if d not in pos_a:
                continue
            if pos_a[d] < pos_b[d]:
                moved_up += 1
                up_rel += r["rel"][d]
            elif pos_a[d] > pos_b[d]:
                moved_down += 1
                down_rel += r["rel"][d]
        for d in rec["engine_docs"]:
            if rec["sources"][d] in TRANSCRIPT_SOURCES:
                tr_n += 1
                tr_rel += r["rel"][d]
            else:
                ntr_n += 1
                ntr_rel += r["rel"][d]
    out.append(f"- Documents the partition moved UP (non-transcripts): {moved_up}, useful {100 * up_rel / max(1, moved_up):.1f}%")
    out.append(f"- Documents the partition moved DOWN (transcripts): {moved_down}, useful {100 * down_rel / max(1, moved_down):.1f}%")
    out.append(f"- All transcripts in the engine's top-10: {tr_n}, useful {100 * tr_rel / max(1, tr_n):.1f}%; all non-transcripts: {ntr_n}, useful {100 * ntr_rel / max(1, ntr_n):.1f}%")
    # position-1 view: what sits at rank 1 under each arm, and how often it is useful
    top1 = {"A": [0, 0], "B": [0, 0]}
    for r in per_trace:
        rec = records[r["trace_id"]]
        for arm in ("A", "B"):
            d = rec["arms"][arm]["10"][0]
            top1[arm][0] += 1
            top1[arm][1] += r["rel"][d]
    out.append(f"- Rank-1 document useful: A {100 * top1['A'][1] / max(1, top1['A'][0]):.1f}%, B {100 * top1['B'][1] / max(1, top1['B'][0]):.1f}%")

    out.append("\n## Grader agreement\n")
    self_pairs = []
    cross_pairs = []
    other_models = sorted({m for (_, _, m, _) in verdicts} - {model})
    for (tid, d, m, p), v in verdicts.items():
        if m != model or p != 1 or v is None:
            continue
        v2 = verdicts.get((tid, d, model, 2))
        if v2 is not None:
            self_pairs.append((v, v2))
        for om in other_models:
            vo = verdicts.get((tid, d, om, 1))
            if vo is not None:
                cross_pairs.append((v, vo))
    a, kp = kappa(self_pairs)
    out.append(f"- {model} self-agreement (repeat pass): raw {a:.3f}, kappa {kp:.3f}, n={len(self_pairs)}")
    if other_models:
        a2, kp2 = kappa(cross_pairs)
        out.append(f"- {model} vs {', '.join(other_models)}: raw {a2:.3f}, kappa {kp2:.3f}, n={len(cross_pairs)}")
    yes_rate = sum(1 for (_, _, m, p), v in verdicts.items() if m == model and p == 1 and v) / max(1, sum(1 for (_, _, m, p), v in verdicts.items() if m == model and p == 1 and v is not None))
    out.append(f"- {model} YES rate over judged documents: {100 * yes_rate:.1f}%")
    n_ans = sum(answered_by.values())
    out.append(f"- Answered by: " + ", ".join(f"{m} {n} ({100 * n / max(1, n_ans):.1f}%)" for m, n in answered_by.most_common()))
    if refused_by_source:
        out.append(f"- `{model}` refusals seen (stop_reason=refusal, re-asked on the fallback), by document source: {dict(refused_by_source)}")
    out.append("- Human calibration (40-query slice): NOT DONE unless `calibrate --human` was scored; see below.")

    out.append("\n## Per-origin and per-tenant NDCG@10 diff (B − A)\n")
    out.append("| slice | n | A | B | diff | wins/losses/ties |")
    out.append("|---|---|---|---|---|---|")
    for label, keyf in (("origin", lambda r: r["origin"]), ("tenant", lambda r: r["customer_id"])):
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in per_trace:
            groups[keyf(r)].append(r)
        for g, rows in sorted(groups.items()):
            pairs = [(r["NDCG@10_A"], r["NDCG@10_B"]) for r in rows if r["NDCG@10_A"] is not None and r["NDCG@10_B"] is not None]
            if not pairs:
                continue
            ma = sum(a for a, _ in pairs) / len(pairs)
            mb = sum(b for _, b in pairs) / len(pairs)
            w, l, t, _, _ = sign_test([b - a for a, b in pairs])
            out.append(f"| {label}={g} | {len(pairs)} | {ma:.3f} | {mb:.3f} | {mb - ma:+.3f} | {w}/{l}/{t} |")

    out.append("\n## Decision rule (plan §5), evaluated\n")
    primary = results.get("NDCG@10")
    guards = {m: results.get(m) for m in ("P@4", "P@5", "P@8")}
    if primary:
        sig = primary["p1"] < 0.05
        breached = [m for m, g in guards.items() if g and g["ci"][0] <= -0.02]
        out.append(f"- Branch 1 (judge calibration): human slice not scored → the human gate is OPEN; model cross-agreement stands in (see above).")
        out.append(f"- Primary NDCG@10: diff {primary['diff']:+.3f}, CI [{primary['ci'][0]:+.3f}, {primary['ci'][1]:+.3f}], p={primary['p1']:.3g} → {'significant' if sig else 'not significant'}")
        for m, g in guards.items():
            if g:
                out.append(f"- Guardrail {m}: diff {g['diff']:+.3f}, CI lower bound {g['ci'][0]:+.3f} → {'BREACHED' if g['ci'][0] <= -0.02 else 'holds'}")
        if breached:
            out.append(f"- **Branch 4: guardrail breached ({', '.join(breached)}). Do not ship the deletion; today's behaviour stays.**")
        elif sig:
            out.append("- **Branch 2: primary significant, guardrails hold → ship PR 1.**")
        else:
            out.append("- **Branch 3: inconclusive on quality, guardrails hold → Richard's call; recommendation: ship on the code argument.**")

    out.append("\n## Cost\n")
    total = collections.defaultdict(float)
    calls = 0
    for u in usage_lines:
        for m, d in (u.get("dollars") or {}).items():
            total[m] += d
        calls += u.get("calls", 0)
    for m, d in total.items():
        out.append(f"- {m}: ${d:.2f}")
    out.append(f"- judge calls: {calls}")

    out.append("\n## Method notes\n")
    out.append("- Arms are pure functions of the engine's delivered 10-document list: A = partition (non-transcripts first) → `_dedupe_by_session` → cut; B = engine order → `_dedupe_by_session` → cut. Both ported verbatim from research-os.")
    out.append("- Not replayed: session self-exclusion (not recorded in a trace) and the workspace/project lenses (membership filters, identical across arms, absent on unscoped searches). They can change which document is cut at k, never the order.")
    out.append("- NDCG@k uses binary gains with the ideal computed over the same 10 documents; traces with zero useful documents are excluded from NDCG (undefined) and kept in P@k/MRR.")
    out.append("- Phase 0 rows use arm D (top-10 by Jev, offline, same scorer) as the engine list; live rows use the delivered order recorded in the trace (`gathered.chunks`), which equals `selection.ranked`.")
    Path(args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


# --------------------------------------------------------------------------
# calibration slice


def cmd_calibrate(args: argparse.Namespace) -> int:
    records = [json.loads(l) for l in open(args.arms)]
    rng = random.Random(args.seed)
    verdicts: dict[tuple[str, str], bool | None] = {}
    if args.verdicts and os.path.exists(args.verdicts):
        for line in open(args.verdicts):
            row = json.loads(line)
            if row["kind"] == "doc" and row["model"] == args.model and row["pass"] == 1:
                verdicts[(row["trace_id"], row["doc"])] = row["verdict"]
    if args.human:
        human = json.load(open(args.human))
        pairs = []
        for key, hv in human.items():
            tid, doc = key.split("|", 1)
            jv = verdicts.get((tid, doc))
            if jv is not None and isinstance(hv, bool):
                pairs.append((hv, jv))
        a, kp = kappa(pairs)
        print(f"human vs {args.model}: raw agreement {a:.3f}, kappa {kp:.3f}, n={len(pairs)} -> {'GATED (>=0.8)' if a >= 0.8 else 'UNGATED (<0.8)'}")
        return 0
    cands = [r for r in records if r["arms"]["A"]["5"] != r["arms"]["B"]["5"]]
    rng.shuffle(cands)
    slice_ = cands[: args.n]
    lines = ["# Hand-grading slice for the post-sort A/B judge", "",
             "For each query, mark each document `[x]` if a person who asked that query would consider it a useful result (it contains, or directly points to, part of the answer). Leave `[ ]` otherwise. Then run:",
             "", "```", "python scripts/jev_shadow/postsort_ab.py calibrate --arms <arms> --verdicts <verdicts> --human answers.json", "```",
             "", "`answers.json` maps `\"<trace_id>|<doc_id>\"` to true/false. The judge's own verdicts are NOT shown here.", ""]
    keys = []
    for i, r in enumerate(slice_, 1):
        a5, b5 = r["arms"]["A"]["5"], r["arms"]["B"]["5"]
        demoted = [d for d in b5 if d not in a5]
        promoted = [d for d in a5 if d not in b5]
        docs = (demoted[:1] + promoted[:1]) or r["engine_docs"][:2]
        lines.append(f"## {i}. `{r['trace_id']}` ({r['customer_id']})")
        lines.append(f"**Query:** {r['query']}")
        for d in docs:
            keys.append(f"{r['trace_id']}|{d}")
            lines.append(f"- [ ] `{d}` ({r['sources'][d]})")
            body = r["bodies_short"][d].replace("\n", " ")
            lines.append(f"  > {body[:SET_BODY_LIMIT]}")
        lines.append("")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out + ".keys.json").write_text(json.dumps(keys, indent=1))
    print(f"wrote {args.out} with {len(slice_)} queries, {len(keys)} documents")
    return 0


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parity")
    p.add_argument("--pairs", required=True)
    p.add_argument("--blobs", required=True)
    p.add_argument("--min-ok", type=int, default=30)
    p.set_defaults(fn=cmd_parity)

    p = sub.add_parser("build")
    p.add_argument("--live", required=True)
    p.add_argument("--phase0", default="")
    p.add_argument("--phase0-blobs", default="")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("judge")
    p.add_argument("--arms", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium")
    p.add_argument("--openai-model", default="")
    p.add_argument("--repeat-frac", type=float, default=0.3)
    p.add_argument("--no-sets", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("report")
    p.add_argument("--arms", required=True)
    p.add_argument("--verdicts", required=True)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("tiers")
    p.add_argument("--arms", required=True)
    p.add_argument("--verdicts", required=True)
    p.add_argument("--live", required=True)
    p.add_argument("--phase0", default="")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--penalties", default="0,0.02,0.05,0.08;0,0.03,0.08,0.12;0,0.05,0.10,0.20")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_tiers)

    p = sub.add_parser("calibrate")
    p.add_argument("--arms", required=True)
    p.add_argument("--verdicts", default="")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--human", default="")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="calibration-slice.md")
    p.set_defaults(fn=cmd_calibrate)

    args = ap.parse_args()
    return args.fn(args)



# --------------------------------------------------------------------------
# tiers: the engine-side follow-up, sized on the same labels
#
# Richard (2026-09-23): runs/projects/papers first; github commits and files
# demoted because they are high-volume; transcripts last. Implemented here as
# a per-tier PENALTY subtracted from Jev's probability, so a strong Jev
# judgment still wins (a 0.91 session beats a 0.55 commit) and the tier only
# decides near-ties. Arm C(penalties) = engine list re-sorted by
# (p - penalty[tier]) -> _dedupe_by_session -> cut. Evaluated against the
# verdicts the A/B already collected: no new judge calls.

TIER_OF_KIND = {
    # tier 0: the records Probe owns
    "run": 0, "trial": 0, "project": 0, "group": 0, "paper": 0, "team_note": 0, "experiment": 0,
    # tier 1: authored github records
    "gh_pull_request": 1, "gh_pr": 1, "gh_issue": 1, "gh_review": 1, "gh_release": 1,
    "gh_feature_rationale": 1, "gh_codeowners": 1, "gh_commit_comment": 1,
    # tier 2: high-volume: commits, files, code
    "gh_commit": 2, "file": 2, "code": 2,
    # tier 3: agent-session derivatives
    "transcript": 3, "digest": 3,
}
DEFAULT_TIER = 2


def kind_of(doc_id: str) -> str:
    parts = doc_id.split(":")
    src = parts[0]
    if src in TRANSCRIPT_SOURCES:
        return "transcript"
    if src == CUSTOM_INGEST and len(parts) >= 4:
        sk, tail = parts[2], parts[3]
        if sk == "session_digests":
            return "digest"
        if sk == "team_notes":
            return "team_note"
        if sk == "artifacts" or sk.startswith("workspace") or sk.startswith("shared"):
            return "file"
        if sk == "experiments":
            return tail
        return "custom_other"
    if src == "github":
        return "gh_" + (parts[2] if len(parts) > 2 else "unknown")
    if src == "code_graph":
        return "code"
    return src


def tier_of(doc_id: str) -> int:
    return TIER_OF_KIND.get(kind_of(doc_id), DEFAULT_TIER)


def arm_c(engine_docs: list[str], jev_p: dict[str, float], penalties: list[float], customer_id: str, k: int) -> list[str]:
    order = {d: i for i, d in enumerate(engine_docs)}

    def key(d: str) -> tuple[float, int]:
        p = jev_p.get(d)
        if p is None:
            p = 1.0 - 0.1 * order[d]  # no probability recorded: keep engine order spacing
        return (-(p - penalties[tier_of(d)]), order[d])

    return dedupe_by_session(sorted(engine_docs, key=key), customer_id)[:k]


def _jev_probabilities(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """trace_id -> {doc_id: jev_p} from live blobs (selection.ranked) and Phase 0 rows (doc_ranked)."""
    out: dict[str, dict[str, float]] = {}
    live_ids = {r["trace_id"] for r in records if r["origin"] == "live"}
    for tid, path in blob_index(args.live).items():
        if tid not in live_ids:
            continue
        blob = js.load_blob(path)
        out[tid] = {d: float(p) for d, p in ((blob.get("selection") or {}).get("ranked") or [])}
    if args.phase0:
        p0_ids = {r["trace_id"] for r in records if r["origin"] == "phase0"}
        for line in gzip.open(args.phase0, "rt"):
            row = json.loads(line)
            if row["trace_id"] in p0_ids:
                out[row["trace_id"]] = {d: float(p) for d, p in (row.get("doc_ranked") or [])}
    return out


def cmd_tiers(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    records = [json.loads(l) for l in open(args.arms)]
    by_id = {r["trace_id"]: r for r in records}
    jev_p = _jev_probabilities(args, records)
    verdicts: dict[tuple[str, str], bool] = {}
    for line in open(args.verdicts):
        row = json.loads(line)
        if row["kind"] == "doc" and row["model"] == args.model and row["pass"] == 1 and row["verdict"] is not None:
            verdicts[(row["trace_id"], row["doc"])] = row["verdict"]

    out: list[str] = []
    out.append(f"# Tier penalty sizing (arm C) on the A/B labels ({args.model})\n")
    # volume census: kinds in the engine's top-10 across the corpus, and in the live pools
    kinds_top10: collections.Counter = collections.Counter(kind_of(d) for r in records for d in r["engine_docs"])
    kinds_pool: collections.Counter = collections.Counter()
    n_pool_traces = 0
    for tid, path in blob_index(args.live).items():
        if tid not in by_id:
            continue
        blob = js.load_blob(path)
        n_pool_traces += 1
        for cid, hit in js.pool_chunks(blob.get("prefanout")).items():
            kinds_pool[kind_of(js.doc_of(cid, hit))] += 1
    out.append("## Volume: what kinds show up\n")
    out.append("| kind | tier | in engine top-10 (400 traces) | in live candidate pools (chunks, %d traces) | judged useful (Opus) |" % n_pool_traces)
    out.append("|---|---|---|---|---|")
    useful: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in records:
        for d in r["engine_docs"]:
            v = verdicts.get((r["trace_id"], d))
            if v is not None:
                useful[kind_of(d)][0] += int(v)
                useful[kind_of(d)][1] += 1
    for kind, n in kinds_top10.most_common():
        u = useful[kind]
        out.append(f"| {kind} | {tier_of('x') if False else TIER_OF_KIND.get(kind, DEFAULT_TIER)} | {n} | {kinds_pool.get(kind, 0)} | {100 * u[0] / max(1, u[1]):.0f}% ({u[1]}) |")

    penalty_sets = [[float(x) for x in s.split(",")] for s in args.penalties.split(";")]
    out.append("\n## Arms on the same labels (paired per trace; C = tier penalties applied to Jev's probability)\n")
    out.append("| arm | NDCG@10 | P@4 | P@5 | P@8 | vs B: NDCG diff, 95% CI, wins/losses/ties | traces where C ≠ B |")
    out.append("|---|---|---|---|---|---|---|")

    def metrics_for(orders: dict[str, list[str]]) -> tuple[dict[str, float], dict[str, list[float]]]:
        per: dict[str, list[float]] = collections.defaultdict(list)
        for tid, order in orders.items():
            rec = by_id[tid]
            rel = {d: int(verdicts[(tid, d)]) for d in rec["engine_docs"] if (tid, d) in verdicts}
            if len(rel) != len(rec["engine_docs"]):
                continue
            n10 = ndcg_at(order, rel, 10, rec["engine_docs"])
            per["NDCG@10"].append(n10 if n10 is not None else float("nan"))
            for k in (4, 5, 8):
                per[f"P@{k}"].append(prec_at(order, rel, k))
        means = {m: sum(x for x in v if x == x) / max(1, sum(1 for x in v if x == x)) for m, v in per.items()}
        return means, per

    arms: dict[str, dict[str, list[str]]] = {
        "A (today)": {r["trace_id"]: dedupe_by_session(partition(r["engine_docs"], r["sources"]), r["customer_id"]) for r in records},
        "B (engine order)": {r["trace_id"]: dedupe_by_session(list(r["engine_docs"]), r["customer_id"]) for r in records},
    }
    for pens in penalty_sets:
        label = "C " + "/".join(f"{p:g}" for p in pens)
        arms[label] = {r["trace_id"]: arm_c(r["engine_docs"], jev_p.get(r["trace_id"], {}), pens, r["customer_id"], 10) for r in records}
    b_means, b_per = metrics_for(arms["B (engine order)"])
    for label, orders in arms.items():
        means, per = metrics_for(orders)
        diffs = [c - b for c, b in zip(per["NDCG@10"], b_per["NDCG@10"]) if c == c and b == b]
        lo, hi = bootstrap_ci(diffs, rng)
        w, l, t, _, _ = sign_test(diffs)
        changed = sum(1 for tid in orders if orders[tid] != arms["B (engine order)"][tid])
        out.append(f"| {label} | {means['NDCG@10']:.3f} | {means['P@4']:.3f} | {means['P@5']:.3f} | {means['P@8']:.3f} | {sum(diffs) / max(1, len(diffs)):+.3f} [{lo:+.3f}, {hi:+.3f}] {w}/{l}/{t} | {changed}/{len(orders)} |")

    # what a mid penalty does to the rank-1 slot, by kind
    out.append("\n## Rank-1 kind under each arm\n")
    out.append("| arm | " + " | ".join(k for k, _ in kinds_top10.most_common(8)) + " |")
    out.append("|---|" + "---|" * min(8, len(kinds_top10)))
    for label, orders in arms.items():
        c = collections.Counter(kind_of(o[0]) for o in orders.values() if o)
        out.append(f"| {label} | " + " | ".join(str(c.get(k, 0)) for k, _ in kinds_top10.most_common(8)) + " |")
    out.append("\nJev probabilities found for %d of %d traces; a doc without one keeps its engine spacing." % (sum(1 for r in records if jev_p.get(r["trace_id"])), len(records)))
    Path(args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
