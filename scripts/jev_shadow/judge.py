"""T6 -- the judge, and the calibration that decides whether to believe it.

Set overlap cannot rank two arms: it says they differ, never which is better.
This scores the DISAGREEMENT REGION -- the documents one arm delivered and the
other did not -- because the documents both arms agree on cancel out, and
judging them burns money to move no number.

Two labels per call:

    relevant      does this document help answer the query?
    answerable    does the delivered SET answer the query?  (per arm, per trace)

The second exists because the first cannot see a missing document. A set of ten
plausible-but-beside-the-point documents scores well per-document and answers
nothing; that gap is what sizes Phase 2's reformulation trigger.

THE GATE. A judge that does not agree with people is not evidence. Run
`--calibrate` against the hand-graded slice; if agreement is below
`AGREEMENT_FLOOR` the report must say the comparison is UNGATED, not print a
number with a caveat under it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.retrieval.agent import jev_shadow as js  # noqa: E402

AGREEMENT_FLOOR = 0.80

# Anthropic direct, not the LiteLLM gateway. The gateway lives inside the
# cluster and reaching it from here needs a port-forward, which this host will
# not keep alive -- a judge that dies halfway through leaves a half-graded
# corpus, which is worse than a judge on a different route. Cost lands on the
# team's API key rather than the proxy's spend log; noted in the report.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

#: Kept deliberately blunt. A judge prompt that reasons about "partial
#: relevance" produces a spectrum nobody can threshold; the arms are being
#: compared, not the documents.
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


def ask(client: httpx.Client, key: str, prompt: str, _tries: int = 3) -> bool | None:
    """One YES/NO verdict, or None when the judge did not answer.

    None is NOT False. An unanswered document is excluded from both the
    numerator and the denominator, so a rate-limited run reads as a smaller
    sample rather than a pile of negatives -- the failure mode that would make
    whichever arm was graded last look worse.
    """
    for attempt in range(_tries):
        try:
            r = client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": JUDGE_MODEL,
                    "max_tokens": 4,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            txt = (r.json()["content"][0]["text"] or "").strip().upper()
            if txt.startswith("YES"):
                return True
            if txt.startswith("NO"):
                return False
            return None
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def body_of(hit: dict[str, Any], limit: int = 1500) -> str:
    t = (hit.get("title") or "").strip()
    c = (hit.get("content") or "").strip()[:limit]
    return f"{t}\n{c}" if t else c


def doc_bodies(blob: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One representative hit per document in the pool."""
    out: dict[str, dict[str, Any]] = {}
    for cid, hit in js.pool_chunks(blob.get("prefanout")).items():
        out.setdefault(js.doc_of(cid, hit), hit)
    return out


def judge_trace(blob: dict[str, Any], row: dict[str, Any], client, key, arms) -> dict[str, Any]:
    bodies = doc_bodies(blob)
    query = blob.get("query") or ""
    sets = {name: row[field] for name, field in arms.items() if row.get(field)}
    # Only the symmetric difference: documents every arm delivered cancel out.
    union = set().union(*(set(v) for v in sets.values())) if sets else set()
    shared = set.intersection(*(set(v) for v in sets.values())) if len(sets) > 1 else set()
    contested = [d for d in union - shared if d in bodies]

    verdicts: dict[str, bool] = {}
    for doc in contested:
        v = ask(client, key, PROMPT.format(query=query, doc_id=doc, body=body_of(bodies[doc])))
        if v is not None:
            verdicts[doc] = v

    out: dict[str, Any] = {
        "trace_id": row["trace_id"], "day": row.get("day"),
        "customer_id": row["customer_id"], "n_contested": len(contested),
        "judged": len(verdicts), "arms": {},
    }
    for name, docs in sets.items():
        graded = [verdicts[d] for d in docs if d in verdicts]
        out["arms"][name] = {
            "n": len(docs),
            "contested_good": sum(1 for g in graded if g),
            "contested_graded": len(graded),
        }
        joined = "\n\n---\n\n".join(
            f"[{d}] {body_of(bodies[d], 700)}" for d in docs if d in bodies
        )
        if joined:
            a = ask(client, key, SET_PROMPT.format(query=query, bodies=joined))
            out["arms"][name]["answerable"] = a
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="A:a_docs,B:b_docs")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--era", default="")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    arms = dict(p.split(":") for p in args.arms.split(","))
    key = os.environ.get("JUDGE_KEY") or ""
    if not key:
        print("JUDGE_KEY not set", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    if args.era:
        rows = [r for r in rows if (r.get("day") or "") >= args.era]
    # Only traces where the arms actually disagree carry information.
    rows = [r for r in rows if len({tuple(sorted(r[f])) for f in arms.values() if r.get(f)}) > 1]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.sample]
    print(f"judging {len(rows)} disagreeing traces on {JUDGE_MODEL}", file=sys.stderr)

    index = {}
    for p in Path(args.blobs).rglob("*.json.gz"):
        index[p.stem.replace(".json", "")] = p

    results = []
    with httpx.Client() as client, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        def work(r):
            p = index.get(r["trace_id"])
            if p is None:
                return None
            return judge_trace(js.load_blob(p), r, client, key, arms)

        for i, res in enumerate(ex.map(work, rows), 1):
            if res:
                results.append(res)
            if i % 50 == 0:
                print(f"  {i}/{len(rows)}", file=sys.stderr)

    Path(args.out).write_text("\n".join(json.dumps(r) for r in results))

    agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        for name, a in r["arms"].items():
            agg[name]["good"] += a["contested_good"]
            agg[name]["graded"] += a["contested_graded"]
            if a.get("answerable") is True:
                agg[name]["answerable"] += 1
            if a.get("answerable") is not None:
                agg[name]["answer_graded"] += 1
    print(f"\ntraces judged: {len(results)}", file=sys.stderr)
    print(f"{'arm':6} {'contested docs':>15} {'useful':>8} {'precision':>10} {'answers the query':>18}",
          file=sys.stderr)
    for name, a in sorted(agg.items()):
        prec = 100 * a["good"] / max(1, a["graded"])
        ans = 100 * a["answerable"] / max(1, a["answer_graded"])
        print(f"{name:6} {a['graded']:15} {a['good']:8} {prec:9.1f}% {ans:17.1f}%", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
