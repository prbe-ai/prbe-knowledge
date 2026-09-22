"""Phase 0 replay driver.

    --arms ab     arm A (today) vs arm B (floor only). No LLM, no Jev key.
    --arms abc    adds arm C (Jev). Needs TYPESAFE_API_KEY.

Writes one JSON object per trace to `--out`, plus a summary to stderr. Every
skipped trace is recorded with its reason: a corpus that quietly shrinks is how
a replay starts flattering itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.retrieval.agent import jev_shadow as js  # noqa: E402


def iter_blobs(root: Path):
    for p in sorted(root.rglob("*.json.gz")):
        try:
            yield p, js.load_blob(p)
        except Exception as exc:  # a truncated object in R2 is not a reason to stop
            print(json.dumps({"skip": str(p), "reason": f"unreadable:{type(exc).__name__}"}),
                  file=sys.stderr)


def _score_batch(client, model, query, pool, cids, meta, depth=0):
    """Score one batch, halving and retrying on `max_tokens_exceeded`.

    The size estimate is calibrated on English prose (3.77 chars/token); a pool
    of config files and stack traces tokenizes denser and overruns a budget that
    looked safe. 67 of 3,048 traces did exactly that on the first run. Reacting
    to the server's own verdict is the only way to be right about a tokenizer we
    do not have -- guessing further under the cap just wastes pool on every
    normal request.
    """
    from typesafe_sdk import Noul

    if not cids:
        return {}
    chunks, questions = {}, {}
    for cid in cids:
        body = pool[cid].get("content") or ""
        limit = int(js.JEV_TOKEN_BUDGET * js.JEV_CHARS_PER_TOKEN * 0.9)
        if len(body) > limit:
            body = body[:limit]
            meta["truncated"] += 1
        chunks[cid] = body
        questions[cid] = Noul(
            instructions=(
                f"The passage at key {cid!r} in state.chunks answers, or "
                f"directly supports an answer to, state.query."
            )
        )
    try:
        resp = client.system_one(
            state={"query": query, "chunks": chunks},
            questions=questions,
            **({"model": model} if model else {}),
        )
    except Exception as exc:
        if "max_tokens_exceeded" in str(exc) and len(cids) > 1 and depth < 4:
            meta["splits"] += 1
            mid = len(cids) // 2
            out = _score_batch(client, model, query, pool, cids[:mid], meta, depth + 1)
            out.update(_score_batch(client, model, query, pool, cids[mid:], meta, depth + 1))
            return out
        meta["errors"].append(f"{type(exc).__name__}: {str(exc)[:160] or '<empty>'}")
        return {}
    meta["requests"] += 1
    usage = getattr(resp, "usage", None)
    if usage is not None:
        meta["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
    out = {}
    for cid, ans in (getattr(resp, "answers", {}) or {}).items():
        val = getattr(ans, "noul", None)
        if isinstance(val, (int, float)):
            out[cid] = float(val)
    return out


def score_with_jev(blob, client, model):
    """Score every chunk in one pool. Returns (chunk_id -> P, meta)."""
    pool = js.pool_chunks(blob.get("prefanout"))
    query = blob.get("query") or ""
    meta = {"requests": 0, "input_tokens": 0, "truncated": 0, "splits": 0, "errors": []}
    scores = {}
    for batch in js.batch_pool(pool, query=query):
        scores.update(_score_batch(client, model, query, pool, batch.chunk_ids, meta))
    meta["scored"] = len(scores)
    meta["pool"] = len(pool)
    return scores, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", choices=("ab", "abc"), default="ab")
    ap.add_argument("--thetas", default="0.5,0.7,0.9")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-spend-usd", type=float, default=25.0)
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(",")]
    client = model = None
    if args.arms == "abc":
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            print("TYPESAFE_API_KEY not set", file=sys.stderr)
            return 2
        client = TypeSafeClient(api_key=key, retry=RetryPolicy(max_retries=2), timeout=60.0)

    skips: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    todo = []
    for _p, blob in iter_blobs(Path(args.blobs)):
        ok, why = js.is_replayable(blob)
        if not ok:
            skips[why] += 1
            continue
        todo.append(blob)
        if args.limit and len(todo) >= args.limit:
            break

    t0 = time.perf_counter()
    spent = 0.0

    def work(blob: dict[str, Any]) -> dict[str, Any]:
        cmp = js.compare_a_b(blob)
        row = cmp.to_row()
        if client is not None:
            scores, meta = score_with_jev(blob, client, model)
            row["jev"] = {k: v for k, v in meta.items() if k != "errors"}
            row["jev_errors"] = meta["errors"][:3]
            ds = js.doc_scores(blob, scores)
            row["doc_score_hist"] = Counter(round(v, 1) for v in ds.values())
            # The ranking, kept. Scoring costs money and the rule to apply to
            # the scores is exactly what is in question -- persisting the top of
            # the ranking means a new rule is an offline re-read, not a re-run.
            ranked = sorted(ds.items(), key=lambda kv: -kv[1])[:25]
            row["doc_ranked"] = [[d, round(v, 4)] for d, v in ranked]
            # Arm D: top-N by score, NO threshold. Jev's absolute values are
            # compressed (58% of documents land at or below 0.1) and drift
            # +/-0.03 run to run, so a cutoff is both mostly-empty and unstable
            # near itself -- but the ORDER is the part a scorer is good at.
            row["d_docs"] = [d for d, _ in ranked[: js.DELIVERY_BUDGET_DOCS]]
            # How many documents sit within the measured +/-0.03 nondeterminism
            # band of each theta: a threshold there is a coin flip.
            for th in thetas:
                docs, n_jev = js.arm_jev(blob, scores, theta=th)
                a, b = set(row["a_docs"]), set(row["b_docs"])
                c = set(docs)
                row[f"c@{th}"] = {
                    "docs": docs, "n_jev": n_jev,
                    "c_only_vs_a": len(c - a), "a_only_vs_c": len(a - c),
                    "c_only_vs_b": len(c - b), "b_only_vs_c": len(b - c),
                    "unstable": sum(1 for v in ds.values() if abs(v - th) <= 0.03),
                }
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency if client else 1) as ex:
        for i, row in enumerate(ex.map(work, todo), 1):
            if "jev" in row:
                spent += row["jev"].get("input_tokens", 0) * 0.042 / 1e6
                if spent > args.max_spend_usd:
                    print(f"STOP: spend cap ${args.max_spend_usd} reached at trace {i}",
                          file=sys.stderr)
                    rows.append(row)
                    break
            rows.append(row)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}  ${spent:.2f}  {time.perf_counter()-t0:.0f}s",
                      file=sys.stderr)

    Path(args.out).write_text("\n".join(json.dumps(r, default=str) for r in rows))

    # ---- summary
    def block(label: str, rs: list[dict[str, Any]]) -> None:
        if not rs:
            print(f"\n{label}: no traces", file=sys.stderr)
            return
        n = len(rs)
        exact = sum(1 for r in rs if set(r["a_docs"]) == set(r["b_docs"]))
        tot_a = sum(len(r["a_docs"]) for r in rs)
        tot_shared = sum(r["shared"] for r in rs)
        nochange = sum(1 for r in rs if r["a_only"] == 0)
        print(f"\n{label}  (n={n})", file=sys.stderr)
        print(f"  A == B exactly:            {exact:5}/{n} = {100*exact/n:5.1f}%", file=sys.stderr)
        print(f"  A changed nothing vs B:    {nochange:5}/{n} = {100*nochange/n:5.1f}%", file=sys.stderr)
        print(f"  doc overlap A&B / A:       {100*tot_shared/max(1,tot_a):5.1f}%", file=sys.stderr)
        # Provenance is only recorded from 2026-09-12; withhold it elsewhere
        # rather than let an absent field read as "the model chose everything".
        kn = [r for r in rs if r.get("provenance_known")]
        if kn:
            m = sum(r["n_model_picked"] for r in kn)
            f = sum(r["n_appended"] for r in kn)
            z = sum(1 for r in kn if r["n_model_picked"] == 0)
            print(f"  delivered docs (n={len(kn)} with provenance): model {100*m/max(1,m+f):.1f}%  floor {100*f/max(1,m+f):.1f}%",
                  file=sys.stderr)
            print(f"  traces where the model picked ZERO: {z}/{len(kn)} = {100*z/len(kn):.1f}%",
                  file=sys.stderr)
        else:
            print("  provenance: NOT RECORDED in this era (harness_appended shipped 2026-09-12)",
                  file=sys.stderr)

    print(f"\ntraces compared: {len(rows)}   skipped: {dict(skips)}", file=sys.stderr)
    block("ALL", rows)
    # The pipeline changed on 2026-09-12 (PR #548 made the floor conditional
    # and retrained the gatherer's judgement). Pooling the eras would average
    # two different systems.
    block("ERA pre-2026-09-12", [r for r in rows if r.get("day", "") < "2026-09-12"])
    block("ERA 2026-09-12 onward", [r for r in rows if r.get("day", "") >= "2026-09-12"])
    by_tenant: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r["a_docs"]:
            by_tenant[r["customer_id"]].append(100 * r["shared"] // len(r["a_docs"]))
    print("\nper tenant (all eras):", file=sys.stderr)
    for t, xs in sorted(by_tenant.items(), key=lambda kv: -len(kv[1])):
        xs.sort()
        print(f"  {t:26} n={len(xs):5}  median overlap {xs[len(xs)//2]}%", file=sys.stderr)
    if client is not None:
        print(f"jev spend: ${spent:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
