"""Build a frozen auto-merge decision set from a kb database (read-only).

    python -m scripts.jev_automerge.build_set --customer <id> --out <dir> \
        [--logged <done-lines.jsonl>] [--pr-pairs 60] [--unjudged 170]

Decisions (one per new entity the analyzer judged or would judge):

  stored_auto  entity_merge_audit rows written by auto-merge ("auto:" reason)
  stored_sugg  entity_merge_suggestions rows
  logged       `post_write_worker.done` log lines (optional file; the ONLY
               record of "unique" verdicts -- the analyzer never stores them)
  unjudged     recently updated judgeable nodes with no merge/suggestion record

Each decision's candidates are rebuilt with the analyzer's own logic against
TODAY's graph: its trigram SQL (verbatim, as the app role under RLS), the same
exact cosine kNN (computed here from embeddings pulled once with halfvec_send,
because the production query is a ~20s-per-node seq scan on a small primary),
the same ranking and cap, the same stable-key filter. When the graph has moved
on (relabels, merges, deletions) and a stored primary is no longer among the
rebuilt candidates, it is put back (`mode=rebuilt+injected`, or `pair_only`).

Outputs `<out>/decisions.jsonl`, `<out>/human_merges.jsonl` (what people
approved -- adjudication evidence) and `<out>/graph_merges.jsonl` (every live
alias and merge -- "is this the same entity" for agreement scoring). The output
holds tenant data: keep it OUT of this repository.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from engine.ingest.auto_merge.analyzer import (
    TOTAL_CANDIDATE_CAP,
    TRIGRAM_CANDIDATES_SQL,
    TRIGRAM_FLOOR,
    TRIGRAM_TOP_K,
    VECTOR_TOP_K,
    Candidate,
    _is_path_canonical,
    _properties_conflict,
    rank_candidates,
)
from scripts.jev_automerge import kb

# Labels retired since older decisions: Repo collapsed into Document (migration
# 0091), WikiPerson rows now live as Person. Candidate queries use the new label.
QLABEL = {"Repo": "Document", "WikiPerson": "Person"}
_PR_FMT = re.compile(r"^github:([^:]+):(pr|issue):([0-9]+)$")


def _pr_pair(new_id: str, primary: str) -> bool:
    m = _PR_FMT.match(new_id)
    return bool(m) and primary == f"{m.group(1)}#{m.group(3)}"


# The analyzer's own trigram query text, with its $n parameters inlined as
# escaped literals and each result wrapped as one JSON line.
_TRIGRAM_SQL = (
    "SELECT json_build_object('k', {k}, 'rows', COALESCE((SELECT json_agg(row_to_json(t)) FROM ("
    + TRIGRAM_CANDIDATES_SQL.replace("$1", "{label}").replace("$2", "{cid}").replace("$3", "{name}")
    .replace("$4", "{self_id}").replace("$5", "{floor}").replace("$6", "{topk}")
    + ") t), '[]'::json));"
)


def _decision(did, stratum, customer, label, cid, cur, props, degree, hist, **extra) -> dict:
    return {
        "did": did, "stratum": stratum, "customer_id": customer, "label": label,
        "qlabel": QLABEL.get(label, label), "canonical_id": cid,
        "node_id": cur["node_id"] if cur else None, "has_emb": bool(cur and cur["has_emb"]),
        "properties": props, "degree": degree, "hist": hist, **extra,
    }


def collect(
    customer: str, logged: Path | None, pr_pairs: int, unjudged: int, seed: int
) -> tuple[list[dict], list[dict], list[dict], set]:
    rng = random.Random(seed)
    c = kb.lit(customer)
    alias_rows = kb.rows(
        "select row_to_json(t) from (select label, alias_canonical_id, primary_canonical_id "
        f"from entity_aliases where customer_id = {c}) t"
    )
    aliases = {(a["label"], a["alias_canonical_id"]) for a in alias_rows}
    nodes = kb.rows(
        "select row_to_json(t) from (select node_id, label, canonical_id, properties->>'doc_type' doc_type, "
        "degree, updated_at, (embedding is not null) has_emb from graph_nodes "
        f"where customer_id = {c} and label <> 'CodeSymbol') t"
    )
    by_key = {(n["label"], n["canonical_id"]): n for n in nodes}
    by_id = {n["node_id"]: n for n in nodes}

    def lookup(label, cid):
        return by_key.get((QLABEL.get(label, label), cid))

    audit = kb.rows(
        "select row_to_json(t) from (select a.merge_id::text id, a.label, a.primary_canonical_id primary_cid, "
        "a.merged_alias_canonical_ids[1] new_cid, a.performed_at ts, a.reason, a.status, "
        "s.properties snap_props, s.degree snap_degree from entity_merge_audit a "
        "left join entity_merge_node_snapshot s on s.merge_id = a.merge_id and s.label = a.label "
        f"and s.canonical_id = a.merged_alias_canonical_ids[1] where a.customer_id = {c}) t"
    )
    sugg = kb.rows(
        "select row_to_json(t) from (select x.suggestion_id::text id, x.label, x.primary_canonical_id primary_cid, "
        "x.candidate_canonical_id new_cid, x.created_at ts, x.confidence, x.rationale, x.status, "
        "(select row_to_json(s) from entity_merge_node_snapshot s where s.customer_id = x.customer_id "
        "and s.label = x.label and s.canonical_id = x.candidate_canonical_id order by s.created_at desc limit 1) snap "
        f"from entity_merge_suggestions x where x.customer_id = {c}) t"
    )
    human = [
        {"a": r["new_cid"], "b": r["primary_cid"], "why": "human merge"}
        for r in audit
        if r["status"] == "active" and not (r["reason"] or "").startswith("auto:")
    ] + [{"a": s["new_cid"], "b": s["primary_cid"], "why": "applied suggestion"} for s in sugg if s["status"] == "applied"]
    graph = human + [
        {"a": r["new_cid"], "b": r["primary_cid"], "why": "auto merge"}
        for r in audit
        if r["status"] == "active" and (r["reason"] or "").startswith("auto:")
    ]

    decisions: list[dict] = []
    touched: set = set()
    auto = [r for r in audit if (r["reason"] or "").startswith("auto:")]
    pr = [r for r in auto if _pr_pair(r["new_cid"], r["primary_cid"])]
    chosen = [r for r in auto if not _pr_pair(r["new_cid"], r["primary_cid"])] + rng.sample(pr, min(pr_pairs, len(pr)))
    for r in auto + sugg:
        touched.update({(r["label"], r["new_cid"]), (r["label"], r["primary_cid"])})
    for r in chosen:
        cur = lookup(r["label"], r["new_cid"])
        m = re.search(r"rationale=(.*)$", r["reason"] or "", re.S)
        decisions.append(_decision(
            f"audit:{r['id']}", "stored_auto", customer, r["label"], r["new_cid"], cur, r["snap_props"],
            r["snap_degree"] if r["snap_degree"] is not None else (cur["degree"] if cur else 0),
            {"verdict": "duplicate", "primary": r["primary_cid"], "confidence": "high",
             "rationale": (m.group(1) if m else "").strip(), "ts": r["ts"], "human": None},
            pr_format_pair=_pr_pair(r["new_cid"], r["primary_cid"]),
        ))
    for r in sugg:
        cur = lookup(r["label"], r["new_cid"])
        snap = r.get("snap") or {}
        decisions.append(_decision(
            f"sugg:{r['id']}", "stored_sugg", customer, r["label"], r["new_cid"], cur,
            None if cur else snap.get("properties"), cur["degree"] if cur else snap.get("degree", 0),
            {"verdict": "duplicate", "primary": r["primary_cid"], "confidence": r["confidence"],
             "rationale": r["rationale"], "ts": r["ts"], "human": r["status"]},
            pr_format_pair=_pr_pair(r["new_cid"], r["primary_cid"]),
        ))
    if logged:
        latest: dict[int, dict] = {}
        for r in kb.read_jsonl(logged):
            keep = r.get("customer") == customer and r.get("action") != "skipped" and r["node_id"] in by_id
            if keep and (r["node_id"] not in latest or r["timestamp"] > latest[r["node_id"]]["timestamp"]):
                latest[r["node_id"]] = r
        for nid, r in latest.items():
            n = by_id[nid]
            touched.add((n["label"], n["canonical_id"]))
            dup = r["action"] in ("merged", "suggested", "error") and r.get("primary")
            decisions.append(_decision(
                f"log:{nid}", "logged", customer, n["label"], n["canonical_id"], n, None, n["degree"],
                {"verdict": "duplicate" if dup else "unique", "primary": r.get("primary") if dup else None,
                 "confidence": r.get("confidence") if dup else None, "rationale": None,
                 "ts": r["timestamp"], "human": None},
            ))
    pool = sorted(
        (n for n in nodes
         if (n["label"], n["canonical_id"]) not in touched and (n["label"], n["canonical_id"]) not in aliases
         and not _is_path_canonical(n["label"], n["canonical_id"])),
        key=lambda n: n["updated_at"], reverse=True,
    )[: unjudged * 4]
    for n in rng.sample(pool, min(unjudged, len(pool))):
        decisions.append(_decision(
            f"u:{n['node_id']}", "unjudged", customer, n["label"], n["canonical_id"], n, None, n["degree"],
            {"verdict": "unique?", "primary": None, "confidence": None, "rationale": None,
             "ts": n["updated_at"], "human": None},
        ))
    graph += [{"a": a["alias_canonical_id"], "b": a["primary_canonical_id"], "why": "alias"} for a in alias_rows]
    return decisions, human, graph, aliases


def fill_properties(decisions: list[dict]) -> None:
    need = sorted({d["node_id"] for d in decisions if d["node_id"] and d["properties"] is None})
    props: dict[int, dict] = {}
    for i in range(0, len(need), 200):
        ids = ",".join(str(x) for x in need[i : i + 200])
        for r in kb.rows(f"select row_to_json(t) from (select node_id, properties, degree from graph_nodes where node_id in ({ids})) t"):
            props[r["node_id"]] = r
    for d in decisions:
        if d["properties"] is None:
            got = props.get(d["node_id"]) or {}
            d["properties"] = got.get("properties") or {}
            d["degree"] = got.get("degree", d["degree"])


def trigram_leg(decisions: list[dict], customer: str, cache: Path) -> dict[str, list]:
    """The analyzer's trigram query per decision. ~0.5 s of CPU each on the kb
    primary, so batches pause between them and results are cached in `cache`
    (a re-run reuses it instead of loading production again)."""
    out: dict[str, list] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [d for d in decisions if d["did"] not in out]
    for i in range(0, len(todo), 40):
        chunk = todo[i : i + 40]
        parts = [kb.as_tenant(customer)]
        for d in chunk:
            name = (d["properties"] or {}).get("name", "")
            name = name if isinstance(name, str) else ""
            parts.append(_TRIGRAM_SQL.format(
                k=kb.lit(d["did"]), cid=kb.lit(d["canonical_id"]), name=kb.lit(name), label=kb.lit(d["qlabel"]),
                self_id=d["node_id"] or -1, floor=TRIGRAM_FLOOR, topk=TRIGRAM_TOP_K))
        for r in kb.rows("\n".join(parts)):
            out[r["k"]] = r["rows"] or []
        cache.write_text(json.dumps(out))
        print(f"trigram {i + len(chunk)}/{len(todo)}", file=sys.stderr)
        time.sleep(1.0)
    return out


def vector_leg(decisions: list[dict], customer: str, aliases: set) -> dict[str, list]:
    """Exact cosine top-k per decision, same filters as the analyzer's SQL."""
    out: dict[str, list] = {}
    for label in sorted({d["qlabel"] for d in decisions if d["has_emb"] and d["node_id"]}):
        ids, cids, vecs = [], [], []
        after = 0
        while True:  # keyset pages: one statement per 5,000 rows stays well inside the timeout
            sql = kb.as_tenant(customer) + (
                "COPY (SELECT node_id, canonical_id, translate(encode(halfvec_send(embedding), 'base64'), E'\\n', '') "
                f"FROM graph_nodes WHERE label = {kb.lit(label)} AND embedding IS NOT NULL AND node_id > {after} "
                "ORDER BY node_id LIMIT 5000) TO STDOUT"
            )
            page = 0
            for line in kb.run(sql).splitlines():
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                raw = base64.b64decode(parts[2])
                dim = int.from_bytes(raw[:2], "big")
                ids.append(int(parts[0]))
                cids.append(parts[1])
                vecs.append(np.frombuffer(raw[4 : 4 + 2 * dim], dtype=">f2").astype(np.float32))
                after = int(parts[0])
                page += 1
            if page < 5000:
                break
        if not vecs:
            continue
        m = np.stack(vecs)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        index = {nid: i for i, nid in enumerate(ids)}
        allowed = np.array([(label, cid) not in aliases for cid in cids])
        for d in decisions:
            if d["qlabel"] != label or not d["has_emb"] or d["node_id"] not in index:
                continue
            s = index[d["node_id"]]
            dist = 1.0 - m @ m[s]
            mask = allowed.copy()
            mask[s] = False
            masked = np.where(mask, dist, np.inf)
            top = np.argsort(masked, kind="stable")[:VECTOR_TOP_K]
            out[d["did"]] = [{"node_id": ids[i], "canonical_id": cids[i], "distance": float(dist[i])}
                             for i in top if np.isfinite(masked[i])]
        print(f"vector {label}: {len(ids)} embeddings", file=sys.stderr)
    return out


def assemble(decisions: list[dict], trgm: dict, vec: dict, customer: str) -> None:
    # Properties for every candidate, from ANY decision's trigram rows or a
    # fetch -- a node can be a trigram hit for one decision and vector-only
    # for another, and must carry its real properties in both.
    known: dict[int, dict] = {r["node_id"]: r for rows in trgm.values() for r in rows}
    need = sorted({r["node_id"] for rows in vec.values() for r in rows} - set(known))
    extra: dict[int, dict] = {}
    for i in range(0, len(need), 300):
        ids = ",".join(str(x) for x in need[i : i + 300])
        for r in kb.rows(f"select row_to_json(t) from (select node_id, properties, degree from graph_nodes where node_id in ({ids})) t"):
            extra[r["node_id"]] = r
    for d in decisions:
        merged: dict[str, Candidate] = {}
        for r in trgm.get(d["did"], []):
            p = r["properties"] if isinstance(r["properties"], dict) else json.loads(r["properties"] or "{}")
            merged[r["canonical_id"]] = Candidate(r["canonical_id"], p, r["degree"], float(r["trigram_score"]), None)
        for r in vec.get(d["did"], []):
            if r["canonical_id"] in merged:
                merged[r["canonical_id"]].vector_distance = r["distance"]
                continue
            e = known.get(r["node_id"]) or extra.get(r["node_id"]) or {}
            p = e.get("properties")
            p = p if isinstance(p, dict) else json.loads(p or "{}")
            merged[r["canonical_id"]] = Candidate(r["canonical_id"], p, e.get("degree", 0), None, r["distance"])
        props = d["properties"] if isinstance(d["properties"], dict) else {}
        d["candidates"] = [c.__dict__ for c in rank_candidates(merged) if not _properties_conflict(props, c.properties)]
        d["mode"] = "rebuilt"
    # Put back stored primaries the rebuilt set no longer contains.
    missing = [d for d in decisions if d["hist"].get("primary") and d["hist"]["primary"] not in {c["canonical_id"] for c in d["candidates"]}]
    info: dict[str, dict] = {}
    for i in range(0, len(missing), 60):
        stmts = []
        for d in missing[i : i + 60]:
            p, k = d["hist"]["primary"], d["did"]
            stmts.append(
                f"select json_build_object('k', {kb.lit(k)}, "
                f"'node', (select row_to_json(t) from (select properties, degree from graph_nodes where customer_id = {kb.lit(customer)} "
                f"and canonical_id = {kb.lit(p)} order by (label = 'Document') desc limit 1) t), "
                f"'snap', (select row_to_json(t) from (select properties, degree from entity_merge_node_snapshot where customer_id = {kb.lit(customer)} "
                f"and canonical_id = {kb.lit(p)} order by created_at desc limit 1) t), "
                f"'trgm', similarity(lower({kb.lit(p)}), lower({kb.lit(d['canonical_id'])})));"
            )
        for r in kb.rows("\n".join(stmts)):
            info[r["k"]] = r
    for d in missing:
        r = info.get(d["did"], {})
        src = r.get("node") or r.get("snap") or {}
        cand = {"canonical_id": d["hist"]["primary"], "properties": src.get("properties") or {}, "degree": src.get("degree") or 0,
                "trigram_score": float(r["trgm"]) if r.get("trgm") is not None else None, "vector_distance": None}
        if d["candidates"]:
            d["mode"] = "rebuilt+injected"
            d["candidates"] = [*d["candidates"][: TOTAL_CANDIDATE_CAP - 1], cand]
        else:
            d["mode"] = "pair_only"
            d["candidates"] = [cand]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--customer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--logged", type=Path, help="post_write_worker.done lines, one JSON object per line")
    ap.add_argument("--pr-pairs", type=int, default=60, help="sampled github:<repo>:pr:<n> -> <repo>#<n> auto-merges")
    ap.add_argument("--unjudged", type=int, default=170)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-decisions", type=int, help="random subset, for a cheap smoke run of the harness")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    decisions, human, graph, aliases = collect(args.customer, args.logged, args.pr_pairs, args.unjudged, args.seed)
    if args.max_decisions:
        decisions = random.Random(args.seed).sample(decisions, min(args.max_decisions, len(decisions)))
    fill_properties(decisions)
    trgm = trigram_leg(decisions, args.customer, args.out / "trigram_cache.json")
    assemble(decisions, trgm, vector_leg(decisions, args.customer, aliases), args.customer)
    decisions = [d for d in decisions if d["candidates"]]
    kb.write_jsonl(args.out / "decisions.jsonl", decisions)
    kb.write_jsonl(args.out / "human_merges.jsonl", human)
    kb.write_jsonl(args.out / "graph_merges.jsonl", graph)
    print(json.dumps({"decisions": len(decisions),
                      "by_stratum": collections.Counter(d["stratum"] for d in decisions),
                      "by_mode": collections.Counter(d["mode"] for d in decisions)}), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        # First line only: psql's LINE/context lines echo the failing SQL,
        # which carries tenant ids and names into the terminal.
        first = next((ln for ln in (exc.stderr or "").splitlines() if ln.strip()), "no stderr")
        raise SystemExit(f"KB_PSQL failed: {first[:200]}") from exc
