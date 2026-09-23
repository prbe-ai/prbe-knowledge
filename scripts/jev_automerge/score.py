"""Score a replay: adjudicated precision, agreement, drift, cost. Exits 1 if the
acceptance bar fails.

    python -m scripts.jev_automerge.score --set <dir> --results <dir>/results.jsonl \
        [--gptoss-results <file with "gptoss" answers>] [--min-verified 12] [--min-agreement 0.94]

PRECISION, NOT AGREEMENT. Each auto-merge a gate would make is checked against
hard identity evidence -- never against the other model:

  verified     a human-approved merge of the pair; or (non-Person) the same
               repo + PR/issue number, the same leaf UUID, a shared
               email/login, the same repo name (same owner, or one side
               unowned), or the same id up to case and -/_
  by guard     a Person pair whose only confirmation is the shared email/login
               the execution gate itself required -- NOT independent evidence,
               so it is reported on its own line, never folded into "verified"
  known false  different PR/issue numbers, different leaf UUIDs,
               different repo owners, or different page/incident/commit/session ids
  needs human  anything else -- e.g. two people who share only a name

The Jev gate is scored exactly as production acts: a node whose id is one of
the tenant's documents is skipped before judging (`is_document`, recorded by
build_set); then p >= AUTO_MERGE_JEV_HIGH_AT, an answer from
AUTO_MERGE_JEV_MODEL, AND `jev_judge.execution_evidence` (the analyzer's gate).

"Verified" for a non-Person pair is mostly the SAME deterministic evidence the
gate requires, so for those it shows the judge and the gate agree -- it is
not independent ground truth. Independent checks are the human-approved
merges and the known-false rules; the day-one human review of live merges is
the rest.

Acceptance: 0 known-false and 0 needs-human among Jev auto-merges, >=
--min-verified verified (people by the gate counted in), and (when gpt-oss
answers are given) >= --min-agreement same-entity agreement with gpt-oss.
Before the document-node guard four identical runs gave 80-83 auto-merges;
with it the 2026-09-23 set leaves 14 (68 of the 82 were a document's own node
folded into its mention), hence a default of 12.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

from engine.ingest.auto_merge.jev_judge import (
    NONE_OF_THESE,
    NUMBERED_RE,
    REPO_SHAPED_RE,
    execution_evidence,
    fold_id,
    leaf_uuid,
    repo_parts,
    same_repo,
    shared_identifier,
)
from engine.shared.constants import (
    AUTO_MERGE_JEV_HIGH_AT,
    AUTO_MERGE_JEV_MODEL,
    AUTO_MERGE_JEV_SUGGEST_AT,
    NodeLabel,
)
from scripts.jev_automerge import kb

JEV_PRICE_PER_TOKEN = 0.042e-6
_DISTINCT_ID_PREFIXES = ("notion:page:", "pd:incident:", "slack:", "granola:meeting:", "claude_code:", "agent_session:")


class Clusters:
    """Union-find over (a, b) pairs: `same(x, y)` iff merged in the graph."""

    def __init__(self, pairs: list[dict]) -> None:
        self.parent: dict[str, str] = {}
        for p in pairs:
            self.parent[self.find(p["a"])] = self.find(p["b"])

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def same(self, a: str, b: str) -> bool:
        return a in self.parent and b in self.parent and self.find(a) == self.find(b)


def same_entity(graph: Clusters, a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a == b
    return a == b or graph.same(a, b) or same_repo(a, b)


def adjudicate(human: Clusters, label: str, a: str, pa: dict, b: str, pb: dict) -> tuple[str, str]:
    if human.same(a, b):
        return "verified", "human-approved merge"
    if label == NodeLabel.PERSON:
        why = shared_identifier(a, pa, b, pb)
        return ("by_guard", why) if why else ("needs_human", "person: name-level evidence only")
    al, bl = a.lower(), b.lower()
    ma, mb = NUMBERED_RE.match(a), NUMBERED_RE.match(b)
    if ma and mb:
        same = (ma.group(1).lower(), ma.group(2)) == (mb.group(1).lower(), mb.group(2))
        return ("verified", "same repo + number") if same else ("known_false", "different PR/issue")
    ua, ub = leaf_uuid(a), leaf_uuid(b)
    if ua and ub:
        return ("verified", "same leaf uuid") if ua == ub else ("known_false", "different leaf uuids")
    why = shared_identifier(a, pa, b, pb)
    if why:
        return "verified", why
    if fold_id(a) == fold_id(b):
        return "verified", "same id up to case and -/_"
    if REPO_SHAPED_RE.fullmatch(al) and REPO_SHAPED_RE.fullmatch(bl) and len(repo_parts(a)[1]) > 3:
        if same_repo(a, b):
            return "verified", "same repo name"
        return "known_false", "different repos"
    for prefix in _DISTINCT_ID_PREFIXES:
        if al.startswith(prefix) and bl.startswith(prefix):
            return "known_false", f"different {prefix.rstrip(':')} ids"
    return "needs_human", "no hard evidence either way"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, required=True, help="build_set output dir")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--gptoss-results", type=Path, help="results file carrying 'gptoss' answers (default: --results)")
    ap.add_argument("--min-verified", type=int, default=12)
    ap.add_argument("--min-agreement", type=float, default=0.94)
    args = ap.parse_args()

    decisions = {d["did"]: d for d in kb.read_jsonl(args.set / "decisions.jsonl")}
    human = Clusters(kb.read_jsonl(args.set / "human_merges.jsonl"))
    graph = Clusters(kb.read_jsonl(args.set / "graph_merges.jsonl"))
    results = {r["did"]: r for r in kb.read_jsonl(args.results)}
    gpt_src = {r["did"]: r for r in kb.read_jsonl(args.gptoss_results)} if args.gptoss_results else results

    def cand_props(d: dict, cid: str) -> dict:
        return next((c["properties"] or {} for c in d["candidates"] if c["canonical_id"] == cid), {})

    if not all("is_document" in d for d in decisions.values()):
        print("ERROR: this set predates is_document (the analyzer's document-node guard); "
              "rebuild it with build_set", file=sys.stderr)
        return 2

    def band(d: dict, j: dict) -> str:
        """What production does with one answer."""
        if d["is_document"] or "doc_type" in (d["properties"] or {}):
            return "skipped (document node)"
        if j.get("error") or j.get("primary") is None:
            return "unique"
        if j["p"] < AUTO_MERGE_JEV_SUGGEST_AT:
            # jev_judge.verdict_from_answer: a split pick still suggests.
            none = (j.get("probs") or {}).get(NONE_OF_THESE, 1.0 - j["p"])
            return "suggest (split)" if 1.0 - none >= AUTO_MERGE_JEV_SUGGEST_AT else "unique"
        if j["p"] < AUTO_MERGE_JEV_HIGH_AT:
            return "suggest"
        if j.get("model") != AUTO_MERGE_JEV_MODEL:
            return "suggest (uncalibrated model)"
        if not execution_evidence(
            d["qlabel"], d["canonical_id"], d["properties"], j["primary"], cand_props(d, j["primary"])
        ):
            return "suggest (no execution evidence)"
        return "merge"

    # ---- Jev auto-merges, gated exactly as production acts. Production asks
    # about a node on every upsert, so a pair ANY repetition would execute is
    # an auto-merge: adjudicate each distinct (decision, primary) once.
    merged: dict[tuple[str, str], tuple[dict, dict]] = {}
    bands = collections.Counter()
    for did, r in results.items():
        d = decisions[did]
        for i, j in enumerate(r["jev"] or [{}]):
            b = band(d, j)
            if i == 0:
                bands[b] += 1  # the action table reads the first repetition
            if b == "merge":
                merged.setdefault((did, j["primary"]), (d, j))
    merges = list(merged.values())
    guarded = bands["suggest (no execution evidence)"]
    verdicts = [
        (d, j, *adjudicate(human, d["qlabel"], d["canonical_id"], d["properties"] or {}, j["primary"],
                           cand_props(d, j["primary"])))
        for d, j in merges
    ]
    tally = collections.Counter(v[2] for v in verdicts)
    print(f"Jev actions: {dict(bands)}")
    print(f"Jev auto-merges (distinct pairs over all repetitions): {len(merges)}  verified {tally['verified']}  person-by-guard {tally['by_guard']}  "
          f"known false {tally['known_false']}  needs human {tally['needs_human']}  "
          f"(execution gate downgraded {guarded})")
    for d, j, verdict, why in verdicts:
        if verdict not in ("verified", "by_guard"):
            print(f"   [{verdict}: {why}] {d['did']}  p={j['p']:.2f}")

    # ---- agreement with gpt-oss on the same entity (identical inputs)
    agreement = None
    pairs = []
    for did, r in results.items():
        g = (gpt_src.get(did) or {}).get("gptoss")
        j = (r["jev"] or [{}])[0]
        if not g or g.get("error") or j.get("error"):
            continue
        g_dup = g.get("verdict") == "duplicate"
        j_dup = j.get("primary") is not None
        pairs.append(g_dup == j_dup and (not j_dup or same_entity(graph, j["primary"], g.get("primary"))))
    if pairs:
        agreement = sum(pairs) / len(pairs)
        print(f"same entity as gpt-oss: {sum(pairs)}/{len(pairs)} = {agreement:.1%}")

    # ---- drift, latency, cost, failures
    calls = [c for r in results.values() for c in r["jev"]]
    ok = [c for c in calls if not c.get("error")]
    errors = collections.Counter(c["error"].split(":")[0] for c in calls if c.get("error"))
    def merges_at_high(c: dict) -> bool:
        return c.get("primary") is not None and c["p"] >= AUTO_MERGE_JEV_HIGH_AT

    flips = sum(1 for r in results.values() if len(r["jev"]) > 1 and not any(c.get("error") for c in r["jev"])
                and merges_at_high(r["jev"][0]) != merges_at_high(r["jev"][1]))
    if ok:
        ms = sorted(c["ms"] for c in ok)
        toks = [c["input_tokens"] for c in ok if c.get("input_tokens")]
        print(f"Jev calls: {len(ok)} ok, {sum(errors.values())} failed {dict(errors)}; "
              f"p50 {ms[len(ms) // 2]:.0f} ms, p90 {ms[int(len(ms) * 0.9)]:.0f} ms; "
              f"input tokens p50 {statistics.median(toks):.0f} max {max(toks)}; "
              f"${statistics.mean(toks) * JEV_PRICE_PER_TOKEN:.6f}/call; merge-line flips between repeats {flips}")

    failures = []
    if tally["known_false"] or tally["needs_human"]:
        failures.append("Jev auto-merges that are not verified")
    if tally["verified"] + tally["by_guard"] < args.min_verified:
        failures.append(f"only {tally['verified'] + tally['by_guard']} verified auto-merges (< {args.min_verified})")
    if agreement is not None and agreement < args.min_agreement:
        failures.append(f"agreement {agreement:.1%} < {args.min_agreement:.0%}")
    print("ACCEPTANCE:", "PASS" if not failures else "FAIL -- " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
