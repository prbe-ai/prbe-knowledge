"""Score a replay: adjudicated precision, agreement, drift, cost. Exits 1 if the
acceptance bar fails.

    python -m scripts.jev_automerge.score --set <dir> --results <dir>/results.jsonl \
        [--gptoss-results <file with "gptoss" answers>] [--min-verified 78] [--min-agreement 0.94]

PRECISION, NOT AGREEMENT. Each auto-merge a gate would make is checked against
hard identity evidence -- never against the other model:

  verified     same repo + PR/issue number, a shared UUID, a shared email or
               login (Person), the same repo name once owner/wiki prefix and
               -/_ are ignored, or a human-approved merge of the pair
  known false  different PR/issue numbers, different UUIDs (Documents), or
               different page / incident / commit / session ids
  needs human  anything else -- e.g. two people who share only a name

The Jev gate is scored exactly as production acts: p >= AUTO_MERGE_JEV_HIGH_AT
AND, for a Person, a shared identifier (`jev_judge.shared_identifier`).

Acceptance (the eng-review bar, 2026-09-23): 0 known-false and 0 needs-human
among Jev auto-merges, >= --min-verified verified, and (when gpt-oss answers
are given) >= --min-agreement same-entity agreement with gpt-oss. The verified
count moves with Jev's drift: four runs of byte-identical requests over the
2026-09-23 set gave 83, 82, 81 and 80 (all verified), hence a default of 78.
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics
import sys
from pathlib import Path

from engine.ingest.auto_merge.jev_judge import _repo_key, shared_identifier
from engine.shared.constants import AUTO_MERGE_JEV_HIGH_AT, AUTO_MERGE_JEV_SUGGEST_AT
from scripts.jev_automerge import kb

JEV_PRICE_PER_TOKEN = 0.042e-6
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_NUMBERED = re.compile(r"^(?:github:)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?::(?:pr|issue):|#)(\d+)$")
_REPO_SHAPED = re.compile(r"(?:wiki:repo:)?(?:[a-z0-9_.-]+/)?[a-z0-9_.-]+")
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
    if a == b or graph.same(a, b):
        return True
    shaped = all(_REPO_SHAPED.fullmatch(x.lower()) and "@" not in x for x in (a, b))
    return bool(shaped and len(_repo_key(a)) > 3 and _repo_key(a) == _repo_key(b))


def adjudicate(human: Clusters, label: str, a: str, pa: dict, b: str, pb: dict) -> tuple[str, str]:
    if human.same(a, b):
        return "verified", "human-approved merge"
    if label == "Person":
        why = shared_identifier(a, pa, b, pb)
        return ("verified", why) if why else ("needs_human", "person: name-level evidence only")
    al, bl = a.lower(), b.lower()
    ma, mb = _NUMBERED.match(a), _NUMBERED.match(b)
    if ma and mb:
        same = (ma.group(1).lower(), ma.group(2)) == (mb.group(1).lower(), mb.group(2))
        return ("verified", "same repo + number") if same else ("known_false", "different PR/issue")
    ua, ub = set(_UUID.findall(al)), set(_UUID.findall(bl))
    if ua and ub:
        return ("verified", "shared uuid") if ua & ub else ("known_false", "different uuids")
    why = shared_identifier(a, pa, b, pb)
    if why:
        return "verified", why
    if _REPO_SHAPED.fullmatch(al) and _REPO_SHAPED.fullmatch(bl) and len(_repo_key(a)) > 3:
        return ("verified", "same repo name") if _repo_key(a) == _repo_key(b) else ("known_false", "different names")
    for prefix in _DISTINCT_ID_PREFIXES:
        if al.startswith(prefix) and bl.startswith(prefix):
            return "known_false", f"different {prefix.rstrip(':')} ids"
    return "needs_human", "no hard evidence either way"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, required=True, help="build_set output dir")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--gptoss-results", type=Path, help="results file carrying 'gptoss' answers (default: --results)")
    ap.add_argument("--min-verified", type=int, default=78)
    ap.add_argument("--min-agreement", type=float, default=0.94)
    args = ap.parse_args()

    decisions = {d["did"]: d for d in kb.read_jsonl(args.set / "decisions.jsonl")}
    human = Clusters(kb.read_jsonl(args.set / "human_merges.jsonl"))
    graph = Clusters(kb.read_jsonl(args.set / "graph_merges.jsonl"))
    results = {r["did"]: r for r in kb.read_jsonl(args.results)}
    gpt_src = {r["did"]: r for r in kb.read_jsonl(args.gptoss_results)} if args.gptoss_results else results
    label = lambda d: {"Repo": "Document", "WikiPerson": "Person"}.get(d["label"], d["label"])  # noqa: E731

    def cand_props(d: dict, cid: str) -> dict:
        return next((c["properties"] or {} for c in d["candidates"] if c["canonical_id"] == cid), {})

    # ---- Jev auto-merges, gated exactly as production acts
    merges, guarded, bands = [], 0, collections.Counter()
    for did, r in results.items():
        d, j = decisions[did], (r["jev"] or [{}])[0]
        if j.get("error") or j.get("primary") is None:
            bands["unique"] += 1
            continue
        if j["p"] < AUTO_MERGE_JEV_SUGGEST_AT:
            bands["unique"] += 1
        elif j["p"] < AUTO_MERGE_JEV_HIGH_AT:
            bands["suggest"] += 1
        elif label(d) == "Person" and not shared_identifier(d["canonical_id"], d["properties"], j["primary"], cand_props(d, j["primary"])):
            bands["suggest (person guard)"] += 1
            guarded += 1
        else:
            bands["merge"] += 1
            merges.append((d, j))
    verdicts = [(d, j, *adjudicate(human, label(d), d["canonical_id"], d["properties"] or {}, j["primary"], cand_props(d, j["primary"])))
                for d, j in merges]
    tally = collections.Counter(v[2] for v in verdicts)
    print(f"Jev actions: {dict(bands)}")
    print(f"Jev auto-merges: {len(merges)}  verified {tally['verified']}  known false {tally['known_false']}  "
          f"needs human {tally['needs_human']}  (person guard downgraded {guarded})")
    for d, j, verdict, why in verdicts:
        if verdict != "verified":
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
    if tally["verified"] < args.min_verified:
        failures.append(f"only {tally['verified']} verified auto-merges (< {args.min_verified})")
    if agreement is not None and agreement < args.min_agreement:
        failures.append(f"agreement {agreement:.1%} < {args.min_agreement:.0%}")
    print("ACCEPTANCE:", "PASS" if not failures else "FAIL -- " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
