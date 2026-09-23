"""Jev (TypeSafe) as the entity auto-merge judge.

WHAT THIS REPLACES. `AutoMergeAnalyzer._judge` asked Cerebras gpt-oss-120b for an
`AutoMergeVerdict`. Here Jev answers ONE Choice question instead -- "which of
these candidates is the same real-world thing as the new entity, or none of
them?" -- and this module turns the answer into the same `AutoMergeVerdict`, so
everything in `analyze()` after the judge is unchanged.

    new entity + filtered candidates (<= 10)
        │
        ├─ build_request(): state {new_entity, candidates c0..cN}, long string
        │                   values trimmed (keys never dropped), one Choice
        │                   over {c0..cN, none_of_these}
        ├─ jev.post_choice(model=AUTO_MERGE_JEV_MODEL, breaker=MERGE_BREAKER)
        └─ verdict_from_answer():   p = probability of the chosen key
               none_of_these ─────────────► unique
               p >= AUTO_MERGE_JEV_HIGH_AT ► duplicate / high
               p >= AUTO_MERGE_JEV_SUGGEST_AT duplicate / medium
               otherwise ─────────────────► unique (writes nothing)

Jev cannot write text, so the rationale is a template over the signals the
analyzer already has (see `template_rationale`).

MEASURED, NOT ASSUMED: a replay of 477 managed-plane decisions (2026-09-23)
asked both models the same questions on identical inputs. Jev and gpt-oss picked
the same real-world entity 94.8% of the time (98.8% on everyday traffic), and at
>= 0.95 Jev's 84 auto-merges were 83 verified by hard identity evidence plus one
name-only Person pair -- the case the shared-identifier guard in `analyzer.py`
downgrades. Numbers: docs/jev-contract.md, "Entity auto-merge".

The request shape below IS what the replay measured. Rewording the
instructions or criteria is a model change: re-run `scripts/jev_automerge/`
before shipping a new phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from engine.ingest.auto_merge.models import AutoMergeVerdict
from engine.retrieval.agent.jev import MERGE_BREAKER, ChoiceAnswer, post_choice
from engine.shared.constants import (
    AUTO_MERGE_JEV_HIGH_AT,
    AUTO_MERGE_JEV_MAX_VALUE_CHARS,
    AUTO_MERGE_JEV_MODEL,
    AUTO_MERGE_JEV_SUGGEST_AT,
)

NONE_OF_THESE = "none_of_these"

#: The gpt-oss system prompt's rules, translated one-for-one into a Choice.
#: Nothing added that the production prompt does not already say.
_INSTRUCTIONS = (
    "Is state.new_entity the SAME real-world thing -- same person, same project, same "
    "topic, same channel, same artifact -- as one of state.candidates? Different surface "
    "text is fine as long as the underlying identity matches. Choose that candidate's key. "
    f"Choose `{NONE_OF_THESE}` when none of them is the same thing; that is the safe default when in "
    "doubt. Concrete shared evidence (exact email, exact username, exact ticket or PR id, or "
    "a name + role overlap with no contradicting properties) makes a match; surface-text or "
    "embedding similarity alone does not."
)

_RATIONALE_MAX_CHARS = 240  # AutoMergeVerdict.rationale / the suggestions column


class CandidateLike(Protocol):
    """`analyzer.Candidate`, without importing the analyzer (it imports us)."""

    canonical_id: str
    properties: dict
    degree: int
    trigram_score: float | None
    vector_distance: float | None


@dataclass(frozen=True, slots=True)
class Judgment:
    """A judge's verdict plus what the audit trail needs to name it honestly."""

    verdict: AutoMergeVerdict
    #: The model that actually answered (Jev's response `model`, or the LLM id).
    model: str
    #: Probability of the chosen candidate. None for judges that have none.
    p: float | None = None


# --------------------------------------------------------------------------
# the request


def trim_values(value: Any, limit: int = AUTO_MERGE_JEV_MAX_VALUE_CHARS) -> Any:
    """Trim long STRING values; keep every key and every list element.

    A blanket size cap could cut an identity field off while keeping a
    matching name. Trimming only long free text keeps ids, emails and logins
    intact -- they are never anywhere near `limit`.
    """
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {k: trim_values(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [trim_values(v, limit) for v in value]
    return value


def build_request(
    node: dict[str, Any], candidates: list[CandidateLike]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, CandidateLike]]:
    """(state, question, key -> candidate) for one judgment.

    Keys are `c0..cN` in the analyzer's ranking order, never canonical ids:
    an id can be long, contain any character, or look like instructions.
    """
    keys: dict[str, CandidateLike] = {}
    cands: dict[str, Any] = {}
    for i, c in enumerate(candidates):
        key = f"c{i}"
        keys[key] = c
        signals: dict[str, float] = {}
        if c.trigram_score is not None:
            signals["trigram"] = round(c.trigram_score, 2)
        if c.vector_distance is not None:
            signals["vector_distance"] = round(c.vector_distance, 3)
        cands[key] = {
            "canonical_id": c.canonical_id,
            "properties": trim_values(c.properties or {}),
            "degree": c.degree,
            "signals": signals,
        }
    state = {
        "new_entity": {
            "label": node["label"],
            "canonical_id": node["canonical_id"],
            "properties": trim_values(node.get("properties") or {}),
            "degree": node.get("degree", 0),
        },
        "candidates": cands,
    }
    criteria = {
        key: f"state.candidates.{key} (canonical_id {c.canonical_id}) is the same real-world thing as state.new_entity"
        for key, c in keys.items()
    }
    criteria[NONE_OF_THESE] = "none of the candidates is the same real-world thing as state.new_entity"
    question = {"type": "choice", "instructions": _INSTRUCTIONS, "criteria": criteria}
    return state, question, keys


# --------------------------------------------------------------------------
# the answer


def verdict_from_answer(
    answer: ChoiceAnswer, keys: dict[str, CandidateLike], node: dict[str, Any]
) -> tuple[AutoMergeVerdict, float]:
    """Map Jev's choice + probability onto the analyzer's verdict contract."""
    p = answer.probabilities[answer.choice]
    if answer.choice == NONE_OF_THESE:
        return (
            AutoMergeVerdict(
                verdict="unique",
                rationale=f"Jev: none of the {len(keys)} candidates is the same entity (p={p:.2f})",
            ),
            p,
        )
    cand = keys[answer.choice]
    if p < AUTO_MERGE_JEV_SUGGEST_AT:
        return (
            AutoMergeVerdict(
                verdict="unique",
                rationale=f"Jev's best pick {cand.canonical_id[:120]} is below the suggestion bar (p={p:.2f})",
            ),
            p,
        )
    return (
        AutoMergeVerdict(
            verdict="duplicate",
            primary_canonical_id=cand.canonical_id,
            confidence="high" if p >= AUTO_MERGE_JEV_HIGH_AT else "medium",
            rationale=template_rationale(node, cand, p),
        ),
        p,
    )


async def judge(
    node: dict[str, Any],
    candidates: list[CandidateLike],
    *,
    api_key: str,
    client: httpx.AsyncClient | None = None,
) -> Judgment:
    """One Jev call -> Judgment. Raises what `post_choice` raises."""
    state, question, keys = build_request(node, candidates)
    answer = await post_choice(
        state,
        question,
        api_key=api_key,
        model=AUTO_MERGE_JEV_MODEL,
        breaker=MERGE_BREAKER,
        client=client,
    )
    verdict, p = verdict_from_answer(answer, keys, node)
    return Judgment(verdict=verdict, model=answer.model, p=p)


# --------------------------------------------------------------------------
# identity evidence (also the analyzer's Person guard) and the rationale

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
#: `github:<owner>/<repo>:pr:<n>` or `<owner>/<repo>#<n>` (PRs and issues).
_NUMBERED = re.compile(r"^(?:github:)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?::(?:pr|issue):|#)(\d+)$")
_REPO_SHAPED = re.compile(r"(?:wiki:repo:)?(?:[a-z0-9_.-]+/)?[a-z0-9_.-]+")
_HEX = re.compile(r"\b[0-9a-f]{7,40}\b")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _repo_key(canonical_id: str) -> str:
    """`prbe-ai/prbe-agent-tap`, `wiki:repo:prbe_agent_tap`, `prbe-agent-tap` -> `prbeagenttap`."""
    c = canonical_id.lower()
    c = c.removeprefix("wiki:repo:")
    if re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", c):
        c = c.split("/", 1)[1]
    return _alnum(c)


def shared_identifier(
    a_id: str, a_props: dict[str, Any] | None, b_id: str, b_props: dict[str, Any] | None
) -> str | None:
    """A concrete identifier the two entities share, described; else None.

    Exact email or login (case-insensitive), or one side's email/login being
    the other side's canonical id (Person ids are often an email or a login).
    Names are deliberately NOT identifiers.
    """
    pa, pb = a_props or {}, b_props or {}
    for key, what in (("email", "email"), ("login", "handle")):
        va, vb = _text(pa.get(key)), _text(pb.get(key))
        if va and va.lower() == vb.lower():
            return f"shared {what} {va}"
    for side, other_id in ((pa, b_id), (pb, a_id)):
        for key in ("email", "login"):
            v = _text(side.get(key))
            if v and v.lower() == other_id.strip().lower():
                return f"{key} {v} is the other entity's id"
    return None


def _id_evidence(a: str, b: str) -> str | None:
    ua, ub = set(_UUID.findall(a.lower())), set(_UUID.findall(b.lower()))
    if ua & ub:
        return f"shared id {sorted(ua & ub)[0]}"
    na, nb = _NUMBERED.match(a), _NUMBERED.match(b)
    if na and nb and (na.group(1).lower(), na.group(2)) == (nb.group(1).lower(), nb.group(2)):
        return f"same repo {na.group(1)} and number {na.group(2)}"
    if _alnum(a) and _alnum(a) == _alnum(b):
        return f"ids equal ignoring punctuation ({a} ~ {b})"
    ka = _repo_key(a)
    if (
        len(ka) > 3
        and ka == _repo_key(b)
        and _REPO_SHAPED.fullmatch(a.lower())
        and _REPO_SHAPED.fullmatch(b.lower())
    ):
        return f"same repo name once owner/wiki prefix and -/_ are ignored ({a} ~ {b})"
    return None


def template_rationale(node: dict[str, Any], cand: CandidateLike, p: float | None) -> str:
    """The one strongest shared signal, then the numbers. <= 240 chars.

    Order: shared email/login > shared UUID > same repo + PR/issue number >
    same id ignoring punctuation > same repo name > same display name >
    shared hex token > "no shared identifier".
    """
    a, b = node["canonical_id"], cand.canonical_id
    pa, pb = node.get("properties") or {}, cand.properties or {}
    evidence = shared_identifier(a, pa, b, pb) or _id_evidence(a, b)
    if evidence is None:
        na = _text(pa.get("name")) or _text(pa.get("display_name"))
        nb = _text(pb.get("name")) or _text(pb.get("display_name"))
        if na and _alnum(na) == _alnum(nb):
            evidence = f'same name "{na}"'
    if evidence is None:
        common = set(_HEX.findall(a.lower())) & set(_HEX.findall(b.lower()))
        if common:
            evidence = f"shared id token {max(common, key=len)}"
    numbers = []
    if cand.trigram_score is not None:
        numbers.append(f"name/id trigram {cand.trigram_score:.2f}")
    if cand.vector_distance is not None:
        numbers.append(f"embedding similarity {1 - cand.vector_distance:.2f}")
    if p is not None:
        numbers.append(f"Jev p={p:.2f}")
    text = (evidence or "no shared identifier") + ("; " + ", ".join(numbers) if numbers else "")
    return text[:_RATIONALE_MAX_CHARS]
