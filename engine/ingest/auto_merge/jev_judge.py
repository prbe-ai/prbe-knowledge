"""Jev (TypeSafe) as the entity auto-merge judge.

WHAT THIS REPLACES. `AutoMergeAnalyzer._judge` asked Cerebras gpt-oss-120b for an
`AutoMergeVerdict`. Here Jev answers ONE Choice question instead -- "which of
these candidates is the same real-world thing as the new entity, or none of
them?" -- and this module turns the answer into the same `AutoMergeVerdict`
contract, so the analyzer acts on either judge the same way. (After the judge,
`analyze()` also applies the execution-evidence gate below: an auto-merge needs
a deterministic identifier the pair shares, whichever judge proposed it.)

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
name-only Person pair -- the case `execution_evidence` (the analyzer's gate)
downgrades. Numbers: docs/jev-contract.md, "Entity auto-merge".

The request shape below IS what the replay measured. Rewording the
instructions or criteria is a model change: re-run `scripts/jev_automerge/`
before shipping a new phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from engine.ingest.auto_merge.models import RATIONALE_MAX_CHARS, AutoMergeVerdict
from engine.retrieval.agent.jev import MERGE_BREAKER, ChoiceAnswer, post_choice
from engine.shared.constants import (
    AUTO_MERGE_JEV_HIGH_AT,
    AUTO_MERGE_JEV_MAX_VALUE_CHARS,
    AUTO_MERGE_JEV_MODEL,
    AUTO_MERGE_JEV_SUGGEST_AT,
    NodeLabel,
)

if TYPE_CHECKING:  # the analyzer imports this module; the type is enough here
    from engine.ingest.auto_merge.analyzer import Candidate

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

#: A candidate id longer than this cannot be a verdict's primary
#: (AutoMergeVerdict.primary_canonical_id max_length), so it is never offered.
MAX_PRIMARY_ID_CHARS = 512


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


#: A list property keeps at most this many elements (a marker notes the rest),
#: so one huge array cannot push every judgment it appears in over the cap.
MAX_LIST_ITEMS = 50
#: A map keeps at most this many keys (identity keys always first), and
#: nesting stops at MAX_DEPTH: custom-ingest properties have no size limit,
#: and one tenant's huge map must not push its judgments over the cap.
MAX_KEYS = 100
MAX_DEPTH = 6
#: Kept ahead of every other key when a map is cut.
_IDENTITY_KEYS = ("email", "login", "name", "display_name", "source_system", "doc_type", "kind")


def trim_values(value: Any, limit: int = AUTO_MERGE_JEV_MAX_VALUE_CHARS, _depth: int = 0) -> Any:
    """Trim long strings, long lists, big maps and deep nesting.

    A blanket size cap could cut an identity field off while keeping a
    matching name. So strings are cut only past `limit` (ids, emails and
    logins are never near it), and a map that is cut keeps its identity keys
    first. A marker records what was cut.
    """
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict | list) and _depth >= MAX_DEPTH:
        return "… nested too deep"
    if isinstance(value, dict):
        keys = list(value)
        if len(keys) > MAX_KEYS:
            identity = [k for k in _IDENTITY_KEYS if k in value]
            keys = identity + [k for k in keys if k not in identity][: MAX_KEYS - len(identity)]
        kept = {k: trim_values(value[k], limit, _depth + 1) for k in keys}
        if len(value) > len(keys):
            kept["…"] = f"{len(value) - len(keys)} more keys"
        return kept
    if isinstance(value, list):
        kept_list = [trim_values(v, limit, _depth + 1) for v in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            kept_list.append(f"… {len(value) - MAX_LIST_ITEMS} more")
        return kept_list
    return value


def build_request(
    node: dict[str, Any], candidates: list[Candidate]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Candidate]]:
    """(state, question, key -> candidate) for one judgment.

    Keys are `c0..cN` in the analyzer's ranking order, never canonical ids:
    an id can be long, contain any character, or look like instructions.
    """
    keys: dict[str, Candidate] = {}
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
    answer: ChoiceAnswer, keys: dict[str, Candidate], node: dict[str, Any]
) -> Judgment:
    """Map Jev's choice + probability onto the analyzer's verdict contract."""
    p = answer.probabilities[answer.choice]
    # When the graph holds one entity several ways, Jev's mass splits across
    # the copies (c0 0.48, c1 0.47): no single pick clears the bar although it
    # is sure the node is a duplicate. Then suggest the most likely copy.
    duplicate_mass = 1.0 - answer.probabilities.get(NONE_OF_THESE, 0.0)
    best = max((k for k in keys), key=lambda k: answer.probabilities.get(k, 0.0), default=None)
    if best is not None and p < AUTO_MERGE_JEV_SUGGEST_AT and duplicate_mass >= AUTO_MERGE_JEV_SUGGEST_AT:
        cand = keys[best]
        best_p = answer.probabilities[best]
        verdict = AutoMergeVerdict(
            verdict="duplicate",
            primary_canonical_id=cand.canonical_id,
            confidence="medium",
            rationale=template_rationale(node, cand, best_p),
        )
        return Judgment(verdict=verdict, model=answer.model, p=best_p)
    if answer.choice == NONE_OF_THESE:
        verdict = AutoMergeVerdict(
            verdict="unique",
            rationale=f"Jev: none of the {len(keys)} candidates is the same entity (p={p:.2f})",
        )
    elif p < AUTO_MERGE_JEV_SUGGEST_AT:
        verdict = AutoMergeVerdict(
            verdict="unique",
            rationale=(
                f"Jev's best pick {keys[answer.choice].canonical_id[:120]} "
                f"is below the suggestion bar (p={p:.2f})"
            ),
        )
    else:
        cand = keys[answer.choice]
        # The bands were calibrated on AUTO_MERGE_JEV_MODEL. An answer from any
        # other model may still suggest, but never auto-merge.
        calibrated = answer.model == AUTO_MERGE_JEV_MODEL
        verdict = AutoMergeVerdict(
            verdict="duplicate",
            primary_canonical_id=cand.canonical_id,
            confidence="high" if p >= AUTO_MERGE_JEV_HIGH_AT and calibrated else "medium",
            rationale=template_rationale(node, cand, p),
        )
    return Judgment(verdict=verdict, model=answer.model, p=p)


async def judge(
    node: dict[str, Any],
    candidates: list[Candidate],
    *,
    api_key: str,
    client: httpx.AsyncClient | None = None,
) -> Judgment:
    """One Jev call -> Judgment. Raises what `post_choice` raises.

    With no candidate left to offer (every id too long to store as a
    primary) there is nothing to ask, and the verdict is unique.
    """
    offered = [c for c in candidates if len(c.canonical_id) <= MAX_PRIMARY_ID_CHARS]
    if not offered:
        return Judgment(
            verdict=AutoMergeVerdict(verdict="unique", rationale="no candidate id short enough to merge into"),
            model=AUTO_MERGE_JEV_MODEL,
        )
    state, question, keys = build_request(node, offered)
    answer = await post_choice(
        state,
        question,
        api_key=api_key,
        model=AUTO_MERGE_JEV_MODEL,
        breaker=MERGE_BREAKER,
        client=client,
    )
    return verdict_from_answer(answer, keys, node)


# --------------------------------------------------------------------------
# identity evidence (also the analyzer's Person guard) and the rationale

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
#: `github:<owner>/<repo>:pr:<n>` or `<owner>/<repo>#<n>` (PRs and issues).
NUMBERED_RE = re.compile(r"^(?:github:)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?::(?:pr|issue):|#)(\d+)$")
#: `name`, `owner/name` or `wiki:repo:name` -- the ids repos appear under.
REPO_SHAPED_RE = re.compile(r"(?:wiki:repo:)?(?:[a-z0-9_.-]+/)?[a-z0-9_.-]+")
_HEX = re.compile(r"\b[0-9a-f]{7,40}\b")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def fold_id(value: str) -> str:
    """Case and the -/_ spelling difference only. Dots and every other
    character stay: `model-v1.1` is not `model-v11`."""
    return value.lower().replace("-", "_")


def repo_parts(canonical_id: str) -> tuple[str | None, str]:
    """(owner or None, name folded to [a-z0-9]) for a repo-shaped id.

    `prbe-ai/prbe-agent-tap` -> ("prbe-ai", "prbe_agent_tap");
    `wiki:repo:prbe_agent_tap` and `prbe-agent-tap` -> (None, "prbe_agent_tap").
    """
    c = canonical_id.lower()
    wiki = c.startswith("wiki:repo:")
    c = c.removeprefix("wiki:repo:")
    if "/" in c and not wiki:
        owner, name = c.split("/", 1)
        return owner, fold_id(name)
    return None, fold_id(c)


def same_repo(a: str, b: str) -> bool:
    """Both ids name the same repository: the same name up to case and -/_,
    at least one of them in a repo form (`owner/name` or `wiki:repo:name`),
    and the same owner whenever both carry one (`alice/utils` is not
    `bob/utils`). Two bare slugs are left to the punctuation rule."""
    la, lb = a.lower(), b.lower()
    if not (REPO_SHAPED_RE.fullmatch(la) and REPO_SHAPED_RE.fullmatch(lb)):
        return False
    if not any("/" in x or x.startswith("wiki:repo:") for x in (la, lb)):
        return False
    (oa, na), (ob, nb) = repo_parts(a), repo_parts(b)
    return len(na) > 3 and na == nb and (oa is None or ob is None or oa == ob)


def shared_identifier(
    a_id: str, a_props: dict[str, Any] | None, b_id: str, b_props: dict[str, Any] | None
) -> str | None:
    """A concrete identifier the two entities share, described; else None.

    Email: an exact (case-insensitive) match of the property, or one side's
    email being the other side's canonical id. Emails are global, so any
    source counts. Login: the same, but ONLY within one source system -- a
    GitHub login can collide with an opaque id from somewhere else (a Slack
    user id is also a short upper/lower-case token). Names are NOT identifiers.
    """
    pa, pb = a_props or {}, b_props or {}
    ea, eb = _text(pa.get("email")), _text(pb.get("email"))
    if ea and ea.lower() == eb.lower():
        return f"shared email {ea}"
    for side, other_id in ((pa, b_id), (pb, a_id)):
        e = _text(side.get("email"))
        if e and e.lower() == other_id.strip().lower():
            return f"email {e} is the other entity's id"
    sa, sb = _text(pa.get("source_system")).lower(), _text(pb.get("source_system")).lower()
    if not sa or sa != sb:
        return None
    la, lb = _text(pa.get("login")), _text(pb.get("login"))
    if la and la.lower() == lb.lower():
        return f"shared handle {la}"
    for side, other_id in ((pa, b_id), (pb, a_id)):
        login = _text(side.get("login"))
        if login and login.lower() == other_id.strip().lower():
            return f"login {login} is the other entity's id"
    return None


def leaf_uuid(canonical_id: str) -> str | None:
    """The id's own UUID: its LAST segment, when that segment is a UUID.

    A UUID anywhere else names a parent every sibling shares:
    `linear:<workspace>:issue:<issue>` its workspace, and
    `custom_ingest:<customer>:<source>:<doc>` the TENANT -- so "the last UUID
    anywhere" would make every upload without a UUID of its own match every
    other one.
    """
    last = re.split(r"[:/#]", canonical_id.lower())[-1]
    return last if UUID_RE.fullmatch(last) else None


def _id_evidence(a: str, b: str, *, repo_rules: bool = True) -> str | None:
    """Id-shaped evidence. `repo_rules` adds the two name rules (repo-style
    slug up to case and -/_, the same repo name): for a repo they are
    identity, for any other entity they amount to "same name"."""
    ua, ub = leaf_uuid(a), leaf_uuid(b)
    if ua and ua == ub:
        return f"shared id {ua}"
    na, nb = NUMBERED_RE.match(a), NUMBERED_RE.match(b)
    if na and nb and (na.group(1).lower(), na.group(2)) == (nb.group(1).lower(), nb.group(2)):
        return f"same repo {na.group(1)} and number {na.group(2)}"
    if not repo_rules:
        return None
    if (
        a != b
        and _alnum(a)
        and REPO_SHAPED_RE.fullmatch(a.lower())
        and REPO_SHAPED_RE.fullmatch(b.lower())
        and fold_id(a) == fold_id(b)
    ):
        # Repo-style slugs only: GitHub names ignore case and spell -/_ both
        # ways. An arbitrary id (a customer's upload key) is case-sensitive.
        return f"ids equal ignoring case and -/_ ({a} ~ {b})"
    if same_repo(a, b):
        return f"same repo name once owner/wiki prefix and -/_ are ignored ({a} ~ {b})"
    return None


def execution_evidence(
    label: str, a_id: str, a_props: dict[str, Any] | None, b_id: str, b_props: dict[str, Any] | None
) -> str | None:
    """The deterministic identity evidence an AUTO-merge requires; None if absent.

    A judge proposes; this decides whether the proposal may execute without a
    human. People need a shared email or login (a name is not identity). Other
    entities also accept id evidence: the same leaf UUID or the same repo +
    PR/issue number; Documents (where repos live) also the same repo-style
    slug up to case and -/_, or the same repo name. On the 2026-09-23 replay
    every verified Jev auto-merge carried such evidence; the one that did not
    was a name-only Person pair.
    """
    if label == NodeLabel.PERSON:
        return shared_identifier(a_id, a_props, b_id, b_props)
    return shared_identifier(a_id, a_props, b_id, b_props) or _id_evidence(
        a_id, b_id, repo_rules=label == NodeLabel.DOCUMENT
    )


def template_rationale(node: dict[str, Any], cand: Candidate, p: float | None) -> str:
    """The one strongest shared signal, then the numbers. <= RATIONALE_MAX_CHARS.

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
    return text[:RATIONALE_MAX_CHARS]
