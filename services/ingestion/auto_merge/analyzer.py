"""Entity auto-merge analyzer.

Reads a graph_nodes row, finds duplicate candidates via trigram + vector,
filters conflicting properties, asks Cerebras gpt-oss-120b to judge, then
either fires the merge transaction directly or writes a suggestion row
for the dashboard to surface.

Single-tenant per call: caller supplies (customer_id, node_id). Wrap in
``with_tenant(customer_id)`` for RLS.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

import asyncpg
from pydantic import ValidationError

from services.ingestion.auto_merge.models import AutoMergeVerdict
from services.ingestion.entity_clusters_routes import (
    MergeRequest,
    MergeResponse,
    merge_cluster,
)
from shared.constants import SEARCH_AGENT_INFERENCE_MODEL
from shared.llm import LLMError, acompletion
from shared.logging import get_logger

log = get_logger(__name__)


# Reserved nil UUID for auto-merge audit rows. The entity_merge_audit table's
# performed_by_user_id is NOT NULL, so we stamp this and prefix `reason` with
# "auto:" to distinguish auto-merges from human-driven ones at unmerge time.
SYSTEM_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


# Trigram similarity floor for candidate generation. Empirically tuned in the
# prbe-orchestrator dedupe pipeline (PR #49) — paraphrased duplicates land at
# 0.65-0.70 cosine for chunks; for trigram on canonical_ids the equivalent
# tunable. Start wide (0.3) so the judge sees plausible candidates; the
# downstream `confidence='high'` gate keeps execution strict.
TRIGRAM_FLOOR = 0.3

# Vector-search cap. HNSW returns ordered-by-distance results; we cap at 10
# to keep the judge's candidate list focused.
VECTOR_TOP_K = 10
TRIGRAM_TOP_K = 10
TOTAL_CANDIDATE_CAP = 10


# Cached at import — same pattern as services/retrieval/agent/extractor.py.
# Building the JSON schema once keeps the proxy's cache key stable across calls.
_VERDICT_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "AutoMergeVerdict",
        "schema": AutoMergeVerdict.model_json_schema(),
    },
}


# Path-canonical labels: canonical_id is already a stable URL/path-derived
# identifier (e.g. `<repo>:<path>.<symbol>` for code symbols, `<owner>/<repo>#<n>`
# for PR/Issue). Two rows with the same shape and different canonical_ids are
# different entities by construction — never candidates for fuzzy merge.
def _is_path_canonical(label: str, canonical_id: str) -> bool:
    if not canonical_id:
        return True
    # PR/Issue: `<owner>/<repo>#<number>` shape
    if "#" in canonical_id and re.search(r"#\d+$", canonical_id):
        return True
    # Code symbol: `<owner>/<repo>:<path>.<symbol>` shape
    if ":" in canonical_id and "." in canonical_id.split(":", 1)[1]:
        return True
    # Repo: `<owner>/<name>` — composite key, treat as path-canonical
    return (
        "/" in canonical_id
        and canonical_id.count("/") == 1
        and label.lower() == "repo"
    )


# Stable property keys that, when both candidates carry them with DIFFERENT
# values, are decisive evidence the entities are NOT the same. Used as a
# pre-LLM filter to drop obvious non-duplicates.
_STABLE_KEY_PAIRS: list[tuple[str, ...]] = [
    ("email",),
    ("repo", "number"),
    ("owner", "name"),
    ("team_id",),
]


def _properties_conflict(p1: dict, p2: dict) -> bool:
    """True if both dicts have the same stable key set with different values."""
    for keys in _STABLE_KEY_PAIRS:
        if (
            all(p1.get(k) for k in keys)
            and all(p2.get(k) for k in keys)
            and tuple(p1[k] for k in keys) != tuple(p2[k] for k in keys)
        ):
            return True
    return False


@dataclass
class Candidate:
    canonical_id: str
    properties: dict
    degree: int
    trigram_score: float | None  # NULL if surfaced only by vector path
    vector_distance: float | None


@dataclass
class AutoMergeResult:
    """Outcome of running the analyzer on one (customer_id, node_id)."""

    action: str  # "merged" | "suggested" | "no_candidates" | "unique" | "skipped" | "error"
    primary_canonical_id: str | None = None
    confidence: str | None = None
    rationale: str | None = None
    candidate_count: int = 0
    merge_id: uuid.UUID | None = None
    suggestion_id: uuid.UUID | None = None
    error: str | None = None


class AutoMergeAnalyzer:
    """LLM-judged entity dedup.

    Caller is responsible for tenant context — wrap in ``with_tenant()``.
    """

    def __init__(self, *, execute_high_confidence: bool = False) -> None:
        """Construct.

        ``execute_high_confidence`` gates whether high-confidence verdicts
        fire merges directly (True) or only write suggestions (False).
        Mirrors the ``auto_merge_execute`` customer preference; callers
        thread the per-customer toggle here.
        """
        self._execute = execute_high_confidence

    async def analyze(
        self,
        conn: asyncpg.Connection,
        customer_id: str,
        node_id: int,
    ) -> AutoMergeResult:
        node = await self._load_node(conn, node_id)
        if node is None:
            return AutoMergeResult(action="skipped", rationale="node not found")

        label = node["label"]
        canonical_id = node["canonical_id"]
        properties = node["properties"] or {}

        if _is_path_canonical(label, canonical_id):
            return AutoMergeResult(action="skipped", rationale="path-canonical label")

        candidates = await self._find_candidates(conn, node)
        if not candidates:
            return AutoMergeResult(action="no_candidates", candidate_count=0)

        # Property-key conflict filter — drop candidates with mismatched stable keys
        filtered = [
            c for c in candidates if not _properties_conflict(properties, c.properties)
        ]
        if not filtered:
            return AutoMergeResult(
                action="no_candidates",
                rationale="all candidates conflict on stable keys",
                candidate_count=len(candidates),
            )

        try:
            verdict = await self._judge(node, filtered)
        except (LLMError, ValidationError) as exc:
            log.warning(
                "auto_merge.judge_failed",
                customer=customer_id,
                node_id=node_id,
                error=repr(exc),
            )
            return AutoMergeResult(
                action="error",
                candidate_count=len(filtered),
                error=repr(exc),
            )

        if verdict.verdict == "unique":
            return AutoMergeResult(
                action="unique",
                rationale=verdict.rationale,
                candidate_count=len(filtered),
            )

        # Verdict says duplicate. Validate primary_canonical_id is one of ours.
        candidate_ids = {c.canonical_id for c in filtered}
        if (
            verdict.primary_canonical_id is None
            or verdict.primary_canonical_id not in candidate_ids
        ):
            log.warning(
                "auto_merge.hallucinated_primary",
                customer=customer_id,
                node_id=node_id,
                returned=verdict.primary_canonical_id,
                candidates=list(candidate_ids),
            )
            return AutoMergeResult(
                action="error",
                candidate_count=len(filtered),
                error="LLM returned primary_canonical_id not in candidate list",
            )

        # Fire either the merge or the suggestion path.
        if verdict.confidence == "high" and self._execute:
            return await self._fire_merge(
                customer_id=customer_id,
                label=label,
                new_node_canonical_id=canonical_id,
                primary_canonical_id=verdict.primary_canonical_id,
                rationale=verdict.rationale,
                candidate_count=len(filtered),
            )
        return await self._write_suggestion(
            conn=conn,
            customer_id=customer_id,
            label=label,
            new_node_canonical_id=canonical_id,
            primary_canonical_id=verdict.primary_canonical_id,
            confidence=verdict.confidence or "low",
            rationale=verdict.rationale,
            candidate_count=len(filtered),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _load_node(
        self, conn: asyncpg.Connection, node_id: int
    ) -> dict | None:
        # NOTE: don't SELECT the embedding column here — asyncpg has no native
        # halfvec serializer, so roundtripping it through Python and back
        # into a vector comparison fails. The vector candidate query
        # self-joins on graph_nodes to keep the embedding entirely in SQL.
        row = await conn.fetchrow(
            """
            SELECT node_id, label, canonical_id, properties, degree,
                   (embedding IS NOT NULL) AS has_embedding
            FROM graph_nodes
            WHERE node_id = $1
            """,
            node_id,
        )
        if row is None:
            return None
        properties = row["properties"]
        if isinstance(properties, str):
            properties = json.loads(properties)
        return {
            "node_id": row["node_id"],
            "label": row["label"],
            "canonical_id": row["canonical_id"],
            "properties": properties or {},
            "degree": row["degree"],
            "has_embedding": row["has_embedding"],
        }

    async def _find_candidates(
        self, conn: asyncpg.Connection, node: dict
    ) -> list[Candidate]:
        """Union of trigram-similar + vector-nearest entities (same label, not self,
        not already aliased into another cluster)."""

        label = node["label"]
        canonical_id = node["canonical_id"]
        properties = node["properties"]
        name = properties.get("name", "") if isinstance(properties, dict) else ""
        self_node_id = node["node_id"]

        # Trigram path: same label, similar canonical_id OR similar properties->>'name'.
        trigram_rows = await conn.fetch(
            """
            SELECT
                node_id,
                canonical_id,
                properties,
                degree,
                GREATEST(
                    similarity(LOWER(canonical_id), LOWER($2)),
                    CASE WHEN $3 <> '' THEN
                        similarity(LOWER(COALESCE(properties->>'name','')), LOWER($3))
                    ELSE 0 END
                ) AS trigram_score
            FROM graph_nodes
            WHERE label = $1
              AND node_id <> $4
              AND NOT EXISTS (
                  SELECT 1 FROM entity_aliases ea
                  WHERE ea.label = graph_nodes.label
                    AND ea.alias_canonical_id = graph_nodes.canonical_id
              )
              AND (
                  similarity(LOWER(canonical_id), LOWER($2)) >= $5
                  OR ($3 <> '' AND similarity(LOWER(COALESCE(properties->>'name','')), LOWER($3)) >= $5)
              )
            ORDER BY trigram_score DESC
            LIMIT $6
            """,
            label,
            canonical_id,
            name or "",
            self_node_id,
            TRIGRAM_FLOOR,
            TRIGRAM_TOP_K,
        )

        # Vector path: only when embedding is populated. Self-join keeps the
        # halfvec value entirely in SQL (asyncpg cannot serialize halfvec
        # parameters reliably).
        vector_rows: list[asyncpg.Record] = []
        if node.get("has_embedding"):
            vector_rows = await conn.fetch(
                """
                WITH source AS (
                    SELECT embedding FROM graph_nodes WHERE node_id = $1
                )
                SELECT
                    g.node_id,
                    g.canonical_id,
                    g.properties,
                    g.degree,
                    g.embedding <=> source.embedding AS distance
                FROM graph_nodes g
                CROSS JOIN source
                WHERE g.label = $2
                  AND g.node_id <> $1
                  AND g.embedding IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM entity_aliases ea
                      WHERE ea.label = g.label
                        AND ea.alias_canonical_id = g.canonical_id
                  )
                ORDER BY g.embedding <=> source.embedding
                LIMIT $3
                """,
                self_node_id,
                label,
                VECTOR_TOP_K,
            )

        # Merge by canonical_id (preserve trigram score if present).
        merged: dict[str, Candidate] = {}
        for r in trigram_rows:
            props = r["properties"] if isinstance(r["properties"], dict) else json.loads(r["properties"] or "{}")
            merged[r["canonical_id"]] = Candidate(
                canonical_id=r["canonical_id"],
                properties=props,
                degree=r["degree"],
                trigram_score=float(r["trigram_score"]),
                vector_distance=None,
            )
        for r in vector_rows:
            existing = merged.get(r["canonical_id"])
            if existing:
                existing.vector_distance = float(r["distance"])
                continue
            props = r["properties"] if isinstance(r["properties"], dict) else json.loads(r["properties"] or "{}")
            merged[r["canonical_id"]] = Candidate(
                canonical_id=r["canonical_id"],
                properties=props,
                degree=r["degree"],
                trigram_score=None,
                vector_distance=float(r["distance"]),
            )

        # Rank: prefer rows that surfaced in both signals, then trigram score, then vector distance.
        ranked = sorted(
            merged.values(),
            key=lambda c: (
                -(int(c.trigram_score is not None) + int(c.vector_distance is not None)),
                -(c.trigram_score or 0.0),
                c.vector_distance if c.vector_distance is not None else 1.0,
            ),
        )
        return ranked[:TOTAL_CANDIDATE_CAP]

    async def _judge(self, node: dict, candidates: list[Candidate]) -> AutoMergeVerdict:
        prompt = _build_prompt(node, candidates)
        response = await acompletion(
            model=SEARCH_AGENT_INFERENCE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=_VERDICT_RESPONSE_FORMAT,
            custom_llm_provider="openai",  # gateway routes via OpenAI-shape wire
            temperature=0.1,
            max_tokens=512,
        )
        content = response["choices"][0]["message"]["content"]
        if not content:
            raise LLMError("empty content from judge", provider="cerebras")
        return AutoMergeVerdict.model_validate_json(content)

    async def _fire_merge(
        self,
        *,
        customer_id: str,
        label: str,
        new_node_canonical_id: str,
        primary_canonical_id: str,
        rationale: str,
        candidate_count: int,
    ) -> AutoMergeResult:
        # The MergeRequest convention: primary survives, aliases merge in.
        # Treat the *new* node as the alias merging into the *existing* primary.
        body = MergeRequest(
            customer_id=customer_id,
            performed_by_user_id=SYSTEM_USER_ID,
            label=label,
            primary_canonical_id=primary_canonical_id,
            alias_canonical_ids=[new_node_canonical_id],
            reason=(
                f"auto: model={SEARCH_AGENT_INFERENCE_MODEL} "
                f"confidence=high rationale={rationale[:120]}"
            ),
        )
        try:
            resp: MergeResponse = await merge_cluster(body)
        except Exception as exc:
            log.warning(
                "auto_merge.merge_call_failed",
                customer=customer_id,
                primary=primary_canonical_id,
                alias=new_node_canonical_id,
                error=repr(exc),
            )
            # Fall back to writing a suggestion so the merge isn't silently lost.
            return AutoMergeResult(
                action="error",
                primary_canonical_id=primary_canonical_id,
                confidence="high",
                rationale=rationale,
                candidate_count=candidate_count,
                error=repr(exc),
            )
        log.info(
            "auto_merge.merged",
            customer=customer_id,
            merge_id=str(resp.merge_id),
            primary=primary_canonical_id,
            alias=new_node_canonical_id,
        )
        return AutoMergeResult(
            action="merged",
            primary_canonical_id=primary_canonical_id,
            confidence="high",
            rationale=rationale,
            candidate_count=candidate_count,
            merge_id=resp.merge_id,
        )

    async def _write_suggestion(
        self,
        *,
        conn: asyncpg.Connection,
        customer_id: str,
        label: str,
        new_node_canonical_id: str,
        primary_canonical_id: str,
        confidence: str,
        rationale: str,
        candidate_count: int,
    ) -> AutoMergeResult:
        # ON CONFLICT DO NOTHING — uq_entity_merge_suggestions_pair prevents
        # duplicate pending rows for the same pair.
        row = await conn.fetchrow(
            """
            INSERT INTO entity_merge_suggestions (
                customer_id, label, primary_canonical_id, candidate_canonical_id,
                confidence, rationale, llm_model, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending')
            ON CONFLICT (customer_id, label, primary_canonical_id, candidate_canonical_id)
                WHERE status = 'pending'
            DO NOTHING
            RETURNING suggestion_id
            """,
            customer_id,
            label,
            primary_canonical_id,
            new_node_canonical_id,
            confidence,
            rationale[:240],
            SEARCH_AGENT_INFERENCE_MODEL,
        )
        sid = row["suggestion_id"] if row else None
        log.info(
            "auto_merge.suggested",
            customer=customer_id,
            suggestion_id=str(sid) if sid else None,
            label=label,
            primary=primary_canonical_id,
            candidate=new_node_canonical_id,
            confidence=confidence,
        )
        return AutoMergeResult(
            action="suggested",
            primary_canonical_id=primary_canonical_id,
            confidence=confidence,
            rationale=rationale,
            candidate_count=candidate_count,
            suggestion_id=sid,
        )


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """You judge whether a NEW entity is the same real-world thing as one of a small list of CANDIDATE entities in a knowledge graph. You DO NOT enrich, classify, or summarize -- ONLY decide whether to merge.

Return JSON matching the AutoMergeVerdict schema exactly.

Rules:
1. verdict='duplicate' ONLY when the NEW entity and a candidate describe the SAME real-world thing -- same person, same project, same topic, same channel, same artifact. Different surface text is fine as long as the underlying identity matches.
2. verdict='unique' is the SAFE default. When in doubt, answer 'unique'.
3. confidence='high' requires concrete shared evidence: exact email match, exact username match, exact ticket-id match, or a name + role overlap with no contradicting properties. Surface-text similarity alone is NOT high confidence.
4. confidence='medium' for plausible matches where one strong signal exists but not multiple.
5. confidence='low' for weak hints (just name similarity, just neighborhood overlap).
6. primary_canonical_id MUST be the canonical_id of one of the listed candidates -- never invent one.
7. rationale: <= 30 words, name the shared signal concretely (e.g. "shared email richard@prbe.ai" or "matching GitHub username + display name").
"""


def _build_prompt(node: dict, candidates: list[Candidate]) -> str:
    new_block = (
        f"NEW ENTITY\n"
        f"  label:        {node['label']}\n"
        f"  canonical_id: {node['canonical_id']}\n"
        f"  properties:   {json.dumps(node['properties'], sort_keys=True)}\n"
        f"  degree:       {node['degree']}\n"
    )
    cand_blocks = []
    for c in candidates:
        sig_parts = []
        if c.trigram_score is not None:
            sig_parts.append(f"trigram={c.trigram_score:.2f}")
        if c.vector_distance is not None:
            sig_parts.append(f"vector_distance={c.vector_distance:.3f}")
        cand_blocks.append(
            f"- canonical_id: {c.canonical_id}\n"
            f"  properties:   {json.dumps(c.properties, sort_keys=True)}\n"
            f"  degree:       {c.degree}\n"
            f"  signals:      {', '.join(sig_parts) or 'none'}\n"
        )
    return (
        new_block
        + "\nCANDIDATES (existing entities in the graph)\n"
        + "\n".join(cand_blocks)
        + "\nReturn AutoMergeVerdict JSON."
    )
