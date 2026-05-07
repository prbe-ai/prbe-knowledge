"""LLM-based inferred-edge extractor.

Sends one structured call to Claude Haiku and validates the output.
Every validation drop reason has a counter in ExtractionResult.dropped.

Validation pipeline per edge:
  1. Both endpoints resolve to existing graph_nodes for bundle.customer_id.
  2. edge_type is in the extended EdgeType enum.
  3. confidence in {INFERRED, AMBIGUOUS}; EXTRACTED -> forced to AMBIGUOUS.
  4. why present and <= 200 chars.
  5. from != to (no self-edges).

Kill-switch: if dropped["unknown_endpoint"] / total > 0.5, fail the entire
bundle (probable bad LLM run -- do not pollute the graph).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

import asyncpg

from services.ingestion.inferred_edges.bundle import Bundle
from services.ingestion.inferred_edges.prompts.v1 import PROMPT_VERSION, SYSTEM_PROMPT
from shared.constants import HAIKU_MODEL, EdgeType
from shared.logging import get_logger

try:
    import anthropic as _anthropic_module
except ImportError:
    _anthropic_module = None  # type: ignore[assignment]

log = get_logger(__name__)

# Haiku input/output token pricing (USD per 1M tokens), as of 2026-05.
# Update if Anthropic changes pricing. Used for cost_usd metric only.
_HAIKU_INPUT_COST_PER_1M = 0.80
_HAIKU_OUTPUT_COST_PER_1M = 4.00

# Maximum output tokens from the LLM for the edge-extraction call.
_MAX_OUTPUT_TOKENS = 4096

# Valid confidence values the LLM may emit.
_VALID_CONFIDENCES = {"INFERRED", "AMBIGUOUS"}

# Kill-switch threshold: if more than 50% of all proposed edges have
# unknown endpoints, the whole bundle is failed.
_UNKNOWN_ENDPOINT_FAIL_RATIO = 0.5


@dataclass(slots=True)
class InferredEdge:
    """One validated, upsert-ready edge from the LLM."""

    from_label: str
    from_canonical_id: str
    to_label: str
    to_canonical_id: str
    edge_type: str  # EdgeType.value
    confidence: str  # INFERRED | AMBIGUOUS
    why: str
    extractor_id: str
    extracted_at: datetime


@dataclass
class ExtractionResult:
    """Validated output of one LLM extraction call."""

    edges: list[InferredEdge] = field(default_factory=list)
    # reason -> count for telemetry
    dropped: dict[str, int] = field(default_factory=dict)
    # USD cost estimate for the metric
    cost_usd: float = 0.0
    # Whether the kill-switch fired (too many unknown_endpoint drops)
    bundle_failed: bool = False
    bundle_fail_reason: str = ""


def _inc(dropped: dict[str, int], reason: str) -> None:
    dropped[reason] = dropped.get(reason, 0) + 1


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * _HAIKU_INPUT_COST_PER_1M
        + output_tokens / 1_000_000 * _HAIKU_OUTPUT_COST_PER_1M
    )


# ---- valid edge types set (extended with Lane B types) ---------------------

_VALID_EDGE_TYPES: set[str] = {e.value for e in EdgeType}


# ---- bundle serialisation --------------------------------------------------


def _bundle_to_user_message(bundle: Bundle) -> str:
    """Render bundle contents as a structured user message."""
    lines: list[str] = [
        f"# Bundle for anchor document: {bundle.anchor_doc_id}",
        f"# Customer: {bundle.customer_id}",
        f"# Total documents in bundle: {len(bundle.docs)}",
        "",
    ]
    for i, doc in enumerate(bundle.docs, 1):
        lines.append(f"## Document {i}: {doc.doc_id}")
        lines.append(f"   source_system: {doc.source_system}")
        if doc.title:
            lines.append(f"   title: {doc.title}")
        lines.append("")
        lines.append(doc.content)
        lines.append("")

    # Append the node manifest so the LLM knows which canonical_ids exist.
    lines.append("## Node manifest (use ONLY these canonical_ids):")
    # Dedupe by (doc_id -> label=Document, canonical_id=doc_id)
    for doc in bundle.docs:
        lines.append(f"  - label=Document  canonical_id={doc.doc_id}")

    return "\n".join(lines)


# ---- main extraction function ---------------------------------------------


async def extract_edges(
    bundle: Bundle,
    conn: asyncpg.Connection,
) -> ExtractionResult:
    """Call the LLM and return validated inferred edges.

    `conn` must be a tenant-scoped connection (with_tenant already called)
    for the endpoint existence checks in validation.

    The LLM call itself is skipped if ANTHROPIC_API_KEY is not set (returns
    an empty ExtractionResult). This keeps the worker from crashing in
    environments without credentials.
    """
    result = ExtractionResult()

    if not bundle.docs:
        log.debug("inferred_edges.extractor.empty_bundle", customer=bundle.customer_id)
        return result

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning(
            "inferred_edges.extractor.no_api_key",
            customer=bundle.customer_id,
            anchor=bundle.anchor_doc_id,
        )
        return result

    # ---- LLM call ----------------------------------------------------------
    try:
        if _anthropic_module is None:
            raise ImportError("anthropic package not installed")

        client = _anthropic_module.AsyncAnthropic(api_key=api_key)
        user_message = _bundle_to_user_message(bundle)

        # Prefill the assistant message with `[` so Haiku is forced to start
        # its response inside a JSON array. Without prefill the model is free
        # to emit a preamble ("Here are the edges I found:") or markdown
        # fences, both of which break json.loads. With prefill,
        # response.content[0].text is the body of the array; we re-prepend
        # `[` before parsing.
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "["},
            ],
        )

        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        result.cost_usd = _estimate_cost(input_tokens, output_tokens)

        raw_text = response.content[0].text if response.content else ""
    except Exception as exc:
        log.error(
            "inferred_edges.extractor.llm_call_failed",
            customer=bundle.customer_id,
            anchor=bundle.anchor_doc_id,
            error=str(exc),
        )
        result.bundle_failed = True
        result.bundle_fail_reason = f"llm_call_failed: {type(exc).__name__}"
        return result

    # ---- Reconstruct + parse the JSON array --------------------------------
    # The assistant prefill `[` was sent to the model but is NOT included in
    # raw_text -- raw_text is just what the model continued with. The model
    # has three sensible behaviours:
    #   1. "no edges" -> raw_text is empty / whitespace / just `]`. Treat
    #      as the empty array `[]` and return zero edges; this is the
    #      common case on bundles with nothing inferable.
    #   2. "edges" -> raw_text starts with `{...}, {...}, ..., {...}]`.
    #      Re-prepend `[` and parse.
    #   3. "edges, truncated by max_tokens" -> ends mid-element with no
    #      closing `]`. Best-effort: drop a trailing `,`, append `]`,
    #      try to parse. JSONDecodeError on a truncated-mid-element will
    #      fall through to the failure branch.
    stripped = raw_text.strip()
    if not stripped or stripped == "]":
        return result  # No edges; valid outcome.

    candidate = "[" + stripped
    if not candidate.endswith("]"):
        candidate = candidate.rstrip(",") + "]"

    try:
        raw_edges = json.loads(candidate)
        if not isinstance(raw_edges, list):
            log.warning(
                "inferred_edges.extractor.non_list_response",
                customer=bundle.customer_id,
                anchor=bundle.anchor_doc_id,
            )
            result.bundle_failed = True
            result.bundle_fail_reason = "non_list_response"
            return result
    except json.JSONDecodeError as exc:
        log.warning(
            "inferred_edges.extractor.json_parse_failed",
            customer=bundle.customer_id,
            anchor=bundle.anchor_doc_id,
            error=str(exc),
            raw_text_preview=raw_text[:200],
        )
        result.bundle_failed = True
        result.bundle_fail_reason = f"json_parse_failed: {exc}"
        return result

    if not raw_edges:
        return result  # Empty array is valid

    # ---- Validation pipeline -----------------------------------------------
    total = len(raw_edges)
    now = datetime.now(UTC)

    # Pre-load existing graph nodes for this customer for endpoint validation.
    # Fetch (label, canonical_id) pairs from the DB once to avoid N queries.
    existing_nodes: set[tuple[str, str]] = await _load_existing_nodes(
        conn, bundle.customer_id
    )

    for raw in raw_edges:
        if not isinstance(raw, dict):
            _inc(result.dropped, "bad_format")
            continue

        from_node = raw.get("from") or {}
        to_node = raw.get("to") or {}
        from_label = str(from_node.get("label") or "")
        from_cid = str(from_node.get("canonical_id") or "")
        to_label = str(to_node.get("label") or "")
        to_cid = str(to_node.get("canonical_id") or "")
        edge_type = str(raw.get("edge_type") or "")
        confidence = str(raw.get("confidence") or "")
        why = str(raw.get("why") or "")

        # Rule 5: self-edge
        if from_cid and from_cid == to_cid:
            _inc(result.dropped, "self_edge")
            continue

        # Rule 2: edge type
        if edge_type not in _VALID_EDGE_TYPES:
            _inc(result.dropped, "unknown_type")
            continue

        # Rule 3: confidence
        if confidence == "EXTRACTED":
            # Force-demote to AMBIGUOUS (never trust an LLM claiming EXTRACTED)
            confidence = "AMBIGUOUS"
            _inc(result.dropped, "forced_confidence_demoted")
            # Note: we continue processing this edge after demotion
        elif confidence not in _VALID_CONFIDENCES:
            _inc(result.dropped, "unknown_confidence")
            continue

        # Rule 4: why
        if not why or len(why) > 200:
            _inc(result.dropped, "bad_justification")
            continue

        # Rule 1: endpoint existence
        from_exists = (from_label, from_cid) in existing_nodes
        to_exists = (to_label, to_cid) in existing_nodes
        if not from_exists or not to_exists:
            _inc(result.dropped, "unknown_endpoint")
            log.debug(
                "inferred_edges.extractor.unknown_endpoint",
                customer=bundle.customer_id,
                from_label=from_label,
                from_cid=from_cid,
                to_label=to_label,
                to_cid=to_cid,
                from_exists=from_exists,
                to_exists=to_exists,
            )
            continue

        result.edges.append(
            InferredEdge(
                from_label=from_label,
                from_canonical_id=from_cid,
                to_label=to_label,
                to_canonical_id=to_cid,
                edge_type=edge_type,
                confidence=confidence,
                why=why,
                extractor_id=PROMPT_VERSION,
                extracted_at=now,
            )
        )

    # ---- Kill-switch: >50% unknown_endpoint -> fail bundle -----------------
    # `total` is the count of ALL proposed edges (including non-unknown_endpoint
    # drops like self_edge / bad_justification). This dilutes the ratio
    # intentionally — the spec is "fraction of total proposals that hallucinate
    # endpoints," not "fraction of validation failures." Don't change without
    # updating the spec.
    unknown_count = result.dropped.get("unknown_endpoint", 0)
    if total > 0 and unknown_count / total > _UNKNOWN_ENDPOINT_FAIL_RATIO:
        log.warning(
            "inferred_edges.extractor.bundle_killed_unknown_endpoints",
            customer=bundle.customer_id,
            anchor=bundle.anchor_doc_id,
            unknown_count=unknown_count,
            total=total,
        )
        result.edges = []
        result.bundle_failed = True
        result.bundle_fail_reason = (
            f"unknown_endpoint_ratio={unknown_count}/{total}"
        )

    return result


async def _load_existing_nodes(
    conn: asyncpg.Connection,
    customer_id: str,
) -> set[tuple[str, str]]:
    """Load all (label, canonical_id) pairs for customer_id from graph_nodes.

    The conn must already be scoped via with_tenant(customer_id). We also
    add the explicit WHERE customer_id = $1 for defense-in-depth.
    """
    rows = await conn.fetch(
        """
        SELECT label, canonical_id
        FROM graph_nodes
        WHERE customer_id = $1
        """,
        customer_id,
    )
    return {(r["label"], r["canonical_id"]) for r in rows}
