# Document-grouped retrieval

Switch `/retrieve` and `/query` (+ `/query/stream`) from flat-chunks-per-doc to documents-with-their-matched-chunks. Hard cutover, no flag. Ranking incorporates how many chunks of a doc actually matched. Drop `QueryBundle`; lift its info onto per-chunk `graph_evidence` and a top-level confidence aggregate.

## Why

- Fusion already groups per-doc for scoring (`fusion.py:117–155`) but collapses to ONE best content chunk per doc at output (`fusion.py:196–219`). Multi-chunk hits in the same doc lose visibility.
- Consumers (MCP `search_knowledge`/`query_knowledge`, dashboard search + chat console) want a doc-shaped view: one entry per source with its matched chunks nested.
- `QueryBundle` was a side-channel that worked around an unpopulated `QueryChunk.graph_evidence` field. Wiring up `graph_evidence` per chunk (as `list[GraphEvidence]` for M:N) makes bundles redundant. Confidence breakdown moves to a flat top-level aggregate.
- `/query` and `/query/stream` both call `run_retrieval()` internally — fixing fusion + response shape once propagates to both endpoints.

## Out of scope

- No full-doc body in responses. `chunks[]` under each doc carries ONLY the content chunks that surfaced from the retrievers. No new IO, no `documents.body` joins.
- No doc-fetching helper, no on-demand drill-down — `SourceViewResponse` already exists for that.
- Keep `/query` and `/query/stream` (dashboard chat console depends on them). Just reshape their response.

## Response shape (hard cutover, `/retrieve` + `/query` + `/query/stream`)

```jsonc
{
  "query": "...",
  "documents": [
    {
      "doc_id": "issue:8bcb...",
      "doc_version": 4,
      "source_system": "linear",
      "source_url": "...",
      "title": "...",
      "author_id": "...",
      "created_at": "...",
      "updated_at": "...",
      "score": 0.84,
      "rank": 1,
      "chunk_count": 3,
      "retriever_scores": { "vector": ..., "bm25": ..., "metadata_vector": ... },
      "chunks": [
        {
          "chunk_id": "...",
          "score": 0.42,
          "rank_in_doc": 1,
          "content": "...",
          "graph_evidence": [                     // list, was scalar; usually empty
            { "edge_type": "MENTIONS", "confidence": "EXTRACTED",
              "via_entity": "prbe-ai/prbe-knowledge:Normalizer.process_queue_row",
              "reason": null }
          ]
        },
        { "chunk_id": "...", "score": 0.28, "rank_in_doc": 2,
          "content": "...", "graph_evidence": [] }
      ]
    },
    { "doc_id": "...", ... }
  ],
  "total_candidates": 42,
  "confidence_breakdown": { "EXTRACTED": 12, "INFERRED": 3, "AMBIGUOUS": 1 },  // flat aggregate over all graph-retrieved chunks; absent/zeroed when no graph hits
  "applied_temporal": {...},
  "applied_sort": {...},
  "applied_entity_filter": {...},
  "applied_mode": "...",
  "applied_doc_types": [...],
  "applied_min_confidence": "...",
  "extracted_entities": [...],
  "aggregation": null,
  "timing_ms": {...},
  "trace_id": "...",
  "related_entities": [...] | null,
  "related_entities_error": null
  // bundles: REMOVED
}
```

`AnswerResponse` (`/query`) gets the same `documents` reshape; citations stay chunk-keyed so the answer-side joins still work. `/query/stream` SSE events that carry chunk/doc payloads switch to the doc-grouped frames.

## Doc score formula

Replace `combined_for_doc[doc] = best_content_rrf + metadata_rrf_sum` with:

```
doc_score = max(content_chunk_rrfs)
          + alpha * sum(other_content_chunk_rrfs)
          + metadata_rrf_sum
```

with `alpha = 0.3` (constants.py — `RRF_BREADTH_ALPHA`). Source multiplier + recency decay still applied to the final number, same as today.

Why these choices:
- `max + alpha*sum_of_others` instead of plain `sum`: prevents long docs from drowning shorter, more relevant ones; preserves "best chunk wins ties" semantics.
- Not raw `count`: a doc with 5 weak chunks shouldn't beat a doc with 1 strong + 1 medium chunk.
- 0.3 is a starting alpha — tunable via constant if eval shifts.

## Files to change

### prbe-knowledge (core — most of the work)

- [ ] `shared/constants.py` — add `RRF_BREADTH_ALPHA = 0.3`
- [ ] `shared/models.py`
  - [ ] New `QueryDocument` model (doc-level fields + `chunks: list[QueryChunk]` + `chunk_count`)
  - [ ] `QueryChunk` — add `rank_in_doc: int`; drop the doc-level redundant fields (`doc_id`, `doc_version`, `source_system`, `source_url`, `title`, `author_id`, `created_at`, `updated_at`) since they live on the parent now. Keep `chunk_id`, `content`, `score`, `retriever_scores`.
  - [ ] `QueryChunk.graph_evidence` → `list[GraphEvidence]` (was `GraphEvidence | None`, never populated). Default empty list. Populated only for chunks that surfaced via graph retriever.
  - [ ] `QueryResponse`: `chunks` → `documents: list[QueryDocument]`; add `confidence_breakdown: dict[str, int]` (flat aggregate); REMOVE `bundles`.
  - [ ] `AnswerResponse`: `chunks` → `documents: list[QueryDocument]`; add `confidence_breakdown` to match.
  - [ ] DELETE `QueryBundle` model entirely.
- [ ] `services/retrieval/fusion.py`
  - [ ] `FusedHit` → `FusedDocument` carrying `chunks: list[FusedChunk]` (or a parallel `FusedChunk` dataclass — pick whichever causes less churn in callers)
  - [ ] In `fuse()`: stop selecting `best_content_for_doc`; keep ALL content chunks per doc (`content_chunks_for_doc: dict[doc_id, list[(chunk_id, rrf_score)]]`)
  - [ ] Apply new formula at line ~155 (`max + alpha*sum_other + metadata_sum`)
  - [ ] Source multiplier + recency decay applied at doc-level (use the doc's `updated_at` from any chunk — they share it per the existing comment)
  - [ ] Sort chunks within each doc by their RRF descending; assign `rank_in_doc`
  - [ ] Top-level `top_k` now applies to documents, not chunks
- [ ] `services/retrieval/search_pipeline.py`
  - [ ] Callers of `fuse()` consume new return shape; emit `QueryDocument`s
  - [ ] DELETE `_build_bundles` and the `bundles=...` wiring on QueryResponse
  - [ ] Populate per-chunk `graph_evidence` from the `graph_hits` list — for each chunk_id surviving fusion/dedup/ACL, attach a `GraphEvidence` for every distinct `(via_entity, via_label, edge_type, confidence)` graph hit on that chunk_id (M:N — multiple seeds per chunk OK)
  - [ ] Compute top-level `confidence_breakdown`: count distinct `confidence` tiers across all `graph_evidence` entries on the surviving chunks
- [ ] `services/retrieval/pipeline.py` — `run_retrieval()` wires `documents` + `confidence_breakdown` into `QueryResponse`
- [ ] `services/retrieval/list_pipeline.py` — list mode emits one chunk per doc via the LATERAL join; wrap each in a single-chunk `QueryDocument` so the response shape is uniform across modes (doc score = chunk score, `chunk_count = 1`, `rank_in_doc = 1`, `graph_evidence = []`, `confidence_breakdown = {}`)
- [ ] `services/retrieval/main.py`
  - [ ] `/retrieve` and `/query` handlers — pass through new response model
  - [ ] `/query/stream` SSE — update event payload shapes that today emit chunks (the synthetic `AnswerResponse` at `main.py:362` and any phase events that include chunks/docs) to emit doc-grouped frames
- [ ] `services/retrieval/synthesis.py` — synthesizer iterates over flattened chunks for prompt construction (`[c for d in qr.documents for c in d.chunks]`), re-attaching doc-level fields (title, source_url, source_system) per chunk for the citation/grounding format. Same for streaming variant.
- [ ] `services/retrieval/retrievers/related_entities.py` — input set is now `documents`; walk semantics don't change (was already de-duping by `doc_id`)
- [ ] Tests:
  - [ ] `tests/services/retrieval/test_fusion.py`
  - [ ] `tests/services/retrieval/test_search_pipeline.py`
  - [ ] `tests/services/retrieval/test_pipeline.py`
  - [ ] `tests/services/retrieval/test_list_pipeline.py`
  - [ ] integration tests asserting `response.chunks` shape
  - [ ] add: doc with 3 matched chunks ranks above doc with 1 strong-only chunk
  - [ ] add: chunk-level scores preserved within each doc; `rank_in_doc` monotonic
  - [ ] add: `top_k=5` returns 5 documents, not 5 chunks
  - [ ] add: chunk surfaced via 2 graph seeds carries 2 `graph_evidence` entries
  - [ ] add: top-level `confidence_breakdown` aggregates correctly across docs
  - [ ] DELETE `tests/services/retrieval/test_bundles*` (or equivalent) — bundles are gone

### prbe-knowledge-mcp

- [ ] `app/clients/knowledge.py` — typed response model update to match new shape (both `/retrieve` and `/query` paths)
- [ ] `app/clients/_responses.py` — `compact_search()` and any other reshape: emit doc-grouped output to MCP callers. Keep `verbose=False` compaction; operate on `documents` instead of `chunks`. Drop bundle-related compaction.
- [ ] `app/server.py`
  - [ ] `search_knowledge` tool docstring: describe new shape (LLMs reading this need to know what they're getting); remove `bundles` mention
  - [ ] `query_knowledge` tool docstring: same — describe doc-grouped citation/answer shape
- [ ] Update any fixture asserting `chunks` at top level or `bundles` field

### prbe-dashboard

- [ ] `src/lib/api/knowledge.ts` — TS types for `QueryDocument`, `GraphEvidence` as list, updated `KnowledgeQueryResponse`/answer response; remove `QueryBundle` type
- [ ] `KnowledgeStreamEvent` (SSE event names + payloads) — update frames that carry chunk/doc payloads to the new shape
- [ ] Search/results renderer — group rendering by document; show matched chunks under each doc with their per-chunk scores. (Find call site via `retrieveKnowledge` import sites.)
- [ ] `QueryConsole` (chat console on `/overview`) — consumes `/query/stream` SSE; update to render doc-grouped chunks + citations
- [ ] Any other consumer of `chunks` (history, debug views, etc.)

### Drive-by sweep (check, only change if needed)

- [ ] `prbe-orchestrator` — anything calling `/retrieve` or `/query` directly?
- [ ] `prbe-backend` — BFF proxy is pass-through; confirm no payload-shape introspection

## Rollout (hard cutover)

1. Land prbe-knowledge change first — both endpoints emit new shape (auto-deploys on push to main per `feedback_ci_autodeploys_on_push.md`).
2. **Same day**: land prbe-knowledge-mcp + prbe-dashboard updates against the new shape.

If knowledge ships and MCP/dashboard lag, MCP search results break and dashboard search renders empty until they catch up. Window should be minutes, not hours.

## Verification

- [ ] Unit tests above
- [ ] Local: hit `/retrieve` with a query known to surface ≥3 chunks from the same doc; verify all 3 nested
- [ ] Verify ranking: doc-with-3-chunks out-ranks single-chunk doc with similar top score
- [ ] MCP: invoke `search_knowledge` from dev, verify Claude can parse new shape
- [ ] Dashboard: search bar returns grouped results; chunk drill-down still works

## Review

(filled in after implementation)
