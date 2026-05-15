# LLM call-site migration inventory (Phase 0a -> 0b)

Phase 0a (this PR) added `shared/llm.py` — a LiteLLM-backed wrapper
exposing `acompletion` / `aembedding` plus an `LLMError` shape, per
plan D1 (Adopt LiteLLM as the LLM provider abstraction).

Phase 0b is the call-site migration: every direct provider-SDK call in
the table below must be rewritten on top of `shared/llm.py` so:

  - Self-host customers can route through their own LiteLLM proxy via
    `LLM_GATEWAY_URL` without code changes (plan D15: "self-host
    customer brings own LLM keys; we never see LLM traffic").
  - Per-call token accounting can be done in one place (plan D15:
    LLM-token billing line-item-transparent in managed mode).
  - Adding a new provider becomes a one-line config change instead of
    a fourth SDK integration.

## Production call sites (must migrate in Phase 0b)

| File | Line | Function | Provider | Model used | Call shape | Purpose |
|------|------|----------|----------|------------|------------|---------|
| `shared/embeddings.py` | — | `GeminiEmbedder._ensure_client` + `_embed_subbatch` (`client.aio.models.embed_content`) | Google | `gemini-embedding-2` (`shared.constants.EMBEDDING_V2_MODEL`) | embedding (batched, parallel sub-batches) | Sole production embedder for ingest + retrieval (cutover 2026-05-14, PR #263; OpenAI embedder + SDK stripped in follow-up). Asymmetric prefixing (`title: ... | text: ...` for docs vs `task: search result | query: ...` for queries) |
| `shared/claude_code_extraction.py` | 138 | `extract_units_from_session` | Anthropic | `settings.claude_code_extraction_model` (Sonnet variant) | completion + tool_use forced (`emit_units` tool) | Per-session knowledge-unit extraction (QA / CodeChange / Decision / FileRef) from Claude Code transcripts. Tool-use as structured output |
| `services/synthesis/providers.py` | 116 | `_AnthropicTriage.triage` | Anthropic | `HAIKU_MODEL` (claude-haiku-4-5) | completion + tool_use forced (`emit_triage_verdicts`) | Wiki-synthesis triage stage; decides which docs are wiki-worthy |
| `services/synthesis/providers.py` | 234 | `_gemini_call_json` (used by `_GeminiTriage.triage` line 287) | Google | `gemini-3.1-flash-lite` (GA; alias `gemini-flash-lite`) | completion + `response_schema` (structured JSON) | Alternate triage backend; flippable via `WIKI_TRIAGE_MODEL` constant |
| `services/synthesis/providers.py` | 433 | `_AnthropicDirectedPhrases.generate` | Anthropic | `HAIKU_MODEL` | completion + tool_use forced (directed-phrases tool) | Generate retrieval-trigger phrases per wiki page (legacy/fallback path). Replaced by Gemini per 2026-05-09 eval but kept for A-B |
| `services/synthesis/providers.py` | 472-479 | `_GeminiDirectedPhrases.generate` (calls `_gemini_call_json`) | Google | **`gemini-3-flash-preview`** (PINNED — see project memory `project_gemini_dedupe_model_id`) | completion + `response_schema` | Default directed-phrases generator (per 2026-05-09 model-shootout eval). DO NOT silently downgrade |
| `services/synthesis/triage.py` | 55, 209, 224 | `call_triage` (provider-dispatched via `get_triage_provider`) | Anthropic (when configured) | `HAIKU_MODEL` | completion + tool_use forced | Triage-stage orchestration entry point. Owns the `AsyncAnthropic` client lifetime; provider impl lives in `providers.py` |
| `services/synthesis/triage.py` | 337 | `call_triage_with_split_retry` | Anthropic | `HAIKU_MODEL` | completion + tool_use forced (split-retry on `BadRequestError` overflow) | Defense-in-depth wrapper handling Anthropic 200K context overflow via recursive batch halving |
| `services/synthesis/triage_worker.py` | 159 | `TriageWorker._resolve_client` (constructs `AsyncAnthropic`); `triage_worker` calls flow into `triage.call_triage_with_split_retry` | Anthropic | `HAIKU_MODEL` | completion + tool_use forced | Drains `wiki_synthesis_queue` rows from `pending` -> `triaged`. Owns one `AsyncAnthropic` for the worker process lifetime |
| `services/synthesis/agent_compactor.py` | 110 | `compact_agent_conversation` | Google | `WIKI_AGENT_COMPACTOR_MODEL` (gemini-3.1-flash-lite) | completion (free-text summary) | Wiki-agent loop conversation compactor (fires when context >60% of window) |
| `services/synthesis/gemini_agent_client.py` | 120 | `GeminiAgentClient._ensure_client` + `create_cache` (line 145+) + `generate_with_cache` | Google | `WIKI_AGENT_MODEL` (gemini-3.1-pro-preview) | **CachedContent** + completion + tool-use (function declarations) | Wiki agent loop — uses Gemini's `CachedContent` API for prompt caching + function-calling. **NOTE: LiteLLM does not surface Gemini CachedContent. This call site likely cannot migrate without either (a) keeping the direct SDK for the cache-create path, or (b) switching to LiteLLM's caching layer with feature parity verification. Flag for Phase 0b design discussion** |
| `services/synthesis/index_renderer.py` | 273 | `_render_index_with_llm` | Google | `WIKI_INDEX_MODEL` (gemini-3.1-pro-preview) | completion (free-text Markdown) | LLM-driven wiki index page renderer (replaces deterministic grouping) |
| `services/ingestion/code_graph/cross_repo_deps.py` | 360 | `_call_classifier_llm` | Google | `gemini-3.1-pro-preview` (hardcoded — see line 371) | completion + system_instruction (free-text JSON) | Per-source-repo cross-repo dependency classifier (REAL vs COINCIDENCE). Was Flash-Lite, upgraded to Pro per PRs #184/#186 for directionality |
| `services/ingestion/inferred_edges/extractor.py` | 224, 229 | `_call_anthropic_with_backoff` | Anthropic | `INFERRED_EDGES_MODEL` (when prefix `claude-*`) | completion with assistant-prefill (`[`) trick for JSON-array output | LLM-based inferred-edge extractor; Anthropic branch |
| `services/ingestion/inferred_edges/extractor.py` | ~341 (`_call_gemini`) | `_call_gemini` | Google | `INFERRED_EDGES_MODEL` (when prefix `gemini-*`) | completion + `response_schema` (structured JSON array) | LLM-based inferred-edge extractor; Gemini branch |
| `services/retrieval/router.py` | 43, 431, 438 | `_call_haiku` | Anthropic | `HAIKU_MODEL` | completion + tool_use forced (`route_query`) + Anthropic prompt caching (`cache_control: ephemeral`) | Per-query router: extracts entities + temporal + mode for downstream retrievers. Hot path |
| `services/retrieval/synthesis.py` | 332-344 (streaming) | `synthesize_stream` (anthropic branch) | Anthropic | `model` (parameterized; e.g. `anthropic/claude-sonnet-4-6`) | streaming completion (`messages.stream`) | Streaming retrieval-grounded synthesis. **Streaming via LiteLLM uses `acompletion(..., stream=True)`** |
| `services/retrieval/synthesis.py` | 361-378 (streaming) | `synthesize_stream` (google branch) | Google | `model` (parameterized; e.g. `gemini/gemini-3-flash-preview`) | streaming completion (`generate_content_stream`) + `thinking_config: {thinking_budget: 0}` | Streaming retrieval-grounded synthesis (Google branch). LiteLLM passes `thinking` config through provider-specific kwargs — verify in Phase 0b |
| `services/retrieval/synthesis.py` | 469-487 (non-stream) | `_call_anthropic` | Anthropic | `model` (parameterized) | completion + tool_use forced (`render_answer`) | Non-streaming structured-output synthesis; Anthropic branch |
| `services/retrieval/synthesis.py` | 514-530 (non-stream) | `_call_openai` | OpenAI | `model` (parameterized) | completion + `response_format=json_schema` (strict mode) | Non-streaming structured-output synthesis; OpenAI branch. Strict JSON-schema mode |
| `services/retrieval/synthesis.py` | 578-597 (non-stream) | `_call_google` | Google | `model` (parameterized) | completion + `response_schema` (sanitized JSON-Schema) + `thinking_config: {thinking_budget: 0}` | Non-streaming structured-output synthesis; Google branch. Strips `additionalProperties` (Google rejects it) |

**Total production call sites: 22.**

## Scripts / eval / dev tools (lower priority for Phase 0b)

These are out-of-band utilities and one-off harnesses. Migration is
useful for `LLM_GATEWAY_URL` support but not blocking on the managed-
isolated / self-host split (the production data plane never executes
them).

| File | Provider(s) | Purpose |
|------|-------------|---------|
| `scripts/eval_directed_phrases.py` | Anthropic + OpenAI + Google | Manual eval harness for directed-phrase generation — Haiku 4.5 vs Gemini variants. Uses Opus 4.7 as judge |
| `scripts/synth/llm/anthropic_client.py` | Anthropic | Synth-pipeline (`scripts/synth/`) Anthropic client; tool_use + prompt caching |
| `scripts/synth/llm/gemini_client.py` | Google | Synth-pipeline Gemini client; `response_schema` structured output |

## Notes for Phase 0b

1. **Gemini `gemini-3-flash-preview` model id is pinned** for the
   directed-phrases dedupe judge (project memory
   `project_gemini_dedupe_model_id`). LiteLLM accepts
   `gemini/gemini-3-flash-preview` as the routed id — verify the
   resolved API call shape end-to-end before the migration lands.
   **No fallbacks**; surface 4xx loud rather than silently downgrade.

2. **Tool-use semantics differ across providers.** LiteLLM normalizes
   to OpenAI's `tools` + `tool_choice` shape; the existing Anthropic
   `tool_use` blocks and Google `function_declarations` need response-
   shape adapters. Verify each tool-use call site (router, triage,
   directed-phrases, claude-code-extraction, retrieval synthesis) in a
   shadow eval before flipping the call site.

3. **Streaming**: LiteLLM exposes streaming via
   `acompletion(..., stream=True)` returning an async iterator of
   chunks. The retrieval `synthesize_stream` currently iterates over
   provider-specific text streams; the migration must preserve the
   `StreamDelta` / `StreamFinal` event shape downstream consumers
   (dashboard, MCP `query_knowledge`) depend on.

4. **Gemini CachedContent** (`services/synthesis/gemini_agent_client.py`)
   is the call-site most likely to need a carve-out from the LiteLLM
   migration — LiteLLM does not (today) expose Gemini's per-customer
   CachedContent lifecycle. Either keep the direct SDK for that path
   or switch to LiteLLM's own prompt-cache abstraction with a
   reference-test for cache-hit-rate parity.

5. **OpenAI strict JSON-schema mode** (`response_format` with
   `strict=true`) currently lives only in the OpenAI synthesis
   branch. LiteLLM forwards `response_format` to the OpenAI provider
   verbatim — verify on the migration PR.

6. **Anthropic prompt caching** (`cache_control: ephemeral`) is used on
   the router's system block. LiteLLM passes Anthropic-specific kwargs
   through; the cache-hit-rate is observable from
   `usage.cache_creation_input_tokens` / `usage.cache_read_input_tokens`
   in the response — preserve that telemetry on migration.

## Phase 0c — per-tenant virtual-key attribution (shared-managed)

Phase 0b made every call route through `shared.llm`, which honors a
single process-wide `LLM_GATEWAY_KEY`. That's correct for **managed-
isolated** (one Helm release per tenant → one key baked into the pod's
env). It is NOT sufficient for the **shared-managed** data plane where
one worker handles many tenants — every call would attribute to the
master key and per-tenant cost accounting collapses.

`shared/litellm_key.py` (added in this PR) fills that gap:

  - `get_tenant_virtual_key(customer_id)` fetches the tenant's LiteLLM
    virtual key from the control plane (`GET {backend_base_url}/routing/
    customer/{customer_id}/litellm-key`, auth: `X-Internal-Key`) and
    caches it for 5 minutes per customer.
  - `tenant_virtual_key_context(customer_id)` is an `async with` block
    that binds the key onto a ContextVar; `shared/llm.py`
    `_maybe_inject_gateway` consults the ContextVar first and prefers
    it over `LLM_GATEWAY_KEY` when both `LLM_GATEWAY_URL` and a tenant
    key are set.

**Required entrypoint wiring (follow-up PRs, NOT in this PR).** Every
request/worker-task that knows its `customer_id` should wrap the LLM
call window:

```python
async with tenant_virtual_key_context(customer_id):
    await call_haiku_router(query)
    await synthesize_stream(...)
```

Recommended order, in descending impact:

  1. `services/retrieval/router._call_haiku` + the synthesize path in
     `services/retrieval/synthesis.py` (hot per-query path — every
     user query routes through these).
  2. `services/synthesis/triage_worker.py` + `wiki_synthesis_worker.py`
     drain loops (highest token-volume background jobs).
  3. `services/ingestion/inferred_edges/extractor.py` and
     `services/ingestion/code_graph/cross_repo_deps.py` per-tenant
     ingest paths.

**Control-plane dependency.** The endpoint
`GET /routing/customer/{customer_id}/litellm-key` is owned by
prbe-backend (PR #206 in the control-plane repo, per the triggering
ticket). Until it lands, `get_tenant_virtual_key` raises
`LiteLLMKeyUnavailable`; callers that wrap entrypoints in
`tenant_virtual_key_context` must either gate the call behind a flag
or treat `LiteLLMKeyUnavailable` as a soft-fall-back to the master
key (the existing `LLM_GATEWAY_KEY` path keeps working without the
context wrapper).
