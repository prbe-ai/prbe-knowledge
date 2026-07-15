"""Agent compactor — Flash Lite summarizer for the agent loop's conversation tail.

When AgentLoop's `_estimate_tokens()` crosses 60% of the model context
window, it calls `call_summarizer(messages, runtime_state)` to compress
the conversation history. The runtime state (pending_updates,
pending_creates, applied_queue_ids, skipped_queue_ids) is preserved
verbatim in the summary text — only conversational fluff (model
commentary, intermediate read_page / get_event_body responses) is
dropped.

The summarizer model is gemini-3.1-flash-lite, chosen for cost
(<$0.001 per compaction call) and latency (~2s). Preserves enough
context for the agent to continue making decisions without hallucinating
about already-staged updates.
"""

from __future__ import annotations

import json
from typing import Any

from engine.shared.constants import WIKI_AGENT_COMPACTOR_MODEL
from engine.shared.exceptions import AgentCompactionError
from engine.shared.logging import get_logger

log = get_logger(__name__)


_COMPACTOR_SYSTEM = """You are a conversation summarizer for a wiki-editing agent.

Given the agent's conversation so far AND its current structured
runtime state, produce a compact summary that preserves:

1. The structured runtime state VERBATIM (do not paraphrase queue_ids
   or slugs).
2. Key decisions the agent has made and why (which events it skipped,
   which pages it staged updates for).
3. Open questions the agent was thinking about.

Drop:
  - Verbose tool call/response payloads (just note what was read).
  - Model commentary that didn't change the runtime state.
  - Repeated reasoning the agent has already committed to.

Output is plain text; the next turn's prompt will paste this verbatim
into the conversation. Aim for under 1000 tokens. Lead with the
runtime state block so the agent can find it.
"""


def extract_state_for_summary(runtime_state: dict[str, Any]) -> str:
    """Serialize runtime state into a compact, deterministic block.

    The serialization format is what the agent reads back after a
    compaction; keep it stable across versions or you risk the agent
    re-creating already-staged pages.
    """
    pending_updates = runtime_state.get("pending_updates") or []
    pending_creates = runtime_state.get("pending_creates") or []
    applied = runtime_state.get("applied_queue_ids") or []
    skipped = runtime_state.get("skipped_queue_ids") or []
    return (
        "RUNTIME STATE:\n"
        f"  pending_updates: {len(pending_updates)} pages\n"
        + "\n".join(
            f"    - [{u.get('wiki_type')}/{u.get('slug')}] "
            f"events={u.get('applied_queue_ids') or []}"
            for u in pending_updates
        )
        + (f"\n  pending_creates: {len(pending_creates)} pages\n" if pending_creates else "")
        + "\n".join(
            f"    - [{c.get('wiki_type')}/{c.get('slug')}] "
            f"events={c.get('applied_queue_ids') or []}"
            for c in pending_creates
        )
        + f"\n  applied_queue_ids: {sorted(applied)}\n"
        + f"  skipped_queue_ids: {sorted(skipped)}\n"
    )


async def call_summarizer(
    messages: list[dict[str, Any]],
    runtime_state: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = WIKI_AGENT_COMPACTOR_MODEL,
) -> str:
    """Compress `messages` into a summary string.

    Preserves `runtime_state` verbatim (via extract_state_for_summary)
    and asks the summarizer model only to compress the conversational
    tail.

    Raises `AgentCompactionError` on any failure. The harness re-raises
    as `AgentHaltError('agent.compaction_failed')` so the drain DLQs
    cleanly.

    Phase-0b chunk B: the production call routes through
    `shared.llm.acompletion`. When `LLM_GATEWAY_URL` is set (managed-
    isolated tenant) the wrapper forwards to the central LiteLLM proxy;
    without it LiteLLM falls back to the direct provider call using the
    `GOOGLE_API_KEY` env var (self-host / dev). The ``client`` kwarg is
    preserved for tests that inject a stub mimicking the google-genai
    surface (``client.aio.models.generate_content``) — when supplied we
    drive that path verbatim instead of the wrapper, so existing test
    fixtures keep working.
    """
    state_block = extract_state_for_summary(runtime_state)
    convo_text = _serialize_messages(messages)

    user_prompt = (
        "Conversation so far:\n\n"
        f"{convo_text}\n\n"
        "Current runtime state:\n\n"
        f"{state_block}\n"
    )

    if client is not None:
        # Legacy / test-injected client path — keeps the google-genai
        # call shape so fixtures using AsyncMock'd
        # `.aio.models.generate_content` still work unchanged.
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=user_prompt,
                config={
                    "system_instruction": _COMPACTOR_SYSTEM,
                    "max_output_tokens": 2048,
                },
            )
        except Exception as exc:
            log.warning(
                "agent_compactor.gemini_failed",
                error=str(exc),
                error_class=type(exc).__name__,
            )
            raise AgentCompactionError(f"gemini summarize failed: {exc}") from exc
        text = getattr(resp, "text", None) or ""
    else:
        # Production path: route through shared.llm so the call honors
        # LLM_GATEWAY_URL for gateway-routed tenants. The wrapper
        # auto-injects api_base (and, post-chunk-A, api_key) when the
        # gateway env var is set; otherwise litellm uses GOOGLE_API_KEY.
        from engine.shared import llm as shared_llm
        from engine.shared.config import get_settings

        # Gate on either a configured gateway URL or a direct provider
        # key — without one of these, the compactor cannot reach a model.
        if not (
            shared_llm.gateway_url()
            or get_settings().google_api_key.get_secret_value()
        ):
            raise AgentCompactionError(
                "Neither LLM_GATEWAY_URL nor GOOGLE_API_KEY set; compactor unavailable"
            )

        try:
            resp = await shared_llm.acompletion(
                model=f"gemini/{model}",
                messages=[
                    {"role": "system", "content": _COMPACTOR_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
            )
        except shared_llm.LLMError as exc:
            log.warning(
                "agent_compactor.gemini_failed",
                error=str(exc),
                error_class=type(exc).__name__,
                status_code=exc.status_code,
                provider=exc.provider,
            )
            raise AgentCompactionError(f"gemini summarize failed: {exc}") from exc

        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            log.warning(
                "agent_compactor.gemini_malformed_response",
                error=str(exc),
            )
            raise AgentCompactionError(
                f"gemini returned malformed response: {exc}"
            ) from exc

    if not text.strip():
        raise AgentCompactionError("gemini returned empty summary")
    # Make sure the state block round-trips even if the model decided to
    # paraphrase it. Prepend the verbatim block.
    if state_block.strip().splitlines()[0] not in text:
        text = state_block + "\n\n" + text
    return text


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
    """Render a Gemini-shaped contents list into a flat-text trace.

    The summarizer doesn't need full fidelity (it's the compactor —
    the whole point is dropping detail), so a compact YAML-ish render
    is fine. Function calls render as 'TOOL <name>: <args>'; function
    responses as 'RESULT <name>: <body>'; text parts as the model /
    user prefix + raw text.
    """
    out: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        for part in msg.get("parts") or []:
            if "text" in part:
                txt = (part.get("text") or "").strip()
                if txt:
                    out.append(f"{role}: {txt[:1500]}")
            if "function_call" in part:
                fc = part["function_call"]
                args = json.dumps(fc.get("args") or {}, default=str)[:1000]
                out.append(f"{role} TOOL {fc.get('name')}: {args}")
            if "function_response" in part:
                fr = part["function_response"]
                body = json.dumps(fr.get("response") or {}, default=str)[:1000]
                out.append(f"{role} RESULT {fr.get('name')}: {body}")
    return "\n".join(out)


__all__ = [
    "call_summarizer",
    "extract_state_for_summary",
]
