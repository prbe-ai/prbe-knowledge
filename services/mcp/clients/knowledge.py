"""HTTP client for prbe-knowledge retrieval service.

One client instance per process, lazily constructed. Each call passes
`customer_id` as the X-Prbe-Customer header so retrieval scopes results
to that tenant. Internal-key auth lives on the same call.
"""

from __future__ import annotations

from typing import Any

import httpx

from services.mcp.clients._responses import (
    compact_query,
    compact_search,
    compact_source_view,
)
from services.mcp.config import get_settings

CALLER_KIND = "mcp"


class KnowledgeError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"prbe-knowledge http {status}: {body[:200]}")
        self.status = status
        self.body = body


class KnowledgeClient:
    """Thin HTTP wrapper over prbe-knowledge retrieval service."""

    def __init__(self, http: httpx.AsyncClient, internal_key: str) -> None:
        self._http = http
        self._internal_key = internal_key

    def _headers(self, customer_id: str) -> dict[str, str]:
        return {
            "X-Internal-Knowledge-Key": self._internal_key,
            "X-Prbe-Customer": customer_id,
            "X-Caller-Kind": CALLER_KIND,
            "Content-Type": "application/json",
        }

    async def retrieve(
        self,
        *,
        query: str,
        customer_id: str,
        top_k: int = 5,
        sources: list[str] | None = None,
        entity_must_match: bool | None = None,
        top_k_related: int = 10,
        discovery: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "top_k_related": top_k_related,
        }
        if sources:
            body["sources"] = sources
        if entity_must_match is not None:
            body["entity_must_match"] = entity_must_match
        # Only forward discovery when set so older retrieval deployments
        # that don't recognise the field aren't sent unknown keys.
        if discovery:
            body["discovery"] = True
        resp = await self._http.post("/retrieve", json=body, headers=self._headers(customer_id))
        if resp.status_code >= 400:
            raise KnowledgeError(resp.status_code, resp.text)
        payload = resp.json()
        return payload if verbose else compact_search(payload)

    async def query(
        self,
        *,
        question: str,
        customer_id: str,
        top_k: int = 5,
        entity_must_match: bool | None = None,
        discovery: bool = False,
        top_k_related: int = 0,
        verbose: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": question,
            "top_k": top_k,
            "top_k_related": top_k_related,
        }
        if entity_must_match is not None:
            body["entity_must_match"] = entity_must_match
        if discovery:
            body["discovery"] = True
        resp = await self._http.post("/query", json=body, headers=self._headers(customer_id))
        if resp.status_code >= 400:
            raise KnowledgeError(resp.status_code, resp.text)
        payload = resp.json()
        return payload if verbose else compact_query(payload)

    async def get_source(
        self,
        *,
        doc_id: str,
        customer_id: str,
        mode: str = "preview",
        query: str | None = None,
        pattern: str | None = None,
        start_line: int | None = None,
        limit_lines: int = 80,
        chunk_index: int | None = None,
        context_lines: int = 3,
        max_matches: int = 20,
        cursor: str | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        # doc_id may contain colons (e.g. "linear:org:issue:uuid"); FastAPI's
        # `:path` converter on the server side allows them, but we still
        # URL-encode defensively in case other slashes appear.
        from urllib.parse import quote

        path = f"/source-view/{quote(doc_id, safe=':')}"
        params: dict[str, Any] = {
            "mode": mode,
            "limit_lines": limit_lines,
            "context_lines": context_lines,
            "max_matches": max_matches,
        }
        if query is not None:
            params["query"] = query
        if pattern is not None:
            params["pattern"] = pattern
        if start_line is not None:
            params["start_line"] = start_line
        if chunk_index is not None:
            params["chunk_index"] = chunk_index
        if cursor is not None:
            params["cursor"] = cursor
        resp = await self._http.get(
            path, params=params, headers=self._headers(customer_id)
        )
        if resp.status_code >= 400:
            raise KnowledgeError(resp.status_code, resp.text)
        payload = resp.json()
        return payload if verbose else compact_source_view(payload)


_client: KnowledgeClient | None = None


def get_client() -> KnowledgeClient:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.knowledge_query_url:
            raise RuntimeError("KNOWLEDGE_QUERY_URL is not set")
        if not settings.internal_knowledge_api_key:
            raise RuntimeError("INTERNAL_KNOWLEDGE_API_KEY is not set")
        http = httpx.AsyncClient(
            base_url=settings.knowledge_query_url.rstrip("/"),
            timeout=settings.knowledge_timeout_s,
        )
        _client = KnowledgeClient(http=http, internal_key=settings.internal_knowledge_api_key)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client._http.aclose()
        _client = None
