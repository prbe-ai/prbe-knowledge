"""Regression tests for MCP retrieval transport and result encoding."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent

from services.mcp import server
from services.mcp.clients.knowledge import (
    KnowledgeClient,
    KnowledgeError,
    _build_http_timeout,
)
from services.mcp.config import Settings
from services.mcp.services import response_budget
from services.mcp.services.response_budget import (
    MAX_RESPONSE_BYTES_HARD,
    fit_response_to_budget,
    serialize_tool_response,
)
from services.retrieval.synthesis import SYNTHESIS_TIMEOUT_SECONDS
from shared.constants import (
    SEARCH_AGENT_EXTRACTOR_TIMEOUT_SECONDS,
    SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS,
    SEARCH_AGENT_LOOP_TIMEOUT_SECONDS,
)


def _client(handler: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler, base_url="https://knowledge.example")


async def test_retrieve_timeout_preserves_empty_httpx_diagnostic() -> None:
    sent_body: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        sent_body.update(json.loads(request.content))
        raise httpx.ReadTimeout("", request=request)

    async with _client(httpx.MockTransport(handler)) as http:
        client = KnowledgeClient(http=http, internal_key="test-key")
        with pytest.raises(KnowledgeError) as raised:
            await client.retrieve(query="research-os", customer_id="customer-1")

    assert raised.value.status == 504
    assert raised.value.body == "/retrieve timed out (ReadTimeout)"
    assert raised.value.trace_id == sent_body["trace_id"]
    assert raised.value.trace_id.startswith("q-mcp-")
    assert str(raised.value) == "prbe-knowledge http 504: /retrieve timed out (ReadTimeout)"


async def test_retrieve_transport_error_uses_exception_type_when_message_empty() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("", request=request)

    async with _client(httpx.MockTransport(handler)) as http:
        client = KnowledgeClient(http=http, internal_key="test-key")
        with pytest.raises(KnowledgeError) as raised:
            await client.retrieve(query="research-os", customer_id="customer-1")

    assert raised.value.status == 502
    assert raised.value.body == "/retrieve transport error: ConnectError"
    assert raised.value.trace_id is not None


async def test_retrieve_http_error_keeps_upstream_status_and_body() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="token quota exceeded")

    async with _client(httpx.MockTransport(handler)) as http:
        client = KnowledgeClient(http=http, internal_key="test-key")
        with pytest.raises(KnowledgeError) as raised:
            await client.retrieve(query="research-os", customer_id="customer-1")

    assert raised.value.status == 429
    assert raised.value.body == "token quota exceeded"
    assert raised.value.trace_id is not None


@pytest.mark.parametrize(
    ("response_text", "expected_body"),
    [
        ("not-json", "/retrieve returned invalid JSON (JSONDecodeError)"),
        ("[]", "/retrieve returned list; expected JSON object"),
    ],
)
async def test_retrieve_invalid_json_uses_traceable_transport_error(
    response_text: str,
    expected_body: str,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=response_text)

    async with _client(httpx.MockTransport(handler)) as http:
        client = KnowledgeClient(http=http, internal_key="test-key")
        with pytest.raises(KnowledgeError) as raised:
            await client.retrieve(query="research-os", customer_id="customer-1")

    assert raised.value.status == 502
    assert raised.value.body == expected_body
    assert raised.value.trace_id is not None


async def test_successful_client_calls_share_transport_and_compact_payloads() -> None:
    sent_bodies: dict[str, dict[str, Any]] = {}
    source_query: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in {"/retrieve", "/query"}:
            body = json.loads(request.content)
            sent_bodies[path] = body
            payload: dict[str, Any] = {
                "trace_id": body["trace_id"],
                "timing_ms": {"total": 1},
                "results": [],
            }
            if path == "/query":
                payload["answer"] = "grounded"
            return httpx.Response(200, json=payload)
        source_query.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "doc_id": "github:test:1",
                "content": "source body",
                "metadata": {"internal": True},
                "trace_id": "q-source-success",
            },
        )

    async with _client(httpx.MockTransport(handler)) as http:
        client = KnowledgeClient(http=http, internal_key="test-key")
        search = await client.retrieve(query="research-os", customer_id="customer-1")
        query = await client.query(question="What changed?", customer_id="customer-1")
        source = await client.get_source(
            doc_id="github:test:1",
            customer_id="customer-1",
            mode="full",
        )

    assert search["trace_id"] == sent_bodies["/retrieve"]["trace_id"]
    assert query["trace_id"] == sent_bodies["/query"]["trace_id"]
    assert "timing_ms" not in search
    assert "timing_ms" not in query
    assert source["trace_id"] == "q-source-success"
    assert "metadata" not in source
    assert source_query["mode"] == "full"


def test_default_timeout_exceeds_upstream_search_and_synthesis_budgets() -> None:
    default_timeout = Settings.model_fields["knowledge_timeout_s"].default
    upstream_timeout_envelope = (
        SEARCH_AGENT_EXTRACTOR_TIMEOUT_SECONDS
        + SEARCH_AGENT_LOOP_TIMEOUT_SECONDS
        + SYNTHESIS_TIMEOUT_SECONDS
    )

    assert isinstance(default_timeout, (int, float))
    # A provider-chain turn is nested inside (and must finish before) the
    # gatherer loop; it is not double-counted in the end-to-end envelope.
    assert SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS < SEARCH_AGENT_LOOP_TIMEOUT_SECONDS
    assert default_timeout > upstream_timeout_envelope


def test_http_timeout_only_extends_the_upstream_read_phase() -> None:
    timeout = _build_http_timeout(
        Settings(
            _env_file=None,
            knowledge_timeout_s=180,
            knowledge_connect_timeout_s=7,
            knowledge_write_timeout_s=20,
            knowledge_pool_timeout_s=5,
        )
    )

    assert timeout.read == 180
    assert timeout.connect == 7
    assert timeout.write == 20
    assert timeout.pool == 5


async def test_search_tool_returns_explicit_timeout_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    async with _client(httpx.MockTransport(handler)) as http:
        client = KnowledgeClient(http=http, internal_key="test-key")
        monkeypatch.setattr(server, "get_current_customer", lambda: "customer-1")
        monkeypatch.setattr(server, "get_client", lambda: client)

        async with create_connected_server_and_client_session(server.mcp) as session:
            result = await session.call_tool(
                "search_knowledge",
                {"query": "research-os", "top_k_related": 0},
            )

    assert result.isError is True
    assert result.structuredContent is None
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    payload = json.loads(result.content[0].text)
    assert payload["status"] == 504
    assert payload["trace_id"].startswith("q-mcp-")
    assert payload["error"] == "prbe-knowledge http 504: /retrieve timed out (ReadTimeout)"


async def test_large_mcp_tools_emit_one_json_text_copy_on_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "WIRE-EVIDENCE-MARKER"

    class FakeClient:
        async def retrieve(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "query": "research-os",
                "trace_id": "q-test-wire",
                "results": [
                    {
                        "doc_id": "github:test:1",
                        "chunks": [
                            {
                                "content": marker,
                                "graph_evidence": [
                                    {
                                        "edge_type": "MENTIONS",
                                        "confidence": "EXTRACTED",
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "total_candidates": 1,
            }

        async def query(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "answer": marker,
                "trace_id": "q-test-query-wire",
                "results": [],
            }

        async def get_source(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "doc_id": "github:test:1",
                "content": marker,
                "trace_id": "q-test-source-wire",
            }

    monkeypatch.setattr(server, "get_current_customer", lambda: "customer-1")
    monkeypatch.setattr(server, "get_client", FakeClient)

    async with create_connected_server_and_client_session(server.mcp) as session:
        listed = await session.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        for name in ("search_knowledge", "query_knowledge", "get_source"):
            assert tools[name].outputSchema is None

        calls = {
            "search_knowledge": {"query": "research-os", "top_k_related": 0},
            "query_knowledge": {"question": "What changed?", "top_k_related": 0},
            "get_source": {"doc_id": "github:test:1"},
        }
        payloads: dict[str, dict[str, Any]] = {}
        for name, arguments in calls.items():
            result = await session.call_tool(name, arguments)
            assert result.isError is False
            assert result.structuredContent is None
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)
            payloads[name] = json.loads(result.content[0].text)
            assert result.content[0].text == serialize_tool_response(payloads[name])

    search_payload = payloads["search_knowledge"]
    assert search_payload["results"][0]["chunks"][0]["content"] == marker
    assert search_payload["results"][0]["chunks"][0]["graph_evidence"] == [
        {
            "edge_type": "MENTIONS",
            "confidence": "EXTRACTED",
        }
    ]
    assert search_payload["truncated"] is False
    assert search_payload["trace_id"] == "q-test-wire"
    assert payloads["query_knowledge"]["answer"] == marker
    assert payloads["query_knowledge"]["trace_id"] == "q-test-query-wire"
    assert payloads["get_source"]["content"] == marker
    assert payloads["get_source"]["trace_id"] == "q-test-source-wire"


@pytest.mark.parametrize("result_kind", ["document", "entity"])
async def test_large_search_payload_stays_under_wire_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
    result_kind: str,
) -> None:
    if result_kind == "document":
        results = [
            {
                "node_type": "Document",
                "doc_id": f"github:test:{index}",
                "score": 1 - index / 100,
                "chunks": [
                    {
                        "content": "evidence-" + "x" * 1_200,
                        "graph_evidence": [],
                    }
                ],
            }
            for index in range(30)
        ]
    else:
        results = [
            {
                "node_type": "Entity",
                "canonical_id": f"service:test:{index}",
                "display_name": f"Service {index}",
                "score": 1 - index / 100,
                "properties": {"description": "x" * 1_800},
            }
            for index in range(20)
        ]

    class FakeClient:
        async def retrieve(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "query": "research-os",
                "trace_id": f"q-large-{result_kind}",
                "results": results,
                "total_candidates": len(results),
            }

    monkeypatch.setattr(server, "get_current_customer", lambda: "customer-1")
    monkeypatch.setattr(server, "get_client", FakeClient)

    async with create_connected_server_and_client_session(server.mcp) as session:
        result = await session.call_tool(
            "search_knowledge",
            {"query": "research-os", "top_k_related": 0},
        )

    assert result.isError is False
    assert result.structuredContent is None
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    wire_text = result.content[0].text
    assert len(wire_text.encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD
    payload = json.loads(wire_text)
    assert wire_text == serialize_tool_response(payload)
    assert payload["truncated"] is True
    assert payload["trace_id"] == f"q-large-{result_kind}"
    assert payload["dropped_result_count"] > 0
    assert payload["results"]
    if result_kind == "entity":
        assert payload["results"][0]["canonical_id"] == "service:test:0"
        assert "chunks" not in payload["results"][0]
        assert payload["dropped_chunk_count"] == 0
    else:
        assert payload["results"][0]["doc_id"] == "github:test:0"
        assert all(document["chunks"] for document in payload["results"])


async def test_single_oversized_entity_returns_bounded_explicit_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        async def retrieve(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "query": "research-os",
                "trace_id": "q-huge-entity",
                "results": [
                    {
                        "node_type": "Entity",
                        "canonical_id": "service:test:huge",
                        "properties": {"description": "x" * 30_000},
                    }
                ],
            }

    monkeypatch.setattr(server, "get_current_customer", lambda: "customer-1")
    monkeypatch.setattr(server, "get_client", FakeClient)

    async with create_connected_server_and_client_session(server.mcp) as session:
        result = await session.call_tool(
            "search_knowledge",
            {"query": "research-os", "top_k_related": 0},
        )

    assert result.isError is True
    assert result.structuredContent is None
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    wire_text = result.content[0].text
    assert len(wire_text.encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD
    payload = json.loads(wire_text)
    assert wire_text == serialize_tool_response(payload)
    assert payload["truncated"] is True
    assert payload["error"] == "response exceeded MCP hard byte limit after trimming"
    assert payload["error_code"] == "response_too_large"
    assert payload["status"] == 413
    assert payload["trace_id"] == "q-huge-entity"


@pytest.mark.parametrize(
    "oversized_content",
    ["x" * 100_000, '"' * 12_000],
    ids=["large-raw-content", "json-escape-expansion"],
)
async def test_large_get_source_payload_returns_one_bounded_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
    oversized_content: str,
) -> None:
    class FakeClient:
        async def get_source(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "doc_id": "github:test:huge-source",
                "content": oversized_content,
                "trace_id": "q-huge-source",
            }

    monkeypatch.setattr(server, "get_current_customer", lambda: "customer-1")
    monkeypatch.setattr(server, "get_client", FakeClient)

    async with create_connected_server_and_client_session(server.mcp) as session:
        result = await session.call_tool(
            "get_source",
            {"doc_id": "github:test:huge-source", "mode": "full"},
        )

    assert result.isError is True
    assert result.structuredContent is None
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    wire_text = result.content[0].text
    assert len(wire_text.encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD
    payload = json.loads(wire_text)
    assert payload["status"] == 413
    assert payload["error_code"] == "response_too_large"
    assert payload["trace_id"] == "q-huge-source"


def test_no_results_payload_still_respects_hard_limit() -> None:
    fitted = fit_response_to_budget(
        {"answer": "x" * 30_000, "trace_id": "q-no-results"}
    )

    assert len(serialize_tool_response(fitted).encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD
    assert fitted["truncated"] is True
    assert fitted["trace_id"] == "q-no-results"


def test_related_entities_are_trimmed_before_hard_limit_fallback() -> None:
    fitted = fit_response_to_budget(
        {
            "trace_id": "q-related",
            "results": [
                {
                    "node_type": "Document",
                    "doc_id": "github:test:1",
                    "chunks": [{"content": "primary evidence"}],
                }
            ],
            "related_entities": [
                {
                    "canonical_id": f"service:test:{index}",
                    "properties": {"description": "界" * 500},
                }
                for index in range(30)
            ],
        }
    )

    assert len(serialize_tool_response(fitted).encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD
    assert fitted["truncated"] is True
    assert fitted["results"][0]["doc_id"] == "github:test:1"
    assert fitted["dropped_related_entity_count"] > 0
    assert fitted["related_entities"]


def test_multibyte_content_is_budgeted_by_encoded_bytes() -> None:
    fitted = fit_response_to_budget(
        {
            "trace_id": "q-unicode",
            "results": [
                {
                    "node_type": "Document",
                    "doc_id": "github:test:unicode",
                    "chunks": [{"content": "界" * 9_000}],
                }
            ],
        }
    )

    chunk = fitted["results"][0]["chunks"][0]
    assert len(serialize_tool_response(fitted).encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD
    assert chunk["content"] == "界" * 500
    assert chunk["content_truncated"] is True


def test_single_document_drops_tail_chunks_without_emergency_fallback() -> None:
    original_chunk_count = 20
    fitted = fit_response_to_budget(
        {
            "trace_id": "q-multi-chunk",
            "results": [
                {
                    "node_type": "Document",
                    "doc_id": "github:test:multi-chunk",
                    "chunks": [
                        {
                            "content": f"rank-{index}-" + "x" * 2_000,
                            "graph_evidence": [],
                        }
                        for index in range(original_chunk_count)
                    ],
                }
            ],
        }
    )

    surviving_chunks = fitted["results"][0]["chunks"]
    assert 0 < len(surviving_chunks) < original_chunk_count
    assert surviving_chunks[0]["content"].startswith("rank-0-")
    assert fitted["dropped_chunk_count"] == original_chunk_count - len(surviving_chunks)
    assert fitted["dropped_result_count"] == 0
    assert "error" not in fitted
    assert len(serialize_tool_response(fitted).encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD


def test_response_budget_truncates_oversized_document_without_mutating_input() -> None:
    original = {
        "results": [
            {
                "doc_id": "github:test:1",
                "chunks": [
                    {
                        "content": "x" * 30_000,
                        "graph_evidence": [{"edge_type": "MENTIONS"}],
                    }
                ],
            }
        ]
    }
    before = deepcopy(original)

    fitted = fit_response_to_budget(original)

    assert original == before
    assert len(serialize_tool_response(fitted).encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD
    assert fitted["truncated"] is True
    assert fitted["dropped_chunk_count"] == 0
    assert fitted["dropped_result_count"] == 0
    chunk = fitted["results"][0]["chunks"][0]
    assert chunk["content"] == "x" * 500
    assert chunk["content_truncated"] is True
    assert chunk["graph_evidence"] == []


def test_large_response_trimming_uses_bounded_full_payload_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measure_calls = 0
    original_measure = response_budget._measure

    def counting_measure(payload: dict[str, Any]) -> int:
        nonlocal measure_calls
        measure_calls += 1
        return original_measure(payload)

    monkeypatch.setattr(response_budget, "_measure", counting_measure)
    fitted = response_budget.fit_response_to_budget(
        {
            "results": [
                {
                    "node_type": "Document",
                    "doc_id": f"github:test:{index}",
                    "chunks": [{"content": "x" * 1_000}],
                }
                for index in range(500)
            ]
        }
    )

    assert measure_calls <= 5
    assert len(serialize_tool_response(fitted).encode("utf-8")) <= MAX_RESPONSE_BYTES_HARD
