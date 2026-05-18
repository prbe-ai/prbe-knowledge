"""Tests for the trace-analyzer module (loader + digest + CLI).

Loader/CLI tests run without live_db because the loader uses an async
generator we can drive with a monkeypatched `iter_trace_blobs`. The
SQL-shape regression guard tests the module-level string verbatim —
no DB needed.
"""

from __future__ import annotations

import json
from datetime import date as date_cls
from typing import Any

import pytest

from services.retrieval.agent.trace_analyzer import (
    __main__ as cli,
)
from services.retrieval.agent.trace_analyzer import (
    loader,
)
from services.retrieval.agent.trace_analyzer.digest import summarize_trace

# ============================================================
# loader: regression-guard the SQL shape (no DB needed)
# ============================================================


def test_loader_sql_uses_range_predicate_not_date_cast() -> None:
    """CRITICAL regression guard: the loader must use
    `occurred_at >= $1 AND occurred_at < $2`, NOT
    `occurred_at::date = $1`. The cast form forces a seq scan even
    though `idx_query_traces_customer_time` is on `occurred_at`
    directly. Static-string check prevents anyone "fixing" it back.
    """
    sql = loader._LOAD_SQL
    assert "occurred_at >= $1" in sql
    assert "occurred_at < $2" in sql
    # Belt-and-suspenders: explicit anti-pattern check.
    assert "occurred_at::date" not in sql, (
        "loader SQL must NOT cast occurred_at — that disables the index. "
        "Use the range predicate instead."
    )


def test_loader_sql_requires_trace_blob_key_not_null() -> None:
    """Rows with NULL trace_blob_key are sampled-out or R2-write-failure
    rows. Yielding them would cause the loader to attempt R2 fetches for
    keys that don't exist."""
    assert "trace_blob_key IS NOT NULL" in loader._LOAD_SQL


def test_loader_sql_only_targets_required_columns() -> None:
    """Sanity: the SQL doesn't drift from what the digest reads."""
    sql = loader._LOAD_SQL
    for col in (
        "request_id",
        "customer_id",
        "occurred_at",
        "trace_blob_key",
        "event_type",
        "gatherer_status",
        "tool_calls_count",
        "confidence",
        "cache_hit_rate",
        "dropped_count",
        "need_deeper_extensions",
    ):
        assert col in sql, f"missing column {col!r} from loader SQL"


# ============================================================
# digest: pure function tests
# ============================================================


def _mk_blob(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "trace_id": "trace-abc",
        "customer_id": "cust-1",
        "query": "what shipped today",
        "model": "fireworks/gpt-oss-120b",
        "timestamp_utc": "2026-05-17T20:00:00+00:00",
        "status": "ok",
        "timing_ms": {
            "extraction_ms": 12.0,
            "grounding_ms": 30.0,
            "prefanout_ms": 400.0,
            "agent_ms": 5000.0,
            "agent_loop_ms": 4500.0,
            "agent_tools_ms": 500.0,
        },
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "", "tool_calls": []},
        ],
        "tools_fired": ["vector_search", "bm25_search"],
        "turn_1_tools_fired": ["vector_search", "bm25_search"],
        "turn_count": 2,
        "tool_calls_count": 2,
        "extensions_used": 0,
        "cache_hit_rates": [0.4, 0.8],
        "turn_latencies_ms": [1500.0, 3000.0],
        "tool_latencies_ms": [50.0, 80.0],
        "prose_retries": 0,
        "prefanout": {},
        "prefanout_hit_counts": {"vector": 15, "bm25": 12, "graph": 0, "inferred_edge": 0},
        "reasoning_per_turn": ["picked vector_search first", None],
        "gathered": {
            "entities": [],
            "chunks": [{"doc_id": "d1", "chunk_id": "c1", "content": "x", "matched_via": ["vector"], "why_relevant": "..."}],
            "gatherer_notes": {
                "turns_used": 2,
                "tools_called": ["vector_search"],
                "confidence": "high",
                "dropped": [],
            },
        },
        "_db": {
            "request_id": "trace-abc",
            "customer_id": "cust-1",
            "occurred_at": "2026-05-17T20:00:00+00:00",
            "event_type": "knowledge.retrieve",
            "response_size_bytes": 4096,
            "gatherer_status": "ok",
            "tool_calls_count": 2,
            "confidence": "high",
            "cache_hit_rate": 0.6,
            "dropped_count": 0,
            "need_deeper_extensions": 0,
            "trace_blob_key": "search-traces/2026-05-17/trace-abc.json.gz",
            "bucket_name": "prbe-cust-1",
        },
    }
    base.update(overrides)
    return base


def test_summarize_trace_happy_path_extracts_all_fields() -> None:
    out = summarize_trace(_mk_blob())
    assert out["trace_id"] == "trace-abc"
    assert out["request_id"] == "trace-abc"
    assert out["customer_id"] == "cust-1"
    assert out["query"] == "what shipped today"
    assert out["model"] == "fireworks/gpt-oss-120b"
    assert out["status"] == "ok"
    assert out["confidence"] == "high"
    assert out["chunk_count"] == 1
    assert out["entity_count"] == 0
    assert out["tool_calls_count"] == 2
    assert out["turn_count"] == 2
    assert out["tool_call_sequence"] == ["vector_search", "bm25_search"]
    assert out["turn_1_tools_fired"] == ["vector_search", "bm25_search"]
    assert set(out["turn_1_missed_channels"]) == {"graph_search", "inferred_edge_search"}
    assert out["cache_hit_rate_mean"] == pytest.approx(0.6)
    assert out["turn_latencies_ms"] == [1500.0, 3000.0]
    assert out["agent_ms"] == 5000.0
    assert out["had_reasoning_per_turn"] is True
    # bucket_name + blob_key flow through from _db
    assert out["bucket_name"] == "prbe-cust-1"
    assert out["blob_key"] == "search-traces/2026-05-17/trace-abc.json.gz"


def test_summarize_trace_handles_failure_status() -> None:
    """fatal_provider_error / loop_timeout traces have gathered=None.
    The digest must not crash."""
    blob = _mk_blob(
        status="fatal_provider_error",
        gathered=None,
        turn_count=1,
        tools_fired=[],
        turn_1_tools_fired=[],
        cache_hit_rates=[0.2],
    )
    blob["_db"]["gatherer_status"] = "fatal_provider_error"
    blob["_db"]["confidence"] = None
    out = summarize_trace(blob)
    assert out["status"] == "fatal_provider_error"
    assert out["chunk_count"] == 0
    assert out["entity_count"] == 0
    assert out["confidence"] is None
    assert set(out["turn_1_missed_channels"]) == {
        "vector_search",
        "bm25_search",
        "graph_search",
        "inferred_edge_search",
    }


def test_summarize_trace_handles_zero_turns() -> None:
    """no_llm_configured short-circuit produces a blob with turn_count=0."""
    blob = _mk_blob(
        status="no_llm_configured",
        turn_count=0,
        tools_fired=[],
        turn_1_tools_fired=[],
        cache_hit_rates=[],
        turn_latencies_ms=[],
        reasoning_per_turn=[],
    )
    blob["_db"]["gatherer_status"] = "no_llm_configured"
    out = summarize_trace(blob)
    assert out["turn_count"] == 0
    assert out["cache_hit_rate_mean"] is None
    assert out["had_reasoning_per_turn"] is False


def test_summarize_trace_tolerates_missing_db_stitch() -> None:
    """Fixtures + manual replay may not have the loader's _db stitch.
    Digest still produces a parseable shape; bucket_name falls back to
    the formula derived from customer_id."""
    blob = _mk_blob()
    del blob["_db"]
    out = summarize_trace(blob)
    assert out["request_id"] is None
    assert out["bucket_name"] == "prbe-cust-1"  # formula fallback
    assert out["blob_key"] is None


def test_summarize_trace_reasoning_signal_false_when_all_none() -> None:
    blob = _mk_blob(reasoning_per_turn=[None, None, None])
    out = summarize_trace(blob)
    assert out["had_reasoning_per_turn"] is False


def test_summarize_trace_detects_exploration_tools() -> None:
    blob = _mk_blob(
        tools_fired=["vector_search", "reissue_query", "expand_inferred_neighbors"]
    )
    out = summarize_trace(blob)
    assert out["had_reissue_query"] is True
    assert out["had_expand_inferred_neighbors"] is True


def test_summarize_trace_need_deeper_signal() -> None:
    blob = _mk_blob(extensions_used=1)
    out = summarize_trace(blob)
    assert out["had_need_deeper"] is True


# ============================================================
# __main__ CLI argparse smoke
# ============================================================


def test_cli_rejects_invalid_date() -> None:
    """Bad --date must exit 2 (argparse convention)."""
    rc = cli.main(["--date", "not-a-date"])
    assert rc == 2


def _stub_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub init_pool/close_pool so CLI tests don't touch a real DB.
    The CLI initializes the pool on entry (the FastAPI app does this
    via its startup hook, but the standalone CLI has no lifespan)."""
    from unittest.mock import AsyncMock as _AsyncMock
    monkeypatch.setattr(
        "services.retrieval.agent.trace_analyzer.__main__.init_pool",
        _AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "services.retrieval.agent.trace_analyzer.__main__.close_pool",
        _AsyncMock(return_value=None),
    )


def test_cli_accepts_valid_date_no_traces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path with empty iteration writes 0 lines and exits 0."""
    _stub_pool(monkeypatch)

    async def _empty_iter(*_args: Any, **_kwargs: Any) -> Any:
        if False:
            yield {}
        return

    monkeypatch.setattr(
        "services.retrieval.agent.trace_analyzer.__main__.iter_trace_blobs",
        _empty_iter,
    )
    out_path = tmp_path / "digests.jsonl"
    rc = cli.main(["--date", "2026-05-17", "--out", str(out_path)])
    assert rc == 0
    assert out_path.read_text() == ""
    captured = capsys.readouterr()
    assert "wrote 0 digest" in captured.err


def test_cli_writes_jsonl_lines_for_blobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Each yielded blob produces one JSONL line via summarize_trace."""
    _stub_pool(monkeypatch)
    blobs = [_mk_blob(), _mk_blob(trace_id="trace-2", status="loop_timeout")]

    async def _iter(*_args: Any, **_kwargs: Any) -> Any:
        for b in blobs:
            yield b

    monkeypatch.setattr(
        "services.retrieval.agent.trace_analyzer.__main__.iter_trace_blobs",
        _iter,
    )
    out_path = tmp_path / "digests.jsonl"
    rc = cli.main(["--date", "2026-05-17", "--out", str(out_path)])
    assert rc == 0
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["trace_id"] == "trace-abc"
    assert parsed[0]["status"] == "ok"
    assert parsed[1]["trace_id"] == "trace-2"
    assert parsed[1]["status"] == "loop_timeout"
    # blob_key threaded through from _db on the loaded blob
    assert parsed[0]["blob_key"] == "search-traces/2026-05-17/trace-abc.json.gz"


# ============================================================
# fetch_one CLI — sub-agent's blob reader
# ============================================================


@pytest.mark.asyncio
async def test_fetch_one_happy_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Successful R2 fetch + gunzip + JSON print to stdout."""
    import gzip as _gz
    from services.retrieval.agent.trace_analyzer import fetch_one

    blob = {"trace_id": "t-1", "schema_version": 1, "status": "ok"}
    body = _gz.compress(json.dumps(blob).encode("utf-8"))

    class _Store:
        async def get(self, bucket: str, key: str) -> bytes:
            assert bucket == "prbe-cust-1"
            assert key == "search-traces/2026-05-17/t-1.json.gz"
            return body

    monkeypatch.setattr(fetch_one, "get_store", lambda: _Store())
    rc = fetch_one.main(
        ["--bucket", "prbe-cust-1", "--key", "search-traces/2026-05-17/t-1.json.gz"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed == blob


def test_fetch_one_missing_blob_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """StorageNotFound → exit code 2, error on stderr."""
    from services.retrieval.agent.trace_analyzer import fetch_one
    from shared.exceptions import StorageNotFound

    class _Store:
        async def get(self, bucket: str, key: str) -> bytes:
            raise StorageNotFound(f"{bucket}/{key}")

    monkeypatch.setattr(fetch_one, "get_store", lambda: _Store())
    rc = fetch_one.main(["--bucket", "b", "--key", "k"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_fetch_one_storage_unavailable_distinct_exit_code(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """StorageUnavailable → exit code 3 (distinct from 2 so sub-agents
    can tell 'blob is gone forever' vs 'try later')."""
    from services.retrieval.agent.trace_analyzer import fetch_one
    from shared.exceptions import StorageUnavailable

    class _Store:
        async def get(self, bucket: str, key: str) -> bytes:
            raise StorageUnavailable("R2 down")

    monkeypatch.setattr(fetch_one, "get_store", lambda: _Store())
    rc = fetch_one.main(["--bucket", "b", "--key", "k"])
    assert rc == 3


def test_cli_closes_pool_on_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """If the loader raises mid-iteration, close_pool must still run so
    the next CLI invocation in the same process gets a fresh pool. The
    K8s Job only runs once per pod so this is mostly belt-and-suspenders;
    the local test catches a regression that'd bite in pytest sessions
    or future scripted multi-call usage."""
    from unittest.mock import AsyncMock as _AsyncMock
    init_mock = _AsyncMock(return_value=None)
    close_mock = _AsyncMock(return_value=None)
    monkeypatch.setattr(
        "services.retrieval.agent.trace_analyzer.__main__.init_pool", init_mock
    )
    monkeypatch.setattr(
        "services.retrieval.agent.trace_analyzer.__main__.close_pool", close_mock
    )

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated DB hang")
        yield {}  # makes it a generator

    monkeypatch.setattr(
        "services.retrieval.agent.trace_analyzer.__main__.iter_trace_blobs",
        _boom,
    )
    out_path = tmp_path / "digests.jsonl"
    with pytest.raises(RuntimeError, match="simulated DB hang"):
        cli.main(["--date", "2026-05-17", "--out", str(out_path)])
    assert init_mock.await_count == 1
    assert close_mock.await_count == 1, "close_pool must run even on exception"


# ============================================================
# loader: skip-on-failure paths (monkeypatched storage)
# ============================================================


class _StoreSpy:
    """Minimal ObjectStore stub for loader tests."""

    def __init__(self, bucket_map: dict[str, str], blobs: dict[tuple[str, str], bytes]) -> None:
        self._buckets = bucket_map
        self._blobs = blobs
        self.get_calls: list[tuple[str, str]] = []

    async def bucket_for(self, customer_id: str) -> str:
        if customer_id not in self._buckets:
            from shared.exceptions import StorageUnavailable
            raise StorageUnavailable(f"no bucket for {customer_id}")
        return self._buckets[customer_id]

    async def get(self, bucket: str, key: str) -> bytes:
        self.get_calls.append((bucket, key))
        body = self._blobs.get((bucket, key))
        if body is None:
            from shared.exceptions import StorageNotFound
            raise StorageNotFound(f"{bucket}/{key}")
        return body


@pytest.mark.asyncio
async def test_loader_skips_missing_blobs_continues_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One missing blob shouldn't halt the run — the loader logs +
    continues so the orchestrator gets all readable traces."""
    import gzip

    good_blob = _mk_blob()
    good_body = gzip.compress(json.dumps(good_blob).encode("utf-8"))
    fake_store = _StoreSpy(
        bucket_map={"cust-a": "prbe-cust-a", "cust-b": "prbe-cust-b"},
        blobs={("prbe-cust-a", "search-traces/2026-05-17/r1.json.gz"): good_body},
    )

    async def _customers() -> list[str]:
        return ["cust-a", "cust-b"]

    class _Conn:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._rows = rows

        async def fetch(self, _sql: str, *_args: Any) -> list[dict[str, Any]]:
            return self._rows

    class _CM:
        def __init__(self, conn: _Conn) -> None:
            self._conn = conn

        async def __aenter__(self) -> _Conn:
            return self._conn

        async def __aexit__(self, *_a: Any) -> None:
            return None

    from datetime import UTC, datetime
    rows_by_customer = {
        "cust-a": [
            {
                "request_id": "r1",
                "customer_id": "cust-a",
                "occurred_at": datetime(2026, 5, 17, 20, 0, tzinfo=UTC),
                "trace_blob_key": "search-traces/2026-05-17/r1.json.gz",
                "event_type": "knowledge.retrieve",
                "response_size_bytes": 1024,
                "gatherer_status": "ok",
                "tool_calls_count": 2,
                "confidence": "high",
                "cache_hit_rate": 0.6,
                "dropped_count": 0,
                "need_deeper_extensions": 0,
            }
        ],
        "cust-b": [
            {
                "request_id": "r2",
                "customer_id": "cust-b",
                "occurred_at": datetime(2026, 5, 17, 21, 0, tzinfo=UTC),
                "trace_blob_key": "search-traces/2026-05-17/r2.json.gz",
                "event_type": "knowledge.retrieve",
                "response_size_bytes": 1024,
                "gatherer_status": "ok",
                "tool_calls_count": 1,
                "confidence": "low",
                "cache_hit_rate": 0.2,
                "dropped_count": 0,
                "need_deeper_extensions": 0,
            }
        ],
    }

    def _with_tenant(customer_id: str) -> _CM:
        return _CM(_Conn(rows_by_customer.get(customer_id, [])))

    monkeypatch.setattr(loader, "_iter_customer_ids", _customers)
    monkeypatch.setattr(loader, "with_tenant", _with_tenant)
    monkeypatch.setattr(loader, "get_store", lambda: fake_store)

    yielded: list[dict[str, Any]] = []
    async for b in loader.iter_trace_blobs(date_cls(2026, 5, 17)):
        yielded.append(b)

    # cust-a's blob exists, cust-b's doesn't → only cust-a yields
    assert len(yielded) == 1
    assert yielded[0]["customer_id"] == "cust-1"  # from the fixture blob
    assert yielded[0]["_db"]["request_id"] == "r1"
    # Both customers were attempted
    assert fake_store.get_calls == [
        ("prbe-cust-a", "search-traces/2026-05-17/r1.json.gz"),
        ("prbe-cust-b", "search-traces/2026-05-17/r2.json.gz"),
    ]
