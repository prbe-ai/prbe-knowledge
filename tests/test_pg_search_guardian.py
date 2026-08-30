"""Guardian unit tests: the detection predicate and the repair's failure posture.

The integration half -- building a real pg_search index, truncating its file to
reproduce the 2026-08-25 corruption, and asserting the guardian detects and
drops it -- lives in `tests/test_pg_search_guardian_integration.py` because it
needs a live Postgres with pg_search.

What is covered here is the logic that decides WHETHER to drop, which is the
part that must never be wrong: the guardian runs unattended with DDL rights.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from engine.shared import pg_search_guardian as guardian


class _FakeConn:
    """Minimal asyncpg.Connection stand-in.

    `fetch` returns queued row batches; `execute` records statements so a test
    can assert what DDL was attempted without a database.
    """

    def __init__(self, rows: list[list[dict[str, Any]]] | None = None) -> None:
        self._rows = rows or []
        self.executed: list[str] = []
        self.execute_error: Exception | None = None
        # `to_regclass` result for analyze_tables: None models a table that
        # does not exist.
        self.fetchval_result: Any = "present"

    async def fetch(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return self._rows.pop(0) if self._rows else []

    async def fetchval(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.fetchval_result

    async def execute(self, sql: str, *_args: Any) -> None:
        self.executed.append(sql)
        if self.execute_error is not None:
            raise self.execute_error

    def transaction(self) -> Any:
        conn = self

        class _Txn:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

        del conn
        return _Txn()


def _async_return(value: Any):
    """An async stand-in that ignores its arguments and returns `value`."""

    async def _fn(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return _fn


class _FakePool:
    """`get_pool()` stand-in whose `acquire()` yields a fake connection."""

    def acquire(self) -> Any:
        class _Ctx:
            async def __aenter__(self) -> Any:
                return _FakeConn()

            async def __aexit__(self, *_a: Any) -> bool:
                return False

        return _Ctx()


def _row(
    index_name: str = "idx_chunks_bm25_v2",
    *,
    table: str = "chunks",
    am: str = "bm25",
    size: int = 0,
    reltuples: float = 774109.0,
) -> dict[str, Any]:
    return {
        "index_name": index_name,
        "table_name": table,
        "access_method": am,
        "index_bytes": size,
        "table_reltuples": reltuples,
    }


# ============================================================
# Detection: all three conjuncts have to hold
# ============================================================

async def test_detects_the_incident_shape() -> None:
    """A valid, allowlisted, 0-byte bm25 index on a populated table.

    This is exactly what the 2026-08-25 failover produced, and the state in
    which every statement planning against `chunks` raises XX001."""
    conn = _FakeConn([[_row()]])
    broken = await guardian.find_broken_pg_search_indexes(conn)  # type: ignore[arg-type]
    assert [b["index"] for b in broken] == ["idx_chunks_bm25_v2"]


async def test_skips_an_index_no_contract_declares() -> None:
    """The allowlist is the outer bound on what may be dropped.

    An index this repository never declared is not ours to drop, however
    broken it looks -- it could belong to another application sharing the
    database, or to a migration in flight."""
    conn = _FakeConn([[_row("idx_someone_elses_bm25")]])
    assert await guardian.find_broken_pg_search_indexes(conn) == []  # type: ignore[arg-type]


async def test_skips_an_empty_table() -> None:
    """An index on an empty table is legitimately 0 bytes.

    Without this conjunct the guardian would drop the healthy index on a fresh
    or truncated `chunks` -- causing the outage it exists to prevent, on the
    one instance least able to absorb it."""
    conn = _FakeConn([[_row(reltuples=0)]])
    assert await guardian.find_broken_pg_search_indexes(conn) == []  # type: ignore[arg-type]


async def test_unanalyzed_table_is_treated_as_nonempty() -> None:
    """`reltuples = -1` means "never analyzed", not "empty".

    A freshly promoted standby has no statistics at all, which is precisely the
    instance the guardian is for. Reading -1 as empty would disable it there."""
    conn = _FakeConn([[_row(reltuples=-1.0)]])
    broken = await guardian.find_broken_pg_search_indexes(conn)  # type: ignore[arg-type]
    assert len(broken) == 1


async def test_matches_the_renamed_access_method() -> None:
    """pg_search 0.25 renamed the access method `bm25` -> `paradedb`.

    Matching on the index NAME instead would go silently blind after that
    upgrade, which is the same class of quiet failure the guardian exists to
    end. Both names stay live."""
    assert "paradedb" in guardian.PG_SEARCH_ACCESS_METHODS
    conn = _FakeConn([[_row(am="paradedb")]])
    broken = await guardian.find_broken_pg_search_indexes(conn)  # type: ignore[arg-type]
    assert broken[0]["access_method"] == "paradedb"


async def test_allowlist_is_derived_from_index_contracts() -> None:
    """Derived, not hardcoded, so declaring a new pg_search index in the one
    place this codebase already declares indexes is enough to cover it."""
    assert "idx_chunks_bm25_v2" in guardian.ALLOWED_INDEX_NAMES


# ============================================================
# Debris: alert-only, never dropped
# ============================================================

async def test_invalid_debris_is_reported_not_dropped() -> None:
    """A failed CONCURRENTLY build leaves an invalid index behind.

    It is inert -- the planner will not choose it -- so the guardian reports it
    and stops. An in-progress build looks identical from the catalog, and
    dropping one out from under an attended rebuild would make the guardian the
    cause of the incident."""
    conn = _FakeConn([[{"index_name": "idx_chunks_bm25_v2"}]])
    assert await guardian.find_invalid_index_debris(conn) == ["idx_chunks_bm25_v2"]  # type: ignore[arg-type]
    assert conn.executed == []


# ============================================================
# Drop: bounded, allowlisted, and it yields rather than queues
# ============================================================

async def test_drop_sets_a_lock_timeout_before_the_ddl() -> None:
    """The guardian must never queue ahead of live traffic.

    An unbounded DROP waits for ACCESS EXCLUSIVE, and a queued exclusive lock
    request blocks every reader that arrives behind it -- converting a degraded
    search into a stalled database."""
    conn = _FakeConn()
    assert await guardian.drop_broken_index(conn, "idx_chunks_bm25_v2") is True  # type: ignore[arg-type]
    assert any("lock_timeout" in stmt for stmt in conn.executed)
    assert any("DROP INDEX" in stmt for stmt in conn.executed)


async def test_drop_skips_on_lock_timeout_rather_than_failing() -> None:
    """A busy table is not an error. The next tick retries, and nothing is
    worse off meanwhile than it already was."""
    conn = _FakeConn()
    conn.execute_error = asyncpg.LockNotAvailableError("lock timeout")
    assert await guardian.drop_broken_index(conn, "idx_chunks_bm25_v2") is False  # type: ignore[arg-type]


async def test_drop_refuses_an_index_outside_the_allowlist() -> None:
    """Defence in depth: the caller already filters, so reaching this raise
    means a bug upstream -- and the failure should be loud, not a DROP."""
    conn = _FakeConn()
    with pytest.raises(ValueError, match="allowlist"):
        await guardian.drop_broken_index(conn, "idx_not_ours")  # type: ignore[arg-type]
    assert conn.executed == []


async def test_analyze_refuses_a_suspicious_identifier() -> None:
    """Table names are interpolated (ANALYZE takes no bind parameter), so the
    identifier is validated rather than trusted."""
    conn = _FakeConn()
    with pytest.raises(ValueError, match="suspicious"):
        await guardian.analyze_tables(conn, ['chunks"; DROP TABLE documents; --'])  # type: ignore[arg-type]


# ============================================================
# Alerting must never be able to block the repair
# ============================================================

def test_capture_never_raises_even_when_logging_is_broken(monkeypatch) -> None:
    """`capture` swallows EVERYTHING, including its own logging.

    A regression test with a story: the first version logged before entering
    its try block, on the reasoning that a local structlog call cannot fail.
    It could -- the kwarg was named `event`, which structlog already binds from
    the positional message, so every call raised TypeError. The guardian
    repaired the database, died on the alert, exited nonzero, and never
    recorded its timeline. Unit tests missed it because none of them called
    `capture` for real; a smoke run against a live database found it in one
    tick.

    So the property under test is not "the happy path logs" but "no input and
    no broken dependency can make this raise into its caller"."""
    from engine.shared import ops_alert

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise TypeError("logger is broken")

    monkeypatch.setattr(ops_alert.log, "info", _explode)
    monkeypatch.setattr(ops_alert.log, "warning", _explode)
    assert ops_alert.capture("kb_pg_search_index_repaired", {"indexes": ["x"]}) is False


def test_capture_does_not_collide_with_structlogs_event_key() -> None:
    """The event name rides `alert=`, not `event=`.

    structlog binds `event` from the positional message, so a kwarg of the
    same name is a guaranteed TypeError on every single call -- not a rare
    edge. Asserted on the real logger rather than a mock, because a mock would
    happily accept the colliding kwarg and prove nothing."""
    from engine.shared import ops_alert

    assert ops_alert.capture("kb_pg_search_promotion_detected", {"timeline_id": 2}) is False


def test_capture_is_a_noop_without_an_api_key(monkeypatch) -> None:
    """Self-host and local runs have no analytics configured, and that is not
    a failure -- the guardian still repairs and still logs locally."""
    from engine.shared import ops_alert

    monkeypatch.delenv("POSTHOG_API_KEY", raising=False)
    assert ops_alert.capture("kb_pg_search_index_repaired") is False


async def test_analyze_skips_a_table_that_does_not_exist() -> None:
    """A missing table must not abort the tick.

    The tick ENDS by recording the timeline. A tick that raises before that
    point never advances the marker, so the next one re-detects the same
    promotion and alerts again -- every minute, forever. A missing table (a
    self-host schema, a rename, a fresh database) would turn one alert into an
    unbounded storm, which is how a signal stops being read.

    Found by a smoke run against a database without `documents`: the tick
    failed, the timeline stayed at 0, and the promotion alert repeated on
    every subsequent tick."""
    conn = _FakeConn()
    conn.fetchval_result = None  # to_regclass -> table absent
    await guardian.analyze_tables(conn, ["documents"])  # type: ignore[arg-type]
    assert conn.executed == []


async def test_analyze_failure_does_not_propagate() -> None:
    """ANALYZE is an optimization on top of the repair, not the repair.

    The drop has already restored service by the time this runs, so a failure
    here is worth a log and nothing more -- certainly not worth losing the
    timeline record that keeps the alert from repeating."""
    conn = _FakeConn()
    conn.execute_error = RuntimeError("statement timeout")
    await guardian.analyze_tables(conn, ["chunks"])  # type: ignore[arg-type]


async def test_an_unannounced_repair_fails_the_job(monkeypatch) -> None:
    """A repair nobody was told about must exit nonzero.

    This is the review finding that mattered most: the target cluster's
    `engine-secrets` carries no POSTHOG_API_KEY (verified 2026-08-26), so
    `capture` short-circuits and every repair would otherwise exit 0, land
    under `successfulJobsHistoryLimit: 1`, and vanish within the minute. BM25
    would be silently dropped and stay off with no signal to anyone -- which is
    worse than not repairing, because at least a broken database is noticed.

    Exiting nonzero routes the fact through the CronJob's FAILED history, which
    the chart keeps ten of precisely to serve as the fallback channel when the
    alerting path is what is unavailable."""
    from scripts import cron_pg_search_guardian as cron

    monkeypatch.setattr(cron, "capture", lambda *_a, **_k: False)
    monkeypatch.setattr(cron, "find_broken_pg_search_indexes", _async_return([{"index": "i"}]))
    monkeypatch.setattr(cron, "find_invalid_index_debris", _async_return([]))
    monkeypatch.setattr(cron, "current_timeline_id", _async_return(1))
    monkeypatch.setattr(cron, "read_last_timeline", _async_return(1))
    monkeypatch.setattr(cron, "drop_broken_index", _async_return(True))
    monkeypatch.setattr(cron, "record_timeline", _async_return(None))
    monkeypatch.setattr(cron, "get_pool", lambda: _FakePool())

    assert await cron.run_once() == 1


async def test_a_delivered_repair_exits_clean(monkeypatch) -> None:
    """The same repair, announced, is a success. Red must mean "a change was
    made and nobody was told" -- not merely "a change was made"."""
    from scripts import cron_pg_search_guardian as cron

    monkeypatch.setattr(cron, "capture", lambda *_a, **_k: True)
    monkeypatch.setattr(cron, "find_broken_pg_search_indexes", _async_return([{"index": "i"}]))
    monkeypatch.setattr(cron, "find_invalid_index_debris", _async_return([]))
    monkeypatch.setattr(cron, "current_timeline_id", _async_return(1))
    monkeypatch.setattr(cron, "read_last_timeline", _async_return(1))
    monkeypatch.setattr(cron, "drop_broken_index", _async_return(True))
    monkeypatch.setattr(cron, "record_timeline", _async_return(None))
    monkeypatch.setattr(cron, "get_pool", lambda: _FakePool())

    assert await cron.run_once() == 0


async def test_an_undelivered_promotion_alert_does_not_fail_the_job(monkeypatch) -> None:
    """Only a REPAIR earns a red job.

    A promotion or debris alert that goes undelivered changed nothing in the
    database. Failing on those too would make red mean "something happened"
    rather than "a change was made and nobody was told", and the distinction is
    the whole value of the signal."""
    from scripts import cron_pg_search_guardian as cron

    monkeypatch.setattr(cron, "capture", lambda *_a, **_k: False)
    monkeypatch.setattr(cron, "find_broken_pg_search_indexes", _async_return([]))
    monkeypatch.setattr(cron, "find_invalid_index_debris", _async_return(["debris"]))
    monkeypatch.setattr(cron, "current_timeline_id", _async_return(2))
    monkeypatch.setattr(cron, "read_last_timeline", _async_return(1))  # promoted
    monkeypatch.setattr(cron, "analyze_tables", _async_return(None))
    monkeypatch.setattr(cron, "record_timeline", _async_return(None))
    monkeypatch.setattr(cron, "get_pool", lambda: _FakePool())

    assert await cron.run_once() == 0


# ============================================================
# Absence: the false all-clear the guardian's own repair produces
# ============================================================

async def test_absent_required_index_is_reported() -> None:
    """After the guardian drops a broken index, the index is simply GONE --
    and the broken-index detector, which can only see indexes that exist,
    reads `broken_count: 0`. Observed live 2026-08-26: every tick after the
    04:14 repair reported healthy while BM25 returned nothing. Absence of a
    required index must be its own signal."""
    conn = _FakeConn()

    async def _regclass(_sql: str, name: str) -> Any:
        return None if name == "idx_chunks_bm25_v2" else "present"

    conn.fetchval = _regclass  # type: ignore[method-assign]
    assert await guardian.find_absent_required_indexes(conn) == ["idx_chunks_bm25_v2"]  # type: ignore[arg-type]


async def test_absent_table_is_the_migrations_problem_not_ours() -> None:
    """A required index over a table that does not exist is a fresh or
    part-migrated database, not a search outage. Reporting it would page
    for every `docker compose up`."""
    conn = _FakeConn()

    async def _regclass(_sql: str, _name: str) -> Any:
        return None  # neither index nor table exists

    conn.fetchval = _regclass  # type: ignore[method-assign]
    assert await guardian.find_absent_required_indexes(conn) == []  # type: ignore[arg-type]


async def test_known_absent_guards_against_a_scalar() -> None:
    """A malformed scalar must not become a set of characters.

    `frozenset("present")` is {'p','r','e','s','n','t'} -- the same
    string-iteration trap the lost_channels forwarding shipped with. The
    guard is on the CONTAINER type."""
    conn = _FakeConn()
    conn.fetchval_result = "present"  # a scalar, not a text[]
    assert await guardian.read_known_absent(conn) == frozenset()  # type: ignore[arg-type]


async def test_absence_alerts_fire_on_transitions_not_state(monkeypatch) -> None:
    """An absence persists for HOURS by design -- rebuilds are attended --
    so alerting on the state would fire every minute for the whole window.
    Steady-state known absence: no capture. Newly restored: one capture."""
    from scripts import cron_pg_search_guardian as cron

    events: list[str] = []
    monkeypatch.setattr(cron, "capture", lambda name, *_a, **_k: events.append(name) or True)
    monkeypatch.setattr(cron, "find_broken_pg_search_indexes", _async_return([]))
    monkeypatch.setattr(cron, "find_invalid_index_debris", _async_return([]))
    monkeypatch.setattr(cron, "current_timeline_id", _async_return(4))
    monkeypatch.setattr(cron, "read_last_timeline", _async_return(4))
    monkeypatch.setattr(cron, "record_timeline", _async_return(None))
    monkeypatch.setattr(cron, "record_known_absent", _async_return(None))
    monkeypatch.setattr(cron, "get_pool", lambda: _FakePool())

    # Steady state: absent, and already known absent -> silence.
    monkeypatch.setattr(cron, "find_absent_required_indexes", _async_return(["idx_chunks_bm25_v2"]))
    monkeypatch.setattr(cron, "read_known_absent", _async_return(frozenset({"idx_chunks_bm25_v2"})))
    assert await cron.run_once() == 0
    assert events == []

    # Rebuilt: newly restored -> exactly one closing alert.
    monkeypatch.setattr(cron, "find_absent_required_indexes", _async_return([]))
    assert await cron.run_once() == 0
    assert events == ["kb_pg_search_index_restored"]


async def test_new_absence_alerts_once(monkeypatch) -> None:
    """The transition INTO absence alerts; exit stays 0 -- nothing was
    changed by the guardian, so red would dilute what red means."""
    from scripts import cron_pg_search_guardian as cron

    events: list[str] = []
    monkeypatch.setattr(cron, "capture", lambda name, *_a, **_k: events.append(name) or True)
    monkeypatch.setattr(cron, "find_broken_pg_search_indexes", _async_return([]))
    monkeypatch.setattr(cron, "find_invalid_index_debris", _async_return([]))
    monkeypatch.setattr(cron, "current_timeline_id", _async_return(4))
    monkeypatch.setattr(cron, "read_last_timeline", _async_return(4))
    monkeypatch.setattr(cron, "record_timeline", _async_return(None))
    monkeypatch.setattr(cron, "record_known_absent", _async_return(None))
    monkeypatch.setattr(cron, "get_pool", lambda: _FakePool())
    monkeypatch.setattr(cron, "find_absent_required_indexes", _async_return(["idx_chunks_bm25_v2"]))
    monkeypatch.setattr(cron, "read_known_absent", _async_return(frozenset()))

    assert await cron.run_once() == 0
    assert events == ["kb_pg_search_index_absent"]


# ============================================================
# Post-promotion prewarm
# ============================================================


class _PrewarmConn(_FakeConn):
    """FakeConn whose fetchval dispatches on the statement, because prewarm
    makes three DIFFERENT fetchval calls (to_regproc, to_regclass,
    pg_prewarm) and a single canned result would let a broken dispatch pass."""

    def __init__(self, *, extension_present: bool = True,
                 index_present: bool = True) -> None:
        super().__init__()
        self.extension_present = extension_present
        self.index_present = index_present
        self.prewarmed: list[str] = []

    async def fetchval(self, sql: str, *args: Any, **_kwargs: Any) -> Any:
        if "to_regproc" in sql:
            return "pg_prewarm" if self.extension_present else None
        if "to_regclass" in sql:
            return args[0] if self.index_present else None
        if "pg_prewarm" in sql:
            self.prewarmed.append(args[0])
            return 12345
        raise AssertionError(f"unexpected fetchval: {sql}")


async def test_prewarm_reads_each_index() -> None:
    conn = _PrewarmConn()
    await guardian.prewarm_indexes(conn, ["idx_a", "idx_b"])  # type: ignore[arg-type]
    assert conn.prewarmed == ["idx_a", "idx_b"]


async def test_prewarm_skips_entirely_without_the_extension() -> None:
    """A self-host that never ran migration 0123 must get one log line, not
    an error and not a per-index probe storm."""
    conn = _PrewarmConn(extension_present=False)
    await guardian.prewarm_indexes(conn, ["idx_a"])  # type: ignore[arg-type]
    assert conn.prewarmed == []


async def test_prewarm_skips_a_missing_index_and_continues() -> None:
    class _FirstMissing(_PrewarmConn):
        async def fetchval(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            if "to_regclass" in sql and args and args[0] == "idx_missing":
                return None
            return await super().fetchval(sql, *args, **kwargs)

    conn = _FirstMissing()
    await guardian.prewarm_indexes(conn, ["idx_missing", "idx_b"])  # type: ignore[arg-type]
    assert conn.prewarmed == ["idx_b"]


async def test_prewarm_failure_does_not_abort_the_batch() -> None:
    """Same posture as analyze_tables: best-effort, because a raise here
    would stop the tick before it records the timeline."""
    class _FirstBlows(_PrewarmConn):
        async def fetchval(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            if "pg_prewarm" in sql and args and args[0] == "idx_a":
                raise RuntimeError("disk went away")
            return await super().fetchval(sql, *args, **kwargs)

    conn = _FirstBlows()
    await guardian.prewarm_indexes(conn, ["idx_a", "idx_b"])  # type: ignore[arg-type]
    assert conn.prewarmed == ["idx_b"]


async def test_prewarm_refuses_a_suspicious_identifier() -> None:
    conn = _PrewarmConn()
    with pytest.raises(ValueError, match="suspicious"):
        await guardian.prewarm_indexes(conn, ['idx"; DROP TABLE chunks; --'])  # type: ignore[arg-type]
