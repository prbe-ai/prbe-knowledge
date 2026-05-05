# Synth Tool — Plan 2: Output Layer + Templated Narrative Path

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Output Layer to the synth tool so `python -m scripts.synth run --profile <yaml>` emits deterministic source-shaped Slack and Notion JSON files that round-trip through the existing connectors, with an `--integrate` flag to also write to R2 and insert into `ingestion_queue`.

**Architecture:** Plan 2 sits above Plan 1's immutable `WorldModel`. A new `ScenarioRunner` walks two templated archetypes (`STANDUP_UPDATE`, `ON_CALL_HANDOFF`), producing `ScenarioSpec` → `DocSpec` → `SynthDoc` objects. Source wrappers (`slack.py`, `notion.py`) serialize each `SynthDoc` to a byte-exact envelope matching what `SlackConnector.parse_webhook_event` and `NotionConnector.parse_webhook_event` already consume. `IngestionWriter` writes local files in both modes and additionally calls `bucket.put` + `ingestion_queue` INSERT in `--integrate` mode. A `TenantBootstrap` module handles idempotent tenant init and prefix-guarded teardown.

**Tech Stack:** Python 3.12, frozen dataclasses, structlog, orjson, httpx, pytest+pytest-asyncio, ruff (E/F/W/I/UP/B/SIM/RUF, line-length 100). Postgres via asyncpg for `--integrate` mode. `shared.storage.ObjectStore` for R2 operations; Plan 2 adds a per-key `delete` method to it (Task 13).

---

## File structure

**New files:**

```
scripts/synth/
  archetypes/
    __init__.py                         # empty package marker
    base.py                             # Source, Cadence, Category, ValidatorLevel, Archetype, DocSpec, ScenarioSpec
    library.py                          # ARCHETYPE_LIBRARY, BUILDERS, get_active()
    standup.py                          # STANDUP_UPDATE + build_standup_specs()
    oncall.py                           # ON_CALL_HANDOFF + build_oncall_specs()
  output/
    __init__.py                         # empty package marker
    base.py                             # SynthDoc frozen dataclass + Storage protocol
    slack.py                            # wrap(doc) -> bytes for event_callback envelope
    notion.py                           # wrap(doc) -> bytes for page.updated envelope
    writer.py                           # IngestionWriter (local mode Task 11, integrate Task 14)
    eval_artifacts.py                   # EvalArtifactWriter (manifest.json, docs_index.jsonl, etc.)
  ownership.py                          # OwnershipIndex + build_ownership_index()
  scenarios.py                          # run_scenarios(), TimeWindow, working_days(), weekly_mondays()
  validator.py                          # validate_name_only(), Violation, _extract_proper_nouns()
  bootstrap.py                          # init_tenant(), clean_tenant(), CUSTOMER_OWNED_TABLES

tests/synth/
  test_archetype_base.py               # Task 1
  test_output_base.py                  # Task 2
  test_output_slack_wrapper.py         # Task 3
  test_output_notion_wrapper.py        # Task 4
  test_validator.py                    # Task 5
  test_ownership.py                    # Task 6
  test_archetype_standup.py            # Task 7
  test_archetype_oncall.py             # Task 8
  test_archetype_library.py            # Task 9
  test_scenarios.py                    # Task 10
  test_ingestion_writer.py             # Tasks 11 + 14
  test_eval_artifacts.py               # Task 12
  test_bootstrap.py                    # Task 13
  test_cli_plan2.py                    # Task 15
  test_e2e_run.py                      # Task 16
```

**Modified files:**

```
scripts/synth/cli.py        # ADD init/run/clean subcommands (extract unchanged)
shared/storage.py           # ADD delete(bucket, key) -> None method (Task 13)
```

---

## Conventions

- Every new file starts with `from __future__ import annotations`.
- Every dataclass that is hashable or cached uses `frozen=True`. Mutable working buffers use plain `@dataclass`.
- `tuple[X, ...]` for all immutable sequences on frozen dataclasses. `dict[str, tuple[str, ...]]` on frozen dataclasses is permitted (same pattern as `WorldModel.sha_set`).
- All file paths in code are `pathlib.Path`, never `str`.
- Logging via `shared.logging.get_logger(__name__)` — no `print` in library code. CLI entrypoint may use `print` for user-visible output.
- Async only where it pays: `IngestionWriter.write`, `IngestionWriter.close`, `init_tenant`, `clean_tenant` are async. All archetype builders, wrappers, validator, and ownership index are sync.
- Subprocess calls use `subprocess.run(..., check=True, capture_output=True, text=True)`.
- Tests use `pytest-asyncio` auto mode (configured in `pyproject.toml`).
- Commit per task. Conventional-commit style: `feat(synth): <subject>`.
- Every commit message ends with: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- Mid-file imports where needed (e.g., late `import re`) are marked `# noqa: E402`.
- `ruff check` enforced; `ruff format` not enforced.
- `# type: ignore[...]` is acceptable on known structural-typing stubs.

---

## Task 1: Archetype base dataclasses

**Files:**
- Create: `scripts/synth/archetypes/__init__.py`
- Create: `scripts/synth/archetypes/base.py`
- Test: `tests/synth/test_archetype_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_archetype_base.py`:

```python
"""Smoke tests for archetype base dataclasses and enums."""

from __future__ import annotations

import pytest
from datetime import datetime, UTC

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    DocSpec,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)


def test_source_enum_values() -> None:
    assert Source.SLACK == "slack"
    assert Source.NOTION == "notion"
    assert Source.GRANOLA == "granola"
    assert Source.GITHUB == "github"
    assert Source.LINEAR == "linear"
    assert Source.SENTRY == "sentry"
    assert Source.CLAUDE_CODE == "claude_code"


def test_cadence_enum_values() -> None:
    assert Cadence.DAILY == "daily"
    assert Cadence.WEEKLY == "weekly"
    assert Cadence.BIWEEKLY == "biweekly"
    assert Cadence.MONTHLY == "monthly"
    assert Cadence.SPRINT == "sprint"
    assert Cadence.AD_HOC == "ad_hoc"


def test_archetype_construct_and_frozen() -> None:
    a = Archetype(
        name="STANDUP_UPDATE",
        category=Category.RECURRING,
        cadence=Cadence.DAILY,
        sources_used=(Source.SLACK,),
        cast_size=(1, 1),
        needs_planner_call=False,
        validator_level=ValidatorLevel.NAME_ONLY,
    )
    assert a.name == "STANDUP_UPDATE"
    assert a.cadence == Cadence.DAILY
    with pytest.raises(Exception):
        object.__setattr__(a, "name", "OTHER")  # type: ignore[attr-defined]


def test_doc_spec_construct_and_frozen() -> None:
    ts = datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC)
    doc = DocSpec(
        id="scn-standup-gh-alice-2026-05-01-slack-0",
        source=Source.SLACK,
        occurred_at=ts,
        channel="#standup",
        page_section=None,
        text="Yesterday: shipped payments. Today: auth-service - fix token refresh. Blockers: none.",
        thread_parent_id=None,
        personas=("gh:alice",),
        services_mentioned=("payments",),
    )
    assert doc.channel == "#standup"
    with pytest.raises(Exception):
        object.__setattr__(doc, "text", "mutated")  # type: ignore[attr-defined]


def test_scenario_spec_construct() -> None:
    ts = datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC)
    doc = DocSpec(
        id="scn-standup-gh-alice-2026-05-01-slack-0",
        source=Source.SLACK,
        occurred_at=ts,
        channel="#standup",
        page_section=None,
        text="Yesterday: shipped auth. Today: payments - fix retry. Blockers: none.",
        thread_parent_id=None,
        personas=("gh:alice",),
        services_mentioned=("auth",),
    )
    spec = ScenarioSpec(
        id="scn-standup-gh-alice-2026-05-01",
        archetype_name="STANDUP_UPDATE",
        instance_ts=ts,
        cast=("gh:alice",),
        affected_services=("auth",),
        doc_specs=(doc,),
    )
    assert len(spec.doc_specs) == 1
    assert spec.cast == ("gh:alice",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_archetype_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.archetypes'`

- [ ] **Step 3: Implement the module**

Create `scripts/synth/archetypes/__init__.py` (empty):

```python
```

Create `scripts/synth/archetypes/base.py`:

```python
"""Archetype base dataclasses and enums.

These types are the shared vocabulary for the scenario layer. Every
archetype builder consumes WorldModel + OwnershipIndex and emits
ScenarioSpec objects composed of DocSpec leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Source(StrEnum):
    SLACK = "slack"
    NOTION = "notion"
    GRANOLA = "granola"
    GITHUB = "github"
    LINEAR = "linear"
    SENTRY = "sentry"
    CLAUDE_CODE = "claude_code"


class Cadence(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    SPRINT = "sprint"
    AD_HOC = "ad_hoc"


class Category(StrEnum):
    RECURRING = "recurring"
    PLOT = "plot"
    SLOW_BURN = "slow_burn"


class ValidatorLevel(StrEnum):
    STRICT = "strict"
    NAME_ONLY = "name_only"
    NONE = "none"


@dataclass(frozen=True)
class Archetype:
    name: str
    category: Category
    cadence: Cadence
    sources_used: tuple[Source, ...]
    cast_size: tuple[int, int]       # (min, max) personas per scenario
    needs_planner_call: bool         # False for all Plan 2 archetypes
    validator_level: ValidatorLevel  # NAME_ONLY for all Plan 2 archetypes


@dataclass(frozen=True)
class DocSpec:
    """Specification for a single synthetic document before wrapping."""
    id: str
    source: Source
    occurred_at: datetime
    channel: str | None           # Slack channel name (e.g. "#standup")
    page_section: str | None      # Notion section path (e.g. "Engineering > On-call rotation")
    text: str
    thread_parent_id: str | None  # Slack thread parent doc_spec id, if this is a reply
    personas: tuple[str, ...]     # canonical_ids involved
    services_mentioned: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioSpec:
    """One instance of an archetype producing one or more DocSpecs."""
    id: str
    archetype_name: str
    instance_ts: datetime
    cast: tuple[str, ...]           # canonical_ids in this scenario
    affected_services: tuple[str, ...]
    doc_specs: tuple[DocSpec, ...]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_archetype_base.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/archetypes/__init__.py scripts/synth/archetypes/base.py tests/synth/test_archetype_base.py
git commit -m "$(cat <<'EOF'
feat(synth): archetype base dataclasses — Source, Cadence, Category, DocSpec, ScenarioSpec

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: SynthDoc + Storage protocol

**Files:**
- Create: `scripts/synth/output/__init__.py`
- Create: `scripts/synth/output/base.py`
- Test: `tests/synth/test_output_base.py`

**Key design note:** `shared.storage.ObjectStore` has `put(bucket, key, body, content_type) -> ObjectLocation`, `list_keys(bucket, prefix) -> list[str]`, `ensure_bucket(bucket) -> None`, and `bucket_for(customer_id) -> str`. It does NOT have a per-key `delete` method — only `delete_bucket_recursive`. Task 13 adds `delete(bucket, key) -> None` to `shared/storage.py`. The `Storage` protocol below includes `delete` so callers can depend on it after Task 13.

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_output_base.py`:

```python
"""SynthDoc construction and Storage protocol satisfaction."""

from __future__ import annotations

import asyncio
from datetime import datetime, UTC

import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import Storage, SynthDoc


def _make_doc() -> SynthDoc:
    return SynthDoc(
        id="scn-standup-gh-alice-2026-05-01-slack-0",
        source=Source.SLACK,
        source_event_id="scn-standup-gh-alice-2026-05-01-slack-0",
        text="Yesterday: shipped payments. Today: auth-service - fix token refresh. Blockers: none.",
        occurred_at=datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC),
        channel="#standup",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-standup-gh-alice-2026-05-01",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments", "auth-service"),
        priority=100,
    )


def test_synthdoc_construct() -> None:
    doc = _make_doc()
    assert doc.source == Source.SLACK
    assert doc.priority == 100


def test_synthdoc_frozen() -> None:
    doc = _make_doc()
    with pytest.raises(Exception):
        object.__setattr__(doc, "text", "mutated")  # type: ignore[attr-defined]


def test_storage_protocol_satisfied_by_stub() -> None:
    """A plain class with the right methods satisfies the Storage protocol at runtime."""

    class _FakeStore:
        async def put(self, bucket: str, key: str, data: bytes) -> None:
            pass

        async def list_keys(self, bucket: str, prefix: str) -> list[str]:
            return []

        async def delete(self, bucket: str, key: str) -> None:
            pass

        def bucket_for(self, customer_id: str) -> str:
            return f"bucket-{customer_id}"

        async def ensure_bucket(self, bucket: str) -> None:
            pass

    store = _FakeStore()
    # Verify it satisfies the protocol structurally (runtime_checkable)
    assert isinstance(store, Storage)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_output_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.output'`

- [ ] **Step 3: Implement the module**

Create `scripts/synth/output/__init__.py` (empty):

```python
```

Create `scripts/synth/output/base.py`:

```python
"""SynthDoc — the canonical output unit — and the Storage protocol.

SynthDoc is what every archetype builder ultimately produces (via DocSpec
materialization in ScenarioRunner). Source wrappers consume SynthDoc and
emit bytes. IngestionWriter consumes SynthDoc + bytes and writes files/R2.

Storage is a structural protocol matching shared.storage.ObjectStore
(after Task 13 adds the delete method). Tests can pass a fake stub;
production passes the real ObjectStore instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from scripts.synth.archetypes.base import Source


@dataclass(frozen=True)
class SynthDoc:
    """One synthetic document ready for wrapping and writing."""
    id: str
    source: Source
    source_event_id: str
    text: str
    occurred_at: datetime
    channel: str | None          # Slack channel (e.g. "#standup"); None for Notion
    page_id: str | None          # Notion page id; None for Slack
    thread_parent_id: str | None # Slack thread parent source_event_id; None for root
    scenario_id: str
    archetype: str
    personas: tuple[str, ...]
    services_mentioned: tuple[str, ...]
    priority: int = field(default=100)


@runtime_checkable
class Storage(Protocol):
    """Structural protocol for object storage.

    Matches shared.storage.ObjectStore after Task 13 adds delete().
    Any class implementing these five methods satisfies the protocol.
    """

    async def put(self, bucket: str, key: str, data: bytes) -> None:
        ...

    async def list_keys(self, bucket: str, prefix: str) -> list[str]:
        ...

    async def delete(self, bucket: str, key: str) -> None:
        ...

    def bucket_for(self, customer_id: str) -> str:
        ...

    async def ensure_bucket(self, bucket: str) -> None:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_output_base.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/output/__init__.py scripts/synth/output/base.py tests/synth/test_output_base.py
git commit -m "$(cat <<'EOF'
feat(synth): SynthDoc frozen dataclass + Storage structural protocol

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: SlackWrapper

**Files:**
- Create: `scripts/synth/output/slack.py`
- Test: `tests/synth/test_output_slack_wrapper.py`

**Connector interface understood from `services/ingestion/handlers/slack.py`:** `SlackConnector.parse_webhook_event(customer_id, headers, raw_payload)` reads `raw_payload["event"]["type"]`, `raw_payload["event"]["channel"]`, `raw_payload["event"]["user"]`, `raw_payload["event"]["text"]`, `raw_payload["event"]["ts"]`, `raw_payload["event"]["thread_ts"]` (optional), and `raw_payload["team_id"]`. It returns a `WebhookParseResult` with `source_event_id = f"{channel}:{ts}"` for plain messages and `parse_hint` carrying `channel`, `ts`, `thread_ts`.

The fixture at `fixtures/slack/message_simple.json` has the exact envelope shape:
```json
{"token": "verification", "team_id": "T_TEST", "api_app_id": "A_TEST",
 "event": {"type": "message", "channel": "C_PAYMENTS", "user": "U_ALICE",
            "text": "...", "ts": "1713628800.000100", ...},
 "type": "event_callback", "event_id": "Ev_TEST", "event_time": 1713628800}
```

The wrapper must produce an envelope parseable by `parse_webhook_event`. Round-trip test: parse the wrapper output and recover `(channel, text, ts)`.

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_output_slack_wrapper.py`:

```python
"""SlackWrapper round-trip tests.

The wrapper must produce a byte payload that SlackConnector.parse_webhook_event
accepts. We test the parse_hint fields directly without importing the full
connector (which has heavy deps) — instead we parse the JSON ourselves and
assert the shape matches what the connector expects.
"""

from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.slack import wrap


_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "slack" / "message_simple.json"


def _make_slack_doc(
    *,
    channel: str = "#standup",
    text: str = "Yesterday: shipped payments. Today: auth - fix retry. Blockers: none.",
    thread_parent_id: str | None = None,
    occurred_at: datetime | None = None,
) -> SynthDoc:
    ts = occurred_at or datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC)
    return SynthDoc(
        id="scn-standup-gh-alice-2026-05-01-slack-0",
        source=Source.SLACK,
        source_event_id="scn-standup-gh-alice-2026-05-01-slack-0",
        text=text,
        occurred_at=ts,
        channel=channel,
        page_id=None,
        thread_parent_id=thread_parent_id,
        scenario_id="scn-standup-gh-alice-2026-05-01",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments", "auth"),
        priority=100,
    )


def test_wrap_produces_valid_json() -> None:
    doc = _make_slack_doc()
    raw = wrap(doc)
    payload = json.loads(raw)
    assert payload["type"] == "event_callback"
    assert payload["event"]["type"] == "message"


def test_wrap_recovers_channel_text_ts() -> None:
    """Parsed envelope yields the same channel, text, and ts the doc had."""
    doc = _make_slack_doc(channel="#standup", text="hello world")
    payload = json.loads(wrap(doc))
    event = payload["event"]
    assert event["channel"] == "#standup"
    assert event["text"] == "hello world"
    # ts must be the unix timestamp of occurred_at as "<int>.<6digits>" string
    expected_ts = f"{int(doc.occurred_at.timestamp())}.000000"
    assert event["ts"] == expected_ts


def test_wrap_thread_ts_present_when_reply() -> None:
    parent_id = "scn-oncall-2026-05-05-slack-0"
    doc = _make_slack_doc(thread_parent_id=parent_id)
    payload = json.loads(wrap(doc))
    assert "thread_ts" in payload["event"]


def test_wrap_no_thread_ts_for_root_message() -> None:
    doc = _make_slack_doc(thread_parent_id=None)
    payload = json.loads(wrap(doc))
    assert "thread_ts" not in payload["event"]


def test_fixture_shape_matches_wrapper_shape() -> None:
    """Wrapper output has the same top-level keys as the real fixture."""
    fixture = json.loads(_FIXTURE.read_text())
    doc = _make_slack_doc()
    wrapper_payload = json.loads(wrap(doc))
    fixture_keys = set(fixture.keys())
    wrapper_keys = set(wrapper_payload.keys())
    # Wrapper must have at minimum: type, event, team_id
    assert {"type", "event", "team_id"}.issubset(wrapper_keys)
    # event sub-keys must include channel, text, ts, type, user
    assert {"channel", "text", "ts", "type", "user"}.issubset(set(wrapper_payload["event"].keys()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_output_slack_wrapper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.output.slack'`

- [ ] **Step 3: Implement the wrapper**

Create `scripts/synth/output/slack.py`:

```python
"""SlackWrapper — serialize a SynthDoc to a Slack Events API event_callback envelope.

The output must round-trip through SlackConnector.parse_webhook_event:
  parse_webhook_event(customer_id, {}, json.loads(wrap(doc)))
must return a WebhookParseResult with source_event_id == f"{channel}:{ts}".

Envelope shape (matches fixtures/slack/message_simple.json):
{
  "type": "event_callback",
  "team_id": "T-SYNTH",
  "api_app_id": "A-SYNTH",
  "event_id": "<source_event_id>",
  "event_time": <unix_int>,
  "event": {
    "type": "message",
    "channel": "<channel>",
    "user": "<persona_slug>",
    "text": "<text>",
    "ts": "<unix>.<6digits>",
    "thread_ts": "<parent_ts>"   # only present when thread_parent_id is not None
  }
}
"""

from __future__ import annotations

import orjson

from scripts.synth.output.base import SynthDoc

_SYNTH_TEAM_ID = "T-SYNTH"
_SYNTH_APP_ID = "A-SYNTH"


def _ts_str(dt) -> str:  # type: ignore[no-untyped-def]
    """Convert a datetime to Slack's "<unix_seconds>.<6digits>" string format."""
    unix = int(dt.timestamp())
    return f"{unix}.000000"


def _user_slug(doc: SynthDoc) -> str:
    """Derive a stable pseudo-user id from the first persona canonical_id."""
    if not doc.personas:
        return "U-SYNTH-unknown"
    raw = doc.personas[0].replace(":", "-").replace("@", "").upper()
    return f"U-{raw}"


def wrap(doc: SynthDoc) -> bytes:
    """Produce a Slack Events API event_callback envelope as JSON bytes."""
    channel = doc.channel or "#general"
    ts = _ts_str(doc.occurred_at)

    event: dict = {
        "type": "message",
        "channel": channel,
        "user": _user_slug(doc),
        "text": doc.text,
        "ts": ts,
        "team": _SYNTH_TEAM_ID,
    }

    # Only include thread_ts when this doc is a reply (thread_parent_id set).
    if doc.thread_parent_id is not None:
        # The thread_ts must be a valid ts string. We derive it deterministically
        # from the thread_parent_id by using the same occurred_at timestamp minus
        # 1 second (the parent was posted 1s earlier in the synthetic timeline).
        parent_unix = int(doc.occurred_at.timestamp()) - 1
        event["thread_ts"] = f"{parent_unix}.000000"

    payload = {
        "type": "event_callback",
        "team_id": _SYNTH_TEAM_ID,
        "api_app_id": _SYNTH_APP_ID,
        "event_id": doc.source_event_id,
        "event_time": int(doc.occurred_at.timestamp()),
        "event": event,
    }

    return orjson.dumps(payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_output_slack_wrapper.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/output/slack.py tests/synth/test_output_slack_wrapper.py
git commit -m "$(cat <<'EOF'
feat(synth): SlackWrapper producing event_callback envelopes with round-trip shape

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: NotionWrapper

**Files:**
- Create: `scripts/synth/output/notion.py`
- Test: `tests/synth/test_output_notion_wrapper.py`

**Connector interface understood from `services/ingestion/handlers/notion.py`:** `NotionConnector.parse_webhook_event` calls `_is_notion_webhook(payload)` which checks `isinstance(payload.get("entity"), dict) and isinstance(payload.get("type"), str)`. It then reads `payload["type"]` (must be in `_ACCEPTED_EVENT_TYPES`), `payload["entity"]["type"]` (must be "page"), `payload["entity"]["id"]`, and `payload.get("data", {}).get("last_edited_time")`. The `workspace_id` comes from `payload.get("workspace_id")`.

The fixture at `fixtures/notion/page_updated.json`:
```json
{"id": "evt_01HXYZ", "type": "page.updated", "timestamp": "2026-04-22T12:00:00.000Z",
 "workspace_id": "ws_TEST", "workspace_name": "PRBE",
 "entity": {"type": "page", "id": "page_abc123"},
 "data": {"last_edited_time": "2026-04-22T12:00:00.000Z", "last_edited_by": {"id": "user_alice"},
           "updated_properties": ["title", "Status"]}}
```

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_output_notion_wrapper.py`:

```python
"""NotionWrapper round-trip tests.

The wrapper must produce an envelope that NotionConnector._parse_notion_webhook
accepts. We verify the shape directly by parsing the JSON and checking the
fields the connector reads.
"""

from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.notion import wrap

_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "notion" / "page_updated.json"


def _make_notion_doc(
    *,
    text: str = "## Incident: payments 500s\nOwner: @alice\nStatus: resolved",
    page_section: str = "Engineering > On-call rotation",
    occurred_at: datetime | None = None,
) -> SynthDoc:
    ts = occurred_at or datetime(2026, 5, 5, 9, 0, 0, tzinfo=UTC)
    return SynthDoc(
        id="scn-oncall-2026-05-05-notion-0",
        source=Source.NOTION,
        source_event_id="scn-oncall-2026-05-05-notion-0",
        text=text,
        occurred_at=ts,
        channel=None,
        page_id="page-scn-oncall-2026-05-05-notion-0",
        thread_parent_id=None,
        scenario_id="scn-oncall-2026-05-05",
        archetype="ON_CALL_HANDOFF",
        personas=("gh:alice", "gh:bob"),
        services_mentioned=("payments",),
        priority=100,
    )


def test_wrap_produces_valid_json() -> None:
    doc = _make_notion_doc()
    raw = wrap(doc)
    payload = json.loads(raw)
    assert payload["type"] == "page.updated"


def test_wrap_entity_shape() -> None:
    """Connector reads entity.type and entity.id."""
    doc = _make_notion_doc()
    payload = json.loads(wrap(doc))
    assert payload["entity"]["type"] == "page"
    assert payload["entity"]["id"] == doc.page_id


def test_wrap_data_has_last_edited_time() -> None:
    """Connector reads data.last_edited_time for source_event_id construction."""
    doc = _make_notion_doc(occurred_at=datetime(2026, 5, 5, 9, 0, 0, tzinfo=UTC))
    payload = json.loads(wrap(doc))
    assert "last_edited_time" in payload["data"]
    # Must be an ISO string
    let = payload["data"]["last_edited_time"]
    assert "2026-05-05" in let


def test_fixture_shape_matches_wrapper_top_level_keys() -> None:
    """Wrapper top-level keys are a superset of the fixture's required keys."""
    fixture = json.loads(_FIXTURE.read_text())
    doc = _make_notion_doc()
    wrapper = json.loads(wrap(doc))
    required = {"type", "entity", "data", "workspace_id"}
    assert required.issubset(set(wrapper.keys()))
    # entity must have type + id
    assert {"type", "id"}.issubset(set(wrapper["entity"].keys()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_output_notion_wrapper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.output.notion'`

- [ ] **Step 3: Implement the wrapper**

Create `scripts/synth/output/notion.py`:

```python
"""NotionWrapper — serialize a SynthDoc to a Notion page.updated webhook envelope.

The output must round-trip through NotionConnector._parse_notion_webhook.
_is_notion_webhook() checks: isinstance(entity, dict) and isinstance(type, str).
_parse_notion_webhook reads: type, entity.type, entity.id, data.last_edited_time,
workspace_id.

v1 is minimal: title property + plain-text paragraph blocks. Plan 3 can extend
to richer block shapes when LLM-generated content warrants it.

Envelope shape (matches fixtures/notion/page_updated.json structure):
{
  "id": "<source_event_id>",
  "type": "page.updated",
  "timestamp": "<iso8601>",
  "workspace_id": "ws-synth",
  "workspace_name": "Synth",
  "entity": {"type": "page", "id": "<page_id>"},
  "data": {
    "last_edited_time": "<iso8601>",
    "last_edited_by": {"id": "<user_slug>"},
    "updated_properties": ["title"],
    "properties": {
      "title": {
        "type": "title",
        "title": [{"type": "text", "plain_text": "<title>", "text": {"content": "<title>"}}]
      }
    }
  }
}
"""

from __future__ import annotations

import orjson

from scripts.synth.output.base import SynthDoc

_SYNTH_WORKSPACE_ID = "ws-synth"
_SYNTH_WORKSPACE_NAME = "Synth"


def _iso(dt) -> str:  # type: ignore[no-untyped-def]
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _user_slug(doc: SynthDoc) -> str:
    if not doc.personas:
        return "user-synth-unknown"
    return doc.personas[0].replace(":", "-").replace("@", "")


def _title_from_doc(doc: SynthDoc) -> str:
    """Extract first line of text as the page title."""
    first_line = doc.text.splitlines()[0].strip() if doc.text else ""
    # Strip leading markdown heading markers
    title = first_line.lstrip("#").strip()
    return title[:200] if title else f"Synth page {doc.id}"


def _blocks_from_text(text: str) -> list[dict]:
    """Convert plain text to minimal Notion block list (paragraph per line).

    Heading lines (## prefix) become heading_2 blocks.
    Other lines become paragraph blocks.
    Empty lines are skipped.
    """
    blocks: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            content = stripped[3:].strip()
            btype = "heading_2"
        elif stripped.startswith("# "):
            content = stripped[2:].strip()
            btype = "heading_1"
        else:
            content = stripped
            btype = "paragraph"
        blocks.append({
            "type": btype,
            "id": f"block-{len(blocks)}",
            btype: {
                "rich_text": [
                    {
                        "type": "text",
                        "plain_text": content,
                        "text": {"content": content},
                    }
                ]
            },
        })
    return blocks


def wrap(doc: SynthDoc) -> bytes:
    """Produce a Notion page.updated webhook envelope as JSON bytes."""
    page_id = doc.page_id or f"page-{doc.source_event_id}"
    iso_ts = _iso(doc.occurred_at)
    title = _title_from_doc(doc)

    payload = {
        "id": doc.source_event_id,
        "type": "page.updated",
        "timestamp": iso_ts,
        "workspace_id": _SYNTH_WORKSPACE_ID,
        "workspace_name": _SYNTH_WORKSPACE_NAME,
        "entity": {
            "type": "page",
            "id": page_id,
        },
        "data": {
            "last_edited_time": iso_ts,
            "last_edited_by": {"id": _user_slug(doc)},
            "updated_properties": ["title"],
            "properties": {
                "title": {
                    "type": "title",
                    "title": [
                        {
                            "type": "text",
                            "plain_text": title,
                            "text": {"content": title},
                        }
                    ],
                }
            },
            "blocks": _blocks_from_text(doc.text),
        },
    }

    return orjson.dumps(payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_output_notion_wrapper.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/output/notion.py tests/synth/test_output_notion_wrapper.py
git commit -m "$(cat <<'EOF'
feat(synth): NotionWrapper producing page.updated envelopes with connector-compatible shape

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Validator (name-only Pass 1)

**Files:**
- Create: `scripts/synth/validator.py`
- Test: `tests/synth/test_validator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_validator.py`:

```python
"""Validator Pass 1: name-only WorldModel check."""

from __future__ import annotations

from datetime import datetime, UTC

import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.validator import THIRD_PARTY_ALLOWLIST, validate_name_only
from scripts.synth.world_model import (
    ChannelHint,
    DepEdge,
    Person,
    RepoSummary,
    SectionHint,
    Service,
    ServiceKind,
    TimeAnchor,
    Topic,
    TopicKind,
    WorldModel,
)


def _make_world() -> WorldModel:
    now = datetime(2026, 5, 1, tzinfo=UTC)
    person = Person(
        canonical_id="gh:alice",
        gh_username="alice",
        display_name="Alice Smith",
        email_aliases=("alice@example.com",),
        role_hint=None,
        repos_active_in=("github.com/prbe-ai/prbe-knowledge",),
        activity_score=10.0,
    )
    service = Service(
        name="payments-api",
        qualified="payments-api",
        repo_url="github.com/prbe-ai/prbe-knowledge",
        kind=ServiceKind.API,
        description=None,
        owners=(),
        recent_activity=5.0,
        deploy_target=None,
    )
    channel = ChannelHint(name="#standup", suggested_topic=None, related_services=())
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe-knowledge", sha="abc", default_branch="main"),),
        people=(person,),
        services=(service,),
        topic_pool=(),
        channels=(channel,),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={"github.com/prbe-ai/prbe-knowledge": "abc"},
    )


def _make_doc(text: str, source: Source = Source.SLACK) -> SynthDoc:
    return SynthDoc(
        id="doc-test-1",
        source=source,
        source_event_id="doc-test-1",
        text=text,
        occurred_at=datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC),
        channel="#standup",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-test-1",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments-api",),
        priority=100,
    )


def test_clean_doc_produces_no_violations() -> None:
    world = _make_world()
    doc = _make_doc("Yesterday: shipped payments-api. Today: @alice reviews auth. Blockers: none.")
    violations = validate_name_only((doc,), world)
    assert violations == ()


def test_fabricated_service_name_flagged() -> None:
    world = _make_world()
    doc = _make_doc("shipped fake-service yesterday.")
    violations = validate_name_only((doc,), world)
    assert len(violations) == 1
    assert "fake-service" in violations[0].out_of_world


def test_third_party_saas_not_flagged() -> None:
    world = _make_world()
    # stripe and aws are in THIRD_PARTY_ALLOWLIST
    doc = _make_doc("integrated with stripe and aws.")
    violations = validate_name_only((doc,), world)
    assert violations == ()


def test_world_model_channel_name_not_flagged() -> None:
    world = _make_world()
    doc = _make_doc("posted to #standup channel.")
    violations = validate_name_only((doc,), world)
    assert violations == ()


def test_world_model_service_name_not_flagged() -> None:
    world = _make_world()
    doc = _make_doc("deployed payments-api to production.")
    violations = validate_name_only((doc,), world)
    assert violations == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_validator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.validator'`

- [ ] **Step 3: Implement the validator**

Create `scripts/synth/validator.py`:

```python
"""Validator Pass 1 — name-only WorldModel check.

Extracts proper-noun-shaped tokens from synthetic doc text and verifies
each one appears in the WorldModel's entity set or the third-party allowlist.

This pass is intentionally narrow:
- _TOKEN_RE captures Slack channels (#foo-bar), person mentions (@foo-bar),
  and kebab-cased service names (payments-api).
- False positives (common words that happen to match kebab) are rare in
  templated output; the allowlist handles known SaaS names.
- False negatives (camelCase service names, etc.) are accepted in v1.
  Plan 3's Pass 2 (cheap LLM consistency check) handles the rest.

The validator does NOT raise on violations — it returns them so the caller
(CLI / IngestionWriter) can log and decide whether to abort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.synth.output.base import SynthDoc
from scripts.synth.world_model import WorldModel

# Matches three token shapes likely to be company-internal references:
#   #channel-name   @person-handle   kebab-service-name (at least two segments)
_TOKEN_RE = re.compile(
    r"#[\w-]+"
    r"|@[\w-]+"
    r"|\b[a-z][a-z0-9-]*-[a-z][a-z0-9-]*\b"
)

# Common third-party SaaS names that are obviously not internal services.
# Lowercase, sorted alphabetically.
THIRD_PARTY_ALLOWLIST: frozenset[str] = frozenset({
    "anthropic",
    "aws",
    "datadog",
    "github",
    "granola",
    "linear",
    "notion",
    "openai",
    "sentry",
    "slack",
    "stripe",
})


@dataclass(frozen=True)
class Violation:
    doc_id: str
    out_of_world: tuple[str, ...]


def _extract_proper_nouns(text: str) -> set[str]:
    """Extract tokens matching _TOKEN_RE, lowercased."""
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def validate_name_only(
    docs: tuple[SynthDoc, ...],
    world: WorldModel,
) -> tuple[Violation, ...]:
    """Check that all proper-noun tokens in docs map to known world entities.

    Allowed token set = WorldModel services + people + channels + third-party.
    Returns a tuple of Violation (one per doc with out-of-world tokens).
    """
    allowed: set[str] = set()

    # Services: both bare name and qualified name
    for svc in world.services:
        allowed.add(svc.name.lower())
        allowed.add(svc.qualified.lower())

    # People: display_name, gh_username, channel-mention forms
    for person in world.people:
        if person.display_name:
            allowed.add(person.display_name.lower())
        if person.gh_username:
            allowed.add(person.gh_username.lower())
            allowed.add(f"@{person.gh_username.lower()}")

    # Channels: name as-is (already has # prefix)
    for ch in world.channels:
        allowed.add(ch.name.lower())

    # Third-party SaaS
    for name in THIRD_PARTY_ALLOWLIST:
        allowed.add(name)

    violations: list[Violation] = []
    for doc in docs:
        mentioned = _extract_proper_nouns(doc.text)
        out_of_world = mentioned - allowed
        if out_of_world:
            violations.append(
                Violation(doc_id=doc.id, out_of_world=tuple(sorted(out_of_world)))
            )

    return tuple(violations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_validator.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/validator.py tests/synth/test_validator.py
git commit -m "$(cat <<'EOF'
feat(synth): validator Pass 1 — name-only WorldModel check with regex token extraction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: OwnershipIndex

**Files:**
- Create: `scripts/synth/ownership.py`
- Test: `tests/synth/test_ownership.py`

**Implementation note:** `canonicalize_people`'s helpers `_gh_username_from_noreply` and the email-lowercasing rule live in `scripts/synth/world_model.py`. Plan 2 imports `_gh_username_from_noreply` directly from that module — it is the same package and the cross-module use is justified by a comment. `email_to_gh` resolution is reconstructed locally (one clean pass over signals) rather than duplicating the full `canonicalize_people` merge logic.

Service ownership is derived from `Commit.files_touched` paths: for each file path, find the `Manifest` in the same `RepoSignals` whose `manifest.path.parent` is the longest ancestor of the file path. The manifest's `name` is the service name. Top-3 services per person by frequency (alphabetical tie-break for determinism).

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_ownership.py`:

```python
"""OwnershipIndex tests."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import pytest

from scripts.synth.extractor.git_log import Branch, Commit
from scripts.synth.extractor.manifests import Manifest, ManifestKind
from scripts.synth.extractor.repo import RepoSignals
from scripts.synth.ownership import OwnershipIndex, build_ownership_index
from scripts.synth.world_model import (
    ChannelHint,
    DepEdge,
    Person,
    RepoSummary,
    SectionHint,
    Service,
    ServiceKind,
    TimeAnchor,
    Topic,
    TopicKind,
    WorldModel,
)


def _make_commit(
    sha: str,
    author_email: str,
    author_name: str,
    files: tuple[str, ...],
    ts: datetime | None = None,
) -> Commit:
    return Commit(
        sha=sha,
        author_name=author_name,
        author_email=author_email,
        ts=ts or datetime(2026, 4, 1, tzinfo=UTC),
        subject="fix something",
        body="",
        files_touched=files,
    )


def _make_manifest(path: Path, name: str) -> Manifest:
    return Manifest(
        kind=ManifestKind.PYPROJECT,
        path=path,
        name=name,
        description=None,
        dependencies=(),
    )


def _make_signals(
    url: str,
    commits: list[Commit],
    manifests: list[Manifest],
) -> RepoSignals:
    return RepoSignals(
        url=url,
        clone_path=Path("/tmp/repo"),
        default_branch="main",
        latest_sha="abc123",
        description=None,
        manifests=tuple(manifests),
        readmes=(),
        codeowners=(),
        commits=tuple(commits),
        branches=(Branch(name="main", last_commit_sha="abc123", last_commit_ts=datetime(2026, 4, 1, tzinfo=UTC)),),
        issues=None,
        prs=None,
        contributors=None,
        workflows=None,
    )


def _make_world(people: list[Person], services: list[Service]) -> WorldModel:
    now = datetime(2026, 5, 1, tzinfo=UTC)
    return WorldModel(
        repos=(),
        people=tuple(people),
        services=tuple(services),
        topic_pool=(),
        channels=(),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={},
    )


def _make_person(canonical_id: str, email: str) -> Person:
    return Person(
        canonical_id=canonical_id,
        gh_username=canonical_id.removeprefix("gh:") if canonical_id.startswith("gh:") else None,
        display_name=canonical_id,
        email_aliases=(email,),
        role_hint=None,
        repos_active_in=(),
        activity_score=1.0,
    )


def test_single_repo_single_person() -> None:
    """One person committing to one service is indexed correctly."""
    commits = [
        _make_commit("c1", "alice@example.com", "Alice", ("payments/src/main.py",)),
        _make_commit("c2", "alice@example.com", "Alice", ("payments/src/other.py",)),
    ]
    manifests = [_make_manifest(Path("/repo/payments/pyproject.toml"), "payments")]
    signals = [_make_signals("github.com/prbe-ai/prbe", commits, manifests)]
    person = _make_person("email:alice@example.com", "alice@example.com")
    service = Service(name="payments", qualified="payments", repo_url="github.com/prbe-ai/prbe",
                      kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0, deploy_target=None)
    world = _make_world([person], [service])

    idx = build_ownership_index(signals, world)
    assert "payments" in idx.services_by_person.get("email:alice@example.com", ())


def test_person_without_commits_gets_empty_tuple() -> None:
    signals = [_make_signals("github.com/prbe-ai/prbe", [], [])]
    person = _make_person("gh:bob", "bob@example.com")
    world = _make_world([person], [])
    idx = build_ownership_index(signals, world)
    assert idx.services_by_person.get("gh:bob", ()) == ()


def test_top_3_services_by_frequency() -> None:
    """If a person commits to 4+ services, only top 3 are kept."""
    commits = [
        _make_commit("c1", "alice@example.com", "Alice", ("svc-a/main.py",)),
        _make_commit("c2", "alice@example.com", "Alice", ("svc-a/other.py",)),
        _make_commit("c3", "alice@example.com", "Alice", ("svc-b/main.py",)),
        _make_commit("c4", "alice@example.com", "Alice", ("svc-c/main.py",)),
        _make_commit("c5", "alice@example.com", "Alice", ("svc-d/main.py",)),
    ]
    manifests = [
        _make_manifest(Path("/repo/svc-a/pyproject.toml"), "svc-a"),
        _make_manifest(Path("/repo/svc-b/pyproject.toml"), "svc-b"),
        _make_manifest(Path("/repo/svc-c/pyproject.toml"), "svc-c"),
        _make_manifest(Path("/repo/svc-d/pyproject.toml"), "svc-d"),
    ]
    signals = [_make_signals("github.com/prbe-ai/prbe", commits, manifests)]
    person = _make_person("email:alice@example.com", "alice@example.com")
    world = _make_world([person], [])
    idx = build_ownership_index(signals, world)
    top = idx.services_by_person.get("email:alice@example.com", ())
    assert len(top) <= 3
    # svc-a appears twice so it must be in top 3
    assert "svc-a" in top


def test_people_by_service_inverse() -> None:
    commits = [
        _make_commit("c1", "alice@example.com", "Alice", ("payments/main.py",)),
    ]
    manifests = [_make_manifest(Path("/repo/payments/pyproject.toml"), "payments")]
    signals = [_make_signals("github.com/prbe-ai/prbe", commits, manifests)]
    person = _make_person("email:alice@example.com", "alice@example.com")
    service = Service(name="payments", qualified="payments", repo_url="github.com/prbe-ai/prbe",
                      kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0, deploy_target=None)
    world = _make_world([person], [service])
    idx = build_ownership_index(signals, world)
    assert "email:alice@example.com" in idx.people_by_service.get("payments", ())


def test_deterministic_tie_break() -> None:
    """Equal-frequency services are sorted alphabetically for determinism."""
    commits = [
        _make_commit("c1", "alice@example.com", "Alice", ("svc-z/main.py",)),
        _make_commit("c2", "alice@example.com", "Alice", ("svc-a/main.py",)),
    ]
    manifests = [
        _make_manifest(Path("/repo/svc-z/pyproject.toml"), "svc-z"),
        _make_manifest(Path("/repo/svc-a/pyproject.toml"), "svc-a"),
    ]
    signals = [_make_signals("github.com/prbe-ai/prbe", commits, manifests)]
    person = _make_person("email:alice@example.com", "alice@example.com")
    world = _make_world([person], [])
    idx1 = build_ownership_index(signals, world)
    idx2 = build_ownership_index(signals, world)
    assert idx1.services_by_person == idx2.services_by_person
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_ownership.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.ownership'`

- [ ] **Step 3: Implement OwnershipIndex**

Create `scripts/synth/ownership.py`:

```python
"""OwnershipIndex — services-per-person derived from git commit history.

WorldModel.Service.owners is always () in Plan 2 (Plan 1 left it blank).
The OwnershipIndex is the workaround: we compute service ownership at
scenario-build time by walking commit file paths through manifest ancestry.

Algorithm:
  For each commit in each RepoSignals:
    1. Resolve author_email to a canonical_id using the same email-lowercasing
       rule as canonicalize_people (gh: prefix if noreply, email: otherwise).
    2. For each file in commit.files_touched, find the Manifest in this
       RepoSignals whose manifest.path.parent is the deepest ancestor of the file.
       The manifest.name is the service name.
    3. Record (canonical_id, service_name) pair.
  Aggregate per person: top-3 service names by frequency, alphabetical tie-break.
  Inverse: people_by_service[service_name] = sorted list of canonical_ids.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.synth.extractor.repo import RepoSignals
from scripts.synth.world_model import WorldModel

# Import the noreply resolver from world_model — same package, justified by
# avoiding duplication of the noreply/name-merge logic.
from scripts.synth.world_model import _gh_username_from_noreply  # noqa: PLC2701


@dataclass(frozen=True)
class OwnershipIndex:
    # canonical_id -> top-3 service names (by commit frequency, alpha tie-break)
    services_by_person: dict[str, tuple[str, ...]]
    # service qualified name -> sorted canonical_ids that touched it >= 1 time
    people_by_service: dict[str, tuple[str, ...]]


def _resolve_canonical_id(email: str) -> str:
    """Map a commit author_email to a canonical_id string.

    Mirrors the two-rule logic in canonicalize_people:
      - GitHub noreply emails → gh:<username>
      - All others → email:<lowercased>
    Does NOT consult the Contributor list (no GH API data available here).
    """
    email_lower = email.lower().strip()
    username = _gh_username_from_noreply(email)
    if username:
        return f"gh:{username}"
    return f"email:{email_lower}"


def _deepest_manifest_ancestor(
    file_path: str,
    manifests: tuple,
) -> str | None:
    """Return the manifest name whose directory is the deepest ancestor of file_path.

    file_path is a repo-relative string like "services/payments/src/main.py".
    We compare it against each manifest's path.parent (also repo-relative).
    The deepest match (longest path prefix) wins.
    """
    file_parts = Path(file_path).parts
    best_name: str | None = None
    best_depth: int = -1

    for m in manifests:
        if not m.name:
            continue
        # manifest.path may be absolute; we use only the parts after the repo root.
        # Since paths come from the same repo walk, we use the directory of the manifest.
        manifest_dir_parts = m.path.parent.parts

        # Check that file_parts starts with manifest_dir_parts
        if len(manifest_dir_parts) > len(file_parts):
            continue

        # Match: check prefix alignment
        match = True
        for i, part in enumerate(manifest_dir_parts):
            if i >= len(file_parts):
                match = False
                break
            # Compare last N parts of manifest_dir against file_parts for repo-relative matching.
            # Since manifest paths may be absolute we do a suffix-match heuristic:
            # if the manifest is at /repo/payments/pyproject.toml, its parent parts end with "payments".
            # The file "payments/main.py" starts with "payments". We match the last len(manifest_dir_parts)
            # parts of the manifest against the first len(manifest_dir_parts) parts of the file.
            if file_parts[i] != manifest_dir_parts[-(len(manifest_dir_parts) - i)]:
                match = False
                break

        if match and len(manifest_dir_parts) > best_depth:
            best_depth = len(manifest_dir_parts)
            best_name = m.name

    return best_name


def build_ownership_index(
    signals: list[RepoSignals],
    world: WorldModel,
) -> OwnershipIndex:
    """Build OwnershipIndex from commit history.

    Does not modify WorldModel. Safe to call multiple times with the same
    inputs (pure function of signals + world).
    """
    # (canonical_id, service_name) -> count
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for sig in signals:
        for commit in sig.commits:
            canonical_id = _resolve_canonical_id(commit.author_email)
            for file_path in commit.files_touched:
                svc_name = _deepest_manifest_ancestor(file_path, sig.manifests)
                if svc_name:
                    pair_counts[(canonical_id, svc_name)] += 1

    # Aggregate per person: top-3 by frequency, alphabetical tie-break.
    person_service_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for (canonical_id, svc_name), count in pair_counts.items():
        person_service_counts[canonical_id][svc_name] = count

    services_by_person: dict[str, tuple[str, ...]] = {}
    for canonical_id, counts in person_service_counts.items():
        # Sort by (-frequency, name) for deterministic alphabetical tie-break.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        services_by_person[canonical_id] = tuple(name for name, _ in ranked[:3])

    # Inverse: service -> sorted list of people who touched it >= 1 time.
    service_people: dict[str, set[str]] = defaultdict(set)
    for (canonical_id, svc_name) in pair_counts:
        service_people[svc_name].add(canonical_id)

    people_by_service: dict[str, tuple[str, ...]] = {
        svc: tuple(sorted(people))
        for svc, people in service_people.items()
    }

    return OwnershipIndex(
        services_by_person=services_by_person,
        people_by_service=people_by_service,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_ownership.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/ownership.py tests/synth/test_ownership.py
git commit -m "$(cat <<'EOF'
feat(synth): OwnershipIndex — services-per-person from git commit file paths

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: STANDUP_UPDATE archetype

**Files:**
- Create: `scripts/synth/archetypes/standup.py`
- Test: `tests/synth/test_archetype_standup.py`

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_archetype_standup.py`:

```python
"""STANDUP_UPDATE archetype builder tests."""

from __future__ import annotations

from datetime import date, datetime, UTC, timedelta

import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.archetypes.standup import STANDUP_UPDATE, build_standup_specs
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.world_model import (
    ChannelHint,
    Person,
    RepoSummary,
    SectionHint,
    Service,
    ServiceKind,
    Topic,
    TopicKind,
    WorldModel,
)


def _build_test_world(
    *,
    n_people: int = 3,
    n_topics: int = 5,
    window_end: datetime | None = None,
) -> WorldModel:
    now = window_end or datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    people = tuple(
        Person(
            canonical_id=f"gh:person{i}",
            gh_username=f"person{i}",
            display_name=f"Person {i}",
            email_aliases=(f"person{i}@example.com",),
            role_hint=None,
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=float(n_people - i),
        )
        for i in range(n_people)
    )
    services = (
        Service(name="payments-api", qualified="payments-api",
                repo_url="github.com/prbe-ai/prbe", kind=ServiceKind.API,
                description=None, owners=(), recent_activity=5.0, deploy_target=None),
        Service(name="auth-service", qualified="auth-service",
                repo_url="github.com/prbe-ai/prbe", kind=ServiceKind.API,
                description=None, owners=(), recent_activity=3.0, deploy_target=None),
    )
    topics = tuple(
        Topic(
            text=f"fix issue in payments-api #{j}",
            kind=TopicKind.COMMIT,
            repo_url="github.com/prbe-ai/prbe",
            ts=now - timedelta(days=j),
            mentioned_services=("payments-api",),
            mentioned_people=(),
            weight=1.0 / (j + 1),
        )
        for j in range(n_topics)
    )
    channels = (ChannelHint(name="#standup", suggested_topic=None, related_services=()),)
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe", sha="abc", default_branch="main"),),
        people=people,
        services=services,
        topic_pool=topics,
        channels=channels,
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={"github.com/prbe-ai/prbe": "abc"},
    )


def _make_ownership(world: WorldModel) -> OwnershipIndex:
    """Give every person 1 service for testing."""
    services_by_person = {
        p.canonical_id: ("payments-api",)
        for p in world.people
    }
    people_by_service = {
        "payments-api": tuple(p.canonical_id for p in world.people)
    }
    return OwnershipIndex(
        services_by_person=services_by_person,
        people_by_service=people_by_service,
    )


def test_standup_archetype_metadata() -> None:
    assert STANDUP_UPDATE.name == "STANDUP_UPDATE"
    assert STANDUP_UPDATE.cadence.value == "daily"
    assert STANDUP_UPDATE.needs_planner_call is False
    assert Source.SLACK in STANDUP_UPDATE.sources_used


def test_spec_count_matches_working_days_times_personas() -> None:
    """30-day window has 22 working days × 3 people = 66 specs (approx)."""
    world = _build_test_world(n_people=3)
    ownership = _make_ownership(world)
    end = datetime(2026, 5, 1, tzinfo=UTC)
    # Use a known 5-working-day window: Mon 2026-04-27 to Fri 2026-05-01.
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=end, days=7)
    specs = build_standup_specs(world, ownership, window, seed=42, top_n=3)
    # 5 working days × 3 personas = 15 specs
    assert len(specs) == 15


def test_each_spec_has_one_doc_spec() -> None:
    world = _build_test_world(n_people=2)
    ownership = _make_ownership(world)
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=3)
    specs = build_standup_specs(world, ownership, window, seed=42, top_n=2)
    for spec in specs:
        assert len(spec.doc_specs) == 1
        assert spec.doc_specs[0].source == Source.SLACK


def test_determinism() -> None:
    """Same inputs produce identical output."""
    world = _build_test_world(n_people=2)
    ownership = _make_ownership(world)
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=7)
    specs1 = build_standup_specs(world, ownership, window, seed=42, top_n=2)
    specs2 = build_standup_specs(world, ownership, window, seed=42, top_n=2)
    assert len(specs1) == len(specs2)
    for s1, s2 in zip(specs1, specs2):
        assert s1.id == s2.id
        assert s1.doc_specs[0].text == s2.doc_specs[0].text


def test_person_without_services_skipped() -> None:
    world = _build_test_world(n_people=3)
    # Only person0 has services; others are empty.
    ownership = OwnershipIndex(
        services_by_person={"gh:person0": ("payments-api",)},
        people_by_service={"payments-api": ("gh:person0",)},
    )
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=3)
    specs = build_standup_specs(world, ownership, window, seed=42, top_n=3)
    # Only person0 generates specs (1 working day × 1 person)
    personas_seen = {spec.cast[0] for spec in specs}
    assert personas_seen == {"gh:person0"}


def test_text_template_renders() -> None:
    world = _build_test_world(n_people=1)
    ownership = _make_ownership(world)
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=2)
    specs = build_standup_specs(world, ownership, window, seed=42, top_n=1)
    assert len(specs) >= 1
    text = specs[0].doc_specs[0].text
    assert "Yesterday:" in text
    assert "Today:" in text
    assert "Blockers:" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_archetype_standup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.archetypes.standup'` (or `scripts.synth.scenarios` for `TimeWindow`).

Note: `TimeWindow` is defined in Task 10. The standup builder imports it from `scripts.synth.scenarios`. For this task to work, create a minimal `scripts/synth/scenarios.py` stub that just defines `TimeWindow` (the full implementation comes in Task 10).

**Step 3a: Create minimal `scenarios.py` stub** (will be replaced in Task 10):

```python
"""ScenarioRunner stub — TimeWindow defined here for Task 7+8 archetype builders.

Full implementation in Task 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TimeWindow:
    end: datetime
    days: int
```

- [ ] **Step 3b: Implement STANDUP_UPDATE**

Create `scripts/synth/archetypes/standup.py`:

```python
"""STANDUP_UPDATE archetype — daily Slack standup messages.

For each working day (Mon-Fri) in time_window, for each top-N persona by
activity_score who has at least one service in the OwnershipIndex, emit one
ScenarioSpec with one DocSpec (a Slack message to #standup).

Text template:
  "Yesterday: shipped {topic_a}. Today: {service} - {topic_b}. Blockers: none."

topic_a and topic_b are drawn from world.topic_pool filtered to topics whose
mentioned_services overlap the persona's services and ts >= day - 7 days.
If only one topic is available, topic_b falls back to "ongoing work".
If no topics are available, both slots use "ongoing work".
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, UTC

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    DocSpec,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.world_model import TopicKind, WorldModel

STANDUP_UPDATE = Archetype(
    name="STANDUP_UPDATE",
    category=Category.RECURRING,
    cadence=Cadence.DAILY,
    sources_used=(Source.SLACK,),
    cast_size=(1, 1),
    needs_planner_call=False,
    validator_level=ValidatorLevel.NAME_ONLY,
)


def _working_days(window_end: datetime, days: int) -> list[date]:
    """Return Mon-Fri dates in [window_end - days, window_end], ascending."""
    result: list[date] = []
    end_date = window_end.date()
    start_date = end_date - timedelta(days=days - 1)
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # 0=Mon, 4=Fri
            result.append(current)
        current += timedelta(days=1)
    return result


def _safe_id(canonical_id: str) -> str:
    """Make canonical_id filesystem-safe by replacing ':' with '-'."""
    return canonical_id.replace(":", "-")


def build_standup_specs(
    world: WorldModel,
    ownership: OwnershipIndex,
    time_window: object,  # TimeWindow from scenarios.py
    seed: int,
    top_n: int = 5,
) -> tuple[ScenarioSpec, ...]:
    """Build STANDUP_UPDATE ScenarioSpecs for all working days in time_window."""
    rng = random.Random(seed)

    end: datetime = time_window.end  # type: ignore[attr-defined]
    days: int = time_window.days  # type: ignore[attr-defined]
    work_days = _working_days(end, days)

    # Top-N personas by activity_score (descending), then canonical_id for tie-break.
    sorted_people = sorted(
        world.people,
        key=lambda p: (-p.activity_score, p.canonical_id),
    )[:top_n]

    specs: list[ScenarioSpec] = []

    for work_day in work_days:
        day_start = datetime(work_day.year, work_day.month, work_day.day, 9, 0, 0, tzinfo=UTC)
        lookback = day_start - timedelta(days=7)

        for person in sorted_people:
            person_services = ownership.services_by_person.get(person.canonical_id, ())
            if not person_services:
                continue

            primary_service = person_services[0]

            # Filter topics: mentioned_services overlaps person_services AND ts in lookback window.
            person_svc_set = set(person_services)
            relevant_topics = [
                t for t in world.topic_pool
                if t.ts is not None
                and lookback <= t.ts < day_start
                and (set(t.mentioned_services) & person_svc_set)
            ]

            # Sort by weight desc, then text for determinism.
            relevant_topics.sort(key=lambda t: (-t.weight, t.text))

            if len(relevant_topics) >= 2:
                topic_a_text = relevant_topics[0].text[:50]
                topic_b_text = relevant_topics[1].text[:50]
            elif len(relevant_topics) == 1:
                topic_a_text = relevant_topics[0].text[:50]
                topic_b_text = "ongoing work"
            else:
                topic_a_text = "ongoing work"
                topic_b_text = "ongoing work"

            text = (
                f"Yesterday: shipped {topic_a_text}. "
                f"Today: {primary_service} - {topic_b_text}. "
                f"Blockers: none."
            )

            safe_id = _safe_id(person.canonical_id)
            doc_id = f"scn-standup-{safe_id}-{work_day.isoformat()}-slack-0"
            scenario_id = f"scn-standup-{safe_id}-{work_day.isoformat()}"

            doc_spec = DocSpec(
                id=doc_id,
                source=Source.SLACK,
                occurred_at=day_start,
                channel="#standup",
                page_section=None,
                text=text,
                thread_parent_id=None,
                personas=(person.canonical_id,),
                services_mentioned=tuple(person_services),
            )
            spec = ScenarioSpec(
                id=scenario_id,
                archetype_name="STANDUP_UPDATE",
                instance_ts=day_start,
                cast=(person.canonical_id,),
                affected_services=tuple(person_services),
                doc_specs=(doc_spec,),
            )
            specs.append(spec)

    return tuple(specs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_archetype_standup.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/archetypes/standup.py scripts/synth/scenarios.py tests/synth/test_archetype_standup.py
git commit -m "$(cat <<'EOF'
feat(synth): STANDUP_UPDATE archetype with working-day scheduling and topic filtering

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: ON_CALL_HANDOFF archetype

**Files:**
- Create: `scripts/synth/archetypes/oncall.py`
- Test: `tests/synth/test_archetype_oncall.py`

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_archetype_oncall.py`:

```python
"""ON_CALL_HANDOFF archetype builder tests."""

from __future__ import annotations

from datetime import datetime, UTC, timedelta

import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.archetypes.oncall import ON_CALL_HANDOFF, build_oncall_specs
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.scenarios import TimeWindow
from scripts.synth.world_model import (
    ChannelHint,
    Person,
    RepoSummary,
    SectionHint,
    Service,
    ServiceKind,
    Topic,
    TopicKind,
    WorldModel,
)


def _build_oncall_world(
    *,
    n_people: int = 4,
    n_issues: int = 3,
    window_end: datetime | None = None,
) -> WorldModel:
    now = window_end or datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC)  # Monday
    people = tuple(
        Person(
            canonical_id=f"gh:eng{i}",
            gh_username=f"eng{i}",
            display_name=f"Engineer {i}",
            email_aliases=(f"eng{i}@example.com",),
            role_hint=None,
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=float(n_people - i),
        )
        for i in range(n_people)
    )
    # Some ISSUE topics in last week, some COMMIT topics
    issues = tuple(
        Topic(
            text=f"payments-api returning 500s run {j}",
            kind=TopicKind.ISSUE,
            repo_url="github.com/prbe-ai/prbe",
            ts=now - timedelta(days=j + 1),
            mentioned_services=("payments-api",),
            mentioned_people=(),
            weight=1.0 / (j + 1),
        )
        for j in range(n_issues)
    )
    channels = (
        ChannelHint(name="#oncall", suggested_topic=None, related_services=()),
    )
    sections = (
        SectionHint(title="Engineering > On-call rotation", related_services=()),
    )
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe", sha="abc", default_branch="main"),),
        people=people,
        services=(Service(name="payments-api", qualified="payments-api",
                          repo_url="github.com/prbe-ai/prbe", kind=ServiceKind.API,
                          description=None, owners=(), recent_activity=5.0, deploy_target=None),),
        topic_pool=issues,
        channels=channels,
        notion_sections=sections,
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={"github.com/prbe-ai/prbe": "abc"},
    )


def _make_ownership(world: WorldModel) -> OwnershipIndex:
    services_by_person = {p.canonical_id: ("payments-api",) for p in world.people}
    people_by_service = {"payments-api": tuple(p.canonical_id for p in world.people)}
    return OwnershipIndex(services_by_person=services_by_person, people_by_service=people_by_service)


def test_oncall_archetype_metadata() -> None:
    assert ON_CALL_HANDOFF.name == "ON_CALL_HANDOFF"
    assert ON_CALL_HANDOFF.cadence.value == "weekly"
    assert Source.SLACK in ON_CALL_HANDOFF.sources_used
    assert Source.NOTION in ON_CALL_HANDOFF.sources_used


def test_one_monday_produces_one_scenario_with_three_docs() -> None:
    """One Monday in window → 1 ScenarioSpec with 3 DocSpecs (slack-0, slack-1, notion-0)."""
    # Window: exactly one Monday (2026-04-27)
    world = _build_oncall_world(window_end=datetime(2026, 4, 28, tzinfo=UTC))
    ownership = _make_ownership(world)
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=7)
    specs = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    assert len(specs) == 1
    assert len(specs[0].doc_specs) == 3


def test_doc_spec_sources() -> None:
    """Two Slack docs and one Notion doc per handoff."""
    world = _build_oncall_world(window_end=datetime(2026, 4, 28, tzinfo=UTC))
    ownership = _make_ownership(world)
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=7)
    specs = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    sources = [d.source for d in specs[0].doc_specs]
    assert sources.count(Source.SLACK) == 2
    assert sources.count(Source.NOTION) == 1


def test_thread_parent_id_wiring() -> None:
    """Slack reply (slack-1) has thread_parent_id == slack-0's id."""
    world = _build_oncall_world(window_end=datetime(2026, 4, 28, tzinfo=UTC))
    ownership = _make_ownership(world)
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=7)
    specs = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    slack_docs = [d for d in specs[0].doc_specs if d.source == Source.SLACK]
    parent = slack_docs[0]
    reply = slack_docs[1]
    assert reply.thread_parent_id == parent.id


def test_zero_incident_week_emits_quiet_week_text() -> None:
    """When no issues in lookback window, handoff text contains 'Quiet week'."""
    # Build world with no topics in window range
    now = datetime(2026, 4, 28, tzinfo=UTC)
    world = _build_oncall_world(window_end=now, n_issues=0)
    ownership = _make_ownership(world)
    window = TimeWindow(end=now, days=7)
    specs = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    if specs:
        slack_doc = next(d for d in specs[0].doc_specs if d.source == Source.SLACK)
        assert "Quiet week" in slack_doc.text


def test_rotation_determinism() -> None:
    """Same inputs always produce the same outgoing/incoming pair."""
    world = _build_oncall_world(window_end=datetime(2026, 5, 4, tzinfo=UTC))
    ownership = _make_ownership(world)
    window = TimeWindow(end=datetime(2026, 5, 4, tzinfo=UTC), days=14)
    specs1 = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    specs2 = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    for s1, s2 in zip(specs1, specs2):
        assert s1.cast == s2.cast
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_archetype_oncall.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.archetypes.oncall'`

- [ ] **Step 3: Implement ON_CALL_HANDOFF**

Create `scripts/synth/archetypes/oncall.py`:

```python
"""ON_CALL_HANDOFF archetype — weekly Slack + Notion on-call handoff.

For each Monday in time_window:
  1. Pick outgoing = top_personas[week_index % top_n]
     incoming = top_personas[(week_index + 1) % top_n]
     week_index = ISO week number of the Monday.
  2. Pick up to 3 incidents = topics with kind==ISSUE (fallback COMMIT)
     and ts in [monday - 7d, monday).
  3. Emit 3 DocSpecs:
     - slack-0: Slack parent in #oncall from outgoing, summarizing incidents.
     - slack-1: Slack reply from incoming acknowledging.
     - notion-0: Notion page "On-call handoff <date>" with H2 per incident.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, UTC

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    DocSpec,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.world_model import TopicKind, WorldModel

ON_CALL_HANDOFF = Archetype(
    name="ON_CALL_HANDOFF",
    category=Category.RECURRING,
    cadence=Cadence.WEEKLY,
    sources_used=(Source.SLACK, Source.NOTION),
    cast_size=(2, 2),
    needs_planner_call=False,
    validator_level=ValidatorLevel.NAME_ONLY,
)


def _mondays(window_end: datetime, days: int) -> list[date]:
    """Return all Mondays in [window_end - days, window_end], ascending."""
    result: list[date] = []
    end_date = window_end.date()
    start_date = end_date - timedelta(days=days - 1)
    current = start_date
    # Advance to first Monday
    while current.weekday() != 0:
        current += timedelta(days=1)
    while current <= end_date:
        result.append(current)
        current += timedelta(days=7)
    return result


def _incident_summary(incidents: list) -> str:
    """Render incident list to a bullet summary for the Slack parent message."""
    if not incidents:
        return "Quiet week, nothing on fire."
    lines = ["Incidents this week:"]
    for t in incidents[:3]:
        lines.append(f"- {t.text[:80]}")
    return "\n".join(lines)


def _notion_body(day: date, incidents: list, outgoing_id: str) -> str:
    """Render Notion page body with H2 per incident."""
    lines = [f"On-call handoff {day.isoformat()}", f"Outgoing: @{outgoing_id.replace('gh:', '')}"]
    if not incidents:
        lines.append("## Status")
        lines.append("Quiet week, nothing on fire.")
    else:
        for t in incidents[:3]:
            lines.append(f"## {t.text[:80]}")
            lines.append("Owner: TBD")
            lines.append("Status: resolved")
    return "\n".join(lines)


def build_oncall_specs(
    world: WorldModel,
    ownership: OwnershipIndex,
    time_window: object,
    seed: int,
    top_n: int = 5,
) -> tuple[ScenarioSpec, ...]:
    """Build ON_CALL_HANDOFF ScenarioSpecs for each Monday in time_window."""
    end: datetime = time_window.end  # type: ignore[attr-defined]
    days: int = time_window.days  # type: ignore[attr-defined]
    mondays = _mondays(end, days)

    # Top-N personas by activity_score desc, canonical_id tie-break.
    sorted_people = sorted(
        world.people,
        key=lambda p: (-p.activity_score, p.canonical_id),
    )[:top_n]

    if len(sorted_people) < 2:
        return ()

    specs: list[ScenarioSpec] = []

    for monday in mondays:
        # Use ISO week number as rotation index for determinism.
        week_index = monday.isocalendar()[1]
        outgoing = sorted_people[week_index % len(sorted_people)]
        incoming = sorted_people[(week_index + 1) % len(sorted_people)]

        monday_dt = datetime(monday.year, monday.month, monday.day, 9, 0, 0, tzinfo=UTC)
        lookback = monday_dt - timedelta(days=7)

        # Pick incidents: ISSUE kind preferred, COMMIT as fallback.
        all_in_window = [
            t for t in world.topic_pool
            if t.ts is not None and lookback <= t.ts < monday_dt
        ]
        issues = [t for t in all_in_window if t.kind == TopicKind.ISSUE]
        incidents = issues[:3] if issues else [t for t in all_in_window if t.kind == TopicKind.COMMIT][:3]

        # Stable ordering for determinism: sort by (-weight, text).
        incidents.sort(key=lambda t: (-t.weight, t.text))

        date_str = monday.isoformat()
        slack_parent_id = f"scn-oncall-{date_str}-slack-0"
        slack_reply_id = f"scn-oncall-{date_str}-slack-1"
        notion_id = f"scn-oncall-{date_str}-notion-0"
        scenario_id = f"scn-oncall-{date_str}"

        # slack-0: parent from outgoing
        outgoing_text = _incident_summary(incidents)
        slack_parent = DocSpec(
            id=slack_parent_id,
            source=Source.SLACK,
            occurred_at=monday_dt,
            channel="#oncall",
            page_section=None,
            text=outgoing_text,
            thread_parent_id=None,
            personas=(outgoing.canonical_id,),
            services_mentioned=tuple(ownership.services_by_person.get(outgoing.canonical_id, ())),
        )

        # slack-1: reply from incoming
        incoming_text = "Got it, taking over. Will monitor and escalate if anything resurfaces."
        slack_reply = DocSpec(
            id=slack_reply_id,
            source=Source.SLACK,
            occurred_at=monday_dt,
            channel="#oncall",
            page_section=None,
            text=incoming_text,
            thread_parent_id=slack_parent_id,
            personas=(incoming.canonical_id,),
            services_mentioned=tuple(ownership.services_by_person.get(incoming.canonical_id, ())),
        )

        # notion-0: handoff page
        notion_text = _notion_body(monday, incidents, outgoing.canonical_id)
        page_id = f"page-{notion_id}"
        notion_page = DocSpec(
            id=notion_id,
            source=Source.NOTION,
            occurred_at=monday_dt,
            channel=None,
            page_section="Engineering > On-call rotation",
            text=notion_text,
            thread_parent_id=None,
            personas=(outgoing.canonical_id, incoming.canonical_id),
            services_mentioned=tuple({
                s
                for cid in (outgoing.canonical_id, incoming.canonical_id)
                for s in ownership.services_by_person.get(cid, ())
            }),
        )

        all_cast = (outgoing.canonical_id, incoming.canonical_id)
        all_services = tuple({
            s
            for cid in all_cast
            for s in ownership.services_by_person.get(cid, ())
        })

        spec = ScenarioSpec(
            id=scenario_id,
            archetype_name="ON_CALL_HANDOFF",
            instance_ts=monday_dt,
            cast=all_cast,
            affected_services=all_services,
            doc_specs=(slack_parent, slack_reply, notion_page),
        )
        specs.append(spec)

    return tuple(specs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_archetype_oncall.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/archetypes/oncall.py tests/synth/test_archetype_oncall.py
git commit -m "$(cat <<'EOF'
feat(synth): ON_CALL_HANDOFF archetype with weekly rotation and Slack+Notion docs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Archetype library registry

**Files:**
- Create: `scripts/synth/archetypes/library.py`
- Test: `tests/synth/test_archetype_library.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the archetype library registry."""

from __future__ import annotations

from scripts.synth.archetypes.base import Archetype
from scripts.synth.archetypes.library import (
    ARCHETYPE_LIBRARY,
    BUILDERS,
    get_active,
)
from scripts.synth.archetypes.oncall import ON_CALL_HANDOFF, build_oncall_specs
from scripts.synth.archetypes.standup import STANDUP_UPDATE, build_standup_specs
from scripts.synth.profile import Profile


def _profile(raw: dict | None = None) -> Profile:
    """Construct a minimal Profile for tests, with overridable raw dict."""
    raw = raw or {}
    base = {
        "customer_id": "cust-eval-test-01",
        "repos": [{"url": "github.com/x/y", "local_path": "/tmp/y"}],
        "preset": "tiny-test",
        "seed": 42,
    }
    base.update(raw)
    return Profile(
        customer_id=base["customer_id"],
        repos=(),
        preset=base["preset"],
        seed=base["seed"],
        raw=base,
    )


def test_library_contains_both_archetypes() -> None:
    assert set(ARCHETYPE_LIBRARY.keys()) == {"STANDUP_UPDATE", "ON_CALL_HANDOFF"}
    assert isinstance(ARCHETYPE_LIBRARY["STANDUP_UPDATE"], Archetype)
    assert ARCHETYPE_LIBRARY["STANDUP_UPDATE"] is STANDUP_UPDATE
    assert ARCHETYPE_LIBRARY["ON_CALL_HANDOFF"] is ON_CALL_HANDOFF


def test_builders_resolve_to_correct_functions() -> None:
    assert BUILDERS["STANDUP_UPDATE"] is build_standup_specs
    assert BUILDERS["ON_CALL_HANDOFF"] is build_oncall_specs


def test_get_active_default_returns_full_library() -> None:
    p = _profile()
    active = get_active(p)
    assert set(active.keys()) == {"STANDUP_UPDATE", "ON_CALL_HANDOFF"}


def test_get_active_respects_count_zero_disable() -> None:
    p = _profile({"archetypes": {"STANDUP_UPDATE": {"count": 0}}})
    active = get_active(p)
    assert set(active.keys()) == {"ON_CALL_HANDOFF"}


def test_get_active_respects_archetype_filter() -> None:
    p = _profile()
    active = get_active(p, archetype_filter=("STANDUP_UPDATE",))
    assert set(active.keys()) == {"STANDUP_UPDATE"}


def test_get_active_filter_intersects_with_count_disable() -> None:
    p = _profile({"archetypes": {"STANDUP_UPDATE": {"count": 0}}})
    active = get_active(p, archetype_filter=("STANDUP_UPDATE", "ON_CALL_HANDOFF"))
    assert set(active.keys()) == {"ON_CALL_HANDOFF"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_archetype_library.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.archetypes.library'`.

- [ ] **Step 3: Implement the library**

```python
"""Archetype library — central registry of recurring archetypes.

Plan 2 ships two: STANDUP_UPDATE (daily slack) and ON_CALL_HANDOFF (weekly
slack+notion). Plan 3 will register plot archetypes (INCIDENT, LAUNCH,
BIG_REFACTOR, etc.) here alongside their LLM-driven builders.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from scripts.synth.archetypes.base import Archetype, ScenarioSpec
from scripts.synth.archetypes.oncall import ON_CALL_HANDOFF, build_oncall_specs
from scripts.synth.archetypes.standup import STANDUP_UPDATE, build_standup_specs

if TYPE_CHECKING:
    from scripts.synth.profile import Profile


ARCHETYPE_LIBRARY: dict[str, Archetype] = {
    "STANDUP_UPDATE": STANDUP_UPDATE,
    "ON_CALL_HANDOFF": ON_CALL_HANDOFF,
}

# Builder signatures vary slightly across archetypes (top_n is a kwarg with
# default), so the registry is typed loosely. Callers (run_scenarios) pass
# only the positional args common to all builders.
BUILDERS: dict[str, Callable[..., tuple[ScenarioSpec, ...]]] = {
    "STANDUP_UPDATE": build_standup_specs,
    "ON_CALL_HANDOFF": build_oncall_specs,
}


def get_active(
    profile: Profile,
    archetype_filter: tuple[str, ...] | None = None,
) -> dict[str, Archetype]:
    """Resolve the set of archetypes to run for this profile.

    Profile's optional `archetypes:` block lets the user disable a per-name
    archetype with `count: 0`. CLI's `--archetypes A,B` further restricts
    via `archetype_filter`. Both filters compose (intersection).
    """
    profile_arch = profile.raw.get("archetypes") or {}
    active: dict[str, Archetype] = {}
    for name, archetype in ARCHETYPE_LIBRARY.items():
        cfg = profile_arch.get(name) or {}
        count = cfg.get("count")
        if count == 0:
            continue
        active[name] = archetype
    if archetype_filter is not None:
        active = {k: v for k, v in active.items() if k in archetype_filter}
    return active
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_archetype_library.py -v`
Expected: PASS — 6 tests.

Also: `.venv/bin/ruff check scripts/synth/archetypes/library.py tests/synth/test_archetype_library.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/archetypes/library.py tests/synth/test_archetype_library.py
git commit -m "$(cat <<'EOF'
feat(synth): archetype library registry with disable + filter resolution

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

---

## Task 10: ScenarioRunner + TimeWindow

**Files:**
- Create: `scripts/synth/scenarios.py`
- Test: `tests/synth/test_scenarios.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the scenario runner: TimeWindow + working_days + weekly_mondays + run_scenarios."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    DocSpec,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.profile import Profile
from scripts.synth.scenarios import (
    TimeWindow,
    run_scenarios,
    weekly_mondays,
    working_days,
)
from scripts.synth.world_model import (
    Person,
    Service,
    ServiceKind,
    Topic,
    TopicKind,
    WorldModel,
)


def _profile(raw_extras: dict | None = None) -> Profile:
    raw = {
        "customer_id": "cust-eval-test-01",
        "repos": [{"url": "github.com/x/y", "local_path": "/tmp/y"}],
        "preset": "tiny-test",
        "seed": 7,
    }
    if raw_extras:
        raw.update(raw_extras)
    return Profile(
        customer_id=raw["customer_id"],
        repos=(),
        preset=raw["preset"],
        seed=raw["seed"],
        raw=raw,
    )


def _build_test_world() -> WorldModel:
    """Tiny WorldModel with 3 people, 2 services, 5 topics."""
    people = (
        Person(
            canonical_id="gh:alice",
            gh_username="alice",
            display_name="Alice",
            email_aliases=("alice@x.com",),
            role_hint=None,
            repos_active_in=("github.com/x/y",),
            activity_score=10.0,
        ),
        Person(
            canonical_id="gh:bob",
            gh_username="bob",
            display_name="Bob",
            email_aliases=("bob@x.com",),
            role_hint=None,
            repos_active_in=("github.com/x/y",),
            activity_score=5.0,
        ),
        Person(
            canonical_id="gh:carol",
            gh_username="carol",
            display_name="Carol",
            email_aliases=("carol@x.com",),
            role_hint=None,
            repos_active_in=("github.com/x/y",),
            activity_score=2.0,
        ),
    )
    services = (
        Service(
            name="payments", qualified="payments", repo_url="github.com/x/y",
            kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0,
            deploy_target=None,
        ),
        Service(
            name="billing", qualified="billing", repo_url="github.com/x/y",
            kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0,
            deploy_target=None,
        ),
    )
    topics = (
        Topic(text="fix payments null", kind=TopicKind.COMMIT, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 28, tzinfo=UTC),
              mentioned_services=("payments",), mentioned_people=("gh:alice",), weight=0.8),
        Topic(text="billing rate limit", kind=TopicKind.ISSUE, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 27, tzinfo=UTC),
              mentioned_services=("billing",), mentioned_people=("gh:bob",), weight=0.7),
        Topic(text="payments retry logic", kind=TopicKind.PR, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 26, tzinfo=UTC),
              mentioned_services=("payments",), mentioned_people=("gh:alice",), weight=0.9),
        Topic(text="billing dashboard", kind=TopicKind.PR, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 25, tzinfo=UTC),
              mentioned_services=("billing",), mentioned_people=("gh:carol",), weight=0.6),
        Topic(text="db migration", kind=TopicKind.ISSUE, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 24, tzinfo=UTC),
              mentioned_services=("payments", "billing"), mentioned_people=(), weight=0.5),
    )
    return WorldModel(
        repos=(),
        people=people,
        services=services,
        topic_pool=topics,
        channels=(),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="acme",
        seed=7,
        extracted_at=datetime(2026, 4, 30, tzinfo=UTC),
        sha_set={},
    )


def _ownership_full() -> OwnershipIndex:
    return OwnershipIndex(
        services_by_person={
            "gh:alice": ("payments",),
            "gh:bob": ("billing",),
            "gh:carol": ("billing",),
        },
        people_by_service={
            "payments": ("gh:alice",),
            "billing": ("gh:bob", "gh:carol"),
        },
    )


# --- working_days / weekly_mondays ----------------------------------------

def test_working_days_excludes_weekends() -> None:
    # Window: Mon Apr 27 -> Fri May 1 (5 weekdays). end is exclusive.
    window = TimeWindow(end=datetime(2026, 5, 2, tzinfo=UTC), days=5)
    days = list(working_days(window))
    assert days == [
        date(2026, 4, 27),
        date(2026, 4, 28),
        date(2026, 4, 29),
        date(2026, 4, 30),
        date(2026, 5, 1),
    ]


def test_working_days_zero_days_yields_empty() -> None:
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=0)
    assert list(working_days(window)) == []


def test_weekly_mondays_finds_mondays_in_window() -> None:
    # 30-day window ending 2026-05-01 spans 2026-04-01 .. 2026-04-30.
    # Mondays in that range: Apr 6, Apr 13, Apr 20, Apr 27 (4 mondays).
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=30)
    mondays = list(weekly_mondays(window))
    assert mondays == [
        date(2026, 4, 6),
        date(2026, 4, 13),
        date(2026, 4, 20),
        date(2026, 4, 27),
    ]


def test_weekly_mondays_window_starting_on_monday() -> None:
    # If start_date itself is a Monday, include it.
    window = TimeWindow(end=datetime(2026, 4, 14, tzinfo=UTC), days=7)
    mondays = list(weekly_mondays(window))
    assert mondays == [date(2026, 4, 6), date(2026, 4, 13)]


# --- run_scenarios --------------------------------------------------------

def test_run_scenarios_yields_docs_for_full_library() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=14)
    docs = list(run_scenarios(world, own, p, window))
    # Expect both archetypes to produce docs (count varies; just assert > 0).
    sources = {d.source for d in docs}
    assert Source.SLACK in sources
    assert Source.NOTION in sources


def test_run_scenarios_archetype_filter_restricts_output() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=14)
    docs = list(run_scenarios(world, own, p, window, archetype_filter=("STANDUP_UPDATE",)))
    # STANDUP_UPDATE is slack-only; no notion docs.
    assert all(d.source == Source.SLACK for d in docs)
    assert all(d.archetype == "STANDUP_UPDATE" for d in docs)


def test_run_scenarios_scenario_limit_caps_per_archetype() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=30)
    docs = list(run_scenarios(world, own, p, window, scenario_limit=2))
    # Each archetype contributes <= 2 scenarios (each scenario yields 1+ docs).
    standup_scenarios = {d.scenario_id for d in docs if d.archetype == "STANDUP_UPDATE"}
    oncall_scenarios = {d.scenario_id for d in docs if d.archetype == "ON_CALL_HANDOFF"}
    assert len(standup_scenarios) <= 2
    assert len(oncall_scenarios) <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_scenarios.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.scenarios'`.

- [ ] **Step 3: Implement the runner**

```python
"""ScenarioRunner — walks the active archetype set, builds specs, materializes
SynthDocs. Plan 2 only sees templated builders; Plan 3 will branch on
`archetype.needs_planner_call` to invoke an LLM Planner instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from scripts.synth.archetypes.base import DocSpec, ScenarioSpec, Source
from scripts.synth.archetypes.library import BUILDERS, get_active
from scripts.synth.output.base import SynthDoc

if TYPE_CHECKING:
    from scripts.synth.ownership import OwnershipIndex
    from scripts.synth.profile import Profile
    from scripts.synth.world_model import WorldModel


@dataclass(frozen=True)
class TimeWindow:
    """Half-open window [end - days, end). All times UTC."""

    end: datetime
    days: int


def working_days(window: TimeWindow) -> Iterator[date]:
    """Mon-Fri dates in the window, chronological."""
    start = (window.end - timedelta(days=window.days)).date()
    stop = window.end.date()
    cursor = start
    while cursor < stop:
        if cursor.weekday() < 5:  # 0=Mon ... 4=Fri
            yield cursor
        cursor = cursor + timedelta(days=1)


def weekly_mondays(window: TimeWindow) -> Iterator[date]:
    """Mondays inside the window, chronological."""
    start = (window.end - timedelta(days=window.days)).date()
    stop = window.end.date()
    if start.weekday() == 0:
        cursor = start
    else:
        cursor = start + timedelta(days=(7 - start.weekday()) % 7)
    while cursor < stop:
        yield cursor
        cursor = cursor + timedelta(days=7)


def _materialize(doc_spec: DocSpec, scenario: ScenarioSpec) -> SynthDoc:
    """Convert a planner-emitted DocSpec into the wire-shaped SynthDoc."""
    return SynthDoc(
        id=doc_spec.id,
        source=doc_spec.source,
        source_event_id=doc_spec.id,
        text=doc_spec.text,
        occurred_at=doc_spec.occurred_at,
        channel=doc_spec.channel,
        page_id=doc_spec.id if doc_spec.source == Source.NOTION else None,
        thread_parent_id=doc_spec.thread_parent_id,
        scenario_id=scenario.id,
        archetype=scenario.archetype_name,
        personas=doc_spec.personas,
        services_mentioned=doc_spec.services_mentioned,
        priority=100,
    )


def run_scenarios(
    world: WorldModel,
    ownership: OwnershipIndex,
    profile: Profile,
    time_window: TimeWindow,
    *,
    archetype_filter: tuple[str, ...] | None = None,
    scenario_limit: int | None = None,
) -> Iterator[SynthDoc]:
    """Walk active archetypes, run their builders, materialize SynthDocs.

    `scenario_limit` caps PER ARCHETYPE (each builder's output is sliced).
    """
    active = get_active(profile, archetype_filter=archetype_filter)
    for name in active:
        builder = BUILDERS[name]
        specs = builder(world, ownership, time_window, profile.seed)
        if scenario_limit is not None:
            specs = specs[:scenario_limit]
        for spec in specs:
            for doc_spec in spec.doc_specs:
                yield _materialize(doc_spec, spec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_scenarios.py -v`
Expected: PASS — 7 tests.

Also: `.venv/bin/ruff check scripts/synth/scenarios.py tests/synth/test_scenarios.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/scenarios.py tests/synth/test_scenarios.py
git commit -m "$(cat <<'EOF'
feat(synth): ScenarioRunner with TimeWindow, working_days, weekly_mondays

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

---

## Task 11: IngestionWriter (local mode)

Plan 2 ships local mode here. Integrate mode (R2 + ingestion_queue) is added in Task 14 by extending this same class.

**Files:**
- Create: `scripts/synth/output/writer.py`
- Test: `tests/synth/test_ingestion_writer.py`

### Step 1: Write the failing test

- [ ] **Step 1: Write the failing test** — create `tests/synth/test_ingestion_writer.py`:

```python
"""Tests for IngestionWriter local mode (Plan 2 Task 11).

Integrate-mode tests are added in Task 14.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson
import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.writer import IngestionWriter


def _slack_doc(source_event_id: str = "doc-1") -> SynthDoc:
    return SynthDoc(
        id=source_event_id,
        source=Source.SLACK,
        source_event_id=source_event_id,
        text="hello",
        occurred_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        channel="#standup",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-x",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments",),
    )


def _notion_doc(source_event_id: str = "page-1") -> SynthDoc:
    return SynthDoc(
        id=source_event_id,
        source=Source.NOTION,
        source_event_id=source_event_id,
        text="On-call handoff page body",
        occurred_at=datetime(2026, 5, 4, 10, 0, tzinfo=UTC),
        channel=None,
        page_id=source_event_id,
        thread_parent_id=None,
        scenario_id="scn-y",
        archetype="ON_CALL_HANDOFF",
        personas=("gh:alice", "gh:bob"),
        services_mentioned=("payments",),
    )


@pytest.mark.asyncio
async def test_local_writes_slack_envelope_to_disk(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_slack_doc("doc-1"))
    await writer.close()
    path = tmp_path / "raw" / "slack" / "doc-1.json"
    assert path.exists()
    payload = orjson.loads(path.read_bytes())
    assert payload["type"] == "event_callback"


@pytest.mark.asyncio
async def test_local_writes_notion_envelope_to_disk(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_notion_doc("page-1"))
    await writer.close()
    path = tmp_path / "raw" / "notion" / "page-1.json"
    assert path.exists()
    payload = orjson.loads(path.read_bytes())
    assert payload["type"] == "page.updated"


@pytest.mark.asyncio
async def test_local_overwrite_on_repeat_write(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_slack_doc("doc-1"))
    await writer.write(_slack_doc("doc-1"))  # second write must not raise
    path = tmp_path / "raw" / "slack" / "doc-1.json"
    assert path.exists()


@pytest.mark.asyncio
async def test_local_unsupported_source_raises(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    doc = SynthDoc(
        id="x",
        source=Source.GITHUB,  # GitHub wrapper deferred to Plan 3
        source_event_id="x",
        text="",
        occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
        channel=None,
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-z",
        archetype="STANDUP_UPDATE",
        personas=(),
        services_mentioned=(),
    )
    with pytest.raises(ValueError, match="Plan 2 doesn't support source"):
        await writer.write(doc)
```

### Step 2: Run test to verify it fails

- [ ] **Step 2: Run test to verify it fails** — Run: `.venv/bin/pytest tests/synth/test_ingestion_writer.py -v` — Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.output.writer'`.

### Step 3: Implement local mode

- [ ] **Step 3: Implement local mode** — create `scripts/synth/output/writer.py`:

```python
"""IngestionWriter — writes SynthDocs to local files. Integrate mode (R2 +
ingestion_queue) is added in Task 14 by extending this class.

In both modes, every write produces a local file under <out_dir>/raw/<source>/
for human inspection. Integrate mode additionally calls bucket.put and
batches inserts into ingestion_queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from scripts.synth.archetypes.base import Source
from scripts.synth.output import notion as notion_wrapper
from scripts.synth.output import slack as slack_wrapper
from scripts.synth.output.base import SynthDoc


class IngestionWriter:
    """Plan 2 local-only writer. Task 14 extends with integrate mode."""

    def __init__(self, *, out_dir: Path, mode: Literal["local"] = "local") -> None:
        self.out_dir = out_dir
        self.mode = mode

    async def write(self, doc: SynthDoc) -> None:
        envelope = self._envelope(doc)
        path = self.out_dir / "raw" / doc.source.value / f"{doc.source_event_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(envelope)

    async def close(self) -> None:
        # Local mode has nothing to flush. Integrate mode (Task 14) overrides.
        return None

    def _envelope(self, doc: SynthDoc) -> bytes:
        if doc.source == Source.SLACK:
            return slack_wrapper.wrap(doc)
        if doc.source == Source.NOTION:
            return notion_wrapper.wrap(doc)
        raise ValueError(
            f"Plan 2 doesn't support source: {doc.source.value}. "
            "GitHub/Linear/Sentry/Granola wrappers land in Plan 3."
        )
```

### Step 4: Run test to verify it passes

- [ ] **Step 4: Run test to verify it passes** — Run: `.venv/bin/pytest tests/synth/test_ingestion_writer.py -v` — Expected: PASS — 4 tests.

Also: `.venv/bin/ruff check scripts/synth/output/writer.py tests/synth/test_ingestion_writer.py` — Expected: clean.

### Step 5: Commit

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/output/writer.py tests/synth/test_ingestion_writer.py
git commit -m "$(cat <<'EOF'
feat(synth): IngestionWriter local mode + per-source envelope routing

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

---

## Task 12: Eval artifact writers

**Files:**
- Create: `scripts/synth/output/eval_artifacts.py`
- Test: `tests/synth/test_eval_artifacts.py`

### Step 1: Write the failing test

- [ ] **Step 1: Write the failing test**

```python
"""Tests for eval artifact writers: manifest.json, docs_index.jsonl, profile.yaml, warnings.log."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson
import yaml

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.eval_artifacts import (
    write_docs_index,
    write_manifest,
    write_profile,
    write_warnings,
)
from scripts.synth.profile import Profile
from scripts.synth.validator import Violation
from scripts.synth.world_model import WorldModel


def _profile() -> Profile:
    raw = {
        "customer_id": "cust-eval-test-01",
        "repos": [{"url": "github.com/x/y", "local_path": "/tmp/y"}],
        "preset": "tiny-test",
        "seed": 42,
    }
    return Profile(
        customer_id=raw["customer_id"],
        repos=(),
        preset=raw["preset"],
        seed=raw["seed"],
        raw=raw,
    )


def _world() -> WorldModel:
    return WorldModel(
        repos=(),
        people=(),
        services=(),
        topic_pool=(),
        channels=(),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="acme",
        seed=42,
        extracted_at=datetime(2026, 5, 1, tzinfo=UTC),
        sha_set={"github.com/x/y": "abc123"},
    )


def _doc(source: Source, source_event_id: str, occurred_at: datetime) -> SynthDoc:
    return SynthDoc(
        id=source_event_id,
        source=source,
        source_event_id=source_event_id,
        text="x",
        occurred_at=occurred_at,
        channel="#standup" if source == Source.SLACK else None,
        page_id=source_event_id if source == Source.NOTION else None,
        thread_parent_id=None,
        scenario_id="scn-1",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments",),
    )


def test_write_manifest_produces_expected_keys(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        run_id="run-x",
        profile=_profile(),
        world=_world(),
        totals={
            "archetypes_executed": {"STANDUP_UPDATE": {"requested": 5, "generated": 5, "dropped": 0}},
            "totals": {"scenarios": 5, "documents": 5, "questions": 0},
            "warnings_count": 0,
        },
        mode="local",
        started_at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 1, 10, 5, tzinfo=UTC),
    )
    payload = orjson.loads((tmp_path / "manifest.json").read_bytes())
    assert payload["run_id"] == "run-x"
    assert payload["seed"] == 42
    assert payload["customer_id"] == "cust-eval-test-01"
    assert payload["mode"] == "local"
    assert payload["world_model"]["people_count"] == 0
    assert payload["archetypes_executed"]["STANDUP_UPDATE"]["generated"] == 5
    assert payload["totals"]["scenarios"] == 5


def test_write_docs_index_orders_by_occurred_at_then_id(tmp_path: Path) -> None:
    docs = [
        _doc(Source.SLACK, "doc-2", datetime(2026, 5, 1, 9, 0, tzinfo=UTC)),
        _doc(Source.SLACK, "doc-1", datetime(2026, 5, 1, 8, 0, tzinfo=UTC)),
        _doc(Source.NOTION, "doc-3", datetime(2026, 5, 1, 9, 0, tzinfo=UTC)),
    ]
    write_docs_index(tmp_path, docs)
    raw = (tmp_path / "docs_index.jsonl").read_text().strip().split("\n")
    rows = [orjson.loads(line) for line in raw]
    assert [r["doc_id"] for r in rows] == ["doc-1", "doc-2", "doc-3"]
    assert rows[0]["raw_key"] == "raw/slack/doc-1.json"
    assert rows[2]["raw_key"] == "raw/notion/doc-3.json"


def test_write_profile_dumps_raw_yaml(tmp_path: Path) -> None:
    write_profile(tmp_path, _profile())
    parsed = yaml.safe_load((tmp_path / "profile.yaml").read_text())
    assert parsed["customer_id"] == "cust-eval-test-01"
    assert parsed["seed"] == 42


def test_write_warnings_formats_violations_and_notes(tmp_path: Path) -> None:
    violations = (
        Violation(doc_id="doc-1", out_of_world=("foo",)),
        Violation(doc_id="doc-2", out_of_world=("bar", "baz")),
    )
    notes = ["dropped 1 scenario without recent topics"]
    write_warnings(tmp_path, violations, notes)
    text = (tmp_path / "warnings.log").read_text()
    assert "VIOLATION: doc=doc-1" in text
    assert "VIOLATION: doc=doc-2" in text
    assert "NOTE: dropped 1 scenario" in text


def test_write_docs_index_empty_input_writes_empty_file(tmp_path: Path) -> None:
    write_docs_index(tmp_path, [])
    path = tmp_path / "docs_index.jsonl"
    assert path.exists()
    assert path.read_bytes() == b""


def test_write_warnings_empty_input_writes_empty_file(tmp_path: Path) -> None:
    write_warnings(tmp_path, (), [])
    path = tmp_path / "warnings.log"
    assert path.exists()
    assert path.read_text() == ""
```

### Step 2: Run test to verify it fails

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/synth/test_eval_artifacts.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.output.eval_artifacts'`.

### Step 3: Implement

- [ ] **Step 3: Implement**

```python
"""Eval artifact writers — manifest.json, docs_index.jsonl, profile.yaml, warnings.log.

Each writer is deterministic given its inputs (modulo wall-clock fields like
`started_at` / `finished_at` in the manifest, which the caller passes in).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
import yaml

if TYPE_CHECKING:
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.profile import Profile
    from scripts.synth.validator import Violation
    from scripts.synth.world_model import WorldModel


def write_manifest(
    out_dir: Path,
    *,
    run_id: str,
    profile: Profile,
    world: WorldModel,
    totals: dict,
    mode: str,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Write the run-level summary to `<out_dir>/manifest.json`."""
    manifest = {
        "run_id": run_id,
        "profile_name": profile.preset,
        "seed": profile.seed,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "customer_id": profile.customer_id,
        "mode": mode,
        "repos": [
            {
                "url": repo.url,
                "sha": world.sha_set.get(repo.url, "unknown"),
                "mode": "local",
            }
            for repo in profile.repos
        ],
        "world_model": {
            "people_count": len(world.people),
            "services_count": len(world.services),
            "channels_count": len(world.channels),
        },
        "archetypes_executed": totals.get("archetypes_executed", {}),
        "totals": totals.get("totals", {}),
        "warnings_count": totals.get("warnings_count", 0),
    }
    (out_dir / "manifest.json").write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2)
    )


def write_docs_index(out_dir: Path, docs: list[SynthDoc]) -> None:
    """Write one row per SynthDoc, sorted (occurred_at, id) for determinism."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_docs = sorted(docs, key=lambda d: (d.occurred_at, d.id))
    lines: list[bytes] = []
    for doc in sorted_docs:
        row = {
            "doc_id": doc.id,
            "scenario_id": doc.scenario_id,
            "archetype": doc.archetype,
            "source": doc.source.value,
            "occurred_at": doc.occurred_at.isoformat(),
            "raw_key": f"raw/{doc.source.value}/{doc.source_event_id}.json",
            "personas": list(doc.personas),
            "services_mentioned": list(doc.services_mentioned),
            "is_evidence_for_question_ids": [],
        }
        lines.append(orjson.dumps(row))
    payload = b"\n".join(lines) + (b"\n" if lines else b"")
    (out_dir / "docs_index.jsonl").write_bytes(payload)


def write_profile(out_dir: Path, profile: Profile) -> None:
    """Freeze the resolved profile as YAML next to the run output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile.yaml").write_text(
        yaml.safe_dump(profile.raw, sort_keys=False)
    )


def write_warnings(
    out_dir: Path,
    violations: tuple[Violation, ...],
    notes: list[str],
) -> None:
    """Plain-text log of validator violations + freeform notes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for v in violations:
        lines.append(f"VIOLATION: doc={v.doc_id} out_of_world={list(v.out_of_world)}")
    lines.extend(f"NOTE: {n}" for n in notes)
    payload = "\n".join(lines) + ("\n" if lines else "")
    (out_dir / "warnings.log").write_text(payload)
```

### Step 4: Run test to verify it passes

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/synth/test_eval_artifacts.py -v`

Expected: PASS — 6 tests.

Also: `.venv/bin/ruff check scripts/synth/output/eval_artifacts.py tests/synth/test_eval_artifacts.py` — Expected: clean.

### Step 5: Commit

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/output/eval_artifacts.py tests/synth/test_eval_artifacts.py
git commit -m "$(cat <<'EOF'
feat(synth): eval artifact writers (manifest, docs_index, profile, warnings)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```
```

---

## Task 13: TenantBootstrap (init + clean) + ObjectStore.delete

**Files:**
- Create: `scripts/synth/bootstrap.py`
- Modify: `shared/storage.py` (add a per-key `delete` method — see Step 1.5 below)
- Test: `tests/synth/test_bootstrap.py`

This task lands the prefix-guarded tenant-init/clean pair AND adds a new method to `shared.storage.ObjectStore` that the clean path needs.

### Step 1: Add `delete` to ObjectStore

`shared/storage.py` currently has `put`, `get`, `exists`, `delete_bucket_recursive`, and `list_keys` but no per-key `delete`. The file uses `boto3` (sync) wrapped in `asyncio.to_thread`. Add the following method directly after `list_keys`:

```python
async def delete(self, bucket: str, key: str) -> None:
    """Delete a single object. Idempotent — silent on missing key."""
    def _delete() -> None:
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404"}:
                return
            raise StorageUnavailable(f"delete_object failed: {exc}") from exc

    await asyncio.to_thread(_delete)
```

### Step 2: Write the failing test

- [ ] **Step 2: Write `tests/synth/test_bootstrap.py`** (full file):

```python
"""Tests for TenantBootstrap (init + clean) + the new ObjectStore.delete shim."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.synth.bootstrap import (
    CUSTOMER_OWNED_TABLES,
    clean_tenant,
    init_tenant,
)
from scripts.synth.profile import Profile


def _profile(customer_id: str = "cust-eval-test-01") -> Profile:
    raw = {
        "customer_id": customer_id,
        "repos": [{"url": "github.com/x/y", "local_path": "/tmp/y"}],
        "preset": "tiny-test",
        "seed": 42,
        "sources": ["slack", "notion"],
    }
    return Profile(
        customer_id=raw["customer_id"],
        repos=(),
        preset=raw["preset"],
        seed=raw["seed"],
        raw=raw,
    )


def _mock_db() -> AsyncMock:
    db = AsyncMock()
    db.execute = AsyncMock(return_value=None)
    db.transaction = MagicMock()
    db.transaction.return_value.__aenter__ = AsyncMock()
    db.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


def _mock_bucket() -> AsyncMock:
    bucket = AsyncMock()
    bucket.bucket_for = MagicMock(return_value="prbe-synth-bucket")
    bucket.ensure_bucket = AsyncMock(return_value=None)
    bucket.list_keys = AsyncMock(return_value=[])
    bucket.delete = AsyncMock(return_value=None)
    return bucket


@pytest.mark.asyncio
async def test_init_tenant_inserts_customer_and_tokens() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    profile = _profile()
    await init_tenant(profile, db, bucket)
    # customers insert + 2 source token inserts (slack, notion)
    assert db.execute.await_count == 3
    calls = [c.args[0] for c in db.execute.await_args_list]
    assert any("INSERT INTO customers" in q for q in calls)
    assert sum("INSERT INTO integration_tokens" in q for q in calls) == 2
    bucket.ensure_bucket.assert_awaited_once()


@pytest.mark.asyncio
async def test_init_tenant_idempotent_on_repeat() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    profile = _profile()
    await init_tenant(profile, db, bucket)
    await init_tenant(profile, db, bucket)
    # Both runs are happy-path; ON CONFLICT DO NOTHING keeps things safe.
    # The mock doesn't simulate conflict, but we verify queries always carry
    # the ON CONFLICT clause so a real DB would handle it.
    for call in db.execute.await_args_list:
        assert "ON CONFLICT" in call.args[0]


@pytest.mark.asyncio
async def test_clean_tenant_refuses_non_synth_prefix() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    with pytest.raises(ValueError, match="refuse to clean non-synthetic"):
        await clean_tenant("prod-customer-01", db, bucket)
    db.execute.assert_not_awaited()
    bucket.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_clean_tenant_accepts_eval_prefix() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    await clean_tenant("cust-eval-test-01", db, bucket)
    # One DELETE per CUSTOMER_OWNED_TABLES entry.
    assert db.execute.await_count == len(CUSTOMER_OWNED_TABLES)


@pytest.mark.asyncio
async def test_clean_tenant_accepts_synth_prefix() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    await clean_tenant("cust-synth-test-01", db, bucket)
    assert db.execute.await_count == len(CUSTOMER_OWNED_TABLES)


@pytest.mark.asyncio
async def test_clean_tenant_deletes_only_keys_under_synth_prefix() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    bucket.list_keys = AsyncMock(return_value=[
        "raw/slack/cust-eval-test-01/synth/doc-1.json",
        "raw/notion/cust-eval-test-01/synth/page-1.json",
        "raw/slack/cust-OTHER-01/synth/doc-x.json",  # belongs to a different tenant
    ])
    await clean_tenant("cust-eval-test-01", db, bucket)
    # Only the two keys belonging to cust-eval-test-01/synth/ are deleted.
    assert bucket.delete.await_count == 2
    deleted_keys = [c.args[1] for c in bucket.delete.await_args_list]
    assert all("cust-eval-test-01/synth/" in k for k in deleted_keys)
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth.bootstrap'`

### Step 3: Implement `shared/storage.py` addition and `scripts/synth/bootstrap.py`

- [ ] **Step 3a: Add `delete` to `shared/storage.py`**

Exact diff to apply (insert after the closing `return await asyncio.to_thread(_list)` line of `list_keys`):

```python
# ADD after the existing list_keys method:

async def delete(self, bucket: str, key: str) -> None:
    """Delete a single object. Idempotent — silent on missing key."""
    def _delete() -> None:
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404"}:
                return
            raise StorageUnavailable(f"delete_object failed: {exc}") from exc

    await asyncio.to_thread(_delete)
```

- [ ] **Step 3b: Create `scripts/synth/bootstrap.py`** (full file):

```python
"""TenantBootstrap — idempotent customer init + prefix-guarded clean.

`init_tenant` ensures the customer row, R2 bucket, and stub integration_tokens
rows exist. `clean_tenant` is the dangerous one: hard-guarded by customer_id
prefix to refuse production tenants, then DELETE per known table + R2 prefix.

The customers row is NOT deleted — it stays as a "tenant exists" marker so
init can re-bind without race.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.logging import get_logger

if TYPE_CHECKING:
    from shared.storage import ObjectStore

    from scripts.synth.profile import Profile


log = get_logger(__name__)


# Tables whose rows belong to a customer. Order matters for the explicit
# DELETEs (children before parents) — though the FK CASCADE would handle
# it transparently if any single DELETE were skipped. Keeping explicit
# DELETEs for visibility + idempotency.
CUSTOMER_OWNED_TABLES: tuple[str, ...] = (
    "ingestion_queue",
    "chunks",
    "documents",
    "graph_edges",
    "graph_nodes",
    "graph_node_provenance",
    "integration_tokens",
    "acl_snapshots",
    "failed_chunks",
    "ingestion_events",
    "audit_log",
    "customer_source_mapping",
    "usage_events",
    "backfill_state",
)


async def init_tenant(profile: Profile, db, bucket: ObjectStore) -> None:
    """Idempotent tenant bootstrap.

    Creates the customers row, ensures the R2 bucket, and writes stub
    integration_tokens for each source the profile uses. Re-running is safe
    via ON CONFLICT DO NOTHING.
    """
    customer_id = profile.customer_id
    display_name = profile.raw.get("display_name") or f"synth-{customer_id}"
    sources = profile.raw.get("sources") or ["slack", "notion"]

    await db.execute(
        """
        INSERT INTO customers (customer_id, display_name, status)
        VALUES ($1, $2, 'active')
        ON CONFLICT (customer_id) DO NOTHING
        """,
        customer_id,
        display_name,
    )

    bucket_name = bucket.bucket_for(customer_id)
    await bucket.ensure_bucket(bucket_name)

    for source in sources:
        await db.execute(
            """
            INSERT INTO integration_tokens
              (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, 'synth-stub', 'active')
            ON CONFLICT (customer_id, source_system)
              WHERE device_id IS NULL
            DO NOTHING
            """,
            customer_id,
            source,
        )

    log.info("tenant_init_complete", customer_id=customer_id, sources=sources)


async def clean_tenant(customer_id: str, db, bucket: ObjectStore) -> None:
    """Prefix-guarded teardown of a synthetic tenant.

    Refuses any customer_id NOT starting with cust-eval- or cust-synth-.
    Transactionally DELETEs from every CUSTOMER_OWNED_TABLES row matching
    customer_id, then list-and-delete R2 keys under raw/.../<customer_id>/synth/.
    The customers row is preserved as a tenant marker.
    """
    if not customer_id.startswith(("cust-eval-", "cust-synth-")):
        raise ValueError(
            f"refuse to clean non-synthetic customer: {customer_id!r}"
        )

    async with db.transaction():
        for table in CUSTOMER_OWNED_TABLES:
            await db.execute(
                f"DELETE FROM {table} WHERE customer_id = $1",
                customer_id,
            )

    bucket_name = bucket.bucket_for(customer_id)
    needle = f"/{customer_id}/synth/"
    keys = await bucket.list_keys(bucket_name, "raw/")
    synth_keys = [k for k in keys if needle in k]
    for key in synth_keys:
        await bucket.delete(bucket_name, key)

    log.info(
        "tenant_clean_complete",
        customer_id=customer_id,
        rows_deleted_per_table=len(CUSTOMER_OWNED_TABLES),
        r2_keys_deleted=len(synth_keys),
    )
```

### Step 4: Run tests

- [ ] **Step 4: Run tests** — Run: `.venv/bin/pytest tests/synth/test_bootstrap.py -v` — Expected: PASS — 6 tests.

Also: `.venv/bin/ruff check scripts/synth/bootstrap.py tests/synth/test_bootstrap.py shared/storage.py` — Expected: clean.

### Step 5: Commit

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/bootstrap.py tests/synth/test_bootstrap.py shared/storage.py
git commit -m "$(cat <<'EOF'
feat(synth): TenantBootstrap (init+clean) + ObjectStore.delete shim

Adds idempotent customer/bucket/token init and prefix-guarded teardown.
Customers row is preserved across clean as a tenant marker so init can
re-bind without a race. ObjectStore gets a per-key delete method that
the clean path needs (sibling to the existing delete_bucket_recursive).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: IngestionWriter integrate mode

**Files:**
- Modify: `scripts/synth/output/writer.py` (replace contents — supersedes Task 11)
- Modify: `tests/synth/test_ingestion_writer.py` (append integrate-mode tests)

### Step 1: Append integrate-mode tests

- [ ] **Step 1: Append integrate-mode tests** (full code, append to existing test file):

```python
# Append the following to tests/synth/test_ingestion_writer.py:

import os
from unittest.mock import AsyncMock, MagicMock

import pytest


def _mock_bucket() -> AsyncMock:
    bucket = AsyncMock()
    bucket.bucket_for = MagicMock(return_value="prbe-synth-bucket")
    bucket.put = AsyncMock(return_value=None)
    return bucket


def _mock_db() -> AsyncMock:
    db = AsyncMock()
    db.executemany = AsyncMock(return_value=None)
    return db


@pytest.mark.asyncio
async def test_integrate_writes_local_and_bucket_and_queues_row(tmp_path: Path) -> None:
    bucket = _mock_bucket()
    db = _mock_db()
    writer = IngestionWriter(
        out_dir=tmp_path,
        mode="integrate",
        customer_id="cust-eval-test-01",
        bucket=bucket,
        db=db,
    )
    await writer.write(_slack_doc("doc-1"))
    await writer.close()

    # Local file written
    assert (tmp_path / "raw" / "slack" / "doc-1.json").exists()
    # R2 put called once with the customer-scoped key
    assert bucket.put.await_count == 1
    args = bucket.put.await_args.args
    assert args[1] == "raw/slack/cust-eval-test-01/synth/doc-1.json"
    # ingestion_queue insert flushed on close
    assert db.executemany.await_count == 1
    sql = db.executemany.await_args.args[0]
    assert "INSERT INTO ingestion_queue" in sql
    assert "ON CONFLICT" in sql


@pytest.mark.asyncio
async def test_integrate_batches_at_50_writes(tmp_path: Path) -> None:
    bucket = _mock_bucket()
    db = _mock_db()
    writer = IngestionWriter(
        out_dir=tmp_path,
        mode="integrate",
        customer_id="cust-eval-test-01",
        bucket=bucket,
        db=db,
    )
    for i in range(50):
        await writer.write(_slack_doc(f"doc-{i}"))
    # Flush should have triggered exactly once at the 50th write.
    assert db.executemany.await_count == 1
    await writer.close()
    # close() with empty batch is a no-op.
    assert db.executemany.await_count == 1


@pytest.mark.asyncio
async def test_integrate_close_flushes_residual_batch(tmp_path: Path) -> None:
    bucket = _mock_bucket()
    db = _mock_db()
    writer = IngestionWriter(
        out_dir=tmp_path,
        mode="integrate",
        customer_id="cust-eval-test-01",
        bucket=bucket,
        db=db,
    )
    for i in range(10):
        await writer.write(_slack_doc(f"doc-{i}"))
    assert db.executemany.await_count == 0  # under batch threshold
    await writer.close()
    assert db.executemany.await_count == 1


@pytest.mark.asyncio
async def test_integrate_requires_customer_id_bucket_db(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integrate mode requires"):
        IngestionWriter(out_dir=tmp_path, mode="integrate")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("PRBE_TEST_DB_URL"),
    reason="PRBE_TEST_DB_URL env not set; skipping live integration test",
)
async def test_integrate_round_trip_against_test_db(tmp_path: Path) -> None:
    """Live integration smoke. Requires PRBE_TEST_DB_URL pointing at a
    disposable Postgres + a real ObjectStore. Skipped in standard CI.
    """
    pytest.skip("placeholder: implementer wires this against shared.db helpers")
```

### Step 2: Run integrate tests against the existing local writer

- [ ] **Step 2: Run integrate tests against the existing local writer** — Run: `.venv/bin/pytest tests/synth/test_ingestion_writer.py -v` — Expected: FAIL — `TypeError: IngestionWriter.__init__() got an unexpected keyword argument 'mode'` (or similar) on the new tests; the 4 local-mode tests from Task 11 still pass.

### Step 3: Replace `scripts/synth/output/writer.py` with full integrate-aware version

- [ ] **Step 3: Replace `scripts/synth/output/writer.py` with full integrate-aware version** (full code — supersedes Task 11's file):

```python
"""IngestionWriter — writes SynthDocs to local files in both modes, and
additionally to R2 + ingestion_queue when mode='integrate'.

In local mode (default), each `write` produces one local JSON file under
`<out_dir>/raw/<source>/`. In integrate mode, the same local file is still
written (for inspection), AND the envelope is pushed to R2 at
`raw/<source>/<customer_id>/synth/<id>.json`, AND a row is batched into
`ingestion_queue` (flushed at BATCH_SIZE or on close).

Schema notes (real prbe-knowledge schema, NOT the spec's drift):
- column is `source_system`, not `source`
- payload column is `payload_s3_keys TEXT[]`, not `raw_key TEXT`
- conflict key is (source_system, source_event_id)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from scripts.synth.archetypes.base import Source
from scripts.synth.output import notion as notion_wrapper
from scripts.synth.output import slack as slack_wrapper
from scripts.synth.output.base import SynthDoc

BATCH_SIZE = 50


class IngestionWriter:
    """Plan 2 writer with two modes.

    local mode: writes to <out_dir>/raw/<source>/<id>.json. No DB or R2.
    integrate mode: also writes to bucket and inserts into ingestion_queue.

    integrate mode requires a prior `synth init` to have created the customer
    + bucket + integration_tokens stub rows.
    """

    def __init__(
        self,
        *,
        out_dir: Path,
        mode: Literal["local", "integrate"] = "local",
        customer_id: str | None = None,
        bucket=None,
        db=None,
    ) -> None:
        self.out_dir = out_dir
        self.mode = mode
        self.customer_id = customer_id
        self.bucket = bucket
        self.db = db
        self._batch: list[tuple[SynthDoc, str]] = []
        if mode == "integrate" and (customer_id is None or bucket is None or db is None):
            raise ValueError(
                "integrate mode requires customer_id, bucket, and db arguments"
            )

    async def write(self, doc: SynthDoc) -> None:
        envelope = self._envelope(doc)

        # Always write local file (for inspection in both modes).
        local_path = self.out_dir / "raw" / doc.source.value / f"{doc.source_event_id}.json"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(envelope)

        if self.mode == "local":
            return

        # integrate mode: R2 put + queue batching
        bucket_name = self.bucket.bucket_for(self.customer_id)
        key = f"raw/{doc.source.value}/{self.customer_id}/synth/{doc.source_event_id}.json"
        await self.bucket.put(bucket_name, key, envelope)
        self._batch.append((doc, key))
        if len(self._batch) >= BATCH_SIZE:
            await self._flush_queue()

    async def close(self) -> None:
        if self.mode == "integrate" and self._batch:
            await self._flush_queue()

    async def _flush_queue(self) -> None:
        """Batch-INSERT to ingestion_queue using the actual prbe-knowledge schema."""
        rows = [
            (
                self.customer_id,
                doc.source.value,         # source_system
                doc.source_event_id,
                [key],                     # payload_s3_keys: TEXT[]
                doc.priority,
                doc.occurred_at,
            )
            for doc, key in self._batch
        ]
        await self.db.executemany(
            """
            INSERT INTO ingestion_queue
              (customer_id, source_system, source_event_id, payload_s3_keys,
               status, priority, occurred_at, enqueued_at)
            VALUES ($1, $2, $3, $4, 'pending', $5, $6, NOW())
            ON CONFLICT (source_system, source_event_id) DO NOTHING
            """,
            rows,
        )
        self._batch.clear()

    def _envelope(self, doc: SynthDoc) -> bytes:
        if doc.source == Source.SLACK:
            return slack_wrapper.wrap(doc)
        if doc.source == Source.NOTION:
            return notion_wrapper.wrap(doc)
        raise ValueError(
            f"Plan 2 doesn't support source: {doc.source.value}. "
            "GitHub/Linear/Sentry/Granola wrappers land in Plan 3."
        )
```

### Step 4: Run all tests pass

- [ ] **Step 4: Run all tests pass** — Run: `.venv/bin/pytest tests/synth/test_ingestion_writer.py -v` — Expected: PASS — 8 tests (4 from Task 11 + 4 new integrate tests; the live-DB test is skipped without PRBE_TEST_DB_URL).

Also: `.venv/bin/ruff check scripts/synth/output/writer.py tests/synth/test_ingestion_writer.py` — Expected: clean.

### Step 5: Commit

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/output/writer.py tests/synth/test_ingestion_writer.py
git commit -m "$(cat <<'EOF'
feat(synth): IngestionWriter integrate mode (R2 + ingestion_queue)

Adds the optional integrate path: R2 put + batched ingestion_queue insert
on top of the existing local-file write. Targets the actual prbe-knowledge
schema (source_system column, payload_s3_keys TEXT[]). Conflict key is
(source_system, source_event_id) for idempotent re-runs.

A live integration test (test_integrate_round_trip_against_test_db) is
included but skipped unless PRBE_TEST_DB_URL is set — implementer wires
the real DB pool when running it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: CLI subcommands (init / run / clean)

**Files:**
- Modify: `scripts/synth/cli.py` (add three subcommands; existing `extract` unchanged)
- Test: `tests/synth/test_cli_plan2.py`

**Important context for the implementer:**

The implementer must read the existing `scripts/synth/cli.py` (Plan 1) once before starting because:
- The existing `build_parser`, `_resolve_output_dir`, `_dumps`, `_resolve_company_context`, `_infer_company_name_from_repos`, and `main` functions stay.
- The existing `extract` subcommand stays.
- Task 15 ADDS three subparsers (`init`, `run`, `clean`), three new flag groups, and three new orchestrator functions (`_init_async`, `_run_async`, `_clean_async`).

### Step 1: Write the failing test (full code)

```python
"""Subprocess-driven smoke tests for Plan 2 CLI subcommands.

These tests don't run against a real DB — they cover argument parsing,
help-text shape, and error paths. The integrate-mode end-to-end test
lives in test_e2e_run.py (Task 16) and skips without PRBE_TEST_DB_URL.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "scripts.synth", *args]
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def test_run_help_lists_plan2_flags() -> None:
    result = _run(["run", "--help"])
    assert result.returncode == 0
    assert "--integrate" in result.stdout
    assert "--time-window" in result.stdout
    assert "--archetypes" in result.stdout
    assert "--limit-scenarios" in result.stdout
    assert "--reset" in result.stdout


def test_init_help_lists_profile_flag() -> None:
    result = _run(["init", "--help"])
    assert result.returncode == 0
    assert "--profile" in result.stdout


def test_clean_help_lists_customer_flag() -> None:
    result = _run(["clean", "--help"])
    assert result.returncode == 0
    assert "--customer" in result.stdout


def test_clean_refuses_non_synth_prefix(tmp_path: Path) -> None:
    result = _run(["clean", "--customer", "prod-tenant-01"])
    assert result.returncode != 0
    assert "refuse to clean non-synthetic" in result.stderr


def test_run_local_writes_world_model_and_manifest(tmp_repo_profile_dir: Path) -> None:
    """Smoke: --integrate NOT set -> local files only -> manifest.json + raw/ exist."""
    out_dir = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"
    result = _run([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out_dir),
        "--time-window", "14d",
        "--limit-scenarios", "2",
    ])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "raw").is_dir()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["mode"] == "local"
    assert manifest["customer_id"].startswith("cust-eval-")


def test_run_archetype_filter_restricts_output(tmp_repo_profile_dir: Path) -> None:
    out_dir = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"
    result = _run([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out_dir),
        "--time-window", "14d",
        "--archetypes", "STANDUP_UPDATE",
    ])
    assert result.returncode == 0
    # Notion is only emitted by ON_CALL_HANDOFF; --archetypes filter excludes it.
    assert not (out_dir / "raw" / "notion").exists() or not list((out_dir / "raw" / "notion").glob("*.json"))
```

The `tmp_repo_profile_dir` fixture is provided by `tests/synth/conftest.py` from Plan 1 and writes a profile YAML pointing at the existing `tmp_repo` fixture. If it doesn't exist yet, add this addition to `tests/synth/conftest.py` (note: this is the only conftest change in Plan 2):

```python
# Append to tests/synth/conftest.py:

@pytest.fixture
def tmp_repo_profile_dir(tmp_repo: Path, tmp_path: Path) -> Path:
    """Build a profile YAML pointing at tmp_repo. Returns the dir."""
    profile_dir = tmp_path / "profile_dir"
    profile_dir.mkdir()
    profile_path = profile_dir / "profile.yaml"
    profile_path.write_text(
        f"""
customer_id: cust-eval-fake-01
preset: tiny-test
seed: 7
repos:
  - url: github.com/x/fake
    local_path: {tmp_repo}
world_model:
  min_commits_per_persona: 1
  topic_pool_lookback_days: 9999
""".strip()
    )
    return profile_dir
```

### Step 2: Run test to verify it fails

- [ ] **Step 2: Run test to verify it fails** — Run: `.venv/bin/pytest tests/synth/test_cli_plan2.py -v` — Expected: FAIL — at least the help-output tests fail with `error: invalid choice: 'init'` (and similar for `run`/`clean`) because the CLI parser only knows about `extract`.

### Step 3: Modify `scripts/synth/cli.py`

The implementer takes the EXISTING cli.py and adds:
1. New imports at top (preserve existing imports).
2. New subparsers in `build_parser()` — `init`, `run`, `clean`.
3. Three new orchestrator functions — `_init_async`, `_run_async`, `_clean_async`.
4. New dispatch branches in `main()`.

Below is the COMPLETE replacement file. The implementer overwrites `scripts/synth/cli.py` with this:

```python
"""CLI dispatch for the synth tool.

Plan 1: extract subcommand (WorldModel dump only, no DB).
Plan 2: init / run / clean subcommands.

Plan 2 commands default to local-files mode. The --integrate flag opts into
DB + R2 writes (requires a prior `synth init`).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.synth.bootstrap import clean_tenant, init_tenant
from scripts.synth.cache import DiskCache, default_cache_root
from scripts.synth.company_context import (
    CompanyContext,
    infer_company_context,
    load_company_context,
)
from scripts.synth.extractor.github_api import GithubClient
from scripts.synth.extractor.repo import RepoExtractor, RepoSignals
from scripts.synth.llm_client import LlmClient, LlmClientProtocol
from scripts.synth.output.eval_artifacts import (
    write_docs_index,
    write_manifest,
    write_profile,
    write_warnings,
)
from scripts.synth.output.writer import IngestionWriter
from scripts.synth.ownership import build_ownership_index
from scripts.synth.profile import Profile, load_profile
from scripts.synth.scenarios import TimeWindow, run_scenarios
from scripts.synth.validator import validate_name_only
from scripts.synth.world_model import merge_world_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.synth",
        description="Synthetic company corpus generator for prbe-knowledge eval datasets.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract — Plan 1, unchanged
    extract = sub.add_parser(
        "extract",
        help="Extract WorldModel from repos in a profile (no DB writes).",
    )
    extract.add_argument("--profile", required=True, type=str, help="Path to profile YAML.")
    extract.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write world_model.json (default: eval-datasets/<run-id>/).",
    )

    # init — Plan 2
    init = sub.add_parser(
        "init",
        help="Bootstrap a synthetic tenant: customers row + bucket + integration_tokens stubs.",
    )
    init.add_argument("--profile", required=True, type=str)

    # run — Plan 2
    run = sub.add_parser(
        "run",
        help="Run scenarios for a profile. Local files by default; --integrate writes to DB+R2.",
    )
    run.add_argument("--profile", required=True, type=str)
    run.add_argument("--output-dir", type=str, default=None)
    run.add_argument(
        "--integrate",
        action="store_true",
        help="Also write to R2 + ingestion_queue. Requires prior `synth init`.",
    )
    run.add_argument(
        "--reset",
        action="store_true",
        help="Call `synth clean` before running.",
    )
    run.add_argument(
        "--time-window",
        type=str,
        default=None,
        help="Override profile time_window.days (e.g., 30d, 14d). Default: 30d.",
    )
    run.add_argument(
        "--archetypes",
        type=str,
        default=None,
        help="Comma-separated archetype names to restrict the run.",
    )
    run.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        help="Per-archetype scenario cap (debug).",
    )
    run.add_argument("--verbose", action="store_true")

    # clean — Plan 2
    clean = sub.add_parser(
        "clean",
        help="Tear down a synthetic tenant. Refuses non-synth customer prefixes.",
    )
    clean.add_argument("--customer", required=True, type=str)

    return parser


def _resolve_output_dir(profile: Profile, override: str | None) -> Path:
    if override:
        return Path(override)
    run_id = (
        datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        + f"-{profile.preset}-seed{profile.seed}"
    )
    return Path("eval-datasets") / run_id


def _parse_time_window(arg: str | None, profile: Profile) -> TimeWindow:
    """CLI --time-window beats profile.time_window.days; both default to 30."""
    if arg is not None:
        days = int(arg.rstrip("d"))
    else:
        cfg = profile.raw.get("time_window") or {}
        days = int(cfg.get("days", 30))
    end = datetime.now(UTC).replace(microsecond=0)
    return TimeWindow(end=end, days=days)


# ---------------------------------------------------------------------------
# extract — Plan 1 (unchanged)
# ---------------------------------------------------------------------------


async def _extract_async(profile: Profile, out: Path) -> int:
    cache = DiskCache(default_cache_root("repos"))
    gh_token = os.environ.get("GITHUB_TOKEN")
    gh_client = GithubClient(token=gh_token) if gh_token else None
    extractor = RepoExtractor(github_client=gh_client, cache=cache)

    out.mkdir(parents=True, exist_ok=True)

    wm_cfg = profile.raw.get("world_model") or {}
    min_threshold = int(wm_cfg.get("min_commits_per_persona", 2))
    max_personas = int(wm_cfg.get("max_personas", 25))
    lookback_days = int(wm_cfg.get("topic_pool_lookback_days", 90))
    since = datetime.now(UTC).replace(microsecond=0) - timedelta(days=lookback_days)

    signals: list[RepoSignals] = []
    for repo in profile.repos:
        if repo.local_path is None:
            print(f"warn: repo {repo.url!r} has no local_path; skipping", file=sys.stderr)
            continue
        if gh_client is not None:
            sig = await extractor.extract(repo.url, repo.local_path, since=since, fetch_github=True)
        else:
            sig = extractor.extract_local(repo.url, repo.local_path, since=since)
        signals.append(sig)

    if gh_client is not None:
        await gh_client.close()

    if not signals:
        print("error: no repos extracted; check profile.repos[*].local_path", file=sys.stderr)
        return 3

    cc = await _resolve_company_context(profile, signals, out)

    wm = merge_world_model(
        signals=signals,
        company_name=cc.name,
        seed=profile.seed,
        min_threshold=min_threshold,
        max_personas=max_personas,
        now=datetime.now(UTC),
    )

    (out / "world_model.json").write_text(_dumps(wm))
    (out / "company_context.json").write_text(_dumps(cc))
    print(f"wrote {out}/world_model.json", file=sys.stderr)
    return 0


async def _resolve_company_context(
    profile: Profile,
    signals: list[RepoSignals],
    out: Path,
) -> CompanyContext:
    if profile.company_context_path is not None:
        return load_company_context(profile.company_context_path)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return CompanyContext(
            name=_infer_company_name_from_repos(signals),
            stage="unknown",
            headcount=0,
            inferred=True,
        )
    llm: LlmClientProtocol = LlmClient(api_key=api_key)
    try:
        readme_blob = "\n\n".join(
            r.content for sig in signals for r in sig.readmes if r.content
        )[:20_000]
        repo_descs = [s.description or s.url for s in signals]
        cc, raw_yaml = await infer_company_context(
            readme_blob=readme_blob,
            repo_descriptions=repo_descs,
            llm_client=llm,
            model="claude-opus-4-7",
        )
        (out / "inferred-company.yaml").write_text(raw_yaml)
        return cc
    finally:
        await llm.close()


def _infer_company_name_from_repos(signals: list[RepoSignals]) -> str:
    """Best-effort name when no LLM available: longest-common-prefix
    of repo URL owners; else 'unknown'.

    Bug-fix vs plan spec: skip empty owner segments that arise from
    non-github URL schemes like `repo://fake` where split("/") yields
    ["repo:", "", "fake"] and parts[-2] is "".
    """
    owners: set[str] = set()
    for sig in signals:
        parts = sig.url.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-2]:
            owners.add(parts[-2])
    if len(owners) == 1:
        return next(iter(owners))
    return "unknown"


def _dumps(obj) -> str:
    """Pretty JSON serializer that handles dataclasses + datetimes + Paths."""
    def encode(v):
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {k: encode(getattr(v, k)) for k in (f.name for f in dataclasses.fields(v))}
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple | list):
            return [encode(x) for x in v]
        if isinstance(v, dict):
            return {k: encode(val) for k, val in v.items()}
        return v
    return json.dumps(encode(obj), indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# Plan 2 helpers — DB / bucket connection
# ---------------------------------------------------------------------------


async def _open_db_and_bucket():
    """Construct (db_pool, bucket) for integrate mode. Pulls from settings."""
    # Use the prbe-knowledge shared helpers for connection management.
    from shared.db import get_pool  # type: ignore[import-untyped]
    from shared.storage import ObjectStore  # type: ignore[import-untyped]

    db = await get_pool()
    bucket = ObjectStore()
    return db, bucket


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


async def _init_async(profile: Profile) -> int:
    db, bucket = await _open_db_and_bucket()
    try:
        await init_tenant(profile, db, bucket)
        print(f"initialized tenant {profile.customer_id}", file=sys.stderr)
        return 0
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


async def _clean_async(customer_id: str) -> int:
    db, bucket = await _open_db_and_bucket()
    try:
        await clean_tenant(customer_id, db, bucket)
        print(f"cleaned tenant {customer_id}", file=sys.stderr)
        return 0
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


async def _run_async(profile: Profile, out: Path, args) -> int:
    started_at = datetime.now(UTC)

    if args.reset:
        await _clean_async(profile.customer_id)

    cache = DiskCache(default_cache_root("repos"))
    gh_token = os.environ.get("GITHUB_TOKEN")
    gh_client = GithubClient(token=gh_token) if gh_token else None
    extractor = RepoExtractor(github_client=gh_client, cache=cache)

    out.mkdir(parents=True, exist_ok=True)

    wm_cfg = profile.raw.get("world_model") or {}
    min_threshold = int(wm_cfg.get("min_commits_per_persona", 2))
    max_personas = int(wm_cfg.get("max_personas", 25))
    lookback_days = int(wm_cfg.get("topic_pool_lookback_days", 90))
    since = datetime.now(UTC).replace(microsecond=0) - timedelta(days=lookback_days)

    signals: list[RepoSignals] = []
    for repo in profile.repos:
        if repo.local_path is None:
            print(f"warn: repo {repo.url!r} has no local_path; skipping", file=sys.stderr)
            continue
        if gh_client is not None:
            sig = await extractor.extract(repo.url, repo.local_path, since=since, fetch_github=True)
        else:
            sig = extractor.extract_local(repo.url, repo.local_path, since=since)
        signals.append(sig)

    if gh_client is not None:
        await gh_client.close()

    if not signals:
        print("error: no repos extracted; check profile.repos[*].local_path", file=sys.stderr)
        return 3

    cc = await _resolve_company_context(profile, signals, out)
    world = merge_world_model(
        signals=signals,
        company_name=cc.name,
        seed=profile.seed,
        min_threshold=min_threshold,
        max_personas=max_personas,
        now=datetime.now(UTC),
    )
    ownership = build_ownership_index(signals, world)
    time_window = _parse_time_window(args.time_window, profile)

    archetype_filter: tuple[str, ...] | None = None
    if args.archetypes:
        archetype_filter = tuple(s.strip() for s in args.archetypes.split(",") if s.strip())

    # Setup writer based on mode.
    if args.integrate:
        db, bucket = await _open_db_and_bucket()
        writer = IngestionWriter(
            out_dir=out,
            mode="integrate",
            customer_id=profile.customer_id,
            bucket=bucket,
            db=db,
        )
    else:
        db = None
        writer = IngestionWriter(out_dir=out, mode="local")

    try:
        emitted_docs: list = []
        for doc in run_scenarios(
            world,
            ownership,
            profile,
            time_window,
            archetype_filter=archetype_filter,
            scenario_limit=args.limit_scenarios,
        ):
            await writer.write(doc)
            emitted_docs.append(doc)
        await writer.close()
    finally:
        if db is not None:
            await db.close()

    violations = validate_name_only(tuple(emitted_docs), world)

    finished_at = datetime.now(UTC)
    run_id = out.name if out.name else f"{profile.preset}-seed{profile.seed}"

    archetypes_executed: dict[str, dict] = {}
    for doc in emitted_docs:
        slot = archetypes_executed.setdefault(
            doc.archetype, {"requested": 0, "generated": 0, "dropped": 0}
        )
        slot["generated"] += 1
        slot["requested"] += 1
    totals = {
        "archetypes_executed": archetypes_executed,
        "totals": {
            "scenarios": len({d.scenario_id for d in emitted_docs}),
            "documents": len(emitted_docs),
            "questions": 0,
        },
        "warnings_count": len(violations),
    }

    write_manifest(
        out,
        run_id=run_id,
        profile=profile,
        world=world,
        totals=totals,
        mode="integrate" if args.integrate else "local",
        started_at=started_at,
        finished_at=finished_at,
    )
    write_docs_index(out, emitted_docs)
    write_profile(out, profile)
    write_warnings(out, violations, [])
    (out / "world_model.json").write_text(_dumps(world))
    (out / "company_context.json").write_text(_dumps(cc))

    print(f"wrote {out}/manifest.json ({len(emitted_docs)} docs)", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# main dispatch
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "extract":
        profile = load_profile(Path(args.profile))
        out = _resolve_output_dir(profile, args.output_dir)
        return asyncio.run(_extract_async(profile, out))

    if args.cmd == "init":
        profile = load_profile(Path(args.profile))
        return asyncio.run(_init_async(profile))

    if args.cmd == "run":
        profile = load_profile(Path(args.profile))
        out = _resolve_output_dir(profile, args.output_dir)
        try:
            return asyncio.run(_run_async(profile, out, args))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 4

    if args.cmd == "clean":
        try:
            return asyncio.run(_clean_async(args.customer))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 4

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 4: Run tests

- [ ] **Step 4: Run tests** — Run: `.venv/bin/pytest tests/synth/test_cli_plan2.py -v` — Expected: PASS — 6 tests. (Run also `.venv/bin/pytest tests/synth/ -q` to ensure existing tests still pass.)

Also: `.venv/bin/ruff check scripts/synth/cli.py tests/synth/test_cli_plan2.py` — Expected: clean.

### Step 5: Commit

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/cli.py tests/synth/test_cli_plan2.py tests/synth/conftest.py
git commit -m "$(cat <<'EOF'
feat(synth): wire init/run/clean CLI subcommands end-to-end

- `synth init --profile <yaml>`: idempotent customer + bucket + token bootstrap
- `synth run --profile <yaml> [--integrate] [--time-window 30d] [--archetypes A,B] [--limit-scenarios N] [--reset]`: orchestrates RepoExtractor → merge_world_model → build_ownership_index → run_scenarios → IngestionWriter → eval artifacts
- `synth clean --customer <id>`: prefix-guarded teardown

Plan 1's `extract` subcommand is unchanged. The `--integrate` flag opts
into R2 + ingestion_queue writes; default is local files only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: End-to-end + determinism + integrate smoke

**Files:**
- Create: `tests/synth/test_e2e_run.py`

### Step 1: Write the e2e tests

```python
"""End-to-end + determinism + integrate-smoke tests for `synth run`.

These tests exercise the full Plan 2 pipeline against the existing
`tmp_repo` fixture from Plan 1's conftest, and pin the deterministic
output contract: same (profile, seed, time_window) -> byte-identical
emitted JSON files.

The integrate-mode smoke is gated on PRBE_TEST_DB_URL and is skipped
in standard CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "scripts.synth", *args]
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = ""  # force the company-context stub fallback
    return subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)


def test_e2e_run_local_writes_full_artifact_set(tmp_repo_profile_dir: Path) -> None:
    """Full pipeline: profile -> WorldModel -> scenarios -> wrappers ->
    local files + eval artifacts. Asserts the run-artifact directory has
    everything Plan 2 promises in section 13 of the spec."""
    out = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"

    result = _run_cli([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out),
        "--time-window", "30d",
    ])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    # Manifest + index + frozen profile + world_model snapshot all present.
    assert (out / "manifest.json").exists()
    assert (out / "docs_index.jsonl").exists()
    assert (out / "profile.yaml").exists()
    assert (out / "world_model.json").exists()
    assert (out / "company_context.json").exists()
    assert (out / "warnings.log").exists()

    # raw/ contains both source dirs (STANDUP_UPDATE -> slack, ON_CALL_HANDOFF
    # -> slack + notion). At least one slack and one notion doc.
    slack_docs = list((out / "raw" / "slack").glob("*.json"))
    notion_docs = list((out / "raw" / "notion").glob("*.json"))
    assert len(slack_docs) > 0
    assert len(notion_docs) > 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mode"] == "local"
    assert manifest["seed"] == 7
    assert "STANDUP_UPDATE" in manifest["archetypes_executed"]
    assert "ON_CALL_HANDOFF" in manifest["archetypes_executed"]
    expected_doc_count = len(slack_docs) + len(notion_docs)
    assert manifest["totals"]["documents"] == expected_doc_count

    # docs_index.jsonl row count matches doc count and is sorted by occurred_at.
    rows = [json.loads(line) for line in (out / "docs_index.jsonl").read_text().strip().split("\n")]
    assert len(rows) == expected_doc_count
    occurred_ats = [r["occurred_at"] for r in rows]
    assert occurred_ats == sorted(occurred_ats)


def test_e2e_run_local_is_deterministic(tmp_repo_profile_dir: Path, tmp_path: Path) -> None:
    """Two runs with the same (profile, seed, time_window) must produce
    byte-identical raw/ output. (manifest.json's started_at/finished_at and
    the auto-generated run_id are excluded — those are wall-clock fields.)
    """
    profile = tmp_repo_profile_dir / "profile.yaml"
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    for out in (out_a, out_b):
        result = _run_cli([
            "run",
            "--profile", str(profile),
            "--output-dir", str(out),
            "--time-window", "14d",
        ])
        assert result.returncode == 0, f"stderr:\n{result.stderr}"

    # Compare every raw/<source>/<file>.json byte-for-byte.
    a_files = sorted((out_a / "raw").rglob("*.json"))
    b_files = sorted((out_b / "raw").rglob("*.json"))
    assert [f.relative_to(out_a) for f in a_files] == [f.relative_to(out_b) for f in b_files]
    for fa, fb in zip(a_files, b_files, strict=True):
        assert fa.read_bytes() == fb.read_bytes(), f"diverged: {fa.relative_to(out_a)}"

    # docs_index.jsonl must also be byte-identical.
    assert (out_a / "docs_index.jsonl").read_bytes() == (out_b / "docs_index.jsonl").read_bytes()


def test_e2e_run_archetype_filter_excludes_other(tmp_repo_profile_dir: Path) -> None:
    """--archetypes ON_CALL_HANDOFF -> no slack standup messages emitted.
    (ON_CALL_HANDOFF emits BOTH slack AND notion, but its slack docs go to
    #oncall channel; STANDUP_UPDATE-only docs would be in #standup.)
    The slack/ dir will still have the oncall-thread docs, but no
    STANDUP_UPDATE-archetype docs in docs_index.jsonl.
    """
    out = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"
    result = _run_cli([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out),
        "--time-window", "30d",
        "--archetypes", "ON_CALL_HANDOFF",
    ])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    rows = [json.loads(line) for line in (out / "docs_index.jsonl").read_text().strip().split("\n") if line]
    archetypes = {r["archetype"] for r in rows}
    assert archetypes == {"ON_CALL_HANDOFF"}


import pytest


@pytest.mark.skipif(
    not os.environ.get("PRBE_TEST_DB_URL"),
    reason="PRBE_TEST_DB_URL env not set; skipping integrate smoke",
)
def test_e2e_run_integrate_smoke(tmp_repo_profile_dir: Path) -> None:
    """Live integrate smoke: requires a disposable Postgres + R2 endpoint.

    Sequence: synth init -> synth run --integrate -> assert ingestion_queue
    row count matches local-file count -> synth clean.
    """
    out = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"
    customer_id = "cust-eval-fake-01"

    init = _run_cli(["init", "--profile", str(profile)])
    assert init.returncode == 0, f"init stderr:\n{init.stderr}"

    run = _run_cli([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out),
        "--time-window", "14d",
        "--integrate",
    ])
    assert run.returncode == 0, f"run stderr:\n{run.stderr}"

    local_doc_count = len(list((out / "raw").rglob("*.json")))

    # Cross-check ingestion_queue row count via psql.
    psql = subprocess.run(
        [
            "psql", os.environ["PRBE_TEST_DB_URL"], "-tAc",
            f"SELECT COUNT(*) FROM ingestion_queue WHERE customer_id = '{customer_id}'",
        ],
        check=True, capture_output=True, text=True,
    )
    queue_count = int(psql.stdout.strip())
    assert queue_count == local_doc_count, (
        f"ingestion_queue rows ({queue_count}) != local files ({local_doc_count})"
    )

    clean = _run_cli(["clean", "--customer", customer_id])
    assert clean.returncode == 0, f"clean stderr:\n{clean.stderr}"
```

### Step 2: Run test to verify it currently fails or passes

Run: `.venv/bin/pytest tests/synth/test_e2e_run.py -v`

Expected: at this point, all the `_run_cli` invocations should succeed because Tasks 1-15 are landed; the determinism + filter tests should PASS, and the integrate-smoke is SKIPPED without PRBE_TEST_DB_URL.

If a test fails, the fix is in the implementing task, not this one — work backwards: fail in Task 16 = a regression in Task 7/8 (archetype determinism), Task 11/14 (writer output), or Task 12 (eval artifact ordering).

Expected: PASS — 3 tests, 1 skipped (4 collected).

Also: `.venv/bin/ruff check tests/synth/test_e2e_run.py` — Expected: clean.

### Step 3: Commit

```bash
git add tests/synth/test_e2e_run.py
git commit -m "$(cat <<'EOF'
test(synth): plan 2 end-to-end + determinism + integrate smoke

Three subprocess-driven tests pinning the Plan 2 contract:
- e2e local: full artifact set written, expected counts, JSONL ordering
- determinism: same (profile, seed, time_window) -> byte-identical raw/
- integrate smoke: live Postgres round-trip (skipped without PRBE_TEST_DB_URL)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (writer-run; fix inline)

- [x] Spec coverage: every requirement in `docs/superpowers/specs/2026-05-01-synthetic-output-layer-design.md` maps to a task above
- [x] No "TBD" / "TODO" / placeholder text
- [x] Module layout matches what tests reference
- [x] CLI flags match what `cli.py` actually parses
- [x] Schema columns match `db/schema.sql` (source_system + payload_s3_keys)
- [x] Validator allowlist includes WorldModel-derived sets, not just hand-coded names
- [x] Test plan lists ≥ 1 deterministic-pinning test per templated archetype
- [x] Worktree path correct

## How to test the whole plan locally

```bash
cd ~/Desktop/prbe/prbe-knowledge-worktrees/synthetic-eval-corpus-plan2

# Local mode (no DB / R2)
.venv/bin/python -m scripts.synth run \
  --profile ~/synth-profiles/prbe.yaml \
  --output-dir /tmp/wm \
  --time-window 30d

ls /tmp/wm/                        # manifest.json, docs_index.jsonl, raw/, ...
jq . /tmp/wm/manifest.json
jq . /tmp/wm/docs_index.jsonl | head

# Integrate mode (requires test DB + R2)
PRBE_TEST_DB_URL=postgres://... \
.venv/bin/python -m scripts.synth init --profile ~/synth-profiles/prbe.yaml
.venv/bin/python -m scripts.synth run --profile ~/synth-profiles/prbe.yaml --integrate

# Verify ingestion_queue
psql $PRBE_TEST_DB_URL -c "SELECT source_system, COUNT(*) \
  FROM ingestion_queue WHERE customer_id = 'cust-eval-prbe-01' \
  GROUP BY source_system"

# Tear down
.venv/bin/python -m scripts.synth clean --customer cust-eval-prbe-01
```
