# Synth Tool — Plan 2 Design: Output Layer + Templated Narrative Path

**Date:** 2026-05-01
**Status:** Approved (brainstorming round)
**Stacks on:** [Plan 1 — Deterministic layer](./2026-04-30-synthetic-company-eval-design.md), shipped on `feat/synthetic-eval-corpus` (PR #46).

## 1. Goal

Plan 1 produced an immutable `WorldModel` from local-clone repo signals. Plan 2 adds the **Output layer** plus a **thin templated-only Narrative path** (no LLM) so the tool can emit deterministic synthetic Slack messages and Notion pages, write them through real source-shaped envelopes, and optionally land them in the prbe-knowledge ingestion pipeline.

Concrete success criterion: from a profile pointing at one or more local clones, `python -m scripts.synth run --profile <yaml>` writes a directory of source-shaped JSON files that round-trip cleanly through the existing `Connector.parse` for Slack and Notion. With `--integrate`, the same run also creates a tenant, uploads to R2, and inserts into `ingestion_queue` — verifiable end-to-end against a test DB.

Plan 2 ships **zero LLM calls**. All content is templated over the WorldModel.

## 2. Non-goals (deferred to Plan 3)

- LLM-driven Planner / Writer / Validator Pass 2.
- Plot archetypes (`INCIDENT`, `LAUNCH`, `BIG_REFACTOR`, `PERF_REGRESSION`, `DEPENDENCY_BUMP`, `CUSTOMER_ESCALATION`).
- Slow-burn archetypes (`ROADMAP_INITIATIVE`, `ARCH_RFC`, etc.).
- Meeting archetypes (`WEEKLY_ENG_REVIEW`, `RETRO`, `ALL_HANDS`, `SPRINT_PLANNING`) — better with LLM.
- GitHub source wrapper + `PR_REVIEW` + `DEPLOY_THREAD` archetypes.
- Linear, Sentry, Granola wrappers.
- `questions.jsonl` and `scenarios/*.json` eval artifacts.
- Cost ceiling, mock-LLM mode, prompt cache.

## 3. Architecture

Plan 2 sits between Plan 1's deterministic layer and Plan 3's full narrative layer. The architecture stays consistent with the original spec's three-layer split (`docs/superpowers/specs/2026-04-30-synthetic-company-eval-design.md`):

```
Profile (YAML) ─► RepoExtractor (Plan 1) ─► WorldModelMerger (Plan 1)
                                                   │
                                                   ▼
                                        ScenarioRunner (NEW, Plan 2)
                                          walks templated archetypes
                                                   │
                                                   ▼
                                          Validator pass 1 (NEW)
                                          name-only WorldModel check
                                                   │
                                                   ▼
                                       SOURCE_WRAPPERS (NEW)
                                       slack envelope + notion envelope
                                                   │
                                                   ▼
                                       IngestionWriter (NEW)
                                       local files (default)  OR
                                       R2 + ingestion_queue (--integrate)
                                                   │
                                                   ▼
                                       Eval artifacts (NEW)
                                       manifest.json + docs_index.jsonl
```

Plan 3 swaps `ScenarioRunner.build_spec` for an LLM Planner and adds Validator Pass 2 — the orchestrator and seams stay identical.

## 4. Module layout

**New files:**

```
scripts/synth/
  archetypes/
    __init__.py
    base.py              # Archetype dataclass, Cadence/Category enums, ScenarioSpec, DocSpec
    library.py           # ARCHETYPES = {"STANDUP_UPDATE": ..., "ON_CALL_HANDOFF": ...}
    standup.py           # STANDUP_UPDATE templated spec builder
    oncall.py            # ON_CALL_HANDOFF templated spec builder
  scenarios.py           # ScenarioRunner — walks library, emits SynthDocs
  validator.py           # Pass 1 (name-only) — regex over WorldModel entities
  output/
    __init__.py
    base.py              # SynthDoc dataclass + Storage protocol
    slack.py             # SlackWrapper (event_callback envelope)
    notion.py            # NotionWrapper (page.updated webhook envelope)
    writer.py            # IngestionWriter (local default, --integrate gates DB+R2)
    eval_artifacts.py    # manifest.json + docs_index.jsonl writers
  bootstrap.py           # TenantBootstrap (init/clean DB+bucket logic)
```

**Modified files:**

```
scripts/synth/cli.py     # ADD init/run/clean subcommands; extract unchanged
```

**New test files:**

```
tests/synth/
  test_archetype_standup.py
  test_archetype_oncall.py
  test_scenarios.py
  test_validator.py
  test_output_slack_wrapper.py     # round-trip via fixtures/slack/
  test_output_notion_wrapper.py    # round-trip via fixtures/notion/
  test_ingestion_writer.py         # local mode + integrate mode (skipped without test DB)
  test_bootstrap.py
  test_cli_init.py
  test_cli_run.py
  test_cli_clean.py                # safety guard test
```

Target: ~30 new tests on top of Plan 1's 90.

## 5. CLI surface

```
python -m scripts.synth init     --profile <path>
python -m scripts.synth run      --profile <path> [--reset] [--integrate] \
                                 [--time-window 30d] [--archetypes A,B] \
                                 [--limit-scenarios N] [--verbose]
python -m scripts.synth clean    --customer <id>
python -m scripts.synth extract  --profile <path>     # unchanged from Plan 1
```

**Flag semantics:**

| Flag | Default | Effect |
|---|---|---|
| `--integrate` | off | Opt into DB+R2 writes (γ default; requires prior `init`) |
| `--reset` | off | Call `clean` first, then `run` |
| `--time-window` | from profile (else 30d) | Override `time_window.days` |
| `--archetypes` | all in library | Restrict run to listed archetypes |
| `--limit-scenarios` | unlimited | Per-archetype scenario cap (debug) |
| `--verbose` | off | Structlog level DEBUG |

**Without `--integrate`:** writes to `eval-datasets/<run-id>/raw/<source>/<id>.json` plus the eval artifacts. No DB or R2 contact.

**With `--integrate`:** writes to R2 at `raw/<source>/<customer_id>/synth/<id>.json` and INSERTs to `ingestion_queue`, in addition to local-file output. Requires `synth init --profile <path>` to have been run first.

**`clean`** is hard-guarded by `cust-eval-` / `cust-synth-` prefix:

```python
if not customer_id.startswith(("cust-eval-", "cust-synth-")):
    raise ValueError(f"refuse to clean non-synthetic customer: {customer_id}")
```

## 6. Profile schema additions

Plan 2 extends the profile YAML with `time_window` and `archetypes` blocks (both optional):

```yaml
customer_id: cust-eval-prbe-01
preset: tiny-test
seed: 42
repos:
  - url: github.com/prbe-ai/prbe-knowledge
    local_path: ~/Desktop/prbe/prbe-knowledge

# NEW in Plan 2 (all optional)
time_window:
  end: "2026-05-01"   # default: today UTC
  days: 30            # default: 30
archetypes:
  STANDUP_UPDATE: { count: null }   # null = cadence-driven; integer overrides
  ON_CALL_HANDOFF: { count: null }
```

Plan 1's existing `world_model:` block is untouched.

## 7. Templated archetypes

### 7.1 `Archetype` dataclass

```python
@dataclass(frozen=True)
class Archetype:
    name: str
    category: Category                 # Plan 2: only RECURRING
    cadence: Cadence                   # DAILY | WEEKLY | BIWEEKLY | MONTHLY | SPRINT
    sources_used: tuple[Source, ...]
    cast_size: tuple[int, int]         # min/max personas per scenario
    needs_planner_call: bool           # Plan 2: always False
    validator_level: ValidatorLevel    # Plan 2: NAME_ONLY
```

### 7.2 `STANDUP_UPDATE` (daily, slack)

For each working day in `time_window` (Mon–Fri), for each top-N persona by `activity_score` (default 5):

1. Pick 1–3 services the person commits to. Derived at scenario-build time by walking each `Commit` whose `author_email/name` resolves to this person, then for each of the commit's `files_touched`, finding the `Manifest` whose `path.parent` is the deepest ancestor of the file path. The manifest's `name` is the service. Top-3 by frequency are this person's services. (Tiny new helper — neither `Person` nor `Service` change.)
2. Pick 1–2 recent topics from `wm.topic_pool` mentioning those services in the last 7 days.
3. Emit one Slack message to `#standup`:
   - Template: `"Yesterday: shipped {topic_a}. Today: {service} - {topic_b}. Blockers: none."`
   - Threading: standalone (no `thread_ts`).
   - `ts`: deterministic offset from start-of-working-day.

Per working day × 5 personas = ~110 messages over a 30-day window with 22 working days.

### 7.3 `ON_CALL_HANDOFF` (weekly Mondays, slack + notion)

For each Monday in `time_window`:

1. Pick 2 personas via deterministic seed-rotation over top personas (`outgoing` and `incoming`).
2. Pick 1–3 incidents from prior week's `topic_pool` filtered to `kind in {ISSUE, COMMIT}` with severity-suggesting subject text. Templates fall back gracefully when there are zero recent issues — the handoff still emits with "Quiet week, nothing on fire."
3. Emit:
   - **Slack thread to `#oncall`:** parent message from `outgoing` summarizing the week's incidents; reply from `incoming` acknowledging.
   - **Notion page** titled `"On-call handoff <date>"` with H2 per incident (owner mention, status), under section `Engineering > On-call rotation`.

Both deterministic: same `(WorldModel, profile.seed, time_window)` → byte-identical output.

### 7.4 Determinism contract

For both archetypes:

```
hash(scenario_specs) is a pure function of (WorldModel, profile.seed, time_window)
```

Tests pin this — `test_archetype_standup_is_deterministic` runs the builder twice and asserts identical output.

## 8. ScenarioRunner

```python
def run_scenarios(
    world: WorldModel,
    profile: Profile,
    time_window: TimeWindow,
    *,
    archetype_filter: tuple[str, ...] | None = None,
    scenario_limit: int | None = None,
) -> Iterator[SynthDoc]:
    library = filter_library(profile.archetypes, archetype_filter)
    for archetype in library:
        instances = compute_instances(archetype, time_window)
        for i, instance_ts in enumerate(instances):
            if scenario_limit is not None and i >= scenario_limit:
                break
            spec = archetype.build_spec(world, instance_ts, profile.seed)
            for doc_spec in spec.doc_specs:
                yield materialize(doc_spec, world, spec)
```

Plan 3 will add an alternate path that calls an LLM Planner instead of `archetype.build_spec` for plot archetypes; the per-archetype `needs_planner_call` flag selects between paths.

## 9. SOURCE_WRAPPERS

Each is a pure function `SynthDoc → bytes` producing the byte-exact provider envelope the connector expects.

### 9.1 `SlackWrapper`

```python
def wrap(doc: SynthDoc) -> bytes:
    """Produce a Slack Events API event_callback envelope."""
    payload = {
        "type": "event_callback",
        "event": {
            "type": "message",
            "channel": doc.slack_channel,
            "user": doc.slack_user,
            "text": doc.text,
            "ts": doc.ts.timestamp_str(),
            **({"thread_ts": doc.thread_ts.timestamp_str()} if doc.thread_ts else {}),
        },
        # ...standard event_callback envelope fields
    }
    return orjson.dumps(payload)
```

Round-trip tested: `parse(wrap(doc))` recovers `doc`'s observable fields. Test fixture pulled from `fixtures/slack/message_simple.json`.

### 9.2 `NotionWrapper`

```python
def wrap(doc: SynthDoc) -> bytes:
    """Produce a Notion page.updated webhook envelope."""
    payload = {
        "type": "page.updated",
        "page": {"id": doc.notion_page_id, "properties": {...}},
        "blocks": doc.notion_blocks,  # Notion block JSON list
    }
    return orjson.dumps(payload)
```

Round-trip tested against existing fixtures at `fixtures/notion/`.

### 9.3 What `Connector.parse` must accept

The wrappers deliberately match what the existing prbe-knowledge connectors already parse — no new connector work in Plan 2. Round-trip tests are the contract.

## 10. IngestionWriter (γ semantics)

```python
class IngestionWriter:
    mode: Literal["local", "integrate"]
    out_dir: Path                # always set
    customer_id: str | None      # required for integrate mode
    bucket: ObjectStore | None   # required for integrate mode (uses shared.storage.ObjectStore)
    db: AsyncConnection | None   # required for integrate mode
    batch: list[tuple[SynthDoc, str]]
    BATCH_SIZE: int = 50

    async def write(self, doc: SynthDoc) -> None:
        envelope = WRAPPERS[doc.source].wrap(doc)
        # Local file write happens in BOTH modes for inspection.
        path = self.out_dir / "raw" / doc.source / f"{doc.source_event_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(envelope)
        if self.mode == "local":
            return
        # integrate mode adds R2 + ingestion_queue
        key = f"raw/{doc.source}/{self.customer_id}/synth/{doc.source_event_id}.json"
        await self.bucket.put(key, envelope)
        self.batch.append((doc, key))
        if len(self.batch) >= self.BATCH_SIZE:
            await self._flush_queue()

    async def close(self) -> None:
        if self.batch:
            await self._flush_queue()

    async def _flush_queue(self) -> None:
        # Targets the actual prbe-knowledge schema (source_system, payload_s3_keys[]).
        await self.db.executemany(
            """
            INSERT INTO ingestion_queue
              (customer_id, source_system, source_event_id, payload_s3_keys,
               status, priority, occurred_at, enqueued_at)
            VALUES ($1, $2, $3, ARRAY[$4]::TEXT[], 'pending', $5, $6, NOW())
            ON CONFLICT (source_system, source_event_id) DO NOTHING
            """,
            [
                (self.customer_id, doc.source, doc.source_event_id, key,
                 doc.priority, doc.occurred_at)
                for doc, key in self.batch
            ],
        )
        self.batch.clear()
```

**Schema notes (drift from spec — actual schema wins):**
- Column is `source_system`, not `source`.
- Payload column is `payload_s3_keys TEXT[] NOT NULL DEFAULT '{}'`, not `raw_key TEXT`.
- Plan 2 always writes a single-element array. Plan 3 may use the multi-key form for `claude_code` batches.

**Idempotency:**
- `ON CONFLICT (source_system, source_event_id) DO NOTHING` — re-runs are safe.
- R2 keys are deterministic; `bucket.put` overwrites.
- Local file writes overwrite (last write wins on the same `source_event_id`).

**Backpressure:** batch=50; worker dequeues at its own rate.

## 11. TenantBootstrap

### 11.1 `init`

```python
async def init_tenant(profile: Profile, db: AsyncConnection, bucket: ObjectStore) -> None:
    await db.execute(
        """
        INSERT INTO customers (customer_id, display_name, status, ...)
        VALUES ($1, $2, 'active', ...)
        ON CONFLICT (customer_id) DO NOTHING
        """,
        profile.customer_id,
        profile.display_name or f"synth-{profile.customer_id}",
    )
    await bucket.ensure_exists()  # operationally a no-op for shared buckets
    for source in profile.sources_used:
        await db.execute(
            """
            INSERT INTO integration_tokens
              (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, 'synth-stub', 'active')
            ON CONFLICT (customer_id, source_system) WHERE device_id IS NULL
            DO NOTHING
            """,
            profile.customer_id, source,
        )
```

`init` is idempotent: safe to re-run.

### 11.2 `clean`

Prefix-guarded teardown. The strategy: rely on the `ON DELETE CASCADE` foreign keys on `customers(customer_id)` to clean up child tables, then list-and-delete the customer's R2 keys. The `customers` row is **not** deleted — it stays as a "tenant exists" marker so `init` can re-bind without race.

```python
async def clean_tenant(customer_id: str, db: AsyncConnection, bucket: ObjectStore) -> None:
    if not customer_id.startswith(("cust-eval-", "cust-synth-")):
        raise ValueError(f"refuse to clean non-synthetic customer: {customer_id}")

    # All tables with FK customer_id REFERENCES customers ON DELETE CASCADE
    # are cleaned by the single DELETE on the parent row's child rows. We
    # don't delete the customers row itself; we DELETE WHERE customer_id = $1
    # on each child explicitly so re-runs are idempotent and the operation
    # is auditable in logs.
    async with db.transaction():
        for table in CUSTOMER_OWNED_TABLES:
            await db.execute(f"DELETE FROM {table} WHERE customer_id = $1", customer_id)

    # R2: list all keys under the synth prefix for this tenant, delete each.
    bucket_name = bucket.bucket_for(customer_id)
    prefix = f"raw/"  # any source; filter on customer_id in the key
    keys = await bucket.list_keys(bucket_name, prefix)
    synth_keys = [k for k in keys if f"/{customer_id}/synth/" in k]
    for key in synth_keys:
        await bucket.delete(bucket_name, key)
```

`CUSTOMER_OWNED_TABLES` is a hand-coded list maintained alongside `db/schema.sql`. The implementation includes (verified against current schema): `ingestion_queue`, `chunks`, `documents`, `graph_edges`, `graph_nodes`, `integration_tokens`, `acl_snapshots`, `failed_chunks`, `ingestion_events`, `audit_log`, `graph_node_provenance`, `customer_source_mapping`, `usage_events`, `backfill_state`. (Cascade is the safety net; the explicit DELETEs are for visibility + idempotency.)

Note: `ObjectStore.delete` is added in Plan 2 if it doesn't exist on the shared interface — `list_keys` exists per `shared/storage.py:147` but a per-key delete may need a small addition. The implementation plan locks this in.

## 12. Validator (Pass 1 only — name-only)

```python
@dataclass(frozen=True)
class Violation:
    doc_id: str
    out_of_world: tuple[str, ...]


def validate_name_only(docs: list[SynthDoc], world: WorldModel) -> tuple[Violation, ...]:
    """Extract proper-noun-shaped tokens; assert each maps to world.{services, people, channels}."""
    allowed = {s.name for s in world.services} | {s.qualified for s in world.services}
    allowed |= {p.display_name for p in world.people if p.display_name}
    allowed |= {p.gh_username for p in world.people if p.gh_username}
    allowed |= {c.name for c in world.channels}
    allowed |= THIRD_PARTY_ALLOWLIST  # Stripe, AWS, GitHub itself, etc.

    violations: list[Violation] = []
    for doc in docs:
        mentioned = _extract_proper_nouns(doc.text)
        out = mentioned - allowed
        if out:
            violations.append(Violation(doc.id, tuple(sorted(out))))
    return tuple(violations)
```

`_extract_proper_nouns` is a plain regex pass — it is not trying to be a real NER. It captures three token shapes that are likely company-internal references:

```python
_TOKEN_RE = re.compile(
    r"#[\w-]+"                            # Slack channel mentions: #incidents
    r"|@[\w-]+"                           # Person mentions: @alice
    r"|\b[a-z][a-z0-9-]*-[a-z][a-z0-9-]*\b"  # kebab service names: payments-svc
)
```

This is intentionally narrow. Templated content uses these exact shapes. False positives are stripped via the `THIRD_PARTY_ALLOWLIST`. False negatives (camelCase service names, etc.) are accepted in v1; Plan 3 can layer in a richer extractor when LLM-generated content makes the lossy regex insufficient.

For Plan 2's templated archetypes, this should always pass — but it catches template bugs (e.g., `#stand_up` instead of `#standup`). Plan 3 adds Pass 2 (cheap LLM consistency check) on the same interface.

`THIRD_PARTY_ALLOWLIST` is a small hand-coded set of common SaaS names that are obviously not company-internal services (Stripe, AWS, Datadog, GitHub, Slack, Notion, etc.).

## 13. Eval artifacts

Per run, write to `eval-datasets/<run-id>/`:

| File | Plan 2 | Reason |
|---|---|---|
| `manifest.json` | ✓ | Run receipt, totals per archetype |
| `docs_index.jsonl` | ✓ | One row per doc; useful for debugging |
| `world_model.json` | ✓ | Already from Plan 1 |
| `inferred_company.yaml` | ✓ | Already from Plan 1 (when applicable) |
| `profile.yaml` | ✓ | Frozen resolved profile (preset + overrides merged) |
| `warnings.log` | ✓ | Validator violations + run notes |
| `questions.jsonl` | ✗ | Templated archetypes have `eval_questions=None` |
| `scenarios/*.json` | ✗ | Templated specs are too thin to benefit from per-scenario serialization |

`manifest.json` shape:

```json
{
  "run_id": "2026-05-01T14-30Z-tiny-test-seed42",
  "profile_name": "tiny-test",
  "seed": 42,
  "started_at": "...",
  "finished_at": "...",
  "customer_id": "cust-eval-prbe-01",
  "mode": "local",
  "repos": [{"url": "...", "sha": "...", "mode": "local"}],
  "world_model": {"people_count": 5, "services_count": 5, "channels_count": 15},
  "archetypes_executed": {
    "STANDUP_UPDATE": {"requested": 110, "generated": 110, "dropped": 0},
    "ON_CALL_HANDOFF": {"requested": 4, "generated": 4, "dropped": 0}
  },
  "totals": {"scenarios": 114, "documents": 122, "questions": 0},
  "warnings_count": 0
}
```

`docs_index.jsonl` shape (one row per `SynthDoc`):

```json
{
  "doc_id": "scn-standup-richard-2026-05-01-slack-0",
  "scenario_id": "scn-standup-richard-2026-05-01",
  "archetype": "STANDUP_UPDATE",
  "source": "slack",
  "occurred_at": "2026-05-01T09:00Z",
  "raw_key": "raw/slack/.../scn-standup-richard-2026-05-01-slack-0.json",
  "personas": ["gh:richardwei6"],
  "services_mentioned": ["prbe-knowledge-retrieval"],
  "is_evidence_for_question_ids": []
}
```

## 14. Determinism

Every step in Plan 2 is a pure function of `(WorldModel, profile.seed, time_window)`. Two consecutive runs produce byte-identical:

- `raw/<source>/<id>.json` files
- `docs_index.jsonl` (modulo line ordering — which is also stable per `compute_instances` deterministic order)
- `manifest.json` (modulo `started_at` / `finished_at`)

Tests pin this:

- `test_run_local_files_is_deterministic` — invoke `synth run` twice with the same profile, diff the output dirs, assert empty diff (excluding timestamp fields in `manifest.json`).
- `test_archetype_<name>_is_deterministic` — per-archetype builder test.

## 15. Test plan

**Unit:**
- `test_archetype_standup` — fixture WorldModel, assert spec shape and counts.
- `test_archetype_oncall` — fixture WorldModel with seeded incidents, assert spec emits both Slack and Notion docs.
- `test_scenarios` — `compute_instances` correctness for DAILY/WEEKLY cadences, `--archetypes` filter, `--limit-scenarios` cap.
- `test_validator` — name-only check passes for valid docs; flags fabricated names.
- `test_output_slack_wrapper` — round-trip via `fixtures/slack/`.
- `test_output_notion_wrapper` — round-trip via `fixtures/notion/`.
- `test_ingestion_writer_local` — local mode writes correct paths.
- `test_ingestion_writer_integrate` — pytest-skip unless `PRBE_TEST_DB_URL` is set; full DB+R2 round-trip.
- `test_bootstrap_init` — idempotent re-run.
- `test_bootstrap_clean_refuses_non_synth_customer` — the safety guard.
- `test_cli_init` / `test_cli_run` / `test_cli_clean` — subprocess-driven smoke.

**End-to-end:**
- `test_run_local_e2e_against_tmp_repo` — full pipeline with `tmp_repo` fixture; assert eval-datasets layout and counts match expectations.
- `test_run_local_files_is_deterministic` — pin determinism.

**Target:** ~30 new tests on top of Plan 1's 90; final suite 120/120.

**Skipped automatically when DB unavailable:** the `--integrate` tests are gated on `PRBE_TEST_DB_URL`. CI without that env var sees the local-mode tests only.

## 16. Branch & PR

- Worktree: `~/Desktop/prbe/prbe-knowledge-worktrees/synthetic-eval-corpus-plan2/`.
- Branch: `feat/synthetic-eval-corpus-plan2`, off `feat/synthetic-eval-corpus`.
- Stacked PR base: `feat/synthetic-eval-corpus`.
- When PR #46 (Plan 1) merges to `main`, GitHub auto-rebases the stacked PR.

## 17. Self-review checklist (writer-run; fix inline)

- [ ] No "TBD" / "TODO" / placeholder text.
- [ ] Module layout matches what tests reference.
- [ ] CLI flags match what `cli.py` actually parses.
- [ ] Schema columns match `db/schema.sql`, not the original spec's drift.
- [ ] Validator allowlist includes WorldModel-derived sets, not just hand-coded names.
- [ ] Test plan lists ≥ 1 deterministic-pinning test per templated archetype.
- [ ] Worktree path correct.

## 18. How to test the whole plan locally

```bash
cd ~/Desktop/prbe/prbe-knowledge-worktrees/synthetic-eval-corpus-plan2

# Local-only mode (no DB/R2)
.venv/bin/python -m scripts.synth run \
  --profile ~/synth-profiles/prbe.yaml \
  --time-window 30d

ls eval-datasets/                       # one timestamped subdir
ls eval-datasets/<run>/raw/slack/       # ~110 standup messages
ls eval-datasets/<run>/raw/notion/      # ~4 oncall handoff pages
jq . eval-datasets/<run>/manifest.json
jq . eval-datasets/<run>/docs_index.jsonl | head

# Integrate mode (requires test DB)
PRBE_TEST_DB_URL=postgres://... \
.venv/bin/python -m scripts.synth init --profile ~/synth-profiles/prbe.yaml
.venv/bin/python -m scripts.synth run --profile ~/synth-profiles/prbe.yaml --integrate

# Verify ingestion_queue
psql $PRBE_TEST_DB_URL -c "SELECT source_system, COUNT(*) FROM ingestion_queue WHERE customer_id = 'cust-eval-prbe-01' GROUP BY source_system"

# Tear down
.venv/bin/python -m scripts.synth clean --customer cust-eval-prbe-01
```
