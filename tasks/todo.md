# Wiki gaps — Part B implementation

Plan: `~/.gstack/projects/prbe-ai-research-os/2026-08-17-wiki-gaps-design-plan.md`
Reviewed via /plan-eng-review 2026-08-17. Tasks T1–T10.

Repo split:
- **prbe-knowledge** (this checkout, `feat/wiki-page-append`): T1, T2, T3, T7, T8-engine-half
- **research-os** (separate checkout): T4, T5, T6, T8-consumer-half, T9

## prbe-knowledge

- [x] T1 — `POST /api/wiki/pages/{type}/{slug}/append` with idempotency key + stored result
  - migration `0106_wiki_append_idempotency` (head is `0105_documents_parent_live_index`)
  - reuse `page_lock_key` + `_read_live_page_for_cas` + `Normalizer._persist` — same path as PUT
- [x] T2 — specify + test append serialization vs PUT / delete / restore
- [x] T3 — refuse append on reserved `index` (free via `_validate_wiki_type`; needs the test)
- [x] T7 — research kinds in `_LINK_NODE_MAP`, immutable tenant-scoped canonical ids
- [x] T8a — publish parsed refs (engine half of D11)

## research-os

- [ ] T4 — proxy `POST /v1/wiki/pages/{type}/{slug}/append`, WRITE scope, author from principal
- [ ] T5 — `wiki_append()` SDK + `probe wiki append` CLI + MCP tool, version negotiation
- [ ] T6 — contract-validate the append response
- [ ] T8b — consume + store published refs (migration)
- [ ] T9 — "referenced by" panel

## Not code

- [ ] T10 — registry-vs-wiki ownership of "official version" (blocks gap 4b; design only)

## ⚠ Collisions with live parallel work (checked 2026-08-18)

**`~/kb-wiki-gen` on `feat/wiki-rebuild-generations` is implementing the rebuild-fixes
plan right now**, uncommitted, in this same repo. It modifies `kb/wiki_routes.py`,
`kb/synthesis/persistence.py`, `tests/test_wiki_routes.py`, `tests/conftest.py`.

1. **ALEMBIC SLOT COLLISION — must be fixed before the second merge.**
   They add `db/migrations/versions/20260817_0106_wiki_generations.py`
   (`revision = "0106_wiki_generations"`, `down_revision = "0105_documents_parent_live_index"`).
   Mine declares `revision = "0106_wiki_append_idem"` with the SAME parent. The revision
   STRINGS differ, so alembic will not complain about a duplicate id — it will fail with
   **multiple heads** once both are in one tree.
   Kept parented on 0105 here so this branch stays self-consistent and testable.
   Per the plan's sequencing (rebuild-fixes first), **this one renumbers**: on merge,
   rename to `0107_wiki_append_idem` and set `down_revision = "0106_wiki_generations"`.

2. **`kb/wiki_routes.py` is edited by both.** Mine appends a new route between
   `upsert_wiki_page` and the settings PUT; theirs rewrites the bootstrap/backfill
   trigger flow for generations. Different regions of the file, but expect a merge.

3. **Shared test Postgres.** `tests/conftest.py` hardcodes `localhost:5432` and the
   `live_db` fixture `TRUNCATE`s `customers ... CASCADE`. Two sessions running this suite
   destroy each other's data mid-run. Filed via `feedback`. Until it is fixed, check
   `ps -eo cmd | rg pytest` before running the suite here.

## Notes

- Engine `main` moved to `cab7fde` mid-session (triage DLQ fix, does not touch these files).
- `_validate_wiki_type` already 400s on `index` — T3 is a test, not a change.
- Size cap is enforced in the DB by `0104_wiki_live_page_size`; append must refuse cleanly
  rather than surface a constraint violation as a 500.
