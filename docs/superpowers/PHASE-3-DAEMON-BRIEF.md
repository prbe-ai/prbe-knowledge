# Phase 3 — `prbe-agent-tap` daemon brief

**Status:** Phase 1 (server-side) shipped on `feature/coding-agent-ingestion-v2` (PR #3) + `feature/coding-agent-gateway` (PR #48). The daemon is the only remaining piece before real Claude Code transcripts can flow into the system.

**Read order on resumption:**
1. This file (HTTP contract + scope).
2. `docs/superpowers/specs/2026-04-25-coding-agent-ingestion-design.md` on the held branch — original product design (transcripts, units, retrieval shape).
3. `scripts/smoke_claude_code.py` on prbe-backend `feature/coding-agent-gateway` — the working call sequence the daemon must replicate.

---

## What the daemon does

A long-running process per laptop that:

1. Watches Claude Code's transcript files: `~/.claude/projects/<project>/<session>.jsonl`.
2. On first run, walks an OS-aware install + pair flow: prompts the user for a pairing token from the dashboard, exchanges it for a device token via `POST /agent-tap/pair`, persists the device token in the OS keychain (macOS Keychain, libsecret on Linux).
3. Tails the JSONL files, batches new lines, and ships them to `POST /webhooks/claude_code` with `Authorization: Bearer <device_token>`.
4. Heartbeats every ~5 minutes via `POST /agent-tap/heartbeat`.
5. Handles 401 from the gateway by re-prompting the user for a fresh pairing flow (the device was revoked or the laptop's keychain entry was lost).
6. Cleanly exits + revokes via `POST /agent-tap/revoke` on uninstall.

---

## HTTP contract (post-pivot, gateway-owned)

All daemon traffic terminates at `api.prbe.ai` (prbe-backend's BFF). The daemon never reaches `api-knowledge.prbe.ai` directly.

### `POST /agent-tap/pair`
- Body: `{pairing_token, os, hostname}` (pairing_token is the JWT minted by the dashboard).
- Response 201: `{device_id, device_token, customer_id}` — `device_token` is plaintext, returned exactly once. Hash with SHA-256 to match what the gateway stores.
- Errors: 401 invalid/expired/already-used pairing token; 502 if knowledge is unreachable.

### `POST /webhooks/claude_code`
- Headers: `Authorization: Bearer <device_token>`.
- Body shape (one batch per call):
  ```json
  {
    "device_id": "<uuid>",
    "session_id": "<claude-code-session-uuid>",
    "batch_seq": 0,
    "cwd": "/path/the/user/was/in",
    "events": [
      {"line_no": 0, "raw": {"type": "user_prompt", "content": "..."}},
      {"line_no": 1, "raw": {"type": "assistant_message", "content": "..."}}
    ]
  }
  ```
- `events[].raw` is the verbatim line from Claude Code's JSONL.
- `events[].line_no` is the 0-indexed position in the source JSONL — used by the server-side dedup so retries are safe.
- Response 200: `{status: "accepted"|"duplicate", trace_id, source_event_id}`.
- Errors: 401 invalid/revoked device token; 400 missing `session_id` or `batch_seq`; 502 upstream.

**Important:** the gateway overrides any `employee_id` field in the body with the verified value from the device token. Don't bother sending it.

### `POST /agent-tap/heartbeat`
- Headers: `Authorization: Bearer <device_token>`.
- Response 200: `{ok: true, device_id}`.
- Cadence: every ~5 min. Drives `last_heartbeat_at` on the device row, used by the dashboard to show "this laptop hasn't checked in in 24h" warnings.

### `POST /agent-tap/revoke`
- Headers: `Authorization: Bearer <device_token>`.
- Response 200: `{ok: true, device_id}`.
- Idempotent. Daemon should call this on `prbe-agent-tap uninstall` and on detected token-lost flows before re-pairing.

---

## Stack choice

**Recommended: Go.**
- Single static binary, easy `brew install` / curl-pipe-sh.
- Strong file-watching primitives (`fsnotify`).
- Stdlib HTTP + JSON; no external runtime to install on the user's laptop.
- Cross-compiles cleanly for darwin/amd64, darwin/arm64, linux/amd64, linux/arm64.

Alternatives:
- **Python** — fastest to prototype but introduces a Python install dependency unless we bundle PyInstaller. Drop unless someone strongly prefers it.
- **Rust** — sharper performance + smaller binary, but harder to maintain for the team.

---

## Scope for the first cut

Phase 3 ships the minimum daemon that exercises the entire flow:

1. `prbe-agent-tap pair <pairing-token>` — single-shot CLI command. No daemon yet, just verifies the pair → keychain → first webhook path works.
2. `prbe-agent-tap watch` — long-running. Tails the JSONL files, batches by 10 lines or 5 seconds (whichever first), POSTs.
3. `prbe-agent-tap heartbeat` — invoked by launchd / systemd timer at 5-min cadence (not in-process — keeps the daemon stateless).
4. `prbe-agent-tap revoke` — single-shot.
5. `prbe-agent-tap uninstall` — calls revoke, removes keychain entry, removes the launchd plist.

**Out of scope for Phase 3.1 (defer to 3.2):**
- Sanitization (placeholder; the dashboard surfaces a "review before shipping" toggle).
- Native Claude Code plugin (the daemon is the data path; the plugin would be a UX shortcut to invoke `prbe-agent-tap pair`).
- Cursor / Copilot connectors — separate plan.

---

## Smoke verification

`scripts/smoke_claude_code.py` on prbe-backend runs the exact call sequence the daemon must produce. Get that passing first against locally-running services, then start writing the daemon. The smoke harness is the test fixture for the daemon's HTTP layer.

---

## Repo + worktree convention

When this work starts:

1. Create `~/Desktop/prbe/prbe-agent-tap` (fresh repo).
2. Worktree per branch under `~/Desktop/prbe/prbe-agent-tap-worktrees/<feature>/` (matching the per-session worktree rule from memory `feedback_worktree_per_session.md`).
3. Initial scaffolding: `go mod init github.com/prbe-ai/prbe-agent-tap`, layout `cmd/prbe-agent-tap/main.go`, `internal/{pair,watch,heartbeat,storage}/`.

---

## Architecture decisions still open

- **Where does the dashboard "Pair" UI mint the JWT?** Backend ships `POST /pairing-tokens` (PR #48). The dashboard needs a button that hits it and shows the user the pairing-token string with a "copy" affordance. That's frontend work in the dashboard repo, separate from Phase 3.
- **JSONL → batch boundary.** Default to 10 lines or 5 seconds. Tunable via daemon config; revisit after watching real session sizes.
- **Multi-laptop, same employee.** Already supported by the integration_tokens schema (`(customer_id, source_system, device_id)` partial unique index). Each laptop is a separate device row with its own token.
- **Sanitization.** Phase 2.5 — separate spec, blocks external rollout, doesn't block internal dogfooding.
