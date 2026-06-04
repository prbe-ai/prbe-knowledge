#!/usr/bin/env bash
# Install the Probe MCP server in Claude Code, Codex, and/or Cursor, and add
# agent-instruction snippets so coding agents proactively reach for it.
#
# Usage:
#   curl -fsSL https://mcp.knowledge.prbe.ai/install | bash
#
# Idempotent: re-running skips current entries and updates stale URLs.

set -euo pipefail

MCP_NAME="Probe"
MCP_URL="https://mcp.knowledge.prbe.ai/mcp"
PROBE_HEADING="## Probe MCP server (team operational memory)"

# Use ANSI colors only when stdout is a real TTY; piping to a log file
# or another command shouldn't leak `\033[0;32m` sequences into the
# captured output.
if [ -t 1 ]; then
    green()  { printf "\033[0;32m%s\033[0m\n" "$1"; }
    yellow() { printf "\033[0;33m%s\033[0m\n" "$1"; }
    red()    { printf "\033[0;31m%s\033[0m\n" "$1"; }
    dim()    { printf "\033[0;90m%s\033[0m\n" "$1"; }
else
    green()  { printf "%s\n" "$1"; }
    yellow() { printf "%s\n" "$1"; }
    red()    { printf "%s\n" "$1"; }
    dim()    { printf "%s\n" "$1"; }
fi

# Yes/no prompt that works under `curl ... | bash`.
#
# Reads from /dev/tty so the script's stdin (which is the script body when
# piped from curl) doesn't get consumed. Defaults to YES on Enter and on
# any non-interactive run (no TTY, CI, etc.) so headless usage doesn't stall.
prompt_yn() {
    local prompt="$1"
    # Try to open /dev/tty for reading. If it fails (no TTY, headless, etc.)
    # silently default to yes — don't print a prompt the user can't answer.
    if ! { exec 3</dev/tty; } 2>/dev/null; then
        return 0
    fi
    printf "  %s [Y/n] " "$prompt" >&2
    local answer=""
    read -r answer <&3 || answer=""
    exec 3<&-
    answer="$(printf "%s" "$answer" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    case "$answer" in
        ""|y|yes) return 0 ;;
        n|no)     return 1 ;;
        *)        return 0 ;;
    esac
}

read -r -d '' AGENT_GUIDANCE_SNIPPET <<EOF || true
$PROBE_HEADING

You have access to this team's operational memory via the \`$MCP_NAME\` MCP
server: Slack threads, GitHub PRs, Linear tickets, Notion docs, and Sentry
incidents.

This is not a one-time startup check. At every new user request, task shift,
debugging phase, design decision, or implementation plan, re-evaluate whether
Probe has relevant team context. A search from earlier in the session only
covers that earlier question.

ALWAYS call \`$MCP_NAME.search_knowledge\` when:

- Making architectural or design decisions
- Touching code in an unfamiliar system or area
- The user asks "how do we…", "why did we…", "what about…", "who's
  working on…"
- Before each non-trivial implementation phase or before committing to a new
  approach
- The user references a past event ("that incident", "the bug last
  sprint", "the design doc Y mentioned")
- New information changes the direction of the work after an earlier Probe
  lookup
- Conversation context was just compacted or summarized — prior Probe results
  are no longer in your direct context, so re-search anything you'd
  otherwise rely on from earlier

Pass a bag of entities/keywords as the query — ticket IDs, repos, services,
files, symbols, error strings, feature flags. NOT a question or sentence;
prose dilutes BM25, vector, and entity extraction. Good: "PRB-17 Linear
enrichment per-source toggle workspace_prefs JSONB". Bad: "Why is PRB-17
still broken?". Surface what you find to the user before proceeding —
don't silently fold it into your output.

Do NOT skip Probe on these triggers — a missed lookup is the default
failure mode. If you find yourself proposing a plan, design, refactor,
or "here's how I'd approach this" response without having called
\`$MCP_NAME.search_knowledge\` for the current goal, stop and search first.

Default retrieval loop:
1. Call \`$MCP_NAME.search_knowledge\` with \`top_k=5\`.
2. If a result looks relevant, call \`$MCP_NAME.get_source\` for bounded
   context before relying on it.
3. If recall looks low, retry once with \`top_k=15\` or a narrower
   \`source\` filter.
4. If \`related_entities\` surfaces a high-\`score\` node adjacent to your
   results that wasn't in your original query, drop its \`canonical_id\`
   into the next \`$MCP_NAME.search_knowledge\` call's query to BFS the
   knowledge graph. Set \`top_k_related=0\` to skip the enrichment on
   token-sensitive flows.
5. If results are empty or noisy, say so and continue with repo context.

Long-session checkpoint: if the conversation has moved to a new subsystem,
ticket, PR, design choice, incident, or implementation phase since the last
Probe call, call Probe again before proceeding. By default in a long session,
treat any Probe lookup older than ~5-10 turns as stale and re-search before
relying on it — the cost of a redundant lookup is small; a stale assumption
is large.

Planning gate:
- Before you output an implementation plan, architecture plan, refactor plan,
  <proposed_plan>, or plan-mode response for product code, call
  \`$MCP_NAME.search_knowledge\` unless you have a Probe lookup from this turn
  or the last few turns covering the same goal and subsystem.
- Use the results to constrain the plan. Include a short "Probe context" note
  in the plan: cite the relevant sources, or say no relevant Probe context was
  found.
- Do not treat a startup/session-opening Probe lookup as sufficient for a later
  plan if the user goal, subsystem, or approach has changed.

Use \`$MCP_NAME.query_knowledge\` only when the user asks a direct question
and wants a synthesized answer with citations. For your own reasoning,
prefer \`search_knowledge\` plus \`get_source\`.

This is NOT a source-code search. For code, read the repo directly.
EOF

read -r -d '' CURSOR_RULE_SNIPPET <<EOF || true
---
description: Use Probe team operational memory before architecture, design, debugging, and non-trivial coding decisions.
globs: "**/*"
alwaysApply: true
---

$AGENT_GUIDANCE_SNIPPET
EOF

green "Probe MCP installer"
dim "  Server: $MCP_URL"
dim "  Local alias: $MCP_NAME"
echo ""

# ---------------------------------------------------------------------------
# 1. Claude Code
# ---------------------------------------------------------------------------
if command -v claude >/dev/null 2>&1; then
    claude_probe="$(claude mcp get "$MCP_NAME" 2>/dev/null || true)"
    # Register Probe with "alwaysLoad": true so its tools load into context
    # at session start instead of sitting behind MCP tool search (deferred).
    # Deferred tools force Claude to run a ToolSearch step before it can call
    # them, which makes agents reach for Probe far less often than tools that
    # are always present — so we keep Probe always-loaded and proactive.
    # alwaysLoad needs Claude Code v2.1.121+; older clients store the field
    # harmlessly and ignore it.
    #   https://code.claude.com/docs/en/mcp  ("Exempt a server from deferral")
    #
    # add-json is the only `claude mcp` path that can set alwaysLoad. `-s user`
    # registers it once for every project (the default `local` scope would tie
    # Probe to the cwd's project only).
    probe_json="{\"type\":\"http\",\"url\":\"$MCP_URL\",\"alwaysLoad\":true}"
    if printf "%s" "$claude_probe" | grep -qF "$MCP_URL" \
        && printf "%s" "$claude_probe" | grep -qiE 'always.?load'; then
        yellow "✓ Claude Code: '$MCP_NAME' already set ($MCP_URL, alwaysLoad — skipping)"
    else
        # (Re)register so the entry always ends up at the right URL *and* with
        # alwaysLoad. remove-first makes this idempotent on re-runs and heals a
        # pre-existing URL-only entry from an older installer.
        dim "→ Configuring Claude Code '$MCP_NAME' (user scope, alwaysLoad)…"
        claude mcp remove "$MCP_NAME" >/dev/null 2>&1 || true
        if claude mcp add-json -s user "$MCP_NAME" "$probe_json" >/dev/null 2>&1; then
            green "✓ Claude Code: set '$MCP_NAME' (user scope, alwaysLoad — tools load upfront)"
        else
            red   "✗ Claude Code: couldn't configure '$MCP_NAME' — try manually:"
            echo  "    claude mcp remove $MCP_NAME"
            echo  "    claude mcp add-json -s user $MCP_NAME '$probe_json'"
        fi
    fi
else
    dim "· Claude Code: 'claude' CLI not found (skipping)"
fi

# Note: prbe-knowledge-plugin (the Probe context-injection watcher) is
# built but not yet wired into this installer — see LAUNCH.md in
# prbe-ai/prbe-knowledge-plugin for the prerequisites that need to land
# before re-enabling that path.

# ---------------------------------------------------------------------------
# 2. Codex — global config via `codex mcp add`
# ---------------------------------------------------------------------------
if command -v codex >/dev/null 2>&1; then
    codex_probe="$(codex mcp get "$MCP_NAME" 2>/dev/null || true)"
    if printf "%s" "$codex_probe" | grep -qF "$MCP_URL"; then
        yellow "✓ Codex: '$MCP_NAME' already points at $MCP_URL (skipping)"
    elif [ -n "$codex_probe" ]; then
        dim "→ Updating Codex '$MCP_NAME' URL…"
        if codex mcp remove "$MCP_NAME" >/dev/null 2>&1 \
            && codex mcp add "$MCP_NAME" --url "$MCP_URL" >/dev/null 2>&1; then
            green "✓ Codex: updated '$MCP_NAME' to $MCP_URL"
        else
            red   "✗ Codex: couldn't update '$MCP_NAME' — try manually:"
            echo  "    codex mcp remove $MCP_NAME"
            echo  "    codex mcp add $MCP_NAME --url $MCP_URL"
        fi
    else
        dim "→ Adding to Codex…"
        if codex mcp add "$MCP_NAME" --url "$MCP_URL" >/dev/null 2>&1; then
            green "✓ Codex: added '$MCP_NAME' (global Codex config)"
        else
            red   "✗ Codex: 'codex mcp add' failed — try manually:"
            echo  "    codex mcp add $MCP_NAME --url $MCP_URL"
        fi
    fi
else
    dim "· Codex: 'codex' CLI not found (skipping)"
fi

# ---------------------------------------------------------------------------
# 3. Codex AGENTS.md — global behavior guidance at ~/.codex/AGENTS.md.
#    Codex loads this before repo-local AGENTS.md files.
# ---------------------------------------------------------------------------
GLOBAL_CODEX_AGENTS="$HOME/.codex/AGENTS.md"

_file_has_section() {
    [ -f "$1" ] && grep -qF "$PROBE_HEADING" "$1" 2>/dev/null
}

_append_text() {
    local target="$1" text="$2"
    mkdir -p "$(dirname "$target")"
    if [ -f "$target" ] && [ -s "$target" ]; then
        printf "\n\n%s\n" "$text" >> "$target"
    else
        printf "%s\n" "$text" > "$target"
    fi
}

if command -v codex >/dev/null 2>&1; then
    if _file_has_section "$GLOBAL_CODEX_AGENTS"; then
        yellow "✓ Codex AGENTS.md: Probe section already in ~/.codex/AGENTS.md (skipping)"
    elif prompt_yn "Add Probe guidance to ~/.codex/AGENTS.md so Codex reaches for it?"; then
        _append_text "$GLOBAL_CODEX_AGENTS" "$AGENT_GUIDANCE_SNIPPET"
        green "✓ Codex AGENTS.md: added Probe guidance to ~/.codex/AGENTS.md"
    else
        dim "· Codex AGENTS.md: skipped (you said no)"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Cursor — global config at ~/.cursor/mcp.json
# ---------------------------------------------------------------------------
CURSOR_DIR="$HOME/.cursor"
CURSOR_CFG="$CURSOR_DIR/mcp.json"
_merge_cursor_config() {
    # Atomically merge the Probe entry into ~/.cursor/mcp.json without
    # touching any sibling servers. Writes to a tmp file first so the
    # original is intact if anything fails. Returns 0 on success.
    local cfg="$1" name="$2" url="$3" tmp
    command -v python3 >/dev/null 2>&1 || return 1
    tmp=$(mktemp) || return 1
    if python3 - "$cfg" "$name" "$url" "$tmp" <<'PY' 2>/dev/null
import json, pathlib, sys
src, name, url, dst = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = json.loads(pathlib.Path(src).read_text() or "{}")
cfg.setdefault("mcpServers", {})[name] = {"url": url}
pathlib.Path(dst).write_text(json.dumps(cfg, indent=2) + "\n")
PY
    then
        mv "$tmp" "$cfg"
        return 0
    fi
    rm -f "$tmp"
    return 1
}

_cursor_probe_url() {
    local cfg="$1" name="$2"
    command -v python3 >/dev/null 2>&1 || return 1
    python3 - "$cfg" "$name" <<'PY' 2>/dev/null
import json, pathlib, sys

cfg_path, name = sys.argv[1], sys.argv[2]
cfg = json.loads(pathlib.Path(cfg_path).read_text() or "{}")
server = cfg.get("mcpServers", {}).get(name)
if not isinstance(server, dict):
    raise SystemExit(1)
url = server.get("url")
if not isinstance(url, str):
    raise SystemExit(1)
print(url)
PY
}

if [ -d "$CURSOR_DIR" ]; then
    if [ -f "$CURSOR_CFG" ] && grep -q "\"$MCP_NAME\"[[:space:]]*:" "$CURSOR_CFG" 2>/dev/null; then
        cursor_probe_url="$(_cursor_probe_url "$CURSOR_CFG" "$MCP_NAME" || true)"
        if [ "$cursor_probe_url" = "$MCP_URL" ]; then
            yellow "✓ Cursor: '$MCP_NAME' already points at $MCP_URL (skipping)"
        elif prompt_yn "Update existing Probe entry in Cursor's global config (~/.cursor/mcp.json)?"; then
            if _merge_cursor_config "$CURSOR_CFG" "$MCP_NAME" "$MCP_URL"; then
                green "✓ Cursor: updated '$MCP_NAME' in $CURSOR_CFG"
            else
                yellow "! Cursor: couldn't auto-update $CURSOR_CFG (python3 missing or JSON parse failed). Set this entry under \"mcpServers\":"
                echo  "      \"$MCP_NAME\": { \"url\": \"$MCP_URL\" }"
            fi
        else
            dim "· Cursor: skipped existing '$MCP_NAME' update (you said no)"
        fi
    elif prompt_yn "Add Probe to Cursor's global config (~/.cursor/mcp.json)?"; then
        if [ -f "$CURSOR_CFG" ]; then
            # Existing config — merge atomically via python3 so other
            # MCP servers in the file are preserved.
            if _merge_cursor_config "$CURSOR_CFG" "$MCP_NAME" "$MCP_URL"; then
                green "✓ Cursor: merged '$MCP_NAME' into $CURSOR_CFG"
            else
                yellow "! Cursor: couldn't auto-merge $CURSOR_CFG (python3 missing or JSON parse failed). Add this entry under \"mcpServers\":"
                echo  "      \"$MCP_NAME\": { \"url\": \"$MCP_URL\" }"
            fi
        else
            cat > "$CURSOR_CFG" <<JSON
{
  "mcpServers": {
    "$MCP_NAME": {
      "url": "$MCP_URL"
    }
  }
}
JSON
            green "✓ Cursor: wrote $CURSOR_CFG"
        fi
    else
        dim "· Cursor: skipped (you said no)"
    fi
else
    dim "· Cursor: ~/.cursor not found (skipping)"
fi

# ---------------------------------------------------------------------------
# 5. Cursor rule — project-local behavior guidance. Cursor does not have a
#    stable CLI for global rules, so we add a project rule when Cursor exists.
# ---------------------------------------------------------------------------
LOCAL_CURSOR_RULE=".cursor/rules/probe-knowledge.mdc"

if [ -d "$CURSOR_DIR" ]; then
    if _file_has_section "$LOCAL_CURSOR_RULE"; then
        yellow "✓ Cursor rule: Probe section already in $(pwd)/$LOCAL_CURSOR_RULE (skipping)"
    elif prompt_yn "Add a project Cursor rule at $LOCAL_CURSOR_RULE so Cursor reaches for Probe?"; then
        _append_text "$LOCAL_CURSOR_RULE" "$CURSOR_RULE_SNIPPET"
        green "✓ Cursor rule: added Probe guidance to $(pwd)/$LOCAL_CURSOR_RULE"
    else
        dim "· Cursor rule: skipped (you said no)"
    fi
fi

# ---------------------------------------------------------------------------
# 6. CLAUDE.md — try global first (~/.claude/CLAUDE.md, applies to every
#    project), fall back to per-repo (./CLAUDE.md). Either is sufficient,
#    so we skip if the section is already present in either location.
# ---------------------------------------------------------------------------
GLOBAL_CLAUDE_MD="$HOME/.claude/CLAUDE.md"
LOCAL_CLAUDE_MD="CLAUDE.md"

if _file_has_section "$GLOBAL_CLAUDE_MD"; then
    yellow "✓ CLAUDE.md: Probe section already in global ~/.claude/CLAUDE.md (skipping)"
elif _file_has_section "$LOCAL_CLAUDE_MD"; then
    yellow "✓ CLAUDE.md: Probe section already in $(pwd)/CLAUDE.md (skipping)"
elif prompt_yn "Add a Probe section to your global ~/.claude/CLAUDE.md so EVERY project's agent reaches for it?"; then
    _append_text "$GLOBAL_CLAUDE_MD" "$AGENT_GUIDANCE_SNIPPET"
    green "✓ CLAUDE.md: added Probe section to ~/.claude/CLAUDE.md (applies globally)"
elif prompt_yn "Add it just to this project's CLAUDE.md instead?"; then
    _append_text "$LOCAL_CLAUDE_MD" "$AGENT_GUIDANCE_SNIPPET"
    green "✓ CLAUDE.md: added Probe section to $(pwd)/CLAUDE.md"
else
    dim "· CLAUDE.md: skipped (you said no to both)"
fi

echo ""
green "Done."
echo ""
yellow "→ One more step: authenticate Probe in your AI assistant."
echo  "    Claude Code:  run /mcp, select 'Probe', choose Authenticate"
echo  "    Codex:        run codex mcp login Probe"
echo  "    Cursor:       Settings → MCP → click 'Authenticate' on the Probe entry"
echo ""
dim   "Restart your AI assistant if it was already running so it picks up the new server."
