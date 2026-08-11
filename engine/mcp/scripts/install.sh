#!/usr/bin/env bash
# Install the Probe Knowledge MCP server in Claude Code, Codex, and/or Cursor,
# and add guidance so coding agents use it when team context matters.
#
# Usage:
#   curl -fsSL https://mcp.knowledge.prbe.ai/install | bash
#
# Idempotent: re-running preserves configuration and refreshes managed guidance.

set -euo pipefail

MCP_NAME="probe-knowledge"
MCP_URL="https://mcp.knowledge.prbe.ai/mcp"
PROBE_HEADING="## Probe Knowledge MCP server (team operational memory)"
PROBE_BEGIN="<!-- probe-knowledge:begin (managed by Probe installer) -->"
PROBE_END="<!-- probe-knowledge:end -->"

# Names this server shipped under before, in every client. The rename is not a
# rename to any client: `claude mcp get probe-knowledge` finds nothing on a box
# that has `Probe`, so without an explicit sweep the installer would add a
# second entry pointing at the same URL and the user would load both tool sets.
#
# Note the server name is part of the tool name (`mcp__<server>__<tool>`), so it
# is restricted to [A-Za-z0-9_-] — "Probe Knowledge" with a space is rejected by
# `claude mcp add`.
LEGACY_MCP_NAMES="Probe probe"

# Section headings this snippet shipped under before. The section guard keys on
# the heading, so a changed heading reads as "no section present" and appends a
# duplicate. These are migrated in place instead.
LEGACY_PROBE_HEADINGS="## Probe MCP server (team operational memory)"

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
$PROBE_BEGIN
$PROBE_HEADING

\`$MCP_NAME\` searches team operational history in Slack, GitHub, Linear,
Notion, and Sentry. Use \`$MCP_NAME.search_knowledge\` only when a concrete
history question could change the answer or approach: prior rationale,
incidents, ownership, constraints, or similar/parallel work.

Do not call Probe for repo facts, routine implementation/review, status, or
shipping. A new request, plan, phase change, compaction, or elapsed time is not a
trigger. Reuse a relevant lookup for the same decision.

Query with an entity/keyword bag and \`top_k=5\`. Use
\`$MCP_NAME.get_source\`, retry, or follow \`related_entities\` only when needed
to resolve the decision. Surface useful findings, then continue from repo
evidence. Use \`$MCP_NAME.query_knowledge\` only for a direct question needing a
synthesized, cited answer. Probe is not source-code search.

If Probe informs a plan, cite the useful sources; otherwise omit a Probe note.
$PROBE_END
EOF

read -r -d '' CURSOR_RULE_SNIPPET <<EOF || true
---
description: Use Probe only for concrete team-history questions that can change the current decision.
globs: "**/*"
alwaysApply: true
---

$AGENT_GUIDANCE_SNIPPET
EOF

# Drop an entry this server shipped under previously, in whichever client, but
# only when it still points at our URL — a same-named entry aimed somewhere else
# belongs to someone else and is left alone.
#
# Renaming orphans the client's stored OAuth token: Claude Code keys those as
# `<serverName>|<urlHash>` in the macOS keychain, so the new name starts with an
# empty credential and the user re-auths once via /mcp. That is the cost of not
# leaving two live entries behind, and it is why this warns rather than doing it
# silently.
_sweep_legacy_mcp() {
    local client="$1" legacy found=""
    for legacy in $LEGACY_MCP_NAMES; do
        [ "$legacy" = "$MCP_NAME" ] && continue
        case "$client" in
            claude) found="$(claude mcp get "$legacy" 2>/dev/null || true)" ;;
            codex)  found="$(codex  mcp get "$legacy" 2>/dev/null || true)" ;;
        esac
        printf "%s" "$found" | grep -qF "$MCP_URL" || continue
        dim "→ Removing legacy '$legacy' entry (renamed to '$MCP_NAME')…"
        case "$client" in
            claude) claude mcp remove "$legacy" >/dev/null 2>&1 || true ;;
            codex)  codex  mcp remove "$legacy" >/dev/null 2>&1 || true ;;
        esac
        yellow "! $client: removed legacy '$legacy' — re-authenticate '$MCP_NAME' once (/mcp)"
    done
}

green "Probe Knowledge MCP installer"
dim "  Server: $MCP_URL"
dim "  Local alias: $MCP_NAME"
echo ""

# ---------------------------------------------------------------------------
# 1. Claude Code
# ---------------------------------------------------------------------------
if command -v claude >/dev/null 2>&1; then
    _sweep_legacy_mcp claude
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
    _sweep_legacy_mcp codex
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
    local target="$1" line
    [ -f "$target" ] || return 1
    grep -qF "$PROBE_BEGIN" "$target" 2>/dev/null && return 0
    grep -qF "$PROBE_HEADING" "$target" 2>/dev/null && return 0
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        grep -qF "$line" "$target" 2>/dev/null && return 0
    done <<EOF
$LEGACY_PROBE_HEADINGS
EOF
    return 1
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

# Refresh installer-managed guidance in place. Older installs predate the
# begin/end markers, so migrate their heading-delimited section too.
_upsert_probe_section() {
    local target="$1" text="$2" tmp replacement
    if ! _file_has_section "$target"; then
        _append_text "$target" "$text"
        return 0
    fi

    tmp=$(mktemp) || return 1
    replacement=$(mktemp) || { rm -f "$tmp"; return 1; }
    printf "%s\n" "$text" > "$replacement"

    if awk \
        -v begin="$PROBE_BEGIN" \
        -v end="$PROBE_END" \
        -v heading="$PROBE_HEADING" \
        -v legacy_headings="$LEGACY_PROBE_HEADINGS" \
        -v replacement="$replacement" '
        BEGIN { split(legacy_headings, old_headings, "\n") }
        function is_probe_heading(line,    i) {
            if (line == heading) return 1
            for (i in old_headings) if (line == old_headings[i]) return 1
            return 0
        }
        function emit(    line) {
            while ((getline line < replacement) > 0) print line
            close(replacement)
            inserted = 1
        }
        $0 == begin {
            if (!inserted) emit()
            managed = 1
            next
        }
        managed {
            if ($0 == end) managed = 0
            next
        }
        is_probe_heading($0) {
            if (!inserted) emit()
            legacy = 1
            next
        }
        legacy && $0 == "This is NOT a source-code search. For code, read the repo directly." {
            legacy = 0
            next
        }
        legacy && $0 ~ /^#{1,6}[[:space:]]/ {
            legacy = 0
            print
            next
        }
        !legacy { print }
        END {
            if (!inserted) {
                if (NR > 0) print ""
                emit()
            }
        }
    ' "$target" > "$tmp"
    then
        mv "$tmp" "$target"
        rm -f "$replacement"
        return 0
    fi

    rm -f "$tmp" "$replacement"
    return 1
}

if command -v codex >/dev/null 2>&1; then
    if _file_has_section "$GLOBAL_CODEX_AGENTS"; then
        _upsert_probe_section "$GLOBAL_CODEX_AGENTS" "$AGENT_GUIDANCE_SNIPPET"
        green "✓ Codex AGENTS.md: refreshed Probe guidance in ~/.codex/AGENTS.md"
    elif prompt_yn "Add Probe guidance to ~/.codex/AGENTS.md so Codex reaches for it?"; then
        _upsert_probe_section "$GLOBAL_CODEX_AGENTS" "$AGENT_GUIDANCE_SNIPPET"
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
        _upsert_probe_section "$LOCAL_CURSOR_RULE" "$AGENT_GUIDANCE_SNIPPET"
        green "✓ Cursor rule: refreshed Probe guidance in $(pwd)/$LOCAL_CURSOR_RULE"
    elif prompt_yn "Add a project Cursor rule at $LOCAL_CURSOR_RULE so Cursor reaches for Probe?"; then
        _upsert_probe_section "$LOCAL_CURSOR_RULE" "$CURSOR_RULE_SNIPPET"
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
    _upsert_probe_section "$GLOBAL_CLAUDE_MD" "$AGENT_GUIDANCE_SNIPPET"
    green "✓ CLAUDE.md: refreshed Probe guidance in global ~/.claude/CLAUDE.md"
elif _file_has_section "$LOCAL_CLAUDE_MD"; then
    _upsert_probe_section "$LOCAL_CLAUDE_MD" "$AGENT_GUIDANCE_SNIPPET"
    green "✓ CLAUDE.md: refreshed Probe guidance in $(pwd)/CLAUDE.md"
elif prompt_yn "Add a Probe section to your global ~/.claude/CLAUDE.md so EVERY project's agent reaches for it?"; then
    _upsert_probe_section "$GLOBAL_CLAUDE_MD" "$AGENT_GUIDANCE_SNIPPET"
    green "✓ CLAUDE.md: added Probe section to ~/.claude/CLAUDE.md (applies globally)"
elif prompt_yn "Add it just to this project's CLAUDE.md instead?"; then
    _upsert_probe_section "$LOCAL_CLAUDE_MD" "$AGENT_GUIDANCE_SNIPPET"
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
