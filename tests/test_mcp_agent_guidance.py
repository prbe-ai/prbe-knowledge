"""Regression tests for decision-based Probe prompting."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from engine.mcp.consts import MCP_INSTRUCTIONS

ROOT = Path(__file__).resolve().parents[1]
GUIDANCE_SURFACES = (
    ROOT / "README.md",
    ROOT / "engine" / "mcp" / "consts.py",
    ROOT / "engine" / "mcp" / "server.py",
    ROOT / "engine" / "mcp" / "scripts" / "install.sh",
)


def test_guidance_uses_decision_triggers_not_phase_gates() -> None:
    combined = "\n".join(path.read_text() for path in GUIDANCE_SURFACES)

    assert "At every new user request" not in combined
    assert "Before each non-trivial implementation phase" not in combined
    assert "cost of a redundant lookup is small" not in combined
    assert "Do NOT skip Probe on these triggers" not in combined

    assert "concrete history question could change" in combined
    assert "routine implementation/review, status, or" in combined
    assert "phase change, compaction, or elapsed time is not a" in combined


def test_runtime_instructions_explicitly_exempt_shipping() -> None:
    assert "concrete history question could change" in MCP_INSTRUCTIONS
    assert "routine implementation/review, status, or\nshipping" in MCP_INSTRUCTIONS
    assert "Reuse a relevant lookup for the same decision" in MCP_INSTRUCTIONS
    assert "only when needed to resolve the decision" in MCP_INSTRUCTIONS
    assert "otherwise omit a Probe note" in MCP_INSTRUCTIONS


def test_installer_refreshes_shared_legacy_guidance(tmp_path: Path) -> None:
    home = tmp_path / "home"
    agents = home / ".codex" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text(
        "# Personal rule\n\n"
        "Keep this line.\n\n"
        "## Probe MCP server (team operational memory)\n\n"
        "OLD CODEX GUIDANCE\n\n"
        "This is NOT a source-code search. For code, read the repo directly.\n\n"
        "## Another tool\n\n"
        "Keep this section too.\n"
    )

    claude_md = home / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True)
    claude_md.write_text(
        "# Claude rule\n\n"
        "## Probe MCP server (team operational memory)\n\n"
        "OLD CLAUDE GUIDANCE\n\n"
        "This is NOT a source-code search. For code, read the repo directly.\n"
    )

    cursor_dir = home / ".cursor"
    cursor_rule = tmp_path / ".cursor" / "rules" / "probe-knowledge.mdc"
    cursor_dir.mkdir()
    (cursor_dir / "mcp.json").write_text(
        '{"mcpServers":{"probe-knowledge":{"url":"https://mcp.knowledge.prbe.ai/mcp"}}}\n'
    )
    cursor_rule.parent.mkdir(parents=True)
    cursor_rule.write_text(
        '---\ndescription: Old rule\nglobs: "**/*"\nalwaysApply: true\n---\n\n'
        "## Probe MCP server (team operational memory)\n\n"
        "OLD CURSOR GUIDANCE\n\n"
        "This is NOT a source-code search. For code, read the repo directly.\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for client in ("codex", "claude"):
        executable = fake_bin / client
        executable.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = mcp ] && [ "$2" = get ] '
            '&& [ "$3" = probe-knowledge ]; then\n'
            "  printf 'URL: https://mcp.knowledge.prbe.ai/mcp\\n'\n"
            + ("  printf 'alwaysLoad: true\\n'\n" if client == "claude" else "")
            + "  exit 0\nfi\nexit 1\n"
        )
        executable.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    def run_installer() -> None:
        subprocess.run(
            ["/bin/bash", str(ROOT / "engine" / "mcp" / "scripts" / "install.sh")],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    run_installer()
    migrated = agents.read_text()
    assert "OLD CODEX GUIDANCE" not in migrated
    assert "<!-- probe-knowledge:begin (managed by Probe installer) -->" in migrated
    assert "routine implementation/review, status, or" in migrated
    assert "Keep this line." in migrated
    assert "Keep this section too." in migrated
    assert "OLD CLAUDE GUIDANCE" not in claude_md.read_text()
    assert "# Claude rule" in claude_md.read_text()
    assert "routine implementation/review, status, or" in claude_md.read_text()
    assert "OLD CURSOR GUIDANCE" not in cursor_rule.read_text()
    assert "description: Old rule" in cursor_rule.read_text()
    assert "routine implementation/review, status, or" in cursor_rule.read_text()

    migrated_claude = claude_md.read_text()
    migrated_cursor = cursor_rule.read_text()
    run_installer()
    assert agents.read_text() == migrated
    assert claude_md.read_text() == migrated_claude
    assert cursor_rule.read_text() == migrated_cursor
