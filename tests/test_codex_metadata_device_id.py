"""Regression: CodexConnector inherits device_id propagation from ClaudeCodeConnector.

If CodexConnector ever overrides _build_session_doc and forgets to copy the
device_id field into metadata, per-device stats break silently. This test
catches that drift.
"""

from kb.handlers.claude_code import (
    ClaudeCodeConnector,
    CodexConnector,
)


def test_codex_connector_inherits_build_session_doc() -> None:
    # If a future change overrides _build_session_doc on CodexConnector, this
    # test fails — forcing the author to verify device_id propagation manually.
    assert (
        CodexConnector._build_session_doc  # type: ignore[attr-defined]
        is ClaudeCodeConnector._build_session_doc  # type: ignore[attr-defined]
    ), (
        "CodexConnector overrode _build_session_doc — confirm device_id is "
        "still written into metadata, then update this test."
    )


def test_codex_connector_inherits_build_unit_doc() -> None:
    # Derived Codex artifacts need the same denormalized device_id stamp and
    # historical parent fallback contract as Claude Code artifacts.
    assert (
        CodexConnector._build_unit_doc  # type: ignore[attr-defined]
        is ClaudeCodeConnector._build_unit_doc  # type: ignore[attr-defined]
    ), (
        "CodexConnector overrode _build_unit_doc — confirm device_id is still "
        "copied from the parent session, then update this test."
    )
