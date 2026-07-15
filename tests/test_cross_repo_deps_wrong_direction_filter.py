"""Unit tests for the wrong-direction evidence filter."""

from __future__ import annotations

from kb.code_graph.cross_repo_deps import (
    VerifiedMatch,
    _filter_wrong_direction_evidence,
    _is_wrong_direction_snippet,
)

WRONG_DIRECTION_SAMPLES = [
    "INTERNAL_BACKEND_API_KEY",
    "#   - INTERNAL_BACKEND_API_KEY     — value prbe-backend presents",
    'description="Shared secret prbe-backend presents on X-Internal-Backend-Key"',
    "secret (INTERNAL_BACKEND_API_KEY for prbe-backend,",
    "Mirrors prbe-backend's pattern: fetch JWKS once, cache by `kid`",
    "# Should be prbe-backend's JWKS URL.",
    "internal HTTP routes (used by the prbe-knowledge-plugin watcher, not by",
    "# Internal routes used by the prbe-knowledge-plugin watcher daemon.",
    '"""Internal HTTP routes — for the prbe-knowledge-plugin watcher daemon.',
    "The prbe-knowledge-plugin watcher calls /internal/summarize every ~30s",
    "Middleware implements auth logic based on prbe-orchestrator callers.",
]


FORWARD_DIRECTION_SAMPLES = [
    '"""HTTP client for prbe-knowledge retrieval service.',
    "super().__init__(f\"prbe-knowledge http {status}: {body[:200]}\")",
    "| `KNOWLEDGE_QUERY_URL` | Base URL of `prbe-knowledge` retrieval |",
    "import { something } from '@prbe/sdk'",
    "BACKEND_URL=https://prbe-backend.internal",
    'fetch(f"https://{BACKEND_HOST}/api/...")',
]


def test_wrong_direction_samples_all_match() -> None:
    for snippet in WRONG_DIRECTION_SAMPLES:
        assert _is_wrong_direction_snippet(snippet), (
            f"Expected wrong-direction match for: {snippet!r}"
        )


def test_forward_direction_samples_all_clear() -> None:
    for snippet in FORWARD_DIRECTION_SAMPLES:
        assert not _is_wrong_direction_snippet(snippet), (
            f"Unexpected wrong-direction match for: {snippet!r}"
        )


def test_filter_drops_target_with_only_wrong_direction_snippets() -> None:
    verified = [
        VerifiedMatch(
            file_path="app/config.py",
            line_number=21,
            snippet="INTERNAL_ORCHESTRATOR_API_KEY — value prbe-orchestrator presents",
            target_repo="prbe-ai/prbe-orchestrator",
            reason="auth-key config",
        ),
        VerifiedMatch(
            file_path="app/dependencies/auth_context.py",
            line_number=15,
            snippet="Middleware implements auth logic based on prbe-orchestrator callers",
            target_repo="prbe-ai/prbe-orchestrator",
            reason="caller-identity middleware",
        ),
    ]
    kept = _filter_wrong_direction_evidence(verified)
    assert kept == []


def test_filter_keeps_target_when_one_snippet_is_forward() -> None:
    verified = [
        VerifiedMatch(
            file_path="app/config.py",
            line_number=21,
            snippet="INTERNAL_ORCHESTRATOR_API_KEY — value prbe-orchestrator presents",
            target_repo="prbe-ai/prbe-orchestrator",
            reason="auth-key config",
        ),
        VerifiedMatch(
            file_path="app/clients/orchestrator.py",
            line_number=1,
            snippet='"""HTTP client for prbe-orchestrator agent dispatch."""',
            target_repo="prbe-ai/prbe-orchestrator",
            reason="HTTP client",
        ),
    ]
    kept = _filter_wrong_direction_evidence(verified)
    assert len(kept) == 2
    assert {m.file_path for m in kept} == {
        "app/config.py",
        "app/clients/orchestrator.py",
    }


def test_filter_handles_multiple_targets_independently() -> None:
    verified = [
        VerifiedMatch(
            file_path="app/config.py",
            line_number=21,
            snippet="INTERNAL_ORCHESTRATOR_API_KEY",
            target_repo="prbe-ai/prbe-orchestrator",
            reason="x",
        ),
        VerifiedMatch(
            file_path="app/clients/knowledge.py",
            line_number=1,
            snippet='"""HTTP client for prbe-knowledge retrieval service."""',
            target_repo="prbe-ai/prbe-knowledge",
            reason="x",
        ),
    ]
    kept = _filter_wrong_direction_evidence(verified)
    assert len(kept) == 1
    assert kept[0].target_repo == "prbe-ai/prbe-knowledge"


def test_filter_empty_input_returns_empty() -> None:
    assert _filter_wrong_direction_evidence([]) == []
