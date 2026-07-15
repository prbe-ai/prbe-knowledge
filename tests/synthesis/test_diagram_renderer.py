"""Unit tests for the architecture-diagram Mermaid block renderer."""

from __future__ import annotations

from kb.synthesis.diagram_renderer import _build_mermaid_block
from kb.synthesis.index_renderer import _RepoEdge


def test_build_mermaid_block_empty_returns_empty_string() -> None:
    assert _build_mermaid_block([]) == ""


def test_build_mermaid_block_one_way_uses_labeled_arrow() -> None:
    edges = [
        _RepoEdge(
            source="prbe-ai/repo-a",
            target="prbe-ai/repo-b",
            bidirectional=False,
        )
    ]
    out = _build_mermaid_block(edges)
    assert "repo_a -->|one-way| repo_b" in out
    assert " --- " not in out


def test_build_mermaid_block_bidirectional_uses_double_arrow() -> None:
    edges = [
        _RepoEdge(
            source="prbe-ai/repo-a",
            target="prbe-ai/repo-b",
            bidirectional=True,
        )
    ]
    out = _build_mermaid_block(edges)
    assert "repo_a <--> repo_b" in out
    assert "|one-way|" not in out
    assert " --- " not in out
    # The bidirectional pair must NOT also produce a one-way `-->` line.
    # `<-->` contains `-->` as a substring, so test for the standalone
    # arrow with surrounding spaces instead.
    assert " --> " not in out


def test_build_mermaid_block_emits_each_node_once() -> None:
    edges = [
        _RepoEdge(
            source="prbe-ai/repo-a",
            target="prbe-ai/repo-b",
            bidirectional=False,
        ),
        _RepoEdge(
            source="prbe-ai/repo-a",
            target="prbe-ai/repo-c",
            bidirectional=False,
        ),
        _RepoEdge(
            source="prbe-ai/repo-b",
            target="prbe-ai/repo-c",
            bidirectional=True,
        ),
    ]
    out = _build_mermaid_block(edges)
    lines = out.splitlines()
    # Node-declaration lines look like `  ID[label]`.
    for node_id, label in [
        ("repo_a", "repo-a"),
        ("repo_b", "repo-b"),
        ("repo_c", "repo-c"),
    ]:
        decl = f"  {node_id}[{label}]"
        assert lines.count(decl) == 1, f"expected exactly one {decl!r}, got {lines!r}"


def test_build_mermaid_block_node_id_strips_owner_prefix_and_sanitizes() -> None:
    edges = [
        _RepoEdge(
            source="prbe-ai/prbe-knowledge",
            target="prbe-ai/prbe-backend",
            bidirectional=False,
        )
    ]
    out = _build_mermaid_block(edges)
    # Node ids are sanitized (hyphens -> underscores), labels keep hyphens.
    assert "  prbe_knowledge[prbe-knowledge]" in out
    assert "  prbe_backend[prbe-backend]" in out
    assert "prbe_knowledge -->|one-way| prbe_backend" in out
