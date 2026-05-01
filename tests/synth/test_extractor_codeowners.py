"""CODEOWNERS — used to map service paths to canonical Person ids."""

from __future__ import annotations

from pathlib import Path

from scripts.synth.extractor.codeowners import (
    CodeownerRule,
    find_codeowners_file,
    parse_codeowners,
    resolve_owners,
)


def test_parses_basic_rules() -> None:
    text = """
# infra
/infra/  @alice @bob
/services/payments/   @payments-team
*.md  @docs
""".strip()
    rules = parse_codeowners(text)
    assert rules == (
        CodeownerRule(pattern="/infra/", owners=("@alice", "@bob")),
        CodeownerRule(pattern="/services/payments/", owners=("@payments-team",)),
        CodeownerRule(pattern="*.md", owners=("@docs",)),
    )


def test_skips_blank_and_comment_lines() -> None:
    text = """
# only comments
   # indented comment

   /a/  @x
""".strip()
    [rule] = parse_codeowners(text)
    assert rule == CodeownerRule(pattern="/a/", owners=("@x",))


def test_handles_no_owners_line() -> None:
    """A line with a pattern and no owners is treated as 'unowned' (zero owners)."""
    text = "/legacy/\n"
    [rule] = parse_codeowners(text)
    assert rule.pattern == "/legacy/"
    assert rule.owners == ()


def test_resolve_owners_picks_last_matching_rule() -> None:
    rules = (
        CodeownerRule(pattern="*", owners=("@everyone",)),
        CodeownerRule(pattern="/services/payments/", owners=("@payments-team",)),
    )
    assert resolve_owners("/services/payments/handler.py", rules) == ("@payments-team",)
    assert resolve_owners("/random.py", rules) == ("@everyone",)


def test_finds_codeowners_in_canonical_locations(tmp_path: Path) -> None:
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "CODEOWNERS").write_text("/x  @y\n")
    assert find_codeowners_file(tmp_path) == tmp_path / ".github" / "CODEOWNERS"


def test_finds_codeowners_at_root(tmp_path: Path) -> None:
    (tmp_path / "CODEOWNERS").write_text("/x  @y\n")
    assert find_codeowners_file(tmp_path) == tmp_path / "CODEOWNERS"


def test_returns_none_when_no_codeowners(tmp_path: Path) -> None:
    assert find_codeowners_file(tmp_path) is None
