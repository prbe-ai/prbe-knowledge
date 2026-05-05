"""Unit tests for prompt_typed_confirm — the operator-types-customer-id-back
safety prompt for non-eval seed."""

from io import StringIO

from scripts.synth.seed import prompt_typed_confirm


def test_exact_match(monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("cust-prbe-acme-co\n"))
    assert prompt_typed_confirm("cust-prbe-acme-co") is True


def test_mismatch(monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("cust-prbe-acme\n"))
    assert prompt_typed_confirm("cust-prbe-acme-co") is False


def test_whitespace_stripped(monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("  cust-prbe-acme-co  \n"))
    assert prompt_typed_confirm("cust-prbe-acme-co") is True


def test_empty_input(monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("\n"))
    assert prompt_typed_confirm("cust-prbe-acme-co") is False
