"""The vendored credential scrubber (research-os `tap_core/secrets.py`, synced by
`scripts/sync_credential_scrubber.py`) removes a Capy API key: `capy_` plus 43
characters, org-wide with its owner's full role."""

from engine.ingest._credential_secrets import redact

KEY = "capy_" + "Ab3_-" * 8 + "xyz"


def test_a_capy_key_in_a_transcript_line_is_redacted():
    text, rules = redact(f"export CAPY_API_KEY={KEY}")
    assert KEY not in text
    assert "capy-api-key" in rules


def test_a_word_that_merely_starts_with_capy_is_kept():
    text, _ = redact("the capy_skill folder and capy_" + "a" * 44)
    assert "capy_skill" in text
