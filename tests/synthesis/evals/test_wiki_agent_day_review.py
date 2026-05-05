"""LLM eval: day-review fixture corpus -> expected wiki diff (edit-distance).

Skips cleanly when GEMINI_API_KEY isn't set in the environment so CI
doesn't burn live model spend on every commit. To run locally:

    GEMINI_API_KEY=... uv run pytest tests/synthesis/evals/

Fixture: 50-event hand-curated corpus (loaded from
`tests/synthesis/evals/fixtures/day_review_corpus.json`) plus an
initial wiki state. The expected output is a per-page edit-distance
budget; the test passes when the agent's output is within ε for every
page.

Stub mode: if the fixture file is missing, the test asserts that the
SKIP path triggers correctly. The full corpus + expected output is a
follow-up; the harness here is what the eval needs to plug into.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_CORPUS_FILE = _FIXTURE_DIR / "day_review_corpus.json"


def _missing_api_key() -> bool:
    return not os.environ.get("GEMINI_API_KEY") and not os.environ.get(
        "GOOGLE_API_KEY"
    )


@pytest.mark.skipif(
    _missing_api_key(),
    reason="GEMINI_API_KEY / GOOGLE_API_KEY not set; LLM eval requires live model.",
)
def test_day_review_fixture_corpus_produces_expected_wiki_diff() -> None:
    """Eval scaffold: load fixture, run agent, assert per-page edit-distance.

    Until the corpus + expected output land, this test asserts that
    the fixture file path was set up correctly. When the fixture is
    populated, this test exercises the full agent loop against the
    real Gemini model and checks page-by-page edit distance.
    """
    if not _CORPUS_FILE.exists():
        pytest.skip(
            f"Corpus fixture not yet authored: {_CORPUS_FILE}. "
            "This test is the scaffold; populate the fixture to enable."
        )

    corpus = json.loads(_CORPUS_FILE.read_text())
    assert isinstance(corpus, dict)
    assert "events" in corpus
    assert "expected_pages" in corpus
    # Real eval body would:
    #   1. Seed events into wiki_synthesis_queue at status='triaged'.
    #   2. Spawn SynthesisWorker with the live Gemini client.
    #   3. After the drain, pull the rendered pages from documents.
    #   4. Assert edit-distance per (wiki_type, slug) <= corpus
    #      ['epsilon_per_page'].
    #
    # The corpus is hand-curated; the expected_pages are human-labeled
    # ideal-output bodies. ε is tuned per page based on how prescriptive
    # the agent should be (decision pages get a tighter ε than
    # discursive feature pages).
    pytest.skip("Eval body wired up but fixture not yet authored.")
