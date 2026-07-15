"""Thin deploy wrapper — canonical module: engine.retrieval.agent.trace_analyzer.

Kept so `python -m services.retrieval.agent.trace_analyzer` (the nightly
trace-digest K8s Job) keeps working unchanged.
"""

import sys

from engine.retrieval.agent.trace_analyzer.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
