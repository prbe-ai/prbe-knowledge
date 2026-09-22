"""T1 -- live Jev contract probe.

Nobody on this team has called Jev before, so the replay must not assume the
four things it depends on. This script establishes them by calling the API:

  1. how many Noul questions ride ONE request (docs say "multiple", not a cap);
  2. what the server counts as tokens for a state of known character length --
     the replay's batcher needs a real chars-per-token ratio, not chars/4;
  3. the error shape when a request is over cap, so the batcher can react to it
     instead of guessing its way under it;
  4. latency by request shape, and whether the same request scores the same
     twice (a scorer that drifts cannot be A/B'd -- see the retrieval memory
     "one search is never a measurement").

Reads TYPESAFE_API_KEY from the environment; never prints it. One JSON object
per probe on stdout.

Usage:
    TYPESAFE_API_KEY=... .venv/bin/python scripts/jev_shadow/probe_contract.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from typesafe_sdk import (
    Noul,
    Choice,
    RetryPolicy,
    TypeSafeAPIError,
    TypeSafeClient,
)

# Real gatherer prose, so token counts reflect the corpus rather than lorem.
FILLER = (
    "The gatherer runs single-turn with tool_choice required and hand-curates "
    "the pre-fan-out pool, emitting the chunks it picks. A backfilled chunk was "
    "read by no model, so it can carry no span and no why_relevant. "
)

client = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY"),
    # No retries inside a probe: a retried 4xx would hide the cap we are looking
    # for behind a slower version of the same answer.
    retry=RetryPolicy(max_retries=0),
    timeout=60.0,
)


def emit(rec: dict[str, Any]) -> None:
    print(json.dumps(rec, default=str), flush=True)


def probe(label: str, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    state_chars = len(json.dumps(state))
    t0 = time.perf_counter()
    rec: dict[str, Any] = {
        "probe": label,
        "n_questions": len(questions),
        "state_chars": state_chars,
    }
    try:
        resp = client.system_one(state=state, questions=questions)
        rec["ms"] = round((time.perf_counter() - t0) * 1000)
        rec["ok"] = True
        rec["model"] = getattr(resp, "model", None)
        usage = getattr(resp, "usage", None)
        rec["usage"] = (
            usage.model_dump() if hasattr(usage, "model_dump") else usage
        )
        answers = getattr(resp, "answers", {}) or {}
        rec["n_answers"] = len(answers)
        rec["_values"] = {
            k: (v.model_dump() if hasattr(v, "model_dump") else v)
            for k, v in answers.items()
        }
        first_key = next(iter(answers), None)
        if first_key is not None:
            rec["answer_shape"] = rec["_values"][first_key]
        # chars per billed input token -- the number the batcher actually needs.
        tokens = None
        if isinstance(rec["usage"], dict):
            for k in ("input_tokens", "prompt_tokens", "tokens", "total_tokens"):
                if isinstance(rec["usage"].get(k), int):
                    tokens = rec["usage"][k]
                    break
        if tokens:
            rec["chars_per_token"] = round(state_chars / tokens, 2)
            rec["input_tokens"] = tokens
    except TypeSafeAPIError as exc:
        rec["ms"] = round((time.perf_counter() - t0) * 1000)
        rec["ok"] = False
        rec["error_class"] = type(exc).__name__
        rec["status"] = getattr(exc, "status_code", None)
        # Stringify explicitly: httpx-family timeouts stringify EMPTY, and an
        # empty reason is how a cap probe reads as an unexplained failure.
        rec["error"] = str(exc)[:300] or "<empty str(exc)>"
    except Exception as exc:  # transport, validation, anything else
        rec["ms"] = round((time.perf_counter() - t0) * 1000)
        rec["ok"] = False
        rec["error_class"] = type(exc).__name__
        rec["error"] = str(exc)[:300] or "<empty str(exc)>"
    values = rec.pop("_values", None)
    emit(rec)
    rec["_values"] = values
    return rec


def chunks(n: int, repeat: int = 1) -> dict[str, str]:
    return {f"c{i}": f"chunk {i}: " + FILLER * repeat for i in range(n)}


def nouls(n: int) -> dict[str, Noul]:
    return {
        f"c{i}": Noul(
            instructions=(
                f"Chunk c{i} in the state answers, or directly supports an "
                f"answer to, the query."
            )
        )
        for i in range(n)
    }


QUERY = "why did the gatherer return nothing for a doc that is in the pool"

# ---------------------------------------------------------------- 1. auth
probe(
    "auth_minimal",
    {"query": QUERY, "chunks": {"c0": "the tenant GUC was unset so RLS matched zero rows"}},
    {"c0": Noul(instructions="Chunk c0 answers the query.")},
)

# ------------------------------------------------- 2. questions-per-request
# Small state throughout, so a failure here is a QUESTION-count limit rather
# than a state-size limit. Stops climbing once one rung fails.
for n in (10, 50, 100, 150, 200, 255, 300):
    r = probe(f"questions_{n}", {"query": QUERY, "chunks": chunks(n)}, nouls(n))
    if not r.get("ok"):
        emit({"probe": "questions_ladder", "stopped_at": n})
        break

# ------------------------------------------------------- 3. state-size cap
# Few questions, growing state: isolates the 32k-token state cap and gives the
# chars-per-token ratio at realistic sizes. Our p90 pool is ~120k chars.
for kchars in (40, 80, 110, 130, 160, 200):
    n = max(1, (kchars * 1000) // (len(FILLER) * 6))
    r = probe(
        f"state_{kchars}k_chars",
        {"query": QUERY, "chunks": chunks(n, repeat=6)},
        nouls(min(20, n)),
    )
    if not r.get("ok"):
        emit({"probe": "state_ladder", "stopped_at_kchars": kchars})
        break

# --------------------------------------------- 4. Choice (extraction shape)
opts = {f"cand_{i}": f"candidate entity number {i}" for i in range(120)}
opts["none_of_these"] = "none of the listed candidates is what the query is about"
probe(
    "choice_121_options",
    {"query": "what did the auth refactor change", "candidates": sorted(opts)},
    {
        "pick": Choice(
            instructions="Which candidate is the query about? Choose none_of_these if unsure.",
            criteria=opts,
        )
    },
)

# ------------------------------------------------------------ 5. determinism
st = {"query": QUERY, "chunks": chunks(30)}
qs = nouls(30)
runs = [probe(f"determinism_{k}", st, qs) for k in range(3)]
series = [r.get("_values") or {} for r in runs if r.get("ok")]
if len(series) >= 2:
    keys = sorted(set().union(*(set(v) for v in series)))
    drifts = []
    for k in keys:
        vals = [v.get(k, {}).get("noul") for v in series]
        vals = [x for x in vals if isinstance(x, (int, float))]
        if len(vals) >= 2:
            drifts.append(max(vals) - min(vals))
    emit({
        "probe": "determinism_summary",
        "runs_compared": len(series),
        "questions": len(drifts),
        "identical": all(d == 0 for d in drifts) if drifts else None,
        "max_drift": max(drifts) if drifts else None,
        "mean_drift": round(sum(drifts) / len(drifts), 6) if drifts else None,
    })
else:
    emit({"probe": "determinism_summary", "runs_compared": len(series), "note": "too few OK runs"})

emit({"probe": "done", "runs": len(runs)})
