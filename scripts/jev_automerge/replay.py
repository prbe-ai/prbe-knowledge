"""Ask the PRODUCTION judges about a frozen decision set.

    TYPESAFE_API_KEY=... python -m scripts.jev_automerge.replay \
        --decisions <dir>/decisions.jsonl --out <dir>/results.jsonl [--repeats 2] \
        [--gptoss http://<litellm>/v1]        # needs LLM_GATEWAY_KEY

Jev is asked with `jev_judge.build_request` + `jev.post_choice` -- the exact
request production sends -- so changing the instructions, criteria or value
trimming shows up here before it ships. gpt-oss (optional) gets the analyzer's
own system prompt, user prompt and response schema.

Cost: ~1.8k Jev input tokens per call (~$0.00008); gpt-oss ~$0.0007 a call.
Keys are read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from engine.ingest.auto_merge import analyzer as az
from engine.ingest.auto_merge.jev_judge import NONE_OF_THESE, build_request
from engine.retrieval.agent import jev
from engine.shared.constants import AUTO_MERGE_JEV_MODEL
from scripts.jev_automerge import kb


def _node(d: dict) -> dict:
    return {"label": d["label"], "canonical_id": d["canonical_id"], "properties": d["properties"] or {},
            "degree": d["degree"], "node_id": d.get("node_id")}


async def ask_jev(client: httpx.AsyncClient, api_key: str, d: dict) -> dict:
    cands = [az.Candidate(**c) for c in d["candidates"]]
    state, question, keys = build_request(_node(d), cands)
    t0 = time.perf_counter()
    try:
        ans = await jev.post_choice(state, question, api_key=api_key, model=AUTO_MERGE_JEV_MODEL,
                                    breaker=jev.Breaker(), client=client)
    except jev.JevError as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "ms": (time.perf_counter() - t0) * 1000}
    primary = None if ans.choice == NONE_OF_THESE else keys[ans.choice].canonical_id
    return {
        "ms": ans.elapsed_ms, "choice": ans.choice, "primary": primary,
        "p": ans.probabilities[ans.choice], "model": ans.model, "input_tokens": ans.input_tokens,
        "probs": {(keys[k].canonical_id if k in keys else k): v for k, v in ans.probabilities.items()},
    }


async def ask_gptoss(client: httpx.AsyncClient, url: str, key: str, d: dict) -> dict:
    cands = [az.Candidate(**c) for c in d["candidates"]]
    body = {"model": az.SEARCH_AGENT_INFERENCE_MODEL, **az.gptoss_request(_node(d), cands)}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
    except httpx.HTTPError as exc:
        return {"error": type(exc).__name__, "ms": (time.perf_counter() - t0) * 1000}
    out = {"ms": (time.perf_counter() - t0) * 1000}
    if r.status_code != 200:
        return {**out, "error": f"http_{r.status_code}"}
    j = r.json()
    usage = j.get("usage") or {}
    out.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
    try:
        v = json.loads(((j.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    except ValueError:
        return {**out, "error": "unparseable"}
    out.update(verdict=v.get("verdict"), primary=v.get("primary_canonical_id"),
               confidence=v.get("confidence"), rationale=v.get("rationale"))
    if out["verdict"] == "duplicate" and out["primary"] not in {c["canonical_id"] for c in d["candidates"]}:
        out["error"] = "hallucinated_primary"
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2, help="identical Jev calls per decision (drift)")
    ap.add_argument("--gptoss", help="LiteLLM base URL (…/v1); also asks gpt-oss with the production prompt")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    api_key = os.environ.get("TYPESAFE_API_KEY") or sys.exit("TYPESAFE_API_KEY is not set")
    gkey = os.environ.get("LLM_GATEWAY_KEY", "")
    decisions = kb.read_jsonl(args.decisions)
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async with httpx.AsyncClient() as client:
        async def one(d: dict) -> dict:
            nonlocal done
            async with sem:
                res = {"did": d["did"], "jev": [await ask_jev(client, api_key, d) for _ in range(args.repeats)]}
                if args.gptoss:
                    res["gptoss"] = await ask_gptoss(client, args.gptoss, gkey, d)
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(decisions)}", file=sys.stderr)
            return res

        results = await asyncio.gather(*(one(d) for d in decisions))
    kb.write_jsonl(args.out, results)
    print(f"wrote {len(results)} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
