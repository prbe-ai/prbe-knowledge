#!/usr/bin/env python3
"""Grader-controlled record/replay LLM proxy (HELD-OUT eval infra; sandbox/, agents cannot edit).

Sits between the product (LLM_GATEWAY_URL -> proxy) and the real gateway (REPLAY_UPSTREAM_URL) so the
held-out grade can measure RETRIEVAL latency without the LLM-serving variance that swamps it (the
gatherer's per-call latency dwarfs and hides the candidate's own code-path cost). Modes:

  record : forward real, cache (path+body -> response). No delay. (A recall/warm pass.)
  replay : cached response after a synthetic delay -> wall-clock = deterministic code path, zero LLM
           variance. On a cache MISS (the product produced a request that didn't occur byte-identically
           during record -> gatherer non-determinism), forward it for real + cache so the grade never
           crashes; misses are a minority so p50 stays clean. Misses are logged.
  auto   : record-on-miss (forward, cache, NO delay) AND replay-on-hit (cached + delay). One mode the
           grader can leave set for the WHOLE grade: pass 0 fills the cache at real cost, every repeat
           pass replays deterministically. No /__mode round-trip needed (the host grader can't reach an
           in-container control endpoint without publishing it). This is the default the engine sets.
  off    : passthrough.

THE DELAY (replay / auto-hit) is TOKEN/BYTE-AWARE, not flat: it approximates an LLM's cost shape —
prefill ∝ request bytes, decode ∝ response bytes —

    delay = REPLAY_BASE_MS + REPLAY_PREFILL_MS_PER_KB·(req_kb) + REPLAY_DECODE_MS_PER_KB·(resp_kb)

so a candidate that widens the gatherer's context (bigger requests) or pads its output (bigger
responses) pays REAL graded latency for it — closing the "a wider/padded candidate pool is free"
hole that a flat REPLAY_FIXED_MS left open. When none of the per-kb knobs are set it degrades to the
flat REPLAY_FIXED_MS (back-compat).

Path-agnostic, non-streaming (/retrieve never streams). Retries transient upstream 5xx.

THREAT MODEL (honest): the proxy runs INSIDE the product container, where the agent's code runs as
root, so a candidate that actively tampers (POST /__mode off, kill the proxy, or call
REPLAY_UPSTREAM_URL directly) could dodge the delay. That is an explicit, diff-visible cheat in
sandbox-excluded territory; the durable hardening is to run this proxy on the HOST (like the grader)
so the agent can't reach it. Tracked as a follow-up.
"""
import asyncio
import hashlib
import json
import os

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

UPSTREAM = os.environ["REPLAY_UPSTREAM_URL"].rstrip("/")
STATE = {"mode": os.environ.get("REPLAY_MODE", "off")}
CACHE = {}
LOGF = "/tmp/replay_proxy.log"
_client = httpx.AsyncClient(timeout=180.0)

# --- token/byte-aware replay delay (falls back to the flat REPLAY_FIXED_MS when no per-kb knob is set) ---
BASE_MS = float(os.environ.get("REPLAY_BASE_MS", "0"))
PREFILL_MS_PER_KB = float(os.environ.get("REPLAY_PREFILL_MS_PER_KB", "0"))
DECODE_MS_PER_KB = float(os.environ.get("REPLAY_DECODE_MS_PER_KB", "0"))
FIXED_MS = float(os.environ.get("REPLAY_FIXED_MS", "1000"))
_BYTE_AWARE = BASE_MS > 0 or PREFILL_MS_PER_KB > 0 or DECODE_MS_PER_KB > 0


def _delay_s(req_bytes: int, resp_bytes: int) -> float:
    """Synthetic per-call delay (seconds). Byte-aware when any per-kb knob is set: prefill ∝ request
    bytes, decode ∝ response bytes; else the flat REPLAY_FIXED_MS for back-compat."""
    if not _BYTE_AWARE:
        return FIXED_MS / 1000.0
    ms = BASE_MS + PREFILL_MS_PER_KB * (req_bytes / 1024.0) + DECODE_MS_PER_KB * (resp_bytes / 1024.0)
    return ms / 1000.0


def _key(path, body):
    return hashlib.sha256(path.encode() + b"\n" + body).hexdigest()


def _flog(m):
    try:
        with open(LOGF, "a") as f:
            f.write(m + "\n")
    except Exception:  # noqa: BLE001
        pass


async def _forward(method, path, body, fwd):
    up = None
    for attempt in range(6):
        up = await _client.request(method, UPSTREAM + path, content=body, headers=fwd)
        if up.status_code in (429, 502, 503, 504) and attempt < 5:
            await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
            continue
        break
    return up


def _cache_put(key, up):
    CACHE[key] = {"status": up.status_code, "body": up.content,
                  "ct": up.headers.get("content-type", "application/json")}


async def _proxy(request):
    path = request.url.path + (("?" + request.url.query) if request.url.query else "")
    body = await request.body()
    fwd = {k: v for k, v in request.headers.items()
           if k.lower() not in ("host", "content-length", "accept-encoding")}
    mode = STATE["mode"]
    try:
        model = (json.loads(body).get("model") if body else None)
    except Exception:  # noqa: BLE001
        model = None

    key = _key(path, body)
    hit = CACHE.get(key)
    # REPLAY (or AUTO) cache HIT: serve the recorded response after the byte-aware delay.
    if mode in ("replay", "auto") and hit is not None:
        await asyncio.sleep(_delay_s(len(body), len(hit["body"])))
        return Response(content=hit["body"], status_code=hit["status"], media_type=hit["ct"])
    # REPLAY cache MISS: non-determinism — forward real + cache so the grade never crashes (no delay).
    if mode == "replay":
        _flog(f"REPLAY-MISS {path} model={model} bytes={len(body)}")

    # record / auto-miss / off / replay-miss: forward for real.
    up = await _forward(request.method, path, body, fwd)
    if up.status_code >= 400:
        _flog(f"{request.method} {path} model={model} bytes={len(body)} mode={mode} -> "
              f"{up.status_code} ERR={up.text[:300]!r}")
    # record + auto cache successful responses for later replay (off never caches).
    if mode in ("record", "auto", "replay") and up.status_code < 400:
        _cache_put(key, up)
    return Response(content=up.content, status_code=up.status_code,
                    media_type=up.headers.get("content-type", "application/json"))


async def _mode(request):
    m = (await request.json()).get("mode")
    if m in ("record", "replay", "auto", "off"):
        STATE["mode"] = m
        if m == "record":
            CACHE.clear()
    return JSONResponse({"mode": STATE["mode"], "cached": len(CACHE)})


async def _health(request):
    return JSONResponse({"ok": True, "mode": STATE["mode"], "cached": len(CACHE),
                         "byte_aware": _BYTE_AWARE, "upstream": UPSTREAM})


app = Starlette(routes=[
    Route("/__mode", _mode, methods=["POST"]),
    Route("/__health", _health, methods=["GET"]),
    Route("/{path:path}", _proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("REPLAY_BIND", "127.0.0.1"),
                port=int(os.environ.get("REPLAY_PORT", "8900")), log_level="warning")
