"""The Jev call path -- its failure shapes, not its happy path.

Every test here drives `typesafe_sdk` through an httpx MockTransport, so the
replay's error handling is exercised without a key, a network, or a bill. The
SDK is a thin wrapper over httpx and accepts a transport, which is what makes
this possible at all.
"""

from __future__ import annotations

import json

import httpx2 as httpx  # the SDK is built on httpx2; a plain-httpx mock is rejected at config time
import pytest
from typesafe_sdk import (
    Noul,
    RetryPolicy,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
)


def _client(handler, **kw) -> TypeSafeClient:
    return TypeSafeClient(
        api_key="sk-test-not-a-real-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry=kw.pop("retry", RetryPolicy(max_retries=0)),
        **kw,
    )


def _answers(keys, value=0.8):
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "answers": {k: {"type": "noul", "noul": value} for k in keys},
    }


def test_a_noul_answer_comes_back_per_question():
    c = _client(lambda r: httpx.Response(200, json=_answers(["a", "b"])))
    resp = c.system_one(state={"q": 1}, questions={"a": Noul(instructions="x"),
                                                   "b": Noul(instructions="y")})
    assert {k: v.noul for k, v in resp.answers.items()} == {"a": 0.8, "b": 0.8}
    assert resp.usage.input_tokens == 100


def test_over_cap_is_a_typed_bad_request_carrying_the_reason():
    # The replay branches on this exact string to split a batch and retry, so
    # if the shape changes the batcher must fail loudly rather than give up.
    def handler(request):
        return httpx.Response(400, json={"detail": {"error_type": "max_tokens_exceeded"}})

    c = _client(handler)
    with pytest.raises(TypeSafeBadRequestError) as e:
        c.system_one(state={"q": 1}, questions={"a": Noul(instructions="x")})
    assert "max_tokens_exceeded" in str(e.value)


def test_a_bad_key_is_an_auth_error_not_a_retry_loop():
    c = _client(lambda r: httpx.Response(401, json={"detail": "bad key"}),
                retry=RetryPolicy(max_retries=3))
    with pytest.raises(TypeSafeAuthenticationError):
        c.system_one(state={"q": 1}, questions={"a": Noul(instructions="x")})


def test_a_timeout_raises_a_typed_error_with_a_nonempty_message():
    # httpx-family timeouts are the ones that stringify EMPTY, which is how a
    # replay ends up logging a blank reason for a whole tenant's traces.
    def handler(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    c = _client(handler)
    with pytest.raises(TypeSafeAPITimeoutError) as e:
        c.system_one(state={"q": 1}, questions={"a": Noul(instructions="x")})
    assert str(e.value).strip(), "a blank reason is unusable in a replay log"


def test_a_retryable_status_is_retried_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(200, json=_answers(["a"]))

    c = _client(handler, retry=RetryPolicy(max_retries=2, backoff_initial=0.0))
    resp = c.system_one(state={"q": 1}, questions={"a": Noul(instructions="x")})
    assert calls["n"] == 2 and resp.answers["a"].noul == 0.8


def test_a_missing_answer_is_absent_not_zero():
    # The scorer must not read "the server did not answer for this chunk" as
    # "this chunk is irrelevant" -- that silently deletes a document.
    c = _client(lambda r: httpx.Response(200, json=_answers(["a"])))
    resp = c.system_one(
        state={"q": 1},
        questions={"a": Noul(instructions="x"), "b": Noul(instructions="y")},
    )
    assert "b" not in resp.answers


def test_every_question_key_reaches_the_wire():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content)["questions"])
        return httpx.Response(200, json=_answers(seen))

    c = _client(handler)
    keys = [f"chunk::{i}#0" for i in range(40)]
    c.system_one(state={"q": 1}, questions={k: Noul(instructions=k) for k in keys})
    assert sorted(seen) == sorted(keys)
