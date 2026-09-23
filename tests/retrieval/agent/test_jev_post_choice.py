"""`jev.post_choice`: one Choice question, as entity auto-merge asks it.

Driven through an httpx MockTransport: no key, no network. Each test gets a
fresh breaker so one test's failures cannot open another's.
"""

from __future__ import annotations

import json

import httpx
import pytest

from engine.retrieval.agent import jev
from engine.shared.constants import JEV_BREAKER_FAILURES, JEV_BREAKER_SECONDS

QUESTION = {
    "type": "choice",
    "instructions": "which one?",
    "criteria": {"c0": "the first", "c1": "the second", "none_of_these": "neither"},
}
STATE = {"new_entity": {"canonical_id": "ada@example.com"}, "candidates": {}}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(choice="c0", probs=None, model="jev-9.9.9"):
    body = {
        "model": model,
        "answers": {"match": {"type": "choice", "choice": choice,
                              "probabilities": probs if probs is not None else {"c0": 0.97, "c1": 0.02, "none_of_these": 0.01}}},
        "usage": {"input_tokens": 1234},
    }
    return lambda req: httpx.Response(200, json=body)


async def _ask(handler, breaker=None, api_key="k", model="jev-test"):
    async with _client(handler) as c:
        return await jev.post_choice(
            STATE, QUESTION, api_key=api_key, model=model, breaker=breaker or jev.Breaker(), client=c
        )


async def test_happy_path_returns_choice_probabilities_and_the_answering_model():
    ans = await _ask(_ok())
    assert ans.choice == "c0"
    assert ans.probabilities == {"c0": 0.97, "c1": 0.02, "none_of_these": 0.01}
    assert ans.model == "jev-9.9.9"  # what answered, for the audit row
    assert ans.input_tokens == 1234


async def test_sends_the_explicit_model_and_the_match_question_id():
    seen = {}

    def handler(req: httpx.Request):
        seen.update(json.loads(req.content))
        return _ok()(req)

    await _ask(handler, model="jev-pinned")
    assert seen["model"] == "jev-pinned"  # never the env-overridable JEV_MODEL
    assert list(seen["questions"]) == ["match"]
    assert seen["questions"]["match"] == QUESTION
    assert seen["state"] == STATE


async def test_missing_key_raises_without_a_request():
    with pytest.raises(jev.JevError, match="not configured"):
        await _ask(lambda req: pytest.fail("no request expected"), api_key="")


async def test_open_breaker_raises_breaker_open_and_sends_nothing():
    b = jev.Breaker()
    for _ in range(JEV_BREAKER_FAILURES):
        b.failure()
    with pytest.raises(jev.JevBreakerOpen):
        await _ask(lambda req: pytest.fail("no request expected"), breaker=b)


async def test_max_tokens_exceeded_is_too_large_and_does_not_trip_the_breaker():
    b = jev.Breaker()

    def handler(req):
        return httpx.Response(400, json={"detail": {"error_type": "max_tokens_exceeded"}})

    with pytest.raises(jev.JevRequestTooLarge):
        await _ask(handler, breaker=b)
    assert b.failures == 0


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_a_refused_request_is_rejected_as_permanent_and_does_not_trip_the_breaker(status):
    b = jev.Breaker()

    def handler(req):
        return httpx.Response(status, json={"detail": {"error_type": "validation_error"}})

    with pytest.raises(jev.JevRequestRejected):
        await _ask(handler, breaker=b)
    assert b.failures == 0


# 401/403: a revoked key. 404: a retired model or a wrong base URL. Each fails
# every request alike, so the node must wait (defer), not be dropped.
@pytest.mark.parametrize("status", [401, 402, 403, 404, 408, 429])
async def test_auth_routing_rate_limit_and_timeout_are_outages(status):
    b = jev.Breaker()

    def handler(req):
        return httpx.Response(status, json={"detail": {"error_type": "slow_down"}})

    with pytest.raises(jev.JevError) as err:
        await _ask(handler, breaker=b)
    assert not isinstance(err.value, jev.JevRequestRejected)
    assert b.failures == 1


async def test_server_error_trips_the_breaker_and_keeps_only_the_error_type():
    b = jev.Breaker()

    def handler(req):
        return httpx.Response(503, json={"detail": {"error_type": "overloaded", "msg": "tenant text"}})

    with pytest.raises(jev.JevError) as err:
        await _ask(handler, breaker=b)
    assert b.failures == 1
    assert "overloaded" in str(err.value) and "tenant text" not in str(err.value)


async def test_timeout_names_the_exception_class():
    def handler(req):
        raise httpx.ReadTimeout("")

    b = jev.Breaker()
    with pytest.raises(jev.JevError, match="ReadTimeout"):
        await _ask(handler, breaker=b)
    assert b.failures == 1


async def test_success_resets_the_breaker():
    b = jev.Breaker()
    b.failure()
    await _ask(_ok(), breaker=b)
    assert b.failures == 0


def _answer(choice, probs):
    return {"answers": {"match": {"choice": choice, "probabilities": probs}}}


@pytest.mark.parametrize(
    "body",
    [
        {"answers": {}},
        {"answers": {"match": {"choice": "c0"}}},
        _answer("c0", [0.9]),
        _answer("c7", {"c7": 0.99}),  # not a criteria key
        _answer(["c0"], {"c0": 0.9, "c1": 0.05, "none_of_these": 0.05}),  # unhashable choice
        _answer({"k": 1}, {"c0": 0.9, "c1": 0.05, "none_of_these": 0.05}),
        _answer("c0", {"c0": float("nan"), "c1": 0.0, "none_of_these": 0.0}),
        _answer("c0", {"c0": True, "c1": 0.0, "none_of_these": 0.0}),
        _answer("c0", {"c0": 1.5, "c1": 0.0, "none_of_these": 0.0}),
        _answer("c0", {"c0": 0.99}),  # missing alternatives
        _answer("c0", {"c0": 0.9, "c1": 0.05, "none_of_these": 0.05, "c9": 0.0}),  # an extra key
        _answer("c0", {"c0": 0.99, "c1": 0.99, "none_of_these": 0.99}),  # sums to ~3
        _answer("c0", {"c0": 0.30, "c1": 0.20, "none_of_these": 0.20}),  # sums to 0.7
        _answer("c1", {"c0": 0.90, "c1": 0.05, "none_of_these": 0.05}),  # choice is not the argmax
    ],
)
async def test_malformed_answers_raise_and_count_against_the_breaker(body):
    def handler(req):
        return httpx.Response(200, content=json.dumps(body, allow_nan=True))

    b = jev.Breaker()
    with pytest.raises(jev.JevError, match="malformed"):
        await _ask(handler, breaker=b)
    assert b.failures == 1  # a vendor schema regression must open the breaker


async def test_a_near_tie_within_tolerance_is_accepted():
    ans = await _ask(_ok(choice="c1", probs={"c0": 0.495, "c1": 0.49, "none_of_these": 0.015}))
    assert ans.choice == "c1"


async def test_seconds_until_closed():
    b = jev.Breaker()
    assert b.seconds_until_closed() == 0.0
    for _ in range(JEV_BREAKER_FAILURES):
        b.failure()
    assert 0.0 < b.seconds_until_closed() <= JEV_BREAKER_SECONDS


def test_merge_breaker_is_its_own_breaker():
    assert jev.MERGE_BREAKER is not jev.BREAKER
    assert jev.MERGE_BREAKER is not jev.EXTRACT_BREAKER
