"""Per-request record of which retrieval channels died.

A channel failure used to be invisible to the caller. Each of the four
handlers in `agent/tools.py` catches, logs a warning, and returns `[]`:

    except Exception as exc:
        log.warning("agent.search_bm25_failed", error=str(exc), query=q[:50])
        return []

The fusion layer downstream cannot tell "this channel found nothing" from
"this channel died", so the response reported a clean success. Measured on
the managed data plane over one 16h window:

    102 retrievals lost >= 1 channel to a 30s statement timeout
     24 of those returned status='ok', degraded=False to the caller

That is ~10% of all traffic reporting completeness it did not have, and it
is why the #444 BM25 regression ran for three days without anyone noticing:
the system was built to hide exactly this.

Why a ContextVar rather than threading a parameter: the failure happens
inside nested closures in `search()`, several frames below anything that
holds request state, and the same channels are also called from the turn-0
pre-fan-out path. Passing an accumulator through every signature is the
"sprawling diff" `engine/shared/litellm_key.py` already rejected for the
same reason -- follow the precedent that exists.

The ContextVar holds a MUTABLE set, and that is load-bearing. `asyncio.gather`
gives each child task a COPY of the context, so rebinding the var inside a
channel coroutine would be invisible to the parent. Mutating a shared set
through the copied reference is visible, which is what makes this work under
the 4-way (and 5-subquery) fan-out.
"""

from __future__ import annotations

from contextvars import ContextVar

# Channel names match the keys the fan-out emits ("vector", "bm25", "graph",
# "inferred_edge") so a reason string is readable without a lookup table.
_lost_channels: ContextVar[set[str] | None] = ContextVar(
    "retrieval_lost_channels", default=None
)


def begin_request() -> None:
    """Install a fresh accumulator for this request.

    Call once per retrieval, before any channel runs. Rebinding here (rather
    than at import) is what keeps one request's losses out of the next one on
    a reused worker task.
    """
    _lost_channels.set(set())


def record_channel_loss(channel: str) -> None:
    """Note that `channel` failed and returned no rows.

    Safe to call when no accumulator is installed (unit tests calling a
    retriever directly, list-pipeline callers that never begin_request): the
    loss is simply not recorded rather than raising into an except handler
    that is already handling a failure.
    """
    lost = _lost_channels.get()
    if lost is not None:
        lost.add(channel)


def lost_channels() -> frozenset[str]:
    """Channels that failed during this request, if any."""
    lost = _lost_channels.get()
    return frozenset(lost) if lost else frozenset()
