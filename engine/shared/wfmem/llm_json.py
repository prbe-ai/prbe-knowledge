"""Reading a model's reply: get the text out, then parse the JSON it wrapped.

SHARED BY `classifier` AND `structuring`, deliberately, rather than copied into
each. Two callers is normally too few to justify extracting a helper -- but what
is being shared is a list of the ways a model WRAPS a correct answer (code
fences, a chatty preamble, a response object whose shape shifts with the client
library), and that list only ever grows. Two copies drift the moment somebody
meets a sixth wrapping and fixes it in whichever module they happened to be in,
and the symptom on the other side is not a crash: it is the parse failing, the
response being declared unusable, and a feature quietly getting worse at exactly
the inputs somebody just fixed elsewhere.

Scoped to `engine.shared.wfmem` rather than promoted to `engine.shared.llm`:
that module imports `litellm` at module scope, and both callers go out of their
way to keep that import off the path of everything that merely imports them.

STRICT ABOUT CONTENT, TOLERANT ABOUT PACKAGING. Nothing here repairs a value or
guesses at a missing field -- the caller validates what comes back, because only
the caller knows which of its fields have an honest default.
"""

from __future__ import annotations

import json
from typing import Any


def response_text(response: Any) -> str | None:
    """`.choices[0].message.content` off a LiteLLM response, defensively.

    Every step of that walk is a shape assumption about somebody else's library,
    so a miss returns None -- "the model said nothing usable" -- rather than an
    `AttributeError` surfacing from inside a caller's error handling.
    """
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None


def loads_forgiving(raw: str) -> Any:
    """`json.loads`, retried on the outermost `{...}` if the whole string fails.

    Returns `None` when nothing usable is in there. That is indistinguishable
    from a literal `null` response, and deliberately so: neither is an answer,
    and giving the caller two ways to spell "no" would just move the check.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


__all__ = ["loads_forgiving", "response_text"]
