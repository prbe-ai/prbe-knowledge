"""Scrub structured content before its first persistent write.

The stdlib pass is shared with clients. Gitleaks remains a fail-closed backstop
for its wider vendor corpus. Only redacted values are reconstructed; scanner
findings never become a log message or persisted metadata.
"""
from __future__ import annotations

import asyncio
from typing import Any

from engine.ingest._credential_redaction import default_scrub
from engine.ingest._credential_secrets import redact as redact_spans
from engine.ingest.secret_redaction import redact_documents

#: What the vendored scrubber returns when it gives up a value WHOLE: a NUL, or
#: any `%XX`/`\uXXXX` escape, makes `scrub_string` inspect a decoded view of the
#: entire value and, if that view changes anywhere, return this for all of it.
#: Right for a config field; for a session transcript it is the whole session.
_WHOLE_VALUE = "<redacted>"

#: `scrub_string` refuses text encoded more than four levels deep (`%2525...`):
#: it cannot inspect it. Raised out of one scrub of a whole payload, that refusal
#: failed the entire ingest request with a 500 that clients retry forever, and a
#: session whose transcript held one such value stopped being captured at all.
_TOO_DEEP = "encoded content exceeds scrubber nesting limit"


def _scrub(value: Any, key: str = "") -> Any:
    """`default_scrub`, except a value too deeply encoded to inspect costs only itself.

    The normal path is one `default_scrub` call, unchanged. Only when that
    refuses does this walk the payload the same way `default_scrub` does (keys
    scrubbed, key context passed down, collisions suffixed) so the refusal lands
    on the one value it concerns: that value becomes the placeholder (it could
    not be proven clean), a multi-line value loses only the lines concerned, and
    everything else is scrubbed as usual. A container under a credential name
    never reaches here: `default_scrub` drops it whole before looking inside.
    """
    try:
        return default_scrub(value, key=key)
    except ValueError as exc:
        if _TOO_DEEP not in str(exc):
            raise
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        reserved = {str(item_key) for item_key in value}
        for item_key, item in value.items():
            original = str(item_key)
            clean_key = _scrub(original)
            candidate = clean_key
            suffix = 1
            while candidate in result or (clean_key != original and candidate in reserved):
                candidate = f"{clean_key}:{suffix}"
                suffix += 1
            result[candidate] = _scrub(item, original)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_scrub(item, key) for item in value]
    if isinstance(value, str) and "\n" in value:
        return "\n".join(_scrub(line, key) for line in value.split("\n"))
    return _WHOLE_VALUE


def redact_payload(value: Any) -> Any:
    clean = _scrub(value)
    texts: list[str] = []

    def collect(node):
        if isinstance(node, str):
            texts.append(node)
        elif isinstance(node, dict):
            for key, item in node.items():
                texts.append(key)
                collect(item)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(clean)
    redacted, _ = redact_documents(texts)
    replacements = dict(zip(texts, redacted, strict=True))

    def rebuild(node):
        if isinstance(node, str):
            return replacements[node]
        if isinstance(node, dict):
            result = {}
            reserved = set(node)
            for key, item in node.items():
                cleaned_key = replacements[key]
                candidate = cleaned_key
                suffix = 2
                while candidate in result or (candidate != key and candidate in reserved):
                    candidate = f"{cleaned_key}#{suffix}"
                    suffix += 1
                result[candidate] = rebuild(item)
            return result
        if isinstance(node, list):
            return [rebuild(item) for item in node]
        return node

    return rebuild(clean)


async def redact_payload_async(value: Any) -> Any:
    return await asyncio.to_thread(redact_payload, value)


def _scrub_free_text(text: str) -> str:
    """The stdlib scrub of one free-text value, never costing more than a line.

    A multi-line value the scrubber would replace WHOLE is scrubbed line by line
    instead, so a line whose own decoded view carries a credential is still
    replaced in full, and every other line keeps its content. The tap scanner
    runs over the entire text first, so rules that span lines (a PEM block) still
    see it whole. Values that do not collapse are untouched by this path.
    """
    clean = _scrub(text)
    if clean != _WHOLE_VALUE or text == _WHOLE_VALUE or "\n" not in text:
        return clean
    spans_redacted, _ = redact_spans(text)
    return "\n".join(_scrub(line) for line in spans_redacted.split("\n"))


def redact_texts(texts: list[str]) -> list[str]:
    """Scrub free-text bodies (a document body, pre-chunked pieces).

    Same two passes as `redact_payload` -- the stdlib scrubber, then the gitleaks
    backstop -- except a finding costs the line it sits on, never the body. The
    body is not a field: `redact_payload` once returned a 1.4 MB session as the
    placeholder alone, and the chunk diff then retired every chunk it had.
    """
    redacted, _ = redact_documents([_scrub_free_text(text) for text in texts])
    return redacted


async def redact_texts_async(texts: list[str]) -> list[str]:
    return await asyncio.to_thread(redact_texts, texts)
