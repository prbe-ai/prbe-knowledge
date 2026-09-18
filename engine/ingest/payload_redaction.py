"""Scrub structured content before its first persistent write.

The stdlib pass is shared with clients. Gitleaks remains a fail-closed backstop
for its wider vendor corpus. Only redacted values are reconstructed; scanner
findings never become a log message or persisted metadata.
"""
from __future__ import annotations

import asyncio
from typing import Any

from engine.ingest._credential_redaction import default_scrub
from engine.ingest.secret_redaction import redact_documents


def redact_payload(value: Any) -> Any:
    clean = default_scrub(value)
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
