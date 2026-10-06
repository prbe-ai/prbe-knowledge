"""Project a stored session event onto probe-events/1: keep what it names.

Only ever REMOVES: a key the schema does not name at its level, a content
block's body when the schema has no variant for it (kept as its type), an
attachment's payload (kept as its type), a media block's inline bytes (kept as
media type and size). It never adds or rewrites a value it keeps.

The schema is closed at every level (`additionalProperties: false`), so "what
it names" is exactly what tap 0.9.11 uploads.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).with_name("probe-events-1.schema.json")
_UNKNOWN_BLOCK = "unknown_block"
_MEDIA_TYPES = ("image", "audio", "video")


@cache
def _schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _resolve(node: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        node = _schema()["$defs"][name]
    return node


def _type_matches(type_schema: dict[str, Any] | None, value: Any) -> bool:
    if not type_schema:
        return False
    if "const" in type_schema:
        return type_schema["const"] == value
    if "enum" in type_schema:
        return value in type_schema["enum"]
    return False


def _block(value: dict[str, Any]) -> dict[str, Any]:
    """One content block: the variant its `type` names, else its type alone."""
    block_type = value.get("type")
    variants = [_resolve(v) for v in _resolve({"$ref": "#/$defs/block"})["oneOf"]]
    for variant in variants:
        props = variant.get("properties") or {}
        if _type_matches(props.get("type"), block_type):
            if block_type in _MEDIA_TYPES and "source" in value:
                return _media(value)
            return _object(value, variant)
    if "dropped" in value and isinstance(block_type, str):
        return {"type": block_type, "dropped": value["dropped"]}
    return {"type": _UNKNOWN_BLOCK, "block_type": block_type if isinstance(block_type, str) else None}


def _media(value: dict[str, Any]) -> dict[str, Any]:
    """A media block: an http(s) link is a pointer and stays; inline bytes go."""
    source = value.get("source")
    url = source.get("url") if isinstance(source, dict) and source.get("type") == "url" else None
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        return {"type": value["type"], "source": {"type": "url", "url": url}}
    out: dict[str, Any] = {"type": value["type"]}
    if isinstance(source, dict):
        if isinstance(source.get("media_type"), str):
            out["mimeType"] = source["media_type"]
        if isinstance(source.get("data"), str):
            out["bytes"] = len(source["data"])
    return out


def _value(value: Any, node: dict[str, Any]) -> Any:
    node = _resolve(node)
    if "oneOf" in node:
        if isinstance(value, dict) and node is _resolve({"$ref": "#/$defs/block"}):
            return _block(value)
        for variant in node["oneOf"]:
            variant = _resolve(variant)
            if variant.get("type") == "array" and isinstance(value, list):
                return [_value(item, variant["items"]) for item in value]
            if variant.get("type") == "object" and isinstance(value, dict):
                return _object(value, variant)
        return value
    if isinstance(value, dict) and ("properties" in node or node.get("type") == "object"):
        return _object(value, node)
    if isinstance(value, list) and "items" in node:
        return [_value(item, node["items"]) for item in value]
    return value


def _object(value: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    node = _resolve(node)
    props = node.get("properties")
    if props is None:
        # A free-form object the schema keeps whole (e.g. a media link's source).
        return value
    extra = node.get("additionalProperties")
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in props:
            out[key] = _value(item, props[key])
        elif isinstance(extra, dict):
            out[key] = _value(item, extra)
    return out


def project_event(event: Any) -> Any:
    """`event` (one stored `raw`) reduced to what probe-events/1 names."""
    if not isinstance(event, dict):
        return event
    schema = _schema()
    out = _object(event, schema)
    attachment = event.get("attachment")
    if isinstance(attachment, dict) and attachment.get("type") != "compact_file_reference":
        # Only a compact file reference ships (its name and path); any other
        # attachment held a file, a CLAUDE.md or a diff. Its type is kept.
        out["attachment"] = {"type": attachment["type"]} if isinstance(attachment.get("type"), str) else {}
    return out
