"""Project a stored session event onto probe-events/1: keep what it names.

Only ever REMOVES. A key is dropped when the schema does not name it at its
level, or when its value is not what the schema allows there (the wrong type,
or a value outside a `const` / `enum`): Claude Code's own `origin` object
(which can carry another session's message) is not probe-events/1's
`origin: "user_shell"`. A content block with no schema variant keeps only its
type (`dropped`), an attachment only its type (unless it is a compact file
reference), a media block its media type and size. A kept value is never
rewritten or truncated.

The schema is closed at every level (`additionalProperties: false`), so "what
it allows" is exactly what tap 0.9.11 uploads.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).with_name("probe-events-1.schema.json")
_UNKNOWN_BLOCK = "unknown_block"
_MEDIA_TYPES = ("image", "audio", "video")
#: Returned by _value when the schema allows nothing here: the key is dropped.
_DROP = object()


@cache
def _schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _resolve(node: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        node = _schema()["$defs"][name]
    return node


@cache
def _block_def_id() -> int:
    return id(_resolve({"$ref": "#/$defs/block"}))


def _is_type(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "null":
        return value is None
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, list)
    return False


def _allowed(value: Any, node: dict[str, Any]) -> bool:
    """`value` fits the node's const / enum / type (no other keyword is checked)."""
    if "const" in node:
        return value == node["const"] and type(value) is type(node["const"])
    if "enum" in node:
        return value in node["enum"]
    json_type = node.get("type")
    if json_type is None:
        return True
    return any(_is_type(value, t) for t in (json_type if isinstance(json_type, list) else [json_type]))


def _type_matches(type_schema: dict[str, Any] | None, value: Any) -> bool:
    if not type_schema:
        return False
    if "const" in type_schema:
        return type_schema["const"] == value
    if "enum" in type_schema:
        return value in type_schema["enum"]
    return False


def _block(value: dict[str, Any]) -> Any:
    """One content block: the variant its `type` names, else its type alone."""
    block_type = value.get("type")
    variants = [_resolve(v) for v in _resolve({"$ref": "#/$defs/block"})["oneOf"]]
    for variant in variants:
        props = variant.get("properties") or {}
        if _type_matches(props.get("type"), block_type):
            if block_type in _MEDIA_TYPES and "source" in value:
                return _media(value)
            return _object(value, variant)
    if isinstance(block_type, str):
        # As tap 0.9.11 ships a Claude Code block it does not know: its type alone.
        return {"type": block_type, "dropped": True}
    return {"type": _UNKNOWN_BLOCK, "block_type": None}


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
        if id(node) == _block_def_id():
            return _block(value) if isinstance(value, dict) else _DROP
        for variant in node["oneOf"]:
            variant = _resolve(variant)
            if variant.get("type") == "array" and isinstance(value, list):
                return [_value(item, variant["items"]) for item in value]
            if variant.get("type") == "object" and isinstance(value, dict):
                return _object(value, variant)
            if variant.get("type") not in ("array", "object") and _allowed(value, variant):
                return value
        return _DROP
    if "properties" in node or node.get("type") == "object":
        return _object(value, node) if isinstance(value, dict) else _DROP
    if node.get("type") == "array":
        if not isinstance(value, list):
            return _DROP
        items = node.get("items")
        return [_value(item, items) for item in value] if items else value
    return value if _allowed(value, node) else _DROP


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
            kept = _value(item, props[key])
        elif isinstance(extra, dict):
            kept = _value(item, extra)
        else:
            continue
        if kept is not _DROP:
            out[key] = kept
    return out


def project_event(event: Any) -> Any:
    """`event` (one stored `raw`) reduced to what probe-events/1 allows."""
    if not isinstance(event, dict):
        return event
    out = _object(event, _schema())
    attachment = event.get("attachment")
    if isinstance(attachment, dict) and attachment.get("type") != "compact_file_reference":
        # Only a compact file reference ships (its name and path); any other
        # attachment held a file, a CLAUDE.md or a diff. Its type is kept.
        out["attachment"] = {"type": attachment["type"]} if isinstance(attachment.get("type"), str) else {}
    return out


def dropped_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    """Key paths `before` has and `after` does not (names only, list items as [])."""
    if isinstance(before, dict):
        if not isinstance(after, dict):
            return [prefix or "."]
        out: list[str] = []
        for key, value in before.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in after:
                out.append(path)
            else:
                out.extend(dropped_paths(value, after[key], path))
        return out
    if isinstance(before, list) and isinstance(after, list):
        out = []
        for old, new in zip(before, after, strict=False):
            if isinstance(old, dict) and isinstance(new, dict) and old.get("type") != new.get("type"):
                out.append(f"{prefix}[].{old.get('type')}")
            else:
                out.extend(dropped_paths(old, new, f"{prefix}[]"))
        return out
    return []
