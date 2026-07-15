"""Shared helpers across the per-language extractors."""

from __future__ import annotations


def module_qname_from_path(file_path: str, language: str) -> str:
    """Derive a module-style qualified name from a repo-relative file path.

    Examples (Python):
        services/ingestion/normalizer.py  -> engine.ingest.normalizer
        services/ingestion/__init__.py    -> engine.ingest

    Examples (TypeScript / JavaScript):
        src/lib/auth.ts                   -> src.lib.auth

    Examples (Go):
        internal/auth/middleware.go       -> internal.auth.middleware

    Examples (Java):
        src/main/java/Foo.java            -> src.main.java.Foo
    """
    path = file_path
    # Strip language-specific extensions and __init__ markers.
    if language == "python":
        if path.endswith("/__init__.py"):
            path = path[: -len("/__init__.py")]
        elif path.endswith("__init__.py"):
            path = path[: -len("__init__.py")].rstrip("/")
        elif path.endswith(".py"):
            path = path[: -len(".py")]
    else:
        # Non-Python: drop the trailing extension.
        dot = path.rfind(".")
        slash = path.rfind("/")
        if dot > slash:
            path = path[:dot]
    if not path:
        return ""
    return path.replace("/", ".")


__all__ = ["module_qname_from_path"]
