"""Tests for the default class library shipped at services/kg/templates/.

The library is the seed material for Phase 0 onboarding (spec §5.7): each
template is *structural* (signature shape, edge types, generic playbook
stub) with no tenant-specific runbooks, owners, or file paths. These tests
guarantee:

1. Every JSON template under the directory loads as a valid ``BugClass``.
2. All 7 documented domains (auth, db, deploy, infra, network, observability,
   llm) ship at least one template.
3. Every cross-template ``[[wiki-link]]`` in every body resolves against the
   universe of all template ids — i.e. the library is internally consistent.
4. The templates directory is hygienic — no stray scratch files, only the
   loader, the package marker, domain subdirectories, and ``*.json`` files.
"""

from __future__ import annotations

from pathlib import Path

from services.kg.kg_check import check_class
from services.kg.schema import BugClass
from services.kg.templates._loader import TEMPLATES_DIR, load_all_templates


def test_every_template_loads_and_validates() -> None:
    templates = load_all_templates()
    assert len(templates) >= 23, f"expected >=23 templates, found {len(templates)}"
    for t in templates:
        assert isinstance(t, BugClass)


def test_all_seven_domains_present() -> None:
    templates = load_all_templates()
    by_id = {t.frontmatter.id for t in templates}
    expected_substrings = [
        "auth-",
        "db-",
        "deploy-",
        "pod-",
        "upstream-",
        "metrics-",
        "llm-",
    ]
    for sub in expected_substrings:
        assert any(
            sub in tid for tid in by_id
        ), f"missing template starting with {sub!r}"


def test_default_templates_pass_kg_check() -> None:
    templates = load_all_templates()
    universe = {t.frontmatter.id for t in templates}
    for t in templates:
        # Raises KgCheckError if any cross-template wiki-link is broken.
        check_class(t, universe=universe)


def test_no_extra_files_in_templates_dir() -> None:
    """Sanity: every entry under services/kg/templates/ is a directory,
    a domain ``*.json`` file, ``__init__.py``, ``_loader.py``, or a
    ``__pycache__`` directory. Catches commit hygiene regressions
    (stray ``.bak``, leftover scratch files, accidental README, etc.)."""
    allowed_files = {"__init__.py", "_loader.py"}
    for entry in Path(TEMPLATES_DIR).rglob("*"):
        if entry.is_dir():
            # Allow the package dir itself, domain dirs, and __pycache__.
            continue
        if entry.name in allowed_files:
            continue
        if entry.suffix == ".json":
            continue
        # Ignore compiled bytecode caches; they are filesystem-noise.
        if entry.suffix in {".pyc"} or "__pycache__" in entry.parts:
            continue
        raise AssertionError(f"unexpected file in templates dir: {entry}")
