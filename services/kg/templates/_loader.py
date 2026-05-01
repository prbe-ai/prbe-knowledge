"""Load every JSON template under services/kg/templates/<domain>/ as a BugClass.

Templates are *structural* — signature shape, edge types, generic playbook
stub. No tenant-specific runbooks, owners, or file paths. They get applied
to a tenant during Phase 0 onboarding via the staff dashboard's template
picker (Task 24).
"""

from __future__ import annotations

import json
from pathlib import Path

from services.kg.schema import BugClass

TEMPLATES_DIR = Path(__file__).parent


def load_all_templates() -> list[BugClass]:
    """Load every ``*.json`` file under ``TEMPLATES_DIR`` as a ``BugClass``.

    Recursive walk picks up every domain subdirectory. Returns the list
    in deterministic (sorted) filesystem order.
    """
    out: list[BugClass] = []
    for path in sorted(TEMPLATES_DIR.rglob("*.json")):
        with path.open() as f:
            raw = json.load(f)
        out.append(BugClass.model_validate(raw))
    return out
