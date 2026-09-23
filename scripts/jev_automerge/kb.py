"""Read-only access to a kb database for the auto-merge replay.

Every statement runs through the shell command in $KB_PSQL, which must start
psql against the kb database and read SQL on stdin, e.g. (managed plane):

    export KB_PSQL='kubectl --context <ctx> -n <pg-namespace> exec -i <pg-pod> \
        -c postgres -- psql -q -A -t -U postgres -d <kb-db>'

`-A -t` matters: every query here prints one JSON document per line.

Safety: each session opens with `SET default_transaction_read_only = on`, so
a write fails loudly. Queries that must see what production sees run under
`SET ROLE <app role>` + the tenant GUC, i.e. with the same row-level security.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

APP_ROLE = os.environ.get("KB_APP_ROLE", "probe_app")


def _cmd() -> list[str]:
    raw = os.environ.get("KB_PSQL")
    if not raw:
        raise SystemExit("set KB_PSQL to a psql command for the kb database (see kb.py)")
    return shlex.split(raw)


def run(sql: str, *, timeout_s: int = 120) -> str:
    prefix = f"SET default_transaction_read_only = on; SET statement_timeout = '{timeout_s}s';\n"
    out = subprocess.run(
        [*_cmd(), "-v", "ON_ERROR_STOP=1", "-f", "-"],
        input=prefix + sql,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def rows(sql: str, **kw: Any) -> list[dict]:
    """Run SQL whose output lines are JSON documents (`select row_to_json(t) ...`)."""
    return [json.loads(line) for line in run(sql, **kw).splitlines() if line.strip().startswith(("{", "["))]


def as_tenant(customer_id: str) -> str:
    """SQL prefix: become the app role under the tenant's RLS, output suppressed."""
    return f"SET ROLE {APP_ROLE}; SELECT set_config('app.current_customer_id', {lit(customer_id)}, false) \\g /dev/null\n"


def lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item, default=str) + "\n")
