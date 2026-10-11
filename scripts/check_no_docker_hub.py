#!/usr/bin/env python3
"""Verify no image this repo builds, tests or deploys is pulled from Docker Hub.

Run:  python scripts/check_no_docker_hub.py
Exit: 0 none on Docker Hub, 1 one or more found (each printed).

Anonymous Docker Hub pulls from GitHub's runners failed every research-os image
build for an hour on 2026-10-09 (`429 Too Many Requests`, `auth.docker.io`
timeouts) while both status pages read green, and this repo has no Docker Hub
login either. Every reference names mirror.gcr.io, Google's read-through copy
of Docker Hub: the same digests, served without Hub's per-IP limits. A bare
`python:3.12-slim` IS a Docker Hub pull, so this fails on one rather than on
the next outage. Wired into the lint job beside check_index_contracts.py.

Scans tracked files only: Dockerfile `# syntax=` and FROM lines; literal
`image:` values in workflows, docker-compose, the Helm chart and k8s/; and the
image of every `docker run` in a workflow `run:` block.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIRROR = "mirror.gcr.io/"

#: Refs that are allowed to resolve to Docker Hub, each with its reason.
ALLOWED = {
    # docker-compose builds this image itself (`build:` beside `image:`).
    "prbe-knowledge:local": "built locally by docker-compose, never pulled",
    # MinIO withdrew its free images on 2026-09-25 (Docker Hub `pull access
    # denied`, not on mirror.gcr.io either); CI moved to Chainguard. The local
    # compose stack still names them and is broken until it moves too.
    "minio/minio:latest": "withdrawn upstream; compose needs a Chainguard rework",
    "minio/mc:latest": "withdrawn upstream; compose needs a Chainguard rework",
}

#: `docker run` options that take a separate value (`-e X=1`, `--name n`).
_RUN_VALUE_OPTS = {
    "-e", "--env", "--env-file", "-p", "--publish", "-v", "--volume", "--name",
    "--network", "--net", "--volumes-from", "--entrypoint", "-w", "--workdir",
    "-u", "--user", "--platform", "--add-host", "--mount", "-l", "--label",
    "--cpus", "-m", "--memory", "--shm-size", "--ipc", "--pid", "--hostname",
    "-h", "--restart", "--health-cmd", "--log-driver", "--tmpfs", "--ulimit",
    "--cap-add", "--cap-drop", "--device", "--gpus", "--runtime", "--security-opt",
}


def is_docker_hub(ref: str) -> bool:
    """Docker's own rule: a first path segment with no `.` or `:` (and not
    `localhost`) is not a registry host, so the ref resolves to Docker Hub."""
    first, sep, _ = ref.partition("/")
    if not sep:
        return True
    if first in {"docker.io", "index.docker.io", "registry-1.docker.io"}:
        return True
    return "." not in first and ":" not in first and first != "localhost"


def tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--", *patterns], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    return [REPO / line for line in out.splitlines() if line]


def dockerfile_refs() -> list[tuple[str, str]]:
    found = []
    for path in tracked("*Dockerfile*"):
        rel = str(path.relative_to(REPO))
        lines = path.read_text().splitlines()
        if lines and (m := re.match(r"#\s*syntax=(\S+)", lines[0], re.IGNORECASE)):
            found.append((f"{rel} (# syntax=)", m.group(1)))
        stages: set[str] = set()
        for line in lines:
            m = re.match(
                r"\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?", line, re.IGNORECASE
            )
            if not m:
                continue
            base, stage = m.group(1), m.group(2)
            if base != "scratch" and base not in stages and "$" not in base:
                found.append((rel, base))
            if stage:
                stages.add(stage)
    return found


def yaml_image_refs() -> list[tuple[str, str]]:
    found = []
    paths = tracked(".github/workflows/*.yml", ".github/workflows/*.yaml", "docker-compose*.yml",
                    "deploy/helm/*.yaml", "deploy/helm/**/*.yaml", "k8s/**/*.yaml")
    for path in paths:
        for line in path.read_text().splitlines():
            m = re.match(r"\s*(?:-\s+)?image:\s*[\"']?([^\"'\s#]+)", line)
            if m and not re.search(r"\{\{|\$", m.group(1)):
                found.append((str(path.relative_to(REPO)), m.group(1)))
    return found


def docker_run_image(command: str) -> str | None:
    try:
        tokens = shlex.split(command, comments=False)
    except ValueError:
        return None
    for i in range(len(tokens) - 1):
        if tokens[i] == "docker" and tokens[i + 1] == "run":
            j = i + 2
            while j < len(tokens):
                tok = tokens[j]
                if tok in _RUN_VALUE_OPTS:
                    j += 2
                elif tok.startswith("-"):
                    j += 1
                else:
                    return tok
    return None


def workflow_docker_run_refs() -> list[tuple[str, str]]:
    found = []
    for path in tracked(".github/workflows/*.yml", ".github/workflows/*.yaml"):
        logical, buf = [], ""
        for line in path.read_text().splitlines():
            if line.rstrip().endswith("\\"):
                buf += line.rstrip()[:-1] + " "
                continue
            logical.append(buf + line)
            buf = ""
        for command in logical:
            if "docker run" in command and (ref := docker_run_image(command.strip())):
                found.append((f"{path.relative_to(REPO)} (docker run)", ref))
    return found


def main() -> int:
    refs = dockerfile_refs() + yaml_image_refs() + workflow_docker_run_refs()
    seen = {ref for _, ref in refs}
    # Guard the guard: if a pattern stops matching, the check passes on nothing.
    expected = {
        f"{MIRROR}paradedb/paradedb:0.23.4-pg16",  # tests.yml `docker run`
        f"{MIRROR}pgvector/pgvector:pg16",  # tests.yml service + compose + chart
        f"{MIRROR}library/python:3.12-slim",  # services/*/Dockerfile
    }
    if missing := expected - seen:
        print(f"FAIL: scan no longer sees {sorted(missing)}; a pattern stopped matching")
        return 1
    offenders = [f"{where}: {ref}" for where, ref in refs
                 if ref not in ALLOWED and is_docker_hub(ref)]
    if offenders:
        print(f"FAIL: Docker Hub image references (prefix with {MIRROR}: same digest, "
              "no Hub rate limit):")
        for line in offenders:
            print(f"  {line}")
        return 1
    print(f"PASS: {len(refs)} image references, none on Docker Hub "
          f"({len(ALLOWED)} allowlisted with reasons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
