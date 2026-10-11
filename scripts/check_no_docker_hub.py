#!/usr/bin/env python3
"""Fail when anything this repo builds, tests or deploys pulls an image from Docker Hub.

Run:  python3 scripts/check_no_docker_hub.py            (scan the repo)
      python3 scripts/check_no_docker_hub.py --self-test (the scanner's own cases)
Exit: 0 clean, 1 one or more Docker Hub references (each printed with its file).

Anonymous Docker Hub pulls from GitHub's runners failed every research-os image
build for an hour on 2026-10-09 (`429 Too Many Requests`, `auth.docker.io`
timeouts) while both status pages read green. Every reference names
mirror.gcr.io instead, Google's read-through copy of Docker Hub: the same
digests, without Hub's per-IP limits. A bare `python:3.12-slim` IS a Docker Hub
pull, so this fails on one rather than on the next outage.

Scanned (tracked files only):
  - Dockerfiles (`Dockerfile*`, `*.Dockerfile`, `Containerfile*`): the
    `# syntax=` parser directive, FROM (ARG defaults substituted), and
    `COPY --from=` / `RUN --mount=...,from=` images. Heredoc bodies are skipped.
  - YAML (workflows, compose, Helm, manifests): `image:` scalars and Helm
    `image: {repository:}` maps, `uses: docker://`, scalar `container:`,
    Buildx `driver-opts: image=`; every setup-buildx-action step must name its
    BuildKit image and every setup-qemu-action step its binfmt image.
  - Shell (workflow `run:` blocks, `*.sh`, Makefiles): images of
    `docker run|create|pull` and `docker container run|create`,
    `docker image pull`, and `*IMAGE*` variable defaults (`${X_IMAGE:-ref}`,
    `X_IMAGE=ref`).
Not scanned: images named inside Python or other source (testcontainers and
the like) -- keep those on the mirror by review.

Unresolvable refs (a `$` left in the repository part) fail closed: the scanner
cannot prove where they pull from.

This file is the same in every prbe-ai repo that carries it; only CONFIG below
differs.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
import sys
from pathlib import Path

MIRROR = "mirror.gcr.io/"

# ---------------------------------------------------------------------------
# CONFIG (per repo)
# ---------------------------------------------------------------------------
CONFIG: dict = {
    # git pathspecs of YAML to scan for images (workflows are always scanned).
    "yaml": ["*compose*.yml", "*compose*.yaml", "deploy/helm/*.yaml", "deploy/helm/*.yml",
             "k8s/*.yaml", "k8s/*.yml"],
    # (path glob, ref) -> reason. Refs allowed to resolve to Docker Hub.
    "allowed": {
        ("docker-compose.yml", "prbe-knowledge:local"):
            "built by docker-compose itself (`build:` beside `image:`), never pulled",
        # MinIO withdrew its free images on 2026-09-25: Docker Hub answers `pull
        # access denied` and mirror.gcr.io has nothing either. CI moved to
        # Chainguard; the local compose stack is broken until it moves too.
        ("docker-compose.yml", "minio/minio:latest"): "withdrawn upstream; compose needs a Chainguard rework",
        ("docker-compose.yml", "minio/mc:latest"): "withdrawn upstream; compose needs a Chainguard rework",
    },
    # (scan kind, path glob, ref prefix): each must match at least one finding,
    # so a scanner or glob that silently stops matching fails the check.
    "sentinels": [
        ("docker", ".github/workflows/tests.yml", "mirror.gcr.io/paradedb/paradedb:"),
        ("image", ".github/workflows/tests.yml", "mirror.gcr.io/pgvector/pgvector:"),
        ("image", "docker-compose.yml", "mirror.gcr.io/pgvector/pgvector:"),
        ("image", "deploy/helm/values.yaml", "mirror.gcr.io/pgvector/pgvector:"),
        ("from", "services/worker/Dockerfile", "mirror.gcr.io/library/python:"),
        ("from", "services/worker/Dockerfile", "mirror.gcr.io/library/golang:"),
        ("from", "sandbox/Dockerfile.product", "mirror.gcr.io/pgvector/pgvector:"),
        ("copy-from", "sandbox/Dockerfile.product", "ghcr.io/astral-sh/uv:"),
    ],
}
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent

_DOCKERFILE_GLOBS = ("Dockerfile*", "*.Dockerfile", "*.dockerfile", "Containerfile*")
# From `docker run --help`: every option that takes a separate value.
_RUN_VALUE_OPTS = frozenset(
    "--add-host --annotation --attach --blkio-weight --blkio-weight-device --cap-add "  # noqa: SIM905
    "--cap-drop --cgroup-parent --cgroupns --cidfile --cpu-period --cpu-quota "
    "--cpu-rt-period --cpu-rt-runtime --cpu-shares --cpus --cpuset-cpus --cpuset-mems "
    "--detach-keys --device --device-cgroup-rule --device-read-bps --device-read-iops "
    "--device-write-bps --device-write-iops --dns --dns-option --dns-search --domainname "
    "--entrypoint --env --env-file --expose --gpus --group-add --health-cmd "
    "--health-interval --health-retries --health-start-interval --health-start-period "
    "--health-timeout --hostname --ip --ip6 --ipc --isolation --label --label-file --link "
    "--link-local-ip --log-driver --log-opt --mac-address --memory --memory-reservation "
    "--memory-swap --memory-swappiness --mount --name --network --network-alias "
    "--oom-score-adj --pid --pids-limit --platform --publish --pull --restart --runtime "
    "--security-opt --shm-size --stop-signal --stop-timeout --storage-opt --sysctl "
    "--tmpfs --ulimit --user --userns --uts --volume --volume-driver --volumes-from "
    "--workdir --net -a -c -e -h -l -m -p -u -v -w".split()
)
_GHA_EXPR = re.compile(r"\$\{\{.*?\}\}")
_UNRESOLVED = "<unresolved>"


def is_docker_hub(ref: str) -> bool:
    """Docker's own rule: a first path segment with no `.` or `:` (and not
    `localhost`) is not a registry host, so the ref resolves to Docker Hub."""
    first, sep, _ = ref.partition("/")
    if not sep:
        return True
    if first in {"docker.io", "index.docker.io", "registry-1.docker.io", "registry.hub.docker.com"}:
        return True
    return "." not in first and ":" not in first and first != "localhost"


def repository(ref: str) -> str:
    """`host/name:tag@sha256:x` -> `host/name` (a `:port` in the host is kept)."""
    ref = ref.split("@", 1)[0]
    head, sep, tail = ref.rpartition(":")
    return head if sep and "/" not in tail else ref


def classify(ref: str) -> str | None:
    """None when `ref` is fine, else why it is not."""
    repo = repository(ref)
    if repo == _UNRESOLVED or re.fullmatch(r"\$\{?\w+\}?", repo):
        return None  # wholly a variable (a CI matrix value, a script argument): not checkable here
    if "$" in repo or _UNRESOLVED in repo:
        return "unresolvable (variable in the repository part)"
    if is_docker_hub(repo):
        return "Docker Hub"
    return None


# --- Dockerfiles ------------------------------------------------------------

_ARG_REF = re.compile(r"\$\{(\w+)(?::?[-+]([^}]*))?\}|\$(\w+)")


def _substitute(value: str, args: dict[str, str]) -> str:
    def one(m: re.Match) -> str:
        name = m.group(1) or m.group(3)
        if name in args:
            return args[name]
        if m.group(2) is not None:
            return m.group(2)
        return "$" + name

    return _ARG_REF.sub(one, value)


def dockerfile_refs(text: str) -> list[tuple[str, str]]:
    """(kind, ref) for every image a Dockerfile pulls."""
    lines = text.splitlines()
    found: list[tuple[str, str]] = []
    escape = "\\"
    i = 0
    # Parser directives: leading `# key = value` lines, any order, any case.
    while i < len(lines):
        m = re.match(r"#\s*([a-zA-Z]+)\s*=\s*(\S+)", lines[i])
        if not m:
            break
        key, value = m.group(1).lower(), m.group(2)
        if key == "syntax":
            found.append(("syntax", value))
        elif key == "escape":
            escape = value
        i += 1

    logical: list[str] = []
    buf = ""
    heredoc: str | None = None
    for line in lines[i:]:
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
            continue
        if not buf and re.match(r"\s*#", line):
            continue
        stripped = line.rstrip()
        if stripped.endswith(escape):
            buf += stripped[: -len(escape)] + " "
            continue
        full = buf + line
        buf = ""
        logical.append(full)
        m = re.search(r"<<-?\s*[\"']?(\w+)[\"']?", full)
        if m and re.match(r"\s*(RUN|COPY|ADD)\b", full, re.IGNORECASE):
            heredoc = m.group(1)
    if buf:
        logical.append(buf)

    global_args: dict[str, str] = {}
    stage_args: dict[str, str] = {}
    stages: set[str] = set()
    seen_from = False
    for line in logical:
        m = re.match(r"\s*(\w+)\s*(.*)", line)
        if not m:
            continue
        instr, rest = m.group(1).upper(), m.group(2)
        if instr == "ARG":
            for token in shlex.split(rest, posix=True) if rest else []:
                name, eq, value = token.partition("=")
                target = stage_args if seen_from else global_args
                if eq:
                    target[name] = _substitute(value, {**global_args, **target})
                elif seen_from and name in global_args:
                    stage_args[name] = global_args[name]
        elif instr == "FROM":
            seen_from = True
            stage_args = {}
            words = [w for w in rest.split() if not w.startswith("--")]
            if not words:
                continue
            base = _substitute(words[0], global_args)
            if len(words) >= 3 and words[1].upper() == "AS":
                stages.add(words[2].lower())
            if base.lower() in stages or base == "scratch":
                continue
            found.append(("from", base))
        elif instr in {"COPY", "ADD", "RUN"}:
            args = {**global_args, **stage_args}
            values = []
            for flag in re.finditer(r"(?:^|\s)--(from|mount)=(\S+)", rest):
                if flag.group(1) == "from":
                    values.append(flag.group(2))
                else:
                    values += [kv[5:] for kv in flag.group(2).split(",") if kv.startswith("from=")]
            for raw in values:
                value = _substitute(raw.strip("\"'"), args)
                if value.lower() in stages or value.isdigit():
                    continue
                found.append(("copy-from", value))
    return found


# --- Shell ------------------------------------------------------------------


def shell_commands(text: str) -> list[str]:
    """Logical shell lines: continuations joined, GHA `${{ }}` neutralised."""
    out, buf = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        out.append(_GHA_EXPR.sub(_UNRESOLVED, buf + line))
        buf = ""
    if buf:
        out.append(_GHA_EXPR.sub(_UNRESOLVED, buf))
    return out


def _image_after(tokens: list[str], start: int) -> str | None:
    j = start
    while j < len(tokens):
        tok = tokens[j]
        if tok in _RUN_VALUE_OPTS:
            j += 2
        elif tok.startswith("-"):
            j += 1
        else:
            return tok
    return None


def shell_refs(command: str) -> list[tuple[str, str]]:
    """(kind, ref) for docker pulls and *IMAGE* defaults on one logical line."""
    found: list[tuple[str, str]] = []
    # Only variables NAMED as images (`*IMAGE`): `${IMAGE_KIND:-legacy}` is not one.
    for m in re.finditer(r"\$\{(\w*IMAGE):?[-=]([^}]+)\}", command, re.IGNORECASE):
        value = m.group(2).strip("\"'")
        if value and not value.startswith("$"):
            found.append(("default", value))
    for m in re.finditer(
        r"(?:^|[\s;&|(])(?:export\s+|local\s+|readonly\s+)?(\w*IMAGE)=([\"']?)([^\s\"'$;]+)\2(?=[\s;]|$)",
        command,
        re.IGNORECASE,
    ):
        value = m.group(3)
        if ":" in value or "/" in value:
            found.append(("default", value))
    if "docker" not in command:
        return found
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        return [*found, ("unparseable", command.strip())]
    for i, tok in enumerate(tokens):
        if tok != "docker" or i + 1 >= len(tokens):
            continue
        verb = tokens[i + 1]
        start = None
        if verb in {"run", "create", "pull"}:
            start = i + 2
        elif i + 2 < len(tokens) and (verb, tokens[i + 2]) in {
            ("container", "run"), ("container", "create"), ("image", "pull")
        }:
            start = i + 3
        if start is None:
            continue
        image = _image_after(tokens, start)
        if image and not image.startswith("$") and image not in ";&|()":
            found.append(("docker", image))
    return found


# --- YAML -------------------------------------------------------------------


def _strip_comment(line: str) -> str | None:
    """None for a comment-only line, else the line without a trailing ` # x`."""
    if re.match(r"\s*#", line):
        return None
    return re.sub(r"\s+#.*$", "", line)


def yaml_refs(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        line = _strip_comment(raw)
        if line is None:
            continue
        line = _GHA_EXPR.sub(_UNRESOLVED, line)
        m = re.match(r"\s*(?:-\s+)?image:\s*[\"']?([^\"'\s]*)[\"']?\s*$", line)
        if m:
            value = m.group(1)
            if value and "{{" not in value:
                found.append(("image", _substitute(_GHA_EXPR.sub(_UNRESOLVED, value), {})))
            elif not value:
                indent = len(raw) - len(raw.lstrip())
                fields: dict[str, str] = {}
                for nxt in lines[idx + 1 :]:
                    if not nxt.strip() or re.match(r"\s*#", nxt):
                        continue
                    if len(nxt) - len(nxt.lstrip()) <= indent:
                        break
                    m2 = re.match(r"\s*(registry|repository):\s*[\"']?([^\"'\s#]+)", nxt)
                    if m2:
                        fields[m2.group(1)] = m2.group(2)
                repo = fields.get("repository")
                if repo and "{{" not in repo:
                    reg = fields.get("registry")
                    found.append(("image", f"{reg}/{repo}" if reg and "{{" not in reg else repo))
        for pattern, kind in (
            (r"\buses:\s*[\"']?docker://([^\"'\s]+)", "uses"),
            (r"^\s*container:\s*[\"']?([^\"'\s{]+)[\"']?\s*$", "container"),
            (r"\bimage=([^\s,\"']+)", "buildkit"),
        ):
            for m3 in re.finditer(pattern, line):
                found.append((kind, _GHA_EXPR.sub(_UNRESOLVED, m3.group(1))))
    return found


def action_steps_missing_mirror(text: str) -> list[str]:
    """setup-buildx / setup-qemu steps that would pull their default Hub image."""
    problems = []
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        line = _strip_comment(raw)
        if line is None:
            continue
        m = re.search(r"uses:\s*docker/(setup-buildx-action|setup-qemu-action)@", line)
        if not m:
            continue
        step_indent = len(raw) - len(raw.lstrip())
        if raw.lstrip().startswith("uses:") and idx and lines[idx - 1].lstrip().startswith("-"):
            step_indent = len(lines[idx - 1]) - len(lines[idx - 1].lstrip())
        want = (f"image={MIRROR}moby/buildkit:" if m.group(1) == "setup-buildx-action"
                else f"image: {MIRROR}tonistiigi/binfmt")
        ok = False
        for nxt in lines[idx + 1 :]:
            if not nxt.strip():
                continue
            if len(nxt) - len(nxt.lstrip()) <= step_indent and nxt.lstrip().startswith("-"):
                break
            if len(nxt) - len(nxt.lstrip()) < step_indent:
                break
            body = _strip_comment(nxt)
            if body and want in body.replace('"', "").replace("'", ""):
                ok = True
            if body and m.group(1) == "setup-buildx-action" and re.search(r"\bdriver:\s*docker\s*$", body):
                ok = True  # the docker driver uses the daemon's builder: no BuildKit image pulled
        if not ok:
            problems.append(f"docker/{m.group(1)} without `{want}...`")
    return problems


def workflow_run_blocks(text: str) -> list[str]:
    """Bodies of `run:` steps (inline and block scalars)."""
    blocks, lines = [], text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"(\s*)(?:-\s+)?run:\s*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        indent, rest = len(m.group(1)), m.group(2).strip()
        if rest and rest[0] not in "|>":
            blocks.append(rest)
            i += 1
            continue
        body, folded = [], rest.startswith(">")
        i += 1
        while i < len(lines) and (not lines[i].strip()
                                  or len(lines[i]) - len(lines[i].lstrip()) > indent):
            body.append(lines[i].strip())
            i += 1
        blocks.append(" ".join(body) if folded else "\n".join(body))
    return blocks


# --- Repo scan --------------------------------------------------------------


def tracked(*patterns: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *patterns], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    return sorted({p for p in out.split("\0") if p})


def scan(read=lambda p: (REPO / p).read_text(errors="replace")) -> list[tuple[str, str, str]]:
    """(path, kind, ref) for every image reference, plus action-step problems."""
    results: list[tuple[str, str, str]] = []
    all_files = tracked()
    dockerfiles = [p for p in all_files
                   if any(fnmatch.fnmatch(Path(p).name, g) for g in _DOCKERFILE_GLOBS)
                   and not p.endswith(".dockerignore")]
    for path in dockerfiles:
        for kind, ref in dockerfile_refs(read(path)):
            results.append((path, kind, ref))
    workflows = tracked(".github/workflows/*.yml", ".github/workflows/*.yaml")
    yaml_files = sorted(set(workflows) | set(tracked(*CONFIG["yaml"])) if CONFIG["yaml"]
                        else set(workflows))
    for path in yaml_files:
        text = read(path)
        for kind, ref in yaml_refs(text):
            results.append((path, kind, ref))
        for problem in action_steps_missing_mirror(text):
            results.append((path, "action", problem))
    for path in workflows:
        for block in workflow_run_blocks(read(path)):
            for command in shell_commands(block):
                for kind, ref in shell_refs(command):
                    results.append((path, kind, ref))
    shell_files = [p for p in all_files
                   if p.endswith(".sh") or Path(p).name in {"Makefile", "entrypoint.sh"}
                   or (Path(p).name.startswith("entrypoint") and "." not in Path(p).name)]
    for path in shell_files:
        for command in shell_commands(read(path)):
            for kind, ref in shell_refs(command):
                results.append((path, kind, ref))
    return results


def _allowed(path: str, ref: str) -> str | None:
    for (glob, allowed_ref), reason in CONFIG["allowed"].items():
        if allowed_ref == ref and fnmatch.fnmatch(path, glob):
            return reason
    return None


def main() -> int:
    results = scan()
    offenders, allowed = [], []
    for path, kind, ref in results:
        if kind == "action":
            offenders.append(f"{path}: {ref}")
            continue
        if kind == "unparseable":
            offenders.append(f"{path}: cannot parse this shell line, so cannot vouch for it: {ref}")
            continue
        why = classify(ref)
        if why is None:
            continue
        reason = _allowed(path, ref)
        if reason:
            allowed.append(f"{path}: {ref} -- {reason}")
        else:
            offenders.append(f"{path} ({kind}): {ref} [{why}]")
    missing = [
        f"{kind} in {glob} starting {prefix}"
        for kind, glob, prefix in CONFIG["sentinels"]
        if not any(k == kind and fnmatch.fnmatch(p, glob) and r.startswith(prefix)
                   for p, k, r in results)
    ]
    if missing:
        print("FAIL: the scan no longer sees references it must see, so a pattern or path stopped "
              "matching (or an image moved: update CONFIG['sentinels']):")
        for line in missing:
            print(f"  {line}")
        return 1
    if offenders:
        print(f"FAIL: image references that pull from Docker Hub (prefix with {MIRROR}: same "
              "digest, no Hub rate limit):")
        for line in offenders:
            print(f"  {line}")
        return 1
    print(f"PASS: {len(results)} image references, none on Docker Hub")
    for line in allowed:
        print(f"  allowed: {line}")
    return 0


# --- Self-test --------------------------------------------------------------

_HUB_CASES = [
    ("python:3.12-slim", True), ("library/python", True), ("docker.io/library/python", True),
    ("prom/node-exporter:latest", True), ("mirror.gcr.io/library/python:3.12-slim", False),
    ("ghcr.io/a/b:1", False), ("localhost/x:1", False), ("localhost:5000/x:1", False),
]


def self_test() -> int:
    failures: list[str] = []

    def expect(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    for ref, hub in _HUB_CASES:
        expect(f"is_docker_hub({ref})", is_docker_hub(ref), hub)
    expect("repository with port", repository("localhost:5000/x:1"), "localhost:5000/x")
    expect("repository with digest", repository("a.io/b:1@sha256:ff"), "a.io/b")

    def from_refs(text: str) -> list[str]:
        return [r for _, r in dockerfile_refs(text)]

    df = dockerfile_refs
    expect("syntax with spaces, after escape", df("# escape=`\n# syntax = docker/dockerfile:1\nFROM a.io/b\n"),
           [("syntax", "docker/dockerfile:1"), ("from", "a.io/b")])
    expect("lowercase from + stage", from_refs("from python:3.12 as base\nFROM base\n"), ["python:3.12"])
    expect("ARG default substituted", from_refs("ARG B=python:3.12\nFROM ${B} AS x\n"), ["python:3.12"])
    expect("ARG tag keeps repo", from_refs("ARG V=3.12\nFROM python:${V}-slim\n"), ["python:3.12-slim"])
    expect("ARG without default: wholly variable, skipped", classify(from_refs("ARG B\nFROM ${B}\n")[0]), None)
    expect("variable registry fails closed", classify("$REG/python:3.12") is not None, True)
    expect("Hub alias", is_docker_hub("registry.hub.docker.com/library/python"), True)
    expect("from= inside a RUN command is not an image",
           from_refs('FROM a.io/b\nRUN python -c "f(from=1)" \\\n && echo from pathlib\n'), ["a.io/b"])
    expect("COPY --from image", from_refs("FROM a.io/b AS s\nCOPY --from=busybox:1 /x /y\nCOPY --from=s /a /b\n"),
           ["a.io/b", "busybox:1"])
    expect("RUN --mount from", from_refs("FROM a.io/b\nRUN --mount=type=bind,from=alpine:3,target=/a true\n"),
           ["a.io/b", "alpine:3"])
    expect("heredoc body skipped", from_refs("FROM a.io/b\nRUN <<EOF\nfrom pathlib import Path\nEOF\n"), ["a.io/b"])
    expect("continuation", from_refs("FROM \\\n  python:3.12\n"), ["python:3.12"])
    expect("platform flag", from_refs("FROM --platform=$BUILDPLATFORM golang:1 AS g\n"), ["golang:1"])

    def docker(cmd: str) -> list[str]:
        return [r for k, r in shell_refs(cmd) if k == "docker"]

    expect("run with value opts", docker("docker run -d --name p -e A=1 --health-interval 5s python:3.12 x"),
           ["python:3.12"])
    expect("trailing comment with apostrophe", docker("docker run --rm python:3.12 true  # don't cache"),
           ["python:3.12"])
    expect("two commands", docker("docker run --rm a.io/x true && docker run --rm python:3.12 true"),
           ["a.io/x", "python:3.12"])
    expect("pull/create/container run",
           docker("docker pull alpine:3; docker create redis:7; docker container run busybox"),
           ["alpine:3", "redis:7", "busybox"])
    expect("GHA expr in a volume", docker(shell_commands("docker run -v ${{ github.workspace }}:/w a.io/x true")[0]),
           ["a.io/x"])
    expect("variable image skipped", docker('docker run --rm "$IMAGE" true'), [])
    expect("unbalanced quote is unparseable", [k for k, _ in shell_refs("docker run 'x")], ["unparseable"])
    expect("image defaults",
           [r for k, r in shell_refs('POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:16}"; : "${PG16_IMAGE:=redis:7}"')
            if k == "default"], ["postgres:16", "redis:7"])
    expect("IMAGE_KIND is not an image", shell_refs('echo "${IMAGE_KIND:-legacy}"'), [])
    expect("plain image assignment", [r for k, r in shell_refs("  local BASE_IMAGE=python:3.12") if k == "default"],
           ["python:3.12"])
    expect("yaml forms", sorted(r for _, r in yaml_refs(
        "x:\n  image: python:3.12 # c\n  # image: ignored:1\n  container: node:20\n"
        "  - uses: docker://alpine:3.19\n  img:\n    image:\n      repository: pgvector/pgvector\n"
        "      tag: pg16\n  driver-opts: image=moby/buildkit:x\n  y: {{ .Values.image }}\n")),
        sorted(["python:3.12", "node:20", "alpine:3.19", "pgvector/pgvector", "moby/buildkit:x"]))
    expect("buildx step without driver-opts",
           len(action_steps_missing_mirror("steps:\n  - uses: docker/setup-buildx-action@v3\n  - run: x\n")), 1)
    expect("buildx step with commented driver-opts",
           len(action_steps_missing_mirror(
               "steps:\n  - uses: docker/setup-buildx-action@v3\n    with:\n"
               "      # driver-opts: image=mirror.gcr.io/moby/buildkit:buildx-stable-1\n")), 1)
    expect("buildx multi-line driver-opts",
           action_steps_missing_mirror(
               "steps:\n  - uses: docker/setup-buildx-action@v3\n    with:\n      driver-opts: |\n"
               "        image=mirror.gcr.io/moby/buildkit:buildx-stable-1\n  - run: x\n"), [])
    expect("buildx docker driver",
           action_steps_missing_mirror("steps:\n  - uses: docker/setup-buildx-action@v3\n    with:\n"
                                       "      driver: docker\n"), [])
    expect("compose default substituted", [r for _, r in yaml_refs("  image: ${PG_IMAGE:-postgres:16}\n")],
           ["postgres:16"])
    expect("matrix image key skipped", [classify(r) for _, r in yaml_refs(
        "        image: ${{ fromJSON(needs.x.outputs.images) }}\n")], [None])
    expect("double space in docker run", docker("docker  run --rm python:3.12 true"), ["python:3.12"])
    expect("buildx step on the mirror",
           action_steps_missing_mirror(
               "steps:\n  - name: b\n    uses: docker/setup-buildx-action@v3\n    with:\n"
               "      driver-opts: image=mirror.gcr.io/moby/buildkit:buildx-stable-1\n  - run: x\n"), [])
    expect("qemu step needs its image",
           len(action_steps_missing_mirror("steps:\n  - uses: docker/setup-qemu-action@v3\n")), 1)
    expect("folded run block", [r for b in workflow_run_blocks("    run: >-\n      docker run --rm\n      python:3.12 x\n")
                                for c in shell_commands(b) for _, r in shell_refs(c)], ["python:3.12"])
    if failures:
        print("FAIL: scanner self-test")
        for line in failures:
            print(f"  {line}")
        return 1
    print("PASS: scanner self-test")
    return 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv[1:] else main())
