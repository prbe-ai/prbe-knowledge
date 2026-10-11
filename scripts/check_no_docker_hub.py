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
  - Dockerfiles (`Dockerfile*`, `*.Dockerfile`, `Containerfile*`) and compose
    `dockerfile_inline:`: the `# syntax=` directive, FROM (ARG defaults
    substituted), and the leading `COPY --from=` / `RUN --mount=...,from=` flags.
    Heredoc bodies are skipped.
  - Every tracked YAML file: `image:` values (scalars, Helm `default "x"`,
    flow lists, `image: {repository:}` maps), any `*_IMAGE:` key, `uses:
    docker://`, scalar `container:`, Buildx `driver-opts` images; every
    setup-buildx-action step must name its BuildKit image and every
    setup-qemu-action step its binfmt image. `image:` keys under a `matrix:`
    are values, not pulls, and are skipped. Every scalar is also read as shell.
  - Shell (YAML scalars, `*.sh`, Makefiles): images of `docker run|create|pull`,
    `docker container run|create` and `docker image pull`, including inside
    `$(...)`, backticks and `sh -c '...'`; `docker buildx create`; and `*IMAGE`
    variable defaults and assignments.
A line that names a docker pull the scanner cannot parse, and a ref whose
registry host is a variable, fail closed: it cannot vouch for them. A ref that
is wholly one variable (a script argument, a matrix value) is skipped.

This is a lint, not a security boundary. Not scanned: images named in Python
or other source (testcontainers and the like), and refs a workflow builds at
runtime. Keep those on the mirror by review.

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
    # git globs of tracked YAML NOT to scan (all other *.yml / *.yaml are).
    "yaml_exclude": [],
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
_HUB_HOSTS = {"docker.io", "index.docker.io", "registry-1.docker.io", "registry.hub.docker.com"}
_DOCKERFILE_GLOBS = ("Dockerfile*", "*.Dockerfile", "*.dockerfile", "Containerfile*")
_UNRESOLVED = "<unresolved>"
_GHA_EXPR = re.compile(r"\$\{\{.*?\}\}")
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
_SHORT_VALUE = set("acehlmpuvw")
# `docker` global options that take a value (`docker --context x run ...`).
_GLOBAL_VALUE_OPTS = frozenset(
    "--context -c --config -H --host -l --log-level --tlscacert --tlscert --tlskey".split()  # noqa: SIM905
)
_DOCKER_WORD = re.compile(r"[@+-]*(?:\S*/)?docker")
_DOCKER_PULL = re.compile(
    r"(?:^|[\s;&|(`\"'@+-])(?:[^\s;&|(`\"']*/)?docker"
    r"(?:\s+(?!(?:run|create|pull|container|image|buildx)\b)-{1,2}[\w-]+(?:[= ][^\s-]\S*)?)*"
    r"\s+(?:run|create|pull|container\s+(?:run|create)|image\s+pull|buildx\s+create)\b"
)


def is_docker_hub(ref: str) -> bool:
    """Docker's own rule: a first path segment with no `.` or `:` (and not
    `localhost`) is not a registry host, so the ref resolves to Docker Hub."""
    first, sep, _ = ref.partition("/")
    if not sep:
        return True
    if first in _HUB_HOSTS:
        return True
    return "." not in first and ":" not in first and first != "localhost"


def repository(ref: str) -> str:
    """`host/name:tag@sha256:x` -> `host/name` (a `:port` in the host is kept)."""
    ref = ref.split("@", 1)[0]
    head, sep, tail = ref.rpartition(":")
    return head if sep and "/" not in tail else ref


def classify(ref: str) -> str | None:
    """None when `ref` is fine (or wholly a variable), else why it is not."""
    repo = repository(ref.strip())
    if not repo or repo == _UNRESOLVED or re.fullmatch(r"\$\{?\w+\}?", repo):
        return None
    first, sep, _ = repo.partition("/")
    if "$" in first or _UNRESOLVED in first:
        return "unresolvable (the registry host is a variable)"
    if sep and first not in _HUB_HOSTS and ("." in first or ":" in first or first == "localhost"):
        return None  # a literal non-Hub registry; the rest may vary
    if "$" in repo or _UNRESOLVED in repo:
        return "unresolvable (variable in the repository, no registry host)"
    return "Docker Hub" if is_docker_hub(repo) else None


_VAR_REF = re.compile(r"\$\{(\w+)(?::?([-+=])([^}]*))?\}|\$(\w+)")


def substitute(value: str, args: dict[str, str]) -> str:
    """Apply known values and `${X:-default}` defaults; leave the rest."""

    def one(m: re.Match) -> str:
        name = m.group(1) or m.group(4)
        if name in args:
            return args[name]
        if m.group(2) in {"-", "="}:
            return m.group(3)
        return m.group(0)

    return _VAR_REF.sub(one, value)


# --- Dockerfiles ------------------------------------------------------------

_HEREDOC = re.compile(r"(?:^|\s)\d*<<-?([\"']?)([A-Za-z_]\w*)\1(?=\s|$)")


def _leading_flags(rest: str) -> list[str]:
    flags = []
    for word in rest.split():
        if not word.startswith("--"):
            break
        flags.append(word)
    return flags


def dockerfile_refs(text: str) -> list[tuple[str, str]]:
    """(kind, ref) for every image a Dockerfile pulls."""
    lines = text.splitlines()
    found: list[tuple[str, str]] = []
    escape = "\\"
    i = 0
    while i < len(lines):  # parser directives: leading `# key = value`, any order
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
    pending: list[str] = []
    for line in lines[i:]:
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
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
        if re.match(r"\s*(RUN|COPY|ADD)\b", full, re.IGNORECASE):
            pending = [m.group(2) for m in _HEREDOC.finditer(full)]
    if buf:
        logical.append(buf)
    if pending:
        found.append(("unparseable", f"heredoc `{pending[0]}` never closed"))

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
            try:
                tokens = shlex.split(rest)
            except ValueError:
                tokens = rest.split()
            for token in tokens:
                name, eq, value = token.partition("=")
                target = stage_args if seen_from else global_args
                if eq:
                    target[name] = substitute(value, {**global_args, **target})
                elif seen_from and name in global_args:
                    stage_args[name] = global_args[name]
        elif instr == "FROM":
            seen_from = True
            stage_args = {}
            words = [w for w in rest.split() if not w.startswith("--")]
            if not words:
                continue
            base = substitute(words[0], global_args)
            is_stage = base.lower() in stages or words[0].lower() in stages
            if len(words) >= 3 and words[1].upper() == "AS":
                stages.add(words[2].lower())
            if not is_stage and base != "scratch":
                found.append(("from", base))
        elif instr in {"COPY", "ADD", "RUN"}:
            args = {**global_args, **stage_args}
            values = []
            for flag in _leading_flags(rest):
                if flag.startswith("--from="):
                    values.append(flag[7:])
                elif flag.startswith("--mount="):
                    values += [kv[5:] for kv in flag[8:].split(",") if kv.startswith("from=")]
            for raw in values:
                value = substitute(raw.strip("\"'"), args)
                if value.lower() in stages or value.isdigit():
                    continue
                found.append(("copy-from", value))
    return found


# --- Shell ------------------------------------------------------------------


def strip_comment(line: str) -> str:
    """`cmd  # note` -> `cmd`, ignoring `#` inside quotes or words (`$#`, `a#b`)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"':
                continue
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t;&|("):
            return line[:i].rstrip()
    return line


def shell_commands(text: str) -> list[str]:
    """Logical shell lines: comments dropped, continuations joined, heredoc
    bodies dropped, GHA `${{ }}` neutralised."""
    out, buf = [], ""
    pending: list[str] = []
    for line in text.splitlines():
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        line = strip_comment(line)
        if not line.strip() and not buf:
            continue
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        full = _GHA_EXPR.sub(_UNRESOLVED, buf + line)
        buf = ""
        out.append(full)
        pending = [m.group(2) for m in _HEREDOC.finditer(full)]
    if buf:
        out.append(_GHA_EXPR.sub(_UNRESOLVED, buf))
    return out


def _substitutions(command: str) -> list[str]:
    """Bodies of `$(...)`, backticks and `sh -c '...'` / `bash -c "..."`."""
    bodies = []
    i = 0
    while True:
        start = command.find("$(", i)
        if start < 0:
            break
        depth, j = 1, start + 2
        while j < len(command) and depth:
            depth += {"(": 1, ")": -1}.get(command[j], 0)
            j += 1
        bodies.append(command[start + 2 : j - 1])
        i = start + 2
    bodies += re.findall(r"`([^`]*)`", command)
    bodies += [m[1] for m in re.findall(r"\b(?:ba|z)?sh\s+-c\s+(['\"])(.*?)\1", command)]
    return bodies


def _image_after(tokens: list[str], start: int) -> str | None:
    j = start
    while j < len(tokens):
        tok = tokens[j]
        if tok in _RUN_VALUE_OPTS:
            j += 2
        elif tok.startswith("-") and not tok.startswith("--") and len(tok) > 2 and "=" not in tok:
            j += 2 if tok[-1] in _SHORT_VALUE else 1  # `-dp 5432:5432`
        elif tok.startswith("-"):
            j += 1
        elif tok in {";", "&&", "||", "|", "&", "(", ")"}:
            return None
        else:
            return tok
    return None


def _docker_commands(tokens: list[str]) -> list[tuple[str, str]]:
    found = []
    for i, tok in enumerate(tokens):
        if not _DOCKER_WORD.fullmatch(tok):
            continue
        j = i + 1
        while j < len(tokens) and tokens[j].startswith("-"):  # global options
            j += 2 if tokens[j] in _GLOBAL_VALUE_OPTS else 1
        verb = tokens[j] if j < len(tokens) else ""
        nxt = tokens[j + 1] if j + 1 < len(tokens) else ""
        if verb in {"run", "create", "pull"}:
            start = j + 1
        elif (verb, nxt) in {("container", "run"), ("container", "create"), ("image", "pull")}:
            start = j + 2
        elif (verb, nxt) == ("buildx", "create"):
            rest = tokens[j + 2 :]
            joined = " ".join(rest)
            m = re.search(r"--driver-opt(?:=|\s+)[^\s]*image=([^\s,]+)", joined)
            if m:
                found.append(("buildkit", m.group(1)))
            elif not re.search(r"--driver(?:=|\s+)docker\b", joined):
                found.append(("action", "docker buildx create without --driver-opt image="
                              f"{MIRROR}moby/buildkit:..."))
            else:
                found.append(("buildkit", "docker-driver"))
            continue
        else:
            continue
        image = _image_after(tokens, start)
        if image is not None:
            found.append(("docker", substitute(image, {})))
    return found


def shell_refs(command: str, _depth: int = 0) -> list[tuple[str, str]]:
    """(kind, ref) for docker pulls and *IMAGE defaults in one logical line."""
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
        found.append(("default", m.group(3)))
    if "docker" not in command:
        return found
    pulls = 0
    if _depth < 3:
        for body in _substitutions(command):
            inner = shell_refs(body, _depth + 1)
            found += inner
            pulls += sum(1 for k, _ in inner if k in {"docker", "buildkit", "action", "unparseable"})
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        tokens = None
    if tokens is not None:
        cmds = _docker_commands(tokens)
        found += cmds
        pulls += len(cmds)
    expected = len(_DOCKER_PULL.findall(command))
    if expected > pulls:
        found.append(("unparseable", command.strip()))
    return found


# --- YAML -------------------------------------------------------------------


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


_KEY_LINE = re.compile(r"^(\s*)(-\s+)?([\w.-]+)\s*:(?:\s+(.*?))?\s*$")


def yaml_scalars(text: str) -> list[tuple[list[str], str, str]]:
    """(ancestor keys, key, value) for every mapping key with a scalar value.

    Block scalars are joined (`|` with newlines, `>` with spaces), quoted
    scalars unquoted, plain scalars continued across lines. Lines inside a
    block scalar are never read as keys.
    """
    lines = text.splitlines()
    out: list[tuple[list[str], str, str]] = []
    stack: list[tuple[int, str]] = []  # (key column, key)
    i = 0
    while i < len(lines):
        raw = lines[i]
        m = _KEY_LINE.match(raw)
        if not m or re.match(r"\s*#", raw):
            i += 1
            continue
        col = len(m.group(1)) + len(m.group(2) or "")
        key = m.group(3)
        rest = re.sub(r"\s+#.*$", "", m.group(4) or "").strip()
        while stack and stack[-1][0] >= col:
            stack.pop()
        ancestors = [k for _, k in stack]
        if rest[:1] in {"|", ">"}:
            folded = rest.startswith(">")
            body, j = [], i + 1
            while j < len(lines) and (not lines[j].strip() or _indent(lines[j]) > col):
                body.append(lines[j])
                j += 1
            nonblank = [b for b in body if b.strip()]
            dedent = min((_indent(b) for b in nonblank), default=0)
            texts = [b[dedent:] if b.strip() else "" for b in body]
            value = " ".join(t.strip() for t in texts if t.strip()) if folded else "\n".join(texts)
            out.append((ancestors, key, value))
            i = j
            continue
        if rest:
            j = i + 1
            if rest[0] not in "\"'[{&*":
                while (j < len(lines) and lines[j].strip() and _indent(lines[j]) > col
                       and not re.match(r"\s*(?:-\s|#|[\w.-]+\s*:(?:\s|$))", lines[j])):
                    rest += " " + lines[j].strip()
                    j += 1
            if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "\"'":
                rest = rest[1:-1]
            out.append((ancestors, key, rest))
            i = j
            continue
        stack.append((col, key))
        i += 1
    return out


def _image_values(value: str) -> list[str]:
    value = _GHA_EXPR.sub(_UNRESOLVED, value).strip()
    if "{{" in value:  # Helm: only a literal fallback is a ref
        return re.findall(r"\bdefault\s+\"([^\"]+)\"", value)
    value = re.sub(r"^&\w+\s+", "", value)  # anchor
    if value.startswith("*"):  # alias
        return []
    if value.startswith("["):
        return [v.strip().strip("\"'") for v in value.strip("[]").split(",") if v.strip()]
    if value.startswith("{"):
        return re.findall(r"\bimage:\s*[\"']?([^,}\s\"']+)", value)
    return [substitute(value.split()[0].strip("\"'"), {})] if value.split() else []


def yaml_refs(text: str, workflow: bool = False) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    scalars = yaml_scalars(text)
    by_parent: dict[tuple[str, ...], dict[str, str]] = {}
    for ancestors, key, value in scalars:
        by_parent.setdefault(tuple(ancestors), {})[key] = value
    for ancestors, key, value in scalars:
        lkey = key.lower()
        if lkey == "image":
            if workflow and "matrix" in ancestors:
                continue  # a matrix value; the pull happens where it is used
            found += [("image", v) for v in _image_values(value)]
        elif lkey == "repository" and ancestors and ancestors[-1].lower() == "image":
            if "{{" not in value:
                registry = by_parent.get(tuple(ancestors), {}).get("registry", "")
                repo = value.strip("\"'")
                found.append(("image", f"{registry}/{repo}" if registry and "{{" not in registry else repo))
        elif lkey.endswith("image") or lkey.endswith("_image"):
            found += [("default", v) for v in _image_values(value)]
        elif lkey == "container":
            found += [("container", v) for v in _image_values(value)]
        elif lkey == "uses" and value.startswith("docker://"):
            found.append(("uses", value[len("docker://"):].strip()))
        elif lkey == "driver-opts":
            found += [("buildkit", v) for v in re.findall(r"\bimage=([^\s,\"']+)", value)]
        elif lkey == "dockerfile_inline":
            found += dockerfile_refs(value)
        if lkey in {"image", "repository", "uses", "dockerfile_inline", "driver-opts"}:
            continue
        for command in shell_commands(value):
            found += shell_refs(command)
    for line in text.splitlines():  # flow maps: `db: {image: postgres:16, ...}`
        if not re.match(r"\s*#", line):
            found += [("image", v) for v in re.findall(r"[{,]\s*image:\s*[\"']?([^,}\s\"']+)", line)]
    return found


def action_steps_missing_mirror(text: str) -> list[str]:
    """setup-buildx / setup-qemu steps that would pull their default Hub image."""
    problems = []
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        if re.match(r"\s*#", raw):
            continue
        m = re.search(r"uses:\s*[\"']?docker/(setup-buildx-action|setup-qemu-action)@", raw)
        if not m:
            continue
        step_indent = _indent(raw)
        if raw.lstrip().startswith("uses:"):
            for back in range(idx - 1, -1, -1):  # the step's `- ` line
                if lines[back].strip() and _indent(lines[back]) < step_indent:
                    step_indent = _indent(lines[back]) if lines[back].lstrip().startswith("-") else step_indent
                    break
        buildx = m.group(1) == "setup-buildx-action"
        want = f"image={MIRROR}moby/buildkit:" if buildx else f"image: {MIRROR}tonistiigi/binfmt"
        ok = False
        for nxt in lines[idx + 1 :]:
            if not nxt.strip() or re.match(r"\s*#", nxt):
                continue
            if _indent(nxt) <= step_indent:
                break
            body = re.sub(r"\s+#.*$", "", nxt).replace('"', "").replace("'", "")
            if want in body or (buildx and re.search(r"\bdriver:\s*docker\s*$", body)):
                ok = True
        if not ok:
            problems.append(f"docker/{m.group(1)} without `{want}...`")
    return problems


# --- Repo scan --------------------------------------------------------------


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    return sorted(p for p in out.split("\0") if p)


def scan(read=lambda p: (REPO / p).read_text(errors="replace")) -> list[tuple[str, str, str]]:
    """(path, kind, ref) for every image reference, plus action-step problems."""
    results: list[tuple[str, str, str]] = []
    for path in tracked():
        name = Path(path).name
        if any(fnmatch.fnmatch(name, g) for g in _DOCKERFILE_GLOBS) and not name.endswith(".dockerignore"):
            results += [(path, k, r) for k, r in dockerfile_refs(read(path))]
        elif path.endswith((".yml", ".yaml")):
            if any(fnmatch.fnmatch(path, g) for g in CONFIG["yaml_exclude"]):
                continue
            text = read(path)
            workflow = path.startswith(".github/workflows/")
            results += [(path, k, r) for k, r in yaml_refs(text, workflow=workflow)]
            if workflow:
                results += [(path, "action", p) for p in action_steps_missing_mirror(text)]
        elif path.endswith(".sh") or name == "Makefile" or name.startswith("entrypoint"):
            for command in shell_commands(read(path)):
                results += [(path, k, r) for k, r in shell_refs(command)]
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
            offenders.append(f"{path}: names a docker pull this scanner cannot parse, so it cannot "
                             f"vouch for it (rewrite it, or allowlist it in CONFIG): {ref}")
            continue
        if ref == "docker-driver":
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


def self_test() -> int:
    failures: list[str] = []

    def expect(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    def hub(refs) -> list[str]:
        """The refs that would fail: Docker Hub or unresolvable."""
        return [r for _, r in refs if classify(r) is not None]

    def kinds(refs) -> list[str]:
        return [k for k, _ in refs]

    for ref, want in [
        ("python:3.12-slim", True), ("library/python", True), ("docker.io/library/python", True),
        ("registry.hub.docker.com/library/python", True), ("prom/node-exporter:latest", True),
        ("mirror.gcr.io/library/python:3.12-slim", False), ("ghcr.io/a/b:1", False),
        ("localhost/x:1", False), ("localhost:5000/x:1", False),
    ]:
        expect(f"is_docker_hub({ref})", is_docker_hub(ref), want)
    expect("repository with port", repository("localhost:5000/x:1"), "localhost:5000/x")
    expect("repository with digest", repository("a.io/b:1@sha256:ff"), "a.io/b")
    expect("wholly variable", [classify(r) for r in ("$IMAGE", "${IMAGE}", _UNRESOLVED)], [None] * 3)
    expect("variable host fails", classify("$REG/python:3.12") is not None, True)
    expect("literal host, variable name", classify(f"ghcr.io/prbe-ai/x-{_UNRESOLVED}:1"), None)
    expect("variable name, no host", classify("python-${V}:1") is not None, True)

    df = dockerfile_refs
    expect("syntax with spaces, after escape", df("# escape=`\n# syntax = docker/dockerfile:1\nFROM a.io/b\n"),
           [("syntax", "docker/dockerfile:1"), ("from", "a.io/b")])
    expect("lowercase from + stage", hub(df("from python:3.12 as base\nFROM base\n")), ["python:3.12"])
    expect("FROM node AS node", hub(df("FROM node AS node\nRUN node -v\n")), ["node"])
    expect("ARG default", hub(df("ARG B=python:3.12\nFROM ${B} AS x\n")), ["python:3.12"])
    expect("ARG tag", hub(df("ARG V=3.12\nFROM python:${V}-slim\n")), ["python:3.12-slim"])
    expect("ARG without default skipped", hub(df("ARG B\nFROM ${B}\n")), [])
    expect("COPY --from", hub(df("FROM a.io/b AS s\nCOPY --from=busybox:1 /x /y\nCOPY --from=s /a /b\n")),
           ["busybox:1"])
    expect("RUN --mount from", hub(df("FROM a.io/b\nRUN --mount=type=bind,from=alpine:3,target=/a true\n")),
           ["alpine:3"])
    expect("--from inside a RUN command", hub(df("FROM a.io/b\nRUN pandoc --from=markdown x\n")), [])
    expect("python inside RUN", hub(df('FROM a.io/b\nRUN python -c "f(from=1)" \\\n && echo from x\n')), [])
    expect("heredoc body", hub(df("FROM a.io/b\nRUN <<EOF\nfrom pathlib import Path\nEOF\n")), [])
    expect("<< that is not a heredoc", hub(df('FROM a.io/b\nRUN python -c "print(1<<20)"\nFROM node:20\n')),
           ["node:20"])
    expect("unclosed heredoc fails closed", kinds(df("FROM a.io/b\nRUN <<EOF\nx\n")), ["unparseable", "from"])
    expect("continuation", hub(df("FROM \\\n  python:3.12\n")), ["python:3.12"])
    expect("platform flag", hub(df("FROM --platform=$BUILDPLATFORM golang:1 AS g\n")), ["golang:1"])

    def sh(cmd: str) -> list[tuple[str, str]]:
        return [x for c in shell_commands(cmd) for x in shell_refs(c)]

    expect("run with value opts", hub(sh("docker run -d --name p -e A=1 --health-interval 5s python:3.12 x")),
           ["python:3.12"])
    expect("combined short flags", hub(sh("docker run -dp 5432:5432 python:3.12")), ["python:3.12"])
    expect("trailing comment", hub(sh("docker run --rm python:3.12 true  # don't cache")), ["python:3.12"])
    expect("two commands", hub(sh("docker run --rm a.io/x true && docker run --rm python:3.12 true")),
           ["python:3.12"])
    expect("pull/create/container run",
           hub(sh("docker pull alpine:3; docker create redis:7; docker container run busybox")),
           ["alpine:3", "redis:7", "busybox"])
    expect("double space", hub(sh("docker  run --rm python:3.12 true")), ["python:3.12"])
    expect("global options", hub(sh("docker --context default run --rm python:3.12 true")), ["python:3.12"])
    expect("make recipe prefix and path", hub(sh("\t@docker run -d pgvector/pgvector:pg16\n/usr/bin/docker pull x:1")),
           ["pgvector/pgvector:pg16", "x:1"])
    expect("quoted command substitution", hub(sh('PG_CID="$(docker run -d -e A=1 postgres:16)"')), ["postgres:16"])
    expect("backticks and sh -c", hub(sh("v=`docker run --rm a:1`; bash -c 'docker pull b:2'")), ["a:1", "b:2"])
    expect("GHA expr in a volume", hub(sh("docker run -v ${{ github.workspace }}:/w a.io/x true")), [])
    expect("variable image skipped", hub(sh('docker run --rm "$IMAGE" true')), [])
    expect("variable with Hub default", hub(sh('docker run --rm "${REG:-docker.io}/library/postgres:16" true')),
           ["docker.io/library/postgres:16"])
    expect("unparseable docker line fails closed", "unparseable" in kinds(sh("docker run 'x")), True)
    expect("prose with an apostrophe", sh("echo the docker image didn't build"), [])
    expect("comments", sh("# Smoke test: docker run the image\n#   docker run x  -> y\n"
                          "true  # no `docker pull` needed\n"), [])
    expect("# inside quotes is not a comment", hub(sh('docker run --rm -e "A=#1" python:3.12 x')),
           ["python:3.12"])
    expect("shell heredoc body", sh('cat <<EOF\nThe docker run didn\'t work\nEOF\n'), [])
    expect("image defaults",
           hub(sh('POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:16}"; : "${PG16_IMAGE:=redis:7}"')),
           ["postgres:16", "redis:7"])
    expect("IMAGE_KIND is not an image", sh('echo "${IMAGE_KIND:-legacy}"'), [])
    expect("plain image assignment", hub(sh("  local BASE_IMAGE=python:3.12")), ["python:3.12"])
    expect("buildx create from the Hub", kinds(sh("docker buildx create --use --name ci")), ["action"])
    expect("buildx create on the mirror",
           hub(sh(f"docker buildx create --driver-opt image={MIRROR}moby/buildkit:v1 --use")), [])

    yr = yaml_refs
    expect("yaml forms", sorted(hub(yr(
        "x:\n  image: python:3.12 # c\n  # image: ignored:1\n  container: node:20\n"
        "  steps:\n    - uses: docker://alpine:3.19\n  img:\n    image:\n      repository: pgvector/pgvector\n"
        "      tag: pg16\n  b:\n    driver-opts: image=moby/buildkit:x\n  y: '{{ .Values.image }}'\n"))),
        sorted(["python:3.12", "node:20", "alpine:3.19", "pgvector/pgvector", "moby/buildkit:x"]))
    expect("helm default", hub(yr('  image: {{ .Values.x | default "busybox" }}\n'
                                  '  image: {{ include "probe-knowledge.image" . | quote }}\n')), ["busybox"])
    expect("flow list, flow map, anchor",
           sorted(hub(yr('a:\n  image: ["python:3.11", "mirror.gcr.io/x:1"]\n'
                         '  db: {image: postgres:16, ports: [1]}\n  c:\n    image: &pg redis:7\n'))),
           ["postgres:16", "python:3.11", "redis:7"])
    expect("compose default", hub(yr("  image: ${PG_IMAGE:-postgres:16}\n")), ["postgres:16"])
    expect("matrix image key", hub(yr("jobs:\n  b:\n    strategy:\n      matrix:\n        include:\n"
                                      "          - image: knowledge\n", workflow=True)), [])
    expect("matrix GHA expr", hub(yr("        image: ${{ fromJSON(needs.x.outputs.images) }}\n")), [])
    expect("*_IMAGE keys and build-args",
           sorted(hub(yr("env:\n  PG_IMAGE: postgres:16\nsteps:\n  - with:\n      build-args: |\n"
                         "        BASE_IMAGE=python:3.12-slim\n")), ),
           ["postgres:16", "python:3.12-slim"])
    expect("shell in any key", hub(yr("  - with:\n      command: docker pull pgvector/pgvector:pg16\n")),
           ["pgvector/pgvector:pg16"])
    expect("quoted inline run", hub(yr('  - run: "docker run --rm python:3.12-slim python -V"\n')),
           ["python:3.12-slim"])
    expect("plain run over two lines", hub(yr("  - run: docker run --rm\n      python:3.12-slim python -V\n")),
           ["python:3.12-slim"])
    expect("folded run block", hub(yr("    run: >-\n      docker run --rm\n      python:3.12 x\n")),
           ["python:3.12"])
    expect("GITHUB_OUTPUT image= is not BuildKit",
           hub(yr('  - run: echo "image=${REGISTRY}/${NAME}:${TAG}" >> "$GITHUB_OUTPUT"\n')), [])
    expect("dockerfile_inline", hub(yr("  tools:\n    build:\n      dockerfile_inline: |\n"
                                       "        FROM python:3.12-slim\n")), ["python:3.12-slim"])
    expect("step summary prose", yr('  - run: |\n      cat >> "$GITHUB_STEP_SUMMARY" <<EOF\n'
                                     "      The docker image didn't build.\n      EOF\n"), [])
    steps = action_steps_missing_mirror
    expect("buildx without driver-opts", len(steps("steps:\n  - uses: docker/setup-buildx-action@v3\n  - run: x\n")), 1)
    expect("quoted uses", len(steps('steps:\n  - uses: "docker/setup-buildx-action@v3"\n')), 1)
    expect("commented driver-opts", len(steps(
        "steps:\n  - uses: docker/setup-buildx-action@v3\n    with:\n"
        "      # driver-opts: image=mirror.gcr.io/moby/buildkit:buildx-stable-1\n")), 1)
    expect("buildx on the mirror", steps(
        "steps:\n  - name: b\n    uses: docker/setup-buildx-action@v3\n    with:\n"
        "      driver-opts: image=mirror.gcr.io/moby/buildkit:buildx-stable-1\n  - run: x\n"), [])
    expect("buildx multi-line driver-opts", steps(
        "steps:\n  - uses: docker/setup-buildx-action@v3\n    with:\n      driver-opts: |\n"
        "        image=mirror.gcr.io/moby/buildkit:buildx-stable-1\n  - run: x\n"), [])
    expect("buildx docker driver", steps(
        "steps:\n  - uses: docker/setup-buildx-action@v3\n    with:\n      driver: docker\n"), [])
    expect("qemu needs its image", len(steps("steps:\n  - uses: docker/setup-qemu-action@v3\n")), 1)

    if failures:
        print("FAIL: scanner self-test")
        for line in failures:
            print(f"  {line}")
        return 1
    print("PASS: scanner self-test")
    return 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv[1:] else main())
