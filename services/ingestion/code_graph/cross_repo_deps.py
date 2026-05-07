"""Cross-repo dependency extraction (PR architecture-edges-from-facts).

For each repo's clone, find verified references to OTHER repos in the same
customer's org. Edges flow into ``graph_edges`` as ``DEPENDS_ON`` between
``Repo`` graph nodes. The wiki index renderer reads these edges instead
of letting the LLM hallucinate connections from page summaries.

Pipeline:

  1. Query the customer's other known repos (from ``code_repo_state``).
  2. For each tracked file in the source repo (``git ls-files``), regex-
     match each candidate repo name + variants (snake / kebab / camel /
     screaming snake).
  3. One Flash Lite call per source repo classifies every candidate
     match as REAL or COINCIDENCE, given the surrounding file context.
  4. Aggregate verified matches per (source_repo, target_repo) pair and
     emit one ``DEPENDS_ON`` edge per pair, with provenance in
     ``properties``.

Bidirectionality is computed at READ time (in the wiki index renderer)
by checking whether the reverse edge also exists. That decoupling avoids
needing a "wait for all repos to finish" hook in the event pipeline.

Idempotency: callers should DELETE existing ``DEPENDS_ON`` edges from
the source repo's node before persisting new ones, so a dropped
reference disappears rather than lingering forever.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
import orjson

from shared.db import with_tenant
from shared.logging import get_logger

log = get_logger(__name__)


# Bound the prompt size + cost of the Flash Lite classification call. A
# 30-repo customer with ~150 candidates per repo lands well below this.
_MAX_CANDIDATES_PER_SOURCE_REPO = 500
_MAX_PROMPT_CONTENT_BYTES = 80_000

# Per-file caps so a single 50KB file doesn't dominate the prompt budget.
_FULL_FILE_THRESHOLD_BYTES = 10_000  # files <= this go in whole
_LINES_OF_CONTEXT_AROUND_MATCH = 200  # files > threshold get windowed

# Guardrail against pathological binaries that slipped past .gitignore.
_FILE_SIZE_HARD_CAP_BYTES = 1_048_576

# Regex character class that defines the boundaries of a repo identifier
# in the wild. Alphanumerics + hyphens only — `_` is intentionally
# excluded so `PRBE_KNOWLEDGE` matches inside `PRBE_KNOWLEDGE_URL` (a
# common env-var convention) and `prbe_knowledge` matches inside
# `prbe_knowledge_v2_metadata` (variable-name reference). The LLM
# classifier downstream marks unrelated underscore-adjacent matches as
# coincidences. Hyphen stays in the class so `prbe-knowledge` does NOT
# match as a substring of `prbe-knowledge-mcp` (a distinct repo).
_BOUNDARY_CLASS = r"[A-Za-z0-9-]"


@dataclass(frozen=True)
class CandidateMatch:
    file_path: str
    line_number: int
    snippet: str
    candidate_target: str


@dataclass(frozen=True)
class VerifiedMatch:
    file_path: str
    line_number: int
    snippet: str
    target_repo: str
    reason: str


@dataclass(frozen=True)
class CrossRepoEdge:
    source_repo: str
    target_repo: str
    evidence: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------


def _split_segments(name: str) -> list[str]:
    """Split a repo name on hyphen/underscore/dot into casing tokens."""
    return [s for s in re.split(r"[-_.]+", name) if s]


def repo_name_variants(repo_full_name: str) -> list[str]:
    """Generate the spellings of a repo name we expect to see in code/config.

    Given ``prbe-ai/forward-deployed-agent-demo``, returns:
      - ``forward-deployed-agent-demo`` (kebab; original)
      - ``forward_deployed_agent_demo`` (snake)
      - ``forwardDeployedAgentDemo``    (camelCase)
      - ``ForwardDeployedAgentDemo``    (PascalCase)
      - ``FORWARD_DEPLOYED_AGENT_DEMO`` (SCREAMING_SNAKE)

    Includes ``owner/name`` as a sixth variant so CI workflows that use
    ``uses: prbe-ai/<repo>@v1`` also hit. Single-segment names skip the
    case-split variants since they're identical.
    """
    name = repo_full_name.rsplit("/", 1)[-1]
    segments = _split_segments(name)
    out: list[str] = [name]
    if len(segments) > 1:
        out.append("_".join(segments))
        out.append(segments[0] + "".join(s.capitalize() for s in segments[1:]))
        out.append("".join(s.capitalize() for s in segments))
        out.append("_".join(s.upper() for s in segments))
    out.append(repo_full_name)  # owner/name (CI workflow style)
    # De-dupe while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _build_match_regex(targets: list[tuple[str, list[str]]]) -> re.Pattern[str]:
    """One regex per source-repo-scan that catches any variant of any target.

    Returned pattern groups are anonymous; we re-resolve which target a
    match belongs to via a lookup map. ``targets`` is a list of
    ``(target_repo_full_name, variants)`` tuples.
    """
    parts: list[str] = []
    for _full, variants in targets:
        parts.extend(re.escape(v) for v in variants)
    if not parts:
        # No targets — return a regex that never matches (safe sentinel).
        return re.compile(r"(?!x)x")
    alternation = "|".join(parts)
    pattern = (
        f"(?<!{_BOUNDARY_CLASS})(?:{alternation})(?!{_BOUNDARY_CLASS})"
    )
    return re.compile(pattern)


def _variant_to_target_index(targets: list[tuple[str, list[str]]]) -> dict[str, str]:
    """Map every variant string back to its canonical target repo full name."""
    out: dict[str, str] = {}
    for full, variants in targets:
        for v in variants:
            out[v] = full
    return out


# ---------------------------------------------------------------------------
# File walking
# ---------------------------------------------------------------------------


def list_tracked_files(target_dir: Path) -> list[str]:
    """Return tracked file paths via ``git ls-files``.

    Respects ``.gitignore`` for free. If the directory is not a git
    checkout (rare; defensive), falls back to a recursive walk skipping
    common build artifact dirs.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return [line for line in result.stdout.splitlines() if line]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        log.warning("cross_repo_deps.git_ls_files_failed", target_dir=str(target_dir))
        return _fallback_walk(target_dir)


_FALLBACK_SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", ".venv", "venv", ".git", "dist", "build",
    "__pycache__", "target", ".next", ".nuxt", ".cache",
})


def _fallback_walk(target_dir: Path) -> list[str]:
    out: list[str] = []
    for path in target_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _FALLBACK_SKIP_DIRS for part in path.parts):
            continue
        try:
            rel = path.relative_to(target_dir)
        except ValueError:
            continue
        out.append(str(rel))
    return out


# ---------------------------------------------------------------------------
# Candidate match collection
# ---------------------------------------------------------------------------


def _read_file_safely(path: Path) -> str | None:
    try:
        if path.stat().st_size > _FILE_SIZE_HARD_CAP_BYTES:
            return None
    except OSError:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return None


def collect_candidates(
    target_dir: Path,
    targets: list[tuple[str, list[str]]],
) -> list[CandidateMatch]:
    """Scan tracked files for any variant match against any target repo."""
    if not targets:
        return []
    pattern = _build_match_regex(targets)
    variant_to_target = _variant_to_target_index(targets)
    candidates: list[CandidateMatch] = []
    files = list_tracked_files(target_dir)
    for rel_path in files:
        full_path = target_dir / rel_path
        body = _read_file_safely(full_path)
        if body is None:
            continue
        for line_no, line in enumerate(body.splitlines(), start=1):
            for match in pattern.finditer(line):
                target = variant_to_target.get(match.group(0))
                if not target:
                    continue
                snippet = line.strip()[:200]
                candidates.append(
                    CandidateMatch(
                        file_path=rel_path,
                        line_number=line_no,
                        snippet=snippet,
                        candidate_target=target,
                    )
                )
    return candidates


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------


def _build_file_context_from_contents(
    file_contents: dict[str, str],
    file_to_match_lines: dict[str, list[int]],
) -> str:
    """Content-dict-based variant of the file-context block builder.

    Used by both the initial-backfill path (which preloads files from
    a cloned working tree) and the webhook re-verification path (which
    receives post-push contents from the GitHub Contents API). Same
    windowing logic for files larger than the single-file threshold;
    same total-bytes cap so the LLM input stays bounded.
    """
    parts: list[str] = []
    used_bytes = 0
    for rel_path, match_lines in file_to_match_lines.items():
        if used_bytes >= _MAX_PROMPT_CONTENT_BYTES:
            break
        body = file_contents.get(rel_path)
        if body is None:
            continue
        if len(body.encode("utf-8")) <= _FULL_FILE_THRESHOLD_BYTES:
            block = f"=== {rel_path} ===\n{body}\n"
        else:
            lines = body.splitlines()
            kept_ranges: list[tuple[int, int]] = []
            for line_no in sorted(set(match_lines)):
                start = max(1, line_no - _LINES_OF_CONTEXT_AROUND_MATCH // 2)
                end = min(len(lines), line_no + _LINES_OF_CONTEXT_AROUND_MATCH // 2)
                if kept_ranges and kept_ranges[-1][1] + 1 >= start:
                    kept_ranges[-1] = (kept_ranges[-1][0], max(kept_ranges[-1][1], end))
                else:
                    kept_ranges.append((start, end))
            slices: list[str] = []
            for start, end in kept_ranges:
                slices.append(
                    f"[lines {start}-{end}]\n"
                    + "\n".join(lines[start - 1 : end])
                )
            block = f"=== {rel_path} (windowed) ===\n" + "\n[...]\n".join(slices) + "\n"
        block_bytes = len(block.encode("utf-8"))
        if used_bytes + block_bytes > _MAX_PROMPT_CONTENT_BYTES:
            allowed = _MAX_PROMPT_CONTENT_BYTES - used_bytes
            block = block.encode("utf-8")[:allowed].decode("utf-8", errors="ignore")
            parts.append(block)
            used_bytes += len(block.encode("utf-8"))
            break
        parts.append(block)
        used_bytes += block_bytes
    return "\n".join(parts)


def _build_file_context(
    target_dir: Path,
    file_to_match_lines: dict[str, list[int]],
) -> str:
    """Disk-backed variant — reads each requested file from `target_dir`."""
    contents: dict[str, str] = {}
    for rel_path in file_to_match_lines:
        body = _read_file_safely(target_dir / rel_path)
        if body is not None:
            contents[rel_path] = body
    return _build_file_context_from_contents(contents, file_to_match_lines)


async def _call_classifier_llm(
    *,
    source_repo: str,
    user_prompt: str,
    client: Any | None = None,
    log_prefix: str = "cross_repo_deps",
) -> list[dict[str, Any]] | None:
    """Shared Flash Lite call. Returns the parsed `verdicts` list, or
    None on any failure (caller decides fallback semantics)."""
    if client is None:
        try:
            from google import genai

            from shared.config import get_settings

            api_key = get_settings().google_api_key.get_secret_value()
            if not api_key:
                log.warning(f"{log_prefix}.no_google_api_key", source_repo=source_repo)
                return None
            client = genai.Client(api_key=api_key)
        except ImportError as exc:
            log.warning(f"{log_prefix}.google_genai_missing", error=str(exc))
            return None

    # Hard-coded to Flash Lite. Versioned alias matters — the
    # unversioned `gemini-flash-lite-preview` returns 404 (verified
    # 2026-05-07).
    model_name = "gemini-3.1-flash-lite-preview"

    try:
        resp = await client.aio.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config={
                "system_instruction": _CLASSIFY_SYSTEM_PROMPT,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:
        log.warning(
            f"{log_prefix}.gemini_failed",
            source_repo=source_repo,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        return None

    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        log.warning(f"{log_prefix}.empty_response", source_repo=source_repo)
        return None

    try:
        payload = orjson.loads(text)
    except orjson.JSONDecodeError:
        log.warning(
            f"{log_prefix}.malformed_response",
            source_repo=source_repo,
            preview=text[:200],
        )
        return None

    verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
    if not isinstance(verdicts, list):
        return None
    return verdicts


_CLASSIFY_SYSTEM_PROMPT = (
    "You are a static-analysis pre-filter. For a single source repo, "
    "below is a list of textual matches we found via regex. Each match "
    "is a candidate reference to ANOTHER repo in the same org. Your job: "
    "classify each as REAL or COINCIDENCE.\n\n"
    "REAL = the match genuinely references the other repo. Examples:\n"
    "  - HTTP client base URL pointing at <other>.internal / <other>.fly.dev\n"
    "  - env var name like <OTHER>_URL or BACKEND_URL=<other>.internal\n"
    "  - import / require of a package owned by the other repo\n"
    "  - README or doc cross-link to the other repo\n"
    "  - GitHub Actions `uses: <org>/<other>@...`\n"
    "  - Comment that meaningfully describes interaction with the other repo\n\n"
    "COINCIDENCE = the match does not actually reference the target. Examples:\n"
    "  - the regex matched a substring of a longer identifier\n"
    "  - a CHANGELOG entry from a deprecated era\n"
    "  - a generic comment that names the repo but does not depend on it\n"
    "  - vendored / generated content where the mention is irrelevant\n\n"
    "Output JSON ONLY. Do NOT include prose, explanation, or markdown "
    "code fences. Schema:\n"
    '  {"verdicts": [{"number": 1, "real": true, "reason": "..."}, ...]}'
)


async def classify_with_llm(
    *,
    source_repo: str,
    candidates: list[CandidateMatch],
    target_dir: Path,
    client: Any | None = None,
) -> list[VerifiedMatch]:
    """One Flash Lite call classifies every candidate as REAL or COINCIDENCE.

    Returns the kept (REAL) verifications. On any failure (no API key,
    Gemini error, malformed response) returns an empty list and logs the
    fall-through; the caller proceeds without cross-repo edges for this
    source repo rather than persisting unverified matches.
    """
    if not candidates:
        return []

    # Cap candidate volume so we never blow up the LLM input. Diverse
    # sampling: keep at least one match per (file, target) pair before
    # adding additional matches from the same pair.
    capped = _cap_candidates(candidates, _MAX_CANDIDATES_PER_SOURCE_REPO)

    file_to_lines: dict[str, list[int]] = {}
    for c in capped:
        file_to_lines.setdefault(c.file_path, []).append(c.line_number)
    file_context = _build_file_context(target_dir, file_to_lines)

    candidate_lines = [
        f"{i + 1}. file={c.file_path} line={c.line_number} candidate={c.candidate_target}"
        for i, c in enumerate(capped)
    ]
    user_prompt = (
        f"Source repo: {source_repo}\n\n"
        "=== Files ===\n\n"
        f"{file_context}\n\n"
        "=== Candidate matches ===\n"
        + "\n".join(candidate_lines)
    )

    verdicts = await _call_classifier_llm(
        source_repo=source_repo,
        user_prompt=user_prompt,
        client=client,
        log_prefix="cross_repo_deps",
    )
    if verdicts is None:
        return []

    verified: list[VerifiedMatch] = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        idx = v.get("number")
        if not isinstance(idx, int) or idx < 1 or idx > len(capped):
            continue
        if not v.get("real"):
            continue
        c = capped[idx - 1]
        reason = str(v.get("reason") or "")[:240]
        verified.append(
            VerifiedMatch(
                file_path=c.file_path,
                line_number=c.line_number,
                snippet=c.snippet,
                target_repo=c.candidate_target,
                reason=reason,
            )
        )
    return verified


# ---------------------------------------------------------------------------
# Per-edge re-verification (push webhook path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceEntry:
    """One row of provenance attached to a stored DEPENDS_ON edge.

    Mirrors the shape we persist in ``graph_edges.properties.evidence``.
    Re-verification operates on these directly so the caller doesn't
    have to redo the regex collection step.
    """

    target_repo: str
    file_path: str
    line: int
    snippet: str


async def reverify_evidence_with_llm(
    *,
    source_repo: str,
    evidence: list[EvidenceEntry],
    file_contents: dict[str, str],
    client: Any | None = None,
) -> set[int] | None:
    """Decide which existing-edge evidence rows are still valid.

    Built for the push-webhook path: when changed files overlap with
    previously-recorded evidence, re-verify each affected entry against
    the post-push file content. Returns the set of evidence indices
    (0-based, into the input list) that the LLM keeps as REAL. Indices
    not returned should be treated as dropped — the caller decides
    whether to delete the whole edge (no remaining evidence) or update
    the edge's properties (still has other surviving entries).

    Evidence whose ``file_path`` is not in ``file_contents`` is skipped
    entirely (returned as kept) — the caller chose to only verify a
    subset; an entry pointing at an unchanged file remains trusted.

    Returns ``None`` (NOT an empty set) when the LLM is unavailable.
    Caller distinguishes this from "LLM ran and dropped everything"
    so it can DLQ the push for later retry rather than persist
    deletions based on a missing verifier.
    """
    if not evidence:
        return set()

    # Indices the caller wants verified (file in changed set) vs trusted.
    to_verify_idx: list[int] = []
    trusted_idx: set[int] = set()
    for i, e in enumerate(evidence):
        if e.file_path in file_contents:
            to_verify_idx.append(i)
        else:
            trusted_idx.add(i)

    if not to_verify_idx:
        return trusted_idx

    file_to_lines: dict[str, list[int]] = {}
    for i in to_verify_idx:
        e = evidence[i]
        file_to_lines.setdefault(e.file_path, []).append(e.line)
    file_context = _build_file_context_from_contents(file_contents, file_to_lines)

    # 1-indexed numbering matches the existing classifier so the prompt
    # template is reusable. The mapping back to evidence indices is
    # stable order of `to_verify_idx`.
    listing_lines: list[str] = []
    for n, ev_idx in enumerate(to_verify_idx, start=1):
        e = evidence[ev_idx]
        prior_snippet = e.snippet.replace("\n", " ")[:160]
        listing_lines.append(
            f"{n}. file={e.file_path} line={e.line} target={e.target_repo}\n"
            f"   prior_snippet: {prior_snippet}"
        )
    user_prompt = (
        f"Source repo: {source_repo}\n\n"
        "These are previously-verified architecture-graph edges. The "
        "files below are the CURRENT (post-change) contents. For each "
        "numbered evidence entry, decide if the cited line still "
        "genuinely references the target repo. A line that was deleted, "
        "moved, or changed to point elsewhere is COINCIDENCE; a line "
        "still doing the same job (HTTP call, env var, import, doc "
        "link, etc.) is REAL.\n\n"
        "=== Files (post-change) ===\n\n"
        f"{file_context}\n\n"
        "=== Evidence to re-verify ===\n"
        + "\n".join(listing_lines)
    )

    verdicts = await _call_classifier_llm(
        source_repo=source_repo,
        user_prompt=user_prompt,
        client=client,
        log_prefix="cross_repo_deps.reverify",
    )
    if verdicts is None:
        # LLM unavailable / errored. Signal failure to the caller via
        # None so the push can be DLQ'd for retry. The caller is
        # responsible for the conservative "keep everything for now"
        # behavior in the live edge state — we don't apply silent
        # deletions based on a missing verifier.
        return None

    kept = set(trusted_idx)
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        n = v.get("number")
        if not isinstance(n, int) or n < 1 or n > len(to_verify_idx):
            continue
        if not v.get("real"):
            continue
        kept.add(to_verify_idx[n - 1])
    return kept


async def update_edges_after_push(
    *,
    customer_id: str,
    source_repo: str,
    removed_files: list[str],
    modified_files: list[str],
    file_contents: dict[str, str],
    client: Any | None = None,
) -> dict[str, int]:
    """Re-evaluate this repo's outbound edges after a push.

    Finds evidence whose ``file_path`` is in ``removed_files`` (auto-
    drop) or ``modified_files`` (LLM re-verify). Persists the result:
    update ``properties.evidence`` when an edge survives, delete the
    edge when no evidence remains. New edges introduced by the push
    are NOT discovered here — that's the nightly cross-repo refresh
    pass.

    Returns counts for logging:
      - ``edges_examined`` : edges read from DB before any change
      - ``evidence_dropped``: rows removed across all edges
      - ``edges_deleted``   : edges whose evidence list became empty
      - ``edges_updated``   : edges whose evidence shrank but survived
      - ``verifier_failures``: edges whose LLM call failed; the caller
        should DLQ the push when this is non-zero. Removal-only paths
        (no LLM needed) NEVER bump this counter — they always succeed.
    """
    counts = {
        "edges_examined": 0,
        "evidence_dropped": 0,
        "edges_deleted": 0,
        "edges_updated": 0,
        "verifier_failures": 0,
    }
    if not removed_files and not modified_files:
        return counts

    removed_set = set(removed_files)
    # Convert modified file paths to a content dict keyed identically.
    # `file_contents` may also include added files — caller normalizes.

    async with with_tenant(customer_id) as conn:
        # Pull existing outbound DEPENDS_ON edges + each edge's evidence.
        rows = await conn.fetch(
            """
            SELECT e.edge_id,
                   n_to.canonical_id AS target_repo,
                   e.properties->'evidence' AS evidence
            FROM graph_edges e
            JOIN graph_nodes n_from
                 ON n_from.node_id = e.from_node_id AND n_from.customer_id = e.customer_id
            JOIN graph_nodes n_to
                 ON n_to.node_id = e.to_node_id AND n_to.customer_id = e.customer_id
            WHERE e.customer_id = $1
              AND e.edge_type = 'DEPENDS_ON'
              AND n_from.label = 'Repo'
              AND n_from.canonical_id = $2
              AND n_to.label = 'Repo'
              AND e.valid_to IS NULL
            """,
            customer_id,
            source_repo,
        )

    counts["edges_examined"] = len(rows)
    if not rows:
        return counts

    for row in rows:
        target_repo = row["target_repo"]
        raw_evidence = row["evidence"]
        if isinstance(raw_evidence, (str, bytes, bytearray)):
            try:
                raw_evidence = orjson.loads(raw_evidence)
            except orjson.JSONDecodeError:
                raw_evidence = []
        if not isinstance(raw_evidence, list):
            raw_evidence = []

        # Build typed evidence list, dropping rows whose file was REMOVED.
        evidence: list[EvidenceEntry] = []
        dropped_for_removal = 0
        for e in raw_evidence:
            if not isinstance(e, dict):
                continue
            file_path = str(e.get("file_path") or "")
            if not file_path or file_path in removed_set:
                if file_path in removed_set:
                    dropped_for_removal += 1
                continue
            evidence.append(
                EvidenceEntry(
                    target_repo=target_repo,
                    file_path=file_path,
                    line=int(e.get("line") or 0),
                    snippet=str(e.get("snippet") or ""),
                )
            )

        # Re-verify against modified-file contents only when at least
        # one evidence row needs the LLM. Pure removal-only paths skip
        # the LLM entirely (no verifier_failures risk) — important so
        # that pushes that delete a referencing file always make
        # progress even when Gemini is down.
        needs_llm = any(e.file_path in file_contents for e in evidence)
        if needs_llm:
            kept_indices = await reverify_evidence_with_llm(
                source_repo=source_repo,
                evidence=evidence,
                file_contents=file_contents,
                client=client,
            )
            if kept_indices is None:
                # Verifier failed. Persist removal-only changes (we
                # already filtered evidence above for files in
                # removed_set), keep modified-file evidence as-is, bump
                # the failure counter so the caller can DLQ the push.
                counts["verifier_failures"] += 1
                kept_evidence = list(evidence)
            else:
                kept_evidence = [evidence[i] for i in sorted(kept_indices)]
        else:
            kept_evidence = list(evidence)

        new_dropped = (len(evidence) - len(kept_evidence)) + dropped_for_removal
        counts["evidence_dropped"] += new_dropped

        if not kept_evidence:
            # No evidence survives — drop the edge.
            async with with_tenant(customer_id) as conn:
                await conn.execute(
                    "DELETE FROM graph_edges WHERE edge_id = $1 AND customer_id = $2",
                    row["edge_id"],
                    customer_id,
                )
            counts["edges_deleted"] += 1
        elif new_dropped > 0:
            # Edge survives with reduced evidence — refresh properties.
            new_props = {
                "evidence": [
                    {
                        "file_path": e.file_path,
                        "line": e.line,
                        "snippet": e.snippet,
                    }
                    for e in kept_evidence
                ],
            }
            async with with_tenant(customer_id) as conn:
                await conn.execute(
                    """
                    UPDATE graph_edges
                    SET properties = $1
                    WHERE edge_id = $2 AND customer_id = $3
                    """,
                    orjson.dumps(new_props).decode(),
                    row["edge_id"],
                    customer_id,
                )
            counts["edges_updated"] += 1

    log.info(
        "cross_repo_deps.post_push_update",
        customer=customer_id,
        source_repo=source_repo,
        **counts,
    )
    return counts


def _cap_candidates(
    candidates: list[CandidateMatch], cap: int
) -> list[CandidateMatch]:
    """Diverse-first sampling so the cap retains coverage across pairs.

    Walk candidates in original order but keep at least one per (file,
    target) bucket before accepting a second from any bucket. Caller
    relies on indices for LLM round-trip, so order of the returned list
    must be stable.
    """
    if len(candidates) <= cap:
        return list(candidates)
    bucket_counts: dict[tuple[str, str], int] = {}
    primary: list[CandidateMatch] = []
    overflow: list[CandidateMatch] = []
    for c in candidates:
        key = (c.file_path, c.candidate_target)
        if bucket_counts.get(key, 0) == 0:
            primary.append(c)
            bucket_counts[key] = 1
        else:
            overflow.append(c)
    out = primary[:cap]
    if len(out) < cap:
        out.extend(overflow[: cap - len(out)])
    return out


# ---------------------------------------------------------------------------
# Edge aggregation
# ---------------------------------------------------------------------------


def aggregate_to_edges(
    source_repo: str,
    verified: list[VerifiedMatch],
    *,
    max_evidence_per_pair: int = 5,
) -> list[CrossRepoEdge]:
    """Collapse verified matches into one edge per (source, target) pair."""
    by_pair: dict[str, list[VerifiedMatch]] = {}
    for v in verified:
        by_pair.setdefault(v.target_repo, []).append(v)
    edges: list[CrossRepoEdge] = []
    for target_repo, matches in by_pair.items():
        evidence = [
            {
                "file_path": m.file_path,
                "line": m.line_number,
                "snippet": m.snippet[:160],
                "reason": m.reason,
            }
            for m in matches[:max_evidence_per_pair]
        ]
        edges.append(
            CrossRepoEdge(
                source_repo=source_repo,
                target_repo=target_repo,
                evidence=evidence,
            )
        )
    return edges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def persist_cross_repo_edges(
    customer_id: str,
    source_repo: str,
    edges: list[CrossRepoEdge],
) -> None:
    """Idempotently replace this source repo's outbound DEPENDS_ON edges.

    In a single transaction:

      1. Look up the source repo's ``Repo`` node (create if missing).
      2. Delete ALL existing ``DEPENDS_ON`` edges whose ``from_node`` is
         this source — drops any reference the source repo no longer
         contains, so a removed import disappears from the diagram.
      3. For each new edge: look up / create the target ``Repo`` node,
         insert the edge with provenance in ``properties``.

    Runs in its own transaction (separate from the main code-graph
    extract / persist) so a failure here does not roll back symbol
    extraction. Cross-repo edges are advisory data for the wiki
    architecture diagram; partial state is acceptable.
    """
    async with with_tenant(customer_id) as conn, conn.transaction():
        source_node_id = await _get_or_create_repo_node(
            conn, customer_id, source_repo
        )
        await conn.execute(
            """
                DELETE FROM graph_edges
                WHERE customer_id = $1
                  AND edge_type = 'DEPENDS_ON'
                  AND from_node_id = $2
                """,
            customer_id,
            source_node_id,
        )
        for edge in edges:
            target_node_id = await _get_or_create_repo_node(
                conn, customer_id, edge.target_repo
            )
            if target_node_id == source_node_id:
                # Self-reference (variants matched the source repo's
                # own name in its own files). Drop silently.
                continue
            await conn.execute(
                """
                    INSERT INTO graph_edges
                        (customer_id, edge_type, from_node_id, to_node_id,
                         properties, source_system, confidence)
                    VALUES ($1, 'DEPENDS_ON', $2, $3, $4, 'code_graph', 'EXTRACTED')
                    ON CONFLICT (customer_id, edge_type, from_node_id, to_node_id)
                    DO UPDATE SET properties = EXCLUDED.properties
                    """,
                customer_id,
                source_node_id,
                target_node_id,
                orjson.dumps({"evidence": edge.evidence}).decode(),
            )


async def _get_or_create_repo_node(
    conn: asyncpg.Connection,
    customer_id: str,
    repo: str,
) -> int:
    """Find or create a ``Repo`` graph node for ``repo`` (``owner/name``)."""
    row = await conn.fetchrow(
        """
        INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
        VALUES ($1, 'Repo', $2, $3)
        ON CONFLICT (customer_id, label, canonical_id)
        DO UPDATE SET updated_at = NOW()
        RETURNING node_id
        """,
        customer_id,
        repo,
        orjson.dumps({"name": repo.rsplit("/", 1)[-1]}).decode(),
    )
    if row is None:
        # Should not happen given RETURNING, but cast to int defensively.
        raise RuntimeError(f"failed to upsert Repo node for {repo!r}")
    return int(row["node_id"])


async def list_other_known_repos(customer_id: str, source_repo: str) -> list[str]:
    """All other repos this customer has previously code-graph-extracted.

    Used as the candidate target list for cross-repo dep extraction. The
    first repo a customer ever ingests has no candidates and skips this
    pass entirely; subsequent repos discover edges progressively.
    """
    async with with_tenant(customer_id) as conn:
        rows: list[asyncpg.Record] = await conn.fetch(
            """
            SELECT DISTINCT repo
            FROM code_repo_state
            WHERE customer_id = $1 AND repo <> $2
            """,
            customer_id,
            source_repo,
        )
    return [r["repo"] for r in rows]


async def extract_cross_repo_deps(
    *,
    customer_id: str,
    source_repo: str,
    target_dir: Path,
    client: Any | None = None,
) -> list[CrossRepoEdge]:
    """Per-repo entry point. Returns aggregated outbound edges.

    Caller persists these via graph_edges. Idempotency requirement: the
    caller MUST delete pre-existing ``DEPENDS_ON`` edges from this
    source repo's node before persisting new ones; otherwise stale
    references linger forever after a repo drops a dep.
    """
    other_repos = await list_other_known_repos(customer_id, source_repo)
    if not other_repos:
        log.info(
            "cross_repo_deps.no_other_repos",
            customer=customer_id,
            source_repo=source_repo,
        )
        return []
    targets = [(r, repo_name_variants(r)) for r in other_repos]
    candidates = await asyncio.to_thread(collect_candidates, target_dir, targets)
    if not candidates:
        return []
    log.info(
        "cross_repo_deps.candidates_collected",
        customer=customer_id,
        source_repo=source_repo,
        count=len(candidates),
    )
    verified = await classify_with_llm(
        source_repo=source_repo,
        candidates=candidates,
        target_dir=target_dir,
        client=client,
    )
    log.info(
        "cross_repo_deps.classified",
        customer=customer_id,
        source_repo=source_repo,
        candidates=len(candidates),
        verified=len(verified),
    )
    return aggregate_to_edges(source_repo, verified)


# ---------------------------------------------------------------------------
# DLQ (deferred verifier failures)
# ---------------------------------------------------------------------------


# Cap retry attempts so a permanently-broken token / sha doesn't loop
# forever. Three nightly tries is a week of opportunity.
_DLQ_MAX_ATTEMPTS = 3


async def enqueue_reverify_dlq(
    *,
    customer_id: str,
    source_repo: str,
    sha: str,
    removed_files: list[str],
    modified_files: list[str],
    integration_token_id: str | None,
    error: str | None = None,
) -> int:
    """Park a failed verifier run for later retry. Returns the dlq_id.

    Idempotency: do NOT collapse on (customer, repo, sha) — a single
    sha may arrive via multiple webhooks (rebase, force-push) and each
    one's file lists matter. Let the retry pass deduplicate by SHA in
    its own SELECT if needed.
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO cross_repo_reverify_dlq
                (customer_id, source_repo, sha, removed_files, modified_files,
                 integration_token_id, status, last_error)
            VALUES ($1, $2, $3, $4, $5, $6, 'pending', $7)
            RETURNING dlq_id
            """,
            customer_id,
            source_repo,
            sha,
            orjson.dumps(removed_files).decode(),
            orjson.dumps(modified_files).decode(),
            integration_token_id,
            error,
        )
    if row is None:
        raise RuntimeError("DLQ insert returned no row")
    log.info(
        "cross_repo_deps.dlq_enqueued",
        customer=customer_id,
        source_repo=source_repo,
        dlq_id=row["dlq_id"],
        modified_count=len(modified_files),
        removed_count=len(removed_files),
    )
    return int(row["dlq_id"])


async def drain_reverify_dlq(
    *,
    fetch_files_at_sha: Any,
    resolve_token: Any,
    max_rows: int = 100,
) -> dict[str, int]:
    """Replay pending DLQ rows. Used by the nightly cron.

    `fetch_files_at_sha` and `resolve_token` are passed as callables to
    keep this module free of an ingestion-handlers import cycle. The
    nightly cron module wires them up.

    For each pending row:
      1. Resolve the token (from integration_token_id) via `resolve_token`
      2. Fetch modified file contents at the recorded sha via
         `fetch_files_at_sha`
      3. Re-run `update_edges_after_push`
      4. Mark `done` on success, increment attempts on retry-able
         failure, mark `failed` after _DLQ_MAX_ATTEMPTS strikes

    Returns counts: ``processed``, ``done``, ``retry``, ``failed``.
    """
    summary = {"processed": 0, "done": 0, "retry": 0, "failed": 0}
    # Read all pending rows up to the cap. We don't FOR UPDATE SKIP
    # LOCKED here — only one cron machine drains the DLQ, contention
    # is a non-issue.
    pending: list[asyncpg.Record] = []
    async with _raw_pool_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT dlq_id, customer_id, source_repo, sha, removed_files,
                   modified_files, integration_token_id, attempts
            FROM cross_repo_reverify_dlq
            WHERE status = 'pending'
              AND attempts < $1
            ORDER BY enqueued_at ASC
            LIMIT $2
            """,
            _DLQ_MAX_ATTEMPTS,
            max_rows,
        )
        pending = list(rows)

    for row in pending:
        summary["processed"] += 1
        dlq_id = int(row["dlq_id"])
        customer_id = row["customer_id"]
        source_repo = row["source_repo"]
        sha = row["sha"]
        token_id = row["integration_token_id"]

        await _mark_dlq_processing(dlq_id)

        removed_raw = row["removed_files"]
        modified_raw = row["modified_files"]
        removed_files = (
            orjson.loads(removed_raw)
            if isinstance(removed_raw, (str, bytes, bytearray))
            else (removed_raw or [])
        )
        modified_files = (
            orjson.loads(modified_raw)
            if isinstance(modified_raw, (str, bytes, bytearray))
            else (modified_raw or [])
        )

        try:
            token = await resolve_token(customer_id, token_id) if token_id else None
            file_contents: dict[str, str] = {}
            if modified_files:
                fetched = await fetch_files_at_sha(
                    repo=source_repo,
                    sha=sha,
                    paths=modified_files,
                    token=token,
                    customer_id=customer_id,
                )
                for f in fetched:
                    if not getattr(f, "not_found", False):
                        file_contents[f.rel_path] = f.content

            counts = await update_edges_after_push(
                customer_id=customer_id,
                source_repo=source_repo,
                removed_files=removed_files,
                modified_files=list(file_contents.keys()),
                file_contents=file_contents,
            )

            if counts["verifier_failures"] > 0:
                # LLM still down — bump attempts, leave row pending.
                await _bump_dlq_attempts(
                    dlq_id, error="verifier_failures persisting"
                )
                summary["retry"] += 1
            else:
                await _mark_dlq_done(dlq_id)
                summary["done"] += 1
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc!s}"[:480]
            await _bump_dlq_attempts(dlq_id, error=err)
            log.warning(
                "cross_repo_deps.dlq_retry_error",
                customer=customer_id,
                dlq_id=dlq_id,
                error=err,
            )
            summary["retry"] += 1

    # Promote attempts==MAX still-pending rows to 'failed' so they stop
    # showing up in the next run's pending scan.
    async with _raw_pool_conn() as conn:
        promoted = await conn.execute(
            """
            UPDATE cross_repo_reverify_dlq
            SET status = 'failed'
            WHERE status = 'pending' AND attempts >= $1
            """,
            _DLQ_MAX_ATTEMPTS,
        )
        # promoted is "UPDATE n"; not strictly needed for caller
        log.info(
            "cross_repo_deps.dlq_drain_done",
            promoted_to_failed=promoted,
            **summary,
        )
        summary["failed"] = _parse_update_count(promoted)

    return summary


async def _mark_dlq_processing(dlq_id: int) -> None:
    async with _raw_pool_conn() as conn:
        await conn.execute(
            """
            UPDATE cross_repo_reverify_dlq
            SET status = 'processing', last_attempt_at = NOW()
            WHERE dlq_id = $1
            """,
            dlq_id,
        )


async def _mark_dlq_done(dlq_id: int) -> None:
    async with _raw_pool_conn() as conn:
        await conn.execute(
            """
            UPDATE cross_repo_reverify_dlq
            SET status = 'done', last_attempt_at = NOW()
            WHERE dlq_id = $1
            """,
            dlq_id,
        )


async def _bump_dlq_attempts(dlq_id: int, *, error: str) -> None:
    async with _raw_pool_conn() as conn:
        await conn.execute(
            """
            UPDATE cross_repo_reverify_dlq
            SET status = 'pending',
                attempts = attempts + 1,
                last_error = $2,
                last_attempt_at = NOW()
            WHERE dlq_id = $1
            """,
            dlq_id,
            error,
        )


async def _raw_pool_conn():
    """Connection helper for the DLQ table. Uses the standard pool with
    no tenant GUC because cross_repo_reverify_dlq has no RLS — it's
    cross-customer admin data, same convention as ingestion_queue.
    """
    from shared.db import raw_conn

    return raw_conn()


def _parse_update_count(stmt_status: str | None) -> int:
    """Extract the row-count from asyncpg's `UPDATE n` status string."""
    if not stmt_status:
        return 0
    parts = stmt_status.split()
    if len(parts) >= 2 and parts[0] == "UPDATE":
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


__all__ = [
    "CandidateMatch",
    "CrossRepoEdge",
    "EvidenceEntry",
    "VerifiedMatch",
    "aggregate_to_edges",
    "classify_with_llm",
    "collect_candidates",
    "drain_reverify_dlq",
    "enqueue_reverify_dlq",
    "extract_cross_repo_deps",
    "list_other_known_repos",
    "list_tracked_files",
    "persist_cross_repo_edges",
    "repo_name_variants",
    "reverify_evidence_with_llm",
    "update_edges_after_push",
]
