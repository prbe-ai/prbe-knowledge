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


def _build_file_context(
    target_dir: Path,
    file_to_match_lines: dict[str, list[int]],
) -> str:
    """Render the file-context block fed to the LLM.

    For each file with matches: full file if small, else windowed slices
    around each match line. Files appear in the same order as the
    candidate list so line references stay stable.
    """
    parts: list[str] = []
    used_bytes = 0
    for rel_path, match_lines in file_to_match_lines.items():
        if used_bytes >= _MAX_PROMPT_CONTENT_BYTES:
            break
        full_path = target_dir / rel_path
        body = _read_file_safely(full_path)
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
            # Truncate this block to fit the budget; better to give partial
            # context than to drop the file entirely.
            allowed = _MAX_PROMPT_CONTENT_BYTES - used_bytes
            block = block.encode("utf-8")[:allowed].decode("utf-8", errors="ignore")
            parts.append(block)
            used_bytes += len(block.encode("utf-8"))
            break
        parts.append(block)
        used_bytes += block_bytes
    return "\n".join(parts)


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

    if client is None:
        try:
            from google import genai

            from shared.config import get_settings

            api_key = get_settings().google_api_key.get_secret_value()
            if not api_key:
                log.warning("cross_repo_deps.no_google_api_key", source_repo=source_repo)
                return []
            client = genai.Client(api_key=api_key)
        except ImportError as exc:
            log.warning("cross_repo_deps.google_genai_missing", error=str(exc))
            return []

    # Cross-repo classification is a cheap pre-filter, not a triage call.
    # Hard-coded to Flash Lite regardless of WIKI_TRIAGE_MODEL — flipping
    # triage to Haiku/Anthropic should not bring this Anthropic-bill-flavored
    # call along for the ride. Edit here if Flash Lite ever stops being
    # the right tool.
    model_name = "gemini-flash-lite-preview"

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
            "cross_repo_deps.gemini_failed",
            source_repo=source_repo,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        return []

    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        log.warning("cross_repo_deps.empty_response", source_repo=source_repo)
        return []

    try:
        payload = orjson.loads(text)
    except orjson.JSONDecodeError:
        log.warning(
            "cross_repo_deps.malformed_response",
            source_repo=source_repo,
            preview=text[:200],
        )
        return []

    verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
    if not isinstance(verdicts, list):
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


__all__ = [
    "CandidateMatch",
    "CrossRepoEdge",
    "VerifiedMatch",
    "aggregate_to_edges",
    "classify_with_llm",
    "collect_candidates",
    "extract_cross_repo_deps",
    "list_other_known_repos",
    "list_tracked_files",
    "persist_cross_repo_edges",
    "repo_name_variants",
]
