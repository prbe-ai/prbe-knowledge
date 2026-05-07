"""CodeGraph connector — symbol-level structural ingestion of git repos.

Reads synthetic events written by `services/ingestion/code_graph/bridge.py`,
dispatches by `kind`:

  initial_backfill  — shallow-clone + walk + extract via the pipeline.
  incremental       — fetch changed files via Contents API; diff-extract
                      against code_repo_state cache.
  disconnect        — soft-delete code.symbol Documents for the affected
                      repos; close graph_node_provenance for code_graph.

verify_signature is a no-op return-True — events arrive only from the
internal bridge (which is invoked from already-authenticated source
connector code, e.g., handlers/github.py after HMAC-verified push).
There is no public webhook surface for code_graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from services.ingestion.code_graph.clone import (
    FileEntry,
    prune_scratch,
    repo_dir,
    shallow_clone,
    walk_files,
)
from services.ingestion.code_graph.cross_repo_deps import (
    extract_cross_repo_deps,
    persist_cross_repo_edges,
)
from services.ingestion.code_graph.fetch import fetch_files_at_sha
from services.ingestion.code_graph.pipeline import extract_files_to_result
from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from shared.backend_client import fetch_github_installation_token
from shared.constants import SourceSystem
from shared.db import with_tenant
from shared.exceptions import (
    GitHubAuthError,
    InvalidWebhookPayload,
)
from shared.logging import get_logger
from shared.models import (
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

log = get_logger(__name__)

KIND_INITIAL_BACKFILL = "initial_backfill"
KIND_INCREMENTAL = "incremental"
KIND_DISCONNECT = "disconnect"
_KNOWN_KINDS = frozenset({KIND_INITIAL_BACKFILL, KIND_INCREMENTAL, KIND_DISCONNECT})

# Languages we extract today. Used to filter the worktree walk so we
# don't bother reading files we'd just throw away. Mirrors the union of
# `Extractor.file_extensions` from registered extractors.
_SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".go",
    ".java",
)


@register_connector(SourceSystem.CODE_GRAPH)
class CodeGraphConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.CODE_GRAPH
    display_name: ClassVar[str] = "code-graph"

    # ---- 1. signature verification --------------------------------------

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        return True

    # ---- 2. event parsing -----------------------------------------------

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        kind = raw_payload.get("kind")
        if kind not in _KNOWN_KINDS:
            raise InvalidWebhookPayload(
                f"unknown code_graph payload kind: {kind!r}"
            )
        return WebhookParseResult(
            source_event_id=_recompute_event_id(raw_payload),
            received_at=datetime.now(UTC),
        )

    # ---- 3. hydration ---------------------------------------------------

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: IntegrationToken | None,
    ) -> dict[str, Any]:
        # `_load_token` returns None for code_graph (we don't store a
        # code_graph token row). The actual auth is the GitHub App
        # installation token; fetch it from prbe-backend.
        kind = event.raw_payload.get("kind")
        if kind == KIND_DISCONNECT:
            return {}
        try:
            gh_token, _expires = await fetch_github_installation_token(
                self.http, customer_id=event.customer_id
            )
        except GitHubAuthError as exc:
            log.warning(
                "code_graph.fetch_supplementary.github_auth_failed",
                customer=event.customer_id,
                error=str(exc),
            )
            return {"github_token": None}
        return {"github_token": gh_token}

    # ---- 4. normalization -----------------------------------------------

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        kind = event.raw_payload.get("kind")
        if kind == KIND_INITIAL_BACKFILL:
            return await self._normalize_initial_backfill(event, hydrated)
        if kind == KIND_INCREMENTAL:
            return await self._normalize_incremental(event, hydrated)
        if kind == KIND_DISCONNECT:
            return await self._normalize_disconnect(event)
        raise InvalidWebhookPayload(
            f"unknown code_graph payload kind: {kind!r}"
        )

    # ---- backfill -------------------------------------------------------

    async def _normalize_initial_backfill(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        repo: str = event.raw_payload.get("repo", "")
        sha: str = event.raw_payload.get("sha", "HEAD")
        if not repo:
            raise InvalidWebhookPayload(
                "code_graph initial_backfill missing 'repo'"
            )

        token: str | None = hydrated.get("github_token")
        target_dir = repo_dir(event.customer_id, repo)

        await shallow_clone(repo=repo, sha=sha, token=token, target_dir=target_dir)

        files: list[FileEntry] = []
        async for entry in walk_files(target_dir, _SUPPORTED_EXTENSIONS):
            files.append(entry)

        log.info(
            "code_graph.backfill.walked",
            customer=event.customer_id,
            repo=repo,
            files=len(files),
        )

        result = await extract_files_to_result(
            customer_id=event.customer_id,
            repo=repo,
            sha=sha,
            files=files,
        )

        # Cross-repo dependency edges. Runs after symbol extraction so
        # the repo's tracked files are still on disk, but BEFORE
        # prune_scratch. Persisted via a separate transaction (its own
        # idempotent delete-then-insert) so a failure in this advisory
        # path does not roll back the symbol-graph work above. Skips
        # the call entirely when no other repos exist yet for this
        # customer (first-ever code-graph backfill).
        try:
            cross_repo_edges = await extract_cross_repo_deps(
                customer_id=event.customer_id,
                source_repo=repo,
                target_dir=target_dir,
            )
            if cross_repo_edges:
                await persist_cross_repo_edges(
                    customer_id=event.customer_id,
                    source_repo=repo,
                    edges=cross_repo_edges,
                )
        except Exception as exc:
            log.warning(
                "code_graph.cross_repo_deps_failed",
                customer=event.customer_id,
                repo=repo,
                error=str(exc),
                error_class=type(exc).__name__,
            )

        # Backfill is a single-shot in PR-A. Prune the clone scratch dir
        # now that extraction is complete; incremental updates fetch via
        # Contents API and don't need the worktree.
        prune_scratch(event.customer_id, repo)
        return result

    # ---- incremental ----------------------------------------------------

    async def _normalize_incremental(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        repo: str = event.raw_payload.get("repo", "")
        sha: str = event.raw_payload.get("sha", "")
        added: list[str] = list(event.raw_payload.get("files_added", []) or [])
        modified: list[str] = list(event.raw_payload.get("files_modified", []) or [])
        removed: list[str] = list(event.raw_payload.get("files_removed", []) or [])
        token: str | None = hydrated.get("github_token")

        if not repo or not sha:
            raise InvalidWebhookPayload(
                "code_graph incremental missing 'repo' or 'sha'"
            )

        # Fetch added + modified files via Contents API. Filter to supported
        # extensions first so we don't waste API budget on unindexable files.
        to_fetch = [
            p
            for p in (added + modified)
            if any(p.endswith(ext) for ext in _SUPPORTED_EXTENSIONS)
        ]

        files: list[FileEntry] = []
        if to_fetch:
            fetched = await fetch_files_at_sha(
                repo=repo,
                sha=sha,
                paths=to_fetch,
                token=token,
                customer_id=event.customer_id,
            )
            for f in fetched:
                if f.not_found:
                    # File listed in payload but absent at SHA — push race;
                    # treat as a removal.
                    removed.append(f.rel_path)
                    continue
                files.append(FileEntry(rel_path=f.rel_path, content=f.content))

        result = await extract_files_to_result(
            customer_id=event.customer_id,
            repo=repo,
            sha=sha,
            files=files,
        )

        if removed:
            tombstones = await self._build_removed_documents(
                event.customer_id, repo, removed
            )
            result.documents.extend(tombstones)
        return result

    async def _build_removed_documents(
        self,
        customer_id: str,
        repo: str,
        removed_paths: list[str],
    ) -> list:
        """Look up live code.symbol Documents for `removed_paths` and
        return tombstone copies (deleted_at = NOW()).

        Uses an in-list LIKE-prefix scan: doc_id starts with
        `code_graph:<repo>:<file_path>:`.
        """
        if not removed_paths:
            return []

        tombstones: list = []
        now = datetime.now(UTC)

        async with with_tenant(customer_id) as conn:
            for path in removed_paths:
                prefix = f"code_graph:{repo}:{path}:"
                rows = await conn.fetch(
                    """
                    SELECT doc_id, version, source_id, source_url, doc_class,
                           doc_type, content_type, language, content_hash,
                           title, body_preview, body_size_bytes, body_token_count,
                           author_id, created_at, updated_at, valid_from,
                           ingested_at, parent_doc_id, supersedes_doc_id,
                           acl, metadata, entities, attachments, doc_references,
                           normalizer_version
                    FROM documents
                    WHERE customer_id = $1
                      AND doc_id LIKE $2
                      AND valid_to IS NULL
                      AND deleted_at IS NULL
                    """,
                    customer_id,
                    prefix + "%",
                )
                for r in rows:
                    tombstones.append(_row_to_tombstone(r, customer_id, now))
        log.info(
            "code_graph.incremental.removed",
            customer=customer_id,
            repo=repo,
            paths=len(removed_paths),
            tombstoned=len(tombstones),
        )
        return tombstones

    # ---- disconnect -----------------------------------------------------

    async def _normalize_disconnect(
        self,
        event: WebhookEvent,
    ) -> NormalizationResult:
        repos = list(event.raw_payload.get("repos", []) or [])
        if not repos:
            raise InvalidWebhookPayload(
                "code_graph disconnect missing 'repos'"
            )

        # Disconnect is a bulk operation — we deliberately bend the
        # "Connector should never directly touch the DB" contract here
        # because emitting one Document per symbol for a 10k-symbol repo
        # would balloon Phase B to a multi-minute write txn. Instead we
        # do bulk SQL UPDATEs inside a tenant-scoped txn, then return an
        # empty NormalizationResult with no further persist work needed.
        async with with_tenant(event.customer_id) as conn:
            for repo in repos:
                prefix = f"code_graph:{repo}:"
                # Tombstone live Documents.
                await conn.execute(
                    """
                    UPDATE documents
                    SET deleted_at = NOW(), valid_to = NOW()
                    WHERE customer_id = $1
                      AND source_system = $2
                      AND doc_id LIKE $3
                      AND valid_to IS NULL
                    """,
                    event.customer_id,
                    SourceSystem.CODE_GRAPH.value,
                    prefix + "%",
                )
                # Mark live chunks stale.
                await conn.execute(
                    """
                    UPDATE chunks
                    SET valid_to = NOW()
                    WHERE customer_id = $1
                      AND valid_to IS NULL
                      AND doc_id LIKE $2
                    """,
                    event.customer_id,
                    prefix + "%",
                )
                # Drop code_repo_state for this repo so a future reconnect
                # re-extracts cleanly.
                await conn.execute(
                    """
                    DELETE FROM code_repo_state
                    WHERE customer_id = $1 AND repo = $2
                    """,
                    event.customer_id,
                    repo,
                )
                # Close graph_node_provenance for code_graph on the affected
                # nodes. Anchor on graph_nodes.canonical_id (which we wrote
                # as `<repo>:<qname>` or `<repo>:<file_path>`); a substring
                # match against `qualified_name` would tombstone unrelated
                # nodes when one repo's name is a prefix/substring of
                # another's (acme/app vs acme/app-staging).
                await conn.execute(
                    """
                    UPDATE graph_node_provenance gnp
                    SET last_seen_at = NOW()
                    FROM graph_nodes n
                    WHERE gnp.node_id = n.node_id
                      AND gnp.customer_id = $1
                      AND gnp.source_system = $2
                      AND n.canonical_id LIKE $3
                    """,
                    event.customer_id,
                    SourceSystem.CODE_GRAPH.value,
                    f"{repo}:%",
                )

        log.info(
            "code_graph.disconnect.done",
            customer=event.customer_id,
            repos=len(repos),
        )
        # All writes happened above via raw SQL — return a result the
        # normalizer will treat as a deliberate no-op. is_empty + a
        # skipped_reason short-circuits to DuplicateEventIgnored at
        # normalizer.py, which marks the queue row completed instead of
        # raising NormalizationError.
        return NormalizationResult(
            skipped_reason="code_graph.disconnect.bulk_applied",
        )


def _recompute_event_id(payload: Mapping[str, Any]) -> str:
    kind = payload["kind"]
    if kind == KIND_DISCONNECT:
        repos = payload.get("repos") or []
        repos_label = "+".join(sorted(str(r) for r in repos))[:200]
        ts = payload.get("enqueued_at", "")
        return f"code_graph:disconnect:{repos_label}:{ts}"
    repo = payload.get("repo", "")
    sha = payload.get("sha", "")
    if kind == KIND_INITIAL_BACKFILL:
        return f"code_graph:backfill:{repo}:{sha}"
    return f"code_graph:incremental:{repo}:{sha}"


def _row_to_tombstone(row, customer_id: str, now: datetime):
    """Construct a Document with deleted_at=now from an existing DB row.

    Used by incremental's removed-files path to emit tombstones the
    normalizer's _persist + chunk-soft-delete cascade can chew through.
    """
    import orjson

    from shared.models import ACLSnapshot, Document  # local import to avoid cycle

    def _decode(field: str):
        v = row[field]
        return orjson.loads(v) if isinstance(v, (str, bytes)) else v

    acl_dict = _decode("acl")
    metadata = _decode("metadata")
    entities = _decode("entities")
    attachments = _decode("attachments")
    doc_references = _decode("doc_references")

    return Document(
        doc_id=row["doc_id"],
        customer_id=customer_id,
        version=row["version"],
        source_system=SourceSystem.CODE_GRAPH,
        source_id=row["source_id"],
        source_url=row["source_url"],
        doc_class=row["doc_class"],
        doc_type=row["doc_type"],
        content_type=row["content_type"],
        language=row["language"],
        content_hash=row["content_hash"],
        title=row["title"],
        body_preview=row["body_preview"],
        body_size_bytes=row["body_size_bytes"],
        body_token_count=row["body_token_count"],
        author_id=row["author_id"],
        created_at=row["created_at"],
        updated_at=now,
        valid_from=row["valid_from"],
        deleted_at=now,
        ingested_at=now,
        parent_doc_id=row["parent_doc_id"],
        supersedes_doc_id=row["supersedes_doc_id"],
        acl=ACLSnapshot.model_validate(acl_dict),
        metadata=metadata,
        entities=entities,
        attachments=attachments,
        doc_references=doc_references,
        normalizer_version=row["normalizer_version"],
    )
