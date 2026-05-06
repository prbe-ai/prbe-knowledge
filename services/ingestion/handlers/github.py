"""GitHub connector — pull requests, issues, commits, reviews, and CODEOWNERS.

Covers these webhook event types in Phase 0:
- `pull_request` (opened/edited/synchronize/closed/reopened) → DocType.GITHUB_PULL_REQUEST
- `issues` (opened/edited/closed/reopened) → DocType.GITHUB_ISSUE
- `push` → one DocType.GITHUB_COMMIT per commit; if CODEOWNERS changed, also
  emits a DocType.GITHUB_CODEOWNERS doc with the parsed ownership map and
  per-pattern OWNS graph edges.
- `pull_request_review` (submitted) → DocType.GITHUB_REVIEW

Signature: X-Hub-Signature-256 (HMAC-SHA256 of raw body with
`settings.github_webhook_secret`). Dev bypass when the secret is None and the
environment is local — matches the Slack connector pattern.

ACL: Phase 0 captures the repo as the resource (`github.repository`,
`<owner>/<repo>`) and the owner login as the workspace-level principal with
READ permission. Repo `visibility` goes into metadata so the retrieval layer
can tell public vs private apart without re-fetching.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from services.ingestion.chunker import count_tokens
from services.ingestion.code_graph import bridge as code_graph_bridge
from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from shared.backend_client import fetch_github_installation_token
from shared.constants import (
    GITHUB_INSTALLATION_SCOPE_PREFIX,
    DocClass,
    DocType,
    EdgeType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    RefType,
    SourceSystem,
)
from shared.customer_prefs import code_graph_indexed_branch
from shared.exceptions import (
    InvalidWebhookPayload,
    NotSupportedByConnector,
)
from shared.logging import get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    DocRef,
    Document,
    ExternalWorkspaceRef,
    GraphEdgeSpec,
    GraphNodeSpec,
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

log = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_SIGNATURE_PREFIX = "sha256="

# X-GitHub-Event header values we care about.
_EVENT_PULL_REQUEST = "pull_request"
_EVENT_ISSUES = "issues"
_EVENT_PUSH = "push"
_EVENT_PR_REVIEW = "pull_request_review"
# GitHub App lifecycle events — wired into the code-graph bridge so an
# install/uninstall/repo-add/repo-remove keeps the symbol graph in sync.
_EVENT_INSTALLATION = "installation"
_EVENT_INSTALLATION_REPOSITORIES = "installation_repositories"

# Actions we care about per event type. `deleted` produces a tombstone document.
# GitHub also uses `transferred` when an issue moves repos — treat that as a delete
# of the original (the target repo will fire its own `opened` webhook).
_PR_ACTIONS = frozenset(
    {"opened", "edited", "synchronize", "closed", "reopened", "deleted"}
)
_ISSUE_ACTIONS = frozenset(
    {"opened", "edited", "closed", "reopened", "deleted", "transferred"}
)
_REVIEW_ACTIONS = frozenset({"submitted"})
_DELETE_ACTIONS = frozenset({"deleted", "transferred"})

# Paths that trigger CODEOWNERS reparse. GitHub checks these in order.
_CODEOWNERS_PATHS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")

# Resource type strings for ACL snapshots (stable, not enum candidates).
_ACL_RESOURCE_REPO = "github.repository"

# Inline reference syntax.
_SAME_REPO_REF = re.compile(r"(?<![\w/])#(\d+)\b")
_CROSS_REPO_REF = re.compile(r"\b([\w.-]+/[\w.-]+)#(\d+)\b")

# Git "Co-authored-by:" trailer. Convention is one trailer per line in the
# message footer; the email is the identity key. Match is case-insensitive
# on the trailer key only — name and email retain their original casing for
# display, then we lowercase the email for dedup since RFC 5321 treats the
# local-part as case-sensitive but real-world SMTP routing doesn't.
_COAUTHOR_TRAILER = re.compile(
    r"^\s*co-authored-by:\s*(?P<name>[^<\n]+?)\s*<(?P<email>[^>\n]+)>\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@register_connector(SourceSystem.GITHUB)
class GitHubConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.GITHUB
    display_name: ClassVar[str] = "GitHub"

    # ------------------------------------------------------------------
    # 1. signature verification
    # ------------------------------------------------------------------

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        secret = self.settings.github_webhook_secret
        if secret is None:
            # Dev mode: accept unsigned payloads only when running locally.
            return self.settings.is_local

        sig = _header(headers, "x-hub-signature-256")
        if not sig or not sig.startswith(_SIGNATURE_PREFIX):
            return False

        expected = (
            _SIGNATURE_PREFIX
            + hmac.new(
                secret.get_secret_value().encode(),
                raw_body,
                hashlib.sha256,
            ).hexdigest()
        )
        return hmac.compare_digest(expected, sig)

    # ------------------------------------------------------------------
    # 2. event parsing
    # ------------------------------------------------------------------

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        event_type = _header(headers, "x-github-event")
        if event_type is None:
            raise InvalidWebhookPayload("github payload missing X-GitHub-Event header")

        # Installation lifecycle events arrive without `repository` — handled
        # before the repository-required check below.
        if event_type == _EVENT_INSTALLATION:
            return self._parse_installation(raw_payload)
        if event_type == _EVENT_INSTALLATION_REPOSITORIES:
            return self._parse_installation_repositories(raw_payload)

        repo = raw_payload.get("repository")
        if not isinstance(repo, dict):
            # Events like `ping` lack a repo; treat them as ignorable noise.
            return None
        full_name = repo.get("full_name")
        if not isinstance(full_name, str) or not full_name:
            raise InvalidWebhookPayload("github payload missing repository.full_name")

        if event_type == _EVENT_PULL_REQUEST:
            return self._parse_pull_request(full_name, raw_payload)
        if event_type == _EVENT_ISSUES:
            return self._parse_issue(full_name, raw_payload)
        if event_type == _EVENT_PUSH:
            return self._parse_push(full_name, raw_payload)
        if event_type == _EVENT_PR_REVIEW:
            return self._parse_review(full_name, raw_payload)

        # Everything else (watch, star, fork, check_run, ...) is Phase 0 noise.
        return None

    def _parse_pull_request(
        self, full_name: str, raw_payload: Mapping[str, Any]
    ) -> WebhookParseResult | None:
        action = raw_payload.get("action")
        if action not in _PR_ACTIONS:
            return None

        pr = raw_payload.get("pull_request")
        if not isinstance(pr, dict):
            raise InvalidWebhookPayload("pull_request event missing pull_request object")

        number = pr.get("number")
        updated_at = pr.get("updated_at")
        if number is None or not updated_at:
            raise InvalidWebhookPayload("pull_request missing number/updated_at")

        # GitHub's `updated_at` is per-second; two distinct rapid actions
        # (bot synchronize + bot synchronize, label add + label add) within
        # the same second collide on UNIQUE and the second is silently
        # dropped. Append a payload fingerprint to disambiguate while
        # staying stable across true webhook retries.
        source_event_id = (
            f"pr:{full_name}:{number}:{action}:{updated_at}:{_payload_fp(pr)}"
        )
        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=_parse_iso8601(updated_at),
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "event_type": _EVENT_PULL_REQUEST,
                "action": action,
                "repo": full_name,
                "number": number,
            },
        )

    def _parse_issue(
        self, full_name: str, raw_payload: Mapping[str, Any]
    ) -> WebhookParseResult | None:
        action = raw_payload.get("action")
        if action not in _ISSUE_ACTIONS:
            return None

        issue = raw_payload.get("issue")
        if not isinstance(issue, dict):
            raise InvalidWebhookPayload("issues event missing issue object")

        number = issue.get("number")
        updated_at = issue.get("updated_at")
        if number is None or not updated_at:
            raise InvalidWebhookPayload("issue missing number/updated_at")

        source_event_id = (
            f"issue:{full_name}:{number}:{action}:{updated_at}:{_payload_fp(issue)}"
        )
        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=_parse_iso8601(updated_at),
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "event_type": _EVENT_ISSUES,
                "action": action,
                "repo": full_name,
                "number": number,
            },
        )

    def _parse_push(
        self, full_name: str, raw_payload: Mapping[str, Any]
    ) -> WebhookParseResult | None:
        head = raw_payload.get("head_commit")
        if not isinstance(head, dict):
            # Branch deletions send push events with a null head_commit — skip.
            return None

        head_id = head.get("id")
        if not isinstance(head_id, str) or not head_id:
            raise InvalidWebhookPayload("push event missing head_commit.id")

        timestamp = head.get("timestamp") or datetime.now(UTC).isoformat()
        touches_codeowners = _push_touches_codeowners(raw_payload)

        # Force-push that resets back to a previous head_id would otherwise
        # collide with the original push event. Include the timestamp +
        # commits-array fingerprint so the revert lands as a distinct row.
        commits_fp = _payload_fp(raw_payload.get("commits") or [])
        source_event_id = f"push:{full_name}:{head_id}:{timestamp}:{commits_fp}"
        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=_parse_iso8601(timestamp),
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "event_type": _EVENT_PUSH,
                "repo": full_name,
                "head_commit_id": head_id,
                "touches_codeowners": touches_codeowners,
            },
        )

    def _parse_review(
        self, full_name: str, raw_payload: Mapping[str, Any]
    ) -> WebhookParseResult | None:
        action = raw_payload.get("action")
        if action not in _REVIEW_ACTIONS:
            return None

        review = raw_payload.get("review")
        pr = raw_payload.get("pull_request")
        if not isinstance(review, dict) or not isinstance(pr, dict):
            raise InvalidWebhookPayload("pull_request_review missing review/pull_request")

        review_id = review.get("id")
        pr_number = pr.get("number")
        submitted_at = review.get("submitted_at")
        if review_id is None or pr_number is None or not submitted_at:
            raise InvalidWebhookPayload("pull_request_review missing required fields")

        source_event_id = f"review:{full_name}:{pr_number}:{review_id}"
        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=_parse_iso8601(submitted_at),
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "event_type": _EVENT_PR_REVIEW,
                "action": action,
                "repo": full_name,
                "pr_number": pr_number,
                "review_id": review_id,
            },
        )

    def _parse_installation(
        self, raw_payload: Mapping[str, Any]
    ) -> WebhookParseResult | None:
        """Parse the GitHub App `installation` event.

        Actions: created, deleted, suspend, unsuspend, new_permissions_accepted.
        We act on `created` (initial backfill for every accessible repo) and
        `deleted` (disconnect cascade for every repo). Other actions are
        recorded as queue noise — `received_at` carries the event timestamp
        so the row's idempotency key includes it.
        """
        action = raw_payload.get("action")
        installation = raw_payload.get("installation") or {}
        installation_id = installation.get("id")
        if installation_id is None:
            raise InvalidWebhookPayload("installation event missing installation.id")
        # Time fields vary across actions; fall back to NOW so we always
        # have a valid timestamp.
        ts = installation.get("updated_at") or installation.get("created_at")
        return WebhookParseResult(
            source_event_id=f"installation:{installation_id}:{action}:{ts or _payload_fp(raw_payload)}",
            received_at=_parse_iso8601(ts) if ts else datetime.now(UTC),
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "event_type": _EVENT_INSTALLATION,
                "action": action,
                "installation_id": installation_id,
            },
        )

    def _parse_installation_repositories(
        self, raw_payload: Mapping[str, Any]
    ) -> WebhookParseResult | None:
        """Parse `installation_repositories` (repos added/removed from an install).

        Action is one of `added` / `removed`. Used to fan in
        bridge.enqueue_initial_backfill or enqueue_disconnect for the
        listed repos.
        """
        action = raw_payload.get("action")
        installation = raw_payload.get("installation") or {}
        installation_id = installation.get("id")
        if installation_id is None:
            raise InvalidWebhookPayload(
                "installation_repositories event missing installation.id"
            )
        # No reliable timestamp on the payload — synthesize one.
        ts = datetime.now(UTC).isoformat()
        return WebhookParseResult(
            source_event_id=(
                f"installation_repositories:{installation_id}:{action}:{ts}:{_payload_fp(raw_payload)}"
            ),
            received_at=datetime.now(UTC),
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "event_type": _EVENT_INSTALLATION_REPOSITORIES,
                "action": action,
                "installation_id": installation_id,
            },
        )

    # ------------------------------------------------------------------
    # 3. hydration — fetch CODEOWNERS file contents for push events
    # ------------------------------------------------------------------

    async def _resolve_installation_bearer(
        self, token: IntegrationToken, *, customer_id: str
    ) -> str:
        """Return the bearer to use for GitHub API calls.

        If token.scope starts with 'installation:', fetch a fresh installation
        token from prbe-backend (which mints + caches it server-side using the
        GitHub App private key). Otherwise return token.access_token as-is
        (legacy path — assumes caller already provided a valid token).
        """
        scope = token.scope or ""
        if not scope.startswith(GITHUB_INSTALLATION_SCOPE_PREFIX):
            return token.access_token

        bearer, _expires = await fetch_github_installation_token(
            self.http,
            customer_id=customer_id,
        )
        return bearer

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: IntegrationToken | None,
    ) -> dict[str, Any]:
        # We only need to fetch CODEOWNERS for pushes that touched it.
        headers = event.headers or {}
        event_type = _header(headers, "x-github-event")
        if event_type != _EVENT_PUSH:
            return {}

        if not _push_touches_codeowners(event.raw_payload):
            return {}

        repo = event.raw_payload.get("repository", {})
        full_name = repo.get("full_name")
        if not isinstance(full_name, str) or not full_name or token is None:
            # No installation token — defer to the fallback path in normalize().
            return {}

        bearer = await self._resolve_installation_bearer(
            token, customer_id=event.customer_id
        )

        # Try each canonical CODEOWNERS path in order. GitHub accepts all three.
        for path in _CODEOWNERS_PATHS:
            content = await self._fetch_codeowners_content(full_name, path, bearer)
            if content is not None:
                return {"codeowners_content": content, "codeowners_path": path}

        return {}

    async def _fetch_codeowners_content(
        self, full_name: str, path: str, bearer: str
    ) -> str | None:
        url = f"{_GITHUB_API}/repos/{full_name}/contents/{path}"
        try:
            resp = await self.http.get(
                url,
                headers={
                    "Authorization": f"Bearer {bearer}",
                    "Accept": "application/vnd.github.raw",
                },
            )
        except (OSError, ValueError) as exc:
            log.warning("github.fetch_codeowners_failed", error=str(exc), path=path)
            return None

        if resp.status_code != 200:
            return None
        return resp.text

    # ------------------------------------------------------------------
    # 6. OAuth install + exchange
    # ------------------------------------------------------------------

    def oauth_install_url(self, customer_id: str, redirect_uri: str) -> str:
        slug = self.settings.github_app_slug
        if not slug:
            raise NotSupportedByConnector("GITHUB_APP_SLUG not configured")
        # GitHub Apps don't accept redirect_uri on the install URL — the
        # post-install redirect is controlled in the App's settings. We still
        # accept the arg for API compatibility; state carries the customer_id
        # through the round-trip.
        from urllib.parse import urlencode

        params = urlencode({"state": customer_id})
        return f"https://github.com/apps/{slug}/installations/new?{params}"

    async def exchange_oauth_code(
        self,
        code: str | None,
        redirect_uri: str,
        extra_params: dict[str, str] | None = None,
    ) -> IntegrationToken:
        extra = extra_params or {}
        installation_id = extra.get("installation_id")
        if not installation_id:
            raise InvalidWebhookPayload(
                "github OAuth callback missing installation_id — was the app installed?"
            )

        # GitHub App credentials live in prbe-backend now. We don't validate
        # the installation here (we have no JWT to do so); the first real call
        # via fetch_github_installation_token will surface any mint failure.
        return IntegrationToken(
            customer_id="",  # caller fills in — connector does not know the tenant
            source_system=SourceSystem.GITHUB,
            # access_token column is NOT NULL. GitHub installation tokens are
            # fetched on-demand from prbe-backend so the stored value is never
            # read by the connector.
            access_token="installation-minted-on-demand",
            scope=f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{installation_id}",
        )

    # ------------------------------------------------------------------
    # 7. workspace identification
    # ------------------------------------------------------------------

    async def identify_workspaces(self, token):  # type: ignore[override]
        """Resolve the GitHub installation id for customer_source_mapping.

        `token.scope` carries `installation:<id>` from `exchange_oauth_code`.
        Pre-PR-B we'd also call `GET /app/installations/<id>` with the App JWT
        to fetch the account login — that required the App private key, which
        now lives only in prbe-backend. Without it we return just the
        installation_id; webhook routing still works because
        `extract_external_id_from_payload` keys on the same id.
        """
        scope = token.scope or ""
        if not scope.startswith(GITHUB_INSTALLATION_SCOPE_PREFIX):
            return []
        installation_id = scope.split(":", 1)[1]
        if not installation_id:
            return []

        return [
            ExternalWorkspaceRef(
                external_id=installation_id,
                external_name=None,
                metadata={"installation_id": installation_id},
            )
        ]

    def extract_external_id_from_payload(self, headers, raw_payload):
        install = raw_payload.get("installation") or {}
        iid = install.get("id")
        return str(iid) if iid is not None else None

    # ------------------------------------------------------------------
    # 5. backfill
    # ------------------------------------------------------------------

    async def backfill(
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ):
        """Historical GitHub backfill — walks installation repos, PRs, and issues.

        When `token.scope` starts with `installation:` we fetch a fresh App
        installation bearer from prbe-backend (via `shared.backend_client`).
        Otherwise we use `token.access_token` verbatim (legacy / test path).

        Resumable via the `cursor` arg: an opaque JSON blob capturing which
        repos remain, the current repo + phase (pulls/issues), and the next
        page URL. Yields synthetic WebhookEvents shaped like live webhook
        deliveries so the normalizer has one code path.
        """
        import asyncio as _asyncio
        import json as _json

        import httpx

        from shared.models import WebhookEvent

        state = _decode_github_cursor(cursor)
        bearer = await self._resolve_installation_bearer(token, customer_id=customer_id)
        auth_headers = {
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        if not state["repos_remaining"] and state["current_repo"] is None:
            state["repos_remaining"] = await _list_installation_repos(
                self.http, auth_headers
            )

        repo_objs: dict[str, Mapping[str, Any]] = state.get("repo_objs") or {}

        while state["current_repo"] or state["repos_remaining"]:
            if state["current_repo"] is None:
                next_repo = state["repos_remaining"].pop(0)
                if isinstance(next_repo, dict):
                    full_name = next_repo.get("full_name")
                    if not full_name:
                        continue
                    repo_objs[full_name] = next_repo
                    state["current_repo"] = full_name
                else:
                    state["current_repo"] = next_repo
                state["current_phase"] = "pulls"
                state["next_url"] = (
                    f"{_GITHUB_API}/repos/{state['current_repo']}/pulls"
                    "?state=all&sort=updated&direction=desc&per_page=100"
                )
                state["repo_objs"] = repo_objs

            full_name = state["current_repo"]
            repo = repo_objs.get(full_name) or {"full_name": full_name}

            url = state["next_url"]
            if not url:
                if state["current_phase"] == "pulls":
                    state["current_phase"] = "issues"
                    state["next_url"] = (
                        f"{_GITHUB_API}/repos/{full_name}/issues"
                        "?state=all&sort=updated&direction=desc&per_page=100"
                    )
                    continue
                state["current_repo"] = None
                state["current_phase"] = "pulls"
                state["next_url"] = None
                repo_objs.pop(full_name, None)
                state["repo_objs"] = repo_objs
                continue

            try:
                resp = await self.http.get(url, headers=auth_headers)
            except httpx.HTTPError as exc:
                log.warning(
                    "github.backfill_http_error",
                    repo=full_name,
                    phase=state["current_phase"],
                    error=str(exc),
                )
                state["next_url"] = None
                continue

            if resp.status_code == 429 or (
                resp.status_code == 403
                and resp.headers.get("x-ratelimit-remaining") == "0"
            ):
                # Respect GitHub's rate-limit backoff: honor retry-after if
                # present, else compute delta from x-ratelimit-reset (unix ts).
                retry_after = resp.headers.get("retry-after")
                if retry_after is not None:
                    try:
                        delay = int(retry_after)
                    except ValueError:
                        delay = 5
                else:
                    reset = resp.headers.get("x-ratelimit-reset")
                    try:
                        delay = max(
                            int(float(reset)) - int(datetime.now(UTC).timestamp()), 1
                        ) if reset else 5
                    except ValueError:
                        delay = 5
                await _asyncio.sleep(max(delay, 1))
                continue

            if resp.status_code != 200:
                log.warning(
                    "github.backfill_non_200",
                    repo=full_name,
                    phase=state["current_phase"],
                    status=resp.status_code,
                )
                state["next_url"] = None
                continue

            rows = resp.json()
            if not isinstance(rows, list):
                state["next_url"] = None
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                if state["current_phase"] == "pulls":
                    number = row.get("number")
                    updated_at = row.get("updated_at")
                    if number is None or not updated_at:
                        continue
                    raw_payload = {
                        "action": "opened",
                        "repository": repo,
                        "pull_request": row,
                        "_cursor": _json.dumps(state),
                    }
                    source_event_id = (
                        f"pr:{full_name}:{number}:opened:{updated_at}:{_payload_fp(row)}"
                    )
                    yield WebhookEvent(
                        customer_id=customer_id,
                        source_system=SourceSystem.GITHUB,
                        source_event_id=source_event_id,
                        received_at=_parse_iso8601(updated_at),
                        payload_s3_key="",
                        raw_payload=raw_payload,
                        headers={"X-GitHub-Event": _EVENT_PULL_REQUEST},
                    )
                else:
                    # GitHub's /issues endpoint returns PRs as issues (with a
                    # `pull_request` field). Skip those — they're covered in
                    # the pulls phase above.
                    if row.get("pull_request") is not None:
                        continue
                    number = row.get("number")
                    updated_at = row.get("updated_at")
                    if number is None or not updated_at:
                        continue
                    raw_payload = {
                        "action": "opened",
                        "repository": repo,
                        "issue": row,
                        "_cursor": _json.dumps(state),
                    }
                    source_event_id = f"issue:{full_name}:{number}:opened:{updated_at}"
                    yield WebhookEvent(
                        customer_id=customer_id,
                        source_system=SourceSystem.GITHUB,
                        source_event_id=source_event_id,
                        received_at=_parse_iso8601(updated_at),
                        payload_s3_key="",
                        raw_payload=raw_payload,
                        headers={"X-GitHub-Event": _EVENT_ISSUES},
                    )

            state["next_url"] = _next_link(resp)

    # ------------------------------------------------------------------
    # 4. normalization
    # ------------------------------------------------------------------

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        headers = event.headers or {}
        event_type = _header(headers, "x-github-event")
        if event_type is None:
            return NormalizationResult(skipped_reason="missing X-GitHub-Event header")

        if event_type == _EVENT_PULL_REQUEST:
            return self._normalize_pr(event)
        if event_type == _EVENT_ISSUES:
            return self._normalize_issue(event)
        if event_type == _EVENT_PUSH:
            result = self._normalize_push(event, hydrated)
            # Forward push deltas to the code-graph bridge so symbol
            # extraction stays in sync. Fire-and-await: a bridge failure
            # shouldn't block the github.commit ingestion path, so
            # exceptions are logged and swallowed.
            await self._fire_codegraph_incremental(event)
            return result
        if event_type == _EVENT_PR_REVIEW:
            return self._normalize_review(event)
        if event_type == _EVENT_INSTALLATION:
            return await self._normalize_installation(event)
        if event_type == _EVENT_INSTALLATION_REPOSITORIES:
            return await self._normalize_installation_repositories(event)

        return NormalizationResult(skipped_reason=f"unhandled github event {event_type}")

    # ---- PR ----------------------------------------------------------

    def _normalize_pr(self, event: WebhookEvent) -> NormalizationResult:
        payload = event.raw_payload
        repo = payload.get("repository") or {}
        pr = payload.get("pull_request") or {}
        full_name = repo.get("full_name") or ""
        number = pr.get("number")
        if not full_name or number is None:
            return NormalizationResult(skipped_reason="pr missing repo/number")

        action = payload.get("action")
        is_delete = action in _DELETE_ACTIONS

        author = (pr.get("user") or {}).get("login") or "unknown"
        title = pr.get("title") or ""
        body = pr.get("body") or ""
        html_url = pr.get("html_url") or ""
        created = _parse_iso8601(pr.get("created_at"))
        updated = _parse_iso8601(pr.get("updated_at"))

        doc_id = f"github:{full_name}:pr:{number}"
        source_id = f"{full_name}#{number}"
        deleted_at = event.received_at if is_delete else None
        if is_delete:
            body = ""
            content_hash = _sha256(
                f"{doc_id}|__deleted__|{event.received_at.isoformat()}"
            )
        else:
            content_hash = _sha256(f"{doc_id}|{title}|{body}")

        base_ref = (pr.get("base") or {}).get("ref")
        head_ref = (pr.get("head") or {}).get("ref")

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.GITHUB,
            source_id=source_id,
            source_url=html_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.GITHUB_PULL_REQUEST,
            content_type="text/markdown",
            content_hash=content_hash,
            title=title[:240] if title else None,
            body_preview=body[:280] if body else None,
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=author,
            created_at=created,
            updated_at=updated,
            valid_from=created,
            deleted_at=deleted_at,
            ingested_at=datetime.now(UTC),
            acl=_repo_acl_snapshot(repo, event.received_at),
            metadata={
                "action": action,
                "repo_full_name": full_name,
                "number": number,
                "base_ref": base_ref,
                "head_ref": head_ref,
                "changed_files": pr.get("changed_files"),
                "additions": pr.get("additions"),
                "deletions": pr.get("deletions"),
                "merged": pr.get("merged"),
                "state": pr.get("state"),
                "visibility": _repo_visibility(repo),
            },
            body=body,
            doc_references=_references_from_text(body, full_name, html_url),
        )

        nodes = [
            _repo_node(repo),
            GraphNodeSpec(
                label=NodeLabel.PR,
                canonical_id=f"{full_name}#{number}",
                properties={"repo": full_name, "number": number},
            ),
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=author,
                properties={"source_system": SourceSystem.GITHUB.value},
            ),
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": DocType.GITHUB_PULL_REQUEST.value},
            ),
        ]

        edges = [
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=author,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                valid_from=created,
            ),
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=author,
                to_label=NodeLabel.PR,
                to_canonical_id=f"{full_name}#{number}",
                valid_from=created,
            ),
            GraphEdgeSpec(
                edge_type=EdgeType.TOUCHES,
                from_label=NodeLabel.PR,
                from_canonical_id=f"{full_name}#{number}",
                to_label=NodeLabel.REPO,
                to_canonical_id=full_name,
                valid_from=created,
            ),
            # Document → Repo: the list-pipeline entity filter walks
            # graph_edges from the Document node looking for a matching
            # entity. Without this edge, "last commit on prbe-backend"
            # finds zero docs because the only Repo connection is via
            # the PR/Issue/etc. node, not the Document itself.
            GraphEdgeSpec(
                edge_type=EdgeType.TOUCHES,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.REPO,
                to_canonical_id=full_name,
                valid_from=created,
            ),
        ]
        edges.extend(
            _mention_edges(
                body,
                full_name,
                from_label=NodeLabel.PR,
                from_canonical_id=f"{full_name}#{number}",
                valid_from=created,
            )
        )

        acl_rows = [_repo_acl_row(repo, created)]

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ---- Issue -------------------------------------------------------

    def _normalize_issue(self, event: WebhookEvent) -> NormalizationResult:
        payload = event.raw_payload
        repo = payload.get("repository") or {}
        issue = payload.get("issue") or {}
        full_name = repo.get("full_name") or ""
        number = issue.get("number")
        if not full_name or number is None:
            return NormalizationResult(skipped_reason="issue missing repo/number")

        action = payload.get("action")
        is_delete = action in _DELETE_ACTIONS

        author = (issue.get("user") or {}).get("login") or "unknown"
        title = issue.get("title") or ""
        body = issue.get("body") or ""
        html_url = issue.get("html_url") or ""
        created = _parse_iso8601(issue.get("created_at"))
        updated = _parse_iso8601(issue.get("updated_at"))

        doc_id = f"github:{full_name}:issue:{number}"
        source_id = f"{full_name}#{number}"
        deleted_at = event.received_at if is_delete else None
        if is_delete:
            body = ""
            content_hash = _sha256(
                f"{doc_id}|__deleted__|{event.received_at.isoformat()}"
            )
        else:
            content_hash = _sha256(f"{doc_id}|{title}|{body}")

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.GITHUB,
            source_id=source_id,
            source_url=html_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.GITHUB_ISSUE,
            content_type="text/markdown",
            content_hash=content_hash,
            title=title[:240] if title else None,
            body_preview=body[:280] if body else None,
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=author,
            created_at=created,
            updated_at=updated,
            valid_from=created,
            deleted_at=deleted_at,
            ingested_at=datetime.now(UTC),
            acl=_repo_acl_snapshot(repo, event.received_at),
            metadata={
                "action": action,
                "repo_full_name": full_name,
                "number": number,
                "state": issue.get("state"),
                "labels": [la.get("name") for la in issue.get("labels", []) or []],
                "visibility": _repo_visibility(repo),
            },
            body=body,
            doc_references=_references_from_text(body, full_name, html_url),
        )

        nodes = [
            _repo_node(repo),
            GraphNodeSpec(
                label=NodeLabel.ISSUE,
                canonical_id=f"{full_name}#{number}",
                properties={"repo": full_name, "number": number},
            ),
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=author,
                properties={"source_system": SourceSystem.GITHUB.value},
            ),
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": DocType.GITHUB_ISSUE.value},
            ),
        ]

        edges = [
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=author,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                valid_from=created,
            ),
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=author,
                to_label=NodeLabel.ISSUE,
                to_canonical_id=f"{full_name}#{number}",
                valid_from=created,
            ),
            GraphEdgeSpec(
                edge_type=EdgeType.TOUCHES,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.REPO,
                to_canonical_id=full_name,
                valid_from=created,
            ),
            GraphEdgeSpec(
                edge_type=EdgeType.TOUCHES,
                from_label=NodeLabel.ISSUE,
                from_canonical_id=f"{full_name}#{number}",
                to_label=NodeLabel.REPO,
                to_canonical_id=full_name,
                valid_from=created,
            ),
        ]
        edges.extend(
            _mention_edges(
                body,
                full_name,
                from_label=NodeLabel.ISSUE,
                from_canonical_id=f"{full_name}#{number}",
                valid_from=created,
            )
        )

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=[_repo_acl_row(repo, created)],
        )

    # ---- Push --------------------------------------------------------

    def _normalize_push(
        self, event: WebhookEvent, hydrated: Mapping[str, Any]
    ) -> NormalizationResult:
        payload = event.raw_payload
        repo = payload.get("repository") or {}
        full_name = repo.get("full_name") or ""
        if not full_name:
            return NormalizationResult(skipped_reason="push missing repo.full_name")

        commits_raw = payload.get("commits") or []
        if not isinstance(commits_raw, list):
            raise InvalidWebhookPayload("push payload commits must be a list")

        documents: list[Document] = []
        nodes: list[GraphNodeSpec] = [_repo_node(repo)]
        edges: list[GraphEdgeSpec] = []
        seen_people: set[str] = set()

        for commit in commits_raw:
            if not isinstance(commit, dict):
                continue
            doc, commit_nodes, commit_edges = _commit_to_doc(
                event=event,
                commit=commit,
                repo=repo,
                seen_people=seen_people,
            )
            if doc is None:
                continue
            documents.append(doc)
            nodes.extend(commit_nodes)
            edges.extend(commit_edges)

        # CODEOWNERS handling (only when a commit touched the file).
        acl_rows: list[ACLSnapshotRow] = [
            _repo_acl_row(repo, _parse_iso8601(payload.get("head_commit", {}).get("timestamp")))
        ]
        if _push_touches_codeowners(payload):
            co_doc, co_nodes, co_edges = _codeowners_artifacts(
                event=event,
                repo=repo,
                hydrated=hydrated,
            )
            documents.append(co_doc)
            nodes.extend(co_nodes)
            edges.extend(co_edges)

        return NormalizationResult(
            documents=documents,
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ---- Review ------------------------------------------------------

    def _normalize_review(self, event: WebhookEvent) -> NormalizationResult:
        payload = event.raw_payload
        repo = payload.get("repository") or {}
        review = payload.get("review") or {}
        pr = payload.get("pull_request") or {}
        full_name = repo.get("full_name") or ""
        review_id = review.get("id")
        pr_number = pr.get("number")
        if not full_name or review_id is None or pr_number is None:
            return NormalizationResult(skipped_reason="review missing required ids")

        author = (review.get("user") or {}).get("login") or "unknown"
        body = review.get("body") or ""
        html_url = review.get("html_url") or ""
        submitted_at = _parse_iso8601(review.get("submitted_at"))

        doc_id = f"github:{full_name}:review:{review_id}"
        source_id = f"{full_name}#review:{review_id}"
        parent_doc_id = f"github:{full_name}:pr:{pr_number}"
        content_hash = _sha256(f"{doc_id}|{body}|{review.get('state', '')}")

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.GITHUB,
            source_id=source_id,
            source_url=html_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.GITHUB_REVIEW,
            content_type="text/markdown",
            content_hash=content_hash,
            title=_derive_title(body) or f"Review on PR #{pr_number}",
            body_preview=body[:280] if body else None,
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=author,
            created_at=submitted_at,
            updated_at=submitted_at,
            valid_from=submitted_at,
            ingested_at=datetime.now(UTC),
            parent_doc_id=parent_doc_id,
            acl=_repo_acl_snapshot(repo, event.received_at),
            metadata={
                "repo_full_name": full_name,
                "pr_number": pr_number,
                "review_id": review_id,
                "state": review.get("state"),
                "visibility": _repo_visibility(repo),
            },
            body=body,
            doc_references=_references_from_text(body, full_name, html_url),
        )

        nodes = [
            _repo_node(repo),
            GraphNodeSpec(
                label=NodeLabel.PR,
                canonical_id=f"{full_name}#{pr_number}",
                properties={"repo": full_name, "number": pr_number},
            ),
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=author,
                properties={"source_system": SourceSystem.GITHUB.value},
            ),
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": DocType.GITHUB_REVIEW.value},
            ),
        ]

        edges = [
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=author,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                valid_from=submitted_at,
            ),
            # Document → Repo so the list-pipeline entity filter
            # ("PR reviews on prbe-backend") can reach the doc.
            GraphEdgeSpec(
                edge_type=EdgeType.TOUCHES,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.REPO,
                to_canonical_id=full_name,
                valid_from=submitted_at,
            ),
        ]

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=[_repo_acl_row(repo, submitted_at)],
        )

    # ---- code-graph bridge fan-outs --------------------------------------

    async def _fire_codegraph_incremental(self, event: WebhookEvent) -> None:
        """After a verified push lands, forward the changed-files set to the
        code-graph bridge for symbol-level diff extraction.

        Branch gating: only pushes to the repo's tracked branch fire the
        bridge. The tracked branch is per-(customer, repo), resolved via
        `code_graph_indexed_branch` — falls back to the repo's
        `default_branch` when no override is set. This prevents feature
        branches from polluting the (customer, repo, file_path)-keyed
        `code_repo_state` cache with non-default-branch content.

        Failures are logged and swallowed — symbol extraction must never
        block the github.commit ingestion path. The bridge writes a
        separate ingestion_queue row keyed on (customer, code_graph,
        repo:sha) so a transient failure here can be replayed via a
        resync without re-ingesting commit Documents.
        """
        payload = event.raw_payload
        repository = payload.get("repository") or {}
        repo = repository.get("full_name") or ""
        head = payload.get("head_commit") or {}
        sha = head.get("id") or payload.get("after") or ""
        if not repo or not sha:
            return

        push_branch = _push_branch_from_ref(payload.get("ref"))
        default_branch = repository.get("default_branch")
        if not push_branch or not isinstance(default_branch, str) or not default_branch:
            log.info(
                "code_graph.bridge.skip_unknown_branch",
                customer=event.customer_id,
                repo=repo,
                ref=payload.get("ref"),
            )
            return
        tracked_branch = await code_graph_indexed_branch(
            event.customer_id, repo, default_branch
        )
        if push_branch != tracked_branch:
            log.info(
                "code_graph.bridge.skip_non_tracked_branch",
                customer=event.customer_id,
                repo=repo,
                push_branch=push_branch,
                tracked_branch=tracked_branch,
            )
            return

        added: list[str] = []
        modified: list[str] = []
        removed: list[str] = []
        for commit in payload.get("commits") or []:
            if not isinstance(commit, dict):
                continue
            for path in commit.get("added") or []:
                if isinstance(path, str):
                    added.append(path)
            for path in commit.get("modified") or []:
                if isinstance(path, str):
                    modified.append(path)
            for path in commit.get("removed") or []:
                if isinstance(path, str):
                    removed.append(path)
        # Dedup; per-commit lists overlap when several commits touched the
        # same file in one push.
        added = sorted(set(added))
        modified = sorted(set(modified))
        removed = sorted(set(removed))
        if not (added or modified or removed):
            return
        try:
            await code_graph_bridge.enqueue_incremental(
                customer_id=event.customer_id,
                repo=repo,
                sha=sha,
                files_added=added,
                files_modified=modified,
                files_removed=removed,
                integration_token_id=None,
                originating_source=SourceSystem.GITHUB,
            )
        except Exception as exc:
            log.warning(
                "code_graph.bridge.enqueue_incremental_failed",
                customer=event.customer_id,
                repo=repo,
                sha=sha,
                error=str(exc),
            )

    async def _normalize_installation(self, event: WebhookEvent) -> NormalizationResult:
        """Handle GitHub App installation lifecycle.

        On `created`: enumerate accessible repositories and fan in
        bridge.enqueue_initial_backfill per repo.
        On `deleted`: enumerate the installation's prior repos (from the
        payload, since GitHub embeds them on uninstall) and fan in
        bridge.enqueue_disconnect.
        Other actions (suspend/unsuspend/permissions) are recorded as
        idempotent no-ops.
        """
        payload = event.raw_payload
        action = payload.get("action")
        repositories = payload.get("repositories") or []
        repo_full_names = [
            r.get("full_name") for r in repositories if isinstance(r, dict)
        ]
        repo_full_names = [r for r in repo_full_names if r]

        if action == "created":
            for repo in repo_full_names:
                try:
                    await code_graph_bridge.enqueue_initial_backfill(
                        customer_id=event.customer_id,
                        repo=repo,
                        head_sha="HEAD",
                        integration_token_id=None,
                        originating_source=SourceSystem.GITHUB,
                    )
                except Exception as exc:
                    log.warning(
                        "code_graph.bridge.enqueue_initial_backfill_failed",
                        customer=event.customer_id,
                        repo=repo,
                        error=str(exc),
                    )
        elif action == "deleted" and repo_full_names:
            try:
                await code_graph_bridge.enqueue_disconnect(
                    customer_id=event.customer_id,
                    repos=repo_full_names,
                    originating_source=SourceSystem.GITHUB,
                )
            except Exception as exc:
                log.warning(
                    "code_graph.bridge.enqueue_disconnect_failed",
                    customer=event.customer_id,
                    repos=repo_full_names,
                    error=str(exc),
                )
        return NormalizationResult(
            skipped_reason=f"installation.{action} processed (code_graph bridge fan-out)"
        )

    async def _normalize_installation_repositories(
        self, event: WebhookEvent
    ) -> NormalizationResult:
        """Handle `installation_repositories` events.

        Action 'added': bridge.enqueue_initial_backfill per repo.
        Action 'removed': bridge.enqueue_disconnect for the listed repos.
        """
        payload = event.raw_payload
        action = payload.get("action")
        added = payload.get("repositories_added") or []
        removed = payload.get("repositories_removed") or []
        added_names = [r.get("full_name") for r in added if isinstance(r, dict)]
        added_names = [r for r in added_names if r]
        removed_names = [r.get("full_name") for r in removed if isinstance(r, dict)]
        removed_names = [r for r in removed_names if r]

        if action == "added":
            for repo in added_names:
                try:
                    await code_graph_bridge.enqueue_initial_backfill(
                        customer_id=event.customer_id,
                        repo=repo,
                        head_sha="HEAD",
                        integration_token_id=None,
                        originating_source=SourceSystem.GITHUB,
                    )
                except Exception as exc:
                    log.warning(
                        "code_graph.bridge.enqueue_initial_backfill_failed",
                        customer=event.customer_id,
                        repo=repo,
                        error=str(exc),
                    )
        elif action == "removed" and removed_names:
            try:
                await code_graph_bridge.enqueue_disconnect(
                    customer_id=event.customer_id,
                    repos=removed_names,
                    originating_source=SourceSystem.GITHUB,
                )
            except Exception as exc:
                log.warning(
                    "code_graph.bridge.enqueue_disconnect_failed",
                    customer=event.customer_id,
                    repos=removed_names,
                    error=str(exc),
                )
        return NormalizationResult(
            skipped_reason=(
                f"installation_repositories.{action} processed "
                f"(code_graph bridge fan-out: +{len(added_names)} -{len(removed_names)})"
            )
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _list_installation_repos(http, headers: dict[str, str]) -> list[dict]:
    """Enumerate repositories the installation can access. Paginated."""
    repos: list[dict] = []
    url: str | None = (
        f"{_GITHUB_API}/installation/repositories?per_page=100"
    )
    while url:
        try:
            resp = await http.get(url, headers=headers)
        except Exception as exc:
            log.warning("github.list_repos_error", error=str(exc))
            return repos
        if resp.status_code != 200:
            return repos
        body = resp.json()
        page_repos = body.get("repositories") if isinstance(body, dict) else None
        if isinstance(page_repos, list):
            for r in page_repos:
                if isinstance(r, dict) and r.get("full_name"):
                    repos.append(r)
        url = _next_link(resp)
    return repos


def _next_link(resp) -> str | None:
    """Parse a GitHub Link header for the `rel="next"` URL."""
    link_header = resp.headers.get("link") or resp.headers.get("Link")
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' not in part:
            continue
        if part.startswith("<") and ">" in part:
            return part.split(">", 1)[0][1:]
    return None


def _decode_github_cursor(cursor: str | None) -> dict:
    import json as _json

    default = {
        "repos_remaining": [],
        "current_repo": None,
        "current_phase": "pulls",
        "next_url": None,
        "repo_objs": {},
    }
    if not cursor:
        return default
    try:
        parsed = _json.loads(cursor)
    except _json.JSONDecodeError:
        return default
    if not isinstance(parsed, dict):
        return default
    for key, value in default.items():
        parsed.setdefault(key, value)
    return parsed


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for k, v in headers.items():
        if k.lower() == lowered:
            return v
    return None


def _parse_iso8601(value: Any) -> datetime:
    """Parse a GitHub-style ISO8601 timestamp (with trailing Z) into UTC datetime.

    Falls back to now-UTC if the value is missing — GitHub should always send
    one, but we don't want one missing field to sink an entire ingest.
    """
    if not isinstance(value, str) or not value:
        return datetime.now(UTC)
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise InvalidWebhookPayload(f"invalid iso8601 timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_fp(obj: Any) -> str:
    """Stable 16-char fingerprint of a JSON-serialisable webhook subobject.

    Used to disambiguate two distinct logical events that share an
    `updated_at` second (GitHub timestamps are second-resolution). The
    same payload bytes always hash to the same fingerprint, so true
    webhook retries still collide on the queue's UNIQUE constraint and
    dedupe correctly.

    Strict JSON: we DON'T pass `default=str` because `str(datetime)` is
    timezone-dependent and would let two equivalent timestamps fingerprint
    differently. Callers always feed dicts/lists that came from JSON
    deserialization of a webhook body, so non-JSON types are a bug; let
    `TypeError` raise loudly rather than silently producing a non-stable
    hash.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _derive_title(text: str) -> str | None:
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    return first_line[:120] if first_line else None


def _repo_visibility(repo: Mapping[str, Any]) -> str:
    return "private" if repo.get("private") else "public"


def _repo_node(repo: Mapping[str, Any]) -> GraphNodeSpec:
    full_name = repo.get("full_name") or ""
    return GraphNodeSpec(
        label=NodeLabel.REPO,
        canonical_id=full_name,
        properties={
            "name": repo.get("name"),
            "owner": (repo.get("owner") or {}).get("login"),
            "visibility": _repo_visibility(repo),
            "html_url": repo.get("html_url"),
        },
    )


def _repo_acl_snapshot(
    repo: Mapping[str, Any], captured_at: datetime
) -> ACLSnapshot:
    owner_login = (repo.get("owner") or {}).get("login") or "unknown"
    return ACLSnapshot(
        principals=[
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=owner_login,
                permission=Permission.READ,
            )
        ],
        captured_at=captured_at,
    )


def _repo_acl_row(repo: Mapping[str, Any], valid_from: datetime) -> ACLSnapshotRow:
    owner_login = (repo.get("owner") or {}).get("login") or "unknown"
    full_name = repo.get("full_name") or ""
    return ACLSnapshotRow(
        source_system=SourceSystem.GITHUB,
        principal_type=PrincipalType.WORKSPACE,
        principal_id=owner_login,
        resource_type=_ACL_RESOURCE_REPO,
        resource_id=full_name,
        permission=Permission.READ,
        valid_from=valid_from,
        metadata={"visibility": _repo_visibility(repo)},
    )


def _push_branch_from_ref(ref: object) -> str | None:
    """Strip `refs/heads/` from a push payload's `ref` field.

    Returns None for tag pushes (`refs/tags/...`), missing refs, or any
    non-branch ref shape we don't want to extract code-graph from.
    """
    if not isinstance(ref, str) or not ref.startswith("refs/heads/"):
        return None
    branch = ref[len("refs/heads/"):]
    return branch or None


def _push_touches_codeowners(payload: Mapping[str, Any]) -> bool:
    commits = payload.get("commits") or []
    head = payload.get("head_commit") or {}
    candidates: list[Mapping[str, Any]] = []
    if isinstance(head, dict):
        candidates.append(head)
    if isinstance(commits, list):
        for c in commits:
            if isinstance(c, dict):
                candidates.append(c)

    for commit in candidates:
        for bucket in ("added", "modified", "removed"):
            files = commit.get(bucket) or []
            if not isinstance(files, list):
                continue
            for fp in files:
                if fp in _CODEOWNERS_PATHS:
                    return True
    return False


def _parse_co_authors(message: str) -> list[dict[str, str]]:
    """Extract `Co-authored-by: Name <email>` trailers from a commit message.

    Returns a list of `{"name": ..., "email": ...}` dicts in source order,
    deduplicated by lowercased email. Email is the identity key because the
    GitHub UI uses email (not name) to attribute commits in the contributor
    graph for trailer-based co-authorship.
    """
    if not message:
        return []
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for match in _COAUTHOR_TRAILER.finditer(message):
        name = match.group("name").strip()
        email = match.group("email").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        out.append({"name": name, "email": email})
    return out


def _commit_to_doc(
    *,
    event: WebhookEvent,
    commit: Mapping[str, Any],
    repo: Mapping[str, Any],
    seen_people: set[str],
) -> tuple[Document | None, list[GraphNodeSpec], list[GraphEdgeSpec]]:
    commit_id = commit.get("id")
    if not isinstance(commit_id, str) or not commit_id:
        return None, [], []

    full_name = repo.get("full_name") or ""
    message = commit.get("message") or ""
    timestamp = _parse_iso8601(commit.get("timestamp"))
    author_info = commit.get("author") or {}
    author = author_info.get("username") or author_info.get("email") or "unknown"
    primary_email = (author_info.get("email") or "").strip().lower()
    html_url = commit.get("url") or ""

    co_authors = _parse_co_authors(message)
    # Drop any co-author whose email matches the primary committer — git
    # tooling sometimes adds a trailer for the primary author too.
    co_authors = [c for c in co_authors if c["email"] != primary_email]

    doc_id = f"github:{full_name}:commit:{commit_id}"
    source_id = f"{full_name}@{commit_id}"
    content_hash = _sha256(f"{doc_id}|{message}")

    metadata: dict[str, Any] = {
        "repo_full_name": full_name,
        "commit_id": commit_id,
        "added": commit.get("added") or [],
        "modified": commit.get("modified") or [],
        "removed": commit.get("removed") or [],
        "visibility": _repo_visibility(repo),
    }
    if co_authors:
        metadata["co_authors"] = co_authors

    doc = Document(
        doc_id=doc_id,
        customer_id=event.customer_id,
        source_system=SourceSystem.GITHUB,
        source_id=source_id,
        source_url=html_url,
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.GITHUB_COMMIT,
        content_type="text/plain",
        content_hash=content_hash,
        title=_derive_title(message),
        body_preview=message[:280] if message else None,
        body_size_bytes=len(message.encode("utf-8")),
        body_token_count=count_tokens(message),
        author_id=author,
        created_at=timestamp,
        updated_at=timestamp,
        valid_from=timestamp,
        ingested_at=datetime.now(UTC),
        acl=_repo_acl_snapshot(repo, event.received_at),
        metadata=metadata,
        body=message,
        doc_references=_references_from_text(message, full_name, html_url),
    )

    nodes: list[GraphNodeSpec] = [
        GraphNodeSpec(
            label=NodeLabel.DOCUMENT,
            canonical_id=doc_id,
            properties={"doc_type": DocType.GITHUB_COMMIT.value},
        ),
    ]
    if author not in seen_people:
        nodes.append(
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=author,
                properties={"source_system": SourceSystem.GITHUB.value},
            )
        )
        seen_people.add(author)

    edges: list[GraphEdgeSpec] = [
        GraphEdgeSpec(
            edge_type=EdgeType.AUTHORED,
            from_label=NodeLabel.PERSON,
            from_canonical_id=author,
            to_label=NodeLabel.DOCUMENT,
            to_canonical_id=doc_id,
            valid_from=timestamp,
        ),
        GraphEdgeSpec(
            edge_type=EdgeType.TOUCHES,
            from_label=NodeLabel.DOCUMENT,
            from_canonical_id=doc_id,
            to_label=NodeLabel.REPO,
            to_canonical_id=full_name,
            valid_from=timestamp,
        ),
    ]

    # Co-authors get their own Person node (keyed by email) and an AUTHORED
    # edge to this commit. They are NOT placed in `documents.author_id` —
    # that field is single-valued by design — so they're discoverable only
    # via the graph (Person → AUTHORED → Document) and via the
    # `metadata.co_authors` payload on get_source. Identity resolution
    # across the email and the primary author's GitHub login form is
    # deliberate scope; see TODOS.md.
    for co in co_authors:
        person_id = co["email"]
        if person_id not in seen_people:
            nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=person_id,
                    properties={
                        "source_system": SourceSystem.GITHUB.value,
                        "name": co["name"],
                    },
                )
            )
            seen_people.add(person_id)
        edges.append(
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=person_id,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                valid_from=timestamp,
            )
        )

    edges.extend(
        _mention_edges(
            message,
            full_name,
            from_label=NodeLabel.DOCUMENT,
            from_canonical_id=doc_id,
            valid_from=timestamp,
        )
    )
    return doc, nodes, edges


def _codeowners_artifacts(
    *,
    event: WebhookEvent,
    repo: Mapping[str, Any],
    hydrated: Mapping[str, Any],
) -> tuple[Document, list[GraphNodeSpec], list[GraphEdgeSpec]]:
    full_name = repo.get("full_name") or ""
    content = hydrated.get("codeowners_content")
    path = hydrated.get("codeowners_path") or _CODEOWNERS_PATHS[0]
    head_commit = event.raw_payload.get("head_commit") or {}
    commit_id = head_commit.get("id") or "unknown"
    timestamp = _parse_iso8601(head_commit.get("timestamp"))

    doc_id = f"github:{full_name}:codeowners:{commit_id}"
    source_id = f"{full_name}@codeowners:{commit_id}"
    html_url = f"{repo.get('html_url') or ''}/blob/HEAD/{path}" if path else ""

    ownership_map: dict[str, list[str]] = {}
    edges: list[GraphEdgeSpec] = []
    nodes: list[GraphNodeSpec] = []
    skipped = False

    if isinstance(content, str) and content.strip():
        ownership_map = parse_codeowners(content)
        for pattern, owners in ownership_map.items():
            for owner in owners:
                is_team = "/" in owner
                owner_id = owner.lstrip("@")
                nodes.append(
                    GraphNodeSpec(
                        label=NodeLabel.PERSON,
                        canonical_id=owner_id,
                        properties={
                            "source_system": SourceSystem.GITHUB.value,
                            "is_team": is_team,
                        },
                    )
                )
                edges.append(
                    GraphEdgeSpec(
                        edge_type=EdgeType.OWNS,
                        from_label=NodeLabel.PERSON,
                        from_canonical_id=owner_id,
                        to_label=NodeLabel.REPO,
                        to_canonical_id=full_name,
                        properties={"path_pattern": pattern, "is_team": is_team},
                        valid_from=timestamp,
                    )
                )
    else:
        skipped = True

    body = content if isinstance(content, str) else ""
    doc = Document(
        doc_id=doc_id,
        customer_id=event.customer_id,
        source_system=SourceSystem.GITHUB,
        source_id=source_id,
        source_url=html_url,
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.GITHUB_CODEOWNERS,
        content_type="text/plain",
        content_hash=_sha256(f"{doc_id}|{body}"),
        title=f"CODEOWNERS @ {full_name}",
        body_preview=body[:280] if body else None,
        body_size_bytes=len(body.encode("utf-8")),
        body_token_count=count_tokens(body),
        author_id=None,
        created_at=timestamp,
        updated_at=timestamp,
        valid_from=timestamp,
        ingested_at=datetime.now(UTC),
        acl=_repo_acl_snapshot(repo, event.received_at),
        metadata={
            "repo_full_name": full_name,
            "path": path,
            "ownership_map": ownership_map,
            "codeowners_fetch_skipped": skipped,
            "visibility": _repo_visibility(repo),
        },
        body=body,
    )

    nodes.append(
        GraphNodeSpec(
            label=NodeLabel.DOCUMENT,
            canonical_id=doc_id,
            properties={"doc_type": DocType.GITHUB_CODEOWNERS.value},
        )
    )
    # Document → Repo for entity-filter reachability.
    edges.append(
        GraphEdgeSpec(
            edge_type=EdgeType.TOUCHES,
            from_label=NodeLabel.DOCUMENT,
            from_canonical_id=doc_id,
            to_label=NodeLabel.REPO,
            to_canonical_id=full_name,
            valid_from=timestamp,
        )
    )
    return doc, nodes, edges


def parse_codeowners(text: str) -> dict[str, list[str]]:
    """Parse a CODEOWNERS file into a {pattern: [owner, ...]} mapping.

    Each non-blank, non-comment line is `<pattern> <owner1> [<owner2> ...]`.
    Owners are `@user` or `@org/team-name`; a bare owner without `@` is also
    accepted (GitHub tolerates email-style owners in some configurations).
    Inline `#` comments are stripped.

    The function is deliberately forgiving: malformed lines (pattern with no
    owners) are skipped rather than raising — a broken CODEOWNERS file should
    not fail the whole push ingest.
    """
    result: dict[str, list[str]] = {}
    if not text:
        return result

    for raw_line in text.splitlines():
        # Strip inline comments. GitHub treats `#` anywhere as start of comment.
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            # Pattern with no owner — nothing to emit.
            continue

        pattern = parts[0]
        owners: list[str] = []
        for owner in parts[1:]:
            owner = owner.strip()
            if not owner:
                continue
            owners.append(owner)
        if owners:
            # Later duplicate lines override (same as GitHub's "last match wins").
            result[pattern] = owners
    return result


def _references_from_text(
    text: str, repo_full_name: str, source_html_url: str
) -> list[DocRef]:
    refs: list[DocRef] = []
    if not text:
        return refs

    # Cross-repo refs first so `owner/repo#123` isn't double-counted as `#123`.
    seen_spans: list[tuple[int, int]] = []
    for match in _CROSS_REPO_REF.finditer(text):
        other_repo, number = match.group(1), match.group(2)
        seen_spans.append(match.span())
        refs.append(
            DocRef(
                external_url=f"https://github.com/{other_repo}/issues/{number}",
                ref_type=RefType.MENTIONS,
            )
        )

    for match in _SAME_REPO_REF.finditer(text):
        start, _end = match.span()
        if any(s <= start < e for s, e in seen_spans):
            continue
        number = match.group(1)
        if not repo_full_name:
            continue
        refs.append(
            DocRef(
                external_url=f"https://github.com/{repo_full_name}/issues/{number}",
                ref_type=RefType.MENTIONS,
            )
        )

    # Preserve the source URL as a self-link for traceability.
    if source_html_url:
        refs.append(DocRef(external_url=source_html_url, ref_type=RefType.LINKS_TO))
    return refs


def _mention_edges(
    text: str,
    repo_full_name: str,
    *,
    from_label: NodeLabel,
    from_canonical_id: str,
    valid_from: datetime,
) -> list[GraphEdgeSpec]:
    edges: list[GraphEdgeSpec] = []
    if not text:
        return edges

    seen_spans: list[tuple[int, int]] = []
    for match in _CROSS_REPO_REF.finditer(text):
        other_repo, number = match.group(1), match.group(2)
        seen_spans.append(match.span())
        edges.append(
            GraphEdgeSpec(
                edge_type=EdgeType.MENTIONS,
                from_label=from_label,
                from_canonical_id=from_canonical_id,
                to_label=NodeLabel.ISSUE,
                to_canonical_id=f"{other_repo}#{number}",
                valid_from=valid_from,
            )
        )

    for match in _SAME_REPO_REF.finditer(text):
        start, _ = match.span()
        if any(s <= start < e for s, e in seen_spans):
            continue
        number = match.group(1)
        if not repo_full_name:
            continue
        edges.append(
            GraphEdgeSpec(
                edge_type=EdgeType.MENTIONS,
                from_label=from_label,
                from_canonical_id=from_canonical_id,
                to_label=NodeLabel.ISSUE,
                to_canonical_id=f"{repo_full_name}#{number}",
                valid_from=valid_from,
            )
        )
    return edges


# Expose the matcher so tests can introspect without import gymnastics.
__all__ = ["GitHubConnector", "parse_codeowners"]
