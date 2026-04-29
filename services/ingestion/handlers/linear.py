"""Linear connector — issue + comment ingestion.

Covers:
- Webhook `Issue` and `Comment` types with `action in {create, update}`
- Signature verification via `Linear-Signature` (HMAC-SHA256 of raw body)
- Document shape: DocType.LINEAR_ISSUE per-issue, DocType.LINEAR_COMMENT per-comment
- Graph: Ticket + Person + Document nodes; AUTHORED / ASSIGNED_TO / MENTIONS edges

ACL: Linear resources are scoped by workspace (`organizationId`) and team.
Phase 0 captures a snapshot with the workspace as a WORKSPACE principal and
the team as a GROUP principal. Per-member visibility (private teams, issue
subscribers) comes in Phase 1 alongside backfill of team membership.
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
from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from shared.constants import (
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
from shared.exceptions import (
    InvalidWebhookPayload,
    PermanentSourceError,
    TransientSourceError,
)
from shared.logging import get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    DocRef,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

log = get_logger(__name__)

_LINEAR_OAUTH_AUTHORIZE = "https://linear.app/oauth/authorize"
_LINEAR_OAUTH_TOKEN = "https://api.linear.app/oauth/token"
_LINEAR_SIGNATURE_HEADER = "linear-signature"

# Linear's webhook payload `type` values we process. Anything else
# (Project, Cycle, Reaction, ...) is ignored at the parse step.
_TYPE_ISSUE = "Issue"
_TYPE_COMMENT = "Comment"
_HANDLED_TYPES = frozenset({_TYPE_ISSUE, _TYPE_COMMENT})

# Linear's `action` values we process. `remove` produces a tombstone
# (deleted_at set, chunks reconciled as all-removed by the normalizer diff).
_HANDLED_ACTIONS = frozenset({"create", "update", "remove"})

# Resource types used on ACL rows. Keyed to DocType values so the ACL
# layer can join back to the document type without a second lookup.
_RESOURCE_ISSUE = DocType.LINEAR_ISSUE.value
_RESOURCE_COMMENT = DocType.LINEAR_COMMENT.value

# Matches Linear-style issue keys (team key + dash + number): ENG-123, OPS-42, etc.
# The leading `(?<![A-Z0-9-])` + trailing `(?![A-Z0-9-])` guards keep us from
# matching inside larger identifiers.
_LINEAR_REF_RE = re.compile(r"(?<![A-Z0-9-])([A-Z][A-Z0-9]{1,9}-\d+)(?![A-Z0-9-])")

# Fallback URL ref regex — bare URLs mentioned in an issue body often
# link to runbooks / dashboards worth tracking as cross-doc references.
_URL_RE = re.compile(r"https?://\S+")


@register_connector(SourceSystem.LINEAR)
class LinearConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.LINEAR
    display_name: ClassVar[str] = "Linear"

    # ------------------------------------------------------------------
    # 1. signature verification
    # ------------------------------------------------------------------

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        secret = self.settings.linear_webhook_secret
        if secret is None:
            # Dev mode: accept unsigned payloads only when running locally.
            return self.settings.is_local

        sig = _header(headers, _LINEAR_SIGNATURE_HEADER)
        if not sig:
            return False

        expected = hmac.new(
            secret.get_secret_value().encode(), raw_body, hashlib.sha256
        ).hexdigest()
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
        type_ = raw_payload.get("type")
        action = raw_payload.get("action")
        data = raw_payload.get("data")

        # Phase 0 only handles Issue and Comment create/update events.
        if type_ not in _HANDLED_TYPES:
            return None
        if action not in _HANDLED_ACTIONS:
            return None
        if not isinstance(data, dict):
            raise InvalidWebhookPayload("linear payload missing 'data' dict")

        resource_id = data.get("id")
        if not resource_id:
            raise InvalidWebhookPayload("linear payload missing data.id")

        # Linear sends ISO-8601 with a `Z` suffix. `fromisoformat` handles
        # full offsets in 3.11+ including trailing Z.
        received_at_raw = raw_payload.get("createdAt")
        received_at = (
            _parse_iso(received_at_raw) if received_at_raw else datetime.now(UTC)
        )

        # Prefix with the type so (issue:X, comment:X) stay distinct ids even
        # if Linear ever reuses the same uuid across resource kinds.
        # Include action + a stable event clock so a create/update/remove cycle
        # on the same resource id produces distinct queue rows.
        #
        # The clock must be STABLE across webhook retries (Linear replays the
        # same payload bytes on delivery failure). `createdAt` is always present
        # in real Linear webhooks; if it's missing, fall back to `data.updatedAt`
        # so retries still dedupe. Never use wall-clock now() here — that would
        # bypass the UNIQUE (customer_id, source_system, source_event_id) constraint.
        stable_clock = received_at_raw or data.get("updatedAt") or data.get("createdAt")
        if not stable_clock:
            raise InvalidWebhookPayload(
                "linear payload missing createdAt/updatedAt — cannot compute stable source_event_id"
            )
        # Linear's createdAt/updatedAt is per-second, so two distinct edits
        # within the same second by the same user produce identical clocks
        # and the second event is silently dropped by the queue's UNIQUE
        # constraint. A deterministic hash of the data dict disambiguates
        # them while staying stable across webhook retries (same payload
        # bytes => same hash => dedup still works).
        payload_fp = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        source_event_id = (
            f"{type_.lower()}:{resource_id}:{action}:{stable_clock}:{payload_fp}"
        )

        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=received_at,
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "type": type_,
                "action": action,
                "resource_id": resource_id,
                "organization_id": raw_payload.get("organizationId"),
                "team_id": data.get("teamId") or (data.get("team") or {}).get("id"),
                "issue_id": data.get("issueId") or (data.get("issue") or {}).get("id"),
            },
        )

    # ------------------------------------------------------------------
    # 3. hydration
    # ------------------------------------------------------------------

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: IntegrationToken | None,
    ) -> dict[str, Any]:
        # Phase 0: nothing to hydrate. A Comment webhook only ships the
        # parent issue id, so a future pass could GraphQL-fetch the
        # parent issue title/state here to enrich `metadata` and build
        # a proper title for the comment document. Left as a no-op to
        # keep Phase 0 webhook → normalize fully in-memory.
        return {}

    # ------------------------------------------------------------------
    # 4. normalization
    # ------------------------------------------------------------------

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        payload = event.raw_payload
        type_ = payload.get("type")
        action = payload.get("action")
        data = payload.get("data")
        org_id = payload.get("organizationId") or ""

        if not isinstance(data, dict) or not type_:
            return NormalizationResult(skipped_reason="missing type/data after parse")

        is_delete = action == "remove"

        if type_ == _TYPE_ISSUE:
            return self._normalize_issue(event, data, org_id, is_delete=is_delete)
        if type_ == _TYPE_COMMENT:
            return self._normalize_comment(event, data, org_id, is_delete=is_delete)

        # Defensive — parse_webhook_event should have filtered these.
        return NormalizationResult(skipped_reason=f"unsupported linear type {type_}")

    # ---- issue normalization -----------------------------------------

    def _normalize_issue(
        self,
        event: WebhookEvent,
        data: Mapping[str, Any],
        org_id: str,
        *,
        is_delete: bool = False,
    ) -> NormalizationResult:
        issue_id = data.get("id")
        if not issue_id:
            return NormalizationResult(skipped_reason="linear issue missing id")

        title = data.get("title") or None
        body = data.get("description") or ""
        source_url = data.get("url") or ""

        # For a delete, body is empty and we write a tombstone. The content_hash
        # must differ from the prior live version so the normalizer bumps the
        # version and the chunk diff marks all previous chunks stale.
        deleted_at = event.received_at if is_delete else None
        if is_delete:
            body = ""

        # Linear sometimes ships full objects, sometimes just ids. Support both.
        author_id = _pick_person_id(data, "creator", "creatorId")
        assignee_id = _pick_person_id(data, "assignee", "assigneeId")
        team_id = data.get("teamId") or (data.get("team") or {}).get("id") or ""
        state = (data.get("state") or {}).get("name") if isinstance(
            data.get("state"), dict
        ) else None
        priority = data.get("priority")

        created = _parse_iso(data.get("createdAt")) or event.received_at
        updated = _parse_iso(data.get("updatedAt")) or created

        doc_id = f"linear:{org_id}:issue:{issue_id}"
        if is_delete:
            content_hash = _sha256(
                f"{doc_id}|__deleted__|{event.received_at.isoformat()}"
            )
        else:
            content_hash = _sha256(f"{doc_id}|{body}|{title or ''}")

        principals = [
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=org_id,
                permission=Permission.READ,
            )
        ]
        if team_id:
            principals.append(
                ACLPrincipal(
                    principal_type=PrincipalType.GROUP,
                    principal_id=team_id,
                    permission=Permission.READ,
                )
            )

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.LINEAR,
            source_id=f"issue:{issue_id}",
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.LINEAR_ISSUE,
            content_type="text/markdown",
            content_hash=content_hash,
            title=title,
            body_preview=body[:280] if body else None,
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=author_id,
            created_at=created,
            updated_at=updated,
            valid_from=created,
            deleted_at=deleted_at,
            ingested_at=datetime.now(UTC),
            acl=ACLSnapshot(principals=principals, captured_at=event.received_at),
            metadata={
                "body": body,
                "identifier": data.get("identifier"),
                "organization_id": org_id,
                "team_id": team_id,
                "state": state,
                "priority": priority,
                "assignee_id": assignee_id,
            },
            doc_references=_references_from_text(body),
        )

        nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.TICKET,
                canonical_id=issue_id,
                properties={
                    "source_system": SourceSystem.LINEAR.value,
                    "identifier": data.get("identifier"),
                    "team_id": team_id,
                    "state": state,
                    "priority": priority,
                    "assignee_id": assignee_id,
                },
            ),
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": DocType.LINEAR_ISSUE.value},
            ),
        ]
        if author_id:
            nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=author_id,
                    properties={"source_system": SourceSystem.LINEAR.value},
                )
            )
        if assignee_id and assignee_id != author_id:
            nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=assignee_id,
                    properties={"source_system": SourceSystem.LINEAR.value},
                )
            )

        edges: list[GraphEdgeSpec] = [
            # Document → Ticket so list-pipeline entity filter
            # ("last ticket in PROJ-X") can find the doc.
            GraphEdgeSpec(
                edge_type=EdgeType.LINKED_FROM,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.TICKET,
                to_canonical_id=issue_id,
                valid_from=created,
            ),
        ]
        if author_id:
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.AUTHORED,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id=author_id,
                    to_label=NodeLabel.DOCUMENT,
                    to_canonical_id=doc_id,
                    valid_from=created,
                )
            )
        if assignee_id:
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.ASSIGNED_TO,
                    from_label=NodeLabel.TICKET,
                    from_canonical_id=issue_id,
                    to_label=NodeLabel.PERSON,
                    to_canonical_id=assignee_id,
                    valid_from=created,
                )
            )
        # Cross-ticket mentions (e.g. "see [OPS-42]") emit MENTIONS edges from
        # the document to the referenced ticket. The referenced ticket may not
        # yet have a node — the graph writer upserts by (label, canonical_id).
        for ref_key in _extract_issue_refs(body, exclude=data.get("identifier")):
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.MENTIONS,
                    from_label=NodeLabel.DOCUMENT,
                    from_canonical_id=doc_id,
                    to_label=NodeLabel.TICKET,
                    to_canonical_id=ref_key,
                    valid_from=created,
                )
            )

        acl_rows = [
            ACLSnapshotRow(
                source_system=SourceSystem.LINEAR,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=org_id,
                resource_type=_RESOURCE_ISSUE,
                resource_id=issue_id,
                permission=Permission.READ,
                valid_from=created,
            )
        ]
        if team_id:
            acl_rows.append(
                ACLSnapshotRow(
                    source_system=SourceSystem.LINEAR,
                    principal_type=PrincipalType.GROUP,
                    principal_id=team_id,
                    resource_type=_RESOURCE_ISSUE,
                    resource_id=issue_id,
                    permission=Permission.READ,
                    valid_from=created,
                )
            )

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ---- comment normalization ---------------------------------------

    def _normalize_comment(
        self,
        event: WebhookEvent,
        data: Mapping[str, Any],
        org_id: str,
        *,
        is_delete: bool = False,
    ) -> NormalizationResult:
        comment_id = data.get("id")
        if not comment_id:
            return NormalizationResult(skipped_reason="linear comment missing id")

        body = data.get("body") or ""
        source_url = data.get("url") or ""

        deleted_at = event.received_at if is_delete else None
        if is_delete:
            body = ""
        author_id = _pick_person_id(data, "user", "userId")
        issue_id = data.get("issueId") or (data.get("issue") or {}).get("id")
        team_id = (data.get("team") or {}).get("id") or (
            (data.get("issue") or {}).get("teamId")
        )

        created = _parse_iso(data.get("createdAt")) or event.received_at
        updated = _parse_iso(data.get("updatedAt")) or created

        doc_id = f"linear:{org_id}:comment:{comment_id}"
        parent_doc_id = (
            f"linear:{org_id}:issue:{issue_id}" if issue_id else None
        )
        if is_delete:
            content_hash = _sha256(
                f"{doc_id}|__deleted__|{event.received_at.isoformat()}"
            )
        else:
            content_hash = _sha256(f"{doc_id}|{body}|")

        principals = [
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=org_id,
                permission=Permission.READ,
            )
        ]
        if team_id:
            principals.append(
                ACLPrincipal(
                    principal_type=PrincipalType.GROUP,
                    principal_id=team_id,
                    permission=Permission.READ,
                )
            )

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.LINEAR,
            source_id=f"comment:{comment_id}",
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.LINEAR_COMMENT,
            content_type="text/markdown",
            content_hash=content_hash,
            title=_derive_title(body),
            body_preview=body[:280] if body else None,
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=author_id,
            created_at=created,
            updated_at=updated,
            valid_from=created,
            deleted_at=deleted_at,
            ingested_at=datetime.now(UTC),
            parent_doc_id=parent_doc_id,
            acl=ACLSnapshot(principals=principals, captured_at=event.received_at),
            metadata={
                "body": body,
                "organization_id": org_id,
                "team_id": team_id,
                "issue_id": issue_id,
            },
            doc_references=_references_from_text(body),
        )

        nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": DocType.LINEAR_COMMENT.value},
            ),
        ]
        if issue_id:
            nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.TICKET,
                    canonical_id=issue_id,
                    properties={
                        "source_system": SourceSystem.LINEAR.value,
                        "team_id": team_id,
                    },
                )
            )
        if author_id:
            nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=author_id,
                    properties={"source_system": SourceSystem.LINEAR.value},
                )
            )

        edges: list[GraphEdgeSpec] = []
        if issue_id:
            # Document → Ticket so a comment doc surfaces under "tickets in PROJ-X"
            # filtering (the comment lives on a ticket; the entity filter walks
            # to the parent ticket via this edge).
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.LINKED_FROM,
                    from_label=NodeLabel.DOCUMENT,
                    from_canonical_id=doc_id,
                    to_label=NodeLabel.TICKET,
                    to_canonical_id=issue_id,
                    valid_from=created,
                )
            )
        if author_id:
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.AUTHORED,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id=author_id,
                    to_label=NodeLabel.DOCUMENT,
                    to_canonical_id=doc_id,
                    valid_from=created,
                )
            )
        # Comments frequently cross-reference other tickets ("dupe of [ENG-99]").
        for ref_key in _extract_issue_refs(body, exclude=None):
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.MENTIONS,
                    from_label=NodeLabel.DOCUMENT,
                    from_canonical_id=doc_id,
                    to_label=NodeLabel.TICKET,
                    to_canonical_id=ref_key,
                    valid_from=created,
                )
            )

        acl_rows = [
            ACLSnapshotRow(
                source_system=SourceSystem.LINEAR,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=org_id,
                resource_type=_RESOURCE_COMMENT,
                resource_id=comment_id,
                permission=Permission.READ,
                valid_from=created,
            )
        ]
        if team_id:
            acl_rows.append(
                ACLSnapshotRow(
                    source_system=SourceSystem.LINEAR,
                    principal_type=PrincipalType.GROUP,
                    principal_id=team_id,
                    resource_type=_RESOURCE_COMMENT,
                    resource_id=comment_id,
                    permission=Permission.READ,
                    valid_from=created,
                )
            )

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ------------------------------------------------------------------
    # 5. OAuth install (for completeness — real redirect wired in Tier 7)
    # ------------------------------------------------------------------

    def oauth_install_url(self, customer_id: str, redirect_uri: str) -> str:
        cid = self.settings.linear_client_id
        if not cid:
            from shared.exceptions import MissingSecret

            raise MissingSecret("LINEAR_CLIENT_ID not configured")
        scopes = ",".join(["read", "write", "issues:create"])
        return (
            f"{_LINEAR_OAUTH_AUTHORIZE}"
            f"?client_id={cid}&redirect_uri={redirect_uri}"
            f"&response_type=code&scope={scopes}&state={customer_id}"
        )

    async def exchange_oauth_code(
        self,
        code: str | None,
        redirect_uri: str,
        extra_params: dict[str, str] | None = None,
    ) -> IntegrationToken:
        cid = self.settings.linear_client_id
        secret = self.settings.linear_client_secret
        if not cid or secret is None:
            from shared.exceptions import MissingSecret

            raise MissingSecret("LINEAR_CLIENT_ID / LINEAR_CLIENT_SECRET not configured")

        resp = await self.http.post(
            _LINEAR_OAUTH_TOKEN,
            data={
                "client_id": cid,
                "client_secret": secret.get_secret_value(),
                "redirect_uri": redirect_uri,
                "code": code,
                "grant_type": "authorization_code",
            },
        )
        if resp.status_code >= 500:
            # 5xx — Linear edge / app outage. Caller can retry.
            raise TransientSourceError(
                f"linear /oauth/token returned {resp.status_code}",
                status=resp.status_code,
                body=resp.text[:500],
            )
        if resp.status_code >= 400:
            # 4xx — bad code, redirect_uri mismatch, revoked client, etc.
            # No retry will help; surface the response so the caller can
            # render a useful message in the dashboard.
            raise PermanentSourceError(
                f"linear /oauth/token returned {resp.status_code}",
                status=resp.status_code,
                body=resp.text[:500],
            )
        body = resp.json()
        access_token = body.get("access_token")
        if not access_token:
            raise PermanentSourceError(
                "linear /oauth/token 200 but missing access_token",
                body=str(body)[:500],
            )
        return IntegrationToken(
            customer_id="",  # caller fills in — connector does not know the tenant
            source_system=SourceSystem.LINEAR,
            access_token=access_token,
            scope=body.get("scope"),
        )

    # ------------------------------------------------------------------
    # 7. workspace identification
    # ------------------------------------------------------------------

    async def identify_workspaces(self, token: IntegrationToken):  # type: ignore[override]
        """GraphQL `viewer { organization { id name urlKey } }` to get org info."""
        from shared.logging import get_logger
        from shared.models import ExternalWorkspaceRef

        lg = get_logger(__name__)
        query = "{ viewer { organization { id name urlKey } } }"
        try:
            resp = await self.http.post(
                "https://api.linear.app/graphql",
                headers={
                    "Authorization": f"Bearer {token.access_token}",
                    "Content-Type": "application/json",
                },
                json={"query": query},
            )
        except Exception as exc:
            lg.warning("linear.identify_workspaces_failed", error=str(exc))
            return []
        if resp.status_code != 200:
            return []
        org = (
            ((resp.json().get("data") or {}).get("viewer") or {}).get("organization")
            or {}
        )
        org_id = org.get("id")
        if not org_id:
            return []
        return [
            ExternalWorkspaceRef(
                external_id=org_id,
                external_name=org.get("name"),
                metadata={"url_key": org.get("urlKey")},
            )
        ]

    def extract_external_id_from_payload(self, headers, raw_payload):
        org_id = raw_payload.get("organizationId")
        return str(org_id) if org_id else None

    # ------------------------------------------------------------------
    # 5. backfill
    # ------------------------------------------------------------------

    async def backfill(
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ):
        """Paginate Linear issues + comments via GraphQL.

        Cursor is Linear's pageInfo.endCursor (opaque string). Yields
        synthetic webhook events with `type=Issue, action=create` shaped
        identically to live Linear webhooks so the normalizer doesn't care.

        Phase 0 limitation: we only backfill issues (and their nested
        comments per-issue). Projects, cycles, docs, etc. are ignored.
        """
        from shared.models import WebhookEvent

        page_cursor = cursor
        # Get organizationId once to populate webhook-shaped top-level field.
        org_id = await _fetch_org_id(self.http, token.access_token) or ""

        while True:
            query = _BACKFILL_ISSUES_QUERY
            variables: dict[str, Any] = {"first": 50}
            if page_cursor:
                variables["after"] = page_cursor

            try:
                resp = await self.http.post(
                    "https://api.linear.app/graphql",
                    headers={
                        "Authorization": f"Bearer {token.access_token}",
                        "Content-Type": "application/json",
                    },
                    json={"query": query, "variables": variables},
                )
            except Exception as exc:
                log.warning("linear.backfill_http_error", error=str(exc))
                return

            if resp.status_code != 200:
                log.warning(
                    "linear.backfill_non_200",
                    status=resp.status_code,
                    body=resp.text[:500],
                )
                return
            body = resp.json()
            if body.get("errors"):
                # Linear returns 200 with a non-empty `errors` array on
                # GraphQL validation failures (e.g. unknown field). Without
                # this branch the loop would silently see no `data.issues`
                # and the runner would mark the backfill complete with 0
                # events enqueued — exactly the failure mode that masked
                # this handler's bad query for two installs.
                log.warning("linear.backfill_graphql_errors", errors=body["errors"])
                return
            issues = ((body.get("data") or {}).get("issues") or {})
            for node in issues.get("nodes", []):
                # Yield the issue itself as an Issue webhook-shaped event.
                issue_payload = {
                    "action": "create",
                    "type": "Issue",
                    "data": node,
                    "organizationId": org_id,
                    "createdAt": node.get("updatedAt") or node.get("createdAt"),
                    "_cursor": page_cursor,
                }
                yield WebhookEvent(
                    customer_id=customer_id,
                    source_system=SourceSystem.LINEAR,
                    source_event_id=f"issue:{node.get('id')}:backfill",
                    received_at=_parse_iso(
                        node.get("updatedAt") or node.get("createdAt")
                    ) or datetime.now(UTC),
                    payload_s3_key="",
                    raw_payload=issue_payload,
                    headers={},
                )

                # Yield each comment as a Comment event.
                for comment in ((node.get("comments") or {}).get("nodes") or []):
                    comment_payload = {
                        "action": "create",
                        "type": "Comment",
                        "data": {
                            **comment,
                            "issueId": node.get("id"),
                            "issue": {"id": node.get("id"), "identifier": node.get("identifier")},
                            "teamId": (node.get("team") or {}).get("id"),
                        },
                        "organizationId": org_id,
                        "createdAt": comment.get("updatedAt") or comment.get("createdAt"),
                        "_cursor": page_cursor,
                    }
                    yield WebhookEvent(
                        customer_id=customer_id,
                        source_system=SourceSystem.LINEAR,
                        source_event_id=f"comment:{comment.get('id')}:backfill",
                        received_at=_parse_iso(
                            comment.get("updatedAt") or comment.get("createdAt")
                        ) or datetime.now(UTC),
                        payload_s3_key="",
                        raw_payload=comment_payload,
                        headers={},
                    )

            page_info = issues.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                return
            page_cursor = page_info.get("endCursor")
            if not page_cursor:
                return

    # ------------------------------------------------------------------


_BACKFILL_ISSUES_QUERY = """
query Backfill($first: Int!, $after: String) {
  issues(first: $first, after: $after, orderBy: updatedAt) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id identifier title description url createdAt updatedAt
      state { name type }
      priority
      team { id key name }
      creator { id name email }
      assignee { id name email }
      comments(first: 50) {
        nodes { id body url createdAt updatedAt user { id name } }
      }
    }
  }
}
"""


async def _fetch_org_id(http, token: str) -> str | None:
    try:
        resp = await http.post(
            "https://api.linear.app/graphql",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"query": "{ organization { id } }"},
        )
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    return (
        ((resp.json().get("data") or {}).get("organization") or {}).get("id")
    )


# ---- helpers ---------------------------------------------------------------


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    # `datetime.fromisoformat` in 3.11+ accepts trailing `Z`.
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _pick_person_id(
    data: Mapping[str, Any],
    object_key: str,
    id_key: str,
) -> str | None:
    """Linear sometimes sends `creator: {id, ...}`, sometimes just `creatorId`.

    Prefer the nested object id if present — it's what the webhook ships in
    the current format — and fall back to the flat id field so older / API-
    backfilled payloads still resolve.
    """
    obj = data.get(object_key)
    if isinstance(obj, dict):
        oid = obj.get("id")
        if oid:
            return str(oid)
    flat = data.get(id_key)
    if flat:
        return str(flat)
    return None


def _derive_title(text: str) -> str | None:
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    return first_line[:120] if first_line else None


def _extract_issue_refs(text: str, exclude: str | None) -> list[str]:
    """Pull ENG-123 style issue keys out of body text.

    Deduplicates case-sensitively and drops the issue's own identifier when
    provided (no need to MENTIONS-link a ticket to itself).
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _LINEAR_REF_RE.finditer(text):
        key = match.group(1)
        if exclude and key == exclude:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _references_from_text(text: str) -> list[DocRef]:
    """Emit DocRefs for bare URLs + Linear issue keys in the body.

    Issue keys become internal MENTIONS refs keyed by the public Linear URL
    shape; URLs become LINKS_TO refs. The resolver downstream decides whether
    each ref collapses to a known doc_id.
    """
    refs: list[DocRef] = []
    if not text:
        return refs
    seen_urls: set[str] = set()
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,);]")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        refs.append(DocRef(external_url=url, ref_type=RefType.LINKS_TO))
    seen_keys: set[str] = set()
    for match in _LINEAR_REF_RE.finditer(text):
        key = match.group(1)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        refs.append(
            DocRef(
                external_url=f"linear://issue/{key}",
                ref_type=RefType.MENTIONS,
            )
        )
    return refs
