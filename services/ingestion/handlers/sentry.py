"""Sentry connector — error observability source.

Covers:
- `issue` resource with action `created|resolved|unresolved|assigned|archived`
  → `DocType.SENTRY_ISSUE`. One doc per lifecycle transition so the timeline
  is reconstructable.
- `event_alert` / `error` resources → `DocType.SENTRY_EVENT`. **Sampled**: we
  keep exactly ONE representative event per issue (the first event we see,
  containing its stacktrace + tags). Subsequent events for the same issue
  land at the same deterministic (doc_id, content_hash), which the normalizer
  content-hash dedup collapses into a no-op. Live, up-to-the-second event
  data (counts, fresh stacks, per-release breakdowns) is served directly
  from Sentry's API by a separate tool surface, not by this index.

  Rationale: Sentry's event firehose is volume-heavy and retrieval-value-
  thin — the 2,000,000th occurrence of an issue looks identical to the 100th
  for reasoning purposes. The sample gives agents enough content to describe
  the error; fresh state comes from Sentry's API directly.

- Signature verification via `Sentry-Hook-Signature` (HMAC-SHA256 over the raw
  body with the integration's client secret as key).

ACL: Sentry permissions are workspace-wide by default, so principals are
(a) the whole workspace (organization.slug) and (b) a project-scoped group
(`sentry-project:<slug>`) so finer-grained ACL rules can land later without
a re-ingest. No per-user ACL capture in Phase 0.
"""

from __future__ import annotations

import hashlib
import hmac
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
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload
from shared.logging import get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

log = get_logger(__name__)

# --- Sentry header + resource constants -------------------------------------

_HDR_RESOURCE = "sentry-hook-resource"
_HDR_SIGNATURE = "sentry-hook-signature"

_RESOURCE_ISSUE = "issue"
_RESOURCE_EVENT_ALERT = "event_alert"
_RESOURCE_ERROR = "error"
_RESOURCE_INSTALLATION = "installation"

# Issue actions we persist. Everything else on the `issue` resource is ignored.
_ISSUE_ACTIONS = frozenset(
    {"created", "resolved", "unresolved", "assigned", "archived"}
)

# Actor types we never treat as the authoring human — Sentry uses these for
# rule-triggered / system-initiated webhooks.
_SYSTEM_ACTOR_TYPES = frozenset({"application", "system"})

_SENTRY_BOT_IDS = frozenset({"sentry"})

# Max stacktrace frames embedded in an event body. Beyond this is noise — the
# deep frames tend to be framework internals and blow up chunking cost.
_MAX_FRAMES_IN_BODY = 10


@register_connector(SourceSystem.SENTRY)
class SentryConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.SENTRY
    display_name: ClassVar[str] = "Sentry"

    # ------------------------------------------------------------------
    # 1. signature verification
    # ------------------------------------------------------------------

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        secret = self.settings.sentry_webhook_secret
        if secret is None:
            # Dev mode: accept unsigned payloads only when running locally.
            return self.settings.is_local

        sig = _header(headers, _HDR_SIGNATURE)
        if not sig:
            return False

        expected = hmac.new(
            secret.get_secret_value().encode(),
            raw_body,
            hashlib.sha256,
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
        resource = (_header(headers, _HDR_RESOURCE) or "").lower()

        # Installation / lifecycle hooks aren't content — ignore.
        if resource in {"", _RESOURCE_INSTALLATION}:
            return None

        data = raw_payload.get("data")
        if not isinstance(data, dict):
            raise InvalidWebhookPayload("sentry payload missing 'data' dict")

        if resource == _RESOURCE_ISSUE:
            return self._parse_issue(raw_payload, data)

        if resource in {_RESOURCE_EVENT_ALERT, _RESOURCE_ERROR}:
            return self._parse_event(raw_payload, data)

        # Unknown resource type — skip instead of crash. Keeps forward-compat
        # when Sentry adds new hook resources we haven't wired up yet.
        return None

    def _parse_issue(
        self,
        raw_payload: Mapping[str, Any],
        data: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        action = raw_payload.get("action")
        if action not in _ISSUE_ACTIONS:
            return None

        issue = data.get("issue")
        if not isinstance(issue, dict):
            raise InvalidWebhookPayload("sentry issue payload missing 'data.issue'")

        issue_id = issue.get("id")
        if not issue_id:
            raise InvalidWebhookPayload("sentry issue missing id")

        received_at = _parse_iso8601(issue.get("lastSeen")) or datetime.now(UTC)

        return WebhookParseResult(
            source_event_id=f"issue:{issue_id}:{action}",
            received_at=received_at,
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "resource": _RESOURCE_ISSUE,
                "action": action,
                "issue_id": issue_id,
                "project_slug": _project_slug(raw_payload, issue),
            },
        )

    def _parse_event(
        self,
        raw_payload: Mapping[str, Any],
        data: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        ev = data.get("event")
        if not isinstance(ev, dict):
            raise InvalidWebhookPayload("sentry event payload missing 'data.event'")

        event_id = ev.get("event_id")
        if not event_id:
            raise InvalidWebhookPayload("sentry event missing event_id")

        received_at = _parse_iso8601(ev.get("timestamp")) or datetime.now(UTC)

        return WebhookParseResult(
            source_event_id=f"event:{event_id}",
            received_at=received_at,
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "resource": _RESOURCE_EVENT_ALERT,
                "event_id": event_id,
                "group_id": ev.get("groupID"),
                "project_slug": _project_slug(raw_payload, ev),
            },
        )

    # ------------------------------------------------------------------
    # 3. normalization
    # ------------------------------------------------------------------

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        resource = (event.headers.get(_HDR_RESOURCE) or "").lower()
        if not resource:
            # Headers may have been stored lowercased or titlecased — scan.
            resource = (_header(event.headers, _HDR_RESOURCE) or "").lower()

        if resource == _RESOURCE_ISSUE:
            return self._normalize_issue(event)
        if resource in {_RESOURCE_EVENT_ALERT, _RESOURCE_ERROR}:
            return self._normalize_event(event)

        return NormalizationResult(
            skipped_reason=f"unsupported sentry resource: {resource!r}"
        )

    # ---- issue path ---------------------------------------------------

    def _normalize_issue(self, event: WebhookEvent) -> NormalizationResult:
        payload = event.raw_payload
        action = payload.get("action", "created")
        data = payload.get("data") or {}
        issue = data.get("issue") or {}

        issue_id = issue.get("id")
        if not issue_id:
            return NormalizationResult(skipped_reason="missing issue.id")

        project_slug = _project_slug(payload, issue)
        project_id = _project_id(payload, issue)
        org_slug = _org_slug(payload)
        platform = _platform(payload, issue)

        culprit = issue.get("culprit") or ""
        level = issue.get("level")
        title = issue.get("title") or issue.get("shortId") or f"Sentry issue {issue_id}"
        source_url = issue.get("url") or issue.get("permalink") or ""
        first_seen = _parse_iso8601(issue.get("firstSeen")) or event.received_at
        last_seen = _parse_iso8601(issue.get("lastSeen")) or event.received_at

        # Body = culprit + metadata signature. Events carry the stacktrace;
        # the issue doc is the stable summary.
        body = _issue_body(issue)
        doc_id = f"sentry:issue:{issue_id}"
        content_hash = _sha256(f"{doc_id}|{body}|{culprit}|{level or ''}")

        assignee = _assignee(issue)
        actor_id = _human_actor_id(payload)
        author_id = assignee or actor_id

        acl_principals = [
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=org_slug,
                permission=Permission.READ,
            ),
            ACLPrincipal(
                principal_type=PrincipalType.GROUP,
                principal_id=f"sentry-project:{project_slug}",
                permission=Permission.READ,
            ),
        ]

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.SENTRY,
            source_id=issue_id,
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.SENTRY_ISSUE,
            content_type="text/plain",
            content_hash=content_hash,
            title=title[:240] if title else None,
            body_preview=body[:280],
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=author_id,
            created_at=first_seen,
            updated_at=last_seen,
            valid_from=last_seen,
            ingested_at=datetime.now(UTC),
            acl=ACLSnapshot(principals=acl_principals, captured_at=event.received_at),
            metadata={
                "body": body,
                "action": action,
                "project_slug": project_slug,
                "project_id": project_id,
                "platform": platform,
                "level": level,
                "status": issue.get("status"),
                "short_id": issue.get("shortId"),
                "environment": None,
                "release": None,
                "tags": [],
            },
        )

        # --- graph ----------------------------------------------------
        nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.ERROR_GROUP,
                canonical_id=issue_id,
                properties={
                    "culprit": culprit,
                    "first_seen": first_seen.isoformat(),
                    "last_seen": last_seen.isoformat(),
                    "level": level,
                },
            ),
            GraphNodeSpec(
                label=NodeLabel.SERVICE,
                canonical_id=project_slug,
                properties=_service_properties(payload, platform),
            ),
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": doc.doc_type.value},
            ),
        ]

        edges: list[GraphEdgeSpec] = [
            GraphEdgeSpec(
                edge_type=EdgeType.FIRES_IN,
                from_label=NodeLabel.ERROR_GROUP,
                from_canonical_id=issue_id,
                to_label=NodeLabel.SERVICE,
                to_canonical_id=project_slug,
                valid_from=first_seen,
            ),
            # Document → ErrorGroup + Document → Service so the
            # list-pipeline entity filter can find a Sentry doc under
            # either "errors in service api" or "issues in error_group X".
            GraphEdgeSpec(
                edge_type=EdgeType.LINKED_FROM,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.ERROR_GROUP,
                to_canonical_id=issue_id,
                valid_from=first_seen,
            ),
            # LINKED_FROM (not FIRES_IN) — only the ERROR_GROUP→SERVICE
            # edge represents "this error fires in this service" semantics.
            # The doc just references the service.
            GraphEdgeSpec(
                edge_type=EdgeType.LINKED_FROM,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.SERVICE,
                to_canonical_id=project_slug,
                valid_from=first_seen,
            ),
        ]

        if assignee:
            nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=assignee,
                    properties={"source_system": SourceSystem.SENTRY.value},
                )
            )
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.ASSIGNED_TO,
                    from_label=NodeLabel.ERROR_GROUP,
                    from_canonical_id=issue_id,
                    to_label=NodeLabel.PERSON,
                    to_canonical_id=assignee,
                    valid_from=last_seen,
                )
            )

        # AUTHORED edge only when a human actor triggered the webhook. Sentry's
        # own "rule fired" hooks have an application actor — don't attribute.
        if actor_id:
            # Ensure node exists even when the actor isn't the assignee.
            if actor_id != assignee:
                nodes.append(
                    GraphNodeSpec(
                        label=NodeLabel.PERSON,
                        canonical_id=actor_id,
                        properties={"source_system": SourceSystem.SENTRY.value},
                    )
                )
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.AUTHORED,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id=actor_id,
                    to_label=NodeLabel.DOCUMENT,
                    to_canonical_id=doc_id,
                    valid_from=last_seen,
                )
            )

        acl_rows = [
            ACLSnapshotRow(
                source_system=SourceSystem.SENTRY,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=org_slug,
                resource_type="sentry.issue",
                resource_id=issue_id,
                permission=Permission.READ,
                valid_from=first_seen,
            ),
            ACLSnapshotRow(
                source_system=SourceSystem.SENTRY,
                principal_type=PrincipalType.GROUP,
                principal_id=f"sentry-project:{project_slug}",
                resource_type="sentry.issue",
                resource_id=issue_id,
                permission=Permission.READ,
                valid_from=first_seen,
            ),
        ]

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ---- event path (sampled: one representative event per issue) -----

    def _normalize_event(self, event: WebhookEvent) -> NormalizationResult:
        """Produce a single representative sample doc per Sentry issue.

        Identity collapses to (issue, sample) — all events for the same issue
        produce the same `doc_id` AND the same deterministic `content_hash`, so
        after the first event writes the sample row, subsequent events hit
        the content_hash dedup in the normalizer and no-op.

        An event with no `groupID` is skipped entirely — without an anchor to
        the parent issue, a sample isn't retrievable via the graph and just
        clutters the index.
        """
        payload = event.raw_payload
        data = payload.get("data") or {}
        ev = data.get("event") or {}

        event_id = ev.get("event_id")
        if not event_id:
            return NormalizationResult(skipped_reason="missing event.event_id")

        group_id = ev.get("groupID")
        if not group_id:
            return NormalizationResult(
                skipped_reason="sentry event without groupID — no issue to anchor sample to"
            )

        project_slug = _project_slug(payload, ev)
        project_id = _project_id(payload, ev)
        org_slug = _org_slug(payload)
        platform = _platform(payload, ev)
        environment = ev.get("environment")
        release = ev.get("release")
        culprit = ev.get("culprit") or ""
        title = ev.get("title") or f"Sentry issue {group_id} sample"
        source_url = ev.get("url") or ev.get("web_url") or ""
        occurred_at = _parse_iso8601(ev.get("timestamp")) or event.received_at

        body = _event_body(ev)

        # Deterministic identity per issue: every event for the same issue
        # produces these exact two strings. The normalizer's content-hash
        # dedup (services/ingestion/normalizer.py:_upsert_document) then
        # no-ops every subsequent event, so we never re-embed or bump versions
        # for the firehose. Fresh event details come from Sentry's API directly
        # via a separate tool surface.
        doc_id = f"sentry:issue:{group_id}:sample"
        content_hash = _sha256(f"{doc_id}|representative_sample")

        parent_doc_id = f"sentry:issue:{group_id}"

        acl_principals = [
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=org_slug,
                permission=Permission.READ,
            ),
            ACLPrincipal(
                principal_type=PrincipalType.GROUP,
                principal_id=f"sentry-project:{project_slug}",
                permission=Permission.READ,
            ),
        ]

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.SENTRY,
            source_id=f"issue:{group_id}:sample",
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.SENTRY_EVENT,
            content_type="text/plain",
            content_hash=content_hash,
            title=title[:240] if title else None,
            body_preview=body[:280],
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            # `author_id` is intentionally unset — a sample isn't authored
            # by the actor whose webhook happened to carry it. Actor attribution
            # lives on the issue doc (for lifecycle actions) or in Sentry directly.
            author_id=None,
            created_at=occurred_at,
            updated_at=occurred_at,
            valid_from=occurred_at,
            ingested_at=datetime.now(UTC),
            parent_doc_id=parent_doc_id,
            acl=ACLSnapshot(principals=acl_principals, captured_at=event.received_at),
            metadata={
                "body": body,
                "group_id": group_id,
                "project_slug": project_slug,
                "project_id": project_id,
                "platform": platform,
                # These reflect the FIRST event we saw, not the latest — live
                # Sentry lookup is the source of truth for current release/env.
                "first_sample_environment": environment,
                "first_sample_release": release,
                "first_sample_event_id": event_id,
                "first_sample_tags": ev.get("tags", []),
                "sample_strategy": "first_event_per_issue",
            },
        )

        nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": doc.doc_type.value},
            ),
            GraphNodeSpec(
                label=NodeLabel.SERVICE,
                canonical_id=project_slug,
                properties=_service_properties(payload, platform),
            ),
            GraphNodeSpec(
                label=NodeLabel.ERROR_GROUP,
                canonical_id=group_id,
                properties={"culprit": culprit, "level": "error"},
            ),
        ]
        edges: list[GraphEdgeSpec] = [
            GraphEdgeSpec(
                edge_type=EdgeType.FIRES_IN,
                from_label=NodeLabel.ERROR_GROUP,
                from_canonical_id=group_id,
                to_label=NodeLabel.SERVICE,
                to_canonical_id=project_slug,
                valid_from=occurred_at,
            ),
            # Document → ErrorGroup + Document → Service for entity-filter
            # reachability on the list path.
            GraphEdgeSpec(
                edge_type=EdgeType.LINKED_FROM,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.ERROR_GROUP,
                to_canonical_id=group_id,
                valid_from=occurred_at,
            ),
            GraphEdgeSpec(
                edge_type=EdgeType.LINKED_FROM,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.SERVICE,
                to_canonical_id=project_slug,
                valid_from=occurred_at,
            ),
        ]

        acl_rows = [
            ACLSnapshotRow(
                source_system=SourceSystem.SENTRY,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=org_slug,
                resource_type="sentry.event_sample",
                resource_id=f"issue:{group_id}",
                permission=Permission.READ,
                valid_from=occurred_at,
            ),
            ACLSnapshotRow(
                source_system=SourceSystem.SENTRY,
                principal_type=PrincipalType.GROUP,
                principal_id=f"sentry-project:{project_slug}",
                resource_type="sentry.event_sample",
                resource_id=f"issue:{group_id}",
                permission=Permission.READ,
                valid_from=occurred_at,
            ),
        ]

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ------------------------------------------------------------------
    # 7. workspace identification
    # ------------------------------------------------------------------

    async def identify_workspaces(self, token):  # type: ignore[override]
        """Sentry internal integrations don't go through a standard OAuth
        code flow — the installation webhook carries organization info.
        Phase 0 returns []; webhooks will be routed via
        `extract_external_id_from_payload` + single_customer_fallback.

        Phase 1 TODO: handle `installation.created` webhook specially to
        record the mapping at install time instead of on first real webhook.
        """
        return []

    def extract_external_id_from_payload(self, headers, raw_payload):
        # Sentry payload shapes: top-level organization.slug OR installation.organization.slug
        org = raw_payload.get("organization") or {}
        slug = org.get("slug")
        if not slug:
            install = raw_payload.get("installation") or {}
            slug = (install.get("organization") or {}).get("slug")
        return str(slug) if slug else None

    # ------------------------------------------------------------------
    # backfill
    # ------------------------------------------------------------------

    async def backfill(
        self,
        customer_id: str,
        token,
        cursor: str | None = None,
    ):
        """Paginate Sentry `/api/0/organizations/{slug}/issues/`.

        Emits synthetic issue.created events so the normalizer has one code
        path. Sentry's issues endpoint uses Link-header cursor pagination;
        we follow `next` until exhausted or the server reports no more.

        Phase 0 limitation: we resolve the org slug from `identify_workspaces`
        if available, else fall back to `/api/0/organizations/` (first org).
        Multi-org installations only backfill the first org; the others will
        fill in as webhooks arrive.
        """
        from shared.models import WebhookEvent

        auth_headers = {"Authorization": f"Bearer {token.access_token}"}

        org_slug = await _sentry_org_slug(self.http, auth_headers)
        if org_slug is None:
            log.warning("sentry.backfill_no_org", customer=customer_id)
            return

        url = f"https://sentry.io/api/0/organizations/{org_slug}/issues/"
        if cursor:
            url = f"{url}?cursor={cursor}"

        while url:
            try:
                resp = await self.http.get(url, headers=auth_headers)
            except Exception as exc:
                log.warning("sentry.backfill_http_error", error=str(exc))
                return
            if resp.status_code != 200:
                return

            for issue in resp.json():
                issue_id = issue.get("id")
                if not issue_id:
                    continue
                link = resp.headers.get("link") or ""
                next_cursor = _parse_next_cursor(link)
                payload = {
                    "action": "created",
                    "actor": {"type": "system", "name": "sentry-backfill"},
                    "data": {"issue": issue},
                    "installation": {"organization": {"slug": org_slug}},
                    "organization": {"slug": org_slug},
                    "_cursor": next_cursor,
                }
                yield WebhookEvent(
                    customer_id=customer_id,
                    source_system=SourceSystem.SENTRY,
                    source_event_id=f"issue:{issue_id}:backfill",
                    received_at=_parse_iso8601(issue.get("lastSeen"))
                    or datetime.now(UTC),
                    payload_s3_key="",
                    raw_payload=payload,
                    headers={"sentry-hook-resource": "issue"},
                )

            # Sentry's cursor pagination: the Link header holds a `rel="next"`
            # URL if there's more. If cursor results=false, we're done.
            link = resp.headers.get("link") or ""
            next_url = _parse_next_link(link)
            if not next_url:
                return
            url = next_url

    # ------------------------------------------------------------------


async def _sentry_org_slug(http, headers: dict[str, str]) -> str | None:
    try:
        resp = await http.get("https://sentry.io/api/0/organizations/", headers=headers)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    orgs = resp.json()
    if not orgs:
        return None
    return orgs[0].get("slug")


_SENTRY_CURSOR_RE = None  # set lazily in _parse_next_link


def _parse_next_link(link_header: str) -> str | None:
    """Parse Sentry's Link header for a `rel="next"; results="true"` URL."""
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' not in part:
            continue
        if 'results="false"' in part:
            return None
        # format: <URL>; rel="next"; results="true"; cursor="..."
        if part.startswith("<") and ">" in part:
            return part.split(">", 1)[0][1:]
    return None


def _parse_next_cursor(link_header: str) -> str | None:
    """Extract cursor=... from the next-rel link, for persistence between rows."""
    url = _parse_next_link(link_header)
    if not url or "cursor=" not in url:
        return None
    return url.split("cursor=", 1)[1].split("&", 1)[0]


# ---- helpers ---------------------------------------------------------------


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    # Sentry uses trailing "Z" — normalise to +00:00 for fromisoformat.
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _project_slug(payload: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    # Top-level `project` is the authoritative shape on new hooks; older
    # `data.issue.project` is a fallback. Default to "unknown" so normalize
    # never explodes on a stripped payload.
    project = payload.get("project") or inner.get("project") or {}
    if isinstance(project, dict):
        slug = project.get("slug")
        if isinstance(slug, str) and slug:
            return slug
    return "unknown"


def _project_id(payload: Mapping[str, Any], inner: Mapping[str, Any]) -> str | None:
    project = payload.get("project") or inner.get("project") or {}
    if isinstance(project, dict):
        pid = project.get("id")
        if pid is not None:
            return str(pid)
    # event payloads sometimes carry a bare integer `project` field.
    raw = inner.get("project")
    if isinstance(raw, (int, str)) and raw:
        return str(raw)
    return None


def _org_slug(payload: Mapping[str, Any]) -> str:
    org = payload.get("organization") or {}
    if isinstance(org, dict):
        slug = org.get("slug")
        if isinstance(slug, str) and slug:
            return slug
    return "unknown"


def _platform(payload: Mapping[str, Any], inner: Mapping[str, Any]) -> str | None:
    project = payload.get("project")
    if isinstance(project, dict) and isinstance(project.get("platform"), str):
        return project["platform"]
    if isinstance(inner.get("platform"), str):
        return inner["platform"]
    return None


def _service_properties(
    payload: Mapping[str, Any], platform: str | None
) -> dict[str, Any]:
    props: dict[str, Any] = {"source_system": SourceSystem.SENTRY.value}
    if platform:
        props["platform"] = platform
    project = payload.get("project")
    if isinstance(project, dict):
        team = project.get("team")
        if isinstance(team, dict) and isinstance(team.get("slug"), str):
            props["team"] = team["slug"]
    return props


def _assignee(issue: Mapping[str, Any]) -> str | None:
    assigned = issue.get("assignedTo")
    if not isinstance(assigned, dict):
        return None
    for key in ("username", "email", "id"):
        val = assigned.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _human_actor_id(payload: Mapping[str, Any]) -> str | None:
    """Return the actor id iff it's a human. Sentry tags rule-fired webhooks
    with an `application` actor (name="Sentry") — we don't want to AUTHORED
    an event to a bot."""
    actor = payload.get("actor")
    if not isinstance(actor, dict):
        return None
    actor_type = (actor.get("type") or "").lower()
    if actor_type in _SYSTEM_ACTOR_TYPES:
        return None
    actor_id = actor.get("id") or actor.get("name")
    if not isinstance(actor_id, str) or not actor_id:
        return None
    if actor_id.lower() in _SENTRY_BOT_IDS:
        return None
    return actor_id


def _issue_body(issue: Mapping[str, Any]) -> str:
    parts: list[str] = []
    culprit = issue.get("culprit")
    if isinstance(culprit, str) and culprit:
        parts.append(culprit)

    metadata = issue.get("metadata")
    if isinstance(metadata, dict):
        exc_type = metadata.get("type")
        value = metadata.get("value")
        filename = metadata.get("filename")
        function = metadata.get("function")
        signature_parts = [
            p for p in (exc_type, value) if isinstance(p, str) and p
        ]
        if signature_parts:
            parts.append(": ".join(signature_parts))
        location_parts = [
            p for p in (filename, function) if isinstance(p, str) and p
        ]
        if location_parts:
            parts.append(" in ".join(location_parts))

    if not parts:
        title = issue.get("title")
        if isinstance(title, str):
            parts.append(title)
    return "\n\n".join(parts)


def _event_body(ev: Mapping[str, Any]) -> str:
    """Render the top exception + up to _MAX_FRAMES_IN_BODY stack frames."""
    lines: list[str] = []
    exception = ev.get("exception")
    values: list[Any] = []
    if isinstance(exception, dict):
        raw_values = exception.get("values")
        if isinstance(raw_values, list):
            values = raw_values

    if values and isinstance(values[0], dict):
        top = values[0]
        exc_type = top.get("type")
        exc_value = top.get("value")
        if isinstance(exc_type, str) or isinstance(exc_value, str):
            header = ": ".join(
                str(p) for p in (exc_type, exc_value) if isinstance(p, str) and p
            )
            if header:
                lines.append(header)

        stacktrace = top.get("stacktrace")
        frames: list[Any] = []
        if isinstance(stacktrace, dict) and isinstance(
            stacktrace.get("frames"), list
        ):
            frames = stacktrace["frames"]
        # Sentry lists deepest-first in some SDKs, but the top of our body
        # should be the most specific frame — take the tail slice and reverse.
        selected = frames[-_MAX_FRAMES_IN_BODY:]
        for frame in reversed(selected):
            if not isinstance(frame, dict):
                continue
            filename = frame.get("filename") or frame.get("abs_path") or "<unknown>"
            function = frame.get("function") or "<anon>"
            lineno = frame.get("lineno")
            ctx = frame.get("context_line")
            loc = f"{filename}:{lineno}" if lineno is not None else str(filename)
            line = f"  at {function} ({loc})"
            if isinstance(ctx, str) and ctx.strip():
                line += f"\n    {ctx.strip()}"
            lines.append(line)

    if not lines:
        culprit = ev.get("culprit")
        title = ev.get("title")
        for candidate in (title, culprit):
            if isinstance(candidate, str) and candidate:
                lines.append(candidate)
                break

    return "\n".join(lines)


__all__ = ["SentryConnector"]
