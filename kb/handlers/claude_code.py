"""Claude Code transcript connector.

Data flow:
- Daemon (prbe-agent-tap) ships per-session JSONL batches to the BFF gateway
  (prbe-backend) at api.prbe.ai/webhooks/claude_code; the gateway authenticates
  the device token, then forwards to /webhooks/claude_code on prbe-knowledge
  with X-Internal-Knowledge-Key + X-Prbe-Customer.
- extract_external_id_from_payload returns device_id; resolve_customer maps to customer.
- verify_signature is a defense-in-depth path (the gateway's auth is the primary
  guard — see comment in services/ingestion/main.py).
- parse_webhook_event keys the queue row by <session_id>:<batch_seq>.
- Worker invokes fetch_supplementary (assemble all R2 batches for session) +
  normalize (emit session doc + per-unit child docs).

Pairing/heartbeat/revoke are public lifecycle endpoints on prbe-backend
(api.prbe.ai/agent-tap/*); prbe-knowledge exposes only the internal
/api/devices/* endpoints that the gateway calls.
"""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

import orjson

from engine.ingest.handlers.base import Connector
from engine.ingest.handlers.registry import register_connector
from engine.shared import claude_code_extraction as _ext
from engine.shared.constants import (
    DocClass,
    DocType,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from engine.shared.exceptions import InvalidWebhookPayload, NotSupportedByConnector
from engine.shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    ExternalWorkspaceRef,
    GraphEdgeSpec,
    GraphNodeSpec,
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)
from engine.shared.storage import get_store

# Cap on simultaneous R2 GETs per fetch_supplementary call. With WORKER
# concurrency=4 and a long session of ~100 batches, unbounded asyncio.gather
# would peak at ~400 in-flight S3 GETs per machine and balloon memory. 16
# is enough to drain a typical session in ~1.5s while keeping in-flight
# envelopes bounded to a few MB.
_FETCH_SUPP_R2_CONCURRENCY = 16


def _nonempty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


@register_connector(SourceSystem.CLAUDE_CODE)
class ClaudeCodeConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.CLAUDE_CODE
    display_name: ClassVar[str] = "Claude Code"
    doc_type_prefix: ClassVar[str] = "claude_code."
    # Queue priority 75: bursty, deprioritized vs interactive webhooks (100).
    # Sessions are search-indexable, not user-blocking; one chatty CC user
    # shouldn't block other connectors at the queue claim layer.
    ingestion_priority: ClassVar[int] = 75
    # CC transcripts are high-volume and lower-signal-density than authored
    # team artifacts (Slack threads, Linear tickets, PR descriptions); the
    # 0.5 post-RRF demotion keeps authored content surfacing first.
    score_multiplier: ClassVar[float] = 0.5
    # A CC session is a point-in-time scratchpad — by week two it's almost
    # always stale or contradicted by something authored elsewhere.
    half_life_days: ClassVar[float | None] = 7.0
    # Per-source identifiers used to tag persisted artifacts. Subclasses
    # (e.g. CodexConnector below) override these so docs/edges/ACL rows
    # carry the correct provenance label even though the doc shape and
    # extraction pipeline are shared with Claude Code.
    _doc_id_prefix: ClassVar[str] = "claude_code"
    _agent_label: ClassVar[str] = "claude_code"
    _session_title_prefix: ClassVar[str] = "Claude Code session"

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        # The BFF gateway (prbe-backend) authenticates the daemon via its
        # bearer device token before forwarding here. services/ingestion/main.py
        # gates this endpoint on X-Internal-Knowledge-Key + X-Prbe-Customer
        # and never calls verify_signature, so this is unreachable in
        # production. Returning True keeps the abstract method satisfied.
        return True

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        # source_event_id is the bare session_id for both live batches AND
        # finalize events. _enqueue (services/ingestion/main.py) UPSERTs on
        # this key for claude_code, so every batch + the cron finalize all
        # coalesce into one queue row per session. The worker detects
        # finalize via the presence of a `finalize.marker` key in
        # payload_s3_keys, not via a source_event_id suffix.
        #
        # Legacy `<session>:<batch>` and `<session>:finalize` source_event_ids
        # may still exist on in-flight queue rows from before migration 0026;
        # they continue to drain through the worker's old single-payload path.
        if raw_payload.get("finalize") is True:
            session_id = raw_payload.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise InvalidWebhookPayload("claude_code: finalize event missing session_id")
            return WebhookParseResult(
                source_event_id=session_id,
                received_at=datetime.now(UTC),
                parse_hint={"session_id": session_id, "finalize": True},
            )

        # Normal batch event
        session_id = raw_payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise InvalidWebhookPayload("claude_code: missing session_id")

        batch_seq = raw_payload.get("batch_seq")
        if not isinstance(batch_seq, int):
            raise InvalidWebhookPayload("claude_code: batch_seq must be int")

        events = raw_payload.get("events") or []
        if not events:
            return None  # empty post, nothing to enqueue

        return WebhookParseResult(
            source_event_id=session_id,
            received_at=datetime.now(UTC),
            parse_hint={"session_id": session_id, "batch_seq": batch_seq},
        )

    def extract_external_id_from_payload(
        self,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> str | None:
        device_id = raw_payload.get("device_id")
        if not isinstance(device_id, str) or not device_id:
            return None
        return device_id

    async def identify_workspaces(
        self, token: IntegrationToken
    ) -> list[ExternalWorkspaceRef]:
        if token.device_id is None:
            raise ValueError("identify_workspaces called without device_id")
        meta = token.device_metadata or {}
        return [
            ExternalWorkspaceRef(
                external_id=token.device_id,
                external_name=meta.get("hostname"),
            )
        ]

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: IntegrationToken | None,
    ) -> dict[str, Any]:
        """Merge events from every R2 payload coalesced into the queue row.

        After migration 0026, all batches for a session land in one queue
        row's `payload_s3_keys` array (services/ingestion/main.py:_enqueue
        UPSERTs on session_id and appends each batch's key). This method
        reads every key in parallel (semaphore-bounded), parses the
        webhook envelope's `payload.events`, and merges them into a single
        line_no-ordered event list.

        Session-complete detection has two paths:
        1. Live: any merged event with raw.type == 'session_end'.
        2. Cron-finalize: any payload_s3_keys entry ending in
           `finalize.marker` (written by session_completer.py when a
           session goes idle past the threshold).

        Legacy in-flight rows from before migration 0026 still flow
        through here naturally: their `payload_s3_keys` was backfilled
        to ARRAY[payload_s3_key], so the array is single-element. The
        legacy `<session>:<batch>` and `<session>:finalize` source_event_ids
        on those rows still trigger complete=True via the suffix check
        below — they'll drain through one last time under old semantics.
        """
        session_id = event.raw_payload.get("session_id") or event.source_event_id.split(":", 1)[0]
        if not session_id:
            raise ValueError(
                f"fetch_supplementary: cannot determine session_id from event {event.source_event_id!r}"
            )

        merged_events: list[dict[str, Any]] = []
        seen_line_nos: set[int] = set()
        session_identity: dict[str, str] = {}

        def _remember_payload_identity(payload: Mapping[str, Any]) -> None:
            # In coalesced sessions, Normalizer.event.raw_payload is the
            # oldest payload. A session that started before a gateway identity
            # deploy can therefore have name/email/hostname only on later
            # payloads. Keep the latest non-empty batch-level identity so
            # normalize() can still render the live doc with human labels.
            for key in (
                "employee_id",
                "employee_name",
                "employee_email",
                "employee_hostname",
            ):
                val = _nonempty_str(payload.get(key))
                if val is not None:
                    session_identity[key] = val

        def _ingest(obj: dict[str, Any]) -> None:
            """Append `obj` to merged_events, deduplicating by line_no when present."""
            line_no = obj.get("line_no")
            # Only deduplicate events with explicit line_no values. Events that
            # lack line_no must always be included — otherwise the first None
            # poisons the dedup set and silently drops every subsequent
            # line_no-less event.
            if line_no is not None:
                if line_no in seen_line_nos:
                    return
                seen_line_nos.add(line_no)
            merged_events.append(obj)

        keys: list[str] = list(event.payload_s3_keys or [])
        if not keys and event.payload_s3_key:
            # Defensive: if the queue row was inserted before migration 0026
            # and somehow the array backfill missed it, fall back to the
            # single-key column.
            keys = [event.payload_s3_key]

        store = get_store()
        bucket = await store.bucket_for(event.customer_id)
        sem = asyncio.Semaphore(_FETCH_SUPP_R2_CONCURRENCY)

        async def _fetch(key: str) -> tuple[str, bytes]:
            async with sem:
                body = await store.get(bucket, key)
            return key, body

        # Parallel R2 fetches, semaphore-capped. Order doesn't matter
        # for the merge — we re-sort by line_no after.
        fetched: list[tuple[str, bytes]] = await asyncio.gather(*(_fetch(k) for k in keys))

        finalize_marker_seen = False
        for key, body in fetched:
            if key.endswith("/finalize.marker"):
                finalize_marker_seen = True
            try:
                envelope = orjson.loads(body)
            except orjson.JSONDecodeError:
                # Tolerate non-JSON or corrupted blobs — log and skip.
                continue
            payload = envelope.get("payload", envelope) if isinstance(envelope, dict) else {}
            if not isinstance(payload, dict):
                continue
            _remember_payload_identity(payload)
            for obj in payload.get("events") or []:
                if isinstance(obj, dict):
                    _ingest(obj)

        merged_events.sort(key=lambda e: (e.get("line_no") is None, e.get("line_no") or 0))

        # Session-complete detection: live SessionEnd, cron-injected
        # finalize.marker, or legacy `:finalize` source_event_id from
        # in-flight pre-migration rows.
        complete = any(
            (e.get("raw") or {}).get("type") == "session_end"
            for e in merged_events
        )
        if finalize_marker_seen:
            complete = True
        if event.source_event_id.endswith(":finalize"):
            complete = True

        return {
            "session_id": session_id,
            "events": merged_events,
            "session_complete": complete,
            "cwd": event.raw_payload.get("cwd"),
            **session_identity,
        }

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        session_id = hydrated["session_id"]
        events = hydrated.get("events") or []
        cwd = hydrated.get("cwd")
        complete = bool(hydrated.get("session_complete"))
        employee_id = (
            _nonempty_str(hydrated.get("employee_id"))
            or self._employee_id_from_event(event, events)
        )
        employee_name = (
            _nonempty_str(hydrated.get("employee_name"))
            or self._employee_name_from_event(event, events)
        )
        employee_email = (
            _nonempty_str(hydrated.get("employee_email"))
            or self._employee_email_from_event(event, events)
        )
        employee_hostname = (
            _nonempty_str(hydrated.get("employee_hostname"))
            or self._employee_hostname_from_event(event, events)
        )

        now = datetime.now(UTC)
        session_doc = self._build_session_doc(
            event=event,
            session_id=session_id,
            cwd=cwd,
            employee_id=employee_id,
            employee_name=employee_name,
            employee_email=employee_email,
            employee_hostname=employee_hostname,
            events=events,
            complete=complete,
            now=now,
        )

        documents: list[Document] = [session_doc]
        # Stamp employee_name + employee_email + hostname on the Person
        # node when the gateway provided them. name/email power
        # name-keyed graph filters via idx_graph_nodes_lower_props_name;
        # hostname rides along so dashboard queries can disambiguate
        # multi-device users. Absent (not None/empty) keys mean
        # "no value" — never index empty strings, since the
        # LOWER(properties->>'name') index would otherwise hold a useless
        # "" entry per employee.
        person_props: dict[str, Any] = {"employee_id": employee_id}
        if employee_name:
            person_props["name"] = employee_name
        if employee_email:
            person_props["email"] = employee_email
        if employee_hostname:
            person_props["hostname"] = employee_hostname

        graph_nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=employee_id,
                properties=person_props,
            ),
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=session_doc.doc_id,
                properties={"doc_type": DocType.CLAUDE_CODE_SESSION.value},
            ),
        ]
        graph_edges: list[GraphEdgeSpec] = [
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=employee_id,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=session_doc.doc_id,
                properties={"session_id": session_id},
            )
        ]
        acl = self._acl(employee_id)
        acl_rows: list[ACLSnapshotRow] = [
            ACLSnapshotRow(
                source_system=self.source_system,
                principal_type=p.principal_type,
                principal_id=p.principal_id,
                resource_type="document",
                resource_id=session_doc.doc_id,
                permission=p.permission,
                valid_from=now,
            )
            for p in acl.principals
        ]

        if not complete:
            return NormalizationResult(
                documents=documents,
                graph_nodes=graph_nodes,
                graph_edges=graph_edges,
                acl_snapshots=acl_rows,
            )

        bundle = await _ext.extract_units_from_session(
            session_id=session_id,
            events=events,
            cwd=cwd,
        )

        for idx, qa in enumerate(bundle.qa):
            documents.append(
                self._build_unit_doc(
                    event=event,
                    parent=session_doc,
                    employee_id=employee_id,
                    employee_name=employee_name,
                    employee_email=employee_email,
                    employee_hostname=employee_hostname,
                    doc_type=DocType.CLAUDE_CODE_QA,
                    unit_kind="qa",
                    idx=idx,
                    metadata={
                        "prompt": qa.prompt,
                        "outcome": qa.outcome,
                        "tags": list(qa.tags),
                    },
                    body=f"Q: {qa.prompt}\n\nA: {qa.outcome}",
                    now=now,
                )
            )
        for idx, cc in enumerate(bundle.code_change):
            documents.append(
                self._build_unit_doc(
                    event=event,
                    parent=session_doc,
                    employee_id=employee_id,
                    employee_name=employee_name,
                    employee_email=employee_email,
                    employee_hostname=employee_hostname,
                    doc_type=DocType.CLAUDE_CODE_CODE_CHANGE,
                    unit_kind="code_change",
                    idx=idx,
                    metadata={
                        "file": cc.file,
                        "before": cc.before,
                        "after": cc.after,
                        "intent": cc.intent,
                    },
                    body=f"FILE: {cc.file}\nINTENT: {cc.intent}\nBEFORE:\n{cc.before}\n\nAFTER:\n{cc.after}",
                    now=now,
                )
            )
        for idx, dec in enumerate(bundle.decision):
            documents.append(
                self._build_unit_doc(
                    event=event,
                    parent=session_doc,
                    employee_id=employee_id,
                    employee_name=employee_name,
                    employee_email=employee_email,
                    employee_hostname=employee_hostname,
                    doc_type=DocType.CLAUDE_CODE_DECISION,
                    unit_kind="decision",
                    idx=idx,
                    metadata={
                        "question": dec.question,
                        "options_considered": list(dec.options_considered),
                        "chosen": dec.chosen,
                        "rationale": dec.rationale,
                    },
                    body=(
                        f"Q: {dec.question}\nOPTIONS: {', '.join(dec.options_considered)}\n"
                        f"CHOSE: {dec.chosen}\nRATIONALE: {dec.rationale}"
                    ),
                    now=now,
                )
            )
        for idx, fr in enumerate(bundle.file_ref):
            documents.append(
                self._build_unit_doc(
                    event=event,
                    parent=session_doc,
                    employee_id=employee_id,
                    employee_name=employee_name,
                    employee_email=employee_email,
                    employee_hostname=employee_hostname,
                    doc_type=DocType.CLAUDE_CODE_FILE_REF,
                    unit_kind="file_ref",
                    idx=idx,
                    metadata={
                        "files": list(fr.files),
                        "context": fr.context,
                    },
                    body=f"CONTEXT: {fr.context}\nFILES: {', '.join(fr.files)}",
                    now=now,
                )
            )

        # Note: unit documents are linked to the session document via
        # `parent_doc_id` in Postgres rather than via direct graph edges. The
        # graph writer walks `parent_doc_id` to navigate from a session node
        # to its child units. Adding redundant DOCUMENT/AUTHORED graph nodes
        # for each unit would bloat the graph without enabling any new
        # queries — the session graph node + parent_doc_id chain is sufficient.

        # Mirror ACL onto every unit doc
        for d in documents[1:]:  # skip session doc, already snapshotted
            for p in acl.principals:
                acl_rows.append(
                    ACLSnapshotRow(
                        source_system=self.source_system,
                        principal_type=p.principal_type,
                        principal_id=p.principal_id,
                        resource_type="document",
                        resource_id=d.doc_id,
                        permission=p.permission,
                        valid_from=now,
                    )
                )

        return NormalizationResult(
            documents=documents,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            acl_snapshots=acl_rows,
        )

    # ---- helpers ----------------------------------------------------------

    def _employee_id_from_event(
        self,
        event: WebhookEvent,
        merged_events: list[dict[str, Any]] | None = None,
    ) -> str:
        emp = event.raw_payload.get("employee_id")
        if not isinstance(emp, str) or not emp:
            # Finalize events have no employee_id in raw_payload — fall back to
            # the first merged batch event that carries one.
            for e in (merged_events or []):
                candidate = e.get("employee_id")
                if isinstance(candidate, str) and candidate:
                    return candidate
            raise InvalidWebhookPayload("claude_code: missing employee_id")
        return emp

    def _employee_name_from_event(
        self,
        event: WebhookEvent,
        merged_events: list[dict[str, Any]] | None = None,
    ) -> str | None:
        # Optional. Gateway (prbe-backend PR #67) injects this from
        # neon_auth.user.name when present. Mirrors the finalize-fallback
        # pattern of _employee_id_from_event: finalize events carry no
        # per-event identity in raw_payload, so we walk merged_events for
        # the first one with a populated value.
        val = event.raw_payload.get("employee_name")
        if isinstance(val, str) and val:
            return val
        for e in (merged_events or []):
            candidate = e.get("employee_name")
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    def _employee_email_from_event(
        self,
        event: WebhookEvent,
        merged_events: list[dict[str, Any]] | None = None,
    ) -> str | None:
        # Optional. See _employee_name_from_event for the rationale.
        val = event.raw_payload.get("employee_email")
        if isinstance(val, str) and val:
            return val
        for e in (merged_events or []):
            candidate = e.get("employee_email")
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    def _employee_hostname_from_event(
        self,
        event: WebhookEvent,
        merged_events: list[dict[str, Any]] | None = None,
    ) -> str | None:
        # Optional. Gateway (prbe-backend PR #2 lane B) injects this from
        # the device's registered hostname so the human-readable machine
        # label rides along with each event. Same finalize-fallback shape
        # as _employee_name_from_event / _employee_email_from_event.
        val = event.raw_payload.get("employee_hostname")
        if isinstance(val, str) and val:
            return val
        for e in (merged_events or []):
            candidate = e.get("employee_hostname")
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    def _acl(self, employee_id: str) -> ACLSnapshot:
        return ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.USER,
                    principal_id=employee_id,
                    permission=Permission.READ,
                )
            ],
            captured_at=datetime.now(UTC),
        )

    def _build_session_doc(
        self,
        *,
        event: WebhookEvent,
        session_id: str,
        cwd: str | None,
        employee_id: str,
        employee_name: str | None,
        employee_email: str | None,
        employee_hostname: str | None,
        events: list[dict[str, Any]],
        complete: bool,
        now: datetime,
    ) -> Document:
        rendered_body = _events_to_text(events)
        body_bytes = rendered_body.encode("utf-8")
        content_hash = hashlib.sha256(body_bytes).hexdigest()
        doc_id = f"{self._doc_id_prefix}:{event.customer_id}:{session_id}"
        first_content = ""
        if events:
            raw = events[0].get("raw") or {}
            first_content = raw.get("content", "") or ""
            if not isinstance(first_content, str):
                first_content = ""
        # Title format graceful-degrades through all 8 combinations of
        # (name, email, hostname) presence. With nothing it falls back to
        # the pre-Lane-B "Claude Code session XXXXXXXX" status quo.
        title = _format_session_title(
            short_id=session_id[:8],
            name=employee_name,
            email=employee_email,
            hostname=employee_hostname,
            kind=self._session_title_prefix,
        )
        # Identity keys land on metadata only when present — keeps JSONB
        # null-free and matches the rest of the handler's "omit when
        # absent" convention.
        md: dict[str, Any] = {
            "agent": self._agent_label,
            "cwd": cwd,
            "device_id": event.raw_payload.get("device_id"),
            "session_complete": complete,
            "event_count": len(events),
        }
        if employee_name:
            md["employee_name"] = employee_name
        if employee_email:
            md["employee_email"] = employee_email
        if employee_hostname:
            md["employee_hostname"] = employee_hostname
        return Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=self.source_system,
            source_id=session_id,
            source_url=f"https://prbe.ai/dashboard/agent-sessions/{session_id}",
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.CLAUDE_CODE_SESSION,
            content_type="application/json",
            content_hash=content_hash,
            title=title,
            body_preview=first_content[:200],
            body_size_bytes=len(body_bytes),
            body_token_count=0,
            author_id=employee_id,
            created_at=event.received_at,
            updated_at=now,
            valid_from=now,
            ingested_at=now,
            metadata=md,
            # Body drives Normalizer._stringify_body -> chunker. Lives on the
            # transient Document.body field, never on metadata jsonb.
            body=rendered_body,
            # Coalesce in-place while the session is still incomplete: each
            # tick UPDATEs the live row instead of opening a new SCD2 version.
            # Without this, a 5-tick session writes 5 doc rows + 5 chunk
            # versions for the same conversation. See migration 0036 for the
            # one-time cleanup that prunes the prior accumulation.
            coalesce_into_live=not complete,
            acl=self._acl(employee_id),
        )

    def _build_unit_doc(
        self,
        *,
        event: WebhookEvent,
        parent: Document,
        employee_id: str,
        employee_name: str | None,
        employee_email: str | None,
        employee_hostname: str | None,
        doc_type: DocType,
        unit_kind: str,
        idx: int,
        metadata: dict[str, Any],
        body: str,
        now: datetime,
    ) -> Document:
        source_id = f"{parent.source_id}:{unit_kind}:{idx}"
        doc_id = f"{self._doc_id_prefix}:{event.customer_id}:{source_id}"
        body_bytes = body.encode("utf-8")
        content_hash = hashlib.sha256(body_bytes).hexdigest()
        # Mirror the session-doc title shape so a unit doc surface like
        # "Richard Wei's (richard@prbe.ai) decision 82861aa0 (Richards-Macbook-Pro)"
        # — same identity-bearing prefix, with unit_kind substituted for
        # the session-prefix so retrieval can rank both equally.
        title = _format_session_title(
            short_id=parent.source_id[:8],
            name=employee_name,
            email=employee_email,
            hostname=employee_hostname,
            kind=unit_kind,
        )
        # Identity keys land on metadata only when present — same
        # "omit when absent" convention as the session doc above.
        md = dict(metadata)
        device_id = _nonempty_str(parent.metadata.get("device_id"))
        if device_id:
            # Keep derived artifacts directly attributable for new ingests.
            # Historical unit docs only have parent_doc_id, so readers must
            # retain parent fallback rather than relying on this denormalized
            # field being universally present.
            md["device_id"] = device_id
        if employee_name:
            md["employee_name"] = employee_name
        if employee_email:
            md["employee_email"] = employee_email
        if employee_hostname:
            md["employee_hostname"] = employee_hostname
        return Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=self.source_system,
            source_id=source_id,
            source_url=parent.source_url + f"#{unit_kind}-{idx}",
            doc_class=DocClass.RAW_SOURCE,
            doc_type=doc_type,
            content_type="text/plain",
            content_hash=content_hash,
            title=title,
            body_preview=body[:200],
            body_size_bytes=len(body_bytes),
            body_token_count=0,
            author_id=employee_id,
            created_at=now,
            updated_at=now,
            valid_from=now,
            ingested_at=now,
            parent_doc_id=parent.doc_id,
            metadata=md,
            # body drives Normalizer._stringify_body -> chunker. Without it,
            # only the 200-char preview gets indexed and the unit's full
            # content (Q+A, code change, decision text, file ref) is lost.
            body=body,
            acl=parent.acl,
        )

    def backfill(
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ) -> AsyncIterator[WebhookEvent]:
        raise NotSupportedByConnector(
            "claude_code backfill happens client-side via the agent-tap daemon"
        )


def _format_session_title(
    short_id: str,
    name: str | None,
    email: str | None,
    hostname: str | None,
    kind: str = "Claude Code session",
) -> str:
    """Render the human-friendly identity-bearing Document title.

    Builds something like:
        "Richard Wei's (richard@prbe.ai) Claude Code session 82861aa0 (Richards-Macbook-Pro)"

    Each identity field contributes only when present so all 8 combinations
    of (name, email, hostname) presence/absence yield a sensible string.
    With nothing extra it degrades to "Claude Code session 82861aa0", which
    matches the pre-Lane-B status quo.

    `kind` lets unit docs (decision/qa/code_change/file_ref) reuse the same
    shape — they pass their unit_kind in place of "Claude Code session".
    """
    name_part = f"{name}'s " if name else ""
    email_part = f"({email}) " if email else ""
    base = f"{kind} {short_id}"
    host_part = f" ({hostname})" if hostname else ""
    return f"{name_part}{email_part}{base}{host_part}"


def _events_to_text(events: list[dict[str, Any]]) -> str:
    """Render merged Claude Code events into a chunkable text body.

    Output is human-readable prose — what the chunker + embedder consume,
    so it has to look like the conversation. NOT JSON dumps.

    Each turn becomes a block separated by blank lines:

        USER: how does auth work?

        ASSISTANT (thinking): the flow uses JWT in cookie X.
        ASSISTANT: we use JWT.

        TOOL_USE: Bash — git status
        TOOL_RESULT (toolu_xxx): ok

        USER: refactor it.

    Ordering matches the line_no-sorted merged stream so a chunker walking
    sequentially sees the session in transcript order.

    Events the plugin sanitizer already drops (file-history-snapshot,
    last-prompt, ai-title, permission-mode, stop_hook_summary, turn_duration)
    don't reach here. Anything else without a renderer is silently skipped
    rather than dumped as JSON — JSON noise in the embedded text was the
    original problem this rewrite solves.
    """
    blocks: list[str] = []
    for ev in events:
        raw = ev.get("raw") if isinstance(ev, dict) else None
        if not isinstance(raw, dict):
            continue
        rendered = _render_event(raw)
        if rendered:
            blocks.append(rendered)
    return "\n\n".join(blocks)


def _render_event(raw: dict[str, Any]) -> str:
    ev_type = raw.get("type")
    if ev_type == "user":
        return _render_user(raw)
    if ev_type == "assistant":
        return _render_assistant(raw)
    if ev_type == "system":
        sub = raw.get("subtype") or ""
        content = raw.get("content")
        if isinstance(content, str) and content:
            return f"SYSTEM ({sub}): {content}" if sub else f"SYSTEM: {content}"
        # System event with no string content — note the subtype but skip
        # dumping the rest. Keeps the conversation flow readable.
        return f"SYSTEM ({sub})" if sub else ""

    # Top-level string `content` for unknown event types — preserve
    # forward-compat without leaking raw JSON into embeddings.
    content = raw.get("content")
    if isinstance(content, str) and content:
        label = (ev_type or "EVENT").upper()
        return f"{label}: {content}"
    return ""


def _render_user(raw: dict[str, Any]) -> str:
    msg = raw.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str) and content:
        return f"USER: {content}"
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            text = b.get("text") or ""
            if text:
                parts.append(f"USER: {text}")
        elif bt == "tool_result":
            # Sanitizer already stripped the heavy `content` field; only
            # tool_use_id (+ optional is_error) reach here. Surface
            # success/failure as a one-liner so the conversation flow stays
            # readable.
            tool_id = b.get("tool_use_id") or ""
            label = f"TOOL_RESULT ({tool_id})" if tool_id else "TOOL_RESULT"
            parts.append(f"{label}: error" if b.get("is_error") else f"{label}: ok")
    return "\n".join(parts)


def _render_assistant(raw: dict[str, Any]) -> str:
    msg = raw.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str) and content:
        return f"ASSISTANT: {content}"
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            text = b.get("text") or ""
            if text:
                parts.append(f"ASSISTANT: {text}")
        elif bt == "thinking":
            text = b.get("thinking") or ""
            if text and text.strip():
                parts.append(f"ASSISTANT (thinking): {text}")
        elif bt == "tool_use":
            name = b.get("name") or "tool"
            summary = b.get("summary") or ""
            parts.append(f"TOOL_USE: {name} — {summary}" if summary else f"TOOL_USE: {name}")

    # Note non-default stop_reasons (max_tokens, refusal, …); end_turn is
    # the boring case and noting it would just clutter every assistant turn.
    stop_reason = msg.get("stop_reason")
    if stop_reason and stop_reason not in ("end_turn", "tool_use") and parts:
        parts.append(f"[stop: {stop_reason}]")
    return "\n".join(parts)

    def oauth_install_url(self, customer_id: str, redirect_uri: str) -> str:
        raise NotSupportedByConnector("claude_code uses pairing, not OAuth")

    async def exchange_oauth_code(
        self,
        code: str | None,
        redirect_uri: str,
        extra_params: dict[str, str] | None = None,
    ) -> IntegrationToken:
        raise NotSupportedByConnector("claude_code uses pairing, not OAuth")


@register_connector(SourceSystem.CODEX)
class CodexConnector(ClaudeCodeConnector):
    """Codex CLI sessions, shimmed into Claude-Code shape by the plugin's
    sanitizer. Inherits all parsing / fetch_supplementary / normalize logic
    from ClaudeCodeConnector — only the source label and doc-id prefix differ
    so dashboard queries can distinguish provenance.

    Codex-only fields (sandbox_policy, network policy, developer_instructions,
    sub-agent metadata, MessagePhase, structured reasoning_summary) ride along
    on each event under the `_codex_extras` key. They survive into raw R2
    storage but are not currently parsed into units; a v0.2 native pipeline
    can read them without re-ingest.
    """
    source_system: ClassVar[SourceSystem] = SourceSystem.CODEX
    display_name: ClassVar[str] = "Codex"
    # Source profile (doc_type_prefix "claude_code.", priority 75, 0.5
    # multiplier, 7d half-life) is inherited from ClaudeCodeConnector on
    # purpose: same doc shape, same coalescing semantics, same staleness
    # curve — only the provenance label differs.
    _doc_id_prefix: ClassVar[str] = "codex"
    _agent_label: ClassVar[str] = "codex"
    _session_title_prefix: ClassVar[str] = "Codex session"
