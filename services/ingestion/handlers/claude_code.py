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

from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from shared import claude_code_extraction as _ext
from shared.constants import (
    DocClass,
    DocType,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload, NotSupportedByConnector
from shared.models import (
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
from shared.storage import get_store

# Cap on simultaneous R2 GETs per fetch_supplementary call. With WORKER
# concurrency=4 and a long session of ~100 batches, unbounded asyncio.gather
# would peak at ~400 in-flight S3 GETs per machine and balloon memory. 16
# is enough to drain a typical session in ~1.5s while keeping in-flight
# envelopes bounded to a few MB.
_FETCH_SUPP_R2_CONCURRENCY = 16


@register_connector(SourceSystem.CLAUDE_CODE)
class ClaudeCodeConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.CLAUDE_CODE
    display_name: ClassVar[str] = "Claude Code"

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
        bucket = store.bucket_for(event.customer_id)
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
        employee_id = self._employee_id_from_event(event, events)

        now = datetime.now(UTC)
        session_doc = self._build_session_doc(
            event=event,
            session_id=session_id,
            cwd=cwd,
            employee_id=employee_id,
            events=events,
            complete=complete,
            now=now,
        )

        documents: list[Document] = [session_doc]
        graph_nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=employee_id,
                properties={"employee_id": employee_id},
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
                source_system=SourceSystem.CLAUDE_CODE,
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
                        source_system=SourceSystem.CLAUDE_CODE,
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
        events: list[dict[str, Any]],
        complete: bool,
        now: datetime,
    ) -> Document:
        body_str = orjson.dumps({"events": events}).decode("utf-8")
        body_bytes = body_str.encode("utf-8")
        content_hash = hashlib.sha256(body_bytes).hexdigest()
        doc_id = f"claude_code:{event.customer_id}:{session_id}"
        first_content = ""
        if events:
            raw = events[0].get("raw") or {}
            first_content = raw.get("content", "") or ""
            if not isinstance(first_content, str):
                first_content = ""
        return Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.CLAUDE_CODE,
            source_id=session_id,
            source_url=f"https://prbe.ai/dashboard/agent-sessions/{session_id}",
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.CLAUDE_CODE_SESSION,
            content_type="application/json",
            content_hash=content_hash,
            title=f"Claude Code session {session_id[:8]}",
            body_preview=first_content[:200],
            body_size_bytes=len(body_bytes),
            body_token_count=0,
            author_id=employee_id,
            created_at=event.received_at,
            updated_at=now,
            valid_from=now,
            ingested_at=now,
            metadata={
                "agent": "claude_code",
                "cwd": cwd,
                "device_id": event.raw_payload.get("device_id"),
                "session_complete": complete,
                "event_count": len(events),
                # `body` drives Normalizer._stringify_body → chunker.
                # Surface a human-readable rendering of the merged events so
                # full-session retrieval works (every other connector follows
                # the same metadata["body"] convention).
                "body": _events_to_text(events),
            },
            acl=self._acl(employee_id),
        )

    def _build_unit_doc(
        self,
        *,
        event: WebhookEvent,
        parent: Document,
        employee_id: str,
        doc_type: DocType,
        unit_kind: str,
        idx: int,
        metadata: dict[str, Any],
        body: str,
        now: datetime,
    ) -> Document:
        source_id = f"{parent.source_id}:{unit_kind}:{idx}"
        doc_id = f"claude_code:{event.customer_id}:{source_id}"
        body_bytes = body.encode("utf-8")
        content_hash = hashlib.sha256(body_bytes).hexdigest()
        return Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.CLAUDE_CODE,
            source_id=source_id,
            source_url=parent.source_url + f"#{unit_kind}-{idx}",
            doc_class=DocClass.RAW_SOURCE,
            doc_type=doc_type,
            content_type="text/plain",
            content_hash=content_hash,
            title=f"{unit_kind} from {parent.source_id[:8]}",
            body_preview=body[:200],
            body_size_bytes=len(body_bytes),
            body_token_count=0,
            author_id=employee_id,
            created_at=now,
            updated_at=now,
            valid_from=now,
            ingested_at=now,
            parent_doc_id=parent.doc_id,
            # `body` drives Normalizer._stringify_body → chunker. Without it,
            # only the 200-char preview gets indexed and the unit's full
            # content (Q+A, code change, decision text, file ref) is lost.
            metadata={**metadata, "body": body},
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


def _events_to_text(events: list[dict[str, Any]]) -> str:
    """Render merged Claude Code events into a chunkable text body.

    Each event becomes one block: `<type>: <content>` for events that have
    string content, or a JSON dump for everything else. Ordering matches the
    line_no-sorted merged stream, so a chunker walking sequentially sees the
    session in transcript order.
    """
    blocks: list[str] = []
    for ev in events:
        raw = ev.get("raw") if isinstance(ev, dict) else None
        if not isinstance(raw, dict):
            blocks.append(orjson.dumps(ev).decode("utf-8"))
            continue
        ev_type = raw.get("type")
        content = raw.get("content")
        if isinstance(content, str) and content:
            blocks.append(f"{ev_type or 'event'}: {content}")
        else:
            blocks.append(orjson.dumps(raw).decode("utf-8"))
    return "\n\n".join(blocks)

    def oauth_install_url(self, customer_id: str, redirect_uri: str) -> str:
        raise NotSupportedByConnector("claude_code uses pairing, not OAuth")

    async def exchange_oauth_code(
        self,
        code: str | None,
        redirect_uri: str,
        extra_params: dict[str, str] | None = None,
    ) -> IntegrationToken:
        raise NotSupportedByConnector("claude_code uses pairing, not OAuth")
