"""Probe knowledge KB — the integrations domain.

Source connectors (kb/handlers/), pollers (kb/polling/, kb/poller/), the
webhook ingestion app (kb/ingestion_app.py), the composed worker process
(kb/worker.py), code-graph ingestion, and wiki synthesis (kb/synthesis/).

Layering rule: kb/ imports engine/; engine/ must NEVER import kb/.
"""
