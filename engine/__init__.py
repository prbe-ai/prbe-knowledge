"""Probe knowledge ENGINE — the content-agnostic core.

Subpackages: shared (config/db/tenancy/models/registry), ingest (queue ->
normalize -> chunk -> embed pipeline + generic ingest doors), retrieval
(vector/bm25/graph retrievers + fusion + HTTP API), mcp (agent tool
surface), community, system_settings.

Layering rule: engine/ must NEVER import kb/. kb/ (source integrations,
polling, webhook app, wiki synthesis) imports engine/; composition of the
two happens in the services/* deploy wrappers.
"""
