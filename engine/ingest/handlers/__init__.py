"""Generic connector contract + engine-door connectors.

base.py / registry.py define the Connector ABC, the connector registry and
the per-source profile registration; custom_ingest.py and manual_upload.py
are the engine's own ingest doors. Source-specific integration connectors
(slack, github, ...) live in kb/handlers/ and register themselves through
the same registry.
"""
