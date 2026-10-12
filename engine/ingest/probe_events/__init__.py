"""probe-events/1: the shape of one uploaded coding-agent session event.

`probe-events-1.schema.json` is a COPY of research-os
`agent/src/probe/tap_core/probe-events-1.schema.json` (taken at research-os
b5282848d, tap 0.9.11). research-os owns it; refresh this copy when it changes.
One field is here first: `media_block.sha256`, the digest of a pasted image the
client uploaded (kb/session_media.py). research-os's copy gains it with the
tap change that sends it; until then a refresh must keep it.

`project` keeps only what the schema names. Taps before 0.9.10 uploaded more
than the consent screen allows (a second copy of every tool's output, files,
pasted images); projecting a stored event onto the schema removes it.
"""

from engine.ingest.probe_events.project import project_event

__all__ = ["project_event"]
