"""Companion injection infrastructure: the engine half of the transport.

Deliberately a SIBLING of `engine.shared.wfmem`, not a child. The companion is
the pipe that carries context into a live coding session; workflow memory is
one future WRITER into that pipe. Nothing here imports the store, the
classifier or the serve ledger -- `clause_ids` and `serve_ledger_id` exist in
the schema as reserved columns and stay NULL until the intelligent layer
arrives. Keeping the package outside `wfmem/` makes that decoupling structural
rather than a comment somebody deletes.
"""
