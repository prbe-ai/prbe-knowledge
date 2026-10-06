"""Agent sessions as ATIF trajectories.

`build.build_trajectory` turns a captured session's merged events into an ATIF
v1.8 document, with the frozen `build_reference` builder or by folding the
events' per-event `fragment`s with `fold`; `lines.lines_from_trajectory` turns
that document back into the text the search index reads; `models/` is Harbor's
own pydantic schema, vendored, for validation; `store` reads and writes the
per-session `trajectory.json` in R2.
"""
