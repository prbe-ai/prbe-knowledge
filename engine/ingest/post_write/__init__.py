"""Post-write pipeline: drains node_post_write_queue and runs analyzers.

Currently runs one analyzer — `AutoMergeAnalyzer` — but the queue schema
+ worker shape are designed so additional analyzers can plug in later
(see `services/ingestion/auto_merge/__init__.py` docstring for the
unified-pipeline plan).
"""

from engine.ingest.post_write.worker import PostWriteWorker

__all__ = ["PostWriteWorker"]
