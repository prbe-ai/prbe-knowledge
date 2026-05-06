"""Per-source bootstrap crawler agents.

The ``BackfillWorker`` (``services.synthesis.backfill_app``) claims
``wiki_synthesis_runs`` rows at ``status='pending'`` and instantiates
one ``BackfillAgent`` per claim from this REGISTRY. Each subclass
lives in its own module here (``github_agent.py``, ``slack_agent.py``,
...) and registers itself at import time via ``register_crawler``.

Lane C shipped the registry empty. Lane D added GitHub. Subsequent
lanes add Slack, Linear, Notion, Granola, Claude Code, codebase.
"""

from __future__ import annotations

from services.synthesis.crawlers.base import (
    BackfillAgent,
    BackfillAgentResult,
    BearerResolver,
)

# Module-level registry. Populated at import time by `register_crawler`
# calls in each concrete crawler module. The orchestrator looks up
# subclasses here when resolving the `sources=[...]` payload.
REGISTRY: dict[str, type[BackfillAgent]] = {}


def _register_default_crawlers() -> None:
    """Eagerly register every concrete crawler module that ships today.

    Called at module load time below — matches the eager-registration
    convention used elsewhere in this codebase. The crawler imports happen
    inside the function (rather than at module top) so unit tests that
    monkeypatch REGISTRY don't have to drag a real GitHub client through
    test collection.
    """
    from services.synthesis.crawlers.github import GitHubCrawlerAgent

    REGISTRY[GitHubCrawlerAgent.source] = GitHubCrawlerAgent


_register_default_crawlers()


def register_crawler(cls: type[BackfillAgent]) -> type[BackfillAgent]:
    """Decorator (or plain call) to add a crawler to ``REGISTRY``.

    Usage:

        @register_crawler
        class GitHubCrawlerAgent(BackfillAgent):
            source = "github"
            ...

    Re-registering the same source name overwrites the prior entry —
    tests rely on this so they can substitute a mock crawler. Production
    code shouldn't double-register.
    """
    source = getattr(cls, "source", "")
    if not source:
        raise ValueError(f"{cls.__name__} must set a non-empty `source` ClassVar")
    REGISTRY[source] = cls
    return cls


__all__ = [
    "REGISTRY",
    "BackfillAgent",
    "BackfillAgentResult",
    "BearerResolver",
    "register_crawler",
]
