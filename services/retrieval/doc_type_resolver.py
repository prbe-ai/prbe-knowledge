"""Map Haiku's unqualified doc_type tokens to dotted DocType values.

Haiku says "commit"; the documents table stores `github.commit`. Users in
natural language don't qualify ("show me commits", not "show me github
commits") so we resolve at the dispatcher.

Resolution narrows by `sources` when the caller specified one. Without a
source, "issue" resolves to all source-qualified issue types (Linear,
GitHub, Sentry) and the SQL retriever ANYs them.
"""

from __future__ import annotations

from shared.constants import DocType, SourceSystem

# Token → unqualified bucket. The dotted DocType values that match.
# Keys MUST match shared/services/retrieval/router.py::DOC_TYPE_TOKENS.
_TOKEN_TO_DOC_TYPES: dict[str, tuple[DocType, ...]] = {
    "commit": (DocType.GITHUB_COMMIT,),
    "pr": (DocType.GITHUB_PULL_REQUEST,),
    "issue": (DocType.LINEAR_ISSUE, DocType.GITHUB_ISSUE, DocType.SENTRY_ISSUE),
    "review": (DocType.GITHUB_REVIEW,),
    "release": (DocType.GITHUB_RELEASE,),
    "message": (DocType.SLACK_MESSAGE,),
    "thread": (DocType.SLACK_THREAD,),
    "page": (DocType.NOTION_PAGE,),
    # "ticket" is what users say for Linear issues colloquially.
    "ticket": (DocType.LINEAR_ISSUE,),
    # "comment" maps both kinds — Linear ticket comments and GitHub
    # commit comments. Users say "comment" without qualifying which.
    "comment": (DocType.LINEAR_COMMENT, DocType.GITHUB_COMMIT_COMMENT),
    "session": (DocType.CLAUDE_CODE_SESSION,),
    "meeting": (DocType.GRANOLA_MEETING,),
}


# Source prefix → dotted DocType set, used to narrow by sources filter.
# CODEX shares the `claude_code.` doc_type prefix because the connector
# emits CLAUDE_CODE_SESSION docs for both — provenance differs at
# `source_system`, doc shape is identical. Mapping CODEX here lets a
# `sources=[codex]` retrieval filter resolve a token like "session" to
# `claude_code.session` without dropping zero results.
_SOURCE_PREFIX: dict[SourceSystem, str] = {
    SourceSystem.SLACK: "slack.",
    SourceSystem.LINEAR: "linear.",
    SourceSystem.GITHUB: "github.",
    SourceSystem.NOTION: "notion.",
    SourceSystem.SENTRY: "sentry.",
    SourceSystem.GRANOLA: "granola.",
    SourceSystem.CLAUDE_CODE: "claude_code.",
    SourceSystem.CODEX: "claude_code.",
}


def resolve_doc_type_token(
    token: str | None,
    sources: list[SourceSystem] | None = None,
) -> list[str] | None:
    """Resolve an unqualified token to a list of dotted DocType strings.

    Returns None when the token is None/empty/unknown — callers treat
    None as "no doc_type filter". Returns a non-empty list otherwise.

    When `sources` is provided, narrow the dotted set to those whose
    source-prefix matches at least one allowed source. If the narrowing
    eliminates everything (e.g. token="message" but sources=[github]),
    return None — the user's query is internally inconsistent and we'd
    rather return zero results from the wider filter than hard-fail.
    """
    if not token:
        return None
    matches = _TOKEN_TO_DOC_TYPES.get(token.lower())
    if not matches:
        return None

    if sources:
        allowed_prefixes = {_SOURCE_PREFIX[s] for s in sources if s in _SOURCE_PREFIX}
        narrowed = [
            dt.value for dt in matches if any(dt.value.startswith(p) for p in allowed_prefixes)
        ]
        if not narrowed:
            return None
        return narrowed

    return [dt.value for dt in matches]
