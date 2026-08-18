"""Phase 0 canonical enums. Every string used as a type/label/edge/status lives here."""

import os
from dataclasses import dataclass
from enum import StrEnum


class SourceSystem(StrEnum):
    SLACK = "slack"
    LINEAR = "linear"
    GITHUB = "github"
    NOTION = "notion"
    SENTRY = "sentry"
    GRANOLA = "granola"
    CLAUDE_CODE = "claude_code"
    # Codex CLI sessions arrive shimmed into Claude-Code shape by the plugin's
    # sanitizer. Doc shape and unit extraction are identical to claude_code;
    # this label exists so dashboard queries can distinguish provenance.
    CODEX = "codex"
    MANUAL_UPLOAD = "manual_upload"
    CUSTOM_INGEST = "custom_ingest"
    # Curated team-knowledge layer (runbooks, decisions, service cards, feature
    # notes). Pages are authored programmatically via /api/wiki/pages/* — no
    # external webhook. doc_class distinguishes human authorship (MANUAL_ENTRY)
    # from agent-compiled summaries (COMPILED_WIKI).
    WIKI = "wiki"
    CODE_GRAPH = "code_graph"
    PAGERDUTY = "pagerduty"
    INCIDENT_IO = "incident_io"


# Canonical display labels for each SourceSystem. Exposed to the
# dashboard via /api/sources (TODO) and mirrored verbatim in
# prbe-dashboard/src/lib/sources.ts. The connector classes also carry
# `display_name: ClassVar[str]` for the same string at handler-instance
# scope; these are kept aligned by code review (a future cleanup can
# derive one from the other through the connector registry).
SOURCE_DISPLAY_NAMES: dict[SourceSystem, str] = {
    SourceSystem.SLACK: "Slack",
    SourceSystem.LINEAR: "Linear",
    SourceSystem.GITHUB: "GitHub",
    SourceSystem.NOTION: "Notion",
    SourceSystem.SENTRY: "Sentry",
    SourceSystem.GRANOLA: "Granola",
    SourceSystem.CLAUDE_CODE: "Claude Code",
    SourceSystem.CODEX: "Codex",
    SourceSystem.MANUAL_UPLOAD: "Manual upload",
    SourceSystem.CUSTOM_INGEST: "Custom Ingest",
    SourceSystem.WIKI: "Wiki",
    SourceSystem.CODE_GRAPH: "Code",
    SourceSystem.PAGERDUTY: "PagerDuty",
    SourceSystem.INCIDENT_IO: "incident.io",
}


class DocClass(StrEnum):
    RAW_SOURCE = "raw_source"
    COMPILED_WIKI = "compiled_wiki"
    MANUAL_ENTRY = "manual_entry"
    AGENT_ARTIFACT = "agent_artifact"


class DocType(StrEnum):
    SLACK_MESSAGE = "slack.message"
    SLACK_THREAD = "slack.thread"
    LINEAR_ISSUE = "linear.issue"
    LINEAR_COMMENT = "linear.comment"
    GITHUB_PULL_REQUEST = "github.pull_request"
    GITHUB_ISSUE = "github.issue"
    GITHUB_COMMIT = "github.commit"
    GITHUB_COMMIT_COMMENT = "github.commit_comment"
    GITHUB_REVIEW = "github.review"
    GITHUB_RELEASE = "github.release"
    GITHUB_CODEOWNERS = "github.codeowners"
    NOTION_PAGE = "notion.page"
    NOTION_DATABASE = "notion.database"
    SENTRY_ISSUE = "sentry.issue"
    SENTRY_EVENT = "sentry.event"
    GRANOLA_MEETING = "granola.meeting"
    CLAUDE_CODE_SESSION = "claude_code.session"
    CLAUDE_CODE_QA = "claude_code.qa"
    CLAUDE_CODE_CODE_CHANGE = "claude_code.code_change"
    CLAUDE_CODE_DECISION = "claude_code.decision"
    CLAUDE_CODE_FILE_REF = "claude_code.file_ref"
    #: A standing instruction the researcher gave about HOW to work — verify
    #: before calling it done, spec first then approve, look here, don't touch
    #: that. Separate from `decision` because it is a norm rather than a choice:
    #: it outlives the session it was stated in, which is what makes it worth
    #: aggregating per person. Scoped deliberately narrow — see the DirectiveKind
    #: comment in engine/shared/claude_code_extraction.py for what was measured
    #: out of it and why.
    CLAUDE_CODE_DIRECTIVE = "claude_code.directive"
    MANUAL_UPLOAD_TEXT = "manual_upload.text"
    MANUAL_UPLOAD_MARKDOWN = "manual_upload.markdown"
    MANUAL_UPLOAD_DOCX = "manual_upload.docx"
    MANUAL_UPLOAD_FILE = "manual_upload.file"
    CUSTOM_DOCUMENT = "custom.document"
    # Wiki pages use a free-form `wiki.<type>` doc_type stamped at write
    # time from the LLM-emitted `wiki_type` slug — no enum, no validation
    # gate. The synthesis agent decides what page kinds are useful for a
    # given customer's corpus (typically `repo`, `runbook`, `person`, but
    # nothing prevents new ones). The auto-generated overview page is
    # written under `wiki.index`. Anywhere we need to filter for wiki
    # pages in SQL: `WHERE doc_type LIKE 'wiki.%'`.
    # LEGACY (PR-A pre-Path-2): one Document per symbol. Migration 0050
    # hard-deletes existing rows of this type when the file-as-Document
    # rewrite (CODE_FILE below) ships. Keep the constant defined so the
    # search pipeline + dashboard renderer can recognize stragglers
    # (e.g. an old chunk that escaped DELETE) and still display them.
    CODE_SYMBOL = "code.symbol"
    # Path 2: one Document per file. Body is None; chunks are pre-emitted
    # by the pipeline (one ChunkPiece per symbol body + one metadata chunk
    # carrying repo+file+symbol-list identifying text). The repo name
    # lives in the embedded metadata chunk so semantic search ranks
    # repo-qualified queries correctly.
    CODE_FILE = "code.file"
    INCIDENT = "incident"
    INCIDENT_INVESTIGATION = "incident.investigation"
    # Standalone Document carrying the LLM-drafted + human-approved "why
    # this PR exists" rationale produced by prbe-apps on PR merge. Persisted
    # alongside the FEATURE GraphNode (see feature_nodes_routes.py) so the
    # rationale text lands in BM25 + vector indexes. Prefix is `github.`
    # so doc_type_resolver's SourceSystem.GITHUB narrowing includes it.
    FEATURE_RATIONALE = "github.feature_rationale"
    # Post-approval wiki artifacts authored by the postmortem / wiki-edit
    # agents after an incident investigation is approved AND resolved.
    # These doc_types share the `wiki.` prefix so existing wiki listings
    # (`doc_type LIKE 'wiki.%'`) include them, while remaining
    # distinguishable from human-authored wiki pages.
    #
    # Visibility (DRAFT vs APPROVED) gates retrieval -- DRAFT artifacts are
    # excluded from search until a reviewer approves them via the
    # wiki_review_queue lifecycle.
    WIKI_POSTMORTEM = "wiki.postmortem"
    WIKI_KNOWLEDGE_PAGE = "wiki.knowledge_page"
    WIKI_CORRECTION = "wiki.correction"
    # TODO(post-approval): wiki-listing queries in services/ingestion/wiki_routes.py,
    #   services/synthesis/wiki_agent.py, and services/synthesis/persistence.py fan
    #   over `doc_type LIKE 'wiki.%'` without filtering by visibility. Once the
    #   writeback route (Component 5) starts persisting these doc types as drafts,
    #   those queries need a `visibility = 'approved'` predicate.


class Visibility(StrEnum):
    """Retrieval-visibility gate on a Document / Chunk.

    DRAFT rows are excluded from search and synthesis until promoted to
    APPROVED via the post-approval review pipeline. Used by the
    post-approval wiki artifacts (postmortems, knowledge pages,
    corrections); existing wiki/source documents default to APPROVED at
    write time, matching pre-existing behavior.
    """

    DRAFT = "draft"
    APPROVED = "approved"


# SQL pattern matching every wiki page doc_type (excludes the singleton
# index page so listings don't show themselves). The schema stamps
# wiki pages as `wiki.<wiki_type>` with no validation; this prefix +
# the explicit `<> 'wiki.index'` exclusion is the canonical filter.
WIKI_DOC_TYPE_PREFIX = "wiki."
WIKI_INDEX_DOC_TYPE = "wiki.index"


class NodeLabel(StrEnum):
    """Graph node labels — four canonical kinds post-migration 0091.

    Sub-type discrimination (Module vs Function for code; PR vs Issue for
    documents) lives in ``properties['kind']`` using the typed enums below
    (CodeSymbolKind, DocumentKind). Emit via the factories in shared.models
    (`make_code_symbol`, `make_document`, `make_person`, `make_feature`)
    rather than constructing GraphNodeSpec directly — the factories enforce
    the label-to-kind relationship.

    Other domain labels (SERVICE, SERVICE_CARD, DECISION, RUNBOOK,
    ERROR_GROUP, AGENT, WORKFLOW, FIX_ARTIFACT, VERIFICATION_RESULT) are
    out of scope for the collapse — they're either unused for acme
    today or carry distinct semantics worth preserving.
    """

    # ---- Canonical labels ----
    PERSON = "Person"
    DOCUMENT = "Document"
    FEATURE = "Feature"
    CODE_SYMBOL = "CodeSymbol"

    # ---- Domain labels (untouched by 0091) ----
    SERVICE = "Service"
    ERROR_GROUP = "ErrorGroup"
    SERVICE_CARD = "ServiceCard"
    DECISION = "Decision"
    RUNBOOK = "Runbook"
    AGENT = "Agent"
    WORKFLOW = "Workflow"
    FIX_ARTIFACT = "FixArtifact"
    VERIFICATION_RESULT = "VerificationResult"

    # ---- Research domain ----
    # Generic experiment-tracking vocabulary, not specific to any one
    # tracker: a Project groups Experiments, an Experiment has Runs, a Run
    # produces Artifacts, an Asset is a reusable input/output pinned across
    # runs. Any custom-ingest client modelling that shape can address these.
    #
    # These are deliberately NOT collapsed into DOCUMENT the way migration
    # 0091 collapsed PR/Issue/Ticket/Channel/Repo. That collapse applied to
    # things that were really *documents from a source*; 0091's own docstring
    # keeps distinct-semantics domain labels (Service, Decision, Runbook,
    # Agent, Workflow) separate. A Run is a domain object, not a document
    # about one -- it sits on the same side of that line.
    #
    # Each is paired with its own Document node at ingest and reached through
    # it: the graph retriever's neighbour join is `AND n.label = 'Document'`,
    # so an entity node is an ANCHOR to traverse from, never a returned hit.
    PROJECT = "Project"
    EXPERIMENT = "Experiment"
    RUN = "Run"
    ARTIFACT = "Artifact"
    ASSET = "Asset"

    # ---- Coding-agent sessions ----
    # A coding-agent conversation (Claude Code today; Codex has a handler but
    # no capture plugin yet). It is the JOIN between two independent ingest
    # pipelines: the transcript connector emits it beside the session
    # document, and research-os emits it beside the run document, so a session
    # and the runs it produced become one-hop neighbours through it.
    #
    # It has to be an entity rather than a reference to the session Document:
    # custom-ingest rewrites EVERY Document-labelled edge endpoint through
    # custom_ingest_doc_id() (see engine/ingest/handlers/custom_ingest.py),
    # which is the namespace-confinement control that stops one system
    # attaching edges to another tenant's or another connector's documents.
    # That rewrite is unconditional, so a same-tenant cross-connector
    # reference is mangled just like a hostile one, accepted with a 200, and
    # parked in pending_edges until the reaper drops it. Non-Document labels
    # pass through untouched.
    AGENT_SESSION = "AgentSession"


class CodeSymbolKind(StrEnum):
    """Sub-type of a CODE_SYMBOL node, stored in ``properties['kind']``.

    Tree-sitter extractors classify each emitted symbol with one of these.
    Comparison sites (e.g. ``if symbol.kind == CodeSymbolKind.MODULE``)
    use this enum to keep symbol-kind reasoning type-safe.
    """

    MODULE = "Module"
    FUNCTION = "Function"
    CLASS = "Class"
    METHOD = "Method"
    # Generic "Symbol" — tree-sitter emits this for symbols that don't fit
    # the four categories above (interfaces, enums, constants, etc.).
    SYMBOL = "Symbol"


class DocumentKind(StrEnum):
    """Sub-type of a DOCUMENT node, stored in ``properties['kind']``.

    Optional — plain source documents (a slack message, a notion page) leave
    properties['kind'] unset. The five values below are the categories that
    collapsed INTO the Document label during migration 0091, where the
    sub-type is still meaningful enough to query against.
    """

    PR = "PR"
    ISSUE = "Issue"
    TICKET = "Ticket"
    CHANNEL = "Channel"
    REPO = "Repo"


# ---------------------------------------------------------------------------
# Router entity_type -> graph_nodes.label mapping (single source of truth).
#
# The router (services/retrieval/router.py) emits typed entities like
# {"entity_type": "pr", "canonical_id": "175"}. The graph retriever
# (services/retrieval/retrievers/graph.py) needs to know which NodeLabel
# corresponds to each router type to look those entities up. Historically
# this dict lived inside the retriever, drifted from the router's enum
# (services/retrieval/router.py:118-133), and silently dropped any
# router-emitted type the retriever didn't recognise -- producing zero
# graph hits with no error message.
#
# Keeping this in shared/ alongside NodeLabel guarantees the retriever
# can't fall behind when the router adds a new entity type. mypy catches
# any value that isn't a real NodeLabel; missing keys still need a human
# to add them, but the graph retriever now logs + falls back gracefully
# (see services/retrieval/retrievers/graph.py).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EntityTypeSpec:
    """One router/LLM-facing entity_type and how it addresses graph_nodes.

    Post-0091 the label vocabulary is coarser than the router's, so a label
    alone no longer identifies an entity type: ``pr``, ``issue``, ``ticket``,
    ``channel`` and ``repo`` all live at ``label='Document'`` and are told
    apart by ``properties['kind']``. A flat ``label -> entity_type`` dict
    cannot invert that collapse, which is why grounding's private copy went
    stale (it still keyed on the pre-0091 labels and matched zero rows).

    ``kind is None`` means the label carries no sub-type discriminator, or
    the entity type addresses plain untyped documents.

    ``llm_extractable`` says whether the entity extractor's constrained
    decoding may emit this type. It DEFAULTS TO TRUE deliberately: the
    failure this flag exists to prevent is a type being addressable in the
    graph while the extractor is forbidden from naming it, which is silent
    (the entity validates away and the search runs with no anchors) and cost
    us production searches on the research corpus for an unknown period.
    Opting a type OUT is a visible, deliberate act at the callsite; opting a
    new type IN happens for free, which is the direction that fails safely.
    """

    entity_type: str
    label: NodeLabel
    kind: str | None = None
    llm_extractable: bool = True


# The single source of truth for the entity vocabulary. Everything that needs
# to know "what entity types exist" or "how does one address graph_nodes"
# derives from this tuple:
#
#   ROUTER_ENTITY_TO_LABEL      forward:  entity_type -> label
#   entity_type_for_node()      reverse:  (label, kind) -> entity_type
#   GROUNDING_ENTITY_LABELS     which labels grounding's fuzzy channel scans
#   engine/retrieval/agent/models.py  EntityType Literal (asserted in sync)
#
# Adding a type is a one-line change here. Previously it meant editing three
# hand-maintained maps that had already drifted twice.
ENTITY_TYPE_REGISTRY: tuple[EntityTypeSpec, ...] = (
    # Entity-bearing labels: a node here IS the thing, not a document about it.
    EntityTypeSpec("person", NodeLabel.PERSON),
    EntityTypeSpec("service", NodeLabel.SERVICE),
    EntityTypeSpec("feature", NodeLabel.FEATURE),
    EntityTypeSpec("decision", NodeLabel.DECISION),
    EntityTypeSpec("error_group", NodeLabel.ERROR_GROUP),

    # Collapsed into DOCUMENT by 0091; discriminated by properties['kind'].
    EntityTypeSpec("pr", NodeLabel.DOCUMENT, DocumentKind.PR),
    EntityTypeSpec("issue", NodeLabel.DOCUMENT, DocumentKind.ISSUE),
    EntityTypeSpec("ticket", NodeLabel.DOCUMENT, DocumentKind.TICKET),
    EntityTypeSpec("channel", NodeLabel.DOCUMENT, DocumentKind.CHANNEL),
    EntityTypeSpec("repo", NodeLabel.DOCUMENT, DocumentKind.REPO),

    # Document-addressing types with no kind discriminator. `commit_sha` has
    # no Commit node anywhere (DocumentKind has no COMMIT and no handler emits
    # one) -- it is mapped rather than omitted so that a router-extracted commit
    # resolves to an empty typed lookup instead of being silently dropped by
    # the `if label and cid` guard in engine/retrieval/agent/tools.py.
    EntityTypeSpec("file_path", NodeLabel.DOCUMENT),
    EntityTypeSpec("session", NodeLabel.DOCUMENT),
    EntityTypeSpec("commit_sha", NodeLabel.DOCUMENT),
    # Not extractable: `document` is the generic Document fallback used by
    # reverse resolution, not an anchor a query names. An extractor that
    # emitted it would ground to "every document", which is no narrowing at
    # all -- worse than emitting nothing.
    EntityTypeSpec("document", NodeLabel.DOCUMENT, llm_extractable=False),

    # Research-domain entities. Each keeps its own label (see NodeLabel) and
    # carries no kind discriminator, so reverse resolution is label-only.
    EntityTypeSpec("project", NodeLabel.PROJECT),
    EntityTypeSpec("experiment", NodeLabel.EXPERIMENT),
    EntityTypeSpec("run", NodeLabel.RUN),
    EntityTypeSpec("artifact", NodeLabel.ARTIFACT),
    EntityTypeSpec("asset", NodeLabel.ASSET),

    # Coding-agent session. Deliberately EXTRACTABLE: the graph retriever
    # returns early when the router surfaced no entities, so a non-extractable
    # entity is never an anchor and the node is dead weight. Kind-less, so it
    # also supplies its own reverse mapping in _DEFAULT_ENTITY_TYPE_FOR_LABEL.
    # Node naming lives in agent_session_display_name -- NOT the session
    # document's title, which grounds to nothing for an id query.
    EntityTypeSpec("agent_session", NodeLabel.AGENT_SESSION),

    # Code-graph entities (extracted by tree-sitter at ingest). All collapse
    # to CODE_SYMBOL post-0091.
    #
    # Not extractable, preserving the pre-centralisation Literal exactly.
    # Note these were never in that Literal, so the extractor has never been
    # able to emit them -- flipping any of these to True is a real behaviour
    # change and needs its own evaluation, not a drive-by. Two things gate it:
    # CODE_SYMBOL is excluded from grounding on cost (see
    # _GROUNDING_EXCLUDED_LABELS below), so an emitted symbol would have no
    # grounded candidate to reconcile against; and the prose above this tuple
    # historically claimed "the router can still emit these", which contradicts
    # the extractor's own Literal. Resolve that contradiction before enabling.
    EntityTypeSpec(
        "function", NodeLabel.CODE_SYMBOL, CodeSymbolKind.FUNCTION,
        llm_extractable=False,
    ),
    EntityTypeSpec(
        "method", NodeLabel.CODE_SYMBOL, CodeSymbolKind.METHOD,
        llm_extractable=False,
    ),
    EntityTypeSpec(
        "class", NodeLabel.CODE_SYMBOL, CodeSymbolKind.CLASS,
        llm_extractable=False,
    ),
    EntityTypeSpec(
        "module", NodeLabel.CODE_SYMBOL, CodeSymbolKind.MODULE,
        llm_extractable=False,
    ),
    EntityTypeSpec(
        "symbol", NodeLabel.CODE_SYMBOL, CodeSymbolKind.SYMBOL,
        llm_extractable=False,
    ),
)


# The extractor's emittable vocabulary, DERIVED so it cannot drift from the
# registry. This tuple is what `EntityType` in
# engine/retrieval/agent/models.py builds its Literal from, which in turn is
# serialised into the extractor's constrained-decoding response_format.
#
# It replaces a second, hand-maintained copy of this list. That copy went
# stale exactly the way a duplicated list does: the research-domain types
# (experiment / project / run / artifact / asset) were added to the registry
# and to grounding, but nobody added them here, so grounding surfaced
# Experiment candidates, the model correctly emitted entity_type="experiment",
# and Pydantic rejected the whole extraction. Searches returned zero entity
# anchors while still reporting state:"ok". There is now one list.
LLM_EXTRACTABLE_ENTITY_TYPES: tuple[str, ...] = tuple(
    sorted({
        spec.entity_type
        for spec in ENTITY_TYPE_REGISTRY
        if spec.llm_extractable
    })
)


# Reverse resolution for a node whose properties['kind'] is unset. A label
# alone is ambiguous across the kind-less types above (file_path / session /
# commit_sha / document all sit at Document with no kind), so a generic member
# is declared per ambiguous label and everything else is DERIVED.
#
# This map used to be spelled out by hand and it drifted exactly the way the
# registry comment above says a hand-maintained list drifts: AgentSession was
# added to the registry, to ROUTER_ENTITY_TO_LABEL, to GROUNDING_ENTITY_LABELS
# and to the extractor's Literal -- and omitted here. Grounding's fuzzy SQL
# matched those nodes and then dropped every one of them, because its only
# consumer does `if not entity_type: continue`. The node was addressable,
# extractable, grounded by SQL, and unreachable in practice.
#
# Only genuinely ambiguous labels need a hand-picked winner; a label owned by a
# single kind-less spec resolves to that spec.
_AMBIGUOUS_LABEL_WINNER: dict[NodeLabel, str] = {
    NodeLabel.DOCUMENT: "document",
    NodeLabel.CODE_SYMBOL: "symbol",
}

_DEFAULT_ENTITY_TYPE_FOR_LABEL: dict[NodeLabel, str] = {
    **{
        spec.label: spec.entity_type
        for spec in ENTITY_TYPE_REGISTRY
        if spec.kind is None
    },
    **_AMBIGUOUS_LABEL_WINNER,
}




ROUTER_ENTITY_TO_LABEL: dict[str, NodeLabel] = {
    spec.entity_type: spec.label for spec in ENTITY_TYPE_REGISTRY
}


# (label, kind) -> entity_type, for specs that carry a kind discriminator.
_NODE_TO_ENTITY_TYPE: dict[tuple[str, str], str] = {
    (spec.label.value, spec.kind): spec.entity_type
    for spec in ENTITY_TYPE_REGISTRY
    if spec.kind is not None
}


# Labels grounding's fuzzy-entity channel scans.
#
# DOCUMENT is deliberately excluded (long-standing behaviour): it is the
# largest label, and document titles are matched by grounding's separate
# doc-title channel rather than by entity resolution.
#
# CODE_SYMBOL is excluded on cost, not principle. grounding's predicate is
#     coalesce(properties->>'name','') % $2
# while the trigram index is
#     idx_graph_nodes_name_trgm ON (LOWER(properties->>'name') gin_trgm_ops)
# The expressions do not match, so that GIN index cannot serve this query --
# it degrades to a scan of every row for each scanned label. That is fine for
# the small entity labels below, but CodeSymbol is ~100k rows/tenant at
# probe-founders scale. Align the predicate with the index before adding it.
_GROUNDING_EXCLUDED_LABELS: frozenset[NodeLabel] = frozenset({
    NodeLabel.DOCUMENT,
    NodeLabel.CODE_SYMBOL,
})

def agent_session_canonical_id(agent: str, session_id: str) -> str:
    """canonical_id for an AGENT_SESSION node.

    CROSS-REPO CONTRACT. research-os composes this exact string independently
    (it cannot import this package) and asserts an EDGE to it; this connector
    asserts the NODE. One writer owns the node's groundable name, and the
    run-side edge parks in pending_edges until the node lands. Both sides hold
    the same two inputs: the agent label and the agent's session id.

        agent_session:{agent}:{session_id}

    Changing the shape silently un-merges the two sides: each keeps writing a
    node, no error is raised, and the edges simply stop meeting. There is no
    validation that can catch it from inside one repo, which is why the
    post-deploy check asserts the join THROUGH the retrieval API rather than
    against graph_nodes.
    """
    return f"agent_session:{agent}:{session_id}"


def agent_session_display_name(
    agent: str, session_id: str, person: str | None = None
) -> str:
    """properties['name'] for an AGENT_SESSION node.

    The ENGINE owns this name. research-os asserts only the edge, so there is
    exactly one writer and the node's name cannot flip-flop between two sides
    that know different things about the session.

    Grounding decides whether this node is reachable at all, matching
    ``properties->>'name'`` through pg_trgm similarity (diluted by every
    character NOT in the query) and a tsvector word match. Measured against a
    live index, per query style:

        name shape                       full uuid   "<person> ... session"
        session document's title          0.118 no    0.625 yes
        "<agent> session <full id>"       0.673 yes   0.269 no
        "<person> <agent> session <id>"   0.552 yes   0.448 yes   <- this

    The first version reused the session document's title and grounded to
    NOTHING for an id query: the email address in that title contributed enough
    non-matching trigrams to drop similarity under the 0.3 threshold, and the id
    was truncated to 8 characters so naming the real session could not match.
    The node existed, ingest reported success, and the entity was unreachable --
    the failure make_named_entity's docstring warns about, reached by a
    different route.

    So: the FULL id (an id query is the precise one), the person when known
    (the human phrasing), the agent with underscores spaced out so it
    tokenises, and NO email.
    """
    spaced = agent.replace("_", " ")
    if person:
        return f"{person} {spaced} session {session_id}"
    return f"{spaced} session {session_id}"
GROUNDING_ENTITY_LABELS: tuple[str, ...] = tuple(
    sorted({
        spec.label.value
        for spec in ENTITY_TYPE_REGISTRY
        if spec.label not in _GROUNDING_EXCLUDED_LABELS
    })
)


# Every label grounding is allowed to SCAN must reverse to an entity type, or
# grounding silently discards the rows it just matched. Asserted at import so a
# new registry entry cannot reintroduce the AgentSession failure.
_UNREVERSIBLE = [
    label for label in GROUNDING_ENTITY_LABELS
    if NodeLabel(label) not in _DEFAULT_ENTITY_TYPE_FOR_LABEL
]
if _UNREVERSIBLE:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "grounding would scan and then discard these labels because they have "
        f"no reverse entity_type: {_UNREVERSIBLE}. Add them to "
        "ENTITY_TYPE_REGISTRY as a kind-less spec, or to "
        "_AMBIGUOUS_LABEL_WINNER."
    )


# The entity types grounding's fuzzy channel can actually put in front of the
# extractor, as types rather than labels.
#
# Every one of these MUST be llm_extractable, and models.py enforces it at
# import. The reasoning is end-to-end: grounding hands the model a candidate
# of this type, the prompt invites the model to name it, so a model doing its
# job emits it -- and if it is not emittable, Pydantic rejects the extraction.
# That is not a degraded result, it is a zero result, and it is silent. This
# is the precise shape of the research-corpus outage: Experiment and Project
# candidates were grounded and offered for weeks while the extractor was
# forbidden from naming them.
#
# Note this is deliberately NOT "resolvable via entity_type_for_node". Types
# like file_path / session / commit_sha sit at Document with no kind, so the
# label-only default resolves them to `document` and they are unresolvable by
# that route -- yet they are legitimately emittable, reaching the graph through
# bare-ID detection instead. Grounding-addressability is the invariant that
# matches the failure; resolvability is not.
GROUNDING_ADDRESSABLE_ENTITY_TYPES: tuple[str, ...] = tuple(
    sorted({
        spec.entity_type
        for spec in ENTITY_TYPE_REGISTRY
        if spec.label not in _GROUNDING_EXCLUDED_LABELS
    })
)


def entity_type_for_node(label: str, kind: str | None = None) -> str | None:
    """Reverse a graph_nodes row into the router entity_type that names it.

    Returns None for a label outside the registry (e.g. Runbook, Agent),
    which callers treat as "not an addressable entity".
    """
    if kind:
        typed = _NODE_TO_ENTITY_TYPE.get((label, kind))
        if typed is not None:
            return typed
    try:
        return _DEFAULT_ENTITY_TYPE_FOR_LABEL.get(NodeLabel(label))
    except ValueError:
        return None


def node_addressing_for_entity_type(entity_type: str) -> tuple[NodeLabel, str | None] | None:
    """Forward: entity_type -> (label, kind) for an exact graph_nodes lookup.

    Used by grounding's bare-id channel, which must match `PR #340` against
    (label='Document', kind='PR') -- matching on a bare `label='PR'` has
    found nothing since migration 0091.
    """
    for spec in ENTITY_TYPE_REGISTRY:
        if spec.entity_type == entity_type:
            return spec.label, spec.kind
    return None


class EdgeType(StrEnum):
    OWNS = "OWNS"
    MENTIONS = "MENTIONS"
    AUTHORED = "AUTHORED"
    BLOCKS = "BLOCKS"
    SUPERSEDES = "SUPERSEDES"
    DUPLICATES = "DUPLICATES"
    TOUCHES = "TOUCHES"
    FIRES_IN = "FIRES_IN"
    MEMBER_OF = "MEMBER_OF"
    LINKED_FROM = "LINKED_FROM"

    CONFLICTS_WITH = "CONFLICTS_WITH"
    VERIFIED_BY = "VERIFIED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    FIXES = "FIXES"
    REGRESSES = "REGRESSES"
    ASSIGNED_TO = "ASSIGNED_TO"
    COMPILED_FROM = "COMPILED_FROM"
    DESCRIBES = "DESCRIBES"

    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    REFERENCES = "REFERENCES"
    DEFINED_IN = "DEFINED_IN"

    # LLM-inferred edge types (Lane B). These are emitted only by the
    # inferred_edges extractor and carry INFERRED or AMBIGUOUS confidence.
    DISCUSSES = "DISCUSSES"
    DOCUMENTS = "DOCUMENTS"
    RESOLVES = "RESOLVES"
    MENTIONS_ENTITY = "MENTIONS_ENTITY"
    RELATES_TO = "RELATES_TO"


class EdgeConfidence(StrEnum):
    """Confidence tier on a graph edge — mirrors the string literals used
    in `graph_edges.confidence` SQL CASEs and `_stronger_confidence`
    (services/ingestion/graph_writer.py). StrEnum so members compare
    equal to the bare string ("EXTRACTED" == EdgeConfidence.EXTRACTED),
    keeping the existing string-based SQL + asyncpg parameter binding
    paths working unchanged.
    """

    EXTRACTED = "EXTRACTED"  # explicit upstream signal (webhook, API field)
    INFERRED = "INFERRED"    # LLM-derived from text
    AMBIGUOUS = "AMBIGUOUS"  # LLM-derived with low certainty


class PrincipalType(StrEnum):
    USER = "user"
    GROUP = "group"
    CHANNEL = "channel"
    WORKSPACE = "workspace"
    SYSTEM = "system"
    AGENT = "agent"


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class QueueStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    DLQ = "dlq"


class IngestionEventStatus(StrEnum):
    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    SKIPPED = "skipped"


class IngestionEventType(StrEnum):
    WEBHOOK = "webhook"
    SYNC = "sync"
    BACKFILL = "backfill"
    MANUAL = "manual"
    REPROCESS = "reprocess"


class BackfillStatus(StrEnum):
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class IntegrationStatus(StrEnum):
    ACTIVE = "active"
    AUTH_FAILED = "auth_failed"
    REVOKED = "revoked"


class CustomerStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class CompileTrigger(StrEnum):
    SCHEDULED = "scheduled"
    SOURCE_UPDATE = "source_update"
    MANUAL = "manual"
    QUERY_FILING = "query_filing"
    NORMALIZER_REPROCESS = "normalizer_reprocess"


class RefType(StrEnum):
    MENTIONS = "mentions"
    LINKS_TO = "links_to"
    EMBEDS = "embeds"
    REPLIES_TO = "replies_to"


class AttachmentKind(StrEnum):
    IMAGE = "image"
    FILE = "file"
    URL = "url"
    CODE_LINK = "code_link"
    BLOCK_REFERENCE = "block_reference"


class EntityType(StrEnum):
    SERVICE = "service"
    REPO = "repo"
    PERSON = "person"
    TICKET = "ticket"
    PR = "pr"
    ERROR_GROUP = "error_group"
    FEATURE = "feature"
    DECISION = "decision"
    FILE_PATH = "file_path"
    CHANNEL = "channel"


# Gemini-2 is the sole embedder (OpenAI text-embedding-3-large was retired
# 2026-05-14, PR #263; the OpenAI embedder + SDK were stripped in a follow-up).
# The v1 column on `chunks` is nullable and unindexed post-0067; reads target
# `embedding_v2`.
#
# Treat the two V2 constants below as the single source of truth for
# "what's the embedder?" — swapping models later means flipping these
# values, not chasing string literals across the codebase.
EMBEDDING_V2_MODEL = "google/gemini-embedding-2"
EMBEDDING_V2_DIM = 3072
# Bare model id as exposed by the LiteLLM proxy's `model_list` (matches the
# `gemini-embedding-*` alias). Use this when routing through the gateway;
# prefixing with `gemini/` falls through to the proxy's `*` catch-all and
# returns "invalid model ID" because the catch-all routes to OpenAI.
# Direct-SDK callers (google-genai) also accept this bare form.
EMBEDDING_V2_PROXY_ALIAS = "gemini-embedding-2"
# Per https://ai.google.dev/gemini-api/docs/embeddings, gemini-embedding-2
# accepts up to 8192 input tokens. The chunker's DEFAULT_CHUNK_TOKENS (512)
# is well under this; this constant is the absolute upper bound the chunker
# is allowed to use so we can't silently truncate Gemini-side input if the
# chunker is ever retuned.
EMBEDDING_V2_MAX_INPUT_TOKENS = 8192
CHUNKER_VERSION = "naive-v1"

# Per-symbol cap for code_graph chunks. Matches DEFAULT_CHUNK_TOKENS so code
# and prose live on the same retrieval scale: a unified retriever ranks
# candidates across sources and assumes chunks are roughly comparable units.
# Pre-cap, individual code symbols could land as 6000+ token chunks, which
# (a) made BM25 fire harder on identifier tokens than a 512-token prose
# chunk and (b) blew Anthropic 25KB tool-result caps on Probe MCP responses
# (one symbol = 30KB). 0.3x demote (commit 7745043c) was a band-aid for the
# ranking side; this constant attacks the size mismatch at the source.
MAX_SYMBOL_CHUNK_TOKENS = 512
NORMALIZER_VERSION = "v1"

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# Inferred-edges (Lane B) extractor configuration. Lives here per the
# RRF_K / RRF_BREADTH_ALPHA tuning-knob convention so eval sweeps and
# per-tenant overrides don't require code edits.
#
# Model selection: picked Gemini 3.1 Flash Lite over Claude Haiku 4.5
# after a real-prod eval showed Flash Lite + wider bundle produces
# better edge quality (65% specificity vs 49%, 0% vs 8% hallucination
# rate) at the same cost as Haiku-current. See
# scripts/eval_inferred_edges_widebundle.py.
#
# Provider dispatch is by prefix in engine.ingest.inferred_edges.
# extractor: "claude-*" -> anthropic SDK; "gemini-*" -> google-genai.
INFERRED_EDGES_MODEL = "gemini-3.1-flash-lite"

# Pricing per 1M tokens, as of 2026-05. Used only for the cost_usd
# telemetry gauge; pipeline correctness is unaffected by drift here.
INFERRED_EDGES_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M) USD
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}

# Bundle caps. Wider than v1 (60K / 20 / 5 / 10) because Flash Lite's
# 1M context window plus its better specificity at higher candidate
# counts means the LLM picks more specific edge_types when given more
# evidence. Real-prod eval: at v1 caps Flash Lite was 37% specific
# (worse than Haiku); at these caps it's 65% specific (best of all
# tested combos). Wider bundle ate Flash Lite's cost advantage --
# net cost is ~the same as Haiku-v1 -- but quality is higher.
INFERRED_EDGES_BUNDLE_TOKEN_BUDGET = 300_000
INFERRED_EDGES_BUNDLE_MAX_1HOP = 50
INFERRED_EDGES_BUNDLE_MAX_VECTOR_SIMILAR = 20
INFERRED_EDGES_BUNDLE_MAX_TIME_WINDOW = 30

# Models advertised by the /query synthesis layer. Keys are the active
# "<provider>/<model>" identifiers callers (and the dashboard picker) use;
# values are provider names the synthesis dispatcher uses to pick a client.
SYNTHESIS_MODELS: dict[str, str] = {
    "anthropic/claude-haiku-4-5-20251001": "anthropic",
    "anthropic/claude-sonnet-4-6": "anthropic",
    "google/gemini-3.6-flash": "google",
    "google/gemini-3.5-flash-lite": "google",
}

# Submitted model ids from the previous picker remain accepted during the
# rollout, but intentionally live outside SYNTHESIS_MODELS so they are not
# advertised as active choices or included in validation error messages.
SYNTHESIS_MODEL_ALIASES: dict[str, str] = {
    "google/gemini-3-flash-preview": "google/gemini-3.6-flash",
    "google/gemini-3.1-flash-lite": "google/gemini-3.5-flash-lite",
}
DEFAULT_SYNTHESIS_MODEL = "anthropic/claude-sonnet-4-6"

MAX_WEBHOOK_ATTEMPTS = 5
QUEUE_HEARTBEAT_INTERVAL_SECONDS = 30
QUEUE_RECLAIM_THRESHOLD_SECONDS = 300

# Default queue priority at enqueue time. Worker._claim_one orders by
# priority DESC, so higher numbers claim first. Tiers:
#
#   100  — interactive webhooks: github, slack, notion, linear, granola, sentry
#    75  — bursty agent/custom batches (claude_code, codex, custom_ingest)
#    50  — backfill rows (set in backfill_runner.py); never blocks live work
#
# Per-source overrides live on the connector classes and are read through
# shared.source_registry (see SourceProfile.ingestion_priority); sources
# without a registered profile fall back to this default.
DEFAULT_INGESTION_PRIORITY = 100

TOP_K_VECTOR = 50
TOP_K_BM25 = 50
TOP_K_GRAPH = 20
TOP_K_DIRECTED = 20
RRF_K = 60
DEDUP_COSINE_THRESHOLD = 0.95

# Cap on caller-provided `source_keys` in QueryRequest. Each key becomes a
# `documents.metadata->>'source_key' = ANY(...)` array member in every
# retrieval channel's SQL; the cap bounds both the array size and the
# request payload. Consumers scoping to a workspace lens pass 1-2 keys;
# 50 leaves generous headroom for multi-corpus callers.
MAX_REQUEST_SOURCE_KEYS = 50

# Graph-explore endpoint (POST /graph/explore + /graph/search) caps. These
# bound the visualization payload that the dashboard renders client-side --
# force-directed layout starts to crawl above a few thousand nodes, and the
# wire payload itself dwarfs everything else above 5k edges. Lives here per
# the RRF_K / RRF_BREADTH_ALPHA tuning-knob convention so tuning doesn't
# require an env-var deploy.
#
# Default mode: top-N nodes by graph_nodes.degree DESC, 1-hop edges among
# the selected set. Anchor mode: tiered BFS centered on a node, hop1 cap +
# hop2 cap (total = hop1 + hop2). Edge cap is enforced regardless of node
# count; if hit, truncated=True flips in the response. WHY_MAX_CHARS caps
# per-edge LLM-generated rationale at serialization time.
GRAPH_EXPLORE_NODE_CAP = 2000
GRAPH_EXPLORE_EDGE_CAP = 5000
GRAPH_EXPLORE_HOP1_CAP = 500
GRAPH_EXPLORE_HOP2_CAP = 1500
GRAPH_EXPLORE_WHY_MAX_CHARS = 200
GRAPH_SEARCH_DEFAULT_LIMIT = 10
GRAPH_SEARCH_MAX_LIMIT = 25

# Directed-vectors feature: doc-level retrieval signal contributed by
# per-document trigger phrases stored in the directed_vectors table.
# Eval-tuned; commits in the same change that bumps it. Set to 0.0 to
# disable contribution without removing the retriever from the fan-out.
DIRECTED_RETRIEVAL_WEIGHT: float = 1.0

# Cap on LLM-generated directed phrases per wiki document. Engineer-pinned
# phrases get their own cap (MAX_HUMAN_DIRECTED_PER_DOC) so a runaway LLM
# can't bury legitimate pins.
MAX_DIRECTED_VECTORS_PER_DOC: int = 16

# Cap on engineer-pinned directed phrases per wiki document. Higher than
# the LLM cap because explicit pins are intentional, but bounded so a
# malicious / typo'd frontmatter can't balloon the table.
MAX_HUMAN_DIRECTED_PER_DOC: int = 32

# Per-phrase character cap. Trigger phrases are short by design (5-12
# tokens per the prompt); 256 chars is generous slack against natural
# English while still rejecting megabyte-long pathological inputs that
# would bloat embedding cost / storage / log noise.
MAX_DIRECTED_PHRASE_CHARS: int = 256

# Cosine distance threshold below which two candidate trigger phrases are
# considered near-duplicates and one is dropped (humans always win on
# collision; LLM duplicates of human pins are suppressed).
DIRECTED_DEDUPE_COSINE_THRESHOLD: float = 0.05

# Doc-grouped fusion: weight applied to the sum of NON-best content-chunk RRF
# scores when collapsing per-doc. doc_score = max(rrfs) + alpha * sum(others) +
# metadata_sum. Prevents long docs from drowning shorter ones; preserves
# best-chunk-wins-ties; rewards docs whose multiple chunks all matched.
RRF_BREADTH_ALPHA = 0.3

# Per-source post-RRF score multipliers moved to the source registry:
# each connector declares `score_multiplier` where the source is defined
# (services/ingestion/handlers/<source>.py) and fusion reads it through
# shared.source_registry. Unregistered sources get the neutral 1.0.

# Baseline recency half-life (days) applied to every source. Smaller = faster
# decay. Acts as the universal floor so backfilled tenants don't see 8-12 month
# old docs ranked equally with last week's. Per-source overrides below win when
# a source is noisier than baseline and needs faster decay.
#
# At 120d: a 4-month-old doc keeps 50% of its score, 8-month 25%, 12-month 12%.
# Strongly-relevant old docs still win on raw signal; tied semantic matches go
# to the fresher one.
DEFAULT_RECENCY_HALF_LIFE_DAYS = 120.0

# Per-source half-life overrides moved to the source registry: connectors
# declare `half_life_days` where the source is defined and fusion resolves
# per-source override > caller-supplied baseline > the default above via
# shared.source_registry.half_life_days_for.

# Inferred-edge retrieval channel tuning. The channel walks INFERRED Doc-Doc
# edges from top-K primary docs and surfaces the linked docs as additional
# results with `matched_via.channel = "inferred_edge"`. Knobs live here so
# eval sweeps + per-tenant overrides can adjust without code edits, per the
# RRF_K / RRF_BREADTH_ALPHA precedent (feedback_prbe_knowledge_tuning_consts).

# Cap on linked docs returned by `inferred_edge_search`. 5 keeps the result
# set predictable; the inferred-edge channel is supplementary, not primary.
INFERRED_EDGE_TOP_K = 5

# Per-hop dampening applied to inferred-edge results. The final score is:
#   dampening * 1/(1 + anchor_rank) * score_multiplier(src)
#                                  / (1 + ln(linked_edge_count))
# 0.2 keeps a 2-hop result (anchor at rank 1) at base score 0.10 -- well
# below direct vector hits but still surfacing above weakly-matched primary
# docs. Was 0.5 in v1; lowered after observing inferred-edge results
# outranking the primary doc that surfaced them (the codex session #1
# case where a 2-hop chained inference dominated rank 1 for a query the
# linked doc didn't actually contain).
INFERRED_EDGE_DAMPENING = 0.2

# Max chunks per inferred-edge-derived QueryDocumentResult. Hydrated from
# the chunks table by `chunk_index ASC` (first chunks are usually the most
# identity-bearing -- title metadata + opening body). Without hydration the
# chunks list is empty and the dashboard renders "0 matched"; the
# synthesizer also can't cite the doc -- both regressions of the v1 design.
INFERRED_EDGE_HYDRATION_CHUNKS = 3

# ---- Search agent (gatherer) -------------------------------------------------
# The gatherer is the retrieval pipeline (see
# docs/specs/agentic-search.md). Tunables below are read at agent loop
# construction; changing them requires a redeploy (no hot-reload).

# IMPORTANT: no provider-prefix expansion. When `shared.llm.acompletion`
# forces `custom_llm_provider="openai"` for gateway-routed calls (so the
# upstream honors response_format), LiteLLM forwards this model id
# verbatim in the OpenAI chat-completion request body. The LiteLLM proxy
# matches `model_name: "cerebras/*"` in its modelList (per
# prbe-backend/charts/litellm/values.yaml) and the same model id flows
# through to the upstream call.
#
# Env-overridable so we can A/B-test alternative providers without a code
# change. Set SEARCH_AGENT_INFERENCE_MODEL on the retrieval pod to e.g.
# `claude-sonnet-4-6` (routed via the proxy's claude-* modelList entry)
# and that's what gets called.
#
# Default flipped 2026-05-18 from `accounts/fireworks/models/gpt-oss-120b`
# to `cerebras/gpt-oss-120b`. Same model id, ~10x the output throughput
# on Cerebras's wafer-scale chips — eliminated the 90s gatherer-timeout
# cascade on conceptual queries. The Fireworks route was simultaneously
# dropped from the LiteLLM modelList (prbe-backend PR #342), so a stale
# install relying on the old default now 404s at the proxy with a clean
# error instead of silently routing through the catch-all.
SEARCH_AGENT_INFERENCE_MODEL = os.getenv(
    "SEARCH_AGENT_INFERENCE_MODEL",
    "cerebras/gpt-oss-120b",
)

# Soft budget: total tool calls across all turns. Covers turn-1 mandatory 4
# + ~16 exploration calls across 2-3 follow-up turns. The agent may extend
# by emitting need_deeper{reason}; +10 per extension, max 2 extensions.
SEARCH_AGENT_TOOL_BUDGET = 20
SEARCH_AGENT_EXTENSION_GRANT = 10
SEARCH_AGENT_MAX_EXTENSIONS = 2

# Soft turn cap: once the model has completed this many turns without
# emitting the terminal, the next exploration turn triggers a forcing
# nudge to call `emit_gatherer_output` on the turn after that. Same
# mechanism as the budget-exhausted nudge, but tripped by turn count
# instead of tool-call count.
#
# Set to 1 because the parallel 4-channel fan-out (vector + BM25 +
# graph + inferred edges) already runs PRE-LOOP in run_gatherer and
# its results are baked into the model's turn-1 evidence pack as
# `<channel_results>`. Vector and BM25 already surface anything one
# hop from a real answer, so the only useful in-loop exploration is
# at most one 1-hop follow-up (graph_walk / fetch_doc); beyond that
# is provably noise.
#
# Concretely the loop runs at most:
#   model turn 1 — sees prefanout, may emit terminal OR pick one
#                  exploration tool (cap not yet tripped: turn_count
#                  was 0 on entry).
#   model turn 2 — should emit terminal; if it picks another tool,
#                  the cap (turn_count was 1 on entry, >= 1) fires
#                  on the way out of this iteration.
#   model turn 3 — forced-emit turn after the nudge (only reached
#                  if turns 1 and 2 both explored).
#
# Effective ceiling: SOFT_TURN_CAP + 2 = 3 LLM turns (~3-5s on
# Cerebras), down from the prior 67-90s oscillation that hit the
# 90s SEARCH_AGENT_LOOP_TIMEOUT_SECONDS wall-clock.
SEARCH_AGENT_SOFT_TURN_CAP = 1

# Hard ceiling. Even with extensions the agent never exceeds this.
SEARCH_AGENT_HARD_CAP = SEARCH_AGENT_TOOL_BUDGET + (
    SEARCH_AGENT_EXTENSION_GRANT * SEARCH_AGENT_MAX_EXTENSIONS
)  # 40

# Min curated results the agent should return before falling back to "no
# confident match". If the agent emits fewer than this, harness logs a
# `gatherer.under_min_output` anomaly for trace review.
SEARCH_AGENT_MIN_OUTPUT = 5

# Per-tool top_k defaults. The agent may override at call time. See plan
# section "Per-tool top_k defaults" for the bytes/turn budget reasoning.
#
# Vector + BM25 carry the recall load on the turn-1 pre-fan-out: those two
# channels are the only ones that fire when grounding resolves no graph
# entity (the common case for free-text / conversational-memory queries,
# where there's no PR / Linear / Slack anchor to seed graph + inferred_edge
# on). When the answer turn falls outside a 15-deep cosine/BM25 window it
# never reaches the gatherer, and no in-loop reformulation can recover a
# candidate the channels never surfaced — a hard recall ceiling. Widen the
# turn-1 window to 30 so more borderline-relevant turns reach the LLM for
# curation. This is the single pre-fan-out call (no extra turns, so no
# added LLM round-trips) and the gatherer's curation guidance already
# defaults to keeping candidates the consumer can filter, so the broader
# pool lifts recall without re-ranking. Graph / inferred_edge stay at 10:
# they're entity-anchored and rarely the recall bottleneck.
SEARCH_AGENT_VECTOR_TOP_K = 30
SEARCH_AGENT_BM25_TOP_K = 30
SEARCH_AGENT_GRAPH_TOP_K = 10

# Graph-channel budget when QueryRequest.discovery is set. Discovery asks the
# engine to favour structurally surprising connections, and the graph channel
# is the only one that carries that signal: graph_search computes a per-edge
# surprise score (engine/retrieval/surprise.py) and returns its hits already
# sorted by it (retrievers/graph.py). Widening this budget therefore admits
# more of the surprise-ranked tail into the gatherer's evidence pool without
# touching vector or BM25.
#
# This replaces the pre-cutover implementation, which multiplied graph hits'
# reciprocal-rank contribution inside fusion.py. That module was deleted with
# the agentic cutover (see retrieval/pipeline.py) and `discovery` had been
# silently inert ever since -- accepted on the wire, plumbed onto QueryRequest,
# read by nothing.
#
# The invariants the old fusion tests encoded are preserved:
#   * off by default                    -- discovery defaults to False
#   * graph channel ONLY                -- vector/bm25 budgets are untouched
#   * bounded                           -- _clamp_top_k caps at _HARD_TOP_K_CAP
#   * two-sided, not just amplifying    -- hits are surprise-ORDERED, so
#     low-surprise edges (hub<->hub, score < 1.0) sort to the tail and fall
#     outside the budget rather than being promoted
SEARCH_AGENT_GRAPH_TOP_K_DISCOVERY = 30
SEARCH_AGENT_INFERRED_EDGE_TOP_K = 10
SEARCH_AGENT_GRAPH_WALK_TOP_K = 20
SEARCH_AGENT_EXPAND_NEIGHBORS_TOP_K = 10
# fetch_doc page size. fetch_doc is now a PAGINATED whole-doc reader
# (doc_id, offset, limit): each call returns at most this many chunks
# starting at `offset` in chunk_index order, so the agent walks a long doc
# deliberately instead of one call hauling the whole thing. The `offset`
# also fixes the prior bug where `ORDER BY chunk_index ASC LIMIT 10` could
# never reach a matched chunk past index 9 on a long doc.
SEARCH_AGENT_FETCH_CHUNKS_MAX = 10

# ---- Context-overflow protection (gatherer) ---------------------------------
# The gatherer sends its whole growing message history (turn-1 pre-fan-out
# dump + every tool result) to Cerebras gpt-oss-120b on EVERY turn. The
# model's hard context window is 131,072 tokens; exceeding it is a provider
# 400 (ContextWindowExceededError), which the loop now degrades to a 200
# passthrough rather than a 503. Two independent guards keep the payload
# under the ceiling (see loop._render_prefanout_budgeted and
# _enforce_context_budget):
#
#  1. PREFANOUT_TOKEN_BUDGET caps the turn-1 <channel_results> dump, which
#     was previously an uncapped json.dumps of every hit with full content.
#     Docs are kept in fused-RRF order until the budget fills (Top-N, no
#     per-source floor — the recall eval guards against source masking; add
#     a floor only if it regresses). ~35 chunks fit at the 512-tok chunk
#     size (was ~80 while this budget was 40_000).
#
#     NOTE this guard is no longer only about the 131,072 context window.
#     It is now also the primary control on PROVIDER QUOTA — see the budget's
#     own comment below. The context window bounds one request; the quota
#     bounds how many can run at once, and the turn-1 dump is charged again
#     on every turn.
#  2. MAX_CONTEXT_TOKENS is the running backstop enforced before every LLM
#     turn: if accumulated messages exceed it, the oldest tool results are
#     truncated in place (message pairing preserved) until the payload fits.
#     Set well under 131,072 because the cl100k count we estimate with
#     diverges from gpt-oss's true tokenizer — headroom absorbs the drift.
# Lowered 40_000 -> 18_000, and made env-overridable.
#
# This is the highest-leverage knob on provider quota, because the turn-1
# dump is re-sent on EVERY turn: input is charged per turn, output once.
# Cerebras enforces a 250,000 tokens-per-minute ceiling at the ORGANIZATION
# level, shared by every deployment of this engine, and it reserves
# `input + max_completion_tokens` BEFORE running a request. At a 33k-54k
# prompt that admitted only ~2-3 concurrent searches before
# 429 token_quota_exceeded, which is what made concurrent search fail while
# a single search served fine.
#
# 18_000 is a payload cap, NOT a recall cap. The candidate pool is set by
# SEARCH_AGENT_VECTOR_TOP_K / _BM25_TOP_K above and is deliberately left
# alone: cutting those would lower the recall CEILING before fusion, while
# this trims what gets SENT after fusion, keeping the highest-RRF docs. The
# funnel has room -- measured in production, ~280 candidates are rendered to
# return 10-16 results.
#
# Env-overridable because it is the first dial to reach for if recall
# regresses: `kubectl set env DEPLOY SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET=40000`
# restores the old behaviour without a release.
SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET = int(
    os.getenv("SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET", "18000")
)

# The model's hard context window. Providers admit a request on
# `input + max_completion_tokens`, NOT on input alone, so this is the number
# both halves below have to share.
SEARCH_AGENT_MODEL_CONTEXT_WINDOW = int(
    os.getenv("SEARCH_AGENT_MODEL_CONTEXT_WINDOW", "131072")
)

# Cap on one gatherer completion. Sending NO cap is what broke this: the
# provider then books the MODEL MAXIMUM (~40k) as the reservation, and that
# booking is charged twice over --
#
#   * against the context window. On 2026-08-03 a request whose input the
#     115_000 backstop had already bounded was still rejected by Fireworks
#     with `prompt is too long: 143250 tokens exceeds maximum context length
#     of 131071`. The backstop bounded the INPUT; the reservation rode on top
#     and cleared the ceiling. Nothing in the loop had ever measured the sum.
#   * against provider quota. Cerebras reserves `input + max_completion_tokens`
#     BEFORE running a request, against a 250,000 TPM ceiling shared org-wide
#     (see SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET's comment). An uncapped 40k
#     booking per request is quota burned on output that never arrives, and it
#     is the half of the 429s that raising the quota is NOT needed to fix.
#
# MEASURED, not guessed -- which is what `completion_tokens_per_turn` and
# `finish_reasons_per_turn` on LoopState were added to make possible. Live
# retrieval pods, 3h window: min 145, p50 1,635, p90/max 4,698 completion
# tokens. 16_000 is ~3.4x the observed max, the same headroom multiple
# SEARCH_AGENT_EXTRACTOR_MAX_TOKENS settled on after a too-tight cap there
# silently truncated JSON mid-string.
#
# Remember this bounds reasoning + visible content TOGETHER on the OpenAI wire
# shape (gpt-oss reasoning lands inside `usage.completion_tokens`), so it is
# not a JSON-size budget. A turn that hits it returns
# `finish_reason="length"`, which `_run_turn` now reports as a degradation
# rather than letting a half-emitted answer read as a healthy one.
#
# Env-overridable: the right value is a property of the MODEL, and
# SEARCH_AGENT_INFERENCE_MODEL is itself env-overridable. The sample above is
# small (n=10, light traffic), so retune with
# `kubectl set env DEPLOY SEARCH_AGENT_MAX_OUTPUT_TOKENS=24000` if
# `finish_reason="length"` starts appearing in agent.turn_complete.
SEARCH_AGENT_MAX_OUTPUT_TOKENS = int(
    os.getenv("SEARCH_AGENT_MAX_OUTPUT_TOKENS", "16000")
)

# Slack between our token estimate and the provider's real count. We measure
# with cl100k while gpt-oss tokenizes differently, and `_enforce_context_budget`
# counts `messages` only -- the tool schemas ride along on every turn too. This
# absorbs both. Raise it before raising the window if 400s reappear.
SEARCH_AGENT_CONTEXT_SAFETY_MARGIN = int(
    os.getenv("SEARCH_AGENT_CONTEXT_SAFETY_MARGIN", "10000")
)

# DERIVED, never hardcoded. The old flat 115_000 encoded an input budget that
# silently assumed a zero-token completion; whenever the reservation or the
# window moved, the invariant broke with no compiler or test to notice. Stating
# it as subtraction makes "input + output + slack fits the window" true by
# construction -- change any term and the others still add up.
SEARCH_AGENT_MAX_CONTEXT_TOKENS = (
    SEARCH_AGENT_MODEL_CONTEXT_WINDOW
    - SEARCH_AGENT_MAX_OUTPUT_TOKENS
    - SEARCH_AGENT_CONTEXT_SAFETY_MARGIN
)

# FAIL FAST on a config that cannot work. All three terms above are env-
# overridable so an operator can retune without a release, which also means a
# typo is a realistic path: `SEARCH_AGENT_MAX_OUTPUT_TOKENS=200000` yields a
# budget of -78928. Nothing downstream would raise -- `_enforce_context_budget`
# would just evict every tool result on every turn and send anyway, so retrieval
# quality collapses silently and looks like a model regression. A pod that
# refuses to start naming the three knobs is far cheaper to diagnose.
#
# The floor is the prefanout budget: below that the turn-1 evidence dump alone
# cannot fit, so the agent has nothing to reason over and the config is a
# mistake however it was reached.
if SEARCH_AGENT_MAX_CONTEXT_TOKENS < SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET:
    raise ValueError(
        "search-agent token budget is unusable: "
        f"window({SEARCH_AGENT_MODEL_CONTEXT_WINDOW}) "
        f"- output({SEARCH_AGENT_MAX_OUTPUT_TOKENS}) "
        f"- margin({SEARCH_AGENT_CONTEXT_SAFETY_MARGIN}) "
        f"= {SEARCH_AGENT_MAX_CONTEXT_TOKENS}, which is below the prefanout "
        f"budget ({SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET}). Lower "
        "SEARCH_AGENT_MAX_OUTPUT_TOKENS or SEARCH_AGENT_CONTEXT_SAFETY_MARGIN, "
        "or raise SEARCH_AGENT_MODEL_CONTEXT_WINDOW to match the model."
    )

# fetch_chunk_window: neighbors returned on each side of a matched chunk.
# The matched chunk is already surfaced by the pre-fan-out, so this pulls
# just enough adjacent context to repair fixed-window chunk fragmentation
# without hauling the whole doc. Total chunks returned = 2*N + 1.
SEARCH_AGENT_CHUNK_WINDOW_DEFAULT = 1
SEARCH_AGENT_CHUNK_WINDOW_MAX = 5

# Per-channel result byte cap. Node properties / chunk content trimmed to
# this when assembled into a tool return. Keeps the per-turn evidence pack
# around 15K tokens.
SEARCH_AGENT_PER_HIT_PROPERTIES_CAP = 2048

# Cerebras prefix-cache discount only fires when consecutive turns —
# AND consecutive queries from the same customer — land on the same
# replica. We set `x-session-affinity` per customer (not per query) so
# the static prefix (system prompt + tool defs) cache-hits across queries,
# and multi-turn cache continuity is preserved because Cerebras's prefix
# cache is content-addressed (turn 1 still hits the warm prefix turn 0
# wrote to the same replica). This is the acceptance gate observed via
# query_traces.cache_hit_rate; if production rate drops below this,
# hard-query cost roughly doubles. See `loop._affinity_key`.
SEARCH_AGENT_CACHE_HIT_RATE_FLOOR = 0.7

# Wall-clock cap on entity extraction. Extraction is optional enrichment and
# retains the original 30s bound; failure falls back to grounding-only anchors.
#
# 6s, cut from 30s (2026-08-14). This 30 has never been the deadline that fired:
# extraction goes through the same gateway route as the gatherer, because all
# three Cerebras callers send SEARCH_AGENT_INFERENCE_MODEL verbatim and the
# proxy has ONE `cerebras/*` entry, so research-os's
# `litellm.searchAgentTimeout` -- 6s -- has been the real bound all along.
#
# It has to be stated here now because that proxy rung is being raised to 14s to
# stop it cutting healthy GATHERER turns (see
# SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS below). One rung, three callers: without
# this line, widening it for the gatherer would silently hand extraction a 14s
# hang budget too, and extraction runs INSIDE the stage cap during setup, where
# 8 extra seconds come straight out of the turns. 6.0 keeps the bound it
# actually has today, now enforced where the callers can differ.
#
# Measured extraction_ms is p50 ~1.0s / p90 ~1.6s, so 6s stays far above healthy
# traffic. The auto-merge judge is the third caller and is deliberately left on
# the proxy rung: it runs on the ingestion path, not inside a caller's search
# budget, so a longer hang there costs throughput rather than a degraded search.
SEARCH_AGENT_EXTRACTOR_TIMEOUT_SECONDS = 6.0

# Token cap on one entity-extraction completion. This is NOT a JSON-size
# budget: SEARCH_AGENT_INFERENCE_MODEL is a reasoning model, and on the
# OpenAI wire shape `max_tokens` bounds reasoning + visible content
# TOGETHER (reasoning lands in `usage.completion_tokens`, not beside it).
# Extraction reasons ~440-560 tokens before emitting a single character,
# so the old 600 left ~45 tokens for the JSON body -- the extractor
# returned `finish_reason="length"` with the object cut mid-string inside
# `sub_queries`, json.loads raised, and the caller silently got an empty
# EntityExtraction() while the response still read state:"ok".
#
# Measured against cerebras/gpt-oss-120b on the probe tenant with a fully
# populated 22-candidate grounding bundle: reasoning 555 + content 52 =
# 607 completion tokens. The old cap missed by SEVEN tokens, which is why
# it reproduced 100% of the time on that query rather than intermittently.
# 2000 leaves ~3x headroom for a longer bundle without letting a runaway
# reasoning trace burn the extractor's 30s budget.
#
# Env-overridable because the right value is a property of the MODEL, and
# SEARCH_AGENT_INFERENCE_MODEL is itself env-overridable: point the extractor
# at a heavier reasoner and this cap has to move with it. Retune via
# `kubectl set env DEPLOY SEARCH_AGENT_EXTRACTOR_MAX_TOKENS=4000` without a
# deploy -- the failure mode it guards against is silent (empty extraction,
# response still state:"ok"), so waiting on a release to correct it is the
# expensive option.
SEARCH_AGENT_EXTRACTOR_MAX_TOKENS = int(
    os.getenv("SEARCH_AGENT_EXTRACTOR_MAX_TOKENS", "2000")
)

# Client-side cap on one gatherer model turn. Nested inside
# SEARCH_AGENT_LOOP_TIMEOUT_SECONDS, which bounds the complete loop via
# asyncio.wait_for and is the hard backstop.
#
# This number, not the gateway's, is what ends a stalled turn. MEASURED: a
# stalled turn runs to exactly this deadline (repeated `litellm.Timeout ...
# time taken=<deadline>.01` with `x-litellm-attempted-fallbacks: 0`), never to
# the 12s Cerebras deployment deadline, so the Cerebras -> Fireworks failover
# the old 60s value was sized for does not actually fire. Treat this purely as
# "how long are we willing to wait", not as failover headroom.
#
# The stall is reproducible and specific: large context COMBINED WITH
# tool-calling. A 300KB prompt alone returns in ~1.5s; the same prompt with
# `tools` + tool_choice="required" was measured at 61.6s. Neither ingredient
# alone does it, which is why isolated probes look healthy and only real
# gatherer turns hang.
#
# 5s, CUT FROM 70s (2026-08-04), and the strategy changed with it: this is no
# longer "how long are we willing to wait for Cerebras to unstick", it is "how
# long before we give up on Cerebras and finish the run on Fireworks".
#
# Waiting was the wrong trade. Measured over 105 production turns:
#
#     healthy turns (n=92, 87.6%)   mean 971ms   p50 812ms   p90 1.6s   max 3.9s
#     stalled turns (n=13, 12.4%)   59.5s - 63.8s, NOTHING in between
#
# The distribution is bimodal with a 55-second empty gap, so a deadline
# anywhere in 4s..59s separates the two modes perfectly. 5s sits above the
# healthy max (3.9s) with headroom and cuts every stall. It cannot truncate
# healthy traffic: a turn slower than 5s has never been observed to then
# succeed quickly -- it goes to ~60s.
#
# Why waiting out the stall did not work. The 70s value assumed the stalled
# turn eventually returns something useful, and it does (finish_reason
# tool_calls at ~60s). But turns are per-RETRIEVAL dice rolls: at a mean 2.23
# turns/retrieval, a 12.4% per-turn stall rate compounds to ~30% of retrievals
# degrading. And research-os hangs up at 30s (ENGINE_TIMEOUT_SECONDS), so for
# that caller a 60s success is indistinguishable from a failure -- it had
# already returned an empty result set.
#
# The failover this enables is OURS, not the gateway's. The proxy's
# Cerebras -> Fireworks route fallback provably never fires: stalled turns end
# at exactly this client deadline carrying `x-litellm-attempted-fallbacks: 0`,
# never at the gateway's own 12s deployment deadline (see the block above).
# So the loop owns it: on the first stall it flips to
# SEARCH_AGENT_FALLBACK_INFERENCE_MODEL and STAYS there for the rest of the
# run -- see `_run_turn`. Retrying Cerebras every turn would re-pay this 5s on
# each one.
#
# Caller budgets this now fits inside:
#   * research-os /v1/search: 30s total. Worst case is now ~4s deterministic
#     + one 5s cut + Fireworks turns, which fits. It did not before.
#   * MCP search_knowledge: 180s. Unaffected; it just stops spending 60s of
#     that on a stalled provider.
#
# 12s, RAISED FROM 5s (2026-08-14). Everything above is still the right
# reasoning; its INPUT expired. That analysis rested on a bimodal distribution
# with a 55-second empty gap -- healthy max 3.9s, stalls at 59.5-63.8s, nothing
# between -- which made "a turn slower than 5s never then succeeds" true, and a
# 5s cut free. Re-measured over 54 successful production turns:
#
#     healthy turns   p50 2.17s   p90 4.66s   p95 6.69s   max 10.6s
#     stalls at ~60s: NONE observed
#
# The gap is gone and the healthy tail now runs past 10s, so 5s sits at roughly
# p90 of HEALTHY traffic and cuts about one turn in ten that would have
# returned. 12s clears the observed p100 with headroom.
#
# WHY THIS MATTERS MORE THAN THE 7s IT COSTS. Cutting the primary fails the run
# over, and the failover is sticky, and the new provider has no cached prefix.
# Measured on the same conversations: turns BEFORE a failover run p50 926ms at
# mean cache_hit 0.56; turns AFTER it run p50 4.80s, max 10.6s, and 5 of 12
# failovers then died on the gateway's 12s Fireworks deadline. So a cut
# triggered by one slow turn makes every remaining turn ~5x slower, which is
# what pushed ~11-14% of /v1/search past its 30s budget. The failover itself is
# fine -- it rescued 7 of 12 -- it just must not fire on healthy traffic.
#
# NOT the 10s that was tried before and reverted: charts/research-os
# values.yaml records a 10s/10s experiment that clipped healthy turns, and the
# 10.6s max above is exactly why. This raises ONE rung; the Fireworks rung
# stays where it is.
#
# The proxy's own cerebras/* deployment timeout must stay ABOVE this or it cuts
# first and this value never binds -- research-os charts/research-os/values.yaml
# `litellm.searchAgentTimeout`, moved to 14 in the same change.
SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS = 12.0

# Per-turn deadline once the run has failed over.
#
# This was 12.0, on two claims that production contradicts.
#
# 1. "Mirrors the gateway's own Fireworks route timeout (12s)." It does not.
#    The deployed proxy config gives `accounts/fireworks/*` a timeout of 40 and
#    `cerebras/*` a timeout of 12 -- 12 is the CEREBRAS deployment deadline, so
#    the mirror was taken off the wrong route and held the fallback to a
#    deadline 28s tighter than the gateway itself allows.
# 2. "Fireworks ... does not exhibit the large-context + tool-calling stall."
#    It does. The stall is driven by the ~300-chunk pre-fan-out prompt with
#    tool_choice=required, not by the provider, so the hop lands on a model
#    that stalls the same way.
#
# The 5s primary cut is safe because Cerebras is bimodal -- healthy maxes at
# 3.9s, stalls start at 59.5s, and nothing lives in the 55s gap. Fireworks has
# no such gap, so the same reasoning does NOT transfer. Measured over 16h on
# the managed data plane (111 failovers): successful fallback turns ran a
# smooth continuum from 451ms to 11904ms, and 72 of 76 failures landed within
# 100ms of exactly 12000ms. That is a clipped distribution -- the deadline was
# truncating turns that were going to succeed, which is the exact failure the
# old comment claimed to be avoiding by not reusing the 5s value.
#
# 30, not the gateway's 40: SEARCH_AGENT_LOOP_TIMEOUT_SECONDS caps the whole
# stage at 60s with setup subtracted, and measured setup (grounding +
# extraction + pre-fan-out) is p50 32s, leaving ~28s for all turns at the
# median. 40 is unreachable there, so it would be a number that never binds.
# This is a CEILING, not a guarantee: the stage deadline usually fires first
# and degrades through the normal loop_timeout path. It is still strictly
# better than 12, which cut the turn while ~16s of stage budget sat unused.
# Capping the pre-fan-out is what would make this deadline fully reachable.
#
# 12s, cut back from 30s (2026-08-14), because 30 was never the deadline that
# fired. This is a CLIENT deadline, and the gateway enforces its own on the
# same call: `accounts/fireworks/*` carries `timeout: 12` in research-os
# charts/research-os/values.yaml (`litellm.searchAgentFallbackTimeout`). Every
# observed fallback failure in production is that one, not this:
#
#   litellm.Timeout: ... Fireworks_aiException ... Timeout passed=12.0,
#   time taken=12.003 seconds        (n=9, all within 51ms of 12.000)
#
# So raising this to 30 bought nothing and, worse, made the ladder arithmetic
# below reason about a number the system cannot produce -- the stage cap was
# sized against 30s fallback turns that always ended at 12. Setting it to what
# actually binds keeps `test_stage_cap_leaves_room_for_the_failover_to_land`
# honest instead of accidentally true.
#
# If the fallback should really get 30s, raise the GATEWAY value; this one
# follows it, never leads. And check the caller first: research-os abandons at
# 30s total, so a 30s fallback turn cannot land there whatever this says.
SEARCH_AGENT_FALLBACK_TIMEOUT_SECONDS = 12.0

# Where a stalled run finishes. Must be a model id the gateway's modelList
# resolves -- `accounts/fireworks/*` expands to the upstream
# `fireworks_ai/accounts/fireworks/...` route. Empty string disables failover
# entirely (self-hosted installs with no second provider), in which case a
# stalled turn just raises as it did before.
SEARCH_AGENT_FALLBACK_INFERENCE_MODEL = os.getenv(
    "SEARCH_AGENT_FALLBACK_INFERENCE_MODEL",
    "accounts/fireworks/models/gpt-oss-120b",
)

# Overall agent loop cap, and a HARD ceiling on the whole gatherer stage:
# `run_gatherer` converts this into a deadline measured from its own entry, so
# grounding + extraction + pre-fan-out (~4s) come OUT of this budget rather
# than being added to it. Whatever happens, the stage returns within this many
# seconds; timeout degrades to the citable pre-fan-out evidence the harness
# already retrieved (`_backfill_recall_floor` off `state.prefanout`), which is
# a real result set, not an error.
#
# 60s, cut from 90s (2026-08-04). With a 5s primary cut and a 12s fallback
# deadline, a 3-turn run lands ~40s worst case, so 60s is the backstop for
# pathological cases rather than a routine ceiling.
#
# 25s, cut from 60s (2026-08-14), because 60 was never reachable by the caller
# that matters. research-os /v1/search abandons at ENGINE_TIMEOUT_SECONDS = 30,
# so a stage permitted to run 60s spends up to half its budget producing an
# answer nobody is still waiting for -- and the caller reports state:"partial"
# while this pod is still working. A backstop above the caller's own deadline
# is not a backstop, it is a leak.
#
# 25 not 30: the deadline is measured from `run_gatherer` entry, so setup
# (grounding + extraction + pre-fan-out) comes OUT of it, not on top. Setup
# measures p50 6.5s / p90 10.7s in production -- the "~4s" above is stale --
# leaving ~14s of turns at p90 setup, which fits one full 12s primary cut plus
# the degrade to pre-fan-out. The 5s of slack under 30 covers response
# serialisation and the hop back to research-os.
#
# Timing out here is not an error: it degrades to `_backfill_recall_floor` off
# `state.prefanout`, which is a real citable result set. Returning that at 25s
# beats returning nothing at 30.
SEARCH_AGENT_LOOP_TIMEOUT_SECONDS = 25.0

# Fraction of gatherer runs whose full per-turn transcript gets persisted
# to R2 alongside the query_traces summary row. 1.0 = persist every run.
# Drop via `kubectl set env DEPLOY SEARCH_AGENT_TRACE_SAMPLE_RATE=0.1`
# without a deploy if R2 spend spikes. Sampled-out rows still get the
# summary in `query_traces`; only the full blob is skipped.
SEARCH_AGENT_TRACE_SAMPLE_RATE = float(
    os.getenv("SEARCH_AGENT_TRACE_SAMPLE_RATE", "1.0")
)

# ---- Provider token-rate limiting (shared.llm) -------------------------------
# Per-PROCESS ceiling on tokens sent to the provider per minute, enforced in
# shared.llm around every litellm call. See _acquire_token_budget there.
#
# 0 = DISABLED, and that is the default on purpose. A self-host install with
# its own provider account has no shared quota to protect, and enabling this
# without a measured budget throttles for no reason.
#
# Set it where a quota IS shared. Cerebras enforces 250,000 TPM at the
# ORGANIZATION level, and both the research and managed data planes draw on
# that one budget with separate API keys under the same org. There is no
# shared store between them (separate Postgres, no Redis), so the budget is
# SPLIT STATICALLY rather than coordinated: give each process its share and
# keep the sum under the org limit. With N processes across both clusters,
# start near (org_limit / N) and lower it if 429s persist.
#
# This is a burst smoother, not an accountant. The cost estimate is chars/4
# plus max_tokens, so it tracks the provider's own pre-admission reservation
# without trying to match it exactly.
LLM_TPM_BUDGET = int(os.getenv("LLM_TPM_BUDGET", "0"))

# How long a call may wait for budget before proceeding anyway.
#
# The limiter FAILS OPEN. Waiting longer than the caller's own deadline turns
# a fast 429 (which degrades to pre-fan-out evidence) into a slow timeout
# (which returns nothing) -- strictly worse for the user. The gatherer's turn
# deadline is SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS (70s), but the binding
# constraint is research-os abandoning /v1/search at 30s, so this stays well
# inside the tighter of the two.
LLM_TPM_MAX_WAIT_SECONDS = float(os.getenv("LLM_TPM_MAX_WAIT_SECONDS", "5.0"))

# Prefix used in `integration_tokens.scope` to signal the row represents a
# GitHub App installation rather than an OAuth access_token. The installation
# id follows the colon; tokens are minted on demand from the App private key.
GITHUB_INSTALLATION_SCOPE_PREFIX = "installation:"

# Granola: API tier prefix in integration_tokens.scope. Personal keys see only
# the issuing user's notes + shared. Enterprise keys see the whole workspace.
GRANOLA_SCOPE_PERSONAL = "tier:personal"
GRANOLA_SCOPE_ENTERPRISE = "tier:enterprise"

# pg_notify channel the worker LISTENs on for sub-second manual-refresh wake.
# The /admin/.../granola/refresh endpoint NOTIFYs after enqueuing so
# BackfillWorker doesn't wait for its 5s poll cycle.
GRANOLA_REFRESH_CHANNEL = "granola_refresh"

# Steady-state polling cadence: re-enqueue Granola backfills this often once
# the initial sync is complete. Read by services/ingestion/poller.
GRANOLA_POLL_INTERVAL_SECONDS = 300

# Per-customer rate-budget for outbound calls to the Granola API.
# Granola docs: 5 rps / 25 in 5s burst. We sleep this long between calls inside
# the connector's backfill loop, leaving 20% headroom under the documented limit.
GRANOLA_REQUEST_INTERVAL_SECONDS = 0.25

# Manual-refresh debounce. Repeated /refresh hits within this window collapse
# into a single enqueue + notify; the second hit returns 429 with Retry-After.
GRANOLA_REFRESH_DEBOUNCE_SECONDS = 30


# pg_notify channels for the wiki synthesis pipeline.
#
# Pre-redesign: a single `wiki_synthesize` channel — Normalizer._persist
# fired NOTIFY on every webhook, the in-process cron drained immediately,
# resulting in continuous daytime synthesis. That model didn't match the
# wiki's actual scope (slow-moving company knowledge). The redesign:
#
# - Normalizer._persist NO LONGER fires NOTIFY. Queue rows accumulate
#   silently during the day at status='pending'.
# - The wiki-cron fly app fires NOTIFY on `wiki_synthesize_pending`
#   nightly at 02:00 UTC (per opted-in customer with pending rows). The
#   /api/wiki/synthesize/trigger endpoint also fires it for manual wakes
#   from the dashboard "Generate Wiki Now" button.
# - The wiki-worker (triage) app LISTENs on `wiki_synthesize_pending` →
#   drains pending rows through triage → marks them triaged/rejected/
#   verifier_rejected → fires NOTIFY on `wiki_synthesize_triaged` from
#   the same transaction that committed the UPDATE (Postgres delivers
#   NOTIFY only after COMMIT, so listeners never wake on un-visible rows).
# - The wiki-synthesis app LISTENs on `wiki_synthesize_triaged` → drains
#   triaged rows through verifier + synthesize → writes wiki pages →
#   regenerates the index.
WIKI_PENDING_CHANNEL = "wiki_synthesize_pending"
WIKI_TRIAGED_CHANNEL = "wiki_synthesize_triaged"

# Backfill pipeline wake channel — fired by the
# /api/wiki/backfill/trigger route after it inserts pending rows. Empty
# payload; the BackfillWorker treats this as a "drain pending rows now"
# wake hint and claims rows via FOR UPDATE SKIP LOCKED. Distinct from
# WIKI_PENDING_CHANNEL because the daily-replay path operates on the v4
# queue, while backfill reads from source APIs.
WIKI_BACKFILL_CHANNEL = "wiki_backfill_pending"

# Backfill cancel channel — fired by the trigger route's force-cancel
# path. Payload is a JSON object ``{customer_id, run_ids: [int]}``;
# every BackfillWorker LISTENing on this channel cancels in-flight
# tasks whose run_id matches. Coarse 10s drain window — see
# BACKFILL_CANCEL_DRAIN_TIMEOUT_SECONDS.
WIKI_BACKFILL_CANCEL_CHANNEL = "wiki_backfill_cancel"

# Cooperative drain window the trigger route waits after firing the
# cancel NOTIFY before proceeding to wipe + insert new pending rows.
# Sized larger than the worker's per-tick cadence but small enough that
# admin-initiated force-restart still feels interactive in the dashboard.
BACKFILL_CANCEL_DRAIN_TIMEOUT_SECONDS = 10.0

# Per-machine cap on concurrent backfill crawler agents. Read at boot
# from the BACKFILL_PARALLELISM env var by ``BackfillWorker``; the
# constant here is the default. Sized at 6 against the 4 GB / 2 vCPU
# fly machine envelope (idle ~150 MB, ~150-250 MB per active crawler ->
# ~1.5 GB peak crawler load + headroom). Tune via env, not code.
BACKFILL_PARALLELISM = 6

# How many wiki_synthesis_queue rows the cron claims per drain tick. Triage is
# token-budget batched on top of this; this is just the upper bound on rows
# pulled into memory at once.
WIKI_SYNTHESIS_CLAIM_BATCH = 200

# Token budget per Haiku triage call, expressed in *estimated Anthropic
# tokens* (post-multiplier — see `kb.synthesis.triage`). Rows are
# packed greedily until this ceiling is hit, then the batch fires. The
# packer adds prompt + tool-schema + per-event framing overhead on top
# of body tokens before comparing to this budget, so it represents the
# user-content slice of the wire request, not the full request size.
#
# Headroom: Anthropic Haiku's hard context limit is 200K tokens. We
# budget 150K for content; the remaining 50K is left as margin for
# (1) prompt + tool-schema + envelope (~2K), (2) tokenizer drift between
# our cl100k estimate and Anthropic's true tokenizer, and (3) the model's
# own response. Production drains were DLQ'ing entire batches at the
# previous 120K budget because the packer counted only raw body text in
# cl100k and Anthropic's tokenizer + request envelope pushed the wire
# count past 200K (e.g. batch_size=66 produced 208K Anthropic tokens).
WIKI_TRIAGE_TOKEN_BUDGET = 150_000

# Output-side budget for the triage Anthropic call.
#
# Haiku 4.5's `max_tokens` ceiling is 8192. We set 8000 to leave a 192-
# token cushion against SDK-version drift / per-conversation token
# bookkeeping. The original 4096 was way too low: a batch of ~28+ events
# would produce more verdicts than fit in 4096 output tokens, so Haiku
# would stop at max_tokens with NO tool_use block — causing a Pydantic
# crash on the missing `verdicts` field and DLQ-ing the whole batch.
WIKI_TRIAGE_MAX_OUTPUT_TOKENS = 8000

# Per-verdict size estimate, in Anthropic tokens. TriageVerdict is
# {important: bool, score: float, reason: str ≤ ~100 tokens}; with the
# JSON envelope `"queue_id": {...}` and pretty-printing, a verdict lands
# around 80-120 Anthropic tokens. 150 is the conservative cap.
WIKI_TRIAGE_VERDICT_TOKENS = 150

# Output-side cap on events per batch:
#   floor(WIKI_TRIAGE_MAX_OUTPUT_TOKENS / WIKI_TRIAGE_VERDICT_TOKENS)
#   = 8000 / 150 = 53 → round down to 50 for envelope + drift margin.
# The packer enforces MIN(input-token-budget, this-event-cap) so the
# limiting factor is whichever binds first for a given batch.
WIKI_TRIAGE_MAX_EVENTS_PER_BATCH = 50

# Importance threshold for triage to keep an event. Below this score the row
# is marked 'rejected' and never reaches synthesis. Raised from 5.0 → 7.0
# to align triage with the wiki's actual scope (slow-moving company
# knowledge, not a per-event log). Step down stepwise (7.0 → 6.0 → 5.0)
# if the wiki is under-populated; step up if it gets spammy.
WIKI_TRIAGE_SCORE_THRESHOLD = 7.0

# Per-row attempt cap before a queue row is parked in 'failed'.
WIKI_SYNTHESIS_MAX_ATTEMPTS = 3

# NO PERIODIC WAKE. The wiki drains on NOTIFY only.
#
# This was 1800 (30 min), described as "a safety net if a notify is missed
# during a connection drop". It was not a safety net -- it was the scheduler,
# and it silently replaced the design. Only TWO things ever notify the wiki
# channels: the nightly cron (kb/synthesis/nightly_trigger) and the manual
# "generate now" route. Ingestion does not. So the timer was never catching a
# missed notify from a busy pipeline; it was finding rows ingestion had left
# pending and draining them on its own schedule.
#
# The cost of that, measured on 2026-08-12 before this changed: 32 synthesis
# runs and 25 timer-driven triage runs in 24 hours, against ZERO nightly ones,
# rewriting 16 pages 72 times. Every drain is a multi-turn Gemini Pro agent
# loop, and the wiki is explicitly slow-moving knowledge -- see
# WIKI_TRIAGE_SCORE_THRESHOLD above, raised to 7.0 for exactly that reason. It
# also made `updated_at` meaningless as "when did this knowledge change" and
# left the page histories full of rewrites of unchanged material.
#
# WHAT THE RISK ACTUALLY IS, stated so it can be re-decided: if a NOTIFY is
# genuinely lost -- a connection drop at the moment the 02:00 cron fires --
# that tenant waits until the next night. Two senders, one of them a cron at a
# fixed time whose failure is visible in its own run history, is a small enough
# surface that a 30-minute poll is the wrong shape of insurance. Set this to a
# number of seconds if that judgement turns out to be wrong; the workers still
# read it.
WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS: float | None = None

# Provider knob for the triage stage. v4 uses the wiki agent (Gemini
# Pro) for synthesis, so the synthesis + verifier provider knobs are
# gone. Triage is provider-pluggable: flip the value and redeploy.
# Recognized values:
#   "haiku" | "claude-haiku"            -> Anthropic Haiku 4.5
#   "gemini-flash-lite" | "gemini-3.1-flash-lite" -> Gemini 3.1 Flash Lite
#   "gemini-3.5-flash"                  -> Gemini 3.5 Flash (default; 2026-05-19)
# No env-var override path — the prior `getattr(settings, ...)` plumbing
# referenced fields that didn't exist on Settings, so the env var was
# silently inert. Constants-only is honest.
#
# Default flipped 2026-05-19 from "haiku" to "gemini-3.5-flash" after the
# A/B sweep in scripts/eval_3_5_flash_sweep.py (report:
# ~/.gstack/projects/prbe-knowledge/eval-3-5-flash-sweep-20260520T025718Z.md).
# 20 fixtures x 2 trials per model. Label accuracy: both 100%. Opus-judged
# quality: 9.3 (haiku) vs 9.4 (3.5-flash) — statistical tie. p50 latency:
# 1913ms → 1614ms (~16% faster). Cost per call: $0.00225 → $0.00060
# (~3.75x cheaper). Net: equal quality at <30% of the wire cost on a
# high-volume hot path.
WIKI_TRIAGE_MODEL = "gemini-3.5-flash"

# Directed-phrase generation runs once per wiki page during synthesis to
# emit 5-10 trigger phrases that boost retrieval ranking when an engineer's
# symptom-style query semantically matches them. The 2026-05-09 model
# shootout (scripts/eval_directed_phrases.py, judged by Opus 4.7) picked
# Gemini 3 Flash: specificity 8.6/retrieval-fit 8.2 vs Haiku 7.8/7.8, at
# ~1/4 the cost ($0.0005 vs $0.0022 per call). Flip to "haiku" or
# "gemini-3.1-flash-lite" via this constant + redeploy.
DIRECTED_PHRASES_MODEL = "gemini-3-flash-preview"

# Concurrency caps. The wiki-worker fans out customers, then triage
# batches per customer. (The v4 wiki agent uses
# WIKI_AGENT_GLOBAL_CONCURRENCY for synthesis-side fan-out plus a
# per-customer advisory lock; it doesn't cluster events anymore.)
WIKI_SYNTHESIS_CUSTOMER_CONCURRENCY = 4
WIKI_TRIAGE_BATCH_CONCURRENCY = 8

# Manual trigger rate limit (advisory-lock + lookback in the BFF). The
# /api/wiki/synthesize/trigger endpoint here surfaces the same value so
# the dashboard can render an accurate "try again in Xs" toast.
WIKI_TRIGGER_RATE_LIMIT_SECONDS = 300

# Hour-of-day (UTC) the wiki-cron fly machine fires its nightly NOTIFY.
# 02:00 UTC = 18:00 PT / 21:00 ET — picked so the drain finishes before
# the team's morning standup but doesn't compete with the rest of the
# nightly pipeline (Granola steady-poll cycles, etc.).
WIKI_NIGHTLY_HOUR_UTC = 2

# Source systems that never feed wiki synthesis. WIKI because the cron's
# own COMPILED_WIKI writes must not feed back into its own queue;
# CODE_GRAPH because code.symbol docs are deterministic AST extractions
# whose body is a function signature + docstring — triage would burn LLM
# tokens extracting Decisions/Runbooks from them and produce nothing.
# The Normalizer enqueue gate and every queue-seeding path
# (kb.synthesis.persistence.seed_missing_docs) consume THIS tuple; the
# two lists drifted once (the catchup script excluded only WIKI and
# enqueued thousands of code_graph rows) — do not re-list sources inline.
WIKI_ENQUEUE_EXCLUDED_SOURCES: tuple[SourceSystem, ...] = (
    SourceSystem.WIKI,
    SourceSystem.CODE_GRAPH,
)

# Per-statement ceiling for the nightly queue-seed reconcile. The seed is
# a batched INSERT…SELECT anti-join per enabled tenant; on a healthy index
# a batch runs in seconds, so a two-minute cap only exists to stop a
# pathological plan from occupying the nightly window.
WIKI_RECONCILE_STATEMENT_TIMEOUT_MS = 120_000

# Rows per reconcile seed batch. Each batch is its own statement + txn
# under the timeout above, so a huge backlog makes durable progress
# instead of failing all-or-nothing. 20k queue-row inserts is comfortably
# a few seconds of work against the 4-index queue table.
WIKI_RECONCILE_SEED_BATCH = 20_000

# Per-statement ceiling inside the rebuild trigger's transaction and the
# preview's counts. Deliberately UNDER the prbe-backend BFF's 20s total
# client timeout on these proxied calls (_WIKI_TIMEOUT in
# apps/data_plane/routers/dashboard/knowledge.py): a reseed that cannot
# finish inside it fails fast with the whole txn (wipe included) rolled
# back and an honest error, instead of the BFF 502ing while the engine
# commits behind its back and the retry hits a mystery 409. Tenants too
# large for this window rebuild via the catchup CLI.
WIKI_REBUILD_STATEMENT_TIMEOUT_MS = 15_000


# ---------------------------------------------------------------------------
# Wiki agent loop (v4: Gemini 3.1 Pro driving the synthesis stage)
# ---------------------------------------------------------------------------

# Hard cap on agent turns per drain. Picked at 200 to leave headroom for
# pebble's ~3000-event drains; smaller customers typically finish in
# 10-50 turns. Exceeding this cap halts the drain and DLQs the in-flight
# rows; admin reset is the recovery path.
WIKI_AGENT_TURN_CAP = 200

# Hard cap on staged page updates per drain. The wiki is supposed to
# move slowly — 30 page edits per night is generous. Exceeding this cap
# means the agent is hallucinating page mass and we'd rather DLQ than
# write 100 brand-new pages.
WIKI_AGENT_UPDATE_CAP = 30

# Stall threshold. If the agent makes no consequential tool call (no
# page update / create / skip) for this many consecutive turns, halt.
#
# Bumped from 3 to 15 after acme' run 105 stalled with 200
# events DLQ'd despite the agent making real progress on reads. The
# old "one read-page, one read-event, one think" math was wrong — a
# realistic decision flow on a chunk of 200 events looks like:
#   next_events -> list_wiki_pages -> read_page x3 -> get_event_body x2
#   -> update_page (CONSEQUENTIAL)
# That's 7 read-style turns before the first consequential one.
# Three would have halted in the middle of normal exploration. 15
# leaves margin for the agent to read 5 pages and 3 event bodies
# before deciding, with extra room for a re-read or thought-only
# turn. Stuck loops still trip eventually.
WIKI_AGENT_STALL_TURNS = 15

# Auto-compaction trigger. When estimated input tokens cross this
# fraction of Gemini 3.1 Pro's 2M context window, summarize the
# conversation history (preserving structured runtime state) before
# the next turn.
WIKI_AGENT_COMPACT_THRESHOLD = 0.60

# Number of triaged events per next_events() page. Gemini reads the
# day in batches; the agent re-calls next_events() until drain_complete.
WIKI_AGENT_BATCH_SIZE = 200

# Maximum number of customer drains running simultaneously per
# wiki-synthesis machine. Higher than per-customer concurrency (1
# under advisory lock) so two small customers can drain in parallel
# while pebble holds its own machine. Env-overridable because adding
# replicas is a no-op (per-customer advisory locks make extra pods log
# drain_skip_concurrent) — cross-tenant fan-out on the ONE machine is
# the only knob that raises throughput. Within-tenant drains stay
# serial by design (rows claimed in source_ts order).
def _env_positive_int(name: str, default: int) -> int:
    """Parse a positive-int env override; malformed or non-positive values
    fall back to the default instead of crashing every importing service
    at boot (this module is imported by all of them)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


WIKI_AGENT_GLOBAL_CONCURRENCY = _env_positive_int(
    "WIKI_AGENT_GLOBAL_CONCURRENCY", 2
)

# Gemini model used by the wiki agent loop. Triage stays Flash Lite;
# only the agent uses Pro because per-cluster reasoning + cross-event
# pattern recognition need the bigger model.
WIKI_AGENT_MODEL = "gemini-3.1-pro-preview"

# Compactor model. Cheaper Flash variant since it only summarizes the
# conversation; preserves the structured runtime state untouched.
WIKI_AGENT_COMPACTOR_MODEL = "gemini-3.1-flash-lite"

# Per-source backfill crawler models. Default to the same Pro model the
# daily-replay agent uses; per-source knobs let us swap a cheaper /
# bigger model for one source without redeploying the rest. Mentioned
# under "Per-source models" in docs/wiki-backfill-plan.md.
WIKI_BACKFILL_MODEL_GITHUB = "gemini-3.1-pro-preview"

# Stop-walking heuristic for backfill crawlers. After this many
# consecutive source items that don't cause the agent to call
# update_page / create_page, the crawler treats the repo as drained and
# moves on. Picked at 50 to match the system prompt's stopping rule.
WIKI_BACKFILL_QUIET_STREAK = 50

# Time horizon (days) for GitHub PR + issue ingestion. Commits walk
# all-time per the locked plan so old structural commits ("first added
# auth middleware") still surface even when ticket history is bounded.
WIKI_BACKFILL_GITHUB_PRS_DAYS = 365


# Cap on Phase 2 fan-out per (customer, source). After Phase 1 completes,
# the orchestrator queries the source's discoverer for targets (e.g.,
# repos for GitHub) and inserts one Phase 2 row per target up to this
# cap. Above the cap, take the top-N by recent activity. At ~$0.30-0.60
# per Phase 2 agent run (Gemini Pro), 30 caps worst-case spend at
# ~$15/backfill on the largest customers we have today.
BACKFILL_MAX_TARGETS_PER_SOURCE = int(os.environ.get("BACKFILL_MAX_TARGETS_PER_SOURCE", "30"))

# Agent's CachedContent TTL. Re-create on miss; alert if hit_rate < 80%.
WIKI_AGENT_CACHE_TTL = "3600s"


# --- DB pool init backoff ---------------------------------------------------
# Connect-with-backoff knobs for shared.db.init_pool, kept here per the
# RRF_K / RRF_BREADTH_ALPHA tuning-knob convention (feedback_prbe_knowledge_
# tuning_consts) so the ceiling is one explicit number, not buried in db.py.
#
# Sizing: in the k8s data-plane, app pods only start after the migrate
# sentinel exists (prbe-data-plane-image / chart change), so Postgres is up
# and migrated by the time init_pool runs. The remaining retries cover only
# transient boot blips -- NetworkPolicy settling, DNS, pool limits, a
# credential race -- which clear in a second or two, not a minute. So the
# old 6-attempt / base-1s / x2 schedule (1+2+4+8+16 = 31s of pure sleep,
# ~90s worst case with connect timeouts) is far longer than needed.
#
# New ceiling: 4 attempts, base 0.5s, x2, capped per-attempt at 5s ->
# backoffs of 0.5 + 1 + 2 = 3.5s of sleep across 3 retries; worst case
# (connect timeout fully consumed each attempt) ~3.5s + 4 * connect_timeout.
# A real transient blip recovers in single-digit seconds; genuine outages
# still surface a readable DatabaseUnavailable rather than a silent hang.
DB_INIT_RETRY_ATTEMPTS = 4
DB_INIT_RETRY_BASE_SECONDS = 0.5
DB_INIT_RETRY_BACKOFF_CAP_SECONDS = 5.0
