"""Who may see a clause.

Tenant isolation is RLS's job and happens underneath this. This module
answers the narrower question the design's red team raised: a clause whose
evidence comes from exactly ONE human is that human's working note, not the
team's practice, and publishing it to colleagues is how a private habit
becomes false policy. It unlocks when a second DISTINCT human's evidence
appears -- distinctness, not row count, since one person logging three times
is still one person.
"""

from __future__ import annotations

import asyncpg

#: The source classes that represent a HUMAN saying something. The guarantee is
#: "until a second HUMAN appears" -- counting every author_ref would let two
#: agent transcripts unlock a clause no second person ever endorsed, which is
#: precisely the self-reinforcement loop the design exists to prevent.
#: The predicate also excludes `exposure_tainted` evidence: an agent that was
#: SERVED this clause and then echoed it back is not an independent second
#: voice, and letting it unlock the clause is the same loop by another route.
HUMAN_SOURCE_CLASSES = (
    "declared",
    "human_doc",
    "human_message",
    "pr_review",
    "human_wiki_edit",
)

#: Applied to an aliased `clauses c`. $1 = viewer's actor ref, $2 = human classes.
VISIBILITY_PREDICATE = """
    (
        (
            SELECT COUNT(DISTINCT e.author_ref)
              FROM clause_evidence e
             WHERE e.clause_id = c.id
               AND e.author_ref IS NOT NULL
               AND e.source_class = ANY($2::text[])
               AND NOT e.exposure_tainted
        ) >= 2
        OR c.author_ref = $1
    )
"""


async def fetch_visible_clauses(conn: asyncpg.Connection, viewer_ref: str) -> list[asyncpg.Record]:
    """Every clause in the current tenant that `viewer_ref` may see."""
    return await conn.fetch(
        f"""
        SELECT c.*
          FROM clauses c
         WHERE {VISIBILITY_PREDICATE}
         ORDER BY c.created_at
        """,
        viewer_ref,
        list(HUMAN_SOURCE_CLASSES),
    )
