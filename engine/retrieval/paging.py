"""The rest of a retrieval, and the cursor that asks for it.

`/retrieve` ranks a pool of documents and returns `top_k` of them. The surplus
used to be dropped on the floor: the response carried no cursor, so "there were
more" was not a thing the API could say, and an agent that wanted them had to
re-run the whole search with a bigger number -- the full pipeline again, LLM
turn included, for results the server had already ranked.

Here the surplus is written to `retrieve_pages` and named by an opaque cursor.
Three decisions are load-bearing:

* **The page carries content, not ids.** Ids would be smaller, and they would
  also make paging a moving corpus: a document re-indexed between page 1 and
  page 2 moves in the ranking, so the reader sees one result twice and never
  sees another. A page is a SNAPSHOT of what that search found, which is the
  only thing "page 2" can honestly mean.
* **The write is synchronous.** A background write would let a fast caller ask
  for a page that does not exist yet -- a race whose only cure is a retry the
  caller has to understand. The cost is one INSERT, and only when there is a
  surplus to insert.
* **The cursor is not signed.** It names a row; the row is read under the
  tenant GUC, so another tenant's cursor selects nothing. A signature would add
  a key to rotate and protect nothing that RLS is not already protecting.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any
from uuid import UUID

from engine.shared.db import with_tenant
from engine.shared.logging import get_logger

log = get_logger(__name__)

#: How many documents past `top_k` are worth keeping. Two more pages of a
#: default search: past that the caller is better served by a narrower query
#: than by walking a ranking whose tail the fusion itself is not confident in.
MAX_PAGE_ITEMS = 100

#: How long a cursor stays good. A page is a snapshot, and a day-old snapshot
#: of a corpus that re-indexes continuously is not the thing the caller thinks
#: it is asking for -- better to refuse it and let them search again than to
#: serve a ranking the rest of the system has moved past.
PAGE_TTL = "1 day"


class CursorExpired(Exception):
    """The page is gone (or never existed). Re-search; do not silently re-run.

    Silently re-searching is what produces the duplicates and skips paging
    exists to prevent, so this is surfaced rather than papered over.
    """


def encode_cursor(page_id: UUID | str, offset: int) -> str:
    raw = json.dumps({"p": str(page_id), "o": int(offset)}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, int]:
    """Parse a cursor, or raise `CursorExpired` for anything unreadable.

    Total on purpose: a cursor is opaque to its holder, so a malformed one is
    not a caller error to be lectured about -- it is a page that cannot be
    served, which is the same outcome as an expired one.
    """
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        parsed = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        page_id, offset = str(parsed["p"]), int(parsed["o"])
        UUID(page_id)
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise CursorExpired("unreadable cursor") from exc
    if offset < 0:
        raise CursorExpired("negative offset")
    return page_id, offset


async def store_page(customer_id: str, query: str, items: list[dict[str, Any]]) -> str | None:
    """Persist the surplus and return the cursor that asks for it.

    Returns None when there is nothing to page or the write fails: a response
    without a cursor is complete-as-far-as-it-goes, which is exactly what the
    caller got before this existed. Paging is an improvement on the answer, and
    it must never be able to take the answer away.
    """
    if not items:
        return None
    kept = items[:MAX_PAGE_ITEMS]
    try:
        async with with_tenant(customer_id) as conn:
            # Trim on write. A separate cron would be one more thing to deploy,
            # monitor and forget; a page store that cleans itself every time it
            # grows cannot drift, and the DELETE is an index scan over a table
            # that holds at most a day of one tenant's searches.
            await conn.execute(
                f"DELETE FROM retrieve_pages WHERE created_at < NOW() - INTERVAL '{PAGE_TTL}'"
            )
            row = await conn.fetchrow(
                """
                INSERT INTO retrieve_pages (customer_id, query, items)
                VALUES ($1, $2, $3::jsonb)
                RETURNING page_id
                """,
                customer_id,
                query,
                json.dumps(kept, default=str),
            )
    except Exception as exc:
        log.warning("retrieve.page_store_failed", customer_id=customer_id, error=str(exc))
        return None
    return encode_cursor(row["page_id"], 0)


async def load_page(
    customer_id: str,
    cursor: str,
    limit: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """One slice of a stored page, plus the cursor for the slice after it."""
    page_id, offset = decode_cursor(cursor)
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT items FROM retrieve_pages WHERE page_id = $1",
            UUID(page_id),
        )
    if row is None:
        # Either expired, or minted by another tenant -- the GUC makes those
        # the same query and therefore the same answer, which is the point.
        raise CursorExpired("no such page")
    items = json.loads(row["items"]) if isinstance(row["items"], str) else row["items"]
    if not isinstance(items, list):
        raise CursorExpired("malformed page")
    window = items[offset : offset + max(1, limit)]
    consumed = offset + len(window)
    return window, (encode_cursor(page_id, consumed) if consumed < len(items) else None)
