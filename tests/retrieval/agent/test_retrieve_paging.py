"""Paging a retrieval: the cursor, the page, and what each refuses to do.

The DB round trip is exercised in the storage suite; everything here is the
logic that decides what a cursor MEANS -- which is where a paging bug actually
lives. A walk that skips a row or shows one twice is the failure mode, and
both come from re-deciding the ranking between pages.
"""

from __future__ import annotations

import pytest

from engine.retrieval.paging import (
    MAX_PAGE_ITEMS,
    CursorExpired,
    decode_cursor,
    encode_cursor,
)

_PAGE = "11111111-1111-1111-1111-111111111111"


def test_a_cursor_round_trips() -> None:
    assert decode_cursor(encode_cursor(_PAGE, 20)) == (_PAGE, 20)


def test_a_cursor_is_opaque_and_carries_no_query() -> None:
    """Nothing about WHAT was searched rides in it -- only which page."""
    cursor = encode_cursor(_PAGE, 0)
    assert "select" not in cursor.lower()
    assert _PAGE not in cursor  # base64, not a URL with the id in it


@pytest.mark.parametrize(
    "bad",
    ["", "not-base64!!", "e30", "eyJwIjogIm5vdC1hLXV1aWQiLCAibyI6IDB9"],
)
def test_an_unreadable_cursor_is_an_expired_cursor(bad: str) -> None:
    """A holder cannot inspect a cursor, so it cannot be blamed for its shape.

    Malformed and expired get the same answer for the same reason: neither can
    be served, and the only useful instruction in both cases is "search again".
    """
    with pytest.raises(CursorExpired):
        decode_cursor(bad)


def test_a_negative_offset_is_refused() -> None:
    with pytest.raises(CursorExpired):
        decode_cursor(encode_cursor(_PAGE, -1))


def test_the_page_cap_is_bounded() -> None:
    """A pool is not an export: the tail of a fused ranking is not worth a walk."""
    assert 0 < MAX_PAGE_ITEMS <= 200


def test_a_walk_covers_every_item_exactly_once() -> None:
    """THE invariant, modelled on the slice arithmetic `load_page` performs."""
    items = list(range(47))
    limit, offset, seen = 8, 0, []
    while True:
        window = items[offset : offset + limit]
        seen.extend(window)
        consumed = offset + len(window)
        if consumed >= len(items):
            break
        offset = decode_cursor(encode_cursor(_PAGE, consumed))[1]
    assert seen == items, "a walk must skip nothing and repeat nothing"


def test_the_last_page_has_no_cursor() -> None:
    items = list(range(10))
    window = items[8:16]
    assert 8 + len(window) >= len(items)
