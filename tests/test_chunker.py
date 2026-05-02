from services.ingestion.chunker import chunk_text, count_tokens


def test_empty_returns_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_short_text_single_chunk() -> None:
    pieces = chunk_text("hello world")
    assert len(pieces) == 1
    assert pieces[0].chunk_index == 0
    assert pieces[0].content.strip() == "hello world"
    assert pieces[0].token_count == count_tokens("hello world")


def test_overlap_produces_multiple_pieces() -> None:
    text = " ".join(["token"] * 2000)
    pieces = chunk_text(text, chunk_tokens=512, overlap=64)
    assert len(pieces) > 1
    # Chunk indexes should be contiguous starting at 0.
    for i, p in enumerate(pieces):
        assert p.chunk_index == i
    # Token counts should never exceed the window.
    assert all(p.token_count <= 512 for p in pieces)


def test_literal_special_tokens_in_user_content_do_not_raise() -> None:
    # Real user content (e.g. Linear comments quoting LLM prompts) can contain
    # literal "<|endoftext|>" strings. tiktoken's default `disallowed_special`
    # would raise ValueError; the chunker must tolerate them as plain text.
    text = "before <|endoftext|> middle <|im_start|> after"
    pieces = chunk_text(text)
    assert len(pieces) == 1
    assert count_tokens(text) > 0
