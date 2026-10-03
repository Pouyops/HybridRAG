from src.chunker import Chunker

SECTION = "### 9.3. Method Definitions\n\n" + "\n\n".join(
    f"Paragraph {i} about request methods and their semantics." for i in range(60)
)
TEXT = "# RFC 9110 - HTTP Semantics\n\n## 9. Methods\n\n" + SECTION + "\n\n### 9.4. Short\n\nShort text."


def _chunks(max_tokens):
    chunker = Chunker(embedding_fn=None, chunk_overlap=0, markdown_max_tokens=max_tokens)
    return chunker.chunk_documents(TEXT, {"filepath": "rfc.md"}, strategy="Markdown")


def test_markdown_cap_splits_long_sections_and_keeps_heading_path():
    uncapped = _chunks(0)
    capped = _chunks(100)

    long_uncapped = [c for c in uncapped if c.metadata.get("Header 3") == "9.3. Method Definitions"]
    long_capped = [c for c in capped if c.metadata.get("Header 3") == "9.3. Method Definitions"]
    assert len(long_uncapped) == 1
    assert len(long_capped) > 1

    # Continuation pieces carry the heading path and the section metadata.
    for piece in long_capped[1:]:
        assert piece.page_content.startswith(
            "# RFC 9110 - HTTP Semantics\n## 9. Methods\n### 9.3. Method Definitions (continued)"
        )
        assert piece.metadata["filepath"] == "rfc.md"
    # Nothing is lost: every paragraph lands in some piece.
    joined = "\n".join(c.page_content for c in long_capped)
    assert all(f"Paragraph {i} " in joined for i in range(60))


def test_markdown_cap_leaves_short_sections_alone():
    short = [c for c in _chunks(100) if c.metadata.get("Header 3") == "9.4. Short"]
    assert len(short) == 1
    # (MarkdownHeaderTextSplitter itself pads the header line with spaces.)
    lines = [line.strip() for line in short[0].page_content.splitlines() if line.strip()]
    assert lines == ["### 9.4. Short", "Short text."]
