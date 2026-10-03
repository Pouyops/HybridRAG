from src.loader import multiloader


def test_normalize_text_collapses_whitespace_but_keeps_line_breaks():
    loader = multiloader(".")

    assert loader._normalize_text("  Hello \n\n  world \t!  ") == "Hello\n\nworld !"
    assert loader._normalize_text("a\r\n\r\n\r\n\nb") == "a\n\nb"


def test_loaded_markdown_still_splits_on_headers(tmp_path):
    """Regression test: flattening all whitespace onto one line left the
    Markdown chunker with a single header line, i.e. one chunk per file."""
    from src.chunker import Chunker

    (tmp_path / "doc.md").write_text(
        "# Title\n\nIntro text.\n\n## Section A\n\nAlpha.\n\n## Section B\n\nBeta.\n",
        encoding="utf-8",
    )
    doc = multiloader(str(tmp_path))._document_loader()[0]

    chunks = Chunker(embedding_fn=None).chunk_documents(
        doc.page_content, doc.metadata, strategy="Markdown"
    )

    assert [c.metadata.get("Header 2") for c in chunks] == [None, "Section A", "Section B"]
    assert all(c.metadata["Header 1"] == "Title" for c in chunks)


def test_document_loader_returns_files_in_sorted_order(tmp_path):
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")

    documents = multiloader(str(tmp_path))._document_loader()

    assert [d.metadata["filename"] for d in documents] == ["a.txt", "b.txt", "c.txt"]


def test_document_loader_reads_supported_files_and_skips_others(tmp_path):
    (tmp_path / "note.txt").write_text("Hello   world", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")

    loader = multiloader(str(tmp_path))
    documents = loader._document_loader()

    assert len(documents) == 1
    assert documents[0].page_content == "Hello world"
    assert documents[0].metadata["filename"] == "note.txt"


def test_process_file_returns_none_for_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    loader = multiloader(str(tmp_path))
    assert loader._process_file(str(empty_file)) is None
