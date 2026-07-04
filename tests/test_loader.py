from src.loader import multiloader


def test_normalize_text_collapses_whitespace():
    loader = multiloader(".")

    assert loader._normalize_text("  Hello \n\n  world \t!  ") == "Hello world !"


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
