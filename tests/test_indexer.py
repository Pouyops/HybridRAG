from langchain_core.documents import Document

from src.indexer import indexer


def _make_indexer():
    """Build an indexer without running __init__, so no embeddings client
    (and therefore no API key) is required."""
    return object.__new__(indexer)


def test_prepare_metadata_assigns_per_file_chunk_index():
    idx = _make_indexer()
    chunks = [
        Document(page_content="a", metadata={"filepath": "doc1.txt"}),
        Document(page_content="b", metadata={"filepath": "doc1.txt"}),
        Document(page_content="c", metadata={"filepath": "doc2.txt"}),
    ]

    processed = idx._prepare_metadata(chunks)

    assert [c.metadata["chunk_index"] for c in processed] == [0, 1, 0]


def test_prepare_metadata_picks_deepest_available_header_as_heading():
    idx = _make_indexer()
    chunks = [
        Document(
            page_content="a",
            metadata={
                "filepath": "doc.md",
                "Header 1": "Intro",
                "Header 3": "Deep Section",
            },
        )
    ]

    processed = idx._prepare_metadata(chunks)

    assert processed[0].metadata["section_heading"] == "Deep Section"


def test_prepare_metadata_defaults_heading_when_no_headers_present():
    idx = _make_indexer()
    chunks = [Document(page_content="a", metadata={"filepath": "doc.txt"})]

    processed = idx._prepare_metadata(chunks)

    assert processed[0].metadata["section_heading"] == "N/A"
    assert processed[0].metadata["character_count"] == 1
