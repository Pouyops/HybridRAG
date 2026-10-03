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


def test_create_indexes_rebuild_does_not_duplicate_chunks(tmp_path):
    """Regression test: Chroma.from_documents appended to an existing
    persisted collection, so every app restart duplicated the corpus."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    idx = _make_indexer()
    idx.embeddings = DeterministicFakeEmbedding(size=16)

    def chunks():
        return [
            Document(page_content=f"chunk {i}", metadata={"filepath": "doc.txt"})
            for i in range(3)
        ]

    persist_dir = str(tmp_path / "chroma")
    idx.create_indexes(chunks(), persist_directory=persist_dir)
    vectorstore, bm25 = idx.create_indexes(chunks(), persist_directory=persist_dir)

    assert len(vectorstore.get()["ids"]) == 3
    assert len(bm25.docs) == 3


def test_bm25_tokenize_ignores_case_and_punctuation_but_keeps_hyphenated_terms():
    from src.indexer import bm25_tokenize

    assert bm25_tokenize('A "Vary: *" field, If-None-Match and max-age=60.') == [
        "a", "vary", "field", "if-none-match", "and", "max-age", "60",
    ]


def test_bm25_index_matches_terms_regardless_of_punctuation(tmp_path):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    idx = _make_indexer()
    idx.embeddings = DeterministicFakeEmbedding(size=16)
    chunks = [
        Document(page_content='A stored response with "Vary: *" always fails.', metadata={}),
        Document(page_content="Unrelated text about connections.", metadata={}),
        # BM25Okapi's IDF is 0 for a term in exactly half the documents, so
        # the corpus needs a third document for "vary" to score above zero.
        Document(page_content="More unrelated text about methods.", metadata={}),
    ]

    _, bm25 = idx.create_indexes(chunks, persist_directory=str(tmp_path / "chroma"))

    assert bm25.invoke("vary")[0].page_content.startswith("A stored response")
