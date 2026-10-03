from unittest.mock import patch

from langchain_core.documents import Document

from src.retriever import HybridRetriever


def _make_retriever(dense_weight=0.7, sparse_weight=0.3, use_reranker=True):
    """Build a HybridRetriever without running __init__, so the CrossEncoder
    (which downloads a model from the network) is never loaded."""
    retriever = object.__new__(HybridRetriever)
    retriever.dense_weight = dense_weight
    retriever.sparse_weight = sparse_weight
    retriever.use_reranker = use_reranker
    return retriever


class FakeVectorstore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k):
        return self.docs[:k]


class FakeBM25:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


def test_retrieve_and_fuse_ranks_docs_in_both_lists_higher():
    shared = Document(page_content="shared chunk", metadata={})
    dense_only = Document(page_content="dense only chunk", metadata={})
    sparse_only = Document(page_content="sparse only chunk", metadata={})

    retriever = _make_retriever()
    retriever.vectorstore = FakeVectorstore([shared, dense_only])
    retriever.bm25_retriever = FakeBM25([shared, sparse_only])

    fused = retriever.retrieve_and_fuse("query", retrieval_depth=60, rrf_k=60, top_n=10)
    fused_by_content = {item["doc"].page_content: item["score"] for item in fused}

    # The chunk retrieved by both dense and sparse search should outscore
    # either chunk retrieved by only one of the two.
    assert fused_by_content["shared chunk"] > fused_by_content["dense only chunk"]
    assert fused_by_content["shared chunk"] > fused_by_content["sparse only chunk"]


def test_retrieve_and_fuse_respects_top_n():
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(5)]

    retriever = _make_retriever()
    retriever.vectorstore = FakeVectorstore(docs)
    retriever.bm25_retriever = FakeBM25([])

    fused = retriever.retrieve_and_fuse("query", retrieval_depth=60, rrf_k=60, top_n=2)

    assert len(fused) == 2


def test_retrieve_and_fuse_retrieval_depth_and_rrf_k_are_independent():
    """retrieval_depth controls how many docs similarity_search returns;
    rrf_k is only the smoothing constant in the fusion formula. Changing one
    must not change the other's effect."""
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(5)]

    retriever = _make_retriever()
    retriever.vectorstore = FakeVectorstore(docs)
    retriever.bm25_retriever = FakeBM25([])

    shallow = retriever.retrieve_and_fuse("query", retrieval_depth=2, rrf_k=60, top_n=10)
    deep = retriever.retrieve_and_fuse("query", retrieval_depth=5, rrf_k=60, top_n=10)

    assert len(shallow) == 2
    assert len(deep) == 5

    # Same retrieval_depth, different rrf_k must change the fused score.
    low_k = retriever.retrieve_and_fuse("query", retrieval_depth=5, rrf_k=1, top_n=10)
    low_k_by_content = {item["doc"].page_content: item["score"] for item in low_k}
    deep_by_content = {item["doc"].page_content: item["score"] for item in deep}
    assert low_k_by_content["chunk 0"] != deep_by_content["chunk 0"]


class FakeCrossEncoder:
    def predict(self, pairs):
        # Score chunk i as i, so higher index == more relevant.
        return [float(pair[1].split()[-1]) for pair in pairs]


def test_rerank_orders_by_cross_encoder_score_and_truncates():
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(4)]
    candidates = [{"doc": doc, "score": 0.0} for doc in docs]

    retriever = _make_retriever()
    retriever.reranker = FakeCrossEncoder()

    reranked = retriever.rerank("query", candidates, final_k=2)

    assert [doc.page_content for doc, score in reranked] == ["chunk 3", "chunk 2"]
    assert [score for doc, score in reranked] == [3.0, 2.0]


def test_get_relevant_documents_returns_doc_score_pairs():
    """The retrieval confidence fix needs real cross-encoder scores to flow
    all the way out of get_relevant_documents, not just plain documents."""
    shared = Document(page_content="shared chunk", metadata={})
    dense_only = Document(page_content="dense only chunk", metadata={})

    class ConstantCrossEncoder:
        def predict(self, pairs):
            return [1.0 for _ in pairs]

    retriever = _make_retriever()
    retriever.vectorstore = FakeVectorstore([shared, dense_only])
    retriever.bm25_retriever = FakeBM25([shared])
    retriever.reranker = ConstantCrossEncoder()

    results = retriever.get_relevant_documents("query")

    assert all(isinstance(score, float) for _, score in results)
    assert {doc.page_content for doc, _ in results} == {
        "shared chunk",
        "dense only chunk",
    }


def test_get_relevant_documents_with_reranker_disabled_skips_cross_encoder():
    """use_reranker=False must skip .rerank() entirely (no CrossEncoder call)
    and instead return the top final_k RRF-fused candidates, using the fused
    score in place of the cross-encoder score."""
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(5)]

    class ExplodingCrossEncoder:
        def predict(self, pairs):
            raise AssertionError("rerank() must not be called when use_reranker=False")

    retriever = _make_retriever(use_reranker=False)
    retriever.vectorstore = FakeVectorstore(docs)
    retriever.bm25_retriever = FakeBM25([])
    retriever.reranker = ExplodingCrossEncoder()

    with patch("src.retriever.settings.final_k", 3):
        results = retriever.get_relevant_documents("query")

    assert len(results) == 3
    assert [doc.page_content for doc, _ in results] == ["chunk 0", "chunk 1", "chunk 2"]
    assert all(isinstance(score, float) for _, score in results)


def test_get_relevant_documents_reranker_default_true_unchanged():
    """Default construction path (use_reranker not passed) must keep using
    the cross-encoder, matching prior behavior byte-for-byte."""
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(3)]

    retriever = _make_retriever()  # use_reranker defaults to True
    assert retriever.use_reranker is True

    retriever.vectorstore = FakeVectorstore(docs)
    retriever.bm25_retriever = FakeBM25([])
    retriever.reranker = FakeCrossEncoder()

    with patch("src.retriever.settings.final_k", 2):
        results = retriever.get_relevant_documents("query")

    # FakeCrossEncoder scores by trailing index, so highest index chunks win.
    assert [doc.page_content for doc, _ in results] == ["chunk 2", "chunk 1"]


def test_retrieve_and_fuse_counts_a_duplicated_chunk_once_per_retriever():
    """A chunk appearing twice in one retriever's list (e.g. a duplicated
    index) must not collect that retriever's score twice."""
    dup = Document(page_content="dup chunk", metadata={})
    other = Document(page_content="other chunk", metadata={})

    retriever = _make_retriever(dense_weight=1.0, sparse_weight=0.0)
    retriever.vectorstore = FakeVectorstore([dup, dup, other])
    retriever.bm25_retriever = FakeBM25([])

    fused = retriever.retrieve_and_fuse("query", retrieval_depth=60, rrf_k=60, top_n=10)
    fused_by_content = {item["doc"].page_content: item["score"] for item in fused}

    assert fused_by_content["dup chunk"] == 1.0 / 61


def test_retrieve_and_fuse_uses_one_based_ranks_so_rrf_k_zero_is_valid():
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(2)]

    retriever = _make_retriever(dense_weight=1.0, sparse_weight=0.0)
    retriever.vectorstore = FakeVectorstore(docs)
    retriever.bm25_retriever = FakeBM25([])

    fused = retriever.retrieve_and_fuse("query", retrieval_depth=60, rrf_k=0, top_n=10)

    assert [item["score"] for item in fused] == [1.0, 0.5]


def test_retrieve_and_fuse_limits_sparse_results_to_retrieval_depth():
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(5)]

    retriever = _make_retriever()
    retriever.vectorstore = FakeVectorstore([])
    retriever.bm25_retriever = FakeBM25(docs)

    fused = retriever.retrieve_and_fuse("query", retrieval_depth=2, rrf_k=60, top_n=10)

    assert [item["doc"].page_content for item in fused] == ["chunk 0", "chunk 1"]


def test_rerank_with_no_candidates_returns_empty_without_calling_model():
    class ExplodingCrossEncoder:
        def predict(self, pairs):
            raise AssertionError("predict must not be called with no candidates")

    retriever = _make_retriever()
    retriever.reranker = ExplodingCrossEncoder()

    assert retriever.rerank("query", []) == []


def test_injected_reranker_is_used_without_loading_a_model():
    sentinel = FakeCrossEncoder()

    with patch("src.retriever._load_cross_encoder") as load:
        retriever = HybridRetriever(
            vectorstore=None, bm25_retriever=None, use_reranker=True, reranker=sentinel
        )

    load.assert_not_called()
    assert retriever.reranker is sentinel
