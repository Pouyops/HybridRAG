from langchain_core.documents import Document

from src.retriever import HybridRetriever


def _make_retriever(dense_weight=0.7, sparse_weight=0.3):
    """Build a HybridRetriever without running __init__, so the CrossEncoder
    (which downloads a model from the network) is never loaded."""
    retriever = object.__new__(HybridRetriever)
    retriever.dense_weight = dense_weight
    retriever.sparse_weight = sparse_weight
    retriever.initial_k = 10
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

    fused = retriever.retrieve_and_fuse("query", fusion_k=60, top_n=10)
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

    fused = retriever.retrieve_and_fuse("query", fusion_k=60, top_n=2)

    assert len(fused) == 2


def test_rerank_orders_by_cross_encoder_score_and_truncates():
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(4)]
    candidates = [{"doc": doc, "score": 0.0} for doc in docs]

    class FakeCrossEncoder:
        def predict(self, pairs):
            # Score chunk i as i, so higher index == more relevant.
            return [float(pair[1].split()[-1]) for pair in pairs]

    retriever = _make_retriever()
    retriever.reranker = FakeCrossEncoder()

    reranked = retriever.rerank("query", candidates, final_k=2)

    assert [doc.page_content for doc in reranked] == ["chunk 3", "chunk 2"]
