import logging
import time

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_cross_encoder(model_name: str, max_retries: int = 5) -> CrossEncoder:
    """Load a CrossEncoder, retrying on network errors (e.g. interrupted downloads)."""
    for attempt in range(1, max_retries + 1):
        try:
            return CrossEncoder(model_name)
        except Exception as e:
            if attempt < max_retries:
                wait = 5 * attempt
                logger.warning(
                    "CrossEncoder download failed (attempt %d/%d), retrying in %ds: %s",
                    attempt,
                    max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")  # satisfies the type checker


class HybridRetriever:
    def __init__(
        self,
        vectorstore,
        bm25_retriever,
        dense_weight=0.7,
        sparse_weight=0.3,
    ):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = _load_cross_encoder(_RERANKER_MODEL)

    def retrieve_and_fuse(self, query, retrieval_depth=60, rrf_k=60, top_n=20):
        """
        retrieval_depth: how many candidates to pull from each of the dense/
        sparse retrievers before fusion. rrf_k: the smoothing constant in the
        Reciprocal Rank Fusion formula 1/(rrf_k + rank) — independent of how
        deep the initial retrieval goes.
        """
        dense_results = self.vectorstore.similarity_search(query, k=retrieval_depth)
        sparse_results = self.bm25_retriever.invoke(query)

        fused_scores = {}

        for rank, doc in enumerate(dense_results):
            doc_content = doc.page_content
            if doc_content not in fused_scores:
                fused_scores[doc_content] = {"doc": doc, "score": 0}
            fused_scores[doc_content]["score"] += self.dense_weight * (
                1 / (rrf_k + rank)
            )

        for rank, doc in enumerate(sparse_results):
            doc_content = doc.page_content
            if doc_content not in fused_scores:
                fused_scores[doc_content] = {"doc": doc, "score": 0}
            fused_scores[doc_content]["score"] += self.sparse_weight * (
                1 / (rrf_k + rank)
            )

        ranked_results = sorted(
            fused_scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [item for item in ranked_results[:top_n]]

    def rerank(self, query, candidates, final_k=5):
        """Returns [(doc, cross_encoder_score), ...] sorted by score, so callers
        can use the score as a real relevance signal instead of discarding it."""
        pairs = [[query, doc["doc"].page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [(item["doc"], float(score)) for item, score in scored_docs[:final_k]]

    def get_relevant_documents(self, query):
        candidates = self.retrieve_and_fuse(query, top_n=20)
        return self.rerank(query, candidates, final_k=5)
