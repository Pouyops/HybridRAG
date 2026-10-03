import logging
import time

from sentence_transformers import CrossEncoder

from config import settings

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
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        use_reranker: bool = True,
    ):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_reranker = use_reranker
        # Skip the network download entirely when reranking is disabled (e.g.
        # for the reranker-off ablation arm) — no point paying that cost.
        self.reranker = _load_cross_encoder(_RERANKER_MODEL) if use_reranker else None

    def retrieve_and_fuse(
        self,
        query,
        retrieval_depth=settings.retrieval_depth,
        rrf_k=settings.rrf_k,
        top_n=settings.top_n,
    ):
        """
        retrieval_depth: how many candidates to pull from each of the dense/
        sparse retrievers before fusion. rrf_k: the smoothing constant in the
        Reciprocal Rank Fusion formula 1/(rrf_k + rank) — independent of how
        deep the initial retrieval goes. Ranks are 1-based, as in the standard
        RRF formulation (so rrf_k=0 is valid and the top hit scores 1/(rrf_k+1)).
        """
        dense_results = self.vectorstore.similarity_search(query, k=retrieval_depth)
        sparse_results = self.bm25_retriever.invoke(query)[:retrieval_depth]

        fused_scores = {}

        for results, weight in (
            (dense_results, self.dense_weight),
            (sparse_results, self.sparse_weight),
        ):
            # A chunk contributes once per retriever, at its best rank — a
            # duplicate in one result list must not add its score twice.
            seen = set()
            for rank, doc in enumerate(results, start=1):
                doc_content = doc.page_content
                if doc_content in seen:
                    continue
                seen.add(doc_content)
                if doc_content not in fused_scores:
                    fused_scores[doc_content] = {"doc": doc, "score": 0}
                fused_scores[doc_content]["score"] += weight * (1 / (rrf_k + rank))

        ranked_results = sorted(
            fused_scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [item for item in ranked_results[:top_n]]

    def rerank(self, query, candidates, final_k=settings.final_k):
        """Returns [(doc, cross_encoder_score), ...] sorted by score, so callers
        can use the score as a real relevance signal instead of discarding it."""
        if not candidates:
            return []
        pairs =[[query, doc["doc"].page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [(item["doc"], float(score)) for item, score in scored_docs[:final_k]]

    def get_relevant_documents(self, query):
        candidates = self.retrieve_and_fuse(query, top_n=settings.top_n)
        if not self.use_reranker:
            # Return the top final_k straight from RRF fusion, using the
            # fused score in place of the cross-encoder score so the
            # (doc, score) contract stays intact for downstream callers
            # (e.g. score_confidence's rerank_scores parameter). Note these
            # are not calibrated relevance logits: RRF scores are ~0.01, so
            # score_confidence's sigmoid maps them all to ~0.5.
            top = candidates[: settings.final_k]
            return [(item["doc"], float(item["score"])) for item in top]
        return self.rerank(query, candidates, final_k=settings.final_k)
