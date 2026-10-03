import logging
import time
from typing import List

from pydantic import BaseModel, Field
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
        use_reranker: bool = settings.use_reranker,
        reranker=None,
        cross_encoder_confidence: bool = settings.cross_encoder_confidence,
    ):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_reranker = use_reranker
        self.cross_encoder_confidence = cross_encoder_confidence
        # The cross-encoder is needed to reorder candidates (use_reranker) or
        # just to score the final chunks for the confidence gate. Skip the
        # download when neither is wanted. Callers comparing many configs can
        # pass one preloaded `reranker` instead of loading it per retriever.
        needs_model = use_reranker or cross_encoder_confidence
        if needs_model and reranker is None:
            reranker = _load_cross_encoder(_RERANKER_MODEL)
        self.reranker = reranker if needs_model else None

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
        return self.finalize(query, candidates)

    def finalize(self, query, candidates):
        """Pick the final_k chunks from fused `candidates` and attach a score
        to each, returning [(doc, score), ...].

        With use_reranker the cross-encoder reorders the candidates. Without
        it the fused order is kept, and (with cross_encoder_confidence) the
        cross-encoder only scores the chosen chunks, so score_confidence
        still gets calibrated relevance logits. Otherwise the raw RRF score
        is returned; it is ~0.01, so score_confidence's sigmoid maps it to
        ~0.5 for every query, which is why that path is not the default.
        """
        if self.use_reranker:
            return self.rerank(query, candidates, final_k=settings.final_k)
        top = candidates[: settings.final_k]
        if self.cross_encoder_confidence and top:
            scores = self.reranker.predict([[query, item["doc"].page_content] for item in top])
            return [(item["doc"], float(score)) for item, score in zip(top, scores)]
        return [(item["doc"], float(item["score"])) for item in top]


class SubQuestions(BaseModel):
    questions: List[str] = Field(
        description="Self-contained sub-questions, or an empty list for a single-part question"
    )


class DecomposingRetriever:
    """Retrieves separately for each part of a multi-part question.

    A single query with final_k=5 rarely brings back two distant passages:
    on the RFC set, no configuration retrieved all the evidence for more
    than 4 of 9 multi-hop questions. The LLM splits the question into up to
    max_subquestions sub-questions; each (plus the original question) is
    retrieved and fused as usual, and the per-query rankings are combined
    with reciprocal rank fusion so every sub-question's best passages
    compete for the final slots. Single-part questions cost one extra LLM
    call and are otherwise retrieved exactly as before.
    """

    def __init__(self, base, llm, max_subquestions=settings.max_subquestions):
        self.base = base
        self.decomposer = llm.with_structured_output(SubQuestions)
        self.max_subquestions = max_subquestions

    def decompose(self, query):
        prompt = (
            "Decide whether the user's question asks for two or more separate pieces of "
            "information that are likely to be documented in different places. If it "
            "does, rewrite each piece as a standalone sub-question, using the question's "
            "own terms. Do not add background, definition or follow-up questions the "
            "user did not ask. If the question asks for one piece of information, "
            f"return an empty list. Return at most {self.max_subquestions}.\n"
            f"Question: {query}"
        )
        try:
            questions = self.decomposer.invoke(prompt).questions
        except Exception as e:  # noqa: BLE001 - decomposition is best-effort
            logger.warning("Query decomposition failed, retrieving undecomposed: %s", e)
            return []
        seen = {query.strip().lower()}
        cleaned = []
        for q in questions:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                cleaned.append(q.strip())
        return cleaned[: self.max_subquestions]

    def get_relevant_documents(self, query):
        sub_questions = self.decompose(query)
        if not sub_questions:
            return self.base.get_relevant_documents(query)

        combined = {}
        for q in [query] + sub_questions:
            for rank, item in enumerate(
                self.base.retrieve_and_fuse(q, top_n=settings.top_n), start=1
            ):
                key = item["doc"].page_content
                entry = combined.setdefault(key, {"doc": item["doc"], "score": 0.0})
                entry["score"] += 1 / (settings.rrf_k + rank)
        candidates = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return self.base.finalize(query, candidates[: settings.top_n])
