from sentence_transformers import CrossEncoder

class HybridRetriever():
    def __init__(self, vectorestore, bm25_retriever, dense_weight=0.7, sparse_weight=0.3, initial_k=10):
        self.vectorestore = vectorestore
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.initial_k = initial_k
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve_and_fuse(self, query, fusion_k=60, top_n=20):
        dense_results = self.vectorestore.similarity_search(query, k=fusion_k)
        sparse_results = self.bm25_retriever.invoke(query)

        fused_scores = {}

        for rank, doc in enumerate(dense_results):
            doc_content = doc.page_content
            if doc_content not in fused_scores:
                fused_scores[doc_content] = {"doc": doc, "score": 0}
            fused_scores[doc_content]["score"] += self.dense_weight * (1 / (fusion_k + rank))

        for rank, doc in enumerate(sparse_results):
            doc_content = doc.page_content
            if doc_content not in fused_scores:
                fused_scores[doc_content] = {"doc": doc, "score": 0}
            fused_scores[doc_content]["score"] += self.sparse_weight * (1 / (fusion_k + rank))

        ranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

        return [item for item in ranked_results[:top_n]]

    def rerank(self, query, candidates, final_k=5):
        pairs = [[query, doc['doc'].page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [item['doc'] for item, score in scored_docs[:final_k]]

    def get_relevant_documents(self, query):
        candidates = self.retrieve_and_fuse(query, top_n=20)
        final_results = self.rerank(query, candidates, final_k=5)
        return final_results