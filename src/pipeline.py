"""Factory that wires together the full Hybrid RAG pipeline: load documents
-> chunk -> index (dense + BM25) -> HybridRetriever -> AdvancedRAGSystem.

This mirrors the setup that main.py performs for its single-query demo, so
app.py, streamlit_app.py, and any future caller can build the same pipeline
without duplicating that wiring. Uses config.py for every tunable default;
callers can still override anything they need to.
"""

import logging

from langchain_openai import ChatOpenAI

from config import settings
from src.chunker import Chunker
from src.generator import AdvancedRAGSystem
from src.indexer import indexer
from src.loader import multiloader
from src.retriever import DecomposingRetriever, HybridRetriever

logger = logging.getLogger(__name__)


def build_retriever(
    vectorstore, bm25_retriever, llm=None, decompose=None, **retriever_kwargs
):
    """HybridRetriever configured from config.py, wrapped in a
    DecomposingRetriever when query decomposition is on. `llm` is the model
    that splits questions; it is required when decomposition is on.
    Keyword arguments override HybridRetriever's settings (weights,
    use_reranker, reranker, cross_encoder_confidence)."""
    decompose = settings.decompose_queries if decompose is None else decompose
    retriever = HybridRetriever(
        vectorstore=vectorstore, bm25_retriever=bm25_retriever, **retriever_kwargs
    )
    if decompose:
        if llm is None:
            raise ValueError("Query decomposition needs an llm.")
        retriever = DecomposingRetriever(retriever, llm)
    return retriever


def build_pipeline(
    openai_api_key: str,
    data_dir: str = "./data/",
    persist_directory: str = "./chroma_main_db",
    chunking_strategy: str = settings.chunking_strategy,
    max_documents: int = 50,
) -> AdvancedRAGSystem:
    """Build a ready-to-query AdvancedRAGSystem.

    Loads documents from `data_dir`, chunks them, builds a Chroma + BM25
    index at `persist_directory`, wraps it in a HybridRetriever, and returns
    an AdvancedRAGSystem ready to call `generate_robust_answer(query)` on.
    """
    if not openai_api_key:
        raise ValueError("openai_api_key must be a non-empty string.")

    loader = multiloader(data_dir)
    all_documents = loader._document_loader()
    if not all_documents:
        raise ValueError(
            f"No documents loaded from {data_dir!r}. Run `python scripts/fetch_corpus.py` to download the default corpus."
        )
    documents = all_documents[:max_documents]

    logger.info("Loaded %d document(s) from %s", len(documents), data_dir)

    embeddings = None
    if chunking_strategy == "Semantic":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    chunker = Chunker(embedding_fn=embeddings)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(
            chunker.chunk_documents(
                doc.page_content, doc.metadata, strategy=chunking_strategy
            )
        )
    logger.info("Produced %d chunk(s) using the %r strategy", len(all_chunks), chunking_strategy)

    idx = indexer(api_key=openai_api_key)
    vectorstore, bm25_retriever = idx.create_indexes(
        all_chunks, persist_directory=persist_directory
    )

    generator_llm = ChatOpenAI(
        model=settings.generator_model,
        temperature=settings.generator_temperature,
        api_key=openai_api_key,
    )
    retriever = build_retriever(vectorstore, bm25_retriever, llm=generator_llm)

    return AdvancedRAGSystem(llm=generator_llm, retriever=retriever)
