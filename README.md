# Advanced RAG Pipeline with Hybrid Retrieval & Automated Evaluation

An end-to-end, highly robust Retrieval-Augmented Generation (RAG) system built using LangChain, ChromaDB, and Google Gemini. This project features multi-strategy document chunking, hybrid retrieval (Dense + Sparse) with Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and a comprehensive automated evaluation suite using an LLM-as-a-Judge.

![RAG Pipeline Flow Graph](path/to/your/pipeline-flow-graph.png)

---

## 🌟 Key Features

* **Multi-Format Document Loader:** Ingests and normalizes text from PDF, HTML, Markdown, and TXT files.
* **Flexible Chunking:** Supports Recursive Character, Markdown Header, and Semantic chunking.
* **Hybrid Retrieval & RRF:** Combines semantic (vector) search with keyword-based (BM25) search. Results are fused via Reciprocal Rank Fusion (RRF) and reranked using a Cross-Encoder for higher accuracy.
* **Self-Correcting Generation:** Implements citation verification to ensure the generated answer is grounded in retrieved chunks.
* **Automated Evaluation:** Includes synthetic QA dataset generation and an LLM-as-a-Judge pipeline to score performance based on Correctness, Faithfulness, and Retrieval Relevance.

---

## 🛠️ Architecture

1. **Ingestion:** Uses `multiloader` to traverse directories and parse documents.
2. **Indexing:** Employs `indexer` to generate embeddings and build a Chroma vector store alongside a BM25 sparse index.
3. **Retrieval:** The `HybridRetriever` fetches candidates from both indices, performs RRF scoring, and reranks via a Cross-Encoder model.
4. **Generation:** The `AdvancedRAGSystem` generates responses with required citations and runs verification steps.
5. **Evaluation:** The `SyntheticEvaluator` generates testing data, while `RAGEvaluator` benchmarks the system's responses.

---

## 📦 Prerequisites

* Python 3.9+
* Google Gemini API Key
* Required libraries:
  ```bash
  pip install langchain_text_splitters pymupdf langchain_experimental \
  langchain_google_genai chromadb rank_bm25 sentence-transformers \
  langchain_community bs4 pydantic
## 🚀 Usage
1. Initialize and Ingest

```Python
# Initialize models and loaders
loader = multiloader('/path/to/data')
documents = loader._document_loader()

# Chunk documents
chunker = Chunker()
all_chunks = chunker.chunk_documents(documents, strategy="TokenRecursive")
```

2. Indexing and Retrieval
```Python

idx = indexer()
vectorstore, bm25 = idx.create_indexes(all_chunks)

retriever = HybridRetriever(vectorestore=vectorstore, bm25_retriever=bm25)
rag_system = AdvancedRAGSystem(llm=generator_llm, retriever=retriever)
```
3. Generate Answer
```Python

response = rag_system.generate_robust_answer("Your query here?")
print(response)
```
📊 Evaluation

You can benchmark the system by generating a synthetic test set:
```Python

synthetic_evaluator = SyntheticEvaluator(vectorstore=vectorstore, llm=judge_llm)
synthetic_evaluator.build_dataset(total_questions=15)
