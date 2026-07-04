# Advanced RAG Pipeline with Hybrid Retrieval & Automated Evaluation

An end-to-end, highly robust Retrieval-Augmented Generation (RAG) system built using LangChain, ChromaDB, and OpenAI. This project features multi-strategy document chunking, hybrid retrieval (Dense + Sparse) with Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and a comprehensive automated evaluation suite using an LLM-as-a-Judge.

![RAG Pipeline Flow Graph](assets/flowgraph.png)

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
* OpenAI API Key
* Required libraries:
  ```bash
  pip install -r requirements.txt
  ```

## 🚀 Usage

Add an `OPENAI_API_KEY` to a `.env` file in the project root, then run the pipeline against your own documents and question:

```bash
python main.py --data-dir ./data --query "Your question here?"
```

Both flags are optional — running `python main.py` with no arguments uses `./data/` and a built-in sample query. Each run: indexes the documents in `--data-dir`, answers `--query` with cited sources, generates a 15-question synthetic evaluation set, and benchmarks all three chunking strategies against it.

For programmatic use, the same building blocks can be composed directly:

```Python
from src.loader import multiloader
from src.chunker import Chunker
from src.indexer import indexer
from src.retriever import HybridRetriever
from src.generator import AdvancedRAGSystem

loader = multiloader("/path/to/data")
documents = loader._document_loader()

chunker = Chunker(embedding_fn=None)
all_chunks = []
for doc in documents:
    all_chunks.extend(chunker.chunk_documents(doc.page_content, doc.metadata, strategy="TokenRecursive"))

idx = indexer(api_key=OPENAI_API_KEY)
vectorstore, bm25 = idx.create_indexes(all_chunks)

retriever = HybridRetriever(vectorstore=vectorstore, bm25_retriever=bm25)
rag_system = AdvancedRAGSystem(llm=generator_llm, retriever=retriever)

response = rag_system.generate_robust_answer("Your query here?")
print(response)
```

## 📊 Evaluation

`main.py` benchmarks all three chunking strategies (TokenRecursive, Markdown, Semantic) against a synthetic 15-question evaluation set, scoring each on correctness, faithfulness, retrieval relevance, and citation accuracy via an LLM-as-a-Judge. Sample results from a run against a single sample document (`gpt-4o-mini` for both generation and judging):

| Chunking Strategy | Correctness | Faithfulness | Retrieval Relevance | Citation Accuracy | Fallback Rate |
|---|---|---|---|---|---|
| TokenRecursive | 0.9071 | 0.9857 | 0.8429 | 0.9524 | 0.0000 |
| Markdown | 0.8958 | 1.0000 | 0.9583 | 1.0000 | 0.1667 |
| Semantic | 0.8750 | 0.9900 | 0.9300 | 0.9500 | 0.4000 |

Results are generated per-run from a synthetic dataset, so exact numbers will vary between runs and datasets.

## ✅ Testing

Unit tests cover the RRF fusion/reranking math, citation parsing and verification, chunk metadata assignment, and the document loader — all without calling any external LLM or embedding API.

```bash
pip install -r requirements-dev.txt
pytest
```
