# Advanced RAG Pipeline with Hybrid Retrieval & Automated Evaluation

[![CI](https://github.com/Pouyops/HybridRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Pouyops/HybridRAG/actions/workflows/ci.yml)

An end-to-end, highly robust Retrieval-Augmented Generation (RAG) system built using LangChain, ChromaDB, and OpenAI. This project features multi-strategy document chunking, hybrid retrieval (Dense + Sparse) with Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, citation-verified generation, a FastAPI service + Streamlit demo, Docker packaging, and a rigorous, reproducible evaluation suite — including a retrieval/reranker ablation study — using an LLM-as-a-Judge.

![RAG Pipeline Flow Graph](assets/flowgraph.png)

---

## 🌟 Key Features

* **Multi-Format Document Loader:** Ingests and normalizes text from PDF, HTML, Markdown, and TXT files.
* **Flexible Chunking:** Supports Recursive Character, Markdown Header, and Semantic chunking.
* **Hybrid Retrieval & RRF:** Combines semantic (vector) search with keyword-based (BM25) search. Results are fused via Reciprocal Rank Fusion (RRF) and reranked using a Cross-Encoder for higher accuracy. Retrieval depth and the RRF smoothing constant are independent, tunable parameters.
* **Self-Correcting Generation:** Implements citation verification to ensure the generated answer is grounded in retrieved chunks. Retrieval confidence is derived from the cross-encoder's own relevance scores, not a fixed constant.
* **Automated Evaluation:** Synthetic QA dataset generation and an independent LLM-as-a-Judge pipeline score Correctness, Faithfulness, Retrieval Relevance, and Citation Accuracy — citation accuracy is judged from the actual chunks an answer used, not copied from the generator's own self-reported confidence.
* **Served as a real system, not just a script:** a FastAPI service (`POST /query`, `/health`, `/metrics`), a Streamlit demo UI, Docker/Docker Compose packaging, centralized config (`config.py`), and CI running the test suite on every push.
* **Reproducible evaluation rigor:** a frozen, versioned evaluation set, `--runs N` for mean ± std reporting, and a full retrieval/reranker ablation study — see [`RESULTS.md`](RESULTS.md).

---

## 🛠️ Architecture

1. **Ingestion:** Uses `multiloader` to traverse directories and parse documents.
2. **Indexing:** Employs `indexer` to generate embeddings and build a Chroma vector store alongside a BM25 sparse index.
3. **Retrieval:** The `HybridRetriever` fetches candidates from both indices, performs RRF scoring, and reranks via a Cross-Encoder model (optionally skippable — see the ablation study).
4. **Generation:** The `AdvancedRAGSystem` generates responses with required citations and runs verification steps.
5. **Evaluation:** The `SyntheticEvaluator` generates testing data, while `RAGEvaluator` benchmarks the system's responses.
6. **Service:** `app.py` (FastAPI) and `streamlit_app.py` wrap the same pipeline (built once via `src/pipeline.py`) for programmatic and interactive use.

All of the above pull their tunable parameters (retrieval weights, chunk size, model names, confidence threshold, ...) from a single source of truth, [`config.py`](config.py) — see [`docs/SERVICE.md`](docs/SERVICE.md#configuration-configpy) for the full list and how to override any of them via environment variables.

---

## 📦 Prerequisites

* Python 3.9+
* An OpenAI API key
* Required libraries:
  ```bash
  pip install -r requirements.txt
  ```

## 🚀 Usage

Add an `OPENAI_API_KEY` to a `.env` file in the project root, then run the pipeline against your own documents and question:

```bash
python main.py --data-dir ./data --query "Your question here?"
```

Both flags are optional — running `python main.py` with no arguments uses `./data/` (a small original Apollo 11 corpus — see [`data/README.md`](data/README.md)) and a built-in sample query. Each run indexes the documents in `--data-dir`, answers `--query` with cited sources, then benchmarks all three chunking strategies against the committed, frozen evaluation set. Two more flags control the evaluation behavior:

* `--runs N` — repeat the strategy comparison N times and report mean ± std per metric instead of a single number.
* `--regenerate-eval-set` — regenerate `evaluation_dataset.json` via the synthetic evaluator instead of reusing the frozen, committed one (only needed if you change the corpus or want a fresh set).

For programmatic use, the same building blocks can be composed directly — or just use the factory in `src/pipeline.py`, which is what `app.py` and `streamlit_app.py` do:

```python
from src.pipeline import build_pipeline

rag_system = build_pipeline(openai_api_key=OPENAI_API_KEY, data_dir="./data/")
response = rag_system.generate_robust_answer("Your query here?")
print(response)
```

### Running as a service

```bash
uvicorn app:app --reload        # FastAPI: POST /query, GET /health, GET /metrics
streamlit run streamlit_app.py  # interactive demo UI
docker compose up --build       # both, in containers
```

See [`docs/SERVICE.md`](docs/SERVICE.md) for the full API reference, Docker instructions, and the complete `config.py` parameter table.

## 📊 Evaluation

`main.py` benchmarks all three chunking strategies (TokenRecursive, Markdown, Semantic) against a frozen, versioned 14-question evaluation set, scoring each on correctness, faithfulness, retrieval relevance, and citation accuracy via an LLM-as-a-Judge (`gpt-4o-mini` for both generation and judging). Latest results, 3 runs each (mean ± std):

| Chunking Strategy | Correctness | Faithfulness | Retrieval Relevance | Citation Accuracy | Fallback Rate |
|---|---|---|---|---|---|
| TokenRecursive | 0.997 ± 0.005 | 1.000 ± 0.000 | 0.983 ± 0.005 | 0.975 ± 0.000 | 0.400 ± 0.000 |
| Markdown | 0.997 ± 0.005 | 1.000 ± 0.000 | 0.980 ± 0.000 | 0.978 ± 0.002 | 0.400 ± 0.000 |
| Semantic | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.979 ± 0.008 | 0.964 ± 0.030 | 0.452 ± 0.073 |

**For the full picture — headline findings, a retrieval/reranker ablation study (dense-only vs. sparse-only vs. hybrid, reranker on/off, RRF weight sweep), charts, and honestly-reported caveats — see [`RESULTS.md`](RESULTS.md).** The evaluation set is frozen and committed (`evaluation_dataset.json`) so results are reproducible and comparable run over run, rather than regenerated (and therefore shifting) on every invocation.

## ✅ Testing

Unit tests cover the RRF fusion/reranking math (including the reranker on/off path), citation parsing and verification, chunk metadata assignment, and the document loader — all without calling any external LLM or embedding API, so they also run in CI with no API key required.

```bash
pip install -r requirements-dev.txt
pytest
```
