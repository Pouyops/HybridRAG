# Service Layer

This document covers the API service, the Streamlit demo, Docker, configuration,
and CI for HybridRAG. The core retrieval/generation/evaluation pipeline itself
is documented in the root `README.md`; this file only covers the service
wrapper around it.

## Running the API locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

The pipeline (document loading, chunking, dense + BM25 indexing, cross-encoder
loading) is built once at startup, not per-request — expect a delay of tens of
seconds to a couple of minutes on first run depending on corpus size, since
Chroma embeddings and the cross-encoder model are prepared during that
startup window. Startup time is logged to the console.

Requires `OPENAI_API_KEY` in a `.env` file at the repo root (see `.env`,
gitignored). Optionally set `DATA_DIR` to point at a different documents
directory; it defaults to `./data/`.

### Endpoints

- `POST /query` — body `{"query": "<question>"}`. Returns:
  ```json
  {
    "answer": "...",
    "citations": [...],
    "confidence": {...},
    "status": "Success",
    "latency_ms": 1234.5
  }
  ```
  `citations` is the generator's list of flagged (unsupported) citations, and
  `confidence` is the confidence_metrics dict (`retrieval_confidence`,
  `citation_coverage`, `answer_completeness`, `composite_score`,
  `is_confident`). When the system falls back to "Insufficient Information",
  `citations` is `[]` and `confidence` is `null`.

- `GET /health` — `{"status": "ok"}`.

- `GET /metrics` — rolling median latency and average estimated cost across
  requests served so far in this process (in-memory only, resets on
  restart):
  ```json
  {
    "requests_served": 3,
    "median_latency_ms": 1180.2,
    "avg_cost_usd": 0.00041
  }
  ```

### Observability

Each request's latency and an estimated OpenAI cost (via `tiktoken` token
counts against illustrative gpt-4o-mini pricing — **approximate, verify
against current OpenAI pricing** — $0.15/1M input tokens, $0.60/1M output
tokens) are logged via the standard `logging` module. This is intentionally
lightweight: no external time-series database, just console logs plus the
in-memory `/metrics` rollup.

## Running the Streamlit demo

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This builds the pipeline directly (it does not call the FastAPI service), so
it's a single command with no separate API process required. The pipeline
build is cached with `st.cache_resource`, so it only runs once per Streamlit
process, not on every question submitted.

## Running with Docker

```bash
docker compose up --build
```

This starts two services:

- `api` — the FastAPI app on `http://localhost:8000`
- `ui` — the Streamlit demo on `http://localhost:8501`

Both read secrets from `.env` via `env_file:` in `docker-compose.yml` — the
`.env` file is never copied into the image (`.dockerignore` excludes it, and
the `Dockerfile` never `COPY`s it). `./data` is mounted read-only into the
container so both services index the same corpus.

To build only the image without starting anything:

```bash
docker compose build
```

## Configuration (`config.py`)

`config.py` centralizes every tunable parameter that was previously a
scattered magic number/string across `src/retriever.py`, `src/generator.py`,
`src/chunker.py`, `src/indexer.py`, `src/evaluator.py`, and `main.py`. It is a
`pydantic-settings` `Settings` object — every field can be overridden with an
environment variable of the same name (case-insensitive), including via
`.env`, without touching code:

| Setting | Default | Controls |
|---|---|---|
| `dense_weight` | `0.7` | HybridRetriever dense-score weight in RRF fusion |
| `sparse_weight` | `0.3` | HybridRetriever sparse-score weight in RRF fusion |
| `retrieval_depth` | `60` | Candidates pulled from each of dense/sparse before fusion |
| `rrf_k` | `60` | RRF smoothing constant `1/(rrf_k + rank)` |
| `top_n` | `20` | Candidates kept after fusion, before reranking |
| `final_k` | `5` | Chunks kept after cross-encoder reranking |
| `confidence_threshold` | `0.75` | Minimum composite confidence to return "Success" |
| `chunk_size` | `512` | Token-recursive chunk size |
| `chunk_overlap` | `50` | Token-recursive chunk overlap |
| `generator_model` | `gpt-4o-mini` | Answer-generation LLM |
| `judge_model` | `gpt-4o-mini` | Evaluation / citation-judge LLM |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `generator_temperature` | `0` | Generator LLM temperature |
| `judge_temperature` | `0` | Judge LLM temperature |
| `synthetic_temperature` | `0.7` | Synthetic dataset generator LLM temperature |

`config.py` does **not** hold API keys — `OPENAI_API_KEY` / `OPENROUTER_API_KEY`
stay as `os.getenv()` calls at the call sites, loaded via `python-dotenv`, same
as before.

Example override:

```bash
DENSE_WEIGHT=0.6 SPARSE_WEIGHT=0.4 uvicorn app:app --reload
```

## What CI checks (`.github/workflows/ci.yml`)

On every push and pull request to `main`: checks out the repo, sets up Python
3.11, installs `requirements.txt` + `requirements-dev.txt`, and runs
`pytest -q`. The test suite is fully mocked (no OpenAI calls, no real
CrossEncoder download, no `OPENAI_API_KEY` required) and passes with
`OPENAI_API_KEY` unset. The one index test uses a local Chroma store with
fake embeddings.
