# HybridRAG

[![CI](https://github.com/Pouyops/HybridRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Pouyops/HybridRAG/actions/workflows/ci.yml)

A retrieval-augmented generation pipeline that combines dense and BM25 retrieval, reranks with a cross-encoder, and checks its own citations before it answers. If it cannot back an answer with the retrieved text, it says so instead of guessing.

It runs as a CLI, a FastAPI service, or a Streamlit app, and includes an evaluation harness (LLM-as-a-judge, a frozen test set, repeated runs, and an ablation study).

![RAG pipeline flow graph](assets/flowgraph.png)

<sub>The diagram was drawn before the switch from Google to OpenAI embeddings. The dense index now uses OpenAI `text-embedding-3-small`.</sub>

## How it works

| Stage | Module | What happens |
|---|---|---|
| Load | `src/loader.py` | Reads `.pdf`, `.html`/`.htm`, `.md` and `.txt` files from a directory (in sorted order) and normalizes whitespace while keeping line breaks. |
| Chunk | `src/chunker.py` | Splits each document with one of three strategies: `TokenRecursive` (512 tokens, 50 overlap; the default), `Markdown` (by `#`/`##`/`###` headers) or `Semantic` (embedding-distance breakpoints). |
| Index | `src/indexer.py` | Embeds chunks into a persisted Chroma collection (`text-embedding-3-small`) and builds a BM25 index over the same chunks. Rebuilding into an existing directory replaces the collection, so the index never accumulates duplicates. |
| Retrieve | `src/retriever.py` | Takes the top 60 hits from each index, fuses them with weighted Reciprocal Rank Fusion, `score = Σ wᵢ / (k + rankᵢ)` (dense 0.7, sparse 0.3, k = 60), keeps the top 20, and reranks those with `cross-encoder/ms-marco-MiniLM-L-6-v2` down to 5. |
| Generate | `src/generator.py` | Asks the LLM (`gpt-4o-mini`) to answer only from the numbered context blocks and to cite them as `[1]`, `[1][2]` or `[1, 2]`. |
| Verify | `src/generator.py` | Splits the answer into cited claims and has a judge LLM check each claim against the text of the chunks it cites. |
| Gate | `src/generator.py` | Computes `0.3 × retrieval + 0.4 × citation coverage + 0.3 × completeness`. Retrieval is the mean sigmoid of the top-3 cross-encoder scores; completeness is LLM-rated and clamped to 0–1. If the result is below 0.75, the pipeline returns an "Insufficient Information" response that lists the most relevant source files instead of an answer. |

`src/pipeline.py` wires these stages together. `app.py`, `streamlit_app.py` and `scripts/validate_judge.py` all use it.

## Quick start

Requires Python 3.11 (the version used by CI and the Docker image) and an OpenAI API key.

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

The first run downloads the cross-encoder model (~90 MB) from Hugging Face.

### Command line

```bash
python main.py --query "Who was the commander of Apollo 11?"
```

`main.py` indexes `--data-dir` (default `./data/`), answers `--query` and prints the response. It then benchmarks all three chunking strategies against the frozen evaluation set, which makes many LLM calls; see [Evaluation](#evaluation).

| Flag | Default | Purpose |
|---|---|---|
| `--data-dir` | `./data/` | Directory of documents to index |
| `--query` | built-in Apollo 11 question | Question to answer |
| `--runs N` | `1` | Repeat the strategy comparison N times and report mean ± std |
| `--regenerate-eval-set` | off | Rebuild `evaluation_dataset.json` instead of reusing the committed one |

### API service

```bash
uvicorn app:app --reload
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"query": "Who stayed in lunar orbit during Apollo 11?"}'
```

| Endpoint | Returns |
|---|---|
| `POST /query` | `answer`, `status`, `confidence` (the score breakdown), `citations` (claims the verifier flagged as unsupported), `latency_ms` |
| `GET /health` | `{"status": "ok"}` |
| `GET /metrics` | Request count, median latency, and average estimated cost for this process |

The index is built once at startup. Set `DATA_DIR` to index a different directory.

### Streamlit demo

```bash
streamlit run streamlit_app.py
```

This shows the answer, the confidence breakdown, any flagged citations, and the retrieved chunks.

### Docker

```bash
docker compose up --build   # API on :8000, Streamlit on :8501
```

Both containers read `.env` at runtime and mount `./data` read-only. Secrets are never baked into the image.

### As a library

```python
from src.pipeline import build_pipeline

rag = build_pipeline(openai_api_key="sk-...", data_dir="./data/")
result = rag.generate_robust_answer("When did Apollo 11 land?")

if result["status"] == "Success":
    print(result["answer"])               # answer text with [n] citations
    print(result["confidence_metrics"])   # retrieval / coverage / completeness / composite
    print(result["flagged_citations"])    # claims the verifier rejected
else:                                     # "Insufficient Information"
    print(result["reason"], result.get("suggested_documents"))
```

## Configuration

Every tunable parameter lives in [`config.py`](config.py), a `pydantic-settings` object. You can override any of them with an environment variable or a line in `.env`:

```bash
DENSE_WEIGHT=0.5 SPARSE_WEIGHT=0.5 FINAL_K=8 uvicorn app:app
```

| Group | Settings (defaults) |
|---|---|
| Retrieval | `dense_weight` 0.7, `sparse_weight` 0.3, `retrieval_depth` 60, `rrf_k` 60, `top_n` 20, `final_k` 5 |
| Gate | `confidence_threshold` 0.75 |
| Chunking | `chunk_size` 512, `chunk_overlap` 50 |
| Models | `generator_model` / `judge_model` `gpt-4o-mini`, `embedding_model` `text-embedding-3-small` |
| Temperatures | `generator_temperature` 0, `judge_temperature` 0, `synthetic_temperature` 0.7 |

[`docs/SERVICE.md`](docs/SERVICE.md) explains each setting and covers the service layer in more detail.

## Demo corpus

`data/` holds six short, original Markdown files about the Apollo 11 mission. The files reference each other, so some questions need facts from two of them. [`docs/CORPUS.md`](docs/CORPUS.md) describes each file. The description is kept outside `data/` because every file in that directory gets indexed. To use your own documents, point `--data-dir` or `DATA_DIR` at another directory.

## Evaluation

- `main.py` compares the three chunking strategies. A judge LLM scores each answer for **correctness**, **faithfulness**, **retrieval relevance** and **citation accuracy**. Each claim is re-judged independently, without reusing the generator's own verification. The **fallback rate** is the share of questions that ended in "Insufficient Information".
- `scripts/run_ablation.py` compares dense-only, sparse-only and hybrid retrieval, reranker on and off, and three RRF weightings. It saves each result as soon as it finishes, so you can resume an interrupted run with the same command.
- `scripts/validate_judge.py` writes a template for hand-labelling answers and measures how closely the judge agrees with the human labels.

All three scripts use the committed, frozen `evaluation_dataset.json`: 14 synthetic questions (6 lookup, 4 multi-hop, 2 unanswerable, 2 ambiguous). Results from different runs are comparable because the questions never change.

Most recent recorded results (3 runs, `gpt-4o-mini` as generator and judge):

| Strategy | Correctness | Faithfulness | Retrieval relevance | Citation accuracy | Fallback rate |
|---|---|---|---|---|---|
| TokenRecursive | 0.997 ± 0.005 | 1.000 ± 0.000 | 0.983 ± 0.005 | 0.975 ± 0.000 | 0.286 ± 0.000 |
| Markdown\* | 0.997 ± 0.005 | 1.000 ± 0.000 | 0.980 ± 0.000 | 0.978 ± 0.002 | 0.286 ± 0.000 |
| Semantic | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.979 ± 0.008 | 0.964 ± 0.030 | 0.310 ± 0.034 |

A fallback rate of 0.286 is 4 of 14 questions: exactly the 4 unanswerable and ambiguous questions that are meant to fall back. The other metrics only cover questions that were answered.

\* These numbers predate several bug fixes. At the time, the loader flattened newlines, so the "Markdown" row actually measured one chunk per file, and the corpus description file was indexed alongside the data. The fallback rates have been recomputed with the correct denominator. [`RESULTS.md`](RESULTS.md) has the corrections, the full ablation study with charts, and the caveats. To refresh every number, run `python main.py --runs 3` and `python scripts/run_ablation.py --runs 3`.

## Project layout

```
├── main.py                  # CLI: single query + chunking-strategy benchmark
├── app.py                   # FastAPI service
├── streamlit_app.py         # Streamlit demo
├── config.py                # all tunable settings
├── src/
│   ├── loader.py            # multi-format document loading
│   ├── chunker.py           # TokenRecursive / Markdown / Semantic chunking
│   ├── indexer.py           # Chroma + BM25 index construction
│   ├── retriever.py         # RRF fusion + cross-encoder reranking
│   ├── generator.py         # cited generation, verification, confidence gate
│   ├── evaluator.py         # synthetic QA generation + LLM-as-a-judge metrics
│   └── pipeline.py          # build_pipeline() factory
├── scripts/
│   ├── build_index.py       # build a persisted index from a directory
│   ├── run_ablation.py      # retrieval / reranker ablation study
│   └── validate_judge.py    # judge-vs-human agreement scaffold
├── data/                    # demo corpus (everything here is indexed)
├── results/                 # benchmark CSVs and charts
├── tests/                   # offline unit tests
└── docs/                    # service docs and corpus description
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover RRF fusion, reranking, citation parsing and verification, the confidence gate, chunk metadata, index rebuilds, document loading and the evaluation metrics. LLMs, embeddings and the cross-encoder are faked, so no API key or network access is needed. CI runs the suite on every push and pull request to `main`.

## Known limitations

- **Without the reranker, the retrieval-confidence term is not meaningful.** When `use_reranker=False`, raw RRF scores (about 0.01) go into a sigmoid that expects cross-encoder logits, so that term sits near 0.5 for every query. This affects the reranker-off arm of the ablation; see RESULTS.md §2b.
- **An answer with no citations gets full citation coverage.** Uncited text is not counted against coverage, so only the retrieval and completeness terms can gate an uncited answer.
- **Cost estimates are approximate.** `/metrics` counts tokens for one generation call with illustrative `gpt-4o-mini` prices. It does not include the verification and completeness calls each query also makes.
- **The demo corpus is small.** With a few dozen chunks at most, retrieval mode and fusion weights barely change the results. The ablation needs a larger, more varied corpus to show real differences.
