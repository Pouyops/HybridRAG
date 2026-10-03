# HybridRAG

[![CI](https://github.com/Pouyops/HybridRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Pouyops/HybridRAG/actions/workflows/ci.yml)

A retrieval-augmented generation pipeline that combines dense and BM25 retrieval, reranks with a cross-encoder, and checks its own citations before it answers. If it cannot back an answer with the retrieved text, it says so instead of guessing.

The default corpus is the IETF HTTP standards: RFC 9110 (Semantics), RFC 9111 (Caching) and RFC 9112 (HTTP/1.1), about 92,000 words in total. That makes the system an HTTP standards assistant for backend and API developers. It answers questions like "When may a cache reuse a stored response?" or "Why must a proxy strip Content-Length when Transfer-Encoding is present?" with citations to the relevant passages, and declines questions the standards don't cover.

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
python scripts/fetch_corpus.py      # downloads RFC 9110/9111/9112 into data/
```

The first run downloads the cross-encoder model (~90 MB) from Hugging Face.

### Command line

```bash
python main.py --query "What must a 304 Not Modified response include?"
```

`main.py` indexes `--data-dir` (default `./data/`), answers `--query` and prints the response. It then benchmarks all three chunking strategies against the frozen evaluation set, which makes many LLM calls; see [Evaluation](#evaluation).

| Flag | Default | Purpose |
|---|---|---|
| `--data-dir` | `./data/` | Directory of documents to index |
| `--query` | built-in caching question | Question to answer |
| `--runs N` | `1` | Repeat the strategy comparison N times and report mean ± std |
| `--regenerate-eval-set` | off | Generate a synthetic question set (`evaluation_dataset_synthetic.json`) and benchmark on that instead of the curated set |

### API service

```bash
uvicorn app:app --reload
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"query": "Which HTTP methods are idempotent?"}'
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

Run `python scripts/fetch_corpus.py` on the host first. Both containers read `.env` at runtime and mount `./data` read-only. Secrets are never baked into the image.

### As a library

```python
from src.pipeline import build_pipeline

rag = build_pipeline(openai_api_key="sk-...", data_dir="./data/")
result = rag.generate_robust_answer("How does s-maxage interact with max-age?")

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

## Corpus

`python scripts/fetch_corpus.py` downloads the HTTP Working Group's HTML rendering of the three RFCs and converts each one to Markdown. The text itself is unchanged. Numbered sections become headings, so Markdown-header chunking splits on real section boundaries, and ABNF and examples become code blocks. The table of contents and index are dropped. The files are gitignored rather than committed: the RFC text belongs to the IETF Trust, and the script reproduces it exactly.

Why this corpus? It is long (RFC 9110 alone is about 66,000 words), normative, and dense with cross-references between sections and between documents. Many real questions need two passages, and users' vocabulary often differs from the spec's ("resume a download" means `Range` plus `If-Range`). That is the situation hybrid retrieval and reranking are meant for, and a small toy corpus can't test it.

To use your own documents, point `--data-dir` or `DATA_DIR` at another directory. Every `.md`, `.txt`, `.html` and `.pdf` file there is indexed. The original Apollo 11 demo corpus is kept in [`examples/apollo11/`](examples/apollo11/).

## Evaluation

`evaluation_dataset.json` holds 32 hand-written questions:

| Type | Count | Examples |
|---|---|---|
| Lookup | 14 | Which methods are safe? What must a 405 response include? |
| Multi-Hop | 9 | Heuristic caching of a 200 response, which needs RFC 9110 §15.3.1 *and* RFC 9111 §4.2.2 |
| Unanswerable | 5 | HTTP/2 SETTINGS, WebSocket keys, 429, HSTS, QPACK: plausible but not in these RFCs |
| Ambiguous | 4 | "How long can it be cached?" |

Every answerable question cites its RFC sections and includes verbatim **evidence** passages, so retrieval can be scored exactly, without an LLM judge.

| Script | Measures | Needs |
|---|---|---|
| `scripts/eval_retrieval.py` | hit@1, hit@5, full@5 (all evidence retrieved), recall, MRR and context size, for each chunking strategy × dense/sparse/hybrid × reranker on/off | Embeddings only (BM25 + Markdown runs fully offline) |
| `main.py` | Chunking strategies compared end to end. A judge LLM scores **correctness**, **faithfulness**, **retrieval relevance** and **citation accuracy**, and the **fallback rate** is reported alongside | Generator + judge LLM |
| `scripts/run_ablation.py` | Dense vs sparse vs hybrid, reranker on/off and RRF weights, end to end. Resumable | Generator + judge LLM |
| `scripts/validate_judge.py` | How closely the judge's scores agree with human labels | Generator + judge LLM |

**Results** (full analysis, ablation charts and caveats in [`RESULTS.md`](RESULTS.md)):

| Finding | Evidence |
|---|---|
| The cross-encoder reranker *hurts* on these documents | Lowers hit@5 in all 9 chunking × retrieval-mode combinations. 32–50% of chunks exceed its 512-token window, and it was trained on short web passages |
| Multi-hop retrieval is unsolved | At most 4 of 9 multi-hop questions get all their evidence into the top 5. TokenRecursive gets 1 of 9 in every configuration |
| Best retrieval: Semantic chunks, hybrid, no reranker | hit@1 0.826, hit@5 1.000, MRR 0.913, at ~5,000 words of context per query |
| Best answers: Markdown chunking | Correctness 0.937 vs 0.902 for TokenRecursive (the default); citation accuracy 1.000 |
| The gate's real failure is letting a hallucination through, not refusing too often | Under the default setup, 0 of 23 answerable questions were refused, but one out-of-corpus question (HSTS) was answered confidently from model knowledge |

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
│   ├── fetch_corpus.py      # download + convert the RFC corpus into data/
│   ├── build_index.py       # build a persisted index from a directory
│   ├── eval_retrieval.py    # objective retrieval metrics against gold evidence
│   ├── run_ablation.py      # retrieval / reranker ablation study
│   └── validate_judge.py    # judge-vs-human agreement scaffold
├── data/                    # the indexed corpus (fetched, gitignored)
├── evaluation_dataset.json  # 32 curated questions with gold evidence
├── examples/apollo11/       # original small demo corpus + its eval set
├── results/                 # benchmark CSVs and charts (apollo11/ = archived)
├── tests/                   # offline unit tests
└── docs/                    # service docs
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover RRF fusion, reranking, BM25 tokenization, citation parsing and verification, the confidence gate, chunk metadata, index rebuilds, document loading, the RFC converter and the evaluation metrics. LLMs, embeddings and the cross-encoder are faked, so no API key or network access is needed. CI runs the suite on every push and pull request to `main`.

## Known limitations

- **Without the reranker, the retrieval-confidence term is not meaningful.** When `use_reranker=False`, raw RRF scores (about 0.01) go into a sigmoid that expects cross-encoder logits, so that term sits near 0.5 for every query. This affects the reranker-off arm of the ablation.
- **An answer with no citations gets full citation coverage.** Uncited text is not counted against coverage, so only the retrieval and completeness terms can gate an uncited answer.
- **Cost estimates are approximate.** `/metrics` counts tokens for one generation call with illustrative `gpt-4o-mini` prices. It does not include the verification and completeness calls each query also makes.
- **Markdown-header chunks are uneven on real documents.** Some RFC sections are about 4,000 words long ("9.3. Method Definitions"). Splitting only on `#`/`##`/`###` makes those single chunks, which inflates retrieval hit rates and the size of the generator's prompt.
- **The evaluation set is small.** With 23 answerable questions, one question is worth about 4 points of hit@k, so treat small differences as noise.
