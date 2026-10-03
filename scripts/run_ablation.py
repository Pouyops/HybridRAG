"""Ablation study for the hybrid retrieval + reranking knobs.

Measures, against the FROZEN evaluation_dataset.json (see main.py
--regenerate-eval-set), how three knobs affect answer quality:

  (a) retrieval mode   -- dense-only / sparse-only / hybrid (0.7/0.3), reranker on
  (b) reranker         -- on vs off, at the default hybrid 0.7/0.3 weights
  (c) RRF weight sweep -- 0.5/0.5, 0.7/0.3 (default), 0.3/0.7, reranker on

Chunking strategy is held fixed at "TokenRecursive" throughout so it isn't a
confound (it's already compared separately by main.py's strategy comparison).

Uses OpenAI gpt-4o-mini (config.settings.generator_model / judge_model) as
both the generator and judge LLM -- same model main.py's default flow uses.

NOTE: this originally targeted OpenRouter's free tier
(nvidia/nemotron-3-super-120b-a12b:free) to avoid OpenAI spend, but that
free tier caps out at 50 requests/day, and a single (config, run) cell over
the 14-question frozen set costs ~90 calls on its own -- nowhere close to
enough for a 6-config x 3-run matrix. Switched to OpenAI after confirming
that in practice; the retry/backoff plumbing below is kept since it's
harmless and still useful for transient API errors.

Resilience:
  - Every LLM call is wrapped in a stdlib-only exponential-backoff retry loop
    (ResilientRunnable) to survive rate limits / transient API errors.
  - Results are appended to results/ablation_results.csv after EVERY
    (config, run) cell completes, not just at the end.
  - Re-running the script skips (config, run) cells already present in that
    CSV, so a rate-limit stall or a hit daily cap never loses completed work
    -- just re-run the same command to pick up where it left off.

Usage:
    python scripts/run_ablation.py --runs 3
    python scripts/run_ablation.py --runs 1          # cheaper/faster smoke test
    python scripts/run_ablation.py --charts-only      # just re-render charts
"""

import argparse
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config import settings  # noqa: E402
from src.chunker import Chunker  # noqa: E402
from src.evaluator import RAGEvaluator  # noqa: E402
from src.generator import AdvancedRAGSystem  # noqa: E402
from src.indexer import indexer  # noqa: E402
from src.loader import multiloader  # noqa: E402
from src.retriever import HybridRetriever, _RERANKER_MODEL, _load_cross_encoder  # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "ablation_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "ablation_summary.csv")
DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "evaluation_dataset.json")
INDEX_DIR = os.path.join(REPO_ROOT, "chroma_ablation_db")

CSV_COLUMNS = [
    "config_name",
    "dense_weight",
    "sparse_weight",
    "use_reranker",
    "run_idx",
    "avg_correctness",
    "avg_faithfulness",
    "avg_retrieval",
    "avg_citation_accuracy",
    "total_answered",
    "total_failed",
    "failure_rate",
    "elapsed_sec",
    "timestamp",
    "error",
]

# Six unique configurations cover all three comparisons in the task; e.g.
# "hybrid_default" (0.7/0.3, reranker on) is reused as the shared baseline
# for the retrieval-mode comparison, the reranker on/off comparison, AND the
# RRF weight sweep, so it is only ever run once per run_idx, not three times.
CONFIGS = [
    {
        "name": "dense_only",
        "dense_weight": 1.0,
        "sparse_weight": 0.0,
        "use_reranker": True,
        "groups": ["retrieval_mode"],
        "label": "Dense-only",
    },
    {
        "name": "sparse_only",
        "dense_weight": 0.0,
        "sparse_weight": 1.0,
        "use_reranker": True,
        "groups": ["retrieval_mode"],
        "label": "Sparse-only",
    },
    {
        "name": "hybrid_default",
        "dense_weight": settings.dense_weight,
        "sparse_weight": settings.sparse_weight,
        "use_reranker": True,
        "groups": ["retrieval_mode", "reranker", "rrf_sweep"],
        "label": "Hybrid 0.7/0.3 (default)",
    },
    {
        "name": "hybrid_no_reranker",
        "dense_weight": settings.dense_weight,
        "sparse_weight": settings.sparse_weight,
        "use_reranker": False,
        "groups": ["reranker"],
        "label": "Hybrid, reranker OFF",
    },
    {
        "name": "rrf_50_50",
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "use_reranker": True,
        "groups": ["rrf_sweep"],
        "label": "0.5 / 0.5",
    },
    {
        "name": "rrf_30_70",
        "dense_weight": 0.3,
        "sparse_weight": 0.7,
        "use_reranker": True,
        "groups": ["rrf_sweep"],
        "label": "0.3 / 0.7",
    },
]
CONFIG_BY_NAME = {c["name"]: c for c in CONFIGS}

try:
    import openai as _openai_mod

    _RATE_LIMIT_EXC = (_openai_mod.RateLimitError,)
except Exception:
    _RATE_LIMIT_EXC = ()


def _is_rate_limit_error(e: Exception) -> bool:
    status_code = getattr(e, "status_code", None)
    msg = str(e).lower()
    return (
        isinstance(e, _RATE_LIMIT_EXC)
        or status_code == 429
        or "429" in msg
        or "rate limit" in msg
        or "rate_limit" in msg
        or "too many requests" in msg
    )


# Observed empirically against OpenRouter's free nvidia/nemotron endpoint:
# besides plain 429s, it also intermittently returns (a) 502 "Service
# temporarily overloaded" error envelopes, and (b) a 200-status response body
# with `choices: null`, which the OpenAI SDK's parser then crashes on with a
# bare `TypeError: 'NoneType' object is not iterable` deep inside
# openai.lib._parsing._completions.parse_chat_completion. Neither carries a
# "429" or "rate limit" string, so a narrow rate-limit-only retry misses both.
# A small, explicit denylist of genuinely permanent errors is safer than
# trying to keyword-match every transient failure mode this free tier can
# produce -- so retry everything EXCEPT those.
_PERMANENT_ERROR_MARKERS = (
    "invalid_api_key",
    "incorrect api key",
    "401",
    "unauthorized",
    "invalid_request_error",
    "model_not_found",
    "does not exist",
)


def _is_permanent_error(e: Exception) -> bool:
    status_code = getattr(e, "status_code", None)
    msg = str(e).lower()
    if status_code in (401, 403, 404):
        return True
    return any(marker in msg for marker in _PERMANENT_ERROR_MARKERS)


def with_backoff(fn, max_retries=8, base_delay=3.0, max_delay=90.0, label=""):
    """Stdlib-only exponential backoff (time.sleep) for OpenRouter free-tier
    flakiness: 429 rate limits, transient 502 "overloaded" errors, and the
    choices=null parser crash described above. Deliberately NOT using
    tenacity -- that dependency was removed from this project; this loop is
    local to the ablation script. Retries everything except a small
    denylist of permanent errors (bad API key, unknown model, ...)."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - deliberately broad, re-raised below
            last_err = e
            if _is_permanent_error(e) or attempt == max_retries:
                raise
            reason = "rate limit" if _is_rate_limit_error(e) else "transient error"
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 1.5)
            print(
                f"    [{reason}] {label}: attempt {attempt}/{max_retries}, "
                f"retrying in {delay:.1f}s ({e})"
            )
            time.sleep(delay)
    raise last_err  # pragma: no cover


class ResilientRunnable:
    """Wraps a LangChain Runnable so every .invoke() retries with backoff on
    rate limits. Transparent to src/generator.py and src/evaluator.py -- they
    only ever call .invoke() / .with_structured_output(...).invoke(), both of
    which this proxies, so neither file needs to change."""

    def __init__(self, inner, label="llm"):
        self._inner = inner
        self._label = label

    def invoke(self, *args, **kwargs):
        return with_backoff(lambda: self._inner.invoke(*args, **kwargs), label=self._label)

    def with_structured_output(self, *args, **kwargs):
        return ResilientRunnable(self._inner.with_structured_output(*args, **kwargs), label=self._label)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def build_shared_index(data_dir: str, openai_api_key: str):
    """Build ONE Chroma + BM25 index (TokenRecursive chunking, fixed) shared
    across every ablation config -- only retrieval weights/reranker differ
    between configs, so there is no need to re-embed per config."""
    print("Building shared index (TokenRecursive chunking, fixed across all configs)...")
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    loader = multiloader(data_dir)
    documents = loader._document_loader()[:50]
    if not documents:
        raise SystemExit(f"No documents loaded from {data_dir!r}. Run `python scripts/fetch_corpus.py` to download the default corpus.")

    chunker = Chunker(embedding_fn=embeddings)
    chunks = []
    for doc in documents:
        chunks.extend(
            chunker.chunk_documents(doc.page_content, doc.metadata, strategy="TokenRecursive")
        )

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    idx = indexer(api_key=openai_api_key)
    vectorstore, bm25 = idx.create_indexes(chunks, persist_directory=INDEX_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} document(s).")
    return vectorstore, bm25


def load_completed(results_csv):
    """A (config, run) cell counts as done only if it SUCCEEDED (no error) --
    a row recorded after a crash must be retried on the next invocation, not
    treated as permanently skipped."""
    if not os.path.exists(results_csv):
        return set()
    df = pd.read_csv(results_csv)
    succeeded = df[df["error"].isna() | (df["error"] == "")]
    return set(zip(succeeded["config_name"], succeeded["run_idx"]))


def append_result(results_csv, row: dict):
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    df_row = pd.DataFrame([row], columns=CSV_COLUMNS)
    file_exists = os.path.exists(results_csv)
    df_row.to_csv(results_csv, mode="a", header=not file_exists, index=False)


_cross_encoder = None


def _shared_cross_encoder():
    """Load the reranker model once for the whole ablation, not once per cell."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = _load_cross_encoder(_RERANKER_MODEL)
    return _cross_encoder


def run_one_cell(config, run_idx, vectorstore, bm25, generator_llm, judge_llm, dataset_path):
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_retriever=bm25,
        dense_weight=config["dense_weight"],
        sparse_weight=config["sparse_weight"],
        use_reranker=config["use_reranker"],
        reranker=_shared_cross_encoder() if config["use_reranker"] else None,
    )
    rag_pipeline = AdvancedRAGSystem(llm=generator_llm, retriever=retriever)
    evaluator = RAGEvaluator(rag_pipeline=rag_pipeline, judge_llm=judge_llm)

    t0 = time.time()
    metrics = evaluator.run_test_suite(dataset_path)
    elapsed = time.time() - t0

    total_answered = metrics["total_runs"]
    total_failed = len(metrics["failures"])
    denom = total_answered + total_failed
    failure_rate = (total_failed / denom) if denom else float("nan")

    return {
        "config_name": config["name"],
        "dense_weight": config["dense_weight"],
        "sparse_weight": config["sparse_weight"],
        "use_reranker": config["use_reranker"],
        "run_idx": run_idx,
        "avg_correctness": metrics["avg_correctness"],
        "avg_faithfulness": metrics["avg_faithfulness"],
        "avg_retrieval": metrics["avg_retrieval"],
        "avg_citation_accuracy": metrics["avg_citation_accuracy"],
        "total_answered": total_answered,
        "total_failed": total_failed,
        "failure_rate": failure_rate,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": "",
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("config_name")
        .agg(
            mean_correctness=("avg_correctness", "mean"),
            std_correctness=("avg_correctness", "std"),
            mean_faithfulness=("avg_faithfulness", "mean"),
            std_faithfulness=("avg_faithfulness", "std"),
            mean_retrieval=("avg_retrieval", "mean"),
            std_retrieval=("avg_retrieval", "std"),
            mean_citation=("avg_citation_accuracy", "mean"),
            std_citation=("avg_citation_accuracy", "std"),
            mean_failure_rate=("failure_rate", "mean"),
            std_failure_rate=("failure_rate", "std"),
            n_runs=("run_idx", "count"),
        )
        .reset_index()
    )
    for col in agg.columns:
        if col.startswith("std_"):
            agg[col] = agg[col].fillna(0.0)
    return agg


def _bar_chart(agg, config_names, metric_prefix, ylabel, title, filename):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    missing = [c for c in config_names if c not in set(agg["config_name"])]
    if missing:
        print(f"  [skip chart] {filename}: no completed runs yet for {missing}")
        return False

    sub = agg.set_index("config_name").loc[config_names]
    labels = [CONFIG_BY_NAME[c]["label"] for c in config_names]
    means = sub[f"mean_{metric_prefix}"].values
    stds = sub[f"std_{metric_prefix}"].values
    n_runs = sub["n_runs"].values

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(config_names)]
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    for bar, m, n in zip(bars, means, n_runs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.0, m) + 0.03,
            f"{m:.3f}\n(n={int(n)})",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [chart] wrote {out_path}")
    return True


def generate_summary_and_charts():
    if not os.path.exists(RESULTS_CSV):
        print("No results yet -- nothing to summarize.")
        return

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        print("Results file is empty -- nothing to summarize.")
        return

    n_total_rows = len(df)
    df = df[df["error"].isna() | (df["error"] == "")]
    n_failed_rows = n_total_rows - len(df)
    if n_failed_rows:
        print(f"(Excluding {n_failed_rows} failed/errored row(s) from the summary -- see {RESULTS_CSV} for details.)")
    if df.empty:
        print("No successful runs yet -- nothing to summarize.")
        return

    agg = summarize(df)
    agg.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 78)
    print(" ABLATION SUMMARY (mean +/- std across completed runs per config)")
    print("=" * 78)
    display_cols = [
        "config_name",
        "n_runs",
        "mean_retrieval",
        "std_retrieval",
        "mean_correctness",
        "std_correctness",
        "mean_faithfulness",
        "std_faithfulness",
        "mean_citation",
        "std_citation",
        "mean_failure_rate",
    ]
    print(agg[display_cols].to_string(index=False))
    print(f"\nRaw per-run results: {RESULTS_CSV}")
    print(f"Aggregated summary:  {SUMMARY_CSV}")

    print("\nGenerating charts (retrieval relevance is the metric most directly")
    print("affected by these knobs)...")
    _bar_chart(
        agg,
        ["dense_only", "sparse_only", "hybrid_default"],
        "retrieval",
        "Retrieval Relevance (LLM judge, 0-1)",
        "Retrieval Mode: Dense-only vs Sparse-only vs Hybrid",
        "ablation_retrieval_mode.png",
    )
    _bar_chart(
        agg,
        ["hybrid_default", "hybrid_no_reranker"],
        "retrieval",
        "Retrieval Relevance (LLM judge, 0-1)",
        "Cross-Encoder Reranker: On vs Off",
        "ablation_reranker_on_off.png",
    )
    _bar_chart(
        agg,
        ["rrf_50_50", "hybrid_default", "rrf_30_70"],
        "retrieval",
        "Retrieval Relevance (LLM judge, 0-1)",
        "RRF Dense/Sparse Weight Sweep",
        "ablation_rrf_sweep.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Run the retrieval/reranker ablation study.")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Repetitions per configuration (default: 3). Lower this if OpenRouter's "
        "free-tier rate limits make 3 impractical in the time available.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to the frozen evaluation_dataset.json")
    parser.add_argument("--data-dir", default="./data/", help="Corpus directory to index.")
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Skip running the ablation; just (re)generate the summary + charts from the existing CSV.",
    )
    args = parser.parse_args()

    if args.charts_only:
        generate_summary_and_charts()
        return

    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY not set (needed for embeddings, generation, and judging).")
    if not os.path.exists(args.dataset):
        raise SystemExit(
            f"{args.dataset} not found. Run `python main.py --regenerate-eval-set` first, "
            "or restore the committed frozen evaluation_dataset.json."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    vectorstore, bm25 = build_shared_index(args.data_dir, openai_key)

    base_llm = ChatOpenAI(
        model=settings.generator_model,
        api_key=openai_key,
        temperature=0,
    )
    generator_llm = ResilientRunnable(base_llm, label="generator")
    judge_llm = ResilientRunnable(base_llm, label="judge")

    done = load_completed(RESULTS_CSV)
    plan = [(c, r) for c in CONFIGS for r in range(1, args.runs + 1)]
    print(f"\nPlan: {len(CONFIGS)} configs x {args.runs} runs = {len(plan)} (config, run) cells.")
    print(f"Already completed (resume): {len(done)} cell(s).\n")

    for config, run_idx in plan:
        key = (config["name"], run_idx)
        if key in done:
            print(f"[skip] {config['name']} run {run_idx}/{args.runs} -- already in {RESULTS_CSV}")
            continue

        print(
            f"\n=== {config['label']}  (dense={config['dense_weight']}, "
            f"sparse={config['sparse_weight']}, reranker={config['use_reranker']})  "
            f"run {run_idx}/{args.runs} ==="
        )
        try:
            row = run_one_cell(config, run_idx, vectorstore, bm25, generator_llm, judge_llm, args.dataset)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {config['name']} run {run_idx} crashed and was NOT completed: {e}")
            traceback.print_exc()
            row = {
                "config_name": config["name"],
                "dense_weight": config["dense_weight"],
                "sparse_weight": config["sparse_weight"],
                "use_reranker": config["use_reranker"],
                "run_idx": run_idx,
                "avg_correctness": float("nan"),
                "avg_faithfulness": float("nan"),
                "avg_retrieval": float("nan"),
                "avg_citation_accuracy": float("nan"),
                "total_answered": 0,
                "total_failed": 0,
                "failure_rate": float("nan"),
                "elapsed_sec": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)[:500],
            }

        append_result(RESULTS_CSV, row)
        print(
            f"  -> saved: correctness={row['avg_correctness']}, retrieval={row['avg_retrieval']}, "
            f"faithfulness={row['avg_faithfulness']}, citation_accuracy={row['avg_citation_accuracy']}, "
            f"failure_rate={row['failure_rate']}, elapsed={row['elapsed_sec']}s"
        )

    print("\nAll planned cells attempted (or skipped as already-done).")
    generate_summary_and_charts()


if __name__ == "__main__":
    main()
