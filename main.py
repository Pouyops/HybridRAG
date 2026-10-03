import argparse
import logging
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings
from src.chunker import Chunker
from src.evaluator import RAGEvaluator, SyntheticEvaluator
from src.generator import AdvancedRAGSystem
from src.indexer import indexer
from src.loader import multiloader
from src.retriever import HybridRetriever

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _rmtree_with_retry(path, max_retries=5, delay=1.0):
    """shutil.rmtree with a short retry loop for Windows file locking.

    Chroma's persistent client can keep the previous run's SQLite/HNSW file
    handles open for a moment after the Python object holding them goes out
    of scope (garbage collection timing, not an immediate close), so a
    same-process rmtree of the same path -- which only happens when
    `--runs N > 1` reuses a strategy's db_path across runs -- can hit a
    transient PermissionError ([WinError 32]) on Windows even though nothing
    is genuinely still using the directory. A few short retries clears it
    without needing to reach into chromadb's internals to force a close.
    """
    for attempt in range(1, max_retries + 1):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == max_retries:
                raise
            time.sleep(delay)


RESULTS_DIR = "./results"
STRATEGY_CSV = os.path.join(RESULTS_DIR, "strategy_comparison_results.csv")
STRATEGY_CSV_COLUMNS = [
    "strategy",
    "run_idx",
    "Correctness",
    "Faithfulness",
    "Retrieval Relevance",
    "Citation Accuracy",
    "Fallback Rate",
]


def _load_completed_strategy_cells(csv_path=STRATEGY_CSV):
    """(strategy, run_idx) pairs already saved to disk -- lets a killed/crashed
    `--runs N` invocation resume instead of restarting from run 1, the same
    resumability pattern scripts/run_ablation.py uses."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["strategy"], df["run_idx"]))


def _append_strategy_result(row, csv_path=STRATEGY_CSV):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_row = pd.DataFrame([row], columns=STRATEGY_CSV_COLUMNS)
    file_exists = os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)


def run_strategy_comparison_once(gitlab_documents, dataset_path, llm, judge_llm, api_key, run_idx=1):
    """Run the chunking-strategy comparison a single time, saving each
    strategy's row to STRATEGY_CSV immediately after it completes (not just
    at the end) and skipping any (strategy, run_idx) already present there,
    so a killed process can be resumed with the same command.

    `run_idx` also makes each repeated call (from --runs N) use its own
    persist_directory per strategy instead of reusing/deleting the same one,
    which sidesteps Windows file-locking on Chroma's persisted index files
    (see _rmtree_with_retry) rather than relying on retries alone.
    """
    strategies = ["TokenRecursive", "Markdown", "Semantic"]
    done = _load_completed_strategy_cells()

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    for strategy in strategies:
        if (strategy, run_idx) in done:
            print(f"  [skip] {strategy} run {run_idx} -- already in {STRATEGY_CSV}")
            continue

        db_path = f"./chroma_db_{strategy}_run{run_idx}"
        if os.path.exists(db_path):
            _rmtree_with_retry(db_path)

        chunker = Chunker(embedding_fn=embeddings)
        all_chunks = []
        for doc in gitlab_documents:
            chunks = chunker.chunk_documents(
                doc.page_content, doc.metadata, strategy=strategy
            )
            all_chunks.extend(chunks)

        idx = indexer(api_key=api_key)
        vectorstore, bm25 = idx.create_indexes(all_chunks, persist_directory=db_path)

        retriever = HybridRetriever(vectorstore=vectorstore, bm25_retriever=bm25)
        rag_pipeline = AdvancedRAGSystem(llm=llm, retriever=retriever)
        evaluator = RAGEvaluator(rag_pipeline=rag_pipeline, judge_llm=judge_llm)

        metrics = evaluator.run_test_suite(dataset_path)

        row = {
            "strategy": strategy,
            "run_idx": run_idx,
            "Correctness": metrics["avg_correctness"],
            "Faithfulness": metrics["avg_faithfulness"],
            "Retrieval Relevance": metrics["avg_retrieval"],
            "Citation Accuracy": metrics["avg_citation_accuracy"],
            "Fallback Rate": metrics["fallback_rate"],
        }
        _append_strategy_result(row)
        print(f"  -> saved: {strategy} run {run_idx}: {row}")


_METRIC_COLUMNS = [
    "Correctness",
    "Faithfulness",
    "Retrieval Relevance",
    "Citation Accuracy",
    "Fallback Rate",
]


def run_strategy_comparison(
    gitlab_documents, dataset_path, llm, judge_llm, api_key, runs=1
):
    """Run the chunking-strategy comparison `runs` times (resuming any cells
    already saved to STRATEGY_CSV from a prior, interrupted invocation) and
    report mean +/- std per metric per strategy (std is 0 for a single run).
    Returns the formatted report DataFrame (strings like "0.907 +/- 0.021").
    """
    strategies = ["TokenRecursive", "Markdown", "Semantic"]

    for run_idx in range(1, runs + 1):
        print(f"\n--- Strategy comparison run {run_idx}/{runs} ---")
        run_strategy_comparison_once(
            gitlab_documents, dataset_path, llm, judge_llm, api_key, run_idx=run_idx
        )

    # Always recompute the aggregate from disk (not from in-memory results),
    # so a resumed run correctly includes cells saved by an earlier, killed
    # invocation of this same command.
    all_results = pd.read_csv(STRATEGY_CSV)
    all_results = all_results[all_results["run_idx"] <= runs]
    raw_by_strategy = {
        s: {m: all_results[all_results["strategy"] == s][m].tolist() for m in _METRIC_COLUMNS}
        for s in strategies
    }

    report_rows = []
    mean_rows = []  # numeric means, used to pick winners
    for s in strategies:
        formatted = {"Chunking Strategy": s}
        means = {"Chunking Strategy": s}
        for m in _METRIC_COLUMNS:
            values = np.array(raw_by_strategy[s][m], dtype=float)
            mean = values.mean()
            std = values.std() if len(values) > 1 else 0.0
            col_name = f"{m} (mean±std)" if runs > 1 else m
            formatted[col_name] = f"{mean:.3f} ± {std:.3f}" if runs > 1 else round(mean, 4)
            means[m] = mean
        report_rows.append(formatted)
        mean_rows.append(means)

    report_df = pd.DataFrame(report_rows)
    means_df = pd.DataFrame(mean_rows)

    print("\n" + "=" * 70)
    print(f" CHUNKING STRATEGY EVALUATION REPORT ({runs} run{'s' if runs != 1 else ''})")
    print("=" * 70)
    print(report_df.to_string(index=False))

    best_retrieval = means_df.loc[means_df["Retrieval Relevance"].idxmax()][
        "Chunking Strategy"
    ]
    best_citation = means_df.loc[means_df["Citation Accuracy"].idxmax()][
        "Chunking Strategy"
    ]

    print("\n" + "-" * 70)
    print(f"Winner - Retrieval Relevance (mean): {best_retrieval}")
    print(f"Winner - Citation Accuracy (mean): {best_citation}")

    return report_df


DEFAULT_QUERY = (
    "Who was the commander of Apollo 11, and what did he do during the final "
    "phase of the lunar landing?"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Hybrid RAG pipeline.")
    parser.add_argument(
        "--data-dir",
        default="./data/",
        help="Directory containing the documents to index (default: ./data/).",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Question to ask the RAG system (default: the built-in sample query).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Number of times to repeat the chunking-strategy comparison; "
            "reports mean +/- std per metric per strategy instead of a "
            "single number (default: 1)."
        ),
    )
    parser.add_argument(
        "--regenerate-eval-set",
        action="store_true",
        help=(
            "Regenerate evaluation_dataset.json via SyntheticEvaluator before "
            "running the strategy comparison. By default the committed, "
            "frozen evaluation_dataset.json is reused as-is so results stay "
            "comparable run over run. Chunk selection is seeded "
            "(random.seed(42)) for reproducibility, but the LLM's phrasing "
            "of each question/answer is still not perfectly deterministic "
            "even at temperature 0 -- that's an OpenAI API property, not "
            "something this flag controls."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    generator_llm = ChatOpenAI(
        model=settings.generator_model,
        temperature=settings.generator_temperature,
        api_key=OPENAI_API_KEY,
    )

    judge_llm = ChatOpenAI(
        model=settings.judge_model,
        temperature=settings.judge_temperature,
        api_key=OPENAI_API_KEY,
    )

    dataset_path = args.data_dir
    loader = multiloader(dataset_path)
    all_documents = loader._document_loader()
    subset_all_documents = all_documents[:50]

    if not subset_all_documents:
        print(f"No documents loaded. Please add files to the {dataset_path} directory.")
    else:
        print("--- Testing Single Query Execution ---")
        chunker = Chunker(embedding_fn=None)
        all_chunks = []
        for doc in subset_all_documents:
            chunks = chunker.chunk_documents(
                doc.page_content, doc.metadata, strategy="TokenRecursive"
            )
            all_chunks.extend(chunks)

        idx = indexer(api_key=OPENAI_API_KEY)
        vectorstore, bm25 = idx.create_indexes(
            all_chunks, persist_directory="./chroma_main_db"
        )

        retriever = HybridRetriever(vectorstore=vectorstore, bm25_retriever=bm25)
        rag_system = AdvancedRAGSystem(llm=generator_llm, retriever=retriever)

        response = rag_system.generate_robust_answer(args.query)
        print(response)

        dataset_path = "evaluation_dataset.json"
        if args.regenerate_eval_set:
            print("\n--- Regenerating Synthetic Evaluation Dataset ---")
            # Seeds which chunks get sampled for question generation, so a
            # future regeneration selects the same source chunks. The LLM's
            # phrasing of the question/answer text is NOT pinned by this --
            # that depends on OpenAI API determinism, which is outside our
            # control even at temperature 0.
            random.seed(42)
            synthetic_evaluator = SyntheticEvaluator(vectorstore=vectorstore, llm=judge_llm)
            synthetic_evaluator.build_dataset(total_questions=15)
            print(f"Dataset generated and saved to {dataset_path}")
        elif not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"{dataset_path} not found. Run with --regenerate-eval-set to "
                "generate it, or restore the committed frozen version."
            )
        else:
            print(f"\n--- Reusing frozen {dataset_path} (pass --regenerate-eval-set to refresh it) ---")

        print(f"\n--- Running Strategy Comparison ({args.runs} run(s)) ---")
        comparison_report = run_strategy_comparison(
            gitlab_documents=subset_all_documents,
            dataset_path=dataset_path,
            llm=generator_llm,
            judge_llm=judge_llm,
            api_key=OPENAI_API_KEY,
            runs=args.runs,
        )
