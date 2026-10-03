"""Objective retrieval evaluation against the gold evidence in the frozen set.

Every answerable question in evaluation_dataset.json (Lookup and Multi-Hop)
lists one or more verbatim `evidence` passages from the RFCs. For each
chunking strategy x retrieval configuration, this script retrieves the
final_k chunks the generator would see and checks which evidence passages
they contain. No LLM is involved in scoring, so the numbers are exact and
repeatable, unlike the LLM-judged "retrieval relevance" in main.py.

Metrics (averaged over answerable questions):
    hit@1          the top chunk contains an evidence passage
    hit@k          any of the top final_k chunks does
    full@k         ALL of a question's evidence passages are in the top
                   final_k (what a multi-hop question needs)
    recall         fraction of evidence passages retrieved
    mrr            mean reciprocal rank of the first evidence-bearing chunk
    ctx_words      mean words handed to the generator (cost / dilution:
                   huge chunks trivially "contain" evidence, so compare
                   hit rates together with this)

Network needs by option:
    sparse only + Markdown chunking   fully offline
    TokenRecursive chunking           tiktoken's encoding download
    dense / hybrid / Semantic         OpenAI embeddings (OPENAI_API_KEY)
    --reranker on                     Hugging Face model download

Usage:
    python scripts/eval_retrieval.py                       # everything
    python scripts/eval_retrieval.py --strategies Markdown --modes sparse --reranker off
"""

import argparse
import itertools
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from langchain_community.retrievers import BM25Retriever  # noqa: E402

from config import settings  # noqa: E402
from src.chunker import Chunker  # noqa: E402
from src.evaluator import evidence_ranks, retrieval_scores  # noqa: E402
from src.indexer import bm25_tokenize, indexer  # noqa: E402
from src.loader import multiloader  # noqa: E402
from src.pipeline import build_retriever  # noqa: E402
from src.retriever import _RERANKER_MODEL, _load_cross_encoder  # noqa: E402

RESULTS_CSV = os.path.join(REPO_ROOT, "results", "retrieval_eval.csv")
DEFAULT_DATASET = os.path.join(REPO_ROOT, "evaluation_dataset.json")

MODES = {
    "dense": (1.0, 0.0),
    "sparse": (0.0, 1.0),
    "hybrid": (settings.dense_weight, settings.sparse_weight),
}
METRICS = ["hit@1", "hit@k", "full@k", "recall", "mrr"]


class _NoDenseIndex:
    """Stand-in vector store for sparse-only runs, so they need no embeddings."""

    def similarity_search(self, query, k):
        return []


def build_indexes(documents, strategy, need_dense, embeddings, markdown_max_tokens):
    chunker = Chunker(embedding_fn=embeddings, markdown_max_tokens=markdown_max_tokens)
    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk_documents(doc.page_content, doc.metadata, strategy=strategy))

    if need_dense:
        idx = indexer()
        persist_dir = os.path.join(REPO_ROOT, f"chroma_retrieval_eval_{strategy}")
        vectorstore, bm25 = idx.create_indexes(chunks, persist_directory=persist_dir)
    else:
        vectorstore = _NoDenseIndex()
        bm25 = BM25Retriever.from_documents(
            chunks, k=settings.retrieval_depth, preprocess_func=bm25_tokenize
        )
    return chunks, vectorstore, bm25


def evaluate(retriever, cases):
    rows = []
    for case in cases:
        retrieved = [doc for doc, _ in retriever.get_relevant_documents(case["question"])]
        scores = retrieval_scores(evidence_ranks(retrieved, case["evidence"]))
        rows.append(
            {
                "id": case["id"],
                "question_type": case["question_type"],
                "hit@1": scores["hit@1"],
                "hit@k": scores["hit@k"],
                "full@k": scores["full@k"],
                "recall": scores["evidence_recall"],
                "mrr": scores["reciprocal_rank"],
                "ctx_words": sum(len(d.page_content.split()) for d in retrieved),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Objective retrieval evaluation.")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--strategies", nargs="+", default=["TokenRecursive", "Markdown", "Semantic"],
        choices=["TokenRecursive", "Markdown", "Semantic"],
    )
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--reranker", choices=["on", "off", "both"], default="both")
    parser.add_argument(
        "--decompose", choices=["on", "off", "both"], default="both",
        help="Query decomposition (one gpt call per question, cached across configs).",
    )
    parser.add_argument(
        "--markdown-max-tokens", type=int, default=settings.markdown_max_tokens,
        help="Cap for Markdown-header chunks (0 = uncapped).",
    )
    parser.add_argument("--out", default=RESULTS_CSV)
    args = parser.parse_args()

    load_dotenv()
    with open(args.dataset, "r", encoding="utf-8") as f:
        cases = [c for c in json.load(f) if c.get("evidence")]
    documents = multiloader(args.data_dir)._document_loader()
    if not documents:
        raise SystemExit(f"No documents in {args.data_dir}. Run scripts/fetch_corpus.py first.")

    need_dense = "Semantic" in args.strategies or any(m != "sparse" for m in args.modes)
    embeddings = None
    if need_dense:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    flags = {"on": [True], "off": [False], "both": [False, True]}
    reranker_flags = flags[args.reranker]
    decompose_flags = flags[args.decompose]
    cross_encoder = _load_cross_encoder(_RERANKER_MODEL) if True in reranker_flags else None
    llm = None
    decompositions = {}  # question -> sub-questions, shared by every config
    if True in decompose_flags:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=settings.generator_model, temperature=0)

    print(f"{len(cases)} answerable questions, {len(documents)} documents in {args.data_dir}")
    summary = []
    for strategy in args.strategies:
        t0 = time.time()
        chunks, vectorstore, bm25 = build_indexes(
            documents, strategy, need_dense, embeddings, args.markdown_max_tokens
        )
        print(f"\n[{strategy}] {len(chunks)} chunks (indexed in {time.time() - t0:.1f}s)")
        for mode in args.modes:
            dense_w, sparse_w = MODES[mode]
            for use_reranker, decompose in itertools.product(reranker_flags, decompose_flags):
                # Scores don't affect these metrics, so skip confidence scoring.
                retriever = build_retriever(
                    vectorstore, bm25, llm=llm, decompose=decompose,
                    dense_weight=dense_w, sparse_weight=sparse_w,
                    use_reranker=use_reranker, reranker=cross_encoder,
                    cross_encoder_confidence=False,
                )
                if decompose:
                    plain = retriever.decompose
                    retriever.decompose = lambda q, plain=plain: decompositions.setdefault(q, plain(q))
                per_q = evaluate(retriever, cases)
                row = {
                    "strategy": strategy,
                    "markdown_max_tokens": args.markdown_max_tokens if strategy == "Markdown" else "",
                    "mode": mode,
                    "reranker": "on" if use_reranker else "off",
                    "decompose": "on" if decompose else "off",
                    "n_chunks": len(chunks),
                    **{m: per_q[m].mean() for m in METRICS},
                    "full@k_multihop": per_q[per_q["question_type"] == "Multi-Hop"]["full@k"].mean(),
                    "ctx_words": per_q["ctx_words"].mean(),
                    "missed": " ".join(per_q[per_q["hit@k"] == 0]["id"]),
                }
                summary.append(row)
                print(
                    f"  {mode:<6} reranker={row['reranker']:<3} decompose={row['decompose']:<3}  "
                    + "  ".join(f"{m}={row[m]:.3f}" for m in METRICS)
                    + f"  ctx_words={row['ctx_words']:.0f}  missed=[{row['missed']}]"
                )

    df = pd.DataFrame(summary)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out} (final_k={settings.final_k}).")


if __name__ == "__main__":
    main()
