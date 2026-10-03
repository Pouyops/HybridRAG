"""Judge-validation scaffold -- NOT a substitute for real human validation.

`--generate` builds results/judge_validation_template.json: for each
frozen evaluation_dataset.json question, it runs the pipeline once to
get a real generated answer and asks the judge LLM for a correctness score,
then writes a row with an explicit, empty `human_label` field.

This script deliberately does NOT invent or simulate human labels -- an LLM
grading its own judge would defeat the entire point of validating the judge.
Filling in `human_label` is a manual step: a person reads each
question / expected_answer / generated_answer and scores it 0.0-1.0 (the
same scale `judge_correctness_score` uses) directly in the JSON file.

Once at least one row has a `human_label`, `--check-agreement` (or running
with no flags) computes agreement between the judge and the human labels
(mean absolute error, a "near-match" rate, and Pearson correlation). With
zero labeled rows it prints a clear message and exits cleanly -- it does not
crash and does not fabricate a result.

Usage:
    python scripts/validate_judge.py --generate       # (re)build the template
    python scripts/validate_judge.py                  # check agreement (default)
    python scripts/validate_judge.py --check-agreement
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from config import settings  # noqa: E402
from src.evaluator import RAGEvaluator  # noqa: E402
from src.pipeline import build_pipeline  # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
TEMPLATE_PATH = os.path.join(RESULTS_DIR, "judge_validation_template.json")
DATASET_PATH = os.path.join(REPO_ROOT, "evaluation_dataset.json")
PERSIST_DIR = os.path.join(REPO_ROOT, "chroma_judge_validation_db")


def generate_template(data_dir="./data/", dataset_path=DATASET_PATH):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set.")
    if not os.path.exists(dataset_path):
        raise SystemExit(
            f"{dataset_path} not found. Generate the frozen eval set first "
            "(python main.py --regenerate-eval-set)."
        )

    with open(dataset_path, "r") as f:
        cases = json.load(f)

    print(f"Building pipeline against {data_dir} ...")
    rag_pipeline = build_pipeline(
        openai_api_key=api_key, data_dir=data_dir, persist_directory=PERSIST_DIR
    )

    judge_llm = ChatOpenAI(
        model=settings.judge_model,
        temperature=settings.judge_temperature,
        api_key=api_key,
    )
    evaluator = RAGEvaluator(rag_pipeline=rag_pipeline, judge_llm=judge_llm)

    rows = []
    for i, case in enumerate(cases, start=1):
        question = case["question"]
        expected = case["expected_answer"]
        print(f"[{i}/{len(cases)}] {question}")

        response = rag_pipeline.generate_robust_answer(question)

        if response.get("status") != "Success":
            rows.append(
                {
                    "id": i,
                    "question": question,
                    "expected_answer": expected,
                    "question_type": case.get("question_type"),
                    "generated_answer": None,
                    "pipeline_status": response.get("status"),
                    "judge_correctness_score": None,
                    "judge_reasoning": response.get("reason"),
                    "human_label": None,
                }
            )
            continue

        generated_answer = response["answer"]
        score_result = evaluator.measure_correctness(question, expected, generated_answer)

        rows.append(
            {
                "id": i,
                "question": question,
                "expected_answer": expected,
                "question_type": case.get("question_type"),
                "generated_answer": generated_answer,
                "pipeline_status": "Success",
                "judge_correctness_score": score_result.score,
                "judge_reasoning": score_result.reasoning,
                # A real human fills this in by hand: a float 0.0-1.0 on the
                # same scale as judge_correctness_score. Left null on purpose.
                "human_label": None,
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TEMPLATE_PATH, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote {len(rows)} row(s) to {TEMPLATE_PATH}")
    print(
        "Next step: a human reviewer fills in `human_label` for each row "
        "(0.0-1.0, same scale as judge_correctness_score) by reading the "
        "question / expected_answer / generated_answer, then run:\n"
        "    python scripts/validate_judge.py --check-agreement"
    )


def check_agreement(template_path=TEMPLATE_PATH):
    if not os.path.exists(template_path):
        print(
            f"{template_path} does not exist yet. Run "
            "`python scripts/validate_judge.py --generate` first."
        )
        return None

    with open(template_path, "r") as f:
        rows = json.load(f)

    labeled = [
        r
        for r in rows
        if r.get("human_label") is not None and r.get("judge_correctness_score") is not None
    ]

    if not labeled:
        print(
            "No human labels yet -- fill in `human_label` for the rows in "
            f"{template_path} and re-run this script.\n"
            f"({len(rows)} row(s) total, 0 labeled. This is expected until a "
            "person does the manual labeling -- not an error.)"
        )
        return None

    n = len(labeled)
    diffs = [abs(float(r["human_label"]) - float(r["judge_correctness_score"])) for r in labeled]
    mae = sum(diffs) / n
    near_match = sum(1 for d in diffs if d <= 0.2) / n
    exact_match = sum(1 for d in diffs if d <= 1e-9) / n

    human_vals = [float(r["human_label"]) for r in labeled]
    judge_vals = [float(r["judge_correctness_score"]) for r in labeled]
    correlation = None
    if n >= 2 and len(set(human_vals)) > 1 and len(set(judge_vals)) > 1:
        mean_h = sum(human_vals) / n
        mean_j = sum(judge_vals) / n
        cov = sum((h - mean_h) * (j - mean_j) for h, j in zip(human_vals, judge_vals))
        var_h = sum((h - mean_h) ** 2 for h in human_vals)
        var_j = sum((j - mean_j) ** 2 for j in judge_vals)
        if var_h > 0 and var_j > 0:
            correlation = cov / ((var_h**0.5) * (var_j**0.5))

    print(f"Judge-vs-human agreement over {n}/{len(rows)} labeled row(s):")
    print(f"  Mean absolute error:      {mae:.3f}")
    print(f"  Near-match rate (<=0.2):  {near_match:.1%}")
    print(f"  Exact-match rate:         {exact_match:.1%}")
    if correlation is not None:
        print(f"  Pearson correlation:      {correlation:.3f}")
    else:
        print("  Pearson correlation:      n/a (insufficient variance in labeled sample)")

    if n < len(rows):
        print(f"\nNote: {len(rows) - n} row(s) in the template are still unlabeled.")

    return {
        "n_labeled": n,
        "n_total": len(rows),
        "mae": mae,
        "near_match_rate": near_match,
        "exact_match_rate": exact_match,
        "pearson_correlation": correlation,
    }


def main():
    parser = argparse.ArgumentParser(description="Judge-validation scaffold.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="(Re)build the template by running the pipeline once per frozen question.",
    )
    parser.add_argument(
        "--check-agreement",
        action="store_true",
        help="Compute judge-vs-human agreement (this is also the default action).",
    )
    parser.add_argument("--data-dir", default="./data/")
    args = parser.parse_args()

    if args.generate:
        generate_template(data_dir=args.data_dir)
    else:
        check_agreement()


if __name__ == "__main__":
    main()
