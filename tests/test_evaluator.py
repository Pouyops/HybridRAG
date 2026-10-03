from langchain_core.documents import Document

from src.evaluator import RAGEvaluator
from src.generator import CitationVerification


class FakeStructuredLLM:
    """Mimics `llm.with_structured_output(schema)` returning itself, so tests
    can control what `.invoke()` returns without a real LLM."""

    def __init__(self, responses):
        self._responses = iter(responses)

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, prompt):
        return next(self._responses)


def _make_evaluator(rag_pipeline, judge_responses):
    judge_llm = FakeStructuredLLM(judge_responses)
    return RAGEvaluator(rag_pipeline=rag_pipeline, judge_llm=judge_llm)


def test_measure_citation_accuracy_uses_independent_judge_not_generator_self_grade():
    chunks = [Document(page_content="context A", metadata={})]
    claims = [
        {"claim": "claim one", "chunk_ids": [1]},
        {"claim": "claim two", "chunk_ids": [1]},
    ]

    evaluator = _make_evaluator(
        rag_pipeline=None,
        judge_responses=[
            CitationVerification(
                claim="claim one", cited_chunk_ids=[1], is_supported=True, reasoning="ok"
            ),
            CitationVerification(
                claim="claim two",
                cited_chunk_ids=[1],
                is_supported=False,
                reasoning="not supported",
            ),
        ],
    )

    accuracy = evaluator.measure_citation_accuracy(claims, chunks)

    assert accuracy == 0.5


def test_measure_citation_accuracy_defaults_to_full_score_with_no_claims():
    evaluator = _make_evaluator(rag_pipeline=None, judge_responses=[])

    assert evaluator.measure_citation_accuracy([], retrieved_chunks=[]) == 1.0


class ExplodingRetriever:
    """Fails the test if the evaluator retrieves a second time instead of
    reusing the chunks generate_robust_answer already returned."""

    def get_relevant_documents(self, query):
        raise AssertionError(
            "run_test_suite must reuse retrieved_chunks, not retrieve again"
        )


class FakeRagPipeline:
    def __init__(self, response):
        self._response = response
        self.retriever = ExplodingRetriever()

    def generate_robust_answer(self, query):
        return self._response

    def parse_citations(self, answer):
        return [{"claim": "the answer", "chunk_ids": [1]}]


def test_run_test_suite_reuses_retrieved_chunks_without_retrieving_again(tmp_path):
    chunk = Document(page_content="context A", metadata={})
    rag_pipeline = FakeRagPipeline(
        response={
            "status": "Success",
            "answer": "the answer [1]",
            "retrieved_chunks": [chunk],
        }
    )

    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(
        '[{"question": "q?", "expected_answer": "expected"}]', encoding="utf-8"
    )

    class Score:
        def __init__(self, score):
            self.score = score

    evaluator = _make_evaluator(
        rag_pipeline=rag_pipeline,
        judge_responses=[
            Score(1.0),  # correctness
            Score(1.0),  # faithfulness
            Score(1.0),  # retrieval relevance
            CitationVerification(
                claim="the answer", cited_chunk_ids=[1], is_supported=True, reasoning="ok"
            ),
        ],
    )

    metrics = evaluator.run_test_suite(str(dataset_path))

    assert metrics["total_runs"] == 1
    assert metrics["avg_citation_accuracy"] == 1.0


def test_run_test_suite_fallback_rate_is_over_all_questions(tmp_path):
    """Regression test: the fallback rate was failures / answered instead of
    failures / asked, inflating e.g. 4-of-14 (0.286) to 4/10 (0.400)."""
    responses = iter(
        [
            {"status": "Success", "answer": "a [1]", "retrieved_chunks": [Document(page_content="c", metadata={})]},
            {"status": "Insufficient Information", "reason": "low confidence"},
            {"status": "Insufficient Information", "reason": "low confidence"},
            {"status": "Success", "answer": "a [1]", "retrieved_chunks": [Document(page_content="c", metadata={})]},
        ]
    )

    class Pipeline:
        def generate_robust_answer(self, query):
            return next(responses)

        def parse_citations(self, answer):
            return []

    class Score:
        score = 1.0

    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(
        '[' + ",".join('{"question": "q", "expected_answer": "e"}' for _ in range(4)) + ']',
        encoding="utf-8",
    )
    evaluator = _make_evaluator(Pipeline(), judge_responses=[Score()] * 6)

    metrics = evaluator.run_test_suite(str(dataset_path))

    assert metrics["total_runs"] == 2
    assert metrics["fallback_rate"] == 0.5


def test_split_counts_sums_to_requested_total():
    from src.evaluator import _split_counts

    counts = _split_counts(15, {"lookup": 0.4, "multihop": 0.3, "unans": 0.15, "ambig": 0.15})

    assert counts == {"lookup": 6, "multihop": 5, "unans": 2, "ambig": 2}
    assert _split_counts(50, {"a": 0.4, "b": 0.3, "c": 0.15, "d": 0.15}) == {
        "a": 20,
        "b": 15,
        "c": 8,
        "d": 7,
    }


def test_evidence_ranks_finds_first_chunk_containing_each_snippet():
    from src.evaluator import evidence_ranks

    chunks = [
        Document(page_content="Nothing relevant here.", metadata={}),
        Document(page_content="The GET, HEAD,\n  OPTIONS, and TRACE methods are SAFE.", metadata={}),
        Document(page_content="the get, head, options, and trace methods again", metadata={}),
    ]

    ranks = evidence_ranks(
        chunks, ["GET, HEAD, OPTIONS, and TRACE methods", "not in any chunk"]
    )

    # Matching ignores case and whitespace differences; the first hit wins.
    assert ranks == [2, None]


def test_retrieval_scores_distinguishes_partial_and_full_evidence():
    from src.evaluator import retrieval_scores

    partial = retrieval_scores([3, None])
    assert partial == {
        "hit@1": 0.0,
        "hit@k": 1.0,
        "full@k": 0.0,
        "evidence_recall": 0.5,
        "reciprocal_rank": 1 / 3,
    }

    assert retrieval_scores([1, 2])["full@k"] == 1.0
    assert retrieval_scores([1, 2])["hit@1"] == 1.0
    assert retrieval_scores([None])["reciprocal_rank"] == 0.0
