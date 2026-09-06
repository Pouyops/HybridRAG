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
        {"claim": "claim one", "chunk_id": 1},
        {"claim": "claim two", "chunk_id": 1},
    ]

    evaluator = _make_evaluator(
        rag_pipeline=None,
        judge_responses=[
            CitationVerification(
                claim="claim one", cited_chunk_id=1, is_supported=True, reasoning="ok"
            ),
            CitationVerification(
                claim="claim two",
                cited_chunk_id=1,
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
        return [{"claim": "the answer", "chunk_id": 1}]


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
                claim="the answer", cited_chunk_id=1, is_supported=True, reasoning="ok"
            ),
        ],
    )

    metrics = evaluator.run_test_suite(str(dataset_path))

    assert metrics["total_runs"] == 1
    assert metrics["avg_citation_accuracy"] == 1.0
