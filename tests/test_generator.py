import math

from langchain_core.documents import Document

from src.generator import AdvancedRAGSystem, CitationVerification


def _make_system():
    """Build an AdvancedRAGSystem without running __init__, so no real LLM
    or retriever needs to be constructed."""
    return object.__new__(AdvancedRAGSystem)


def test_parse_citations_splits_claims_by_bracketed_reference():
    system = _make_system()
    answer = "The sky is blue [1]. Grass is green [2]."

    claims = system.parse_citations(answer)

    assert claims == [
        {"claim": "The sky is blue", "chunk_id": 1},
        {"claim": ". Grass is green", "chunk_id": 2},
    ]


def test_parse_citations_ignores_text_without_a_citation():
    system = _make_system()
    answer = "This has no citation at all."

    claims = system.parse_citations(answer)

    assert claims == []


def test_verify_citations_computes_coverage_from_judge_results():
    system = _make_system()
    chunks = [Document(page_content="context A", metadata={})]
    claims = [
        {"claim": "claim one", "chunk_id": 1},
        {"claim": "claim two", "chunk_id": 1},
    ]

    results = iter(
        [
            CitationVerification(
                claim="claim one", cited_chunk_id=1, is_supported=True, reasoning="ok"
            ),
            CitationVerification(
                claim="claim two",
                cited_chunk_id=1,
                is_supported=False,
                reasoning="not supported",
            ),
        ]
    )

    class FakeJudgeLLM:
        def invoke(self, prompt):
            return next(results)

    system.judge_llm = FakeJudgeLLM()

    verification = system.verify_citations(claims, chunks)

    assert verification.coverage_percentage == 0.5
    assert len(verification.verifications) == 2


def test_verify_citations_defaults_to_full_coverage_with_no_claims():
    system = _make_system()
    system.judge_llm = None

    verification = system.verify_citations([], retrieved_chunks=[])

    assert verification.coverage_percentage == 1.0
    assert verification.verifications == []


def test_score_confidence_falls_back_when_llm_output_is_not_a_float():
    system = _make_system()
    system.confidence_threshold = 0.75

    class FakeLLM:
        def invoke(self, prompt):
            class Response:
                content = "not a number"

            return Response()

    system.llm = FakeLLM()

    score = system.score_confidence(
        query="q", answer="a", retrieved_chunks=[], coverage=1.0
    )

    assert score.answer_completeness == 0.5


def test_score_confidence_derives_retrieval_confidence_from_rerank_scores():
    """retrieval_confidence must come from the cross-encoder's own scores,
    not a hardcoded constant — different rerank scores must yield different
    retrieval_confidence values."""
    system = _make_system()
    system.confidence_threshold = 0.75

    class FakeLLM:
        def invoke(self, prompt):
            class Response:
                content = "1.0"

            return Response()

    system.llm = FakeLLM()

    strong = system.score_confidence(
        query="q", answer="a", retrieved_chunks=[], coverage=1.0, rerank_scores=[8.0, 7.0]
    )
    weak = system.score_confidence(
        query="q", answer="a", retrieved_chunks=[], coverage=1.0, rerank_scores=[-8.0, -7.0]
    )

    expected_strong = (1 / (1 + math.exp(-8.0)) + 1 / (1 + math.exp(-7.0))) / 2
    assert math.isclose(strong.retrieval_confidence, expected_strong, rel_tol=1e-9)
    assert strong.retrieval_confidence > weak.retrieval_confidence


def test_score_confidence_retrieval_confidence_is_zero_with_no_scores():
    system = _make_system()
    system.confidence_threshold = 0.75

    class FakeLLM:
        def invoke(self, prompt):
            class Response:
                content = "1.0"

            return Response()

    system.llm = FakeLLM()

    score = system.score_confidence(
        query="q", answer="a", retrieved_chunks=[], coverage=1.0, rerank_scores=[]
    )

    assert score.retrieval_confidence == 0.0


def test_generate_robust_answer_returns_retrieved_chunks_for_reuse():
    """The evaluator must be able to reuse the exact chunks the answer was
    generated from, instead of retrieving a second time."""
    system = _make_system()
    system.confidence_threshold = 0.0  # always confident, to reach the Success path

    chunk_a = Document(page_content="context A", metadata={})
    chunk_b = Document(page_content="context B", metadata={})

    class FakeRetriever:
        def get_relevant_documents(self, query):
            return [(chunk_a, 5.0), (chunk_b, 4.0)]

    class FakeLLM:
        def invoke(self, prompt):
            class Response:
                content = "The answer [1]."

            return Response()

    class FakeJudgeLLM:
        def invoke(self, prompt):
            return CitationVerification(
                claim="The answer",
                cited_chunk_id=1,
                is_supported=True,
                reasoning="ok",
            )

    system.retriever = FakeRetriever()
    system.llm = FakeLLM()
    system.judge_llm = FakeJudgeLLM()

    response = system.generate_robust_answer("q")

    assert response["status"] == "Success"
    assert response["retrieved_chunks"] == [chunk_a, chunk_b]
