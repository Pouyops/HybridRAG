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
