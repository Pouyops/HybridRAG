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
        {"claim": "The sky is blue", "chunk_ids": [1]},
        {"claim": ". Grass is green", "chunk_ids": [2]},
    ]


def test_parse_citations_ignores_text_without_a_citation():
    system = _make_system()
    answer = "This has no citation at all."

    claims = system.parse_citations(answer)

    assert claims == []


def test_parse_citations_attaches_back_to_back_citations_to_the_same_claim():
    """Regression test: a claim like "...the Moon [1][2]." must attach BOTH
    chunk ids to the one claim, not silently drop [2] because there's no
    text between the two brackets to anchor a second claim to."""
    system = _make_system()
    answer = "Apollo 11 launched and landed in 1969 [1][2]."

    claims = system.parse_citations(answer)

    assert claims == [
        {"claim": "Apollo 11 launched and landed in 1969", "chunk_ids": [1, 2]},
    ]


def test_parse_citations_three_back_to_back_citations():
    system = _make_system()
    answer = "A compound claim [1][2][3]."

    claims = system.parse_citations(answer)

    assert claims == [{"claim": "A compound claim", "chunk_ids": [1, 2, 3]}]


def test_verify_citations_computes_coverage_from_judge_results():
    system = _make_system()
    chunks = [Document(page_content="context A", metadata={})]
    claims = [
        {"claim": "claim one", "chunk_ids": [1]},
        {"claim": "claim two", "chunk_ids": [1]},
    ]

    results = iter(
        [
            CitationVerification(
                claim="claim one", cited_chunk_ids=[1], is_supported=True, reasoning="ok"
            ),
            CitationVerification(
                claim="claim two",
                cited_chunk_ids=[1],
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


def test_verify_citations_combines_content_from_multiple_cited_chunks():
    """Regression test: a claim citing two chunks (chunk_ids=[1, 2]) must be
    judged against the COMBINED content of both, not just the first -- this
    is what was breaking "when did apollo 11 happen?"-style answers, where
    a single claim spans facts from two different chunks."""
    system = _make_system()
    chunks = [
        Document(page_content="launched July 16, 1969", metadata={}),
        Document(page_content="landed July 20, 1969", metadata={}),
    ]
    claims = [{"claim": "launched and landed in 1969", "chunk_ids": [1, 2]}]

    seen_prompts = []

    class FakeJudgeLLM:
        def invoke(self, prompt):
            seen_prompts.append(prompt)
            return CitationVerification(
                claim="launched and landed in 1969",
                cited_chunk_ids=[1, 2],
                is_supported=True,
                reasoning="both facts present",
            )

    system.judge_llm = FakeJudgeLLM()

    verification = system.verify_citations(claims, chunks)

    assert verification.coverage_percentage == 1.0
    assert "launched July 16, 1969" in seen_prompts[0]
    assert "landed July 20, 1969" in seen_prompts[0]


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
                cited_chunk_ids=[1],
                is_supported=True,
                reasoning="ok",
            )

    system.retriever = FakeRetriever()
    system.llm = FakeLLM()
    system.judge_llm = FakeJudgeLLM()

    response = system.generate_robust_answer("q")

    assert response["status"] == "Success"
    assert response["retrieved_chunks"] == [chunk_a, chunk_b]


def test_parse_citations_handles_comma_separated_citations():
    system = _make_system()
    answer = "Apollo 11 launched and landed in 1969 [1, 2]. Collins stayed in orbit [3]."

    claims = system.parse_citations(answer)

    assert claims == [
        {"claim": "Apollo 11 launched and landed in 1969", "chunk_ids": [1, 2]},
        {"claim": ". Collins stayed in orbit", "chunk_ids": [3]},
    ]


def test_score_confidence_clamps_out_of_range_completeness():
    system = _make_system()
    system.confidence_threshold = 0.75

    class FakeLLM:
        def invoke(self, prompt):
            class Response:
                content = "8"

            return Response()

    system.llm = FakeLLM()

    score = system.score_confidence(
        query="q", answer="a", retrieved_chunks=[], coverage=0.0, rerank_scores=[]
    )

    assert score.answer_completeness == 1.0
    assert score.composite_score == 0.3
    assert score.is_confident is False


def test_unknown_response_lists_suggested_documents_in_rank_order():
    system = _make_system()
    chunks = [
        Document(page_content="x", metadata={"filepath": "b.md"}),
        Document(page_content="y", metadata={"filepath": "a.md"}),
        Document(page_content="z", metadata={"filepath": "b.md"}),
    ]

    response = system._format_unknown_response("q", "reason", chunks)

    assert response["suggested_documents"] == ["b.md", "a.md"]
