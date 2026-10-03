import math
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import settings


def _normalize_cross_encoder_score(score: float) -> float:
    """Squash a raw cross-encoder logit into (0, 1) via a sigmoid."""
    return 1.0 / (1.0 + math.exp(-score))


class CitationVerification(BaseModel):
    claim: str = Field(description="The specific claim extracted from the answer")
    cited_chunk_ids: List[int] = Field(
        description="The IDs of the chunks cited for this claim"
    )
    is_supported: bool = Field(
        description="Whether the source text fully supports the claim"
    )
    reasoning: str = Field(
        description="Explanation of why the claim is or is not supported"
    )


class VerificationResult(BaseModel):
    verifications: List[CitationVerification]
    coverage_percentage: float


class ConfidenceScore(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    composite_score: float
    is_confident: bool


class AdvancedRAGSystem:
    def __init__(self, llm, retriever, confidence_threshold=settings.confidence_threshold):
        self.llm = llm
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.judge_llm = llm.with_structured_output(CitationVerification)

    def parse_citations(self, answer: str) -> List[Dict[str, Any]]:
        """Splits an answer into (claim, cited chunk ids) pairs. Back-to-back
        citations on the same claim (e.g. "...the Moon [1][2].") attach ALL
        of their chunk ids to that one claim, rather than the second (and
        any subsequent) citation being silently dropped because there's no
        text between the brackets to anchor a new claim to."""
        claims = []
        # Matches "[1]" as well as the comma-separated form "[1, 2]".
        citation = r"\[\d+(?:\s*,\s*\d+)*\]"
        parts = re.split(f"({citation})", answer)
        current_claim = ""

        for part in parts:
            if re.fullmatch(citation, part):
                chunk_ids = [int(n) for n in re.findall(r"\d+", part)]
                if current_claim.strip():
                    claims.append(
                        {"claim": current_claim.strip(), "chunk_ids": chunk_ids}
                    )
                elif claims:
                    claims[-1]["chunk_ids"].extend(chunk_ids)
                current_claim = ""
            else:
                current_claim += part

        return claims

    def verify_citations(
        self, claims: List[Dict[str, Any]], retrieved_chunks: List[Any]
    ) -> VerificationResult:
        verifications = []
        supported_count = 0

        for claim_data in claims:
            chunk_ids = claim_data["chunk_ids"]
            claim_text = claim_data["claim"]

            chunk_contents = [
                retrieved_chunks[cid - 1].page_content
                for cid in chunk_ids
                if 0 < cid <= len(retrieved_chunks)
            ]
            source_text = "\n\n".join(chunk_contents)

            prompt = f"""
            Evaluate if the following claim is fully supported by the provided source
            text (which may combine multiple cited chunks).
            Claim: {claim_text}
            Cited Chunk IDs: {chunk_ids}
            Source Text: {source_text}
            """

            result = self.judge_llm.invoke(prompt)
            verifications.append(result)

            if result.is_supported:
                supported_count += 1

        coverage = (supported_count / len(claims)) if claims else 1.0

        return VerificationResult(
            verifications=verifications, coverage_percentage=coverage
        )

    def score_confidence(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Any],
        coverage: float,
        rerank_scores: Optional[List[float]] = None,
    ) -> ConfidenceScore:
        top_scores = (rerank_scores or [])[:3]
        retrieval_score = (
            sum(_normalize_cross_encoder_score(s) for s in top_scores) / len(top_scores)
            if top_scores
            else 0.0
        )

        completeness_prompt = f"""
        Rate how completely the answer addresses the query on a scale of 0.0 to 1.0.
        Query: {query}
        Answer: {answer}
        Output ONLY a float value, nothing else.
        """
        try:
            completeness_result = float(
                self.llm.invoke(completeness_prompt).content.strip()
            )
        except ValueError:
            completeness_result = 0.5
        if not math.isfinite(completeness_result):
            completeness_result = 0.5
        # The LLM is asked for 0.0-1.0 but nothing enforces it; an
        # out-of-range reply (e.g. "8" on a 0-10 scale) must not be able to
        # push the composite past the threshold on its own.
        completeness_result = min(max(completeness_result, 0.0), 1.0)

        composite = (
            (retrieval_score * 0.3) + (coverage * 0.4) + (completeness_result * 0.3)
        )

        return ConfidenceScore(
            retrieval_confidence=retrieval_score,
            citation_coverage=coverage,
            answer_completeness=completeness_result,
            composite_score=composite,
            is_confident=composite >= self.confidence_threshold,
        )

    def generate_robust_answer(self, query: str):
        ranked_chunks = self.retriever.get_relevant_documents(query)

        if not ranked_chunks:
            return self._format_unknown_response(query, "No documents retrieved.")

        chunks = [doc for doc, _ in ranked_chunks]
        rerank_scores = [score for _, score in ranked_chunks]

        context = "\n".join(
            [
                f"Context Block [{i + 1}]:\n{c.page_content}"
                for i, c in enumerate(chunks)
            ]
        )

        qa_prompt = f"""
        You are a precise assistant. Answer the query using ONLY the provided context blocks.
        Cite specific chunks using bracketed references (e.g., [1]).
        Context:
        {context}
        Query: {query}
        """
        raw_answer = self.llm.invoke(qa_prompt).content

        claims = self.parse_citations(raw_answer)
        verification = self.verify_citations(claims, chunks)
        confidence = self.score_confidence(
            query, raw_answer, chunks, verification.coverage_percentage, rerank_scores
        )

        if not confidence.is_confident:
            return self._format_unknown_response(
                query,
                "System confidence fell below threshold.",
                chunks,
                confidence.composite_score,
            )

        return {
            "status": "Success",
            "answer": raw_answer,
            "retrieved_chunks": chunks,
            "confidence_metrics": confidence.model_dump(),
            "flagged_citations": [
                v.model_dump() for v in verification.verifications if not v.is_supported
            ],
        }

    def _format_unknown_response(
        self,
        query: str,
        reason: str,
        chunks: Optional[List[Any]] = None,
        score: float = 0.0,
    ):
        response = {
            "status": "Insufficient Information",
            "reason": reason,
            "confidence_score": score,
            "found_context": "The system retrieved some potentially related information but could not formulate a reliable answer.",
            "missing_information": query,
            "recommended_action": "Review the suggested documents manually or rephrase the query.",
        }

        if chunks:
            # dict.fromkeys de-duplicates while keeping rank order (a set
            # would shuffle the most relevant source out of first place).
            response["suggested_documents"] = list(
                dict.fromkeys(
                    c.metadata.get("filepath", "Unknown source") for c in chunks[:3]
                )
            )

        return response
