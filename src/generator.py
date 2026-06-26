import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class CitationVerification(BaseModel):
    claim: str = Field(description="The specific claim extracted from the answer")
    cited_chunk_id: int = Field(description="The ID of the chunk cited for this claim")
    is_supported: bool = Field(description="Whether the source text fully supports the claim")
    reasoning: str = Field(description="Explanation of why the claim is or is not supported")

class VerificationResult(BaseModel):
    verifications: List[CitationVerification]
    coverage_percentage: float

class ConfidenceScore(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    composite_score: float
    is_confident: bool

class AdvancedRAGSystem():
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.confidence_threshold = 0.75
        self.judge_llm = llm.with_structured_output(CitationVerification)

    def parse_citations(self, answer: str) -> List[Dict[str, Any]]:
        claims = []
        parts = re.split(r'(\[\d+\])', answer)
        current_claim = ""

        for part in parts:
            if re.match(r'\[\d+\]', part):
                chunk_id = int(part.strip('[]'))
                if current_claim.strip():
                    claims.append({
                        "claim": current_claim.strip(),
                        "chunk_id": chunk_id
                    })
                current_claim = ""
            else:
                current_claim += part

        return claims

    def verify_citations(self, claims: List[Dict[str, Any]], retrieved_chunks: List[Any]) -> VerificationResult:
        verifications = []
        supported_count = 0

        for claim_data in claims:
            chunk_id = claim_data["chunk_id"]
            claim_text = claim_data["claim"]

            chunk_content = ""
            if 0 < chunk_id <= len(retrieved_chunks):
                chunk_content = retrieved_chunks[chunk_id - 1].page_content

            prompt = f"""
            Evaluate if the following claim is fully supported by the provided source text.
            Claim: {claim_text}
            Source Text: {chunk_content}
            """

            result = self.judge_llm.invoke(prompt)
            verifications.append(result)

            if result.is_supported:
                supported_count += 1

        coverage = (supported_count / len(claims)) if claims else 1.0

        return VerificationResult(
            verifications=verifications,
            coverage_percentage=coverage
        )

    def score_confidence(self, query: str, answer: str, retrieved_chunks: List[Any], coverage: float) -> ConfidenceScore:
        retrieval_score = 0.85

        completeness_prompt = f"""
        Rate how completely the answer addresses the query on a scale of 0.0 to 1.0.
        Query: {query}
        Answer: {answer}
        Output ONLY a float value, nothing else.
        """
        try:
            completeness_result = float(self.llm.invoke(completeness_prompt).content.strip())
        except ValueError:
            completeness_result = 0.5

        composite = (retrieval_score * 0.3) + (coverage * 0.4) + (completeness_result * 0.3)

        return ConfidenceScore(
            retrieval_confidence=retrieval_score,
            citation_coverage=coverage,
            answer_completeness=completeness_result,
            composite_score=composite,
            is_confident=composite >= self.confidence_threshold
        )

    def generate_robust_answer(self, query: str):
        chunks = self.retriever.get_relevant_documents(query)

        if not chunks:
            return self._format_unknown_response(query, "No documents retrieved.")

        context = "\n".join([f"Context Block [{i+1}]:\n{c.page_content}" for i, c in enumerate(chunks)])

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
        confidence = self.score_confidence(query, raw_answer, chunks, verification.coverage_percentage)

        if not confidence.is_confident:
            return self._format_unknown_response(
                query,
                "System confidence fell below threshold.",
                chunks,
                confidence.composite_score
            )

        return {
            "status": "Success",
            "answer": raw_answer,
            "confidence_metrics": confidence.model_dump(),
            "flagged_citations": [v.model_dump() for v in verification.verifications if not v.is_supported]
        }

    def _format_unknown_response(self, query: str, reason: str, chunks: List[Any] = None, score: float = 0.0):
        response = {
            "status": "Insufficient Information",
            "reason": reason,
            "confidence_score": score,
            "found_context": "The system retrieved some potentially related information but could not formulate a reliable answer.",
            "missing_information": query,
            "recommended_action": "Review the suggested documents manually or rephrase the query."
        }

        if chunks:
            response["suggested_documents"] = list(set([c.metadata.get('filepath', 'Unknown source') for c in chunks[:3]]))

        return response