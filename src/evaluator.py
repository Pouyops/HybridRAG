import random
import json
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class QAPair(BaseModel):
    question: str = Field(description="The generated question")
    expected_answer: str = Field(description="The ground truth answer")
    question_type: str = Field(description="The category of the question")
    source_chunks: list[str] = Field(description="The chunk IDs or filepaths used")

class SyntheticEvaluator():
    def __init__(self, vectorstore, llm=None):
        self.vectorstore = vectorstore
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
        self.structured_llm = self.llm.with_structured_output(QAPair)

        self.all_docs = self.vectorstore.get()['documents']
        self.all_metadatas = self.vectorstore.get()['metadatas']

    def generate_lookup(self):
        idx = random.randint(0, len(self.all_docs) - 1)
        chunk_text = self.all_docs[idx]
        source = self.all_metadatas[idx].get('filepath', 'Unknown')

        prompt = ChatPromptTemplate.from_template(
            "Read this text: {context}\n"
            "Generate a highly specific factual question that can be answered ONLY using this text. "
            "Provide the correct answer. Set question_type to 'Lookup'."
        )

        result = self.structured_llm.invoke(prompt.format(context=chunk_text))
        result.source_chunks = [source]
        return result

    def generate_multihop(self):
        idx1, idx2 = random.sample(range(len(self.all_docs)), 2)
        chunk1, chunk2 = self.all_docs[idx1], self.all_docs[idx2]
        source1 = self.all_metadatas[idx1].get('filepath', 'Unknown')
        source2 = self.all_metadatas[idx2].get('filepath', 'Unknown')

        prompt = ChatPromptTemplate.from_template(
            "Text 1: {context1}\nText 2: {context2}\n"
            "Generate a complex question that REQUIRES information from BOTH texts to answer fully. "
            "Provide the comprehensive answer. Set question_type to 'Multi-Hop'."
        )

        result = self.structured_llm.invoke(prompt.format(context1=chunk1, context2=chunk2))
        result.source_chunks = [source1, source2]
        return result

    def generate_unanswerable(self):
        idx = random.randint(0, len(self.all_docs) - 1)
        chunk_text = self.all_docs[idx]
        source = self.all_metadatas[idx].get('filepath', 'Unknown')

        prompt = ChatPromptTemplate.from_template(
            "Text: {context}\n"
            "Generate a question that sounds highly relevant to this domain and uses the same entities, "
            "but asks for a specific detail explicitly NOT present in the text. "
            "The expected_answer must state what information is missing. Set question_type to 'Unanswerable'."
        )

        result = self.structured_llm.invoke(prompt.format(context=chunk_text))
        result.source_chunks = [source]
        return result

    def generate_ambiguous(self):
        idx = random.randint(0, len(self.all_docs) - 1)
        chunk_text = self.all_docs[idx]
        source = self.all_metadatas[idx].get('filepath', 'Unknown')

        prompt = ChatPromptTemplate.from_template(
            "Text: {context}\n"
            "Generate a vaguely worded question regarding this text. Omit specific names, dates, or context. "
            "The expected_answer should explain why the question cannot be answered as asked and what clarification is needed. "
            "Set question_type to 'Ambiguous'."
        )

        result = self.structured_llm.invoke(prompt.format(context=chunk_text))
        result.source_chunks = [source]
        return result

    def build_dataset(self, total_questions=50):
        dataset = []
        distribution = {
            self.generate_lookup: int(total_questions * 0.4),
            self.generate_multihop: int(total_questions * 0.3),
            self.generate_unanswerable: int(total_questions * 0.15),
            self.generate_ambiguous: int(total_questions * 0.15)
        }

        for func, count in distribution.items():
            for _ in range(count):
                try:
                    qa_pair = func()
                    dataset.append(qa_pair.model_dump())
                except Exception as e:
                    print(f"Generation failed for a {func.__name__} prompt. Skipping.")

        with open('evaluation_dataset.json', 'w') as f:
            json.dump(dataset, f, indent=4)

        return dataset

class EvaluationScore(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief justification for the assigned score")

class RAGEvaluator():
    def __init__(self, rag_pipeline, judge_llm):
        self.rag_pipeline = rag_pipeline
        self.judge_llm = judge_llm.with_structured_output(EvaluationScore)

    def measure_correctness(self, question, expected, generated):
        prompt = ChatPromptTemplate.from_template(
            "Evaluate the correctness of the generated answer against the ground truth.\n"
            "Question: {question}\nGround Truth: {expected}\nGenerated Answer: {generated}\n"
            "Assign a score from 0.0 (completely wrong) to 1.0 (perfectly correct)."
        )
        return self.judge_llm.invoke(prompt.format(
            question=question,
            expected=expected,
            generated=generated
        ))

    def measure_faithfulness(self, generated, context):
        prompt = ChatPromptTemplate.from_template(
            "Evaluate whether the generated answer is entirely grounded in the provided context.\n"
            "Context: {context}\nGenerated Answer: {generated}\n"
            "Assign a score from 0.0 (hallucinated) to 1.0 (perfectly faithful to context)."
        )
        return self.judge_llm.invoke(prompt.format(
            context=context,
            generated=generated
        ))

    def measure_retrieval_relevance(self, question, context):
        prompt = ChatPromptTemplate.from_template(
            "Evaluate how relevant the retrieved context is to answering the question.\n"
            "Question: {question}\nContext: {context}\n"
            "Assign a score from 0.0 (irrelevant) to 1.0 (highly relevant and sufficient)."
        )
        return self.judge_llm.invoke(prompt.format(
            question=question,
            context=context
        ))

    def run_test_suite(self, dataset_path):
        with open(dataset_path, 'r') as f:
            test_cases = json.load(f)

        results = {
            "total_runs": 0,
            "avg_correctness": 0.0,
            "avg_faithfulness": 0.0,
            "avg_retrieval": 0.0,
            "avg_citation_accuracy": 0.0,
            "failures": []
        }

        for case in test_cases:
            query = case["question"]
            expected = case["expected_answer"]

            rag_response = self.rag_pipeline.generate_robust_answer(query)

            if rag_response.get("status") != "Success":
                results["failures"].append({
                    "query": query,
                    "reason": rag_response.get("reason")
                })
                continue

            generated_answer = rag_response["answer"]
            retrieved_chunks = self.rag_pipeline.retriever.get_relevant_documents(query)
            context_str = "\n".join([c.page_content for c in retrieved_chunks])

            correctness = self.measure_correctness(query, expected, generated_answer)
            faithfulness = self.measure_faithfulness(generated_answer, context_str)
            retrieval = self.measure_retrieval_relevance(query, context_str)

            citation_metrics = rag_response.get("confidence_metrics", {})
            citation_accuracy = citation_metrics.get("citation_coverage", 0.0)

            results["avg_correctness"] += correctness.score
            results["avg_faithfulness"] += faithfulness.score
            results["avg_retrieval"] += retrieval.score
            results["avg_citation_accuracy"] += citation_accuracy
            results["total_runs"] += 1

        if results["total_runs"] > 0:
            n = results["total_runs"]
            results["avg_correctness"] /= n
            results["avg_faithfulness"] /= n
            results["avg_retrieval"] /= n
            results["avg_citation_accuracy"] /= n

        return results