import os
from dotenv import load_dotenv
import shutil
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

from src.loader import multiloader
from src.chunker import Chunker
from src.indexer import indexer
from src.retriever import HybridRetriever
from src.generator import AdvancedRAGSystem
from src.evaluator import SyntheticEvaluator, RAGEvaluator

def run_strategy_comparison(gitlab_documents, dataset_path, llm, judge_llm, api_key):
    strategies = ["TokenRecursive", "Markdown", "Semantic"]
    report_data = []

    for strategy in strategies:
        db_path = f"./chroma_db_{strategy}"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        chunker = Chunker(embedding_fn=None)
        all_chunks = []
        for doc in gitlab_documents:
            chunks = chunker.chunk_documents(doc.page_content, doc.metadata, strategy=strategy)
            all_chunks.extend(chunks)

        idx = indexer(api_key=api_key)
        vectorstore, bm25 = idx.create_indexes(all_chunks, persist_directory=db_path)

        retriever = HybridRetriever(vectorestore=vectorstore, bm25_retriever=bm25)
        rag_pipeline = AdvancedRAGSystem(llm=llm, retriever=retriever)
        evaluator = RAGEvaluator(rag_pipeline=rag_pipeline, judge_llm=judge_llm)

        metrics = evaluator.run_test_suite(dataset_path)

        failure_rate = 0.0
        if metrics["total_runs"] > 0:
            failure_rate = len(metrics["failures"]) / metrics["total_runs"]

        report_data.append({
            "Chunking Strategy": strategy,
            "Correctness": round(metrics["avg_correctness"], 4),
            "Faithfulness": round(metrics["avg_faithfulness"], 4),
            "Retrieval Relevance": round(metrics["avg_retrieval"], 4),
            "Citation Accuracy": round(metrics["avg_citation_accuracy"], 4),
            "Fallback Rate": round(failure_rate, 4)
        })

    report_df = pd.DataFrame(report_data)

    print("\n" + "="*70)
    print(" CHUNKING STRATEGY EVALUATION REPORT")
    print("="*70)
    print(report_df.to_string(index=False))

    best_retrieval = report_df.loc[report_df['Retrieval Relevance'].idxmax()]['Chunking Strategy']
    best_citation = report_df.loc[report_df['Citation Accuracy'].idxmax()]['Chunking Strategy']

    print("\n" + "-"*70)
    print(f"Winner - Retrieval Relevance: {best_retrieval}")
    print(f"Winner - Citation Accuracy: {best_citation}")

    return report_df

if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    generator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    dataset_path = './data/'
    loader = multiloader(dataset_path)
    all_documents = loader._document_loader()
    subset_all_documents = all_documents[:50]

    if not subset_all_documents:
        print("No documents loaded. Please add files to the ./data/ directory.")
    else:
        print("--- Testing Single Query Execution ---")
        chunker = Chunker(embedding_fn=None)
        all_chunks = []
        for doc in subset_all_documents:
            chunks = chunker.chunk_documents(doc.page_content, doc.metadata, strategy="TokenRecursive")
            all_chunks.extend(chunks)

        idx = indexer(api_key=GOOGLE_API_KEY)
        vectorstore, bm25 = idx.create_indexes(all_chunks, persist_directory="./chroma_main_db")

        retriever = HybridRetriever(vectorestore=vectorstore, bm25_retriever=bm25)
        rag_system = AdvancedRAGSystem(llm=generator_llm, retriever=retriever)

        test_query = "What specific defensive action did Sergeant Miller take when the enemy armored units initiated the breach at 0400 hours?"
        response = rag_system.generate_robust_answer(test_query)
        print(response)

        print("\n--- Generating Synthetic Evaluation Dataset ---")
        synthetic_evaluator = SyntheticEvaluator(vectorstore=vectorstore, llm=judge_llm)
        synthetic_evaluator.build_dataset(total_questions=15)
        print("Dataset generated and saved to evaluation_dataset.json")

        print("\n--- Running Strategy Comparison ---")
        comparison_report = run_strategy_comparison(
            gitlab_documents=subset_all_documents,
            dataset_path='evaluation_dataset.json',
            llm=generator_llm,
            judge_llm=judge_llm,
            api_key=GOOGLE_API_KEY
        )