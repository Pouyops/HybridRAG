Advanced RAG Pipeline with Automated Evaluation 🚀An end-to-end Retrieval-Augmented Generation (RAG) system built with Python and LangChain. This project goes beyond basic RAG implementations by introducing Hybrid Retrieval with Reciprocal Rank Fusion (RRF), Citation Verification, Synthetic Dataset Generation, and an LLM-as-a-Judge evaluation framework to compare different chunking strategies.🗺️ System Architecture (Graphic Map)The following diagram illustrates the data flow of the pipeline, from document ingestion to automated evaluation.graph TD
    %% Data Ingestion Flow
    subgraph Data Ingestion & Indexing
        A[Raw Documents<br/>PDF, HTML, MD, TXT] --> B(Multi-Format Loader)
        B --> C{Chunking Engine}
        C -->|Token Recursive| D(Vector Embeddings & Indexing)
        C -->|Markdown| D
        C -->|Semantic| D
        D --> E[(ChromaDB Dense Index)]
        D --> F[(BM25 Sparse Index)]
    end

    %% Retrieval Flow
    subgraph Query Processing & Hybrid Retrieval
        G[User Query] --> H(Dense Retrieval)
        G --> I(Sparse Retrieval)
        E --> H
        F --> I
        H --> J(Reciprocal Rank Fusion - RRF)
        I --> J
        J --> K(Cross-Encoder Re-ranker)
    end

    %% Generation Flow
    subgraph Generation & Verification
        K --> L(Context Aggregation)
        L --> M[Gemini LLM Generator]
        G --> M
        M --> N(Citation Parser)
        N --> O{Citation Verifier & Confidence Scorer}
        O -->|Confident| P[Final Robust Answer]
        O -->|Low Confidence| Q[Fallback / Unknown Response]
    end

    %% Evaluation Flow
    subgraph Automated Evaluation
        R[Synthetic Data Generator] -->|Lookup, Multi-hop,<br/>Ambiguous, Unanswerable| S[(Evaluation Dataset)]
        S --> T(LLM-as-a-Judge Evaluator)
        P --> T
        Q --> T
        T --> U[Strategy Comparison Report<br/>Correctness, Faithfulness, Relevance]
    end
✨ Key FeaturesMulti-Format Document Loader: Seamlessly ingests .pdf, .html, .md, and .txt files from complex directory structures, normalizing text and extracting metadata using PyMuPDF and BeautifulSoup.Dynamic Chunking Engine: Implements and evaluates three distinct chunking strategies:Token-Recursive Chunking (standard character/token overlap)Markdown Header Chunking (structure-aware)Semantic Chunking (embedding-based breakpoint detection)Hybrid Retrieval (RRF) & Re-ranking: Combines Dense vector search (ChromaDB + Gemini Embeddings) with Sparse keyword search (BM25) using Reciprocal Rank Fusion, followed by a Cross-Encoder (ms-marco-MiniLM-L-6-v2) for precision re-ranking.Self-Verifying Generation: The LLM generates answers with strict bracketed citations (e.g., [1]). A secondary "Judge" LLM verifies that every cited chunk mathematically supports the claim, calculating a final composite confidence score.Synthetic Q&A Generation: Automatically generates a comprehensive evaluation dataset directly from the source documents, featuring various question types: Lookup, Multi-Hop, Unanswerable, and Ambiguous.Automated "LLM-as-a-Judge" Evaluation: Quantitatively measures pipeline performance based on:CorrectnessFaithfulness to ContextRetrieval RelevanceCitation Accuracy🛠️ Technology StackOrchestration: LangChainLLMs & Embeddings: Google Gemini (gemini-2.5-flash, gemini-2.5-pro, gemini-embedding-001)Vector Store: ChromaDBSparse Retrieval: BM25 (rank_bm25)Re-ranking: Sentence Transformers (cross-encoder)Document Processing: PyMuPDF, BeautifulSoup4🚀 Installation & SetupClone the repository:git clone [https://github.com/pouyops/HybridRAG.git](https://github.com/pouyops/HybridRAG.git)
cd HybridRAG
Install dependencies:pip install langchain_text_splitters pymupdf langchain_google_genai chromadb rank_bm25 sentence-transformers langchain_community bs4 pydantic pandas
Set Environment Variables:You will need a Google Gemini API Key.export GOOGLE_API_KEY="your-google-api-key"
🧠 Usage1. Basic RAG QueryingInitialize the pipeline, chunk your documents, and ask questions:# Assuming 'all_documents' is loaded via multiloader
idx = indexer()
vectorstore, bm25 = idx.create_indexes(all_chunks, persist_directory="./chroma_db")
retriever = HybridRetriever(vectorestore=vectorstore, bm25_retriever=bm25)
rag_system = AdvancedRAGSystem(llm=generator_llm, retriever=retriever)

response = rag_system.generate_robust_answer("Your question here?")
print(response["answer"])
2. Generating an Evaluation DatasetGenerate a synthetic dataset from your documents for testing:synthetic_evaluator = SyntheticEvaluator(vectorstore=vectorstore, llm=judge_llm)
dataset = synthetic_evaluator.build_dataset(total_questions=50) 
# Saves to 'evaluation_dataset.json'
3. Running the Strategy ComparisonCompare how different chunking strategies affect the RAG system's correctness and retrieval relevance:comparison_report = run_strategy_comparison(
    gitlab_documents=all_documents,
    dataset_path='evaluation_dataset.json',
    llm=generator_llm,
    judge_llm=judge_llm
)
# Outputs a Pandas DataFrame comparing Token, Markdown, and Semantic chunking.
📊 Evaluation Metrics ExplainedCorrectness (0-1): Does the generated answer accurately match the expected ground truth?Faithfulness (0-1): Is the answer entirely derived from the provided context (avoiding hallucination)?Retrieval Relevance (0-1): Did the retriever pull chunks that actually contain the answer?Citation Coverage (0-1): What percentage of the claims made in the answer are successfully backed by the cited document chunks?📝 LicenseThis project is licensed under the MIT License.