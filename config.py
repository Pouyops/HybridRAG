"""Centralized, environment-overridable configuration for tunable pipeline
parameters (retrieval weights, chunking sizes, model names, thresholds, ...).

This module intentionally does NOT hold secrets (API keys) — those stay as
os.getenv() calls at the call sites, loaded via python-dotenv, exactly as
before. config.py is a single source of truth for the magic numbers/strings
that used to be scattered across src/retriever.py, src/generator.py,
src/chunker.py, src/indexer.py and main.py, with the SAME defaults they had
before this module existed. Every field can be overridden via an environment
variable of the same name (uppercased), e.g. DENSE_WEIGHT=0.6.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Hybrid retrieval fusion (src/retriever.py: HybridRetriever) ---
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    retrieval_depth: int = 60
    rrf_k: int = 60
    top_n: int = 20
    final_k: int = 5
    # Reorder the fused candidates with the cross-encoder. Off by default:
    # on the RFC corpus it lowered hit@5 in all 9 chunking x mode
    # combinations (RESULTS.md). The cross-encoder still scores the final
    # chunks for the confidence gate (cross_encoder_confidence).
    use_reranker: bool = False
    cross_encoder_confidence: bool = True
    # Ask the LLM to split multi-part questions into sub-questions and
    # retrieve for each (src/retriever.py: DecomposingRetriever). Off by
    # default: on the RFC set it did not raise multi-hop full@5 beyond noise
    # and lowered hit@5 and MRR in most configurations (RESULTS.md).
    decompose_queries: bool = False
    max_subquestions: int = 3

    # --- Generation / confidence (src/generator.py: AdvancedRAGSystem) ---
    confidence_threshold: float = 0.75

    # --- Chunking (src/chunker.py: Chunker) ---
    chunk_size: int = 512
    chunk_overlap: int = 50
    # Default strategy for the service and build_pipeline().
    chunking_strategy: str = "TokenRecursive"
    # Markdown-header sections longer than this many tokens are split further
    # (0 disables the cap). Uncapped, some RFC sections are ~5k tokens.
    markdown_max_tokens: int = 800

    # --- Models (main.py, src/indexer.py, src/evaluator.py) ---
    generator_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Temperatures ---
    generator_temperature: float = 0
    judge_temperature: float = 0
    synthetic_temperature: float = 0.7


settings = Settings()
