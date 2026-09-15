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

    # --- Generation / confidence (src/generator.py: AdvancedRAGSystem) ---
    confidence_threshold: float = 0.75

    # --- Chunking (src/chunker.py: Chunker) ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Models (main.py, src/indexer.py, src/evaluator.py) ---
    generator_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Temperatures ---
    generator_temperature: float = 0
    judge_temperature: float = 0
    synthetic_temperature: float = 0.7


settings = Settings()
