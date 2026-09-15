"""FastAPI service exposing the Hybrid RAG pipeline.

Endpoints:
    POST /query    -> {"answer", "citations", "confidence", "status", "latency_ms"}
    GET  /health   -> {"status": "ok"}
    GET  /metrics  -> rolling median latency and average cost estimate

The pipeline is built ONCE at startup (not per-request) via src/pipeline.py.
"""

import logging
import os
import statistics
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import settings
from src.pipeline import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hybridrag.app")

# Approximate gpt-4o-mini pricing (USD per token). Illustrative only —
# verify against current OpenAI pricing before using for real cost tracking.
_PRICE_PER_INPUT_TOKEN = 0.15 / 1_000_000
_PRICE_PER_OUTPUT_TOKEN = 0.60 / 1_000_000

_TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")

# In-memory, per-process metrics store. Explicitly lightweight: no external
# time-series DB, just enough to eyeball latency/cost while demoing.
_metrics: Dict[str, List[float]] = {"latency_ms": [], "cost_usd": []}

_state: Dict[str, Any] = {"pipeline": None}


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_TOKEN_ENCODING.encode(text))


def _estimate_cost_usd(prompt_text: str, completion_text: str) -> float:
    input_tokens = _count_tokens(prompt_text)
    output_tokens = _count_tokens(completion_text)
    return (
        input_tokens * _PRICE_PER_INPUT_TOKEN
        + output_tokens * _PRICE_PER_OUTPUT_TOKEN
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    data_dir = os.getenv("DATA_DIR", "./data/")

    logger.info("Building RAG pipeline (data_dir=%r)...", data_dir)
    start = time.perf_counter()
    _state["pipeline"] = build_pipeline(openai_api_key=api_key, data_dir=data_dir)
    elapsed = time.perf_counter() - start
    logger.info("Pipeline ready in %.2fs (indexing complete).", elapsed)

    yield

    _state["pipeline"] = None


app = FastAPI(title="HybridRAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    confidence: Optional[Dict[str, Any]]
    status: str
    latency_ms: float


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    latencies = _metrics["latency_ms"]
    costs = _metrics["cost_usd"]
    return {
        "requests_served": len(latencies),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "avg_cost_usd": (sum(costs) / len(costs)) if costs else None,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must be a non-empty string.")

    start = time.perf_counter()
    result = pipeline.generate_robust_answer(request.query)
    latency_ms = (time.perf_counter() - start) * 1000

    status = result.get("status", "Unknown")

    if status == "Success":
        answer = result.get("answer", "")
        citations = result.get("flagged_citations", [])
        confidence = result.get("confidence_metrics")
    else:
        # "Insufficient Information" shaped response — no answer/citations/confidence.
        answer = result.get("found_context", "")
        citations = []
        confidence = None

    retrieved_chunks = result.get("retrieved_chunks", [])
    context_text = "\n".join(c.page_content for c in retrieved_chunks) if retrieved_chunks else ""
    cost_usd = _estimate_cost_usd(context_text + request.query, answer)

    _metrics["latency_ms"].append(latency_ms)
    _metrics["cost_usd"].append(cost_usd)

    logger.info(
        "query served status=%s latency_ms=%.1f est_cost_usd=%.6f",
        status,
        latency_ms,
        cost_usd,
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        status=status,
        latency_ms=latency_ms,
    )
