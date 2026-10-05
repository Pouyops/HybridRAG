# HybridRAG — Roadmap to a Resume-Worthy, Shipped System

Target role: **ML/AI engineer who ships systems.** So this roadmap leans on
deployment, reproducibility, and observability — while keeping the evaluation
rigor that's already this project's biggest differentiator.

Work top-to-bottom. Each item says **what**, **why it matters for the role**, and a
**starting point**. Write all the code yourself — the value is in doing it.

---

## Phase 0 — Fix what's quietly wrong (do this first)

Understanding *why* each of these is a bug is the exact muscle the role tests.
Don't let an interviewer find them before you do.

- [ ] **Stop faking retrieval confidence.**
  `generator.py :: score_confidence` hardcodes `retrieval_score = 0.85` and feeds
  it into the composite that gates every answer. Real signal exists: the
  cross-encoder already produces relevance scores in `rerank()`. Return those
  scores from `rerank()`, normalize the top score (or mean of top-k), and use it.
  *Why:* shows you can tell a measured signal from a decorative one.

- [ ] **Make "citation accuracy" actually measure citations.**
  `evaluator.py` sets `citation_accuracy = citation_metrics["citation_coverage"]`
  — a relabel, not a metric. Compute it from `verify_citations` (fraction of
  claims whose cited chunk genuinely supports them).

- [ ] **Retrieve once per question, not twice.**
  `generate_robust_answer` retrieves, then `run_test_suite` calls
  `get_relevant_documents` again to rebuild the context string — doubling cost and
  risking a different context than the answer used. Have `generate_robust_answer`
  return the chunks it used; reuse them in the evaluator.

- [ ] **Untangle the RRF constants.**
  `retrieve_and_fuse` uses `fusion_k` (60) as *both* dense retrieval depth *and*
  the RRF smoothing constant in `1/(fusion_k + rank)`. Split into two independent
  params (`retrieval_depth`, `rrf_k`). Also delete the unused `initial_k`.

- [ ] **Confirm the vector DBs aren't committed to git.**
  `chroma_db_*/` and `chroma_main_db/` should be in `.gitignore`. Commit a small,
  reproducible index build script instead of binary DB blobs.

---

## Phase 1 — Ship it as a service (the headline for "ships systems")

- [ ] **Wrap the pipeline in a FastAPI service.**
  `POST /query` → `{answer, citations, confidence, latency_ms}`. Add `/health`.
  *Why:* an endpoint is the single clearest "I build systems, not scripts" signal.
  *Start:* one `app.py`, load the retriever/generator once at startup, not per request.

- [ ] **Add a live demo UI (Gradio or Streamlit).**
  A box where anyone types a question and watches retrieve → rerank → cited answer.
  *Why:* a recruiter who can *click* your project remembers it. Host it free on
  Hugging Face Spaces so there's a public URL for your resume.

- [ ] **Dockerize.**
  `Dockerfile` + `docker-compose.yml` so "clone and run" is one command.
  *Why:* reproducibility is a hiring filter for infra-facing ML roles.

- [ ] **Centralize configuration.**
  Move every magic number (RRF weights, `top_n`, `chunk_size`, model names,
  `confidence_threshold`) into `config.yaml` or `pydantic-settings`. No literals
  scattered across modules.

- [ ] **Add observability.**
  Log per-query latency and token cost; expose a rolling summary. Then you can say
  "median 1.2s, ~$0.003/query" — the language of someone who's run things in prod.
  *Stretch:* Prometheus metrics or an OpenTelemetry trace per stage.

- [ ] **CI with GitHub Actions.**
  Run `pytest` on every push; add lint (`ruff`) and format (`black`) checks.
  Put the green badge in the README.

---

## Phase 2 — Make the evaluation bulletproof (your differentiator)

- [ ] **Freeze a versioned eval set.**
  Commit a fixed `evaluation_dataset.json`, seed the RNG. Reproducible numbers,
  apples-to-apples comparisons.

- [ ] **Report mean ± std, not single runs.**
  Run each config N times; report variance. "0.90 ± 0.02 over 5 runs" is what an
  ML engineer says.

- [ ] **Ablation study — the real resume table.**
  Measure the drop when each component is removed: dense-only vs sparse-only vs
  hybrid; reranker on vs off; RRF weight sweep. A row like "reranking: +6 pts
  retrieval relevance" proves you can measure *impact*, not just build features.

- [ ] **Validate the LLM judge.**
  Hand-label ~15 answers yourself; report judge–human agreement (e.g. "87%").
  *Why:* it makes every other number you report trustworthy.

---

## Phase 3 — Give it a story people can see

- [ ] **Swap in a real corpus.**
  Replace the single fictional PDF with real docs (a set of arXiv papers, product
  docs, a Wikipedia topic dump). Makes the demo legible and multi-hop questions
  genuinely hard.

- [ ] **Write a results writeup** (`RESULTS.md` or a blog post) with 2–3 charts:
  the chunking-strategy comparison you already have, plus the reranker and
  RRF-weight ablations. This writeup *is* the resume bullet.

- [ ] **Tighten the README top.**
  Lead with: a one-line pitch, the live-demo link, an architecture diagram, and
  the headline results table. A reader should get value in 15 seconds.

---

## Suggested resume bullets (fill in your real numbers)

- Built and deployed an end-to-end hybrid RAG system (dense + BM25 with reciprocal
  rank fusion, cross-encoder reranking, citation-verified generation) served via
  FastAPI + Docker with a public Gradio demo.
- Designed an automated LLM-as-judge evaluation harness with synthetic QA
  generation; ran ablations showing reranking improved retrieval relevance by
  **X pts** and validated the judge against human labels at **Y%** agreement.
- Instrumented the pipeline for cost/latency observability (median **Z** s,
  **$_** /query) and reproducible builds via config-driven runs and CI.

---

## Order by return-on-effort

1. Phase 0 (bugs) — credibility floor.
2. Phase 1 (FastAPI + demo + Docker) — the "ships systems" story.
3. Phase 2 ablation table + judge validation — the rigor that sets you apart.
4. Phase 3 real corpus + writeup — the polish recruiters actually see.

*I'll explain any concept, review your diffs, or sketch a file's structure when you
get stuck — but the implementation is yours to write. That's where the learning is.*
