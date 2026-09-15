# Evaluation Results

This document reports what was actually measured against the HybridRAG
pipeline's evaluation harness, not aspirational numbers. Where a run failed,
was cut short by a rate limit, or produced a surprising result, that is
called out explicitly rather than smoothed over.

**Corpus:** a 7-file, ~2,885-word original Apollo 11 markdown corpus
(`./data/`), indexed as 12 chunks under the default `TokenRecursive`
chunking strategy.

**Frozen evaluation set:** `evaluation_dataset.json`, generated once via
`SyntheticEvaluator.build_dataset(total_questions=15)` with `random.seed(42)`
for reproducible chunk selection, then committed and reused as-is by every
run below (`main.py`'s default behavior no longer regenerates it; pass
`--regenerate-eval-set` to opt back in). **Note:** the dataset actually
contains **14** questions, not 15 -- `build_dataset` computes each question
type's count as `int(total_questions * fraction)` (6 Lookup + 4 Multi-Hop +
2 Unanswerable + 2 Ambiguous = 14), which silently truncates rather than
rounds. This is pre-existing `SyntheticEvaluator` behavior this phase did
not change; flagging it here for accuracy. Seeding pins *which chunks* get
sampled for question generation, not the LLM's exact phrasing of the
question/answer text -- that remains subject to OpenAI API non-determinism
even at temperature 0.

---

## Headline Findings

1. **All three chunking strategies retrieve well and answer correctly when
   they answer at all.** Across 3 repeated runs, Correctness and
   Faithfulness are at or within a hair of 1.000 for every strategy
   (TokenRecursive and Markdown: 0.997 +/- 0.005; Semantic: 1.000 +/- 0.000),
   and Retrieval Relevance sits in a tight 0.979-0.983 band. On this small,
   clean 7-file corpus, chunking strategy is not the bottleneck.
2. **The real bottleneck is the confidence gate, not retrieval or
   generation quality.** All three strategies fail (fall back to
   "Insufficient Information") on **40-45% of the 14 frozen questions** --
   TokenRecursive and Markdown both at 0.400 +/- 0.000, Semantic worse and
   noisier at 0.452 +/- 0.073. Given the frozen set is deliberately 2/14
   Unanswerable + 2/14 Ambiguous (which are *supposed* to trigger the
   fallback -- see Section 3), a chunk of that 40%+ is expected/correct
   behavior, not a defect; what this number can't distinguish, without the
   per-question breakdown, is how much of the rest is the confidence
   threshold (0.75) being conservative on genuinely answerable questions.
3. **TokenRecursive wins on retrieval relevance, Markdown wins on citation
   accuracy, and Semantic is the least stable strategy on every metric.**
   Semantic has the largest std of all three strategies on retrieval
   relevance (+/-0.008), citation accuracy (+/-0.030, and the lowest mean
   there too at 0.964 vs 0.975-0.978), and fallback rate (+/-0.073) --
   consistent with `SemanticChunker`'s embedding-distance-based split points
   being more sensitive to run-to-run variance than the deterministic
   token/markdown splitters.
4. **Retrieval mode barely matters on this corpus; the reranker trades
   correctness for a lower fallback rate.** Dense-only, sparse-only, and
   hybrid retrieval are statistically indistinguishable on retrieval
   relevance (0.980/0.980/0.980) -- on a 12-chunk corpus this small there
   just isn't much for the fusion weights to disagree about. The one real
   signal in the ablation: turning the cross-encoder reranker **off**
   *raised* retrieval relevance slightly (0.989 vs 0.980) and roughly
   *halved* the fallback rate (14.3% vs 28.6%), but *dropped* correctness
   noticeably (0.925 vs 0.997, and the only config with meaningfully
   nonzero correctness std at +/-0.014). See Section 2 for the full
   breakdown and why this is plausible given how confidence scoring uses
   the reranker's scores.

---

## 1. Chunking Strategy Comparison (mean +/- std, 3 runs, OpenAI gpt-4o-mini)

Command: `python main.py --runs 3`

| Chunking Strategy | Correctness (mean+/-std) | Faithfulness (mean+/-std) | Retrieval Relevance (mean+/-std) | Citation Accuracy (mean+/-std) | Fallback Rate (mean+/-std) |
|---|---|---|---|---|---|
| TokenRecursive | 0.997 +/- 0.005 | 1.000 +/- 0.000 | 0.983 +/- 0.005 | 0.975 +/- 0.000 | 0.400 +/- 0.000 |
| Markdown | 0.997 +/- 0.005 | 1.000 +/- 0.000 | 0.980 +/- 0.000 | 0.978 +/- 0.002 | 0.400 +/- 0.000 |
| Semantic | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.979 +/- 0.008 | 0.964 +/- 0.030 | 0.452 +/- 0.073 |

Winner (Retrieval Relevance, mean): **TokenRecursive** -- also the project
default. Winner (Citation Accuracy, mean): **Markdown** (0.978), ahead of
TokenRecursive (0.975) and Semantic (0.964) -- Semantic is the only strategy
whose own run-to-run std (+/-0.030) is larger than the gap between any pair
of strategies here, so its citation-accuracy number is the least trustworthy
of the three.

This is real output from `run_strategy_comparison`, 3 full passes (3
strategies x 14 questions x 3 runs = 126 pipeline runs, roughly 900-1,100
OpenAI gpt-4o-mini calls counting generation, confidence scoring, citation
verification, and the three judge metrics per question).

**Two bugs found and fixed while producing this table:** first, the
original `run_strategy_comparison` deleted and recreated each strategy's
Chroma persist directory (`shutil.rmtree` then rebuild) once per run, using
the same path every time. On Windows, Chroma's persisted index files can
stay open (via delayed garbage collection of the previous run's client
object) long enough that the second run's `rmtree` call hits a
`PermissionError` (`[WinError 32]`) -- this reliably crashed the original
`--runs 3` mid-way through run 2. `main.py` now gives each run its own
`./chroma_db_{strategy}_run{N}/` directory and retries `rmtree` a few times
before giving up, which is both correct and avoids the Windows-specific
race entirely.

Second: this environment's own background-process execution turned out to
kill long-running commands unpredictably (not a code bug -- an operational
quirk of the sandbox this was run in), and `run_strategy_comparison` had no
way to resume a killed `--runs 3` short of starting over from run 1. It now
saves each (strategy, run) cell to `results/strategy_comparison_results.csv`
immediately after computing it and skips cells already present there on the
next invocation -- the same resumability pattern already used by
`scripts/run_ablation.py`. This is what actually let the table above get
produced at all, across several interruptions and resumes with the same
`python main.py --runs 3` command. Single-run usage (`--runs 1`, the
default) was never
affected since it only ever touched each path once.

---

## 2. Retrieval / Reranker Ablation (mean +/- std, 3 runs, OpenAI gpt-4o-mini)

Command: `python scripts/run_ablation.py --runs 3`

Chunking strategy held fixed at `TokenRecursive` throughout so it isn't a
confound. Generator and judge LLM: OpenAI `gpt-4o-mini` -- the same model
Section 1 uses. Embeddings also OpenAI (`text-embedding-3-small`) for the
one shared index all six configs reuse. All 6 configs x 3 runs = 18 cells
completed successfully; raw data in `results/ablation_results.csv`,
aggregated in `results/ablation_summary.csv`.

**What actually happened, for the record.** This ablation originally
targeted OpenRouter's free-tier `nvidia/nemotron-3-super-120b-a12b:free`
model specifically to avoid OpenAI spend across the repeated eval-suite
runs an ablation requires. The very first call hit `429 Rate limit
exceeded: free-models-per-day (X-RateLimit-Limit: 50, remaining: 0)` --
OpenRouter's free tier caps out at 50 requests/day, and a single (config,
run) cell over these 14 questions costs on the order of 90 calls, so even
one cell doesn't fit in a day's free quota, let alone the 18 needed here.
After confirming that math, the script was switched to OpenAI gpt-4o-mini
(`scripts/run_ablation.py`'s `base_llm` construction), which is what
actually produced every row in `results/ablation_results.csv` -- the
single OpenRouter 429 error row still visible at the top of that CSV is
the one leftover artifact from the original attempt, kept rather than
deleted since the script's resume logic correctly treats an errored row as
not-done and retries it. The retry/backoff plumbing described below was
built and tested against the OpenRouter attempt, then left in place
(harmless, still useful for transient API errors) after the provider
switch.

### 2a. Retrieval mode: dense-only vs sparse-only vs hybrid (reranker on)

| Config | Retrieval Relevance | Correctness | Citation Accuracy | Failure Rate |
|---|---|---|---|---|
| Dense-only (1.0/0.0) | 0.980 +/- 0.000 | 0.993 +/- 0.006 | 0.975 +/- 0.000 | 0.286 +/- 0.000 |
| Sparse-only (0.0/1.0) | 0.980 +/- 0.000 | 0.990 +/- 0.000 | 0.975 +/- 0.000 | 0.286 +/- 0.000 |
| Hybrid 0.7/0.3 (default) | 0.980 +/- 0.000 | 0.997 +/- 0.006 | 0.978 +/- 0.003 | 0.286 +/- 0.000 |

![Retrieval mode comparison](results/ablation_retrieval_mode.png)

Retrieval relevance is identical (0.980) across all three modes -- on this
12-chunk corpus, dense embeddings and BM25 apparently surface essentially
the same top candidates, so fusion weighting has nothing to disagree about.
Hybrid edges out on correctness but the gap (0.997 vs 0.990) is within
noise of the other two strategies' zero std. **On a corpus this small, this
ablation doesn't support a strong claim that hybrid retrieval beats
dense-only or sparse-only** -- it would need a larger, more heterogeneous
corpus (or harder multi-hop-style questions where dense and sparse
retrieval genuinely disagree) to show a real gap.

### 2b. Reranker on vs off (hybrid 0.7/0.3 weights)

| Config | Retrieval Relevance | Correctness | Citation Accuracy | Failure Rate |
|---|---|---|---|---|
| Reranker ON (default) | 0.980 +/- 0.000 | 0.997 +/- 0.006 | 0.978 +/- 0.003 | 0.286 +/- 0.000 |
| Reranker OFF | 0.989 +/- 0.005 | 0.925 +/- 0.014 | 0.981 +/- 0.002 | 0.143 +/- 0.000 |

![Reranker on vs off](results/ablation_reranker_on_off.png)

The one genuinely interesting result in this ablation. With the reranker
off, `get_relevant_documents` returns the top-5 RRF-fused candidates using
the raw fusion score in place of a cross-encoder score. Two things follow
from that, both visible in the numbers: (1) the LLM-judged retrieval
relevance is actually slightly *higher* without reranking (0.989 vs 0.980)
-- on this corpus RRF fusion alone already surfaces good context, so
reranking isn't adding much signal; and (2) `score_confidence`'s
`retrieval_confidence` term, sigmoid-normalized from those raw RRF scores
instead of cross-encoder logits, behaves differently under the 0.75
confidence threshold, which changes *which* questions get answered at all
-- fewer fallbacks (14.3% vs 28.6%) but at a real cost to correctness on the
questions that do get answered (0.925 vs 0.997, and the only config with a
meaningfully nonzero correctness std, +/-0.014, meaning it was also the
least stable across the 3 runs). **Net read: the reranker isn't earning its
keep on retrieval relevance for this corpus, but it is doing real work
keeping the confidence gate calibrated** -- removing it trades a lower
fallback rate for materially worse correctness on the answers it does give.

### 2c. RRF dense/sparse weight sweep (reranker on)

| Config | Retrieval Relevance | Correctness | Citation Accuracy | Failure Rate |
|---|---|---|---|---|
| 0.5 / 0.5 | 0.983 +/- 0.006 | 0.990 +/- 0.000 | 0.983 +/- 0.014 | 0.286 +/- 0.000 |
| 0.7 / 0.3 (default) | 0.980 +/- 0.000 | 0.997 +/- 0.006 | 0.978 +/- 0.003 | 0.286 +/- 0.000 |
| 0.3 / 0.7 | 0.980 +/- 0.000 | 0.993 +/- 0.006 | 0.977 +/- 0.003 | 0.286 +/- 0.000 |

![RRF weight sweep](results/ablation_rrf_sweep.png)

All three weightings land within about 0.003-0.005 of each other on
retrieval relevance -- again consistent with 2a's finding that this small
corpus doesn't give the dense/sparse balance much to disagree about. The
project's default (0.7/0.3) is a reasonable, defensible choice here, but
this sweep doesn't provide strong evidence it's meaningfully better than
0.5/0.5 or 0.3/0.7 on this corpus specifically.

### Resilience notes (retry/backoff design, from the original OpenRouter attempt)

Before the provider switch, the free `nvidia/nemotron-3-super-120b-a12b:free`
backend was observed failing in more ways than plain 429s: intermittent 502
"Service temporarily overloaded" errors, and a raw `TypeError` inside the
OpenAI SDK's response parser when the backend returned a 200-status body
with `choices: null`. Neither carries "429" or "rate limit" in its message,
so `scripts/run_ablation.py`'s backoff retries everything except a small
denylist of permanent errors (bad API key, unknown model, 401/403/404)
rather than matching only rate-limit text -- narrower matching missed both
failure modes in practice. This logic stayed in place after switching to
OpenAI since it's harmless and still useful for ordinary transient API
errors. Separately, results are appended to `results/ablation_results.csv`
after every (config, run) cell, and a cell is only counted as done (skipped
on resume) if it succeeded -- this is what let the OpenAI run above survive
this environment's own unrelated background-process interruptions and
resume cleanly each time, rather than restarting from scratch.

---

## 3. Judge-Validation Scaffold

**Status: built, not yet validated.** `scripts/validate_judge.py --generate`
ran the default OpenAI pipeline once against all 14 frozen questions and
wrote `results/judge_validation_template.json`: for each question, the real
generated answer, the judge's own `judge_correctness_score` (and its
reasoning), and an explicit `human_label: null` field.

No human labels were fabricated -- that would defeat the point of
validating the judge. A human reviewer needs to read each
question/expected_answer/generated_answer and fill in `human_label` (a
float 0.0-1.0 on the same scale) by hand; `python scripts/validate_judge.py
--check-agreement` then reports mean absolute error, a near-match rate, and
Pearson correlation between the judge and the human labels. Run with zero
labels filled in, it correctly prints "no human labels yet" and exits
cleanly rather than crashing or inventing a number.

Observed while generating the template (informational, not a validation
result): all 6 Lookup and all 4 Multi-Hop questions were answered
successfully with `judge_correctness_score = 1.0`; all 2 Unanswerable and
both Ambiguous questions correctly triggered the pipeline's
"Insufficient Information" fallback (confidence below threshold), so they
have no judge score in the template and need a human reviewer's
`pipeline_status`-aware judgment call rather than a 0-1 correctness label.

---

## 4. Known Caveats

- The frozen eval set has 14 questions, not 15 (see above).
- Question/answer phrasing in the frozen set is not perfectly reproducible
  on regeneration even with the chunk-selection seed fixed, because OpenAI
  API sampling is not fully deterministic at temperature 0.
- `AdvancedRAGSystem.generate_robust_answer`'s confidence gate means a
  question that falls below `confidence_threshold` (0.75) never reaches
  `measure_retrieval_relevance` / `measure_faithfulness` /
  `measure_correctness` -- it's recorded as a failure instead. A config with
  a higher fallback/failure rate therefore has its average metrics computed
  over a smaller, easier subset of questions, which can make it look
  artificially strong on the metrics it does report. Failure rates are
  reported alongside every table above for this reason.
- The judge-validation scaffold is unvalidated until a human fills in
  `human_label`; treat the judge's own scores throughout this document as
  provisional, not ground truth.
- Section 1 (chunking comparison) and Section 2 (ablation) both use OpenAI
  `gpt-4o-mini` as generator and judge, so their numbers are on the same
  scale where the nominal configuration matches (e.g. Section 1's
  TokenRecursive row and Section 2's `hybrid_default` row both use the
  project's default retrieval settings). They aren't from the exact same
  run, though, so small differences between them reflect ordinary run-to-run
  variance, not a real effect.
