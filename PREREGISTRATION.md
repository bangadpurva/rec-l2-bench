# Pre-registration

## Original (2026-10-02, project plan)

- Primary metric: NDCG@10 on the test cohort, all users.
- Decision rule: a model beats ANN-only if its paired-bootstrap 95% interval for the
  NDCG@10 difference excludes zero after Holm correction (10,000 resamples over users).
- L1: dense ANN (Qwen3-Embedding-0.6B, recency-weighted user vector), top-200 frozen;
  gate on ANN overlap@100 >= 95% and an L1 recall check before any reranker runs.
- Labels: rating >= 4 in the outcome window; rating = 5 as a sensitivity check.

## Amendment 1 (2026-10-03), made before any L2 model was run

**What was observed.** Only L1 diagnostics had been computed; no reranker scores existed.
Dense-only L1 on the one-day cohorts (100 validation, 500 test users) gave test
Recall@100 = 2.9% against a reachable ceiling of 68% (32% of test positives are items
first seen after the query time). Only 5.8% of test users had any positive in their
top-100 pool, so about 94% of users would score 0 under every reranker. The L1 gate
in the original plan therefore failed.

**Disclosure.** L1 recall diagnostics on the test split were inspected during development
(per-channel Recall@K and users-with-positive@K). No L2 result on test was seen.

**Changes.**
1. Cohorts enlarged to 2,000 validation and 3,000 test users (fixed seed 20261002).
2. L1 pools may merge channels (dense, item co-occurrence, 90-day popularity), all
   computed strictly from interactions before the query time. The channel mix is
   chosen on **validation only** and recorded in `data/pools/l1_report.json`.
3. **Primary metric: NDCG@10 on the conditional test cohort**: users whose frozen
   top-100 pool contains at least one positive. The cohort depends only on L1 and
   labels, is identical for every model, and is frozen and hashed with the pools
   (`data/pools/eval_users_test.parquet`).
4. Secondary: NDCG@10 over all test users, derived exactly from conditional runs
   (users outside the cohort score 0 under every model), plus the original secondary
   metrics.

The decision rule (paired bootstrap, Holm, vs L1 order) is unchanged.

## Amendment 2 (2026-10-03), made before any L2 model was run

**Category.** Video_Games is dropped. Four candidate categories were screened on
**validation only** (2,000 users each, popularity and co-occurrence L1, no embeddings;
`scripts/screen_categories.py`). Musical_Instruments is the benchmark category: it had
the lowest unreachable share by a wide margin (9% of validation positives first seen
after the query time, vs 17-23% elsewhere) and the second-highest share of users with a
positive in a top-100 pool (14.7%). Baby_Products was slightly higher on that share
(15.8%) but nearly twice as many unreachable positives and a weak co-occurrence channel. No test
metric was computed for any candidate; only eligible test-user counts were read.

| Category | Unreachable | Users with positive @100 (best channel) |
|---|---|---|
| Musical_Instruments | 9.1% | 14.7% |
| Baby_Products | 17.8% | 15.8% |
| Industrial_and_Scientific | 23.3% | 10.6% |
| Office_Products | 16.6% | 8.1% |

**Test cohort.** All eligible test users (>= 3 prior interactions, >= 1 positive in the
outcome window) instead of a 3,000-user sample, to maximise the conditional evaluation
cohort. Validation stays at 2,000 users.

Everything else in Amendment 1 is unchanged.

## Amendment 3 (2026-10-03): L1 settings, chosen on validation only

Musical_Instruments, 2,000 validation users, share of users with a positive in the
top-100 pool:

| Pool | Users with positive @100 |
|---|---|
| dense | 6.6% |
| cooc | 6.4% |
| pop | 14.7% |
| dense+pop | 14.0% |
| cooc+pop | 13.2% |
| dense+cooc+pop | 13.0% |

**Pool: dense+pop** (interleaved, dense first), half-life 180 days (30/90/180/365 gave
6.25-6.60%, within noise), HNSW index. Popularity alone scored 0.7 points higher, a gap
within sampling error (about ±0.8 points at n=2,000), but a popularity-only pool is
the same 100 trending items for every user at one query time, and its L1 order would
be identical to the popularity baseline. dense+pop keeps candidates personalised and
keeps L1 order and popularity as distinct baselines. The test table printed by the
same command was not used for this choice.

## Frozen L1 artifacts (2026-10-03 09:23 EDT), canonical

Frozen once, on an Apple M-series Pro (MPS), with `l1-freeze` (dense+pop, half-life 180,
HNSW). Every model is scored against these files; `score.py` verifies the hashes on load.

| File | sha256 | Users |
|---|---|---|
| `pools_valid.parquet` | `0601046bc4f13aa9e972dff6cb2939bc2c19c0be7b5ae9cc95d1f0775dc72d13` | 2,000 |
| `pools_test.parquet` | `68ec129ae4ee905df7721912d2bd73967ae9d50113452cc65a1805c120d70cd4` | 9,104 |
| `eval_users_valid.parquet` | `a7cafc5c6040a3201761ea9234f3e2b1db4590b319fbaf81811927076949c9ae` | 279 |
| `eval_users_test.parquet` | `59a6bba45219fc93ab6fe4b35cfd7b0bf17977631e148fd7a3d4eaa1839ef73c` | 1,200 |

The freeze ran before model revisions were pinned in the configs, so its report shows
the encoder as `@None`. Hugging Face `main` for Qwen/Qwen3-Embedding-0.6B was last changed
2026-04-20, so the effective revision is `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, the
value now pinned in `configs/l1.yaml` and used for the budget tokenizer.

## Amendment 4 (2026-10-03): Track C fusion, fixed before any fusion test run

**Disclosure.** Track A test results were known when this was written: BGE and
Qwen3-Reranker scored below L1 order on test NDCG@10, and a test-set diagnostic
(`scripts/diagnose_runs.py`) showed why: 67% of in-pool positives came from the
popularity channel, and both rerankers moved popularity-sourced items out of the top 10
while improving the ranks of dense-sourced positives. Fusion is motivated by that
observation, so Track C test results are **exploratory**, not a pre-registered
confirmatory test.

**Method, fixed now.** For each reranker R in {BGE, Qwen3} and partner P in
{L1 order, popularity}: weighted reciprocal-rank fusion,
`score = w/(60 + rank_R) + (1 - w)/(60 + rank_P)`, ranks 1-based, ties by L1 rank.
`w` is chosen from {0, 0.1, ..., 1.0} by mean NDCG@10 on the **validation** eval cohort
(279 users), ties to the smaller w. Each pair is then scored **once** on the test eval
cohort with that w. The grid is saved to `results/<CAT>/fusion_tuning.csv`.

Fusion runs enter the same report and Holm family as Track A (four extra comparisons),
which makes every adjusted p-value more conservative.

## Amendment 5 (2026-10-03): Clef Flash, fixed before any Clef test run

Only a 20-user **validation** smoke test of Clef had been run (NDCG@10 not used).

- **Model and mode:** `@cf/cloudflare/clef-flash`, per-pair scoring (one yes/no `noul`
  question per candidate, template `yes_no_v1`), concurrency 32, exponential backoff
  with jitter on HTTP 429/529 (base 1 s, cap 30 s, up to 6 retries). The response
  field `noul` is the score. Failed candidates rank last and are counted.
- **Track A:** Clef on the test eval cohort (1,200 users), compared with L1 order under
  the original decision rule.
- **Track C:** Clef fused with L1 order and with popularity by the Amendment 4 method,
  weight chosen on the validation eval cohort (279 users). Because this is fixed before
  any Clef test result, Clef fusion is **pre-registered**, unlike BGE/Qwen3 fusion.
- Latency is per-user wall clock at concurrency 32 from one client; total backoff time
  and billed input tokens are recorded in each run manifest.

## Amendment 6 (2026-10-03): Jev, fixed before any Jev call

- **Model and mode:** TypeSafe `jev-1.13.0` (pinned; aliases refused), `POST
  /v1/systemone`, per-pair scoring with the same `yes_no_v1` noul question, state and
  backoff as Clef (Amendment 5), concurrency 32 (documented limit 80 req/s).
- **Track A:** Jev on the test eval cohort vs L1 order, original decision rule.
- **Track C:** Jev fused with L1 order and with popularity by the Amendment 4 method,
  weight chosen on validation. Pre-registered, as for Clef.
- Every response's `model` field is logged; a run is invalid if it reports any model
  other than `jev-1.13.0`.
- *Update before any Jev evaluation run:* the 20-user smoke test showed throughput
  limited by per-call latency (about 13 req/s at concurrency 32, roughly 2.4 s per call),
  well under the 80 req/s limit, so Jev runs use **concurrency 96**. Per-user latency is
  reported at that concurrency; it is a wall-clock figure for one client, not model
  compute time.
