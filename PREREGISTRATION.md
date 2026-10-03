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
