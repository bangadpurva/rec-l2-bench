# rec-l2-bench

L2 reranker benchmark: cross-encoder rerankers (Qwen3-Reranker-0.6B, bge-reranker-v2-m3)
vs decision models (Jev, Clef, CLM, OpenAI Decisions) on one frozen L1 candidate pool
from Amazon Reviews'23 (5-core, timestamp split). The category is chosen by a
validation-only screen (`scripts/screen_categories.py`); Video_Games was dropped after
its L1 diagnostics.

Primary metric (pre-registered, see `PREREGISTRATION.md`): **NDCG@10 on the conditional
test cohort** (users whose frozen top-100 pool holds a positive), paired bootstrap vs L1
order with Holm correction. All-user NDCG@10 is reported as secondary.

## Running it

**On Vast.ai (or any Linux GPU box): follow [`VASTAI.md`](VASTAI.md).** One setup script,
then `bash scripts/run_pipeline.sh <stage>` for each stage, with logs and backups.

## Setup (manual)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + tests
pip install -e ".[models,api]"     # GPU models and API clients, when needed
pytest
```

## Layout

```
configs/                 dataset.yaml, l1.yaml, models/*.yaml, templates/yes_no_v1.yaml
data/raw|processed|pools raw data (not committed); pools_<split>.parquet + .sha256 (frozen)
src/recl2bench/
  data/        split.py (timestamp split, cohort, positives, first-seen), leakage_audit.py
  l1/          pools.py (freeze + hash + verified load, 50/100 as prefixes of 200)
  rerankers/   base.py (the contract), text.py (shared budgets), dummy.py,
               cross_encoder.py (BGE, Qwen3), decision.py (Jev/Clef/OpenAI adapter), clm.py
  eval/        metrics.py, bootstrap.py
  manifest.py  one manifest.json per run
  runner.py    pool -> rerank -> per-user metrics
tests/                   metrics, bootstrap, adapter contract, leakage, end-to-end dummy run
```

The plan's `src/data`, `src/eval` etc. live under one package, `src/recl2bench/`, so
imports don't collide with generic names like `data`.

## The contract

```python
reranker.rerank(user_profile, candidate_items) -> RerankResult(scores=[...])
```

One finite score per candidate, in input order. A failed candidate is NaN **and** a
logged failure; the contract rejects unlogged NaNs. Failed candidates rank last, in L1 order.

## Status

| Component | State |
|---|---|
| Metrics, bootstrap + Holm | Done, tested against hand-computed values |
| Contract, dummy rerankers, runner | Done, tested end to end on synthetic pools |
| Split, cohort, leakage audit, pool freezing | Done, tested on a synthetic ratings table |
| BGE / Qwen3 cross-encoders | Written per model cards, **not yet run** (needs GPU) |
| Decision adapter (per-pair, fan-out, retries, failures) | Done, tested with a fake backend |
| Clef Flash backend | Done; verified against a live response (answer field `noul`) |
| Jev backend | Done per docs.typesafe.ai; shares the Clef protocol code; needs a live ping |
| OpenAI Decisions backend | **Stub**: no public spec yet |
| CLM | **Stub**: needs clm-serve API |
| Data loading, profiles, item text | Done, tested on synthetic files in the real format |
| L1 embed, exact + HNSW, gate, freeze | Done, tested with a fake encoder |
| L1 channels: co-occurrence, popularity, interleaved merge | Done, tested (incl. no future data) |
| Baselines: L1 order, random, popularity (90 d), oracle | Done, tested |
| Scoring + results report (paired bootstrap, Holm) | Done, tested end to end |
| SASRec, tabular baseline | Not started |

## Running L1 (GPU box)

```bash
pip install -e ".[models,dev]"
bash scripts/download_data.sh                          # ~4 files into data/raw/
python scripts/prepare_data.py                         # 2,000 valid / 3,000 test users
python scripts/run_l1.py --tune-half-life 30,90,180,365    # validation Recall@100 per half-life
python scripts/run_l1.py --half-life <best> --compare-channels   # recall per channel and merged pool
python scripts/run_l1.py --half-life <best> --index hnsw --channels dense,cooc,pop            # dry run
python scripts/run_l1.py --half-life <best> --index hnsw --channels dense,cooc,pop --freeze   # irreversible
```

Read `data/pools/l1_report.json` before any L2 work: Recall@50/100/200, ANN overlap,
and `max_reachable_share` (positives that are neither already-seen nor first seen after
query time). Recall can never exceed that ceiling. `users_with_pos@100` is the share of
users whose pool contains any positive; users without one score 0 under every reranker.

Dense-only L1 on Video_Games gave test Recall@100 = 2.9% against a 68% reachable ceiling,
so pools merge three channels (dense, item co-occurrence, 90-day popularity). Every model
still scores the same frozen pool, and each pool row records its `source` channel.

## Scoring and the results table

After pools are frozen (one run per model; `--limit-users N` runs are smoke tests and are
left out of the report):

```bash
for m in l1_order random popularity oracle; do python scripts/score.py --model $m --split test; done
python scripts/score.py --model bge   --split test
python scripts/score.py --model qwen3 --split test
python scripts/score.py --model clef  --split valid --limit-users 20   # live smoke test first
python scripts/score.py --model clef  --split test
python scripts/report.py --split test     # -> results/test_m100.md and .csv
```

## Before the first L2 run

