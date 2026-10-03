# rec-l2-bench

L2 reranker benchmark: cross-encoder rerankers (Qwen3-Reranker-0.6B, bge-reranker-v2-m3)
vs decision models (Jev, Clef, CLM, OpenAI Decisions) on one frozen L1 candidate pool
from Amazon Reviews'23 Video_Games (5-core, timestamp split).

Primary metric (pre-registered): **NDCG@10 on the test cohort**, paired bootstrap vs
ANN-only with Holm correction.

## Setup

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
| Jev / Clef / OpenAI backends | **Stubs**: request shapes come from each API reference |
| CLM | **Stub**: needs clm-serve API |
| L1 embedding + retrieval, profiles, baselines, report | Not started |

## Before the first real run

- Pin every `revision: null` in `configs/` to a commit hash.
- Implement `Backend.ask` for Jev and Clef from their API docs; confirm the probability
  field name (the plan's "Noul value").
- Download Video_Games 5-core timestamp files and metadata into `data/raw/`.
