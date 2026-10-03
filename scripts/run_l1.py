"""Step 2: embed items, retrieve top-200, check the gate, freeze pools.

    # tune half-life on validation (Recall@100), then freeze both splits:
    python scripts/run_l1.py --tune-half-life 30,90,180,365
    python scripts/run_l1.py --half-life 180 --index hnsw --freeze
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.data import leakage_audit as LA  # noqa: E402
from recl2bench.data.load import load_ratings  # noqa: E402
from recl2bench.l1 import pools as P  # noqa: E402
from recl2bench.l1 import retrieve as R  # noqa: E402
from recl2bench.l1.embed import STEncoder, embed_items  # noqa: E402


def load_split(proc: Path, split: str):
    cohort = pd.read_parquet(proc / f"cohort_{split}.parquet")
    pos = pd.read_parquet(proc / f"positives_{split}.parquet").groupby("user_id")["parent_asin"].apply(set).to_dict()
    return cohort, pos


def retrieve_split(split, ratings, items, V, item_index, fs, proc, half_life, m, index_kind, cfg_gate):
    cohort, pos = load_split(proc, split)
    users, U, fallback = R.user_vectors(ratings, cohort, item_index, V, half_life)
    ids = items.parent_asin.tolist()
    mask = R.eligibility(users, cohort, ratings, ids, fs)
    ex_idx, ex_sc = R.exact_topm(U, V, mask, m)
    info = {"split": split, "half_life_days": half_life, "users_no_liked_fallback": len(fallback),
            "users_short_pool_exact": int((ex_idx[:, -1] < 0).sum())}
    idx, sc = ex_idx, ex_sc
    if index_kind == "hnsw":
        h_idx, h_sc, short = R.hnsw_topm(U, V, mask, m)
        info["ann_overlap@100"] = R.overlap_at_k(h_idx, ex_idx, 100)
        info["users_short_pool_hnsw"] = short
        info["gate_passed"] = info["ann_overlap@100"] >= cfg_gate
        idx, sc = h_idx, h_sc
    pools = R.to_pool_frame(users, cohort, idx, sc, ids)
    qt = cohort.set_index("user_id")["query_time"]
    q_ms = {u: qt[u].value // 1_000_000 for u in users}
    prior = ratings[ratings.user_id.isin(set(users))]
    prior = prior[prior.timestamp < prior.user_id.map(q_ms)]
    seen = prior.groupby("user_id")["parent_asin"].apply(set).to_dict()
    info.update(R.positive_diagnostics(pools, pos, users, fs, qt, seen))
    return pools, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--l1-config", default="configs/l1.yaml")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--pools-dir", default="data/pools")
    ap.add_argument("--half-life", type=float)
    ap.add_argument("--tune-half-life", help="comma list; reports validation Recall@100 only")
    ap.add_argument("--index", choices=["exact", "hnsw"])
    ap.add_argument("--freeze", action="store_true", help="write and hash pools (irreversible)")
    ap.add_argument("--encoder", default=None, help="override; 'fake' for tests")
    a = ap.parse_args(argv)

    dcfg = yaml.safe_load(Path(a.dataset_config).read_text())
    lcfg = yaml.safe_load(Path(a.l1_config).read_text())
    proc = Path(a.proc)
    m = lcfg["pool"]["stored_top_m"]
    index_kind = a.index or lcfg["index"]["kind"]

    ratings = load_ratings(a.raw, dcfg["category"])
    items = pd.read_parquet(proc / "items.parquet")
    fs = items.set_index("parent_asin")["first_seen"]
    item_index = {x: i for i, x in enumerate(items.parent_asin)}
    if a.encoder == "fake":
        from recl2bench.l1.embed import normalize

        class Fake:
            name = "fake"
            def encode(self, texts):
                rng = np.random.default_rng(0)
                return normalize(rng.normal(size=(len(texts), 16)).astype(np.float32))
        enc = Fake()
    else:
        enc = STEncoder(lcfg["encoder"]["model"], lcfg["encoder"]["revision"])
    V, emb_key = embed_items(items.parent_asin.tolist(), items.text.tolist(), enc, proc)

    if a.tune_half_life:
        rows = []
        for hl in [float(x) for x in a.tune_half_life.split(",")]:
            _, info = retrieve_split("valid", ratings, items, V, item_index, fs, proc, hl, m, "exact", 0)
            rows.append({"half_life_days": hl, **{k: info[k] for k in ("recall@50", "recall@100", "recall@200")}})
        print(pd.DataFrame(rows).to_string(index=False))
        return

    hl = a.half_life or lcfg["user_vector"]["half_life_days"]
    reports = {"encoder": enc.name, "embedding_cache_key": emb_key, "index": index_kind, "splits": {}}
    for split in ("valid", "test"):
        pools, info = retrieve_split(split, ratings, items, V, item_index, fs, proc, hl, m,
                                     index_kind, lcfg["gate"]["min_ann_overlap_at_100"])
        errs = LA.check_pools(pools, ratings, fs)
        if errs:
            raise SystemExit(f"leakage audit failed for {split}: {errs}")
        if index_kind == "hnsw" and split == "valid" and not info["gate_passed"]:
            raise SystemExit(f"ANN gate failed: overlap@100={info['ann_overlap@100']:.3f}")
        if a.freeze:
            info["pool_sha256"] = P.freeze(pools, Path(a.pools_dir) / f"pools_{split}.parquet")
        reports["splits"][split] = info
    out = Path(a.pools_dir if a.freeze else proc) / "l1_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2, default=str))
    print(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    main()
