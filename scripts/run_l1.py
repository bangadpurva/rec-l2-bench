"""Step 2: embed items, retrieve top-200 per channel, merge, check the gate, freeze pools.

    python scripts/run_l1.py --tune-half-life 30,90,180,365       # dense only, validation
    python scripts/run_l1.py --half-life 180 --compare-channels    # recall per channel and merge
    python scripts/run_l1.py --half-life 180 --index hnsw --channels dense,cooc,pop          # dry run
    python scripts/run_l1.py --half-life 180 --index hnsw --channels dense,cooc,pop --freeze # irreversible
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
from recl2bench.l1 import channels as CH  # noqa: E402
from recl2bench.l1 import pools as P  # noqa: E402
from recl2bench.l1 import retrieve as R  # noqa: E402
from recl2bench.l1.embed import STEncoder, embed_items, normalize  # noqa: E402

CHANNELS = ("dense", "cooc", "pop")


def load_split(proc: Path, split: str):
    cohort = pd.read_parquet(proc / f"cohort_{split}.parquet")
    pos = pd.read_parquet(proc / f"positives_{split}.parquet").groupby("user_id")["parent_asin"].apply(set).to_dict()
    return cohort, pos


class Ctx:
    """Everything one split needs, computed once."""

    def __init__(self, split, ratings, items, V, item_index, fs, proc, half_life, m):
        self.split, self.ratings, self.V, self.item_index, self.m = split, ratings, V, item_index, m
        self.ids = items.parent_asin.tolist()
        self.fs = fs
        self.cohort, self.pos = load_split(proc, split)
        self.users, self.U, self.fallback = R.user_vectors(ratings, self.cohort, item_index, V, half_life)
        self.mask = R.eligibility(self.users, self.cohort, ratings, self.ids, fs)
        self.half_life = half_life
        self._lists: dict[str, tuple] = {}

    def channel(self, name: str, index_kind: str = "exact"):
        key = f"{name}:{index_kind}"
        if key not in self._lists:
            info = {}
            if name == "dense" and index_kind == "hnsw":
                ex, _ = R.exact_topm(self.U, self.V, self.mask, self.m)
                idx, sc, short = R.hnsw_topm(self.U, self.V, self.mask, self.m)
                info = {"ann_overlap@100": R.overlap_at_k(idx, ex, 100), "users_short_pool_hnsw": short}
            elif name == "dense":
                idx, sc = R.exact_topm(self.U, self.V, self.mask, self.m)
            elif name == "pop":
                S = CH.popularity_scores(self.ratings, self.cohort, self.users, self.item_index, 90)
                idx, sc = CH.topm_from_scores(S, self.mask, self.m)
            elif name == "cooc":
                S = CH.cooccurrence_scores(self.ratings, self.cohort, self.users, self.item_index,
                                           half_life_days=self.half_life)
                S[S <= 0] = -np.inf          # no co-occurrence -> not a candidate from this channel
                idx, sc = CH.topm_from_scores(S, self.mask, self.m)
            else:
                raise ValueError(name)
            self._lists[key] = (idx, sc, info)
        return self._lists[key]

    def recall(self, idx):
        return R.recall_table(idx, self.users, self.pos, self.ids)


def compare(ctx: Ctx) -> pd.DataFrame:
    rows = []
    lists = {c: ctx.channel(c)[0] for c in CHANNELS}
    for c in CHANNELS:
        rows.append({"pool": c, **ctx.recall(lists[c])})
    for combo in (("dense", "pop"), ("dense", "cooc"), ("cooc", "pop"), ("dense", "cooc", "pop")):
        idx, _ = CH.interleave({c: lists[c] for c in combo}, ctx.m)
        rows.append({"pool": "+".join(combo), **ctx.recall(idx)})
    return pd.DataFrame(rows)


def build_pool(ctx: Ctx, chans: list[str], index_kind: str, gate: float):
    info = {"split": ctx.split, "half_life_days": ctx.half_life, "channels": chans,
            "users_no_liked_fallback": len(ctx.fallback)}
    lists = {}
    for c in chans:
        idx, sc, extra = ctx.channel(c, index_kind if c == "dense" else "exact")
        lists[c] = idx
        info.update(extra)
        info[f"channel_{c}"] = ctx.recall(idx)
    if "ann_overlap@100" in info:
        info["gate_passed"] = info["ann_overlap@100"] >= gate
    if len(chans) == 1:
        idx, sc, _ = ctx.channel(chans[0], index_kind if chans[0] == "dense" else "exact")
        src = np.full(idx.shape, chans[0], dtype=object)
    else:
        idx, src = CH.interleave(lists, ctx.m)
        sc = np.where(idx >= 0, 1.0 / (np.arange(ctx.m)[None, :] + 1.0), -np.inf)
    pools = R.to_pool_frame(ctx.users, ctx.cohort, idx, sc, ctx.ids, src)
    info["source_share_top100"] = pools[pools["rank"] < 100].source.value_counts(normalize=True).round(4).to_dict()
    qt = ctx.cohort.set_index("user_id")["query_time"]
    q_ms = {u: qt[u].value // 1_000_000 for u in ctx.users}
    prior = ctx.ratings[ctx.ratings.user_id.isin(set(ctx.users))]
    prior = prior[prior.timestamp < prior.user_id.map(q_ms)]
    seen = prior.groupby("user_id")["parent_asin"].apply(set).to_dict()
    info.update(R.positive_diagnostics(pools, ctx.pos, ctx.users, ctx.fs, qt, seen))
    return pools, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--l1-config", default="configs/l1.yaml")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--pools-dir", default="data/pools")
    ap.add_argument("--half-life", type=float)
    ap.add_argument("--tune-half-life", help="comma list; dense-only validation recall")
    ap.add_argument("--compare-channels", action="store_true", help="recall per channel and merge; no pools written")
    ap.add_argument("--channels", default="dense", help="comma list from dense,cooc,pop (merge order)")
    ap.add_argument("--index", choices=["exact", "hnsw"])
    ap.add_argument("--freeze", action="store_true", help="write and hash pools (irreversible)")
    ap.add_argument("--encoder", default=None, help="override; 'fake' for tests")
    a = ap.parse_args(argv)

    dcfg = yaml.safe_load(Path(a.dataset_config).read_text())
    lcfg = yaml.safe_load(Path(a.l1_config).read_text())
    proc = Path(a.proc)
    m = lcfg["pool"]["stored_top_m"]
    index_kind = a.index or lcfg["index"]["kind"]
    chans = [c.strip() for c in a.channels.split(",")]
    assert all(c in CHANNELS for c in chans), chans

    ratings = load_ratings(a.raw, dcfg["category"])
    items = pd.read_parquet(proc / "items.parquet")
    fs = items.set_index("parent_asin")["first_seen"]
    item_index = {x: i for i, x in enumerate(items.parent_asin)}
    if a.encoder == "fake":
        class Fake:
            name = "fake"
            def encode(self, texts):
                return normalize(np.random.default_rng(0).normal(size=(len(texts), 16)).astype(np.float32))
        enc = Fake()
    else:
        enc = STEncoder(lcfg["encoder"]["model"], lcfg["encoder"]["revision"])
    V, emb_key = embed_items(items.parent_asin.tolist(), items.text.tolist(), enc, proc)
    hl = a.half_life or lcfg["user_vector"]["half_life_days"]

    if a.tune_half_life:
        rows = []
        for h in [float(x) for x in a.tune_half_life.split(",")]:
            ctx = Ctx("valid", ratings, items, V, item_index, fs, proc, h, m)
            rows.append({"half_life_days": h, **ctx.recall(ctx.channel("dense")[0])})
        print(pd.DataFrame(rows).to_string(index=False))
        return

    if a.compare_channels:
        for split in ("valid", "test"):
            ctx = Ctx(split, ratings, items, V, item_index, fs, proc, hl, m)
            t = compare(ctx)
            print(f"\n== {split} ({len(ctx.users)} users) ==")
            print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            t.to_csv(proc / f"l1_channels_{split}.csv", index=False)
        return

    reports = {"encoder": enc.name, "embedding_cache_key": emb_key, "index": index_kind,
               "channels": chans, "splits": {}}
    for split in ("valid", "test"):
        ctx = Ctx(split, ratings, items, V, item_index, fs, proc, hl, m)
        pools, info = build_pool(ctx, chans, index_kind, lcfg["gate"]["min_ann_overlap_at_100"])
        errs = LA.check_pools(pools, ratings, fs)
        if errs:
            raise SystemExit(f"leakage audit failed for {split}: {errs}")
        if split == "valid" and info.get("gate_passed") is False:
            raise SystemExit(f"ANN gate failed: overlap@100={info['ann_overlap@100']:.3f}")
        top = pools[pools["rank"] < P.EVAL_TOP_M].groupby("user_id")["parent_asin"].apply(set)
        info["eval_cohort_users"] = int(sum(bool(ctx.pos[u] & top.get(u, set())) for u in ctx.users))
        if a.freeze:
            info["pool_sha256"] = P.freeze(pools, Path(a.pools_dir) / f"pools_{split}.parquet")
            sha, n_eval, n_all = P.freeze_eval_cohort(
                pools, ctx.pos, Path(a.pools_dir) / f"eval_users_{split}.parquet")
            info.update({"eval_cohort_sha256": sha, "eval_cohort_users": n_eval, "cohort_users": n_all})
        reports["splits"][split] = info
    out = Path(a.pools_dir if a.freeze else proc) / "l1_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2, default=str))
    print(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    main()
