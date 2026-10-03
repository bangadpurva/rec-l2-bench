"""Screen candidate categories on VALIDATION only, without embeddings.

    python scripts/screen_categories.py
    python scripts/screen_categories.py --categories Musical_Instruments,Baby_Products --n-valid 2000

For each category: downloads the three rating files, checks the split cut points,
samples validation users with the benchmark's cohort rules, and reports how much of
the outcome is reachable and how well popularity and co-occurrence L1 channels do.
Test-split sizes are counted (eligible users only), never scored.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.data.load import load_ratings  # noqa: E402
from recl2bench.data.split import (TEST_START_MS, VALID_START_MS, first_seen,  # noqa: E402
                                   positives, select_cohort)
from recl2bench.l1 import channels as CH  # noqa: E402
from recl2bench.l1 import retrieve as R  # noqa: E402

DEFAULT = "Industrial_and_Scientific,Musical_Instruments,Baby_Products,Office_Products"
ROOT = Path(__file__).resolve().parents[1]


def check_cuts(r: pd.DataFrame) -> str:
    tr, va, te = (r[r.source_split == s].timestamp for s in ("train", "valid", "test"))
    ok = tr.max() < VALID_START_MS <= va.min() and va.max() < TEST_START_MS <= te.min()
    return "ok" if ok else f"MISMATCH train<={tr.max()} valid={va.min()}..{va.max()} test>={te.min()}"


def screen(cat: str, raw_root: Path, n_valid: int, seed: int, chunk: int, download: bool) -> dict:
    d = raw_root / cat
    if download:
        subprocess.run(["bash", str(ROOT / "scripts/download_data.sh"), cat, str(d), "ratings"], check=True)
    r = load_ratings(d, cat)
    out = {"category": cat, "users": r.user_id.nunique(), "items": r.parent_asin.nunique(),
           "ratings": len(r), "cut_points": check_cuts(r)}
    full_v = select_cohort(r, "valid")
    full_t = select_cohort(r, "test")
    out["eligible_valid_users"], out["eligible_test_users"] = len(full_v), len(full_t)
    cohort = select_cohort(r, "valid", n_users=n_valid, seed=seed)
    pos = positives(r, "valid")
    fs = first_seen(r)
    items = sorted(r.parent_asin.unique())
    item_index = {a: i for i, a in enumerate(items)}
    users = sorted(cohort.user_id)
    n_pos = sum(len(pos[u]) for u in users)
    q = cohort.query_time.iloc[0]
    new = sum(1 for u in users for a in pos[u] if not (a in fs.index and fs[a] < q))
    out.update({"sampled_valid_users": len(users), "positives_per_user": n_pos / len(users),
                "unreachable_share": new / n_pos})

    acc = {k: {"recall": [], "hit": []} for k in ("pop", "cooc", "cooc+pop")}
    cache: dict = {}
    for s in range(0, len(users), chunk):
        us = users[s:s + chunk]
        co = cohort[cohort.user_id.isin(us)]
        mask = R.eligibility(us, co, r, items, fs)
        lists = {"pop": CH.topm_from_scores(CH.popularity_scores(r, co, us, item_index, 90), mask, 100)[0]}
        S = CH.cooccurrence_scores(r, co, us, item_index, cache=cache)
        S[S <= 0] = -np.inf
        lists["cooc"] = CH.topm_from_scores(S, mask, 100)[0]
        lists["cooc+pop"] = CH.interleave({"cooc": lists["cooc"], "pop": lists["pop"]}, 100)[0]
        for k, idx in lists.items():
            for i, u in enumerate(us):
                got = {items[j] for j in idx[i] if j >= 0}
                f = len(pos[u] & got)
                acc[k]["recall"].append(f / len(pos[u]))
                acc[k]["hit"].append(f > 0)
    for k, v in acc.items():
        out[f"{k}_recall@100"] = float(np.mean(v["recall"]))
        out[f"{k}_users_with_pos@100"] = float(np.mean(v["hit"]))
    best = max(out[f"{k}_users_with_pos@100"] for k in acc)
    out["est_test_eval_users"] = int(best * out["eligible_test_users"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=DEFAULT)
    ap.add_argument("--raw-root", default="data/screen")
    ap.add_argument("--n-valid", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args(argv)
    rows = []
    for cat in a.categories.split(","):
        print(f"--- {cat}", flush=True)
        rows.append(screen(cat.strip(), Path(a.raw_root), a.n_valid, a.seed, a.chunk, not a.no_download))
        print(pd.Series(rows[-1]).to_string(), flush=True)
    t = pd.DataFrame(rows).set_index("category")
    out = Path(a.raw_root) / "screen_validation.csv"
    t.to_csv(out)
    cols = ["items", "cut_points", "eligible_test_users", "positives_per_user", "unreachable_share",
            "pop_users_with_pos@100", "cooc_users_with_pos@100", "cooc+pop_users_with_pos@100",
            "cooc+pop_recall@100", "est_test_eval_users"]
    print("\n== Validation screen ==")
    print(t[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    return t


if __name__ == "__main__":
    main()
