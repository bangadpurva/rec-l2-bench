"""Step 1: cohorts, positives, item text and profiles for valid and test.

    python scripts/prepare_data.py                       # 2,000 valid / 3,000 test (Amendment 1)
    python scripts/prepare_data.py --n-valid 200 --n-test 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.data import leakage_audit as LA  # noqa: E402
from recl2bench.data.load import load_meta, load_ratings  # noqa: E402
from recl2bench.data.profiles import build_items, build_profiles  # noqa: E402
from recl2bench.data.split import first_seen, positives, select_cohort  # noqa: E402
from recl2bench.tokenizer import get_tokenizer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-valid", type=int)
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--tokenizer", help="override budget tokenizer ('whitespace' for tests)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text())
    co, lab, prof, bud = cfg["cohort"], cfg["label"], cfg["profile"], cfg["budget"]
    n = {"valid": a.n_valid or co.get("valid_users", co["one_day_valid_users"]),
         "test": a.n_test or co.get("test_users", co["one_day_test_users"])}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer(a.tokenizer or bud["tokenizer"])

    ratings = load_ratings(a.raw, cfg["category"])
    fs = first_seen(ratings)
    meta = load_meta(a.raw, set(ratings.parent_asin.unique()), cfg["category"])
    missing_meta = len(set(ratings.parent_asin) - set(meta.parent_asin))
    items = build_items(meta, tok, bud["item_tokens"], fs)
    errs = LA.check_item_fields(items)
    assert not errs, errs
    items.to_parquet(out / "items.parquet", index=False)

    report = {"n_ratings": len(ratings), "n_users": ratings.user_id.nunique(),
              "n_items": len(items), "items_missing_meta": missing_meta,
              "items_truncated": int(items.truncated.sum()), "splits": {}}
    for split in ("valid", "test"):
        cohort = select_cohort(ratings, split, min_prior=co["min_prior_interactions"],
                               min_pos=co["min_positives_in_window"],
                               pos_rating=lab["positive_min_rating"], n_users=n[split], seed=co["seed"])
        pos = positives(ratings, split, lab["positive_min_rating"])
        pos5 = positives(ratings, split, lab["sensitivity_positive_rating"])
        profiles, summary = build_profiles(ratings, cohort, items, tok, bud["profile_tokens"],
                                           prof["max_liked"], prof["max_disliked"],
                                           lab["positive_min_rating"], lab["negative_max_rating"])
        qms = dict(zip(cohort.user_id, cohort.query_time.map(lambda t: t.value // 1_000_000)))
        assert all(summary.last_event_ms < summary.user_id.map(qms)), "profile leak"
        cohort.to_parquet(out / f"cohort_{split}.parquet", index=False)
        summary.drop(columns="last_event_ms").to_parquet(out / f"profiles_{split}.parquet", index=False)
        pd.DataFrame([(u, a_, int(a_ in pos5.get(u, set()))) for u in cohort.user_id for a_ in pos[u]],
                     columns=["user_id", "parent_asin", "is_rating5"]
                     ).to_parquet(out / f"positives_{split}.parquet", index=False)
        report["splits"][split] = {
            "cohort_users": len(cohort), "query_time": str(cohort.query_time.iloc[0]),
            "mean_positives": float(cohort.n_pos.mean()),
            "users_without_disliked": float((summary.n_disliked == 0).mean()),
            "users_without_liked": float((summary.n_liked == 0).mean()),
            "profiles_truncated": float(summary.profile_truncated.mean()),
            "users_with_rating5_positive": float(cohort.user_id.map(lambda u: bool(pos5.get(u))).mean()),
        }
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
