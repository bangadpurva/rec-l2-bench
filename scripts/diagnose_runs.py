"""Where do rerankers move items? Breaks rankings down by L1 source channel.

    python scripts/diagnose_runs.py --category Musical_Instruments

For the latest conditional test run of each model: share of the top-10 drawn from each
channel, and the mean rank of positives by the channel that retrieved them. Reads only
frozen pools, positives and each run's scores.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.l1 import pools as P  # noqa: E402


def rank_runs(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank 0 = best. Descending score, ties broken by L1 rank (as in order_by_scores)."""
    s = scores.copy()
    s["score"] = s["score"].fillna(float("-inf"))
    s = s.sort_values(["user_id", "score", "l1_rank"], ascending=[True, False, True], kind="stable")
    s["new_rank"] = s.groupby("user_id").cumcount()
    return s


def diagnose(pools: pd.DataFrame, positives: pd.DataFrame, runs: dict[str, pd.DataFrame],
             k: int = 10) -> pd.DataFrame:
    src = pools[["user_id", "parent_asin", "source"]]
    pos = positives[["user_id", "parent_asin"]].assign(is_pos=1)
    channels = sorted(src.source.unique())
    rows = []
    for model, scores in runs.items():
        r = rank_runs(scores).merge(src, on=["user_id", "parent_asin"], how="left")
        r = r.merge(pos, on=["user_id", "parent_asin"], how="left").fillna({"is_pos": 0})
        top = r[r.new_rank < k]
        row = {"model": model}
        for c in channels:
            row[f"top{k}_share_{c}"] = float((top.source == c).mean())
            p = r[(r.is_pos == 1) & (r.source == c)]
            row[f"pos_mean_rank_{c}"] = float(p.new_rank.mean()) if len(p) else float("nan")
            row[f"pos_in_top{k}_{c}"] = float((p.new_rank < k).mean()) if len(p) else float("nan")
        p = r[r.is_pos == 1]
        row["pos_mean_rank_all"] = float(p.new_rank.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def latest_runs(runs_dir: Path, split: str = "test", cohort: str = "conditional") -> dict[str, Path]:
    out: dict[str, tuple[str, Path]] = {}
    for d in sorted(runs_dir.glob("*/")):
        mf = d / "manifest.json"
        if not (mf.exists() and (d / "scores.parquet").exists()):
            continue
        m = json.loads(mf.read_text())
        pu = pd.read_parquet(d / "per_user.parquet", columns=["split", "limit_users"])
        if pu.split.iloc[0] != split or m.get("eval_cohort") != cohort or pu.limit_users.iloc[0]:
            continue
        if m["model"] not in out or m["started_at"] > out[m["model"]][0]:
            out[m["model"]] = (m["started_at"], d)
    return {k: v[1] for k, v in out.items() if k != "oracle"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Musical_Instruments")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--top-m", type=int, default=100)
    a = ap.parse_args(argv)
    base = Path("data") / a.category
    pools, _ = P.load(base / "pools" / f"pools_{a.split}.parquet", top_m=a.top_m)
    keep, _ = P.load_eval_cohort(base / "pools" / f"eval_users_{a.split}.parquet")
    pools = pools[pools.user_id.isin(set(keep))]
    positives = pd.read_parquet(base / "processed" / f"positives_{a.split}.parquet")
    positives = positives[positives.user_id.isin(set(keep))]
    runs = {m: pd.read_parquet(d / "scores.parquet")
            for m, d in latest_runs(Path("runs") / a.category, a.split).items()}
    in_pool = positives.merge(pools[["user_id", "parent_asin", "source"]], on=["user_id", "parent_asin"])
    print(f"{len(keep)} users; positives inside the top-{a.top_m} pool by channel:")
    print(in_pool.source.value_counts().to_string(), "\n")
    t = diagnose(pools, positives, runs, a.k).set_index("model")
    print(t.to_string(float_format=lambda x: f"{x:.3f}"))
    out = Path("results") / a.category / f"diagnose_{a.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out)
    return t


if __name__ == "__main__":
    main()
