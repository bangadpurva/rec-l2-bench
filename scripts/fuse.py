"""Track C fusion: tune the weight on validation, score test once, write a run directory.

    python scripts/fuse.py --category Musical_Instruments
    python scripts/fuse.py --rerankers qwen3 --partners l1_order

For each (reranker, partner) pair: picks w on the validation eval cohort, records the
grid in results/<CAT>/fusion_tuning.csv, then writes runs/<CAT>/<id>_C_fusion_.../ for
test, which report.py picks up like any other model.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_runs import latest_runs  # noqa: E402
from recl2bench import fusion as F  # noqa: E402
from recl2bench.manifest import RunManifest  # noqa: E402


def load_split(runs_dir: Path, base: Path, split: str):
    runs = latest_runs(runs_dir, split)
    pos = pd.read_parquet(base / "processed" / f"positives_{split}.parquet")
    pos = pos.groupby("user_id")["parent_asin"].apply(set).to_dict()
    return runs, pos


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Musical_Instruments")
    ap.add_argument("--rerankers", default="bge,qwen3")
    ap.add_argument("--partners", default="l1_order,popularity")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args(argv)

    base = Path(a.data_dir) / a.category
    runs_dir = Path(a.runs_dir or Path("runs") / a.category)
    valid_runs, valid_pos = load_split(runs_dir, base, "valid")
    test_runs, test_pos = load_split(runs_dir, base, "test")

    tuning, written = [], []
    for rr in a.rerankers.split(","):
        for pt in a.partners.split(","):
            need = [(s, m) for s, runs in (("valid", valid_runs), ("test", test_runs))
                    for m in (rr, pt) if m not in runs]
            if need:
                print(f"skip {rr}+{pt}: missing runs {need}")
                continue
            # 1) weight from validation only
            w, grid = F.tune(pd.read_parquet(valid_runs[rr] / "scores.parquet"),
                             pd.read_parquet(valid_runs[pt] / "scores.parquet"), valid_pos)
            tuning.append(grid.assign(reranker=rr, partner=pt, chosen=grid.w == w))
            print(f"{rr}+{pt}: validation grid\n{grid.to_string(index=False)}\n  -> w = {w}")

            # 2) one test evaluation with that weight
            rr_m = json.loads((test_runs[rr] / "manifest.json").read_text())
            rr_pu = pd.read_parquet(test_runs[rr] / "per_user.parquet")
            fused = F.fuse_scores(pd.read_parquet(test_runs[rr] / "scores.parquet"),
                                  pd.read_parquet(test_runs[pt] / "scores.parquet"), w)
            per_user = F.per_user_metrics(fused, test_pos)
            lat = rr_pu.set_index("user_id")["latency_s"]
            per_user = per_user.assign(latency_s=per_user.user_id.map(lat), failures=0, split="test",
                                       top_m=int(rr_pu.top_m.iloc[0]), limit_users=0,
                                       cohort=rr_pu.cohort.iloc[0])
            started = datetime.now(timezone.utc)
            name = f"fusion_{rr}+{pt}"
            run_id = f"{started:%Y%m%dT%H%M%S}_C_{name}_test_{rr_m['eval_cohort']}_m{int(rr_pu.top_m.iloc[0])}"
            m = RunManifest(run_id=run_id, track="C", model=name,
                            model_version=f"rrf(k={F.RRF_K}, w={w}) of {rr_m['model_version']} and {pt}",
                            scoring_mode=None, pool_sha256=rr_m["pool_sha256"],
                            template_sha256=rr_m.get("template_sha256"), budget_tokens=rr_m["budget_tokens"],
                            hardware_or_region=rr_m["hardware_or_region"], started_at=started.isoformat(),
                            eval_cohort=rr_m["eval_cohort"], eval_cohort_sha256=rr_m["eval_cohort_sha256"],
                            n_users_scored=len(per_user), n_users_full_cohort=rr_m["n_users_full_cohort"])
            m.write(runs_dir)
            per_user.to_parquet(runs_dir / run_id / "per_user.parquet", index=False)
            fused.to_parquet(runs_dir / run_id / "scores.parquet", index=False)
            written.append(run_id)
            print(f"  test NDCG@10 = {per_user['ndcg@10'].mean():.4f}  -> {run_id}\n")

    out = Path(a.results_dir) / a.category
    out.mkdir(parents=True, exist_ok=True)
    if tuning:
        pd.concat(tuning).to_csv(out / "fusion_tuning.csv", index=False)
    return written


if __name__ == "__main__":
    main()
