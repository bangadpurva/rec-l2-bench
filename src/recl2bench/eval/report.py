"""Results table across runs: means, and paired bootstrap vs L1 order with Holm.

Only runs on the same split, pool hash and top_m are compared. The oracle is
reported as the ceiling but excluded from the Holm family.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import compare_to_baseline
from .metrics import PRIMARY, SECONDARY

METRIC_COLS = [f"{n}@{k}" for n, k in [PRIMARY, *SECONDARY]]
NO_TEST = {"l1_order", "oracle"}


def load_runs(runs_dir: str | Path) -> list[dict]:
    runs = []
    for d in sorted(Path(runs_dir).glob("*/")):
        mf, pu = d / "manifest.json", d / "per_user.parquet"
        if mf.exists() and pu.exists():
            runs.append({"manifest": json.loads(mf.read_text()), "per_user": pd.read_parquet(pu)})
    return runs


def build_report(runs: list[dict], split: str = "test", top_m: int = 100,
                 n_resamples: int = 10_000, seed: int = 0,
                 cohort: str = "conditional") -> tuple[pd.DataFrame, str]:
    sel = [r for r in runs if (r["per_user"]["split"].iloc[0] == split
                               and int(r["per_user"]["top_m"].iloc[0]) == top_m
                               and r["manifest"].get("eval_cohort", "all") == cohort)]
    if not sel:
        raise ValueError(f"no runs for split={split} top_m={top_m} cohort={cohort}")
    hashes = {r["manifest"]["pool_sha256"] for r in sel}
    if len(hashes) != 1:
        raise ValueError(f"runs use different pools: {hashes}")
    chashes = {r["manifest"].get("eval_cohort_sha256") for r in sel}
    if len(chashes) != 1:
        raise ValueError(f"runs use different eval cohorts: {chashes}")
    # latest run per model wins; partial (limited-user) runs are excluded
    by_model = {}
    for r in sorted(sel, key=lambda r: r["manifest"]["started_at"]):
        if r["per_user"].get("limit_users", pd.Series([0])).fillna(0).iloc[0]:
            continue
        by_model[r["manifest"]["model"]] = r
    if "l1_order" not in by_model:
        raise ValueError("score the l1_order baseline first; every comparison is against it")

    users = sorted(by_model["l1_order"]["per_user"].user_id)
    aligned = {}
    for m, r in by_model.items():
        pu = r["per_user"].set_index("user_id")
        missing = set(users) - set(pu.index)
        if missing:
            raise ValueError(f"{m} is missing {len(missing)} users")
        aligned[m] = pu.loc[users]

    base = aligned["l1_order"]["ndcg@10"].to_numpy()
    tested = {m: a["ndcg@10"].to_numpy() for m, a in aligned.items() if m not in NO_TEST}
    stats = {r.name: r for r in compare_to_baseline(tested, base, n_resamples=n_resamples, seed=seed)} \
        if tested else {}

    rows = []
    for m, a in aligned.items():
        mf = by_model[m]["manifest"]
        n_full = mf.get("n_users_full_cohort") or len(users)
        row = {"model": m, "version": mf.get("model_version"), "mode": mf.get("scoring_mode"),
               **{c: a[c].mean() for c in METRIC_COLS},
               # users outside the conditional cohort score 0 under every model, so this is exact
               "ndcg@10_all_users": a["ndcg@10"].sum() / n_full,
               "latency_p50_s": a.latency_s.quantile(.5), "latency_p95_s": a.latency_s.quantile(.95),
               "failures": mf["failures"], "retries": mf["retries"]}
        s = stats.get(m)
        if s:
            row.update({"d_ndcg@10_vs_l1": s.mean_diff, "ci_low": s.ci_low, "ci_high": s.ci_high,
                        "p_holm": s.p_holm, "beats_l1": s.significant})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False).reset_index(drop=True)

    pool = hashes.pop()
    n_full = by_model["l1_order"]["manifest"].get("n_users_full_cohort") or len(users)
    scope = (f"{len(users)} of {n_full} users (pool top-{top_m} holds >= 1 positive)"
             if cohort == "conditional" else f"all {len(users)} users")
    lines = [f"# Results: {split}, top-{top_m}, {scope}",
             "", f"Pool sha256: `{pool[:16]}…`  ·  primary metric: NDCG@10  ·  "
             f"paired bootstrap ({n_resamples:,} resamples) vs L1 order, Holm-corrected", "",
             "| Model | NDCG@10 | Δ vs L1 [CI] | p (Holm) | Beats L1 | NDCG@10 all users | NDCG@30 | P@10 | R@10 | R@30 | HR@10 | p50 / p95 s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, t in table.iterrows():
        dv = t.get("d_ndcg@10_vs_l1", np.nan)
        if pd.notna(dv):
            dcell = f"{dv:+.4f} [{t.ci_low:+.4f}, {t.ci_high:+.4f}]"
            pcell = f"{t.p_holm:.4f}"
            bcell = "yes" if t.beats_l1 else "no"
        else:
            dcell = pcell = bcell = "—"
        lines.append(f"| {t.model} | {t['ndcg@10']:.4f} | {dcell} | {pcell} | {bcell} | "
                     f"{t['ndcg@10_all_users']:.4f} | "
                     f"{t['ndcg@30']:.4f} | {t['precision@10']:.4f} | {t['recall@10']:.4f} | "
                     f"{t['recall@30']:.4f} | {t['hit_rate@10']:.4f} | "
                     f"{t.latency_p50_s:.3f} / {t.latency_p95_s:.3f} |")
    lines += ["", "Oracle is the within-pool ceiling (uses labels); L1 order is the baseline. "
              "Neither is in the Holm family."]
    return table, "\n".join(lines) + "\n"
