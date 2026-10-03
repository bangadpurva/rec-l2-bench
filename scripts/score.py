"""Score one model on one frozen pool and write runs/<run_id>/.

    python scripts/score.py --model l1_order --split test
    python scripts/score.py --model popularity --split test
    python scripts/score.py --model random --split test
    python scripts/score.py --model oracle --split test
    python scripts/score.py --model bge --split test
    python scripts/score.py --model qwen3 --split test
    python scripts/score.py --model clef --split valid --limit-users 100   # smoke test first

Outputs: manifest.json, per_user.parquet (metrics + latency), scores.parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.data.load import load_ratings  # noqa: E402
from recl2bench.l1 import pools as P  # noqa: E402
from recl2bench.manifest import RunManifest  # noqa: E402
from recl2bench.rerankers.base import UserProfile  # noqa: E402
from recl2bench.runner import run  # noqa: E402

MODELS = ["l1_order", "random", "popularity", "oracle", "bge", "qwen3", "clef"]


def load_template(path: Path):
    from recl2bench.rerankers.decision import YesNoQuestion
    t = yaml.safe_load(path.read_text())
    q = YesNoQuestion(t["id"], t["question"], t["true_means"], t["false_means"])
    return q, hashlib.sha256(path.read_bytes()).hexdigest()


def build(model: str, args, positives, cfg_dir: Path):
    """Return (reranker, model_cfg, template_sha, scoring_mode, concurrency)."""
    tpath = cfg_dir / "templates" / "yes_no_v1.yaml"
    if model == "l1_order":
        from recl2bench.rerankers.dummy import L1OrderReranker
        return L1OrderReranker(), {}, None, None, None
    if model == "random":
        from recl2bench.rerankers.dummy import RandomReranker
        return RandomReranker(args.seed), {}, None, None, None
    if model == "oracle":
        from recl2bench.baselines.simple import OracleReranker
        return OracleReranker(positives), {}, None, None, None
    if model == "popularity":
        from recl2bench.baselines.simple import PopularityReranker
        return PopularityReranker(load_ratings(args.raw, args.category), 90), {}, None, None, None
    if model == "bge":
        from recl2bench.rerankers.cross_encoder import BGEReranker
        c = yaml.safe_load((cfg_dir / "models/bge_reranker.yaml").read_text())
        return (BGEReranker(c["hf_model"], c["revision"], max_length=c["max_length"],
                            batch_size=c["batch_size"]), c, None, None, None)
    if model == "qwen3":
        from recl2bench.rerankers.cross_encoder import Qwen3Reranker
        c = yaml.safe_load((cfg_dir / "models/qwen3_reranker.yaml").read_text())
        q, sha = load_template(tpath)
        return (Qwen3Reranker(q.question, c["hf_model"], c["revision"], max_length=c["max_length"],
                              batch_size=c["batch_size"]), c, sha, None, None)
    if model == "clef":
        from recl2bench.rerankers.decision import DecisionReranker, make_backend
        c = yaml.safe_load((cfg_dir / "models/clef.yaml").read_text())
        q, sha = load_template(tpath)
        mode = args.scoring_mode or c["scoring_mode"]
        be = make_backend("clef", **c)
        return DecisionReranker(be, q, mode, concurrency=c["concurrency"]), c, sha, mode, c["concurrency"]
    raise ValueError(model)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--top-m", type=int, default=100, help="prefix of the frozen 200")
    ap.add_argument("--limit-users", type=int, help="smoke test on the first N users")
    ap.add_argument("--scoring-mode", choices=["per_pair", "fan_out"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default="A")
    ap.add_argument("--hardware", default=platform.platform())
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--pools-dir", default="data/pools")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--budget-tokens", type=int, default=512)
    ap.add_argument("--category", help="override dataset.yaml category (popularity baseline reads ratings)")
    ap.add_argument("--cohort", default="conditional", choices=["conditional", "all"],
                    help="conditional (pre-registered primary): users with >=1 positive in frozen top-100")
    a = ap.parse_args(argv)

    if not a.category:
        a.category = yaml.safe_load((Path(a.configs) / "dataset.yaml").read_text())["category"]
    proc = Path(a.proc)
    pools, pool_sha = P.load(Path(a.pools_dir) / f"pools_{a.split}.parquet", top_m=a.top_m)
    prof = pd.read_parquet(proc / f"profiles_{a.split}.parquet")
    cohort = pd.read_parquet(proc / f"cohort_{a.split}.parquet")
    qt = dict(zip(cohort.user_id, cohort.query_time.astype(str)))
    profiles = {r.user_id: UserProfile(r.user_id, qt[r.user_id], (), (), r.text) for r in prof.itertuples()}
    pos = pd.read_parquet(proc / f"positives_{a.split}.parquet").groupby("user_id")["parent_asin"].apply(set).to_dict()
    items = pd.read_parquet(proc / "items.parquet")
    text = dict(zip(items.parent_asin, items.text))

    n_full = pools.user_id.nunique()
    cohort_sha = None
    if a.cohort == "conditional":
        keep, cohort_sha = P.load_eval_cohort(Path(a.pools_dir) / f"eval_users_{a.split}.parquet")
        pools = pools[pools.user_id.isin(set(keep))]
    users = sorted(pools.user_id.unique())
    if a.limit_users:
        users = users[:a.limit_users]
        pools = pools[pools.user_id.isin(users)]

    reranker, cfg, tsha, mode, conc = build(a.model, a, pos, Path(a.configs))
    if hasattr(reranker, "device"):
        print(f"{a.model}: device={reranker.device} dtype={reranker.dtype}", flush=True)
        a.hardware = f"{a.hardware} [{reranker.device}]"
    started = datetime.now(timezone.utc)
    per_user, scores, totals = run(reranker, pools, profiles, text, pos)

    run_id = f"{started:%Y%m%dT%H%M%S}_{a.track}_{a.model}_{a.split}_{a.cohort}_m{a.top_m}" + \
             (f"_n{a.limit_users}" if a.limit_users else "")
    out = Path(a.runs_dir) / run_id
    m = RunManifest(run_id=run_id, track=a.track, model=a.model,
                    model_version=",".join(sorted(totals["versions"])) or cfg.get("model_version"),
                    scoring_mode=mode, pool_sha256=pool_sha, template_sha256=tsha,
                    budget_tokens=a.budget_tokens, hardware_or_region=a.hardware, concurrency=conc,
                    truncated_pairs=int(getattr(reranker, "truncated_pairs", 0)),
                    retries=totals["retries"], failures=totals["failures"],
                    started_at=started.isoformat(),
                    model_versions_seen=sorted(totals["versions"]),
                    eval_cohort=a.cohort, eval_cohort_sha256=cohort_sha,
                    n_users_scored=len(per_user), n_users_full_cohort=n_full)
    m.write(a.runs_dir)
    per_user.assign(split=a.split, top_m=a.top_m, limit_users=a.limit_users or 0, cohort=a.cohort
                    ).to_parquet(out / "per_user.parquet", index=False)
    scores.to_parquet(out / "scores.parquet", index=False)
    errs = getattr(reranker, "error_samples", {})
    if errs:
        print("backend errors (count: message):")
        for k, v in sorted(errs.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {v}: {k}")
        (out / "errors.json").write_text(__import__("json").dumps(errs, indent=2))
    if hasattr(getattr(reranker, "backend", None), "usage_input_tokens"):
        print(f"input tokens billed: {reranker.backend.usage_input_tokens:,}")
    print(f"{run_id}: users={len(per_user)} ndcg@10={per_user['ndcg@10'].mean():.4f} "
          f"failures={totals['failures']} retries={totals['retries']} "
          f"p50/p95 latency={per_user.latency_s.quantile(.5):.3f}/{per_user.latency_s.quantile(.95):.3f}s")
    return out


if __name__ == "__main__":
    main()
