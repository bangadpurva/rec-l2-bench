"""End-to-end on synthetic raw files in the real Amazon'23 format:
download layout -> prepare_data.py -> run_l1.py (exact + HNSW) -> frozen, audited pools."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recl2bench.data.split import TEST_START_MS, VALID_START_MS
from recl2bench.l1 import pools as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_data  # noqa: E402
import run_l1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = 86_400_000


@pytest.fixture
def raw(tmp_path):
    rng = np.random.default_rng(0)
    items = [f"A{i:04d}" for i in range(400)]
    t0 = VALID_START_MS - 900 * DAY
    rows = []
    for u in range(150):
        n_prior = rng.integers(3, 12)
        for t in rng.uniform(t0, VALID_START_MS - DAY, n_prior):
            rows.append((f"U{u}", rng.choice(items[:300]), int(rng.choice([1, 2, 4, 5, 5])), int(t)))
        for lo, hi in ((VALID_START_MS, TEST_START_MS), (TEST_START_MS, TEST_START_MS + 300 * DAY)):
            for t in rng.uniform(lo, hi, rng.integers(1, 4)):
                rows.append((f"U{u}", rng.choice(items), int(rng.choice([3, 4, 5])), int(t)))
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    df = df.drop_duplicates(["user_id", "parent_asin"])
    df["history"] = ""
    d = tmp_path / "raw"
    d.mkdir()
    for name, part in (("train", df[df.timestamp < VALID_START_MS]),
                       ("valid", df[(df.timestamp >= VALID_START_MS) & (df.timestamp < TEST_START_MS)]),
                       ("test", df[df.timestamp >= TEST_START_MS])):
        part.to_csv(d / f"Video_Games.{name}.csv.gz", index=False)
    with gzip.open(d / "meta_Video_Games.jsonl.gz", "wt") as f:
        for i, a in enumerate(items):
            f.write(json.dumps({"parent_asin": a, "title": f"Game {i}", "store": "Nintendo",
                                "categories": ["Video Games", "Nintendo Switch"], "features": ["fun"],
                                "description": ["long " * 300], "price": 19.99,
                                "average_rating": 4.4, "rating_number": 10,
                                "bought_together": None, "images": []}) + "\n")
    return tmp_path


def test_prepare_and_l1(raw):
    proc, pools_dir = raw / "processed", raw / "pools"
    prepare_data.main(["--config", str(ROOT / "configs/dataset.yaml"), "--raw", str(raw / "raw"),
                       "--out", str(proc), "--n-valid", "40", "--n-test", "60",
                       "--tokenizer", "whitespace"])
    items = pd.read_parquet(proc / "items.parquet")
    assert "average_rating" not in items.columns and items.truncated.all()  # long descriptions cut
    assert items.n_tokens.max() <= 160
    rep = json.loads((proc / "prepare_report.json").read_text())
    assert rep["splits"]["test"]["cohort_users"] == 60

    common = ["--dataset-config", str(ROOT / "configs/dataset.yaml"),
              "--l1-config", str(ROOT / "configs/l1.yaml"), "--raw", str(raw / "raw"),
              "--proc", str(proc), "--pools-dir", str(pools_dir), "--encoder", "fake"]
    run_l1.main(common + ["--tune-half-life", "30,180"])
    run_l1.main(common + ["--index", "exact", "--freeze"])

    report = json.loads((pools_dir / "l1_report.json").read_text())
    t = report["splits"]["test"]
    assert 0 <= t["recall@50"] <= t["recall@100"] <= t["recall@200"] <= t["max_reachable_share"] + 1e-9
    pools, h = P.load(pools_dir / "pools_test.parquet", top_m=100)
    assert h == t["pool_sha256"] and pools.groupby("user_id").size().max() <= 100
    with pytest.raises(FileExistsError):      # frozen: a second freeze refuses
        run_l1.main(common + ["--index", "exact", "--freeze"])


def test_hnsw_matches_exact_on_small_catalog(raw):
    pytest.importorskip("faiss")
    proc = raw / "processed"
    prepare_data.main(["--config", str(ROOT / "configs/dataset.yaml"), "--raw", str(raw / "raw"),
                       "--out", str(proc), "--n-valid", "40", "--n-test", "40",
                       "--tokenizer", "whitespace"])
    run_l1.main(["--dataset-config", str(ROOT / "configs/dataset.yaml"),
                 "--l1-config", str(ROOT / "configs/l1.yaml"), "--raw", str(raw / "raw"),
                 "--proc", str(proc), "--encoder", "fake", "--index", "hnsw"])
    rep = json.loads((proc / "l1_report.json").read_text())
    assert rep["splits"]["valid"]["gate_passed"]
    assert rep["splits"]["valid"]["ann_overlap@100"] > 0.95


def test_score_baselines_and_report(raw):
    import report
    import score
    proc, pools_dir, runs = raw / "processed", raw / "pools", raw / "runs"
    prepare_data.main(["--config", str(ROOT / "configs/dataset.yaml"), "--raw", str(raw / "raw"),
                       "--out", str(proc), "--n-valid", "40", "--n-test", "80",
                       "--tokenizer", "whitespace"])
    run_l1.main(["--dataset-config", str(ROOT / "configs/dataset.yaml"),
                 "--l1-config", str(ROOT / "configs/l1.yaml"), "--raw", str(raw / "raw"),
                 "--proc", str(proc), "--pools-dir", str(pools_dir), "--encoder", "fake",
                 "--index", "exact", "--freeze"])
    common = ["--split", "test", "--raw", str(raw / "raw"), "--proc", str(proc),
              "--pools-dir", str(pools_dir), "--configs", str(ROOT / "configs"), "--runs-dir", str(runs)]
    for m in ("l1_order", "random", "popularity", "oracle"):
        score.main(common + ["--model", m])
    score.main(common + ["--model", "random", "--limit-users", "5", "--seed", "9"])  # excluded
    stem = report.main(["--split", "test", "--runs-dir", str(runs), "--out", str(raw / "results"),
                        "--resamples", "500"])
    table = pd.read_csv(stem.with_suffix(".csv")).set_index("model")
    assert set(table.index) == {"l1_order", "random", "popularity", "oracle"}
    assert table.loc["oracle", "ndcg@10"] >= table["ndcg@10"].max() - 1e-12
    assert pd.isna(table.loc["l1_order", "p_holm"]) and pd.notna(table.loc["random", "p_holm"])
    md = stem.with_suffix(".md").read_text()
    assert "| oracle |" in md and "Beats L1" in md


def test_compare_and_freeze_merged_channels(raw):
    proc, pools_dir = raw / "processed", raw / "pools"
    prepare_data.main(["--config", str(ROOT / "configs/dataset.yaml"), "--raw", str(raw / "raw"),
                       "--out", str(proc), "--n-valid", "30", "--n-test", "30",
                       "--tokenizer", "whitespace"])
    common = ["--dataset-config", str(ROOT / "configs/dataset.yaml"),
              "--l1-config", str(ROOT / "configs/l1.yaml"), "--raw", str(raw / "raw"),
              "--proc", str(proc), "--pools-dir", str(pools_dir), "--encoder", "fake"]
    run_l1.main(common + ["--compare-channels"])
    t = pd.read_csv(proc / "l1_channels_test.csv").set_index("pool")
    assert {"dense", "cooc", "pop", "dense+cooc+pop"} <= set(t.index)
    run_l1.main(common + ["--index", "exact", "--channels", "dense,cooc,pop", "--freeze"])
    pools, _ = P.load(pools_dir / "pools_test.parquet")
    assert set(pools.source) <= {"dense", "cooc", "pop"} and len(set(pools.source)) > 1
    assert not pools.duplicated(["user_id", "parent_asin"]).any()
    rep = json.loads((pools_dir / "l1_report.json").read_text())["splits"]["test"]
    assert "users_with_pos@100" in rep and "channel_cooc" in rep


def test_conditional_cohort_matches_all_users(raw):
    import report
    import score
    proc, pools_dir, runs = raw / "processed", raw / "pools", raw / "runs"
    prepare_data.main(["--config", str(ROOT / "configs/dataset.yaml"), "--raw", str(raw / "raw"),
                       "--out", str(proc), "--n-valid", "30", "--n-test", "80",
                       "--tokenizer", "whitespace"])
    run_l1.main(["--dataset-config", str(ROOT / "configs/dataset.yaml"),
                 "--l1-config", str(ROOT / "configs/l1.yaml"), "--raw", str(raw / "raw"),
                 "--proc", str(proc), "--pools-dir", str(pools_dir), "--encoder", "fake",
                 "--index", "exact", "--channels", "dense,cooc,pop", "--freeze"])
    rep = json.loads((pools_dir / "l1_report.json").read_text())["splits"]["test"]
    keep, _ = P.load_eval_cohort(pools_dir / "eval_users_test.parquet")
    assert len(keep) == rep["eval_cohort_users"] and 0 < len(keep) < 80

    common = ["--split", "test", "--raw", str(raw / "raw"), "--proc", str(proc),
              "--pools-dir", str(pools_dir), "--configs", str(ROOT / "configs"), "--runs-dir", str(runs)]
    for m in ("l1_order", "popularity"):
        cond = score.main(common + ["--model", m])
        full = score.main(common + ["--model", m, "--cohort", "all"])
        pc, pf = pd.read_parquet(cond / "per_user.parquet"), pd.read_parquet(full / "per_user.parquet")
        assert set(pc.user_id) == set(keep)
        # users outside the cohort score exactly 0
        assert (pf[~pf.user_id.isin(keep)]["ndcg@10"] == 0).all()
        assert pc["ndcg@10"].sum() / len(pf) == pytest.approx(pf["ndcg@10"].mean())
    stem = report.main(["--split", "test", "--runs-dir", str(runs), "--out", str(raw / "results"),
                        "--resamples", "300"])
    t = pd.read_csv(stem.with_suffix(".csv")).set_index("model")
    full_mean = pd.read_parquet(full / "per_user.parquet")["ndcg@10"].mean()
    assert t.loc["popularity", "ndcg@10_all_users"] == pytest.approx(full_mean)
    assert "conditional" in stem.name
