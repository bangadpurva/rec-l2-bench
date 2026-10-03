"""Build results/<split>_m<top_m>.md and .csv from every run in runs/.

    python scripts/report.py --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.eval.report import build_report, load_runs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--top-m", type=int, default=100)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resamples", type=int, default=10_000)
    a = ap.parse_args(argv)
    table, md = build_report(load_runs(a.runs_dir), a.split, a.top_m, a.resamples)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a.split}_m{a.top_m}"
    table.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(md)
    print(md)
    return stem


if __name__ == "__main__":
    main()
