"""Freeze L1 pools: one Parquet per split plus a .sha256, verified on every load."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

POOL_COLUMNS = ["user_id", "query_time", "rank", "parent_asin", "l1_score"]
OPTIONAL_COLUMNS = ["source"]   # L1 channel that contributed the item


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(pools: pd.DataFrame, path: str | Path, overwrite: bool = False) -> str:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} is frozen; refusing to overwrite")
    missing = set(POOL_COLUMNS) - set(pools.columns)
    if missing:
        raise ValueError(f"pool table missing columns {missing}")
    cols = POOL_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in pools.columns]
    df = pools[cols].sort_values(["user_id", "rank"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def load(path: str | Path, top_m: int | None = None) -> tuple[pd.DataFrame, str]:
    """Load and verify. top_m reads a prefix (50/100 of the stored 200)."""
    path = Path(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"pool hash mismatch for {path}: {actual} != {expected}")
    df = pd.read_parquet(path)
    if top_m is not None:
        df = df[df["rank"] < top_m].reset_index(drop=True)
    return df, actual


EVAL_TOP_M = 100   # pre-registered: a user is in the eval cohort if top-100 holds a positive


def freeze_eval_cohort(pools: pd.DataFrame, positives: dict[str, set], path: str | Path,
                       top_m: int = EVAL_TOP_M, overwrite: bool = False) -> tuple[str, int, int]:
    """Users whose frozen top-`top_m` pool contains >= 1 positive. Depends only on
    L1 and labels, so it is identical for every reranker. Returns (sha, n_eval, n_all)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} is frozen; refusing to overwrite")
    top = pools[pools["rank"] < top_m]
    users = sorted(pools.user_id.unique())
    got = top.groupby("user_id")["parent_asin"].apply(set).to_dict()
    keep = [u for u in users if positives.get(u, set()) & got.get(u, set())]
    pd.DataFrame({"user_id": keep}).to_parquet(path, index=False)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest, len(keep), len(users)


def load_eval_cohort(path: str | Path) -> tuple[list[str], str]:
    path = Path(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"eval cohort hash mismatch for {path}")
    return pd.read_parquet(path).user_id.tolist(), actual
