"""Load the 5-core timestamp-split CSVs and item metadata; join on parent_asin."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from .leakage_audit import FORBIDDEN_ITEM_FIELDS

RATING_COLS = ["user_id", "parent_asin", "rating", "timestamp"]
META_KEEP = ["parent_asin", "title", "features", "description", "store", "categories",
             "price", "main_category", "details"]


def load_ratings(raw_dir: str | Path, category: str = "Video_Games") -> pd.DataFrame:
    """Concatenate train/valid/test into one timeline. `history` is dropped:
    we rebuild history ourselves from timestamps, which is auditable."""
    raw_dir = Path(raw_dir)
    parts = []
    for s in ("train", "valid", "test"):
        df = pd.read_csv(raw_dir / f"{category}.{s}.csv.gz", usecols=RATING_COLS,
                         dtype={"user_id": str, "parent_asin": str})
        df["source_split"] = s
        parts.append(df)
    r = pd.concat(parts, ignore_index=True)
    r["timestamp"] = r["timestamp"].astype("int64")
    dup = r.duplicated(["user_id", "parent_asin", "timestamp"]).sum()
    if dup:
        r = r.drop_duplicates(["user_id", "parent_asin", "timestamp"])
    return r.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)


def load_meta(raw_dir: str | Path, keep_asins: set[str] | None = None,
              category: str = "Video_Games") -> pd.DataFrame:
    """Stream the metadata JSONL, keeping only catalog items and non-leaky fields."""
    path = Path(raw_dir) / f"meta_{category}.jsonl.gz"
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if keep_asins is not None and d.get("parent_asin") not in keep_asins:
                continue
            rows.append({k: d.get(k) for k in META_KEEP})
    meta = pd.DataFrame(rows, columns=META_KEEP).drop_duplicates("parent_asin")
    assert not set(FORBIDDEN_ITEM_FIELDS) & set(meta.columns)
    return meta.reset_index(drop=True)
