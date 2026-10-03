#!/usr/bin/env bash
# Amazon Reviews'23: 5-core timestamp split (with history) + item metadata for one category.
#   bash scripts/download_data.sh                                # Video_Games -> data/raw
#   bash scripts/download_data.sh Musical_Instruments            # -> data/raw
#   bash scripts/download_data.sh Baby_Products data/screen/Baby_Products ratings   # ratings only
set -euo pipefail
CAT="${1:-Video_Games}"
OUT="${2:-data/raw}"
WHAT="${3:-all}"
mkdir -p "$OUT"
BASE=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023
for s in train valid test; do
  f="$OUT/$CAT.$s.csv.gz"
  [ -s "$f" ] || curl -fL -o "$f" "$BASE/benchmark/5core/timestamp_w_his/$CAT.$s.csv.gz"
done
if [ "$WHAT" = "all" ]; then
  f="$OUT/meta_$CAT.jsonl.gz"
  [ -s "$f" ] || curl -fL -o "$f" "$BASE/raw/meta_categories/meta_$CAT.jsonl.gz"
fi
ls -lh "$OUT"
