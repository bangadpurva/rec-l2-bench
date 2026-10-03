#!/usr/bin/env bash
# Amazon Reviews'23, Video_Games: 5-core timestamp split (with history) + item metadata.
set -euo pipefail
OUT="${1:-data/raw}"
mkdir -p "$OUT"
BASE=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023
for s in train valid test; do
  curl -fL -o "$OUT/Video_Games.$s.csv.gz" "$BASE/benchmark/5core/timestamp_w_his/Video_Games.$s.csv.gz"
done
curl -fL -o "$OUT/meta_Video_Games.jsonl.gz" "$BASE/raw/meta_categories/meta_Video_Games.jsonl.gz"
ls -lh "$OUT"
