#!/usr/bin/env bash
# Pack everything that is expensive to recreate for one category into backups/.
# Rating CSVs are included (the popularity baseline reads them); item metadata is not. Copy the archive off the instance:
#   scp -P <port> root@<host>:/workspace/rec-l2-bench/backups/<file> .
set -euo pipefail
cd "$(dirname "$0")/.."
CAT="${1:?usage: backup.sh <category>}"
mkdir -p backups
OUT="backups/${CAT}_$(date +%Y%m%dT%H%M%S).tar.gz"
paths=()
for p in "data/$CAT/processed" "data/$CAT/pools" "runs/$CAT" "results/$CAT" logs data/screen/screen_validation.csv \
         data/"$CAT"/raw/"$CAT".{train,valid,test}.csv.gz; do
  [ -e "$p" ] && paths+=("$p")
done
[ ${#paths[@]} -gt 0 ] || { echo "nothing to back up for $CAT"; exit 1; }
tar czf "$OUT" "${paths[@]}"
ls -lh "$OUT"
echo "Copy it off this machine before destroying the instance."
