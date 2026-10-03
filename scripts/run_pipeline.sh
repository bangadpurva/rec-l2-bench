#!/usr/bin/env bash
# Every benchmark stage, one command each. Output is also saved to logs/.
#
#   bash scripts/run_pipeline.sh <stage>
#
# Settings come from environment variables (defaults in brackets):
#   CAT       category, e.g. Musical_Instruments        [category in configs/dataset.yaml]
#   HL        L1 recency half-life in days              [180]
#   CHANNELS  L1 channels, comma list of dense,cooc,pop [dense,pop]
#   INDEX     dense index: exact | hnsw                 [hnsw]
#   CATS      categories for the screen stage           [script default]
#
# Example:  CAT=Musical_Instruments HL=90 bash scripts/run_pipeline.sh l1-compare
#
# Stages, in order (STOP = send the output for review before continuing):
#   check       GPU, packages, tests
#   screen      compare candidate categories on validation (CPU only)      STOP
#   data        download CAT and build cohorts, profiles, item text
#   l1-tune     dense half-life on validation                              STOP
#   l1-compare  recall per L1 channel and merged pool                      STOP
#   l1-dryrun   build pools + report, nothing frozen                       STOP
#   l1-freeze   freeze pools and eval cohort (irreversible, asks first)
#   backup      tar processed data, pools, runs, results into backups/
#   baselines   l1_order, random, popularity, oracle on test
#   bge         bge-reranker-v2-m3 on test (GPU)
#   qwen3       Qwen3-Reranker-0.6B on test (GPU)
#   clef-smoke  Clef Flash on 20 validation users (needs .env keys)        STOP
#   clef        Clef Flash on test
#   report      results table -> results/<CAT>/
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-}"
[ -n "$STAGE" ] || { sed -n '2,30p' "$0"; exit 1; }

if [ -f .env ]; then set -a; . ./.env; set +a; fi
CAT="${CAT:-$(python -c "import yaml; print(yaml.safe_load(open('configs/dataset.yaml'))['category'])")}"
HL="${HL:-180}"
CHANNELS="${CHANNELS:-dense,pop}"
INDEX="${INDEX:-hnsw}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

D="data/$CAT"
RUNS="runs/$CAT"
RES="results/$CAT"
mkdir -p logs
LOG="logs/${STAGE}_${CAT}_$(date +%Y%m%dT%H%M%S).log"
HW="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
HW="${HW:-$(uname -m)} $(hostname)"

run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

L1=(--raw "$D/raw" --proc "$D/processed" --pools-dir "$D/pools" --category "$CAT")

COMMON=(--raw "$D/raw" --proc "$D/processed" --pools-dir "$D/pools"
        --runs-dir "$RUNS" --category "$CAT" --hardware "$HW")
SCORE=(--split test "${COMMON[@]}")
SCORE_VALID=(--split valid "${COMMON[@]}")
# Extra args for tests or overrides, e.g. PREP_ARGS="--n-test 1000"
read -r -a PREP_EXTRA <<< "${PREP_ARGS:-}"
read -r -a L1_EXTRA <<< "${L1_ARGS:-}"

echo "stage=$STAGE CAT=$CAT HL=$HL CHANNELS=$CHANNELS INDEX=$INDEX log=$LOG" | tee -a "$LOG"
case "$STAGE" in
  check)
    run nvidia-smi || true
    run python -c "import torch, faiss; print('cuda', torch.cuda.is_available(), 'faiss', faiss.__version__)"
    run python -m pytest -q ;;
  screen)
    if [ -n "${CATS:-}" ]; then run python scripts/screen_categories.py --raw-root data/screen --categories "$CATS"
    else run python scripts/screen_categories.py --raw-root data/screen; fi ;;
  data)
    run bash scripts/download_data.sh "$CAT" "$D/raw"
    run python scripts/prepare_data.py --raw "$D/raw" --out "$D/processed" --category "$CAT" ${PREP_EXTRA[@]+"${PREP_EXTRA[@]}"} ;;
  l1-tune)
    run python scripts/run_l1.py "${L1[@]}" --tune-half-life 30,90,180,365 ${L1_EXTRA[@]+"${L1_EXTRA[@]}"} ;;
  l1-compare)
    run python scripts/run_l1.py "${L1[@]}" --half-life "$HL" --compare-channels ${L1_EXTRA[@]+"${L1_EXTRA[@]}"} ;;
  l1-dryrun)
    run python scripts/run_l1.py "${L1[@]}" --half-life "$HL" --index "$INDEX" --channels "$CHANNELS" ${L1_EXTRA[@]+"${L1_EXTRA[@]}"} ;;
  l1-freeze)
    echo "Freezing $CAT pools: HL=$HL CHANNELS=$CHANNELS INDEX=$INDEX. This cannot be undone."
    if [ "${FORCE:-}" != "1" ]; then read -r -p "Type FREEZE to continue: " ans; [ "$ans" = "FREEZE" ] || exit 1; fi
    run python scripts/run_l1.py "${L1[@]}" --half-life "$HL" --index "$INDEX" --channels "$CHANNELS" --freeze ${L1_EXTRA[@]+"${L1_EXTRA[@]}"} ;;
  backup)
    run bash scripts/backup.sh "$CAT" ;;
  baselines)
    for m in l1_order random popularity oracle; do run python scripts/score.py "${SCORE[@]}" --model "$m"; done ;;
  bge|qwen3)
    run python scripts/score.py "${SCORE[@]}" --model "$STAGE" ;;
  clef-smoke)
    : "${CLOUDFLARE_API_TOKEN:?set CLOUDFLARE_API_TOKEN in .env}" "${CLOUDFLARE_ACCOUNT_ID:?set CLOUDFLARE_ACCOUNT_ID in .env}"
    run python scripts/score.py "${SCORE_VALID[@]}" --model clef --limit-users 20 ;;
  clef)
    : "${CLOUDFLARE_API_TOKEN:?set CLOUDFLARE_API_TOKEN in .env}" "${CLOUDFLARE_ACCOUNT_ID:?set CLOUDFLARE_ACCOUNT_ID in .env}"
    run python scripts/score.py "${SCORE[@]}" --model clef ;;
  report)
    run python scripts/report.py --split test --runs-dir "$RUNS" --out "$RES" ;;
  *)
    echo "unknown stage: $STAGE"; sed -n '2,30p' "$0"; exit 1 ;;
esac
echo "done: $STAGE (log: $LOG)"
