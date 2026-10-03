#!/usr/bin/env bash
# One-time setup on a fresh Vast.ai instance (PyTorch template). Safe to re-run.
#   git clone https://github.com/bangadpurva/rec-l2-bench.git && cd rec-l2-bench
#   bash scripts/vast_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v tmux >/dev/null || ! command -v curl >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq tmux curl git) || sudo apt-get install -y -qq tmux curl git
fi

python -m pip install -q --upgrade pip
python -m pip install -q -e ".[models,api,dev]"

[ -f .env ] || cp .env.example .env
mkdir -p logs backups

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU visible"
python - <<'PY'
import torch, faiss
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("faiss", faiss.__version__)
PY
echo "== disk / RAM =="
df -h . | tail -1
free -g | head -2

python -m pytest -q
echo
echo "Setup done. Next: edit .env if you have API keys, then:  tmux new -s bench"
