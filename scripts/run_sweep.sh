#!/usr/bin/env bash
# Grade completa numa GPU.  Uso: bash scripts/run_sweep.sh [GPU] [N] [PROTOCOLOS...]
# Exemplos:
#   bash scripts/run_sweep.sh 0 200            # grade completa
#   bash scripts/run_sweep.sh 0 200 foa        # só o FoA, variando
#   bash scripts/run_sweep.sh 0 200 debate moa # debate + MoA
set -euo pipefail
GPU="${1:-0}"; N="${2:-200}"; shift 2 || true
cd "$(dirname "$0")/.."
if [ "$#" -gt 0 ]; then
  python scripts/run_sweep.py --gpu "$GPU" --n "$N" --protocols "$@"
else
  python scripts/run_sweep.py --gpu "$GPU" --n "$N"
fi
