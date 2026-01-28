#!/usr/bin/env bash
set -euo pipefail

# Reproduce the transport-regime table and volume-spectrum figure used in the PRL draft.
#
# This script runs a small batch (pf in {0.12,0.14}, 5 seeds each) and then aggregates results.
#
# NOTE: Runtime depends on your machine. Reduce --steps or --genesis_steps if needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p runs results

COMMON_ARGS="--graph_type layered --n 500 --layers 100   --p_skip2 0.02 --p_skip3 0.005   --energy_mode motif --genesis_steps 1500   --rule threshold --threshold 2 --steps 120   --knot_k 20 --knot_density 0.9"

for pf in 0.12 0.14; do
  for seed in 1 2 3 4 5; do
    python -m scdc_lab.experiments.unified_consistency_universe       $COMMON_ARGS --p_forward "$pf" --seed "$seed" --out_prefix "runs/pf${pf}_seed${seed}"
  done
done

python analysis/aggregate_unified_runs.py --runs_dir runs --out_dir results

echo "Done. See results/ for CSVs and figures."
