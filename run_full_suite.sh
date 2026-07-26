#!/bin/bash
# Run all 6 experiments for a single dataset, logging each.
# Usage: bash run_full_suite.sh <dataset> [real_data.npz|csv]
#   With a data file, all experiments load it (real-data override); results
#   still land under results/<dataset>/ so nothing else changes.
cd "$(dirname "$0")"
DATASET="$1"
DATA_ARG=""
[ -n "$2" ] && DATA_ARG="--data $2"

# Single BLAS thread per process so 4 datasets can run in parallel on 4 cores.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LOG_DIR="results/$DATASET/logs"
mkdir -p "$LOG_DIR"

echo "[$DATASET] START $(date)"
for n in 1 2 3 4 5 6; do
    case $n in
        1) script=experiment1_baselines.py ;;
        2) script=experiment2_sensitivity.py ;;
        3) script=experiment3_layer_analysis.py ;;
        4) script=experiment4_overhead.py ;;
        5) script=experiment5_adaptive.py ;;
        6) script=experiment6_semantic_divergence.py ;;
    esac
    echo "[$DATASET] [$n/6] $script $(date)"
    python3 "$script" --dataset "$DATASET" $DATA_ARG \
        > "$LOG_DIR/${DATASET}_exp${n}.log" 2>&1 \
        && echo "[$DATASET] [$n/6] OK" \
        || echo "[$DATASET] [$n/6] FAILED rc=$?"
done
echo "[$DATASET] COMPLETE $(date)"
