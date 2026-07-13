#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# run_all.sh — Full experiment suite with dataset tagging
#
# Usage:
#   bash run_all.sh                        # defaults to nyc-taxi
#   bash run_all.sh --dataset nyc-taxi
#   bash run_all.sh --dataset foursquare   # (loader coming soon)
#   bash run_all.sh --dataset geolife
#   bash run_all.sh --dataset yelp
#
# All output files will be written to:
#   results/<dataset-name>/<dataset-name>_<filename>.csv
#
# Estimated runtime: 40-70 min on CPU / 10-15 min on GPU
# ══════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

# ── Parse dataset argument ────────────────────────────────────────────
DATASET="nyc-taxi"   # default
for arg in "$@"; do
    case $arg in
        --dataset=*) DATASET="${arg#*=}" ;;
        --dataset)   shift; DATASET="$1" ;;
    esac
done

export DATASET
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Byzantine-Resilient FL — Full Experiment Suite"
echo "  Dataset : $DATASET"
echo "  Start   : $(date)"
echo "══════════════════════════════════════════════════════════════"

# ── Install dependencies ──────────────────────────────────────────────
pip install numpy pandas scikit-learn scipy pyarrow fastparquet \
    --quiet 2>/dev/null || true

# ── Create results directory ──────────────────────────────────────────
mkdir -p "results/$DATASET"

# ── Verify dataset config ─────────────────────────────────────────────
echo ""
echo "Checking dataset configuration..."
python3 -c "
from dataset_config import DATASET_NAME, results_dir, tag
print(f'  Dataset name : {DATASET_NAME}')
print(f'  Results dir  : {results_dir()}')
print(f'  Sample file  : {tag(\"table_baselines.csv\")}')
"

# ── Run experiments ───────────────────────────────────────────────────
LOG_DIR="results/$DATASET/logs"
mkdir -p "$LOG_DIR"

echo ""
echo "[1/5] Baseline comparison (FedAvg, Krum, TrimMean, Bulyan, RFVIR, FLAME, Ours)..."
python3 experiment1_baselines.py --dataset "$DATASET" \
    2>&1 | tee "$LOG_DIR/${DATASET}_exp1.log"

echo ""
echo "[2/5] Sensitivity analysis (tau, alpha, delta)..."
python3 experiment2_sensitivity.py --dataset "$DATASET" \
    2>&1 | tee "$LOG_DIR/${DATASET}_exp2.log"

echo ""
echo "[3/5] Per-layer confusion breakdown..."
python3 experiment3_layer_analysis.py --dataset "$DATASET" \
    2>&1 | tee "$LOG_DIR/${DATASET}_exp3.log"

echo ""
echo "[4/5] Computational overhead..."
python3 experiment4_overhead.py --dataset "$DATASET" \
    2>&1 | tee "$LOG_DIR/${DATASET}_exp4.log"

echo ""
echo "[5/5] Adaptive adversary (sleeper / mimicry / slow-drift)..."
python3 experiment5_adaptive.py --dataset "$DATASET" \
    2>&1 | tee "$LOG_DIR/${DATASET}_exp5.log"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Complete: $(date)"
echo "  Dataset : $DATASET"
echo ""
echo "  Result files (all in results/$DATASET/):"
echo ""
ls -1 "results/$DATASET/"*.csv 2>/dev/null | while read f; do
    echo "    $(basename $f)"
done
echo ""
echo "  Key files for paper:"
echo "    ${DATASET}_table_baselines.csv        → Table VI"
echo "    ${DATASET}_sensitivity_tau.csv        → Table IX (τ sweep)"
echo "    ${DATASET}_sensitivity_alpha.csv      → Table IX (α sweep)"
echo "    ${DATASET}_sensitivity_delta.csv      → Table IX (δ sweep)"
echo "    ${DATASET}_layer_confusion.csv        → Table X"
echo "    ${DATASET}_overhead.csv               → Table XI"
echo "    ${DATASET}_adaptive_detection_latency.csv → Table XII"
echo ""
echo "  Send all files in results/$DATASET/ to update the paper."
echo "══════════════════════════════════════════════════════════════"
