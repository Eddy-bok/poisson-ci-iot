#!/bin/bash

# ============================================================
# Reproduce all results from:
# "Certified Real-Time Anomaly Detection for IoT Networks
# with Pre-Deployment Verification"
# S. Spektor and E. Ibokete, 2026
# ============================================================

set -e

# Move to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "========================================"
echo "Reproducing TDSC Paper Results"
echo "========================================"
echo ""

# ------------------------------------------------------------
# Primary DIAD notebook
# ------------------------------------------------------------
echo "[1/3] Running CIC IoT-DIAD 2024 primary notebook..."

python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    notebooks/CIC_IoT2024_IDAD_primary.ipynb \
    --output-dir=results \
    --output CIC_IoT2024_IDAD_primary_executed.ipynb

echo "[1/3] Done."
echo ""

# ------------------------------------------------------------
# ROC analysis notebook
# ------------------------------------------------------------
echo "[2/3] Running ROC analysis notebook..."

python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    notebooks/CIC_IoT2024_IDAD_roc_analysis.ipynb \
    --output-dir=results \
    --output CIC_IoT2024_IDAD_roc_analysis_executed.ipynb

echo "[2/3] Done."
echo ""

# ------------------------------------------------------------
# Cross-dataset notebook
# ------------------------------------------------------------
echo "[3/3] Running CICIoT2023 cross-dataset notebook..."

python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    notebooks/CICIoT2023_crossdataset.ipynb \
    --output-dir=results \
    --output CICIoT2023_crossdataset_executed.ipynb

echo "[3/3] Done."
echo ""

echo "========================================"
echo "All executed notebooks saved to results/"
echo "========================================"