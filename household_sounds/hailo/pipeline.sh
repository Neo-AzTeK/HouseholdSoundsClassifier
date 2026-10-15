#!/usr/bin/env bash
# End-to-end pipeline to compile PyTorch checkpoint into Hailo HEF
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

HW_ARCH="${1:-hailo10h}"
echo "=========================================================="
echo " Household Sounds Classifier - Hailo HEF Pipeline"
echo " Target Architecture: ${HW_ARCH}"
echo " Working Directory:   ${REPO_ROOT}"
echo "=========================================================="

echo ""
echo "[Step 1/3] Exporting PyTorch checkpoint to ONNX..."
uv run sound-export

echo ""
echo "[Step 2/3] Generating calibration dataset for PTQ quantization..."
uv run python -m household_sounds.hailo.calibration --num-samples 128

echo ""
echo "[Step 3/3] Compiling ONNX to Hailo HEF (${HW_ARCH})..."
uv run sound-compile --hw-arch "$HW_ARCH"

echo ""
echo "Pipeline execution finished successfully."
