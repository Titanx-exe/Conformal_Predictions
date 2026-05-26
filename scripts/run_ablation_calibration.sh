#!/bin/bash
# run_ablation_calibration.sh
# Run calibration split ablation studies (test10, test20) for Llama or Qwen models.

# Change directory to the repository root
cd "$(dirname "$0")/.."

MODEL_TYPE=${1:-"qwen"} # "qwen" or "llama"
GPU_ID=${2:-0}
RESULTS_DIR=${3:-""}

if [ "$MODEL_TYPE" = "llama" ]; then
    MODEL_ID="./models_local/llama-3.2-1B"
    DEFAULT_RESULTS_DIR="./conformal_results/llama/calibration_ablation"
    EPSILON=0.05
elif [ "$MODEL_TYPE" = "qwen" ]; then
    MODEL_ID="./models_local/qwen3-embedding-0.6B"
    DEFAULT_RESULTS_DIR="./conformal_results/qwen3-embedding-0.6b/calibration_ablation"
    EPSILON=0.1
else
    echo "ERROR: Invalid MODEL_TYPE '$MODEL_TYPE'. Use 'qwen' or 'llama'."
    exit 1
fi

RESULTS_DIR=${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}

DATASETS=("ace2004" "aida" "aquaint" "iitb-fix" "kore50" "msnbc" "n3reuters128" "n3rss500" "spotlight")
SPLITS=("test10" "test20")
METHODS=("minmax" "softmax" "margin")

echo "========================================================================"
echo "Running Calibration Set Ablation Study for $MODEL_TYPE"
echo "GPU ID:       $GPU_ID"
echo "Results Dir:  $RESULTS_DIR"
echo "Model Path:   $MODEL_ID"
echo "Epsilon:      $EPSILON"
echo "Splits:       ${SPLITS[*]}"
echo "========================================================================"

mkdir -p "$RESULTS_DIR"

for SPLIT in "${SPLITS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Split: $SPLIT, Dataset: $DS, Method: $METHOD"
            
            # Run sequentially:
            python3 run_lbw_hyperparameter_search_conformal.py \
                --gpu_id "$GPU_ID" \
                --model_id "$MODEL_ID" \
                --dataset "$DS" \
                --conformal_method "$METHOD" \
                --conformal_epsilon "$EPSILON" \
                --conformal_coverage entity \
                --conformal_K 40 \
                --calibration_split "$SPLIT" \
                --results_dir "$RESULTS_DIR" > "conformal_${MODEL_TYPE}_split_${SPLIT}_${METHOD}_${DS}.log" 2>&1
        done
    done
done

echo "Calibration split ablation runs completed."
