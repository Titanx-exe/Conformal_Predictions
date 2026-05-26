#!/bin/bash
# run_ablation_epsilon.sh
# Run epsilon ablation studies (varying epsilon: 0.05, 0.1, 0.15) for Llama or Qwen models.

# Change directory to the repository root
cd "$(dirname "$0")/.."

MODEL_TYPE=${1:-"qwen"} # "qwen" or "llama"
GPU_ID=${2:-0}
RESULTS_DIR=${3:-""}

if [ "$MODEL_TYPE" = "llama" ]; then
    MODEL_ID="./models_local/llama-3.2-1B"
    DEFAULT_RESULTS_DIR="./Epsilon_results/Llama/Base"
elif [ "$MODEL_TYPE" = "qwen" ]; then
    MODEL_ID="./models_local/qwen3-embedding-0.6B"
    DEFAULT_RESULTS_DIR="./conformal_results/qwen3-embedding-0.6b"
else
    echo "ERROR: Invalid MODEL_TYPE '$MODEL_TYPE'. Use 'qwen' or 'llama'."
    exit 1
fi

RESULTS_DIR=${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}

DATASETS=("ace2004" "aida" "aquaint" "iitb-fix" "kore50" "msnbc" "n3reuters128" "n3rss500" "spotlight")
EPSILONS=(0.05 0.1 0.15)
METHODS=("minmax" "softmax" "margin")

echo "========================================================================"
echo "Running Epsilon Ablation Study for $MODEL_TYPE"
echo "GPU ID:       $GPU_ID"
echo "Results Dir:  $RESULTS_DIR"
echo "Model Path:   $MODEL_ID"
echo "Epsilons:     ${EPSILONS[*]}"
echo "========================================================================"

mkdir -p "$RESULTS_DIR"

for EPS in "${EPSILONS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Epsilon: $EPS, Dataset: $DS, Method: $METHOD"
            
            # Run sequentially:
            python3 run_lbw_hyperparameter_search_conformal.py \
                --gpu_id "$GPU_ID" \
                --model_id "$MODEL_ID" \
                --dataset "$DS" \
                --conformal_method "$METHOD" \
                --conformal_epsilon "$EPS" \
                --conformal_coverage entity \
                --conformal_K 40 \
                --results_dir "$RESULTS_DIR" > "conformal_${MODEL_TYPE}_epsilon_${EPS}_${METHOD}_${DS}.log" 2>&1
        done
    done
done

echo "Epsilon ablation runs completed."
