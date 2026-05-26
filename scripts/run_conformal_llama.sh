#!/bin/bash
# run_conformal_llama.sh
# Run conformal search (minmax, softmax, margin) for Llama model on all 9 datasets.

# Change directory to the repository root
cd "$(dirname "$0")/.."

GPU_ID=${1:-0}
RESULTS_DIR=${2:-"./Epsilon_results/Llama/Base"}
MODEL_ID=${3:-"./models_local/llama-3.2-1B"}
EPSILON=${4:-0.05}

DATASETS=("ace2004" "aida" "aquaint" "iitb-fix" "kore50" "msnbc" "n3reuters128" "n3rss500" "spotlight")
METHODS=("minmax" "softmax" "margin")

echo "========================================================================"
echo "Running Conformal Hyperparameter Search for Llama-3.2-1B"
echo "GPU ID:      $GPU_ID"
echo "Results Dir: $RESULTS_DIR"
echo "Model Path:  $MODEL_ID"
echo "Epsilon:     $EPSILON"
echo "========================================================================"

mkdir -p "$RESULTS_DIR"

for DS in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting dataset: $DS, method: $METHOD"
        
        # Run sequentially (recommended to prevent GPU out-of-memory or high load):
        python3 run_lbw_hyperparameter_search_conformal.py \
            --gpu_id "$GPU_ID" \
            --model_id "$MODEL_ID" \
            --dataset "$DS" \
            --conformal_method "$METHOD" \
            --conformal_epsilon "$EPSILON" \
            --conformal_coverage entity \
            --conformal_K 40 \
            --results_dir "$RESULTS_DIR" > "conformal_epsilon_${EPSILON}_llama_3.2_1b_conformal_${METHOD}_${DS}.log" 2>&1
            
        # To run in the background (uncomment below and comment out above):
        # setsid python3 run_lbw_hyperparameter_search_conformal.py \
        #     --gpu_id "$GPU_ID" \
        #     --model_id "$MODEL_ID" \
        #     --dataset "$DS" \
        #     --conformal_method "$METHOD" \
        #     --conformal_epsilon "$EPSILON" \
        #     --conformal_coverage entity \
        #     --conformal_K 40 \
        #     --results_dir "$RESULTS_DIR" > "conformal_epsilon_${EPSILON}_llama_3.2_1b_conformal_${METHOD}_${DS}.log" 2>&1 &
    done
done

echo "All Llama conformal jobs finished/launched."
