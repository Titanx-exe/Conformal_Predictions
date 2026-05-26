#!/bin/bash
# run_baseline_qwen.sh
# Run the conformal baseline evaluation for the Qwen model on all 9 datasets.

# Change directory to the repository root
cd "$(dirname "$0")/.."

GPU_ID=${1:-0}
RESULTS_DIR=${2:-"./baseline_results/qwen3-embedding-0.6b"}
MODEL_ID=${3:-"./models_local/qwen3-embedding-0.6B"}

DATASETS=("ace2004" "aida" "aquaint" "iitb-fix" "kore50" "msnbc" "n3reuters128" "n3rss500" "spotlight")

echo "========================================================================"
echo "Running Conformal Baseline Hyperparameter Search for Qwen3-Embedding-0.6B"
echo "GPU ID:      $GPU_ID"
echo "Results Dir: $RESULTS_DIR"
echo "Model Path:  $MODEL_ID"
echo "========================================================================"

mkdir -p "$RESULTS_DIR"

for DS in "${DATASETS[@]}"; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting dataset: $DS"
    
    # Run sequentially (recommended to prevent GPU out-of-memory or high load):
    python3 run_lbw_hyperparameter_search_conformal_baseline.py \
        --gpu_id "$GPU_ID" \
        --model_id "$MODEL_ID" \
        --results_dir "$RESULTS_DIR" \
        --dataset "$DS" > "conformal_baseline_qwen3-embedding-0.6B_${DS}.log" 2>&1
        
    # To run all datasets in the background simultaneously (uncomment below and comment out above):
    # setsid python3 run_lbw_hyperparameter_search_conformal_baseline.py \
    #     --gpu_id "$GPU_ID" \
    #     --model_id "$MODEL_ID" \
    #     --results_dir "$RESULTS_DIR" \
    #     --dataset "$DS" > "conformal_baseline_qwen3-embedding-0.6B_${DS}.log" 2>&1 &
done

echo "All Qwen baseline jobs finished/launched."
