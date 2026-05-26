#!/bin/bash
# run_baseline_llama.sh
# Run the conformal baseline evaluation for the Llama model on all 9 datasets.

# Change directory to the repository root
cd "$(dirname "$0")/.."

GPU_ID=${1:-0}
RESULTS_DIR=${2:-"./baseline_results/LLAMA_3.2_1b/LLLAMA_3.2_1b_TRUEbase"}
MODEL_ID=${3:-"./models_local/llama-3.2-1B"}

DATASETS=("ace2004" "aida" "aquaint" "iitb-fix" "kore50" "msnbc" "n3reuters128" "n3rss500" "spotlight")

echo "========================================================================"
echo "Running Conformal Baseline Hyperparameter Search for Llama-3.2-1B"
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
        --dataset "$DS" > "conformal_baseline_llama_3.2_1b_${DS}.log" 2>&1
        
    # To run all datasets in the background simultaneously (uncomment below and comment out above):
    # setsid python3 run_lbw_hyperparameter_search_conformal_baseline.py \
    #     --gpu_id "$GPU_ID" \
    #     --model_id "$MODEL_ID" \
    #     --results_dir "$RESULTS_DIR" \
    #     --dataset "$DS" > "conformal_baseline_llama_3.2_1b_${DS}.log" 2>&1 &
done

echo "All Llama baseline jobs finished/launched."
