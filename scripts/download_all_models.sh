#!/bin/bash
# download_all_models.sh
# Download llama, qwen, and e5 models using shorthands defined in download_model.py.

set -e

# Change directory to the repository root where download_model.py is located
cd "$(dirname "$0")/.."

echo "========================================================================"
echo "Downloading Llama, Qwen, and E5 models to local folder models_local/..."
echo "========================================================================"

echo -e "\n[1/3] Downloading Llama model..."
python3 download_model.py --model_id llama

echo -e "\n[2/3] Downloading Qwen model..."
python3 download_model.py --model_id qwen

echo -e "\n[3/3] Downloading E5 model..."
python3 download_model.py --model_id e5

echo -e "\nAll models downloaded successfully!"
