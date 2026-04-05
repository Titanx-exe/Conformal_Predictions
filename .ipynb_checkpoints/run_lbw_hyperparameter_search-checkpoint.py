#!/usr/bin/env python3
"""
Hyperparameter Testing Script for LBW (Look-Both-Ways) Configurations

Tests different combinations of:
1. BIDIR layers (10, 8, 4, 2, 1) with INPLACE, EXTEND, EXTRA, INTER
2. BACK layers (10, 8, 4, 2, 1) with INPLACE, EXTEND, EXTRA, INTER  
3. BIDIR with MASK0 (10, 8, 4, 2, 1) - INPLACE only

Usage:
    python run_lbw_hyperparameter_search.py
    
    # With custom GPU:
    python run_lbw_hyperparameter_search.py --gpu_id 1
    
    # Only run specific tests:
    python run_lbw_hyperparameter_search.py --test bidir  # Only BIDIR tests
    python run_lbw_hyperparameter_search.py --test back   # Only BACK tests
    python run_lbw_hyperparameter_search.py --test mask0  # Only MASK0 tests
"""

import subprocess
import re
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import json
import os

# Configuration
LAYER_COUNTS = list(range(16, 0, -1))
ARCHITECTURES = ["INPLACE", "EXTEND", "EXTRA", "INTER"]
# Base command without model_id - will be added dynamically
BASE_COMMAND = [
    "python3", "generic_training.py",
    "--dataset", "aida",
    "--found_model", "llama_decoder",
    "--type_optimization", "all_encoder_layers_llama_decoder",
    "--learning_rate", "3e-8",
    "--evaluate"
]

# Default model ID
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B"


def parse_output(output: str) -> Dict[str, float]:
    """Extract MRR, ECE, and Recall from command output."""
    results = {"mrr": 0.0, "ece": 0.0, "recall": 0.0}
    
    # Extract MRR
    mrr_match = re.search(r'Mean Reciprocal Rank \(MRR\): ([\d.]+)', output)
    if mrr_match:
        results["mrr"] = float(mrr_match.group(1))
    
    # Extract ECE
    ece_match = re.search(r'Validation ECE: ([\d.]+)', output)
    if ece_match:
        results["ece"] = float(ece_match.group(1))
    
    # Extract Recall (the last floating point number before "Evaluation complete")
    # This is the number like "0.07587792642140469"
    recall_matches = re.findall(r'^(0\.\d+)$', output, re.MULTILINE)
    if recall_matches:
        results["recall"] = float(recall_matches[-1])
    
    return results


def run_experiment(gpu_id: str, architecture: str, num_bidir: int, 
                   num_unsink: int, mask_type: str, model_id: str = None,
                   lora_adapter_path: str = None, lora_epoch: str = "final") -> Dict:
    """Run a single experiment and return results."""
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    cmd = BASE_COMMAND + [
        "--gpu_id", gpu_id,
        "--model_id", model_id,
        "--lbw_architecture", architecture,
        "--num_bidir_layers", str(num_bidir),
        "--num_unsink_layers", str(num_unsink),
        "--mask_type", mask_type
    ]

    if lora_adapter_path is not None:
        cmd += ["--lora_adapter_path", lora_adapter_path, "--lora_epoch", lora_epoch]
    
    config = {
        "architecture": architecture,
        "num_bidir_layers": num_bidir,
        "num_unsink_layers": num_unsink,
        "mask_type": mask_type
    }
    
    print(f"\n{'='*60}")
    print(f"Running: {config}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        metrics = parse_output(output)
        
        print(f"  MRR: {metrics['mrr']:.5f}")
        print(f"  ECE: {metrics['ece']:.5f}")
        print(f"  Recall: {metrics['recall']:.5f}")
        
        return {
            "config": config,
            "metrics": metrics,
            "success": True
        }
    except subprocess.TimeoutExpired:
        print("  TIMEOUT!")
        return {"config": config, "metrics": None, "success": False, "error": "timeout"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"config": config, "metrics": None, "success": False, "error": str(e)}


def run_bidir_tests(gpu_id: str, model_id: str = None, lora_adapter_path: str = None, lora_epoch: str = "final") -> List[Dict]:
    """Test BIDIR configurations across all architectures."""
    results = []
    print("\n" + "="*80)
    print("TESTING: BIDIR (Bidirectional Attention) Configurations")
    print("="*80)
    
    for arch in ARCHITECTURES:
        for num_layers in LAYER_COUNTS:
            result = run_experiment(
                gpu_id=gpu_id,
                architecture=arch,
                num_bidir=num_layers,
                num_unsink=0,
                mask_type="BACK",  # Doesn't matter when num_unsink=0
                model_id=model_id,
                lora_adapter_path=lora_adapter_path,
                lora_epoch=lora_epoch
            )
            result["test_type"] = "BIDIR"
            results.append(result)
    
    return results


def run_back_tests(gpu_id: str, model_id: str = None, lora_adapter_path: str = None, lora_epoch: str = "final") -> List[Dict]:
    """Test BACK configurations across all architectures."""
    results = []
    print("\n" + "="*80)
    print("TESTING: BACK (Backward Attention) Configurations")
    print("="*80)
    
    for arch in ARCHITECTURES:
        for num_layers in LAYER_COUNTS:
            result = run_experiment(
                gpu_id=gpu_id,
                architecture=arch,
                num_bidir=0,
                num_unsink=num_layers,
                mask_type="BACK",
                model_id=model_id
            )
            result["test_type"] = "BACK"
            results.append(result)
    
    return results


def run_mask0_tests(gpu_id: str, model_id: str = None, lora_adapter_path: str = None, lora_epoch: str = "final") -> List[Dict]:
    """Test BIDIR with MASK0 (INPLACE only)."""
    results = []
    print("\n" + "="*80)
    print("TESTING: BIDIR with MASK0 (INPLACE architecture only)")
    print("="*80)
    
    for num_layers in LAYER_COUNTS:
        result = run_experiment(
            gpu_id=gpu_id,
            architecture="INPLACE",
            num_bidir=num_layers,
            num_unsink=0,
            mask_type="MASK0",
            model_id=model_id,
            lora_adapter_path=lora_adapter_path,
            lora_epoch=lora_epoch
        )
        result["test_type"] = "BIDIR_MASK0"
        results.append(result)
    
    return results


def find_best_results(results: List[Dict]) -> Dict:
    """Find best configurations for each metric."""
    successful = [r for r in results if r["success"] and r["metrics"]]
    
    if not successful:
        return {"best_mrr": None, "best_ece": None, "best_recall": None}
    
    # Best MRR (higher is better)
    best_mrr = max(successful, key=lambda x: x["metrics"]["mrr"])
    
    # Best ECE (lower is better - but we want well-calibrated)
    best_ece = min(successful, key=lambda x: x["metrics"]["ece"])
    
    # Best Recall (higher is better)
    best_recall = max(successful, key=lambda x: x["metrics"]["recall"])
    
    return {
        "best_mrr": best_mrr,
        "best_ece": best_ece,
        "best_recall": best_recall
    }


def print_summary(all_results: List[Dict], timestamp: str, model_id: str = None):
    """Print summary of all results and find top 3 overall."""
    successful = [r for r in all_results if r["success"] and r["metrics"]]
    
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    if not successful:
        print("No successful experiments!")
        return
    
    # Sort by each metric
    by_mrr = sorted(successful, key=lambda x: x["metrics"]["mrr"], reverse=True)
    by_recall = sorted(successful, key=lambda x: x["metrics"]["recall"], reverse=True)
    by_ece = sorted(successful, key=lambda x: x["metrics"]["ece"])  # Lower is better
    
    print("\n" + "-"*40)
    print("TOP 3 BY MRR (higher is better):")
    print("-"*40)
    for i, r in enumerate(by_mrr[:3], 1):
        c = r["config"]
        m = r["metrics"]
        print(f"{i}. {c['architecture']}, BIDIR={c['num_bidir_layers']}, BACK={c['num_unsink_layers']}, mask={c['mask_type']}")
        print(f"   MRR: {m['mrr']:.5f}, ECE: {m['ece']:.5f}, Recall: {m['recall']:.5f}")
    
    print("\n" + "-"*40)
    print("TOP 3 BY RECALL (higher is better):")
    print("-"*40)
    for i, r in enumerate(by_recall[:3], 1):
        c = r["config"]
        m = r["metrics"]
        print(f"{i}. {c['architecture']}, BIDIR={c['num_bidir_layers']}, BACK={c['num_unsink_layers']}, mask={c['mask_type']}")
        print(f"   MRR: {m['mrr']:.5f}, ECE: {m['ece']:.5f}, Recall: {m['recall']:.5f}")
    
    print("\n" + "-"*40)
    print("TOP 3 BY ECE (lower is better - better calibration):")
    print("-"*40)
    for i, r in enumerate(by_ece[:3], 1):
        c = r["config"]
        m = r["metrics"]
        print(f"{i}. {c['architecture']}, BIDIR={c['num_bidir_layers']}, BACK={c['num_unsink_layers']}, mask={c['mask_type']}")
        print(f"   MRR: {m['mrr']:.5f}, ECE: {m['ece']:.5f}, Recall: {m['recall']:.5f}")
    
    # Overall best (combined score)
    print("\n" + "-"*40)
    print("OVERALL TOP 3 (by combined rank across all metrics):")
    print("-"*40)
    
    # Calculate combined rank for each result
    for i, r in enumerate(successful):
        mrr_rank = by_mrr.index(r) + 1
        recall_rank = by_recall.index(r) + 1
        ece_rank = by_ece.index(r) + 1
        r["combined_rank"] = mrr_rank + recall_rank + ece_rank
    
    by_combined = sorted(successful, key=lambda x: x["combined_rank"])
    
    for i, r in enumerate(by_combined[:3], 1):
        c = r["config"]
        m = r["metrics"]
        print(f"{i}. {c['architecture']}, BIDIR={c['num_bidir_layers']}, BACK={c['num_unsink_layers']}, mask={c['mask_type']}")
        print(f"   MRR: {m['mrr']:.5f}, ECE: {m['ece']:.5f}, Recall: {m['recall']:.5f}")
        print(f"   (Combined Rank Score: {r['combined_rank']})")
    
    # Save results to JSON - include model_id in filename
    # Extract short model name (e.g., "Qwen3-Embedding-4B" from "Qwen/Qwen3-Embedding-4B")
    if model_id:
        model_short = model_id.split("/")[-1] if "/" in model_id else model_id
        output_file = f"lbw_hyperparameter_results_{model_short}_{timestamp}.json"
    else:
        output_file = f"lbw_hyperparameter_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "model_id": model_id,
            "all_results": all_results,
            "top3_by_mrr": [{"config": r["config"], "metrics": r["metrics"]} for r in by_mrr[:3]],
            "top3_by_recall": [{"config": r["config"], "metrics": r["metrics"]} for r in by_recall[:3]],
            "top3_by_ece": [{"config": r["config"], "metrics": r["metrics"]} for r in by_ece[:3]],
            "top3_overall": [{"config": r["config"], "metrics": r["metrics"]} for r in by_combined[:3]]
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LBW Hyperparameter Search")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="Model ID (e.g., Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Embedding-4B, Qwen/Qwen3-Embedding-8B)")
    parser.add_argument("--test", type=str, default="all", 
                        choices=["all", "bidir", "back", "mask0"],
                        help="Which tests to run")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                        help="Path to LoRA adapter directory (e.g., Finetuning/finetuned_models/llama-3.2-1b-lora-dhnm_attn). If None, base model is used.")
    parser.add_argument("--lora_epoch", type=str, default="final",
                        help="Which epoch checkpoint to load (e.g., epoch_1, epoch_5, final). Default: final")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting LBW Hyperparameter Search at {timestamp}")
    print(f"GPU: {args.gpu_id}")
    print(f"Model: {args.model_id}")
    if args.lora_adapter_path:
        print(f"LoRA Adapter: {args.lora_adapter_path} (epoch: {args.lora_epoch})")
    else:
        print(f"LoRA Adapter: None (using base model)")
    print(f"Tests: {args.test}")
    
    all_results = []
    
    if args.test in ["all", "bidir"]:
        all_results.extend(run_bidir_tests(args.gpu_id, args.model_id, args.lora_adapter_path, args.lora_epoch))
    
    if args.test in ["all", "back"]:
        all_results.extend(run_back_tests(args.gpu_id, args.model_id, args.lora_adapter_path, args.lora_epoch))
    
    if args.test in ["all", "mask0"]:
        all_results.extend(run_mask0_tests(args.gpu_id, args.model_id, args.lora_adapter_path, args.lora_epoch))
    
    print_summary(all_results, timestamp, args.model_id)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
