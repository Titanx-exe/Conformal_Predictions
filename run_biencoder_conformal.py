#!/usr/bin/env python3
"""
Conformal Prediction Runner Script for Biencoder & E5 Models

Evaluates the model across multiple datasets using conformal prediction
with ONLY the Minimax method. Automatically names the log folders and files
without Look-Both-Ways (LBW) decoder parameters.
"""

import subprocess
import re
import argparse
from datetime import datetime
from typing import Dict, List
import json
import os
import sys

# Calibration file mapping (testb split per dataset)
CALIBRATION_FILE_MAP = {
    "ace2004": "data/aida/wikidata/ace2004_splits/ACE2004_testb",
    "aida": "data/aida/wikidata/aida_splits/aida_testb",
    "aquaint": "data/aida/wikidata/AQUAINT_splits/AQUAINT_testb",
    "iitb-fix": "data/aida/wikidata/iitb-fix_splits/iitb-fix_testb",
    "kore50": "data/aida/wikidata/KORE50_splits/KORE50_testb",
    "msnbc": "data/aida/wikidata/MSNBC_splits/MSNBC_testb",
    "n3reuters128": "data/aida/wikidata/N3-Reuters-128_splits/N3-Reuters-128_testb",
    "n3rss500": "data/aida/wikidata/N3-RSS-500_splits/N3-RSS-500_testb",
    "spotlight": "data/aida/wikidata/spotlight_splits/spotlight_testb",
}

DEFAULT_BIENCODER_PATH = "./models_local/BLINK_AIDA/pytorch_model.bin"
ABS_BIENCODER_PATH = "/upb/users/h/hpurohit/profiles/unix/cs/RR_Retrieval/Robust_Ranking/RobustRanking/models_local/BLINK_AIDA/pytorch_model.bin"

if not os.path.exists(DEFAULT_BIENCODER_PATH) and os.path.exists(ABS_BIENCODER_PATH):
    DEFAULT_BIENCODER_PATH = ABS_BIENCODER_PATH

def _extract_float(output: str, pattern: str) -> float:
    m = re.search(pattern, output)
    return float(m.group(1)) if m else 0.0


def parse_output(output: str) -> Dict[str, float]:
    """Extract conformal metrics from command output."""
    return {
        "mrr": _extract_float(output, r"MRR:\s+([\d.]+)"),
        "recall": _extract_float(output, r"Recall:\s+([\d.]+)"),
        "empcov": _extract_float(output, r"EmpCov:\s+([\d.]+)"),
        "goldcov": _extract_float(output, r"GoldCov:\s+([\d.]+)"),
        "avg_set_size": _extract_float(output, r"AvgSetSize:\s+([\d.]+)"),
    }


def run_experiment(
    gpu_id: str,
    dataset: str,
    found_model: str,
    path_to_model: str,
    conformal_epsilon: float,
    conformal_coverage: str,
    conformal_K: int,
    conformal_delta: float,
    results_dir: str,
    type_optimization: str,
    top_k: int,
) -> Dict:
    cal_file = CALIBRATION_FILE_MAP.get(dataset)
    if cal_file is None:
        print(f"WARNING: No calibration file for dataset '{dataset}'. Skipping.")
        return {
            "config": {},
            "metrics": None,
            "success": False,
            "error": f"No calibration file for dataset '{dataset}'",
        }

    cmd = [
        "python3",
        "generic_training.py",
        "--dataset", dataset,
        "--found_model", found_model,
        "--model_id", "bert-base-uncased",  # Valid tokenizer ID
        "--type_optimization", type_optimization,
        "--learning_rate", "3e-8",
        "--gpu_id", gpu_id,
        "--evaluate",
        "--top_k", str(top_k),
        "--set_predictor", "conformal",
        "--conformal_method", "minmax",  # Always Minimax
        "--conformal_epsilon", str(conformal_epsilon),
        "--conformal_coverage", conformal_coverage,
        "--conformal_K", str(conformal_K),
        "--conformal_delta", str(conformal_delta),
        "--calibration_file", cal_file,
    ]

    if found_model == "biencoder" and path_to_model:
        cmd += ["--path_to_model", path_to_model]

    if results_dir:
        cmd += ["--results_dir", results_dir]

    config = {
        "dataset": dataset,
        "found_model": found_model,
        "conformal_method": "minmax",
        "conformal_epsilon": conformal_epsilon,
        "conformal_coverage": conformal_coverage,
    }

    print(f"\n{'=' * 60}")
    print(f"Running: {config}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            print(f"  COMMAND FAILED (exit code {result.returncode})!")
            print(output)
            return {"config": config, "metrics": None, "success": False, "error": f"Exit code {result.returncode}"}
            
        metrics = parse_output(output)

        print(f"  [conformal] MRR: {metrics['mrr']:.5f}, Recall: {metrics['recall']:.5f}, "
              f"EmpCov: {metrics['empcov']:.5f}, GoldCov: {metrics['goldcov']:.5f}, "
              f"AvgSetSize: {metrics['avg_set_size']:.5f}")

        return {"config": config, "metrics": metrics, "success": True}
    except subprocess.TimeoutExpired:
        print("  TIMEOUT!")
        return {"config": config, "metrics": None, "success": False, "error": "timeout"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"config": config, "metrics": None, "success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Encoder Conformal Prediction Runner")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--dataset", type=str, default="all", 
                        help="Dataset name ('all' to run the 8 default datasets, or specific dataset)")
    parser.add_argument("--found_model", type=str, default="biencoder", choices=["biencoder", "e5"],
                        help="Model architecture: biencoder or e5")
    parser.add_argument("--path_to_model", type=str, default=DEFAULT_BIENCODER_PATH,
                        help="Path to biencoder model weights")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Root directory for prediction JSON logs. Default: None (outputs to conformal_results/ in cwd)")
    parser.add_argument("--type_optimization", type=str, default="all_encoder_layers_e5",
                        help="Type optimization to use for training/evaluation")
    parser.add_argument("--top_k", type=int, default=20, help="top_k candidate size")
    parser.add_argument("--conformal_epsilon", type=float, default=0.1,
                        help="Desired error rate (epsilon) for conformal filtering")
    parser.add_argument("--conformal_coverage", type=str, default="entity", choices=["entity", "document"],
                        help="Coverage level: entity or document")
    parser.add_argument("--conformal_K", type=int, default=40, help="Top-K candidate pool size")
    parser.add_argument("--conformal_delta", type=float, default=1e-8)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Encoder Conformal Prediction at {timestamp}")
    print(f"GPU: {args.gpu_id}, Model: {args.found_model}")
    print(f"Epsilon: {args.conformal_epsilon}, Coverage: {args.conformal_coverage}")

    if args.dataset == "all":
        # Run only the 8 datasets explicitly requested by the user
        datasets_to_run = ["ace2004", "aquaint", "iitb-fix", "kore50", "msnbc", "n3reuters128", "n3rss500", "spotlight"]
    else:
        if args.dataset not in CALIBRATION_FILE_MAP:
            print(f"ERROR: Dataset '{args.dataset}' is invalid.")
            sys.exit(1)
        datasets_to_run = [args.dataset]

    all_results = []
    for ds in datasets_to_run:
        res = run_experiment(
            gpu_id=args.gpu_id,
            dataset=ds,
            found_model=args.found_model,
            path_to_model=args.path_to_model,
            conformal_epsilon=args.conformal_epsilon,
            conformal_coverage=args.conformal_coverage,
            conformal_K=args.conformal_K,
            conformal_delta=args.conformal_delta,
            results_dir=args.results_dir,
            type_optimization=args.type_optimization,
            top_k=args.top_k,
        )
        all_results.append(res)

    # Save summary log file
    summary_file = f"conformal_results_{args.found_model}_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "model": args.found_model,
                "all_results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary results saved to: {summary_file}")


if __name__ == "__main__":
    main()
