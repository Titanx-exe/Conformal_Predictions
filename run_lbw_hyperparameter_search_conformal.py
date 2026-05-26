#!/usr/bin/env python3
"""
Conformal Hyperparameter Search Script for LBW (Look-Both-Ways)

For each configuration, runs a single evaluation with --set_predictor conformal
and a calibration file.  Extracts per-config conformal metrics (MRR, Recall,
EmpCov, GoldCov, AvgSetSize) and writes aggregated JSON.
"""

import subprocess
import re
import argparse
from datetime import datetime
from typing import Dict, List
import json
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAYER_COUNTS = list(range(16, 0, -1))
ARCHITECTURES = ["INPLACE", "EXTEND", "EXTRA", "INTER"]

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

BASE_COMMAND = [
    "python3",
    "generic_training.py",
    "--found_model", "qwen3_decoder",
    "--type_optimization", "all_encoder_layers_qwen3_decoder",
    "--learning_rate", "3e-8",
    "--evaluate",
    "--top_k", "20",
    "--set_predictor", "conformal",
]

DEFAULT_MODEL_ID = "./models_local/qwen3-embedding-0.6B"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    gpu_id: str,
    dataset: str,
    conformal_method: str,
    conformal_epsilon: float,
    conformal_coverage: str,
    architecture: str,
    num_bidir: int,
    num_unsink: int,
    mask_type: str,
    model_id: str = None,
    lora_adapter_path: str = None,
    lora_epoch: str = "final",
    conformal_K: int = 40,
    conformal_delta: float = 1e-8,
    results_dir: str = None,
    cal_split: str = "testb",
) -> Dict:
    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    cal_file = CALIBRATION_FILE_MAP.get(dataset)
    if cal_file is None:
        print(f"WARNING: No calibration file for dataset '{dataset}'. Skipping.")
        return {
            "config": {},
            "metrics": None,
            "success": False,
            "error": f"No calibration file for dataset '{dataset}'",
        }

    # Append suffix for test10 / test20 splits (e.g., ACE2004_testb_test10.nif)
    if cal_split != "testb":
        cal_file = f"{cal_file}_{cal_split}.nif"

    cmd = BASE_COMMAND + [
        "--gpu_id", gpu_id,
        "--dataset", dataset,
        "--model_id", model_id,
        "--lbw_architecture", architecture,
        "--num_bidir_layers", str(num_bidir),
        "--num_unsink_layers", str(num_unsink),
        "--mask_type", mask_type,
        "--conformal_method", conformal_method,
        "--conformal_epsilon", str(conformal_epsilon),
        "--conformal_coverage", conformal_coverage,
        "--conformal_K", str(conformal_K),
        "--conformal_delta", str(conformal_delta),
        "--calibration_file", cal_file,
    ]

    if lora_adapter_path is not None:
        cmd += ["--lora_adapter_path", lora_adapter_path, "--lora_epoch", lora_epoch]

    if results_dir is not None:
        cmd += ["--results_dir", results_dir]

    config = {
        "dataset": dataset,
        "architecture": architecture,
        "num_bidir_layers": num_bidir,
        "num_unsink_layers": num_unsink,
        "mask_type": mask_type,
        "conformal_method": conformal_method,
        "conformal_epsilon": conformal_epsilon,
        "conformal_coverage": conformal_coverage,
    }

    print(f"\n{'=' * 60}")
    print(f"Running: {config}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
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


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------
def run_bidir_tests(
    dataset: str,
    gpu_id: str,
    conformal_method: str,
    conformal_epsilon: float,
    conformal_coverage: str,
    model_id: str = None,
    lora_adapter_path: str = None,
    lora_epoch: str = "final",
    conformal_K: int = 40,
    conformal_delta: float = 1e-8,
    results_dir: str = None,
    cal_split: str = "testb",
) -> List[Dict]:
    results = []
    print("\n" + "=" * 80)
    print("CONFORMAL: BIDIR Configurations")
    print("=" * 80)
    for arch in ARCHITECTURES:
        for num_layers in LAYER_COUNTS:
            results.append(
                run_experiment(
                    gpu_id, dataset, conformal_method, conformal_epsilon,
                    conformal_coverage, arch, num_layers, 0, "BACK",
                    model_id, lora_adapter_path, lora_epoch,
                    conformal_K, conformal_delta,
                    results_dir=results_dir, cal_split=cal_split,
                )
            )
    return results


def run_back_tests(
    dataset: str,
    gpu_id: str,
    conformal_method: str,
    conformal_epsilon: float,
    conformal_coverage: str,
    model_id: str = None,
    lora_adapter_path: str = None,
    lora_epoch: str = "final",
    conformal_K: int = 40,
    conformal_delta: float = 1e-8,
    results_dir: str = None,
    cal_split: str = "testb",
) -> List[Dict]:
    results = []
    print("\n" + "=" * 80)
    print("CONFORMAL: BACK Configurations")
    print("=" * 80)
    for arch in ARCHITECTURES:
        for num_layers in LAYER_COUNTS:
            results.append(
                run_experiment(
                    gpu_id, dataset, conformal_method, conformal_epsilon,
                    conformal_coverage, arch, 0, num_layers, "BACK",
                    model_id, lora_adapter_path, lora_epoch,
                    conformal_K, conformal_delta,
                    results_dir=results_dir, cal_split=cal_split,
                )
            )
    return results


def run_mask0_tests(
    dataset: str,
    gpu_id: str,
    conformal_method: str,
    conformal_epsilon: float,
    conformal_coverage: str,
    model_id: str = None,
    lora_adapter_path: str = None,
    lora_epoch: str = "final",
    conformal_K: int = 40,
    conformal_delta: float = 1e-8,
    results_dir: str = None,
    cal_split: str = "testb",
) -> List[Dict]:
    results = []
    print("\n" + "=" * 80)
    print("CONFORMAL: BIDIR with MASK0 (INPLACE only)")
    print("=" * 80)
    for num_layers in LAYER_COUNTS:
        results.append(
            run_experiment(
                gpu_id, dataset, conformal_method, conformal_epsilon,
                conformal_coverage, "INPLACE", num_layers, 0, "MASK0",
                model_id, lora_adapter_path, lora_epoch,
                conformal_K, conformal_delta,
                results_dir=results_dir, cal_split=cal_split,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def _print_top3(name: str, results: List[Dict], metric: str):
    print("\n" + "-" * 40)
    print(f"TOP 3 BY {name} (higher is better)")
    print("-" * 40)
    sorted_results = sorted(results, key=lambda x: x["metrics"][metric], reverse=True)
    for i, r in enumerate(sorted_results[:3], 1):
        c = r["config"]
        m = r["metrics"]
        print(
            f"{i}. {c['architecture']}, BIDIR={c['num_bidir_layers']}, "
            f"BACK={c['num_unsink_layers']}, mask={c['mask_type']}, "
            f"MRR: {m['mrr']:.5f}, Recall: {m['recall']:.5f}, "
            f"EmpCov: {m['empcov']:.5f}, GoldCov: {m['goldcov']:.5f}, "
            f"AvgSetSize: {m['avg_set_size']:.5f}"
        )


def print_summary(
    all_results: List[Dict],
    dataset: str,
    timestamp: str,
    model_id: str = None,
):
    successful = [r for r in all_results if r["success"] and r["metrics"]]

    print("\n" + "=" * 80)
    print("SUMMARY OF CONFORMAL RESULTS")
    print("=" * 80)

    if not successful:
        print("No successful experiments!")
        return

    for metric in ["mrr", "recall", "empcov", "goldcov"]:
        _print_top3(metric.upper(), successful, metric)

    # Save JSON summary
    output_file = f"conformal_results_{dataset}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "dataset": dataset,
                "model_id": model_id,
                "all_results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LBW Conformal Hyperparameter Search")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--dataset", type=str, default="ace2004", help="Dataset name")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--lora_epoch", type=str, default="final")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Root directory for per-document prediction JSON logs.")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "bidir", "back", "mask0"])
    parser.add_argument("--conformal_method", type=str, default="minmax",
                        choices=["minmax", "softmax", "margin"])
    parser.add_argument("--conformal_epsilon", type=float, default=0.1)
    parser.add_argument("--conformal_coverage", type=str, default="entity",
                        choices=["entity", "document"])
    parser.add_argument("--conformal_K", type=int, default=40)
    parser.add_argument("--conformal_delta", type=float, default=1e-8)
    parser.add_argument("--calibration_split", type=str, default="testb",
                        choices=["testb", "test10", "test20"],
                        help="Calibration split to use (testb=original, test10=10% removed, test20=20% removed)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Conformal Hyperparameter Search at {timestamp}")
    print(f"GPU: {args.gpu_id}, Dataset: {args.dataset}")
    print(f"Model: {args.model_id}")
    print(f"Conformal: method={args.conformal_method}, epsilon={args.conformal_epsilon}, coverage={args.conformal_coverage}")

    all_results = []

    if args.test in ["all", "bidir"]:
        all_results.extend(run_bidir_tests(
            args.dataset, args.gpu_id, args.conformal_method,
            args.conformal_epsilon, args.conformal_coverage,
            args.model_id, args.lora_adapter_path, args.lora_epoch,
            args.conformal_K, args.conformal_delta,
            results_dir=args.results_dir, cal_split=args.calibration_split,
        ))
    if args.test in ["all", "back"]:
        all_results.extend(run_back_tests(
            args.dataset, args.gpu_id, args.conformal_method,
            args.conformal_epsilon, args.conformal_coverage,
            args.model_id, args.lora_adapter_path, args.lora_epoch,
            args.conformal_K, args.conformal_delta,
            results_dir=args.results_dir, cal_split=args.calibration_split,
        ))
    if args.test in ["all", "mask0"]:
        all_results.extend(run_mask0_tests(
            args.dataset, args.gpu_id, args.conformal_method,
            args.conformal_epsilon, args.conformal_coverage,
            args.model_id, args.lora_adapter_path, args.lora_epoch,
            args.conformal_K, args.conformal_delta,
            results_dir=args.results_dir, cal_split=args.calibration_split,
        ))

    print_summary(all_results, args.dataset, timestamp, args.model_id)

    print("\n" + "=" * 80)
    print("CONFORMAL HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
