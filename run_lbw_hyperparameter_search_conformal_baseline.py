#!/usr/bin/env python3
"""
Hyperparameter Search Script for Conformal Baselines (topk, scoret, platt)

For each configuration, runs a single evaluation pass that computes all three
baseline predictors at once.  Results are written to a JSON summary file.
"""

import subprocess
import re
import argparse
from datetime import datetime
from typing import Dict, List
import json
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAYER_COUNTS = list(range(16, 0, -1))
ARCHITECTURES = ["INPLACE", "EXTEND", "EXTRA", "INTER"]

# Shared command (each run will append config-specific flags)
BASE_COMMAND = [
    "python3",
    "generic_training.py",
    "--found_model", "qwen3_decoder",
    "--type_optimization", "all_encoder_layers_qwen3_decoder",
    "--learning_rate", "3e-8",
    "--evaluate",
    "--top_k", "20",
]

DEFAULT_MODEL_ID = "./models_local/qwen3-embedding-0.6B"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _extract_block_metric(output: str, block: str, metric: str) -> float:
    """Extract a single metric (e.g. Recall) from a baseline block."""
    # e.g.  [topk]   MRR: 0.27488, Recall: 0.66667, EmpCov: 0.92593, GoldCov: 0.68519
    pattern = rf"\[{re.escape(block)}\][^\n]*?{re.escape(metric)}:\s+([\d.]+)"
    m = re.search(pattern, output)
    return float(m.group(1)) if m else 0.0


def parse_output(output: str) -> Dict[str, Dict[str, float]]:
    """Extract MRR, Recall, EmpCov, and GoldCov for the three baselines."""
    results = {"topk": {}, "scoret": {}, "platt": {}}
    for baseline in results:
        results[baseline] = {
            "mrr": _extract_block_metric(output, baseline, "MRR"),
            "recall": _extract_block_metric(output, baseline, "Recall"),
            "empcov": _extract_block_metric(output, baseline, "EmpCov"),
            "goldcov": _extract_block_metric(output, baseline, "GoldCov"),
        }
    return results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    gpu_id: str,
    dataset: str,
    architecture: str,
    num_bidir: int,
    num_unsink: int,
    mask_type: str,
    model_id: str = None,
    found_model: str = None,
    type_optimization: str = None,
    lora_adapter_path: str = None,
    lora_epoch: str = "final",
    results_dir: str = None,
) -> Dict:
    """Run a single configuration and return parsed baseline metrics."""
    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    if found_model is None:
        if "llama" in model_id.lower():
            found_model = "llama_decoder"
        else:
            found_model = "qwen3_decoder"

    if type_optimization is None:
        if "llama" in model_id.lower():
            type_optimization = "all_encoder_layers_llama_decoder"
        else:
            type_optimization = "all_encoder_layers_qwen3_decoder"

    # Filter out '--found_model' and '--type_optimization' from BASE_COMMAND if present
    base_cmd_filtered = []
    skip = False
    for item in BASE_COMMAND:
        if skip:
            skip = False
            continue
        if item in ["--found_model", "--type_optimization"]:
            skip = True
            continue
        base_cmd_filtered.append(item)

    cmd = base_cmd_filtered + [
        "--found_model", found_model,
        "--type_optimization", type_optimization,
        "--gpu_id", gpu_id,
        "--dataset", dataset,
        "--model_id", model_id,
        "--lbw_architecture", architecture,
        "--num_bidir_layers", str(num_bidir),
        "--num_unsink_layers", str(num_unsink),
        "--mask_type", mask_type,
    ]

    if lora_adapter_path is not None:
        cmd += ["--lora_adapter_path", lora_adapter_path, "--lora_epoch", lora_epoch]

    if results_dir is not None:
        cmd += ["--results_dir", results_dir]

    config = {
        "architecture": architecture,
        "num_bidir_layers": num_bidir,
        "num_unsink_layers": num_unsink,
        "mask_type": mask_type,
    }

    print(f"\n{'=' * 60}")
    print(f"Running: {config}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        metrics = parse_output(output)

        for b in metrics:
            m = metrics[b]
            print(f"  [{b}]   MRR: {m['mrr']:.5f}, Recall: {m['recall']:.5f}, "
                  f"EmpCov: {m['empcov']:.5f}, GoldCov: {m['goldcov']:.5f}")

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
    model_id: str = None,
    found_model: str = None,
    type_optimization: str = None,
    lora_adapter_path: str = None,
    lora_epoch: str = "final",
    results_dir: str = None,
) -> List[Dict]:
    results = []
    print("\n" + "=" * 80)
    print("BASELINE: BIDIR Configurations")
    print("=" * 80)
    for arch in ARCHITECTURES:
        for num_layers in LAYER_COUNTS:
            results.append(
                run_experiment(
                    gpu_id, dataset, arch, num_layers, 0, "BACK",
                    model_id, found_model, type_optimization,
                    lora_adapter_path, lora_epoch,
                    results_dir=results_dir,
                )
            )
    return results


def run_back_tests(
    dataset: str,
    gpu_id: str,
    model_id: str = None,
    found_model: str = None,
    type_optimization: str = None,
    lora_adapter_path: str = None,
    lora_epoch: str = "final",
    results_dir: str = None,
) -> List[Dict]:
    results = []
    print("\n" + "=" * 80)
    print("BASELINE: BACK Configurations")
    print("=" * 80)
    for arch in ARCHITECTURES:
        for num_layers in LAYER_COUNTS:
            results.append(
                run_experiment(
                    gpu_id, dataset, arch, 0, num_layers, "BACK",
                    model_id, found_model, type_optimization,
                    lora_adapter_path, lora_epoch,
                    results_dir=results_dir,
                )
            )
    return results


def run_mask0_tests(
    dataset: str,
    gpu_id: str,
    model_id: str = None,
    found_model: str = None,
    type_optimization: str = None,
    lora_adapter_path: str = None,
    lora_epoch: str = "final",
    results_dir: str = None,
) -> List[Dict]:
    results = []
    print("\n" + "=" * 80)
    print("BASELINE: BIDIR with MASK0 (INPLACE only)")
    print("=" * 80)
    for num_layers in LAYER_COUNTS:
        results.append(
            run_experiment(
                gpu_id, dataset, "INPLACE", num_layers, 0, "MASK0",
                model_id, found_model, type_optimization,
                lora_adapter_path, lora_epoch,
                results_dir=results_dir,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def _print_top3_header(name: str):
    print("\n" + "-" * 40)
    print(f"TOP 3 BY {name} (higher is better)")
    print("-" * 40)


def _print_entry(idx: int, config: dict, m: dict, show_metric: str):
    c = config
    print(
        f"{idx}. {c['architecture']}, BIDIR={c['num_bidir_layers']}, "
        f"BACK={c['num_unsink_layers']}, mask={c['mask_type']}"
    )
    print(
        f"   MRR: {m['mrr']:.5f}, Recall: {m['recall']:.5f}, "
        f"EmpCov: {m['empcov']:.5f}, GoldCov: {m['goldcov']:.5f}"
    )


def print_summary(all_results: List[Dict], dataset: str, timestamp: str, model_id: str = None):
    successful = [r for r in all_results if r["success"] and r["metrics"]]

    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    if not successful:
        print("No successful experiments!")
        return

    for baseline in ["topk", "scoret", "platt"]:
        # Gather sub-results for this baseline
        sub = []
        for r in successful:
            sub.append({
                "config": r["config"],
                "metrics": r["metrics"][baseline],
            })

        print(f"\n{'=' * 40}")
        print(f"BASELINE: {baseline.upper()}")
        print("=" * 40)

        for metric in ["mrr", "recall", "empcov", "goldcov"]:
            _print_top3_header(metric.upper())
            s = sorted(sub, key=lambda x: x["metrics"][metric], reverse=True)
            for i, r in enumerate(s[:3], 1):
                _print_entry(i, r["config"], r["metrics"], metric)

    # Save JSON summary
    model_short = model_id.split("/")[-1] if model_id else "default"
    output_file = f"conformal_baseline_results_{dataset}_{timestamp}.json"
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
    parser = argparse.ArgumentParser(description="LBW Conformal Baseline Search")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--dataset", type=str, default="ace2004", help="Dataset name")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--found_model", type=str, default=None,
                        help="Model architecture (e.g., qwen3_decoder, llama_decoder)")
    parser.add_argument("--type_optimization", type=str, default=None,
                        help="Type optimization (e.g., all_encoder_layers_qwen3_decoder, all_encoder_layers_llama_decoder)")
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--lora_epoch", type=str, default="final")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "bidir", "back", "mask0"])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Conformal Baseline Search at {timestamp}")
    print(f"GPU: {args.gpu_id}, Dataset: {args.dataset}")
    print(f"Model: {args.model_id}")

    all_results = []

    if args.test in ["all", "bidir"]:
        all_results.extend(run_bidir_tests(
            args.dataset, args.gpu_id, args.model_id,
            args.found_model, args.type_optimization,
            args.lora_adapter_path, args.lora_epoch,
            results_dir=args.results_dir,
        ))
    if args.test in ["all", "back"]:
        all_results.extend(run_back_tests(
            args.dataset, args.gpu_id, args.model_id,
            args.found_model, args.type_optimization,
            args.lora_adapter_path, args.lora_epoch,
            results_dir=args.results_dir,
        ))
    if args.test in ["all", "mask0"]:
        all_results.extend(run_mask0_tests(
            args.dataset, args.gpu_id, args.model_id,
            args.found_model, args.type_optimization,
            args.lora_adapter_path, args.lora_epoch,
            results_dir=args.results_dir,
        ))

    print_summary(all_results, args.dataset, timestamp, args.model_id)

    print("\n" + "=" * 80)
    print("CONFORMAL BASELINE SEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
