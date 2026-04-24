#!/usr/bin/env python3
"""
Download HuggingFace models to a dedicated local folder.

Usage:
    python3 download_model.py --model_id meta-llama/Llama-3.2-1B --output_dir ./models_local/llama-3.2-1B
"""

import os
import argparse
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download


def download_model(model_id: str, output_dir: str):
    """
    Download model and tokenizer to a dedicated local folder.

    Args:
        model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-3.2-1B')
        output_dir: Local directory to save the model
    """
    print(f"{'=' * 60}")
    print(f"Downloading model: {model_id}")
    print(f"Target directory: {output_dir}")
    print(f"{'=' * 60}")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download using snapshot_download (more reliable than individual downloads)
    print("\n[1/3] Downloading model files...")
    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,  # Copy files instead of symlinks
        resume_download=True,
        etag_timeout=30,
    )
    print(f"✓ Model files downloaded to: {local_path}")

    # Verify by loading tokenizer
    print("\n[2/3] Verifying tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
        print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return None

    # Verify by loading config
    print("\n[3/3] Verifying config...")
    try:
        config = AutoConfig.from_pretrained(output_dir, local_files_only=True)
        print(f"✓ Config loaded: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        print(f"  Num hidden layers: {config.num_hidden_layers}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return None

    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nModel saved to: {output_dir}")
    print(f"\nTo use this model, run:")
    print(f"  python3 generic_training.py --model_id {output_dir} ...")
    print(f"{'=' * 60}\n")

    return local_path


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model to local folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download Llama-3.2-1B
    python3 download_model.py --model_id meta-llama/Llama-3.2-1B --output_dir ./models_local/llama-3.2-1B
    
    # Download Llama-3.2-3B
    python3 download_model.py --model_id meta-llama/Llama-3.2-3B --output_dir ./models_local/llama-3.2-3B
    
    # Download Qwen3-Embedding-0.6B
    python3 download_model.py --model_id Qwen/Qwen3-Embedding-0.6B --output_dir ./models_local/qwen3-embedding-0.6B
        """,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Local directory to save the model (e.g., ./models_local/llama-3.2-1B)",
    )

    args = parser.parse_args()

    # Download the model
    download_model(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
