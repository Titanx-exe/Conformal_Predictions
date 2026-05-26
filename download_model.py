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


SHORTHANDS = {
    "llama": ("meta-llama/Llama-3.2-1B", "./models_local/llama-3.2-1B"),
    "qwen": ("Qwen/Qwen3-Embedding-0.6B", "./models_local/qwen3-embedding-0.6B"),
    "e5": ("intfloat/e5-base-v2", "./models_local/E5"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model to local folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download using shorthand aliases:
    python3 download_model.py --model_id llama
    python3 download_model.py --model_id qwen
    python3 download_model.py --model_id e5
    
    # Download with custom outputs:
    python3 download_model.py --model_id meta-llama/Llama-3.2-1B --output_dir ./models_local/llama-3.2-1B
        """,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID or shorthand alias (llama, qwen, e5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Local directory to save the model. Defaults to ./models_local/<model_name>",
    )

    args = parser.parse_args()

    model_id = args.model_id
    output_dir = args.output_dir

    # Check shorthand
    shorthand_key = model_id.lower().strip()
    if shorthand_key in SHORTHANDS:
        resolved_id, default_dir = SHORTHANDS[shorthand_key]
        print(f"Resolving shorthand '{model_id}' -> Model ID: '{resolved_id}', Default Output Dir: '{default_dir}'")
        model_id = resolved_id
        if output_dir is None:
            output_dir = default_dir

    if output_dir is None:
        model_name = model_id.split("/")[-1]
        output_dir = os.path.join("./models_local", model_name)

    # Download the model
    download_model(model_id, output_dir)


if __name__ == "__main__":
    main()
