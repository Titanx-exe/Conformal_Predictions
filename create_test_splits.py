#!/usr/bin/env python3
"""
Create test10 and test20 NIF/Turtle splits from an input NIF dataset.

- test10  = original dataset minus a random 10% of documents (floor)
- test20  = original dataset minus a random 20% of documents (floor)

Each document is identified by the numeric ID in the NIF subject URI
(e.g. <http://.../0#...>). All blocks (context + entity mentions) that
belong to a removed document are excluded from the corresponding split.
"""

import argparse
import math
import random
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create test10 and test20 NIF/Turtle splits."
    )
    parser.add_argument("input_file", help="Path to the input NIF/Turtle file")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--out-dir", default=".",
        help="Directory where output files will be written (default: .)"
    )
    return parser.parse_args()


def extract_doc_id(block: str) -> str:
    """Extract the document ID from the first line of a Turtle block."""
    first_line = block.splitlines()[0].strip()
    # Support arbitrary path structures in the URI (e.g. http://domain/path/0#char=...)
    m = re.match(r"<https?://[^>]+/(\d+)#", first_line)
    if m:
        return m.group(1)
    return None


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}")
        return

    # ------------------------------------------------------------------
    # 1. Read file and separate prefix declarations from body
    # ------------------------------------------------------------------
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()

    prefix_lines = []
    body_lines = []
    seen_prefix = False
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("@prefix"):
            prefix_lines.append(line)
            seen_prefix = True
        elif stripped == "" and not seen_prefix:
            # skip leading blank lines before any prefix
            continue
        else:
            seen_prefix = True
            body_lines.append(line)

    # ------------------------------------------------------------------
    # 2. Split body into blocks separated by blank lines
    # ------------------------------------------------------------------
    blocks = []
    current_block = []
    for line in body_lines:
        if line.strip() == "":
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block))

    # ------------------------------------------------------------------
    # 3. Group blocks by document ID
    # ------------------------------------------------------------------
    doc_to_blocks = {}
    for block in blocks:
        doc_id = extract_doc_id(block)
        if doc_id is None:
            # Fallback: look for nif:referenceContext inside the block
            m = re.search(r"nif:referenceContext\s+<https?://[^>]+/(\d+)#", block)
            if m:
                doc_id = m.group(1)
            else:
                print(f"[WARNING] Could not determine doc_id for block:\n{block[:200]}...")
                continue
        doc_to_blocks.setdefault(doc_id, []).append(block)

    if not doc_to_blocks:
        print("[ERROR] No documents or document IDs could be parsed from the file.")
        return

    doc_ids = sorted(doc_to_blocks.keys(), key=lambda x: int(x))
    total = len(doc_ids)
    remove_10 = math.ceil(total * 0.10)
    remove_20 = math.ceil(total * 0.20)

    print(f"Total documents found : {total}")
    print(f"  Documents to remove for test10: {remove_10}")
    print(f"  Documents to remove for test20: {remove_20}")

    # ------------------------------------------------------------------
    # 4. Randomly select documents to remove
    # ------------------------------------------------------------------
    random.seed(args.seed)
    remove_10_set = set(random.sample(doc_ids, remove_10)) if remove_10 > 0 else set()

    # Use a different seed stream so test20 is independent but reproducible
    random.seed(args.seed + 1)
    remove_20_set = set(random.sample(doc_ids, remove_20)) if remove_20 > 0 else set()

    keep_10 = [d for d in doc_ids if d not in remove_10_set]
    keep_20 = [d for d in doc_ids if d not in remove_20_set]

    # ------------------------------------------------------------------
    # 5. Write output files
    # ------------------------------------------------------------------
    def write_split(name: str, keep_ids, remove_ids):
        kept_blocks = []
        for doc_id in keep_ids:
            kept_blocks.extend(doc_to_blocks[doc_id])

        out_path = out_dir / f"{input_path.stem}_{name}.nif"
        with open(out_path, "w", encoding="utf-8") as f:
            # Prefix declarations
            for pl in prefix_lines:
                f.write(pl + "\n")
            f.write("\n")
            # Blocks separated by a single blank line
            for i, block in enumerate(kept_blocks):
                f.write(block)
                if i < len(kept_blocks) - 1:
                    f.write("\n\n")

        print(
            f"[{name}] kept {len(keep_ids)} docs, removed {len(remove_ids)} -> {out_path}"
        )
        return out_path

    write_split("test10", keep_10, remove_10_set)
    write_split("test20", keep_20, remove_20_set)

    # ------------------------------------------------------------------
    # 6. Write a small log file with removed document IDs
    # ------------------------------------------------------------------
    log_path = out_dir / f"{input_path.stem}_split.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Input file : {input_path}\n")
        f.write(f"Total docs : {total}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write(f"test10 removed ({remove_10})  : {sorted(remove_10_set, key=int)}\n")
        f.write(f"test20 removed ({remove_20})  : {sorted(remove_20_set, key=int)}\n")
    print(f"Log written to: {log_path}")


if __name__ == "__main__":
    main()
