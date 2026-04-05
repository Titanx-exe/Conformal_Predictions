"""
Create a fine-tuning dataset (JSONL) from ALL train NIF files.

Format: One-pair-per-line — each line is a JSON object with:
  - "document": the full document text
  - "entity":   the entity description text from entity_descriptions.pkl

Entities that are "notInWiki" or missing from entity_descriptions.pkl are skipped.
"""

# import sys
# import os
# import json
# import pickle

# # ── Project setup ────────────────────────────────────────────────────────────
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from nif import NIFDocument
# from data_processing import Aida_joint_el

# # ── Paths ────────────────────────────────────────────────────────────────────
# BASE_PATH = os.path.join(PROJECT_ROOT, "data", "aida", "wikidata")
# ENTITY_DESC_PKL = os.path.join(PROJECT_ROOT, "data", "entity_descriptions.pkl")
# OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "aida_finetune.jsonl")

# TRAIN_FILES = [
#     # ("ACE2004",        os.path.join(BASE_PATH, "ace2004_splits",        "ACE2004_train")),
#     # ("AQUAINT",        os.path.join(BASE_PATH, "AQUAINT_splits",        "AQUAINT_train")),
#     ("AIDA",           os.path.join(BASE_PATH, "aida_splits",           "aida_complete")),
#     # ("iitb-fix",       os.path.join(BASE_PATH, "iitb-fix_splits",       "iitb-fix_train")),
#     # ("KORE50",         os.path.join(BASE_PATH, "KORE50_splits",         "KORE50_train")),
#     # ("MSNBC",          os.path.join(BASE_PATH, "MSNBC_splits",          "MSNBC_train")),
#     # ("N3-Reuters-128", os.path.join(BASE_PATH, "N3-Reuters-128_splits", "N3-Reuters-128_train")),
#     # ("N3-RSS-500",     os.path.join(BASE_PATH, "N3-RSS-500_splits",     "N3-RSS-500_train")),
#     # ("spotlight",      os.path.join(BASE_PATH, "spotlight_splits",      "spotlight_train")),
# ]

# # ── Load shared resources ────────────────────────────────────────────────────
# dp = Aida_joint_el()
# entity_text_dict = pickle.load(open(ENTITY_DESC_PKL, "rb"))
# print(f"Entities in pkl: {len(entity_text_dict)}\n")

# # ── Process all testb files ──────────────────────────────────────────────────
# all_pairs = []
# total_skipped_not_in_wiki = 0
# total_skipped_missing_desc = 0

# print(f"{'Dataset':<20} {'Docs':<8} {'Pairs':<8} {'notInWiki':<12} {'Missing'}")
# print("-" * 65)

# for name, path in TRAIN_FILES:
#     if not os.path.exists(path):
#         print(f"{name:<20} FILE NOT FOUND: {path}")
#         continue

#     nif_doc = dp.load(path)
#     document_map = dp.groupNifDocumentByRefContext(nif_doc)

#     ds_pairs = 0
#     ds_skipped_niw = 0
#     ds_skipped_missing = 0

#     for doc_key, doc in document_map.items():
#         doc_text = None
#         for cont in doc.nifContent:
#             if cont.is_string is not None:
#                 doc_text = cont.is_string
#                 break
#         if doc_text is None:
#             continue

#         for cont in doc.nifContent:
#             if cont.anchor_of is not None:
#                 uri = cont.taIdentRef
#                 if "notInWiki" in uri:
#                     ds_skipped_niw += 1
#                     continue
#                 if uri not in entity_text_dict:
#                     ds_skipped_missing += 1
#                     continue
#                 all_pairs.append({"document": doc_text, "entity": entity_text_dict[uri]})
#                 ds_pairs += 1

#     print(f"{name:<20} {len(document_map):<8} {ds_pairs:<8} {ds_skipped_niw:<12} {ds_skipped_missing}")
#     total_skipped_not_in_wiki += ds_skipped_niw
#     total_skipped_missing_desc += ds_skipped_missing

# # ── Write JSONL ──────────────────────────────────────────────────────────────
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for pair in all_pairs:
#         f.write(json.dumps(pair, ensure_ascii=False) + "\n")

# # ── Summary ──────────────────────────────────────────────────────────────────
# print("-" * 65)
# print(f"\n{'=' * 60}")
# print(f"Dataset created: {OUTPUT_FILE}")
# print(f"{'=' * 60}")
# print(f"Total positive pairs written:       {len(all_pairs)}")
# print(f"Skipped (notInWiki):                {total_skipped_not_in_wiki}")
# print(f"Skipped (missing from pkl):         {total_skipped_missing_desc}")
# print(f"Total annotations processed:        {len(all_pairs) + total_skipped_not_in_wiki + total_skipped_missing_desc}")


""" For Blink Dataset """

"""
dataset_transform_blink.py

Transforms the raw BLINK .jsonl dataset into the flat format expected by
finetuning_dhnm_blink.py:

    {"document": "<input text with [START_ENT]...[END_ENT] markers>",
     "entity":   "<full entity description from pickle>"}

Pipeline
--------
1. Load the wikipedia_id → wikidata_uri mapping from entity_update_blink.py's output
   (or rebuild it on-the-fly from the pickle keys).
2. Load entity descriptions from entity_descriptions.pkl (keyed by wikidata_uri).
3. Stream the raw BLINK .jsonl, resolve each record's entity, and write the
   transformed output.

Records whose entity cannot be resolved are logged and skipped.
"""

import json
import os
import pickle
import sys
import logging
from tqdm import tqdm

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths (edit these as needed) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Raw BLINK dataset (the one with "input", "output", "meta" fields)
RAW_BLINK_PATH = os.path.join(
    PROJECT_ROOT, "data", "aida", "wikidata", "BLINK", "blink-train-kilt_sampled_100.jsonl"
)

# Entity descriptions pickle: { wikidata_uri: description_string }
ENTITY_DESC_PKL = os.path.join(PROJECT_ROOT, "data", "ent_descriptions_update.pkl")

# Output: the clean training file consumed by finetuning_dhnm_blink.py
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "blink_training_100.jsonl")


def count_lines(path: str) -> int:
    """Count total lines in a file for progress bar accuracy."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_wikipedia_to_wikidata_map(entity_desc: dict) -> dict:
    """
    Build a reverse lookup: wikipedia_title (lowercased) → wikidata_uri.

    The entity_descriptions pickle stores descriptions as strings that
    start with "title: <Title> ..." (or "label: <Label> ...").
    We extract the title/label and map it back so we can join with
    BLINK's `output.provenance.title`.

    Since this is a fuzzy heuristic, we also keep the raw wikidata URIs
    around so the caller can try a direct QID-based lookup if available.
    """
    title_to_uri = {}
    for uri, desc in entity_desc.items():
        # Extract the first line / title from the description string
        # Typical format: "title <EntityName> description ..."
        # or after cleanup:  "title <EntityName>  ..."
        first_line = desc.split("\n")[0].strip()

        # 1. Handle optional prefixes (title, label, etc.)
        content = first_line
        for prefix in ("title:", "title ", "label:", "label "):
            if content.lower().startswith(prefix):
                content = content[len(prefix):].strip()
                break
        
        # 2. Extract title before description markers
        # We look for the FIRST occurrence of any marker, even if fused (e.g. "Swanikerdesc:")
        title_val = content
        earliest_idx = len(content)
        for marker in ("desc:", "description:", " desc ", " description ", " alt:", " comment:"):
            idx = content.lower().find(marker)
            if 0 <= idx < earliest_idx:
                # If idx == 0, it means it was just prefix then marker (unlikely but possible)
                earliest_idx = idx
        
        title_val = content[:earliest_idx].strip()

        if title_val:
            title_lower = title_val.lower()
            # Register full title
            if title_lower not in title_to_uri:
                title_to_uri[title_lower] = uri
            
            # Register parts if colon-separated (aliases)
            if ":" in title_lower:
                for part in title_lower.split(":"):
                    p = part.strip()
                    if p and p not in title_to_uri:
                        title_to_uri[p] = uri
            
            # 3. Register normalized title (remove brackets like "Apple (company)")
            import re
            norm_title = re.sub(r"\s*\(.*?\)$", "", title_lower).strip()
            if norm_title and norm_title != title_lower:
                if norm_title not in title_to_uri:
                    title_to_uri[norm_title] = uri

    return title_to_uri


def build_wpid_to_wikidata_map(entity_desc: dict) -> dict:
    """
    If the wikidata URIs follow the pattern http://www.wikidata.org/entity/Q...,
    we can't directly reverse-map to Wikipedia page IDs without the Wikipedia API.

    Instead we rely on title matching (see build_wikipedia_to_wikidata_map).
    This function is a placeholder in case a cached wpid→wd mapping exists.
    """
    # Check if a cached mapping file exists from entity_update_blink.py
    cache_path = os.path.join(PROJECT_ROOT, "data", "wp_to_wd_mapping.json")
    if os.path.exists(cache_path):
        logger.info(f"Found cached Wikipedia→Wikidata mapping: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_entity(record: dict, entity_desc: dict,
                   title_to_uri: dict, wpid_to_wd: dict) -> str | None:
    """
    Try to resolve a BLINK record to a full entity description.

    Resolution order:
      1. Wikipedia page ID → Wikidata URI (via cached mapping) → description
      2. Wikipedia title → Wikidata URI (via title matching)   → description
      3. Answer name  → Wikidata URI (via title matching)      → description
    """
    outputs = record.get("output", [])
    if not outputs:
        return None

    answer = outputs[0].get("answer", "")
    provenance = outputs[0].get("provenance", [])

    for prov in provenance:
        wp_id = str(prov.get("wikipedia_id", ""))
        wp_title = prov.get("title", "")

        # Strategy 1: Direct Wikipedia ID → Wikidata URI lookup
        if wp_id and wp_id in wpid_to_wd:
            wd_uri = wpid_to_wd[wp_id]
            if wd_uri in entity_desc:
                return entity_desc[wd_uri]

        # Strategy 2: Match by Wikipedia title
        if wp_title:
            title_lower = wp_title.lower()
            wd_uri = title_to_uri.get(title_lower)
            
            # Fallback: try removing brackets like " (company)"
            if not wd_uri:
                import re
                norm_title = re.sub(r"\s*\(.*?\)$", "", title_lower).strip()
                if norm_title != title_lower:
                    wd_uri = title_to_uri.get(norm_title)
            
            if wd_uri and wd_uri in entity_desc:
                return entity_desc[wd_uri]

    # Strategy 3: Fall back to answer name
    if answer:
        answer_lower = answer.lower()
        wd_uri = title_to_uri.get(answer_lower)
        
        # Fallback for answer too
        if not wd_uri:
            import re
            norm_answer = re.sub(r"\s*\(.*?\)$", "", answer_lower).strip()
            if norm_answer != answer_lower:
                wd_uri = title_to_uri.get(norm_answer)

        if wd_uri and wd_uri in entity_desc:
            return entity_desc[wd_uri]

    return None


def main():
    logger.info("=" * 60)
    logger.info("  BLINK Dataset Transformation")
    logger.info("  Raw → {document, entity} for finetuning")
    logger.info("=" * 60)

    # ── 1. Load entity descriptions ──
    logger.info(f"Loading entity descriptions from: {ENTITY_DESC_PKL}")
    if not os.path.exists(ENTITY_DESC_PKL):
        logger.error(f"Entity descriptions pickle not found: {ENTITY_DESC_PKL}")
        sys.exit(1)

    with open(ENTITY_DESC_PKL, "rb") as f:
        entity_desc = pickle.load(f)
    logger.info(f"Loaded {len(entity_desc):,} entity descriptions.")

    # ── 2. Build lookup maps ──
    logger.info("Building title → Wikidata URI lookup map...")
    title_to_uri = build_wikipedia_to_wikidata_map(entity_desc)
    logger.info(f"Title lookup map has {len(title_to_uri):,} entries.")

    wpid_to_wd = build_wpid_to_wikidata_map(entity_desc)
    if wpid_to_wd:
        logger.info(f"Wikipedia ID lookup map has {len(wpid_to_wd):,} entries.")
    else:
        logger.info("No cached Wikipedia ID → Wikidata mapping found. Using title matching only.")

    # ── 3. Transform dataset ──
    logger.info(f"Reading raw BLINK dataset: {RAW_BLINK_PATH}")
    total_lines = count_lines(RAW_BLINK_PATH)
    logger.info(f"Total records: {total_lines:,}")

    written = 0
    skipped = 0
    skipped_reasons = {"no_output": 0, "no_description": 0, "parse_error": 0}

    with open(RAW_BLINK_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total_lines, desc="Transforming", unit="record"):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                skipped_reasons["parse_error"] += 1
                continue

            # Document = the 'input' field (contains [START_ENT]...[END_ENT] markers)
            document = record.get("input", "").strip()
            if not document:
                skipped += 1
                skipped_reasons["no_output"] += 1
                continue

            # Entity = full description from pickle
            entity = resolve_entity(record, entity_desc, title_to_uri, wpid_to_wd)
            if not entity:
                skipped += 1
                skipped_reasons["no_description"] += 1
                continue

            # Write the clean training record
            out_record = {"document": document, "entity": entity}
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written += 1

    # ── 4. Summary ──
    logger.info("=" * 60)
    logger.info(f"  Transformation complete!")
    logger.info(f"  Written:  {written:,} records")
    logger.info(f"  Skipped:  {skipped:,} records")
    logger.info(f"    - No output/input:    {skipped_reasons['no_output']:,}")
    logger.info(f"    - No description:     {skipped_reasons['no_description']:,}")
    logger.info(f"    - Parse errors:       {skipped_reasons['parse_error']:,}")
    logger.info(f"  Output:   {OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
