"""
training_data.py
──────────────────────────────────────────────────────────────────────────────
Prepares filtered_reports.json for PEFT fine-tuning on Qwen3.5-35B-A3B.

Steps:
  1. Load filtered_reports.json
  2. Build prompt strings using the Qwen3.5 chat template
  3. Compute labelset powerset key per sample
  4. Stratified split (train / val / test) with optional forced ID overrides
  5. Compute class weights from the TRAINING split distribution only
  6. Save train/val/test as HuggingFace Datasets + a weights JSON

Usage:
    python finetuning/training_data.py \
        --input filtered_reports.json \
        --powerset label_powerset.json \
        --out_dir ./dataset \
        --train_ratio 0.70 \
        --val_ratio 0.15 \
        --force_ids_train 101 204 388 \
        --force_ids_val   512 \
        --force_ids_test  99 1024
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("prepare_data")

# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a system specialized in classifying brain MRI reports. "
    "Classify the report into one or more of the following categories: "
    "Neoformazione, Vascolare, Infiammazione o Infezione, Neurodegenerativo, Altro. "
    "Use multiple labels if applicable. "
    "Assign categories only to findings involving the intracranial/central nervous system; "
    "otherwise classify them as Altro. Do not infer unstated pathologies. "
    "Confidence scale: 1 = very uncertain, 2 = uncertain, 3 = moderate, "
    "4 = fairly confident, 5 = very confident. "
    "Output strict JSON only: {'labels': 'category1, category2, ...', 'confidence': <integer 1-5>}"
)

LABEL_ORDER = [
    "Vascolare",
    "Neurodegenerativo",
    "Infiammazione o Infezione",
    "Neoformazione",
    "Altro",
]

def format_labels(labels: list[str]) -> str:
    """
    Returns labels in a canonical, alphabetically-by-LABEL_ORDER sorted string.
    Consistent ordering is important so the model learns a predictable output format.
    """
    ordered = [l for l in LABEL_ORDER if l in labels]
    # append any unexpected labels at the end (safety net)
    ordered += [l for l in sorted(labels) if l not in LABEL_ORDER]
    return ", ".join(ordered)


def build_prompt(record: dict, tokenizer=None) -> str:
    """
    Builds the full training string for one report using the Qwen3.5 chat template.

    With a tokenizer, the Qwen3.5 chat template is applied with thinking disabled,
    which writes an empty <think></think> block before the answer. Without a
    tokenizer, the same string is written by hand.
    """
    user_content = record["data"]["TestoRefertoStringa"].strip()
    num_annotations = len(record["annotations"])
    for dict_info in record["annotations"][num_annotations - 1]["result"]:
        if dict_info["from_name"] == "category_class":
            labels = dict_info["value"]["choices"]
        if dict_info["from_name"] == "confidence_rating":
            confidence = dict_info["value"]["choices"][0]
    assistant_content = json.dumps({
        "labels": format_labels(labels),
        "confidence": confidence,          # kept as the annotation string, e.g. "5"
    }, ensure_ascii=False)

    if tokenizer is not None:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_content}<|im_end|>\n"
    )


# ── Powerset key ──────────────────────────────────────────────────────────────

def labelset_key(labels: list[str]) -> str:
    return " + ".join(sorted(labels))


# ── Stratified split with forced IDs ─────────────────────────────────────────

def stratified_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    forced_train: set,
    forced_val: set,
    forced_test: set,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Splits records into train/val/test with:
      - Forced ID assignment (overrides stratification)
      - Stratification by labelset powerset key for the remaining records
      - Tier 4 labelsets (count <= 4) forced to train to guarantee model sees them

    Returns (train_records, val_records, test_records)
    """
    random.seed(seed)
    test_ratio = 1.0 - train_ratio - val_ratio

    # Separate forced records
    pool = []
    train, val, test = [], [], []

    for rec in records:
        rid = rec["id"]
        if rid in forced_train:
            train.append(rec)
        elif rid in forced_val:
            val.append(rec)
        elif rid in forced_test:
            test.append(rec)
        else:
            pool.append(rec)

    log.info(f"  Forced → train:{len(train)}  val:{len(val)}  test:{len(test)}")
    log.info(f"  Pool for stratified split: {len(pool)} records")

    # Group pool by labelset key
    by_labelset: dict[str, list[dict]] = defaultdict(list)
    for rec in pool:
        num_annotations = len(rec["annotations"])
        for dict_info in rec["annotations"][num_annotations - 1]["result"]:
            if dict_info["from_name"] == "category_class":
                labels = dict_info["value"]["choices"]
        key = labelset_key(labels)
        by_labelset[key].append(rec)

    # Stratified split per labelset group
    for key, group in by_labelset.items():
        random.shuffle(group)
        n = len(group)

        # Force Tier 4 groups (≤ 4 samples) entirely into training
        # — they're too rare to waste on val/test
        if n <= 4:
            log.info(f"  Tier4 → all {n} samples of [{key}] forced to train")
            train.extend(group)
            continue

        n_val  = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test

        # Guard: if group is too small for 3-way split, prioritise train
        if n_train <= 0:
            train.extend(group)
            continue

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    return train, val, test


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_weights_from_split(train_records: list[dict]) -> dict[str, float]:
    """
    Computes balanced class weights using the TRAINING SPLIT ONLY.

    Formula: weight(i) = N_total / (N_classes * N_i)
    Same as sklearn compute_class_weight("balanced").

    Using only the training split is critical — computing weights on the
    full dataset would let test set distribution influence training dynamics.

    Returns: { "Vascolare + Neoformazione": 2.34, ... }
    """
    counts: Counter = Counter()
    for rec in train_records:
        num_annotations = len(rec["annotations"])
        for dict_info in rec["annotations"][num_annotations - 1]["result"]:
            if dict_info["from_name"] == "category_class":
                labels = dict_info["value"]["choices"]
        counts[labelset_key(labels)] += 1

    total   = len(train_records)
    n_cls   = len(counts)
    weights = {}
    for key, count in counts.items():
        # raw balanced weight; qwen_finetune.py applies the smoothing (weight_smoothing)
        weights[key]     = round(total / (n_cls * count), 4)

    return weights


def attach_weights(
    records: list[dict],
    weights: dict[str, float],
    default_weight: float = 1.0,
) -> list[dict]:
    """
    Adds a 'class_weight' field to each record.
    Val/test records get default_weight=1.0 — weights only affect training loss.
    """
    for rec in records:
        num_annotations = len(rec["annotations"])
        for dict_info in rec["annotations"][num_annotations - 1]["result"]:
            if dict_info["from_name"] == "category_class":
                labels = dict_info["value"]["choices"]
        key = labelset_key(labels)
        rec["class_weight"] = weights.get(key, default_weight)
    return records


# ── HuggingFace Dataset builder ───────────────────────────────────────────────

def records_to_hf_dataset(records: list[dict], tokenizer=None, split=None, out_dir: Path | None = None) -> Dataset:
    """
    Converts records to a HuggingFace Dataset with columns:
        id, text (formatted prompt), labels, labelset_key, class_weight
    """
    rows = {
        "id":            [],
        "text":          [],
        "labels":        [],
        "labelset_key":  [],
        "class_weight":  [],
    }
    for rec in records:
        num_annotations = len(rec["annotations"])
        for dict_info in rec["annotations"][num_annotations - 1]["result"]:
            if dict_info["from_name"] == "category_class":
                labels = dict_info["value"]["choices"]
        rows["id"].append(rec["id"])
        rows["text"].append(build_prompt(rec, tokenizer))
        rows["labels"].append(labels)
        rows["labelset_key"].append(labelset_key(labels))
        rows["class_weight"].append(rec.get("class_weight", 1.0))

    if split and out_dir is not None:
        with open(out_dir / f"human_{split}_dataset.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    return Dataset.from_dict(rows)


# ── Split statistics ──────────────────────────────────────────────────────────

def log_split_stats(name: str, records: list[dict], weights: dict):
    labelset_counts = Counter()
    for rec in records:
        num_annotations = len(rec["annotations"])
        for dict_info in rec["annotations"][num_annotations - 1]["result"]:
            if dict_info["from_name"] == "category_class":
                labels = dict_info["value"]["choices"]
        labelset_counts[labelset_key(labels)] += 1
    log.info(f"\n  ── {name} split: {len(records)} records, "
             f"{len(labelset_counts)} unique labelsets ──")
    for key, count in labelset_counts.most_common():
        w = weights.get(key, 1.0)
        log.info(f"    [{count:>4}]  w={w:.3f}  {key}")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",           default="filtered_reports.json")
    p.add_argument("--powerset",        default="label_powerset.json",
                   help="Optional: if provided, logs a comparison with pre-computed weights")
    p.add_argument("--out_dir",         default="./dataset")
    p.add_argument("--train_ratio",     type=float, default=0.80)
    p.add_argument("--val_ratio",       type=float, default=0.10)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--force_ids_train", nargs="*",  type=int, default=[],
                   help="Report IDs to force into training set")
    p.add_argument("--force_ids_val",   nargs="*",  type=int, default=[],
                   help="Report IDs to force into validation set")
    p.add_argument("--force_ids_test",  nargs="*",  type=int, default=[],
                   help="Report IDs to force into test set")
    p.add_argument("--use_tokenizer",   action="store_true",
                   help="Load Qwen3.5 tokenizer to build prompts with apply_chat_template. "
                        "Slower but more accurate. Omit for fast inspection.")
    p.add_argument("--model_id",        default="Qwen/Qwen3.5-35B-A3B")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info(f"Loading {args.input} ...")
    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)
    log.info(f"Total records: {len(records)}")

    # ── Tokenizer (optional) ──────────────────────────────────────────────────
    tokenizer = None
    if args.use_tokenizer:
        log.info(f"Loading tokenizer from {args.model_id} ...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        log.info("Tokenizer loaded")


    # ── Split ─────────────────────────────────────────────────────────────────
    log.info("\nSplitting dataset ...")
    train_recs, val_recs, test_recs = stratified_split(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        forced_train=set(args.force_ids_train),
        forced_val=set(args.force_ids_val),
        forced_test=set(args.force_ids_test),
        seed=args.seed,
    )
    log.info(f"\n  Final split: train={len(train_recs)}  "
             f"val={len(val_recs)}  test={len(test_recs)}  "
             f"total={len(train_recs)+len(val_recs)+len(test_recs)}")

    # ── Weights from training split only ──────────────────────────────────────
    log.info("\nComputing class weights from training split ...")
    weights = compute_weights_from_split(train_recs)

    # Attach weights (train gets real weights, val/test get 1.0)
    train_recs = attach_weights(train_recs, weights, default_weight=1.0)
    val_recs   = attach_weights(val_recs,   {},      default_weight=1.0)
    test_recs  = attach_weights(test_recs,  {},      default_weight=1.0)

    # ── Log split stats ───────────────────────────────────────────────────────
    log_split_stats("TRAIN", train_recs, weights)
    log_split_stats("VAL",   val_recs,   weights)
    log_split_stats("TEST",  test_recs,  weights)

    # ── Compare with pre-computed powerset weights (if provided) ──────────────
    if Path(args.powerset).exists():
        with open(args.powerset, encoding="utf-8") as f:
            powerset_data = json.load(f)
        pre_weights = {
            entry["labelset"]: entry["class_weight"]
            for entry in powerset_data.get("labelsets", [])
            if "labelset" in entry
        }
        log.info("\n  ── Weight comparison: pre-computed vs training-split ──")
        log.info(f"  {'Labelset':<45} {'Pre':>7}  {'Train-split':>11}")
        for key in sorted(weights.keys()):
            pre = pre_weights.get(key, "N/A")
            train_w = weights[key]
            pre_str = f"{pre:.4f}" if isinstance(pre, float) else str(pre)
            log.info(f"  {key:<45} {pre_str:>7}  {train_w:>11.4f}")
    
    # ── Build HF Datasets ─────────────────────────────────────────────────────
    log.info("\nBuilding HuggingFace DatasetDict ...")
    dataset_dict = DatasetDict({
        "train": records_to_hf_dataset(train_recs, tokenizer, "training", out_dir),
        "val":   records_to_hf_dataset(val_recs,   tokenizer, "validation", out_dir),
        "test":  records_to_hf_dataset(test_recs,  tokenizer, "test", out_dir),
    })
    log.info(dataset_dict)

    # ── Save ──────────────────────────────────────────────────────────────────
    dataset_dict.save_to_disk(str(out_dir / "hf_dataset"))
    log.info(f"HF DatasetDict saved → {out_dir / 'hf_dataset'}")

    weights_path = out_dir / "class_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump({
            "source":       "training_split_only",
            "total_train":  len(train_recs),
            "n_labelsets":  len(weights),
            "weights":      weights,
        }, f, ensure_ascii=False, indent=2)
    log.info(f"Class weights saved   → {weights_path}")

    # ── Split index (for reproducibility / audit) ─────────────────────────────
    split_index = {
        "seed":        args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio":   args.val_ratio,
        "test_ratio":  round(1 - args.train_ratio - args.val_ratio, 4),
        "forced_train": args.force_ids_train,
        "forced_val":   args.force_ids_val,
        "forced_test":  args.force_ids_test,
        "counts": {
            "train": len(train_recs),
            "val":   len(val_recs),
            "test":  len(test_recs),
        },
        "train_ids": [r["id"] for r in train_recs],
        "val_ids":   [r["id"] for r in val_recs],
        "test_ids":  [r["id"] for r in test_recs],
    }
    index_path = out_dir / "split_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(split_index, f, ensure_ascii=False, indent=2)
    log.info(f"Split index saved     → {index_path}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
