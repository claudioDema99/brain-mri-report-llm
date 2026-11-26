"""
label_powerset.py
──────────────────────────────────────────────────────────────────────────────
1. Reads referti.json
2. Filters: keeps only Encefalo + Patologici (excludes Negativo)
3. Builds a concise record per report: id, text, labels, label_evidence
4. Saves a compact copy of the filtered records (id, text, labels, evidence)
   to filtered_reports.json, for inspection (training uses filter_reports.py)
5. Computes the full label powerset (every unique label combination)
   and saves it to label_powerset.json

Usage:
    python finetuning/label_powerset.py
    python finetuning/label_powerset.py --input reports.json --out_dir ./data/derived
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

# from_name values that carry highlighted text evidence per label
LABEL_FROM_NAMES = {
    "label_vasc",
    "label_neur",
    "label_inf",
    "label-neo",
    "label_altro",
}

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("prepare_dataset")


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_latest_annotation(report: dict) -> dict:
    """
    Returns the most recent annotation for a report.
    Mirrors the helper used in the original codebase.
    Raises ValueError if no annotations are present.
    """
    annotations = report.get("annotations", [])
    if not annotations:
        raise ValueError(f"No annotations for report id={report.get('id')}")
    # Sort by updated_at descending; fall back to list order if field is absent
    try:
        return max(annotations, key=lambda a: a.get("updated_at") or "")
    except Exception:
        return annotations[-1]


def get_choice(results: list, from_name: str) -> str | None:
    """
    Returns the first choice value for a given from_name field, or None.
    """
    entry = next((r for r in results if r.get("from_name") == from_name), None)
    if entry is None:
        return None
    choices = entry.get("value", {}).get("choices", [])
    return choices[0] if choices else None


def extract_label_evidence(results: list) -> dict[str, list[str]]:
    """
    Collects all highlighted text spans per label.
    A label may have multiple highlighted spans, so we collect them as a list.

    Returns:
        { "Vascolare": ["testo highlight 1", "testo highlight 2"], ... }
    """
    evidence: dict[str, list[str]] = {}
    for r in results:
        if r.get("from_name") not in LABEL_FROM_NAMES:
            continue
        if r.get("type") != "labels":
            continue
        labels = r.get("value", {}).get("labels", [])
        text   = r.get("value", {}).get("text", "").strip()
        for label in labels:
            if label not in evidence:
                evidence[label] = []
            if text:
                evidence[label].append(text)
    return evidence


def labelset_key(labels: list[str]) -> str:
    """
    Canonical string key for a label combination, alphabetically sorted.
    e.g. ["Neoformazione", "Vascolare"] → "Neoformazione + Vascolare"
    """
    return " + ".join(sorted(labels))


# ── Step 1+2: Filter ───────────────────────────────────────────────────────────

def filter_reports(data: list[dict]) -> tuple[list[dict], dict]:
    """
    Applies both filters sequentially:
      - keep only district == "Encefalo"
      - exclude status == "Negativo"

    Returns:
        (filtered_reports, stats)
    """
    stats = {
        "total":            len(data),
        "no_annotation":    0,
        "encefalo":         0,
        "non_encefalo":     0,
        "negativo":         0,
        "final":            0,
    }

    encefalo_reports = []
    for report in data:
        try:
            results = get_latest_annotation(report)["result"]
        except (ValueError, KeyError, TypeError):
            stats["no_annotation"] += 1
            continue

        district = get_choice(results, "district_type")
        if district == "Encefalo":
            encefalo_reports.append(report)
        else:
            stats["non_encefalo"] += 1

    stats["encefalo"] = len(encefalo_reports)

    positive_reports = []
    for report in encefalo_reports:
        try:
            results = get_latest_annotation(report)["result"]
        except (ValueError, KeyError, TypeError):
            continue

        status = get_choice(results, "status_class")
        if status == "Negativo":
            stats["negativo"] += 1
            continue

        positive_reports.append(report)

    stats["final"] = len(positive_reports)
    return positive_reports, stats


# ── Step 3: Build concise records ─────────────────────────────────────────────

def build_records(reports: list[dict]) -> tuple[list[dict], dict]:
    """
    For each report extracts:
        id, report_text, labels (list), label_evidence (dict)

    Returns:
        (records, extraction_stats)
    """
    records = []
    extraction_stats = {"ok": 0, "skipped": 0, "skip_reasons": Counter()}

    for report in reports:
        report_id   = report.get("id")
        report_text = report.get("data", {}).get("TestoRefertoStringa", "").strip()

        try:
            results = get_latest_annotation(report)["result"]

            # ── Labels ───────────────────────────────────────────────────────
            category_dict = next(
                (r for r in results if r.get("from_name") == "category_class"),
                None,
            )
            if category_dict is None:
                raise KeyError("missing category_class")

            labels = category_dict["value"].get("choices", [])
            if not labels:
                raise ValueError("empty labels list")

            if not report_text:
                raise ValueError("empty report text")

            # ── Evidence ─────────────────────────────────────────────────────
            label_evidence = extract_label_evidence(results)

            records.append({
                "id":             report_id,
                "report_text":    report_text,
                "labels":         sorted(labels),          # sorted for reproducibility
                "label_evidence": label_evidence,
            })
            extraction_stats["ok"] += 1

        except Exception as e:
            reason = type(e).__name__
            extraction_stats["skipped"] += 1
            extraction_stats["skip_reasons"][reason] += 1
            log.debug(f"Skipped report id={report_id}: {e}")

    return records, extraction_stats


# ── Step 5: Compute label powerset ────────────────────────────────────────────

def compute_powerset(records: list[dict]) -> dict:
    """
    Enumerates every unique label combination (labelset) across all records.
    Each record may have 1, 2, 3 or more labels — all treated as a single
    powerset entry without splitting.

    Returns a structured dict:
    {
        "total_records": int,
        "n_unique_labelsets": int,
        "labelsets": [
            {
                "labelset":   "Vascolare + Neoformazione",
                "labels":     ["Neoformazione", "Vascolare"],
                "n_labels":   2,
                "count":      44,
                "frequency":  0.038,
                "report_ids": [...]
            },
            ...
        ]
    }
    Sorted by count descending.
    """
    combo_counts:   Counter = Counter()
    combo_labels:   dict[str, list[str]] = {}
    combo_ids:      dict[str, list] = {}

    for rec in records:
        key    = labelset_key(rec["labels"])
        labels = sorted(rec["labels"])

        combo_counts[key]  += 1
        combo_labels[key]   = labels
        if key not in combo_ids:
            combo_ids[key] = []
        combo_ids[key].append(rec["id"])

    total = len(records)
    labelsets = []
    for key, count in combo_counts.most_common():
        labelsets.append({
            "labelset":   key,
            "labels":     combo_labels[key],
            "n_labels":   len(combo_labels[key]),
            "count":      count,
            "frequency":  round(count / total, 4) if total else 0,
            "report_ids": combo_ids[key],
        })

    return {
        "total_records":      total,
        "n_unique_labelsets": len(labelsets),
        "labelsets":          labelsets,
    }


def compute_class_weights(powerset: dict) -> list[dict]:
    """
    Computes balanced class weights for each labelset using:
        weight(i) = N_total / (N_classes * N_i)

    This mirrors sklearn's compute_class_weight("balanced") logic and is
    directly usable as token-level loss weights in WeightedLossTrainer.

    Returns the labelsets list enriched with a 'class_weight' field.
    """
    total   = powerset["total_records"]
    n_cls   = powerset["n_unique_labelsets"]
    for entry in powerset["labelsets"]:
        entry["class_weight"] = round(total / (n_cls * entry["count"]), 4)
    return powerset["labelsets"]


# ── CLI + orchestration ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Filter referti.json and compute label powerset")
    p.add_argument("--input",   default="referti.json",      help="Path to referti.json")
    p.add_argument("--out_dir", default=".",                 help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info(f"Reading {input_path} ...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    log.info(f"Total reports in file: {len(data)}")

    # ── Filter ────────────────────────────────────────────────────────────────
    log.info("Applying filters (Encefalo + Patologici) ...")
    filtered, fstats = filter_reports(data)

    log.info(f"  Total input:        {fstats['total']}")
    log.info(f"  No annotation:      {fstats['no_annotation']}")
    log.info(f"  Non-Encefalo:       {fstats['non_encefalo']}")
    log.info(f"  Encefalo:           {fstats['encefalo']}")
    log.info(f"  Excluded Negativo:  {fstats['negativo']}")
    log.info(f"  → Final (Enc+Pat):  {fstats['final']}")

    # ── Build records ─────────────────────────────────────────────────────────
    log.info("Extracting concise records ...")
    records, estats = build_records(filtered)

    log.info(f"  Extracted OK:  {estats['ok']}")
    log.info(f"  Skipped:       {estats['skipped']}")
    if estats["skip_reasons"]:
        for reason, cnt in estats["skip_reasons"].items():
            log.info(f"    {reason}: {cnt}")

    # ── Save filtered records ─────────────────────────────────────────────────
    filtered_path = out_dir / "filtered_reports.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info(f"Saved filtered records → {filtered_path}")

    # ── Powerset ──────────────────────────────────────────────────────────────
    log.info("Computing label powerset ...")
    powerset = compute_powerset(records)
    compute_class_weights(powerset)  # enriches in place

    log.info(f"  Unique labelsets found: {powerset['n_unique_labelsets']}")
    log.info("  Top 10 labelsets by frequency:")
    for entry in powerset["labelsets"][:10]:
        log.info(
            f"    [{entry['count']:>4}  {entry['frequency']*100:5.1f}%"
            f"  w={entry['class_weight']:.3f}]  {entry['labelset']}"
        )

    powerset_path = out_dir / "label_powerset.json"
    with open(powerset_path, "w", encoding="utf-8") as f:
        json.dump(powerset, f, ensure_ascii=False, indent=2)
    log.info(f"Saved label powerset   → {powerset_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info(f"Done. {len(records)} records | {powerset['n_unique_labelsets']} unique labelsets")
    log.info(f"Outputs: {filtered_path}, {powerset_path}")


if __name__ == "__main__":
    main()
