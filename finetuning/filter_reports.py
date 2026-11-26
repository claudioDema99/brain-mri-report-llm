"""
filter_reports.py
────────────────────────────────────────
Keep the reports used for the multiclass task and for fine-tuning:
district "Encefalo" and status other than "Negativo", according to the most
recent annotation. Records are copied unchanged (Label Studio format), which
is the input format expected by training_data.py.

The multiclass prompting pipeline applies the same filter internally.

Usage:
    python finetuning/filter_reports.py --input reports.json --output filtered_reports.json
"""

import argparse
import json


def latest_annotation(task: dict) -> dict | None:
    annotations = task.get("annotations", [])
    if not annotations:
        return None
    return max(annotations, key=lambda a: a.get("updated_at") or "")


def choice(task: dict, from_name: str) -> str | None:
    annotation = latest_annotation(task)
    if annotation is None:
        return None
    item = next((r for r in annotation["result"] if r.get("from_name") == from_name), None)
    return item["value"]["choices"][0] if item else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Label Studio JSON export")
    parser.add_argument("--output", required=True, help="where to write the filtered reports")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        tasks = json.load(f)
    kept = [
        t for t in tasks
        if choice(t, "district_type") == "Encefalo" and choice(t, "status_class") != "Negativo"
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"{len(kept)} of {len(tasks)} reports kept -> {args.output}")


if __name__ == "__main__":
    main()
