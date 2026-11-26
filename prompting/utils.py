import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class LogEntry:
    id: str                                      # Label Studio task id
    report: str                                  # report text
    true_label: str                              # positivo/negativo, or comma-separated classes
    confidence: int                              # annotator confidence in true_label
    id_refertante: int                           # annotator id
    status: str                                  # success / failed

    uuid: Optional[str] = None
    set_id: Optional[str] = None                           # few-shot set used
    task: Optional[str] = None                             # "binary" o "multiclass"
    label_evidence: Optional[dict[str, str]] = None        # report span highlighted for each label (multiclass only)
    shot_ids: List[str] = field(default_factory=list)
    num_shot: int = 0
    predicted_label: Optional[str] = None
    model_confidence: Optional[int] = None
    model_motivation: Optional[str] = None
    response: Optional[str] = None                         # raw ChatCompletion object at runtime
    error: Optional[str] = None
    prompt: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────

def save_logs_as_json(log_entries: list, filepath: str) -> None:
    """Save log entries as pretty-formatted JSON."""
    def default_serializer(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, ensure_ascii=False, indent=2, default=default_serializer)
    except (IOError, OSError, PermissionError, TypeError) as e:
        print(f"Error saving logs to {filepath}: {e}")

def get_latest_annotation(report: dict) -> dict:
    """
    Return the most recent annotation for a Label Studio task.

    When a user re-annotates a task, Label Studio appends the new annotation
    to the 'annotations' list rather than overwriting the old one. This function
    returns the annotation with the latest 'updated_at' timestamp, falling back
    to the last element in the list if timestamps are missing.

    Args:
        report (dict): a single Label Studio task dict
    Returns:
        dict: the latest annotation dict
    Raises:
        ValueError: if the task has no annotations
    """
    annotations = report.get("annotations", [])
    if not annotations:
        raise ValueError(f"Task ID {report.get('id', 'UNKNOWN')} has no annotations.")

    # Sort by updated_at descending; fall back to list order if field is absent
    try:
        return max(annotations, key=lambda a: a.get("updated_at") or "")
    except Exception:
        return annotations[-1]


# ──────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────


def filter_reports(data: list, id_sets: list, id_key: str = "id") -> list:
    """
    Remove from data all reports whose id appears in any of the given id_sets.

    Args:
        data (list): original list of Label Studio tasks
        id_sets (list): list of sets/lists of IDs to exclude (e.g. shot sets)
        id_key (str): key used as identifier (default: 'id')
    Returns:
        list: filtered list
    """
    all_ids_to_remove = set()
    for id_set in id_sets:
        all_ids_to_remove.update(id_set)
    return [item for item in data if item.get(id_key) not in all_ids_to_remove]


# ──────────────────────────────────────────────
# Shot builders
# ──────────────────────────────────────────────

def create_shots(data: list, id_set: list, id_key: str = "id") -> str:
    shots = []
    id_to_dict = {item.get(id_key): item for item in data if id_key in item}

    for i, target_id in enumerate(id_set):
        if target_id not in id_to_dict:
            print(f"Warning: ID {target_id} not found in the JSON")
            continue

        item       = id_to_dict[target_id]
        annotation = get_latest_annotation(item)
        results    = annotation["result"]

        value1 = item["data"]["TestoRefertoStringa"]

        # Use from_name lookup instead of fragile index access
        status_dict = next(
            (r for r in results if r.get("from_name") == "status_class"), None
        )
        if status_dict is None:
            print(f"Warning: status_class not found for ID {target_id}")
            continue
        value2 = str(status_dict["value"]["choices"][0]).lower()

        shots.append(f"Esempio {i+1}\nReport: {value1} \nValutazione: {value2}")

    return "\n\n".join(shots)


def create_shots_multiclass(data: list, id_set: list, id_key: str = "id", include_evidence: bool = False) -> str:
    shots = []
    id_to_dict = {item.get(id_key): item for item in data if id_key in item}

    LABEL_FROM_NAMES = {"label_vasc", "label_altro", "label_neur", "label_inf", "label-neo"}

    for i, target_id in enumerate(id_set):
        if target_id not in id_to_dict:
            print(f"Warning: ID {target_id} not found in the JSON")
            continue

        item       = id_to_dict[target_id]
        annotation = get_latest_annotation(item)
        results    = annotation["result"]
        shot       = []

        full_text = item["data"]["TestoRefertoStringa"]
        category_dict = next(
            (r for r in results if r.get("from_name") == "category_class"), None
        )
        if category_dict is None:
            print(f"Warning: category_class not found for ID {target_id}")
            continue

        classes = category_dict.get("value", {}).get("choices", [])
        str_classes = ", ".join(classes)
        shot.append(f"Example {i+1}\nReport: {full_text} \nLabels: {str_classes}")

        if include_evidence:
            label_evidence = {
                r["value"]["labels"][0]: r["value"]["text"]
                for r in results
                if r.get("from_name") in LABEL_FROM_NAMES
                and r.get("type") == "labels"
                and r["value"].get("labels")           # non-empty list
                and r["value"]["labels"][0] in classes
            }
            for label, text_span in label_evidence.items():
                shot.append(f"Key evidence for class '{label}': '{text_span}'")

        shots.append("\n".join(shot))

    return "\n\n".join(shots)


# ──────────────────────────────────────────────
# Statistics — binary
# ──────────────────────────────────────────────


def calculate_statistics_full(log_entries: list) -> dict:
    """Accuracy, precision, recall, F1 grouped by shot-set (uses 'set_id' key)."""
    statistics = {}
    for entry in log_entries:
        set_id = entry["set_id"]
        if set_id not in statistics:
            statistics[set_id] = {
                "correct": 0, "total": 0, "accuracy": 0.0,
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            }
        s = statistics[set_id]
        s["total"] += 1
        label = entry["true_label"]
        pred  = entry["predicted_label"]
        if pred is not None:
            if label == "positivo" and pred == "positivo":   s["tp"] += 1
            elif label != "positivo" and pred == "positivo": s["fp"] += 1
            elif label == "positivo" and pred != "positivo": s["fn"] += 1
        if label == pred:
            s["correct"] += 1

    for s in statistics.values():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        s["accuracy"]  = (s["correct"] / s["total"]) * 100 if s["total"] > 0 else 0
        s["precision"] = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        s["recall"]    = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        p, r = s["precision"], s["recall"]
        s["f1"] = (2 * p * r) / (p + r) if (p + r) > 0 else 0
    return statistics


# ──────────────────────────────────────────────
# Statistics — multiclass
# ──────────────────────────────────────────────

def calculate_statistics_multiclass(log_entries: list) -> dict:
    """
    Multilabel metrics: exact-match accuracy, Hamming score,
    micro/macro precision/recall/F1, and per-class breakdown.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, hamming_loss, classification_report,
    )
    from sklearn.preprocessing import MultiLabelBinarizer

    stats = {
        "total": 0, "valid_predictions": 0, "invalid_predictions": 0,
        "exact_match_accuracy": 0.0, "hamming_score": 0.0,
        "precision_micro": 0.0, "recall_micro": 0.0, "f1_micro": 0.0,
        "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0,
        "per_class_metrics": {},
    }
    y_true, y_pred = [], []

    for entry in log_entries:
        stats["total"] += 1
        try:
            # handle both LogEntry dataclass and plain dicts
            if hasattr(entry, "true_label"):
                true_raw = entry.true_label
                pred_raw = entry.predicted_label
            else:
                true_raw = entry.get("true_label")
                pred_raw = entry.get("predicted_label")

            if not true_raw or not pred_raw:
                stats["invalid_predictions"] += 1
                continue

            true_labels = sorted(set(s.strip() for s in true_raw.split(",")))
            pred_labels = sorted(set(s.strip() for s in pred_raw.split(",")))

            y_true.append(true_labels)
            y_pred.append(pred_labels)
            stats["valid_predictions"] += 1
        except Exception as e:
            print(f"Warning: could not process entry: {e}")
            stats["invalid_predictions"] += 1

    if stats["valid_predictions"] == 0:
        print("Warning: no valid predictions found.")
        return stats

    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    exact = sum(1 for yt, yp in zip(y_true, y_pred) if set(yt) == set(yp))
    stats["exact_match_accuracy"] = (exact / stats["valid_predictions"]) * 100
    stats["hamming_score"]        = (1 - hamming_loss(y_true_bin, y_pred_bin)) * 100
    stats["precision_micro"]      = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0) * 100
    stats["recall_micro"]         = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0) * 100
    stats["f1_micro"]             = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0) * 100
    stats["precision_macro"]      = precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0) * 100
    stats["recall_macro"]         = recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0) * 100
    stats["f1_macro"]             = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0) * 100

    class_names = mlb.classes_
    prec_cls  = precision_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    rec_cls   = recall_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    f1_cls    = f1_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        stats["per_class_metrics"][name] = {
            "precision": prec_cls[i] * 100,
            "recall":    rec_cls[i] * 100,
            "f1":        f1_cls[i] * 100,
            "support":   sum(1 for labels in y_true if name in labels),
        }

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true_bin, y_pred_bin, target_names=class_names, zero_division=0))
    return stats
