"""
Label co-occurrence analysis for a dataset of radiology reports (Label Studio export).
Produces:
  - co-occurrence matrix (heatmap)
  - distribution of the number of labels per sample
  - counts for each unique label combination
  - a text summary

Usage:
    python analysis/cooccurrence_analysis.py --input reports.json [--output ./output]
"""

import json
import argparse
import os
from itertools import combinations
from collections import Counter

# ── optional plotting dependencies ─────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib/seaborn/numpy not found, text output only.\n"
          "       Install with: pip install matplotlib seaborn numpy")

# ── label set ──────────────────────────────────────────────────────────────────
LABELS = [
    "Neoformazione",
    "Vascolare",
    "Infiammazione o Infezione",
    "Neurodegenerativo",
    "Altro",
]

# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_labels(referto: dict) -> tuple[int | None, list[str] | None]:
    """
    Return the id and the label list of one dataset record,
    or (id, None) if the record has no valid annotation.
    """
    report_id = referto.get("id")
    annotations = referto.get("annotations", [])

    if not annotations:
        return report_id, None

    # Use the most recent annotation if there are several
    annotation = (
        max(annotations, key=lambda a: a.get("updated_at") or "")
        if len(annotations) > 1
        else annotations[0]
    )

    category_dict = next(
        (r for r in annotation["result"] if r.get("from_name") == "category_class"),
        None,
    )

    if category_dict is None:
        return report_id, None

    labels = category_dict["value"]["choices"]
    # Drop unknown labels
    labels = [l for l in labels if l in LABELS]
    return report_id, labels if labels else None


def load_dataset(path: str) -> tuple[list[list[str]], list[int]]:
    """Load the JSON and return (label sets, ids without labels)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_label_sets = []
    missing = []

    for referto in data:
        rid, labels = extract_labels(referto)
        if labels is None:
            missing.append(rid)
        else:
            all_label_sets.append(sorted(labels))  # canonical order

    return all_label_sets, missing


# ─────────────────────────────────────────────────────────────────────────────
# Co-occurrence counts
# ─────────────────────────────────────────────────────────────────────────────

def build_cooccurrence_matrix(label_sets: list[list[str]]) -> dict:
    """
    Build the N×N co-occurrence matrix:
    cooc[i][j] = number of samples with both label i and label j,
    cooc[i][i] = total occurrences of label i.
    """
    n = len(LABELS)
    idx = {l: i for i, l in enumerate(LABELS)}
    matrix = [[0] * n for _ in range(n)]

    for label_set in label_sets:
        for l in label_set:
            i = idx[l]
            matrix[i][i] += 1                          # diagonal = occurrences
        for l1, l2 in combinations(label_set, 2):
            i, j = idx[l1], idx[l2]
            matrix[i][j] += 1
            matrix[j][i] += 1                          # symmetric

    return matrix


def count_combinations(label_sets: list[list[str]]) -> Counter:
    """Count each unique label combination."""
    return Counter(tuple(ls) for ls in label_sets)


# ─────────────────────────────────────────────────────────────────────────────
# Text output
# ─────────────────────────────────────────────────────────────────────────────

def print_report(label_sets, matrix, combo_counter, missing, total):
    n_valid = len(label_sets)
    label_counts = {LABELS[i]: matrix[i][i] for i in range(len(LABELS))}

    print("=" * 65)
    print("  DATASET ANALYSIS: RADIOLOGY REPORTS")
    print("=" * 65)
    print(f"\nSamples in file        : {total}")
    print(f"Samples with labels    : {n_valid}")
    print(f"Samples without labels : {len(missing)}")
    if missing:
        print(f"  IDs without labels: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    print("\n── Occurrences per label ────────────────────────────────────")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / n_valid * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<35} {count:>4}  {pct:5.1f}%  {bar}")

    print("\n── Number of labels per sample ──────────────────────────────")
    nlabel_dist = Counter(len(ls) for ls in label_sets)
    for k in sorted(nlabel_dist):
        pct = nlabel_dist[k] / n_valid * 100
        print(f"  {k} label(s)  →  {nlabel_dist[k]:>4} samples  ({pct:.1f}%)")

    print(f"\n  Mean labels per sample: {sum(len(ls) for ls in label_sets)/n_valid:.2f}")

    print("\n── Co-occurrence matrix ─────────────────────────────────────")
    # Header
    short = ["Neo", "Vasc", "Inf/Inf", "Neurod", "Altro"]
    header = f"  {'':>10}" + "".join(f"{s:>9}" for s in short)
    print(header)
    for i, label in enumerate(LABELS):
        row = f"  {short[i]:>10}" + "".join(f"{matrix[i][j]:>9}" for j in range(len(LABELS)))
        print(row)

    print("\n── 15 most frequent label combinations ──────────────────────")
    for combo, count in combo_counter.most_common(15):
        pct = count / n_valid * 100
        label_str = " + ".join(combo) if combo else "(none)"
        print(f"  {count:>4}  ({pct:5.1f}%)  {label_str}")

    # ── Imbalance check ──
    print("\n── Imbalance check ──────────────────────────────────────────")
    min_label = min(label_counts, key=label_counts.get)
    min_count = label_counts[min_label]
    max_label = max(label_counts, key=label_counts.get)
    max_count = label_counts[max_label]
    ratio = max_count / min_count if min_count > 0 else float("inf")

    print(f"  Rarest label     : {min_label} ({min_count})")
    print(f"  Most common label: {max_label} ({max_count})")
    print(f"  Max/min ratio    : {ratio:.1f}×")

    if ratio > 3:
        print("  ⚠ Strong imbalance: consider oversampling or class weights")
    else:
        print("  ✓ Imbalance is acceptable")

    # co-occurrences of the rarest label
    min_idx = LABELS.index(min_label)
    print(f"\n  Co-occurrences of '{min_label}' with the other labels:")
    for j, other in enumerate(LABELS):
        if j != min_idx and matrix[min_idx][j] > 0:
            pct_of_min = matrix[min_idx][j] / label_counts[min_label] * 100
            print(f"    + {other:<35} {matrix[min_idx][j]:>4}  ({pct_of_min:.1f}% of samples with {min_label})")

    print("\n" + "=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_cooccurrence(matrix, output_dir: str):
    if not HAS_PLOT:
        return

    import numpy as np

    arr = np.array(matrix, dtype=float)
    # Normalize each row by its diagonal
    diag = np.diag(arr).copy()
    diag[diag == 0] = 1
    arr_norm = arr / diag[:, None]   # co-occurrence relative to the row label's occurrences

    short = ["Neoform.", "Vascolare", "Inf/Infez.", "Neurodeg.", "Altro"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Label co-occurrence: radiology reports", fontsize=13, fontweight="bold")

    # ─ Plot 1: absolute counts ─
    mask_diag = np.eye(len(LABELS), dtype=bool)
    abs_off = arr.copy()
    abs_off[mask_diag] = np.nan   # hide the diagonal for readability

    sns.heatmap(
        abs_off, annot=True, fmt=".0f", cmap="Blues",
        xticklabels=short, yticklabels=short,
        ax=axes[0], linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Co-occurrences (n)"}
    )
    axes[0].set_title("Absolute co-occurrences (off-diagonal)", fontsize=11)

    # ─ Plot 2: row-normalized ─
    arr_norm_off = arr_norm.copy()
    arr_norm_off[mask_diag] = np.nan

    sns.heatmap(
        arr_norm_off, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=short, yticklabels=short,
        ax=axes[1], linewidths=0.5, linecolor="white",
        vmin=0, vmax=1,
        cbar_kws={"label": "Fraction (0–1)"}
    )
    axes[1].set_title("Relative co-occurrence (row-normalized)", fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cooccurrence_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Heatmap saved to: {out_path}")
    plt.close()


def plot_label_distribution(label_sets, output_dir: str):
    if not HAS_PLOT:
        return

    import numpy as np

    label_counts = {l: 0 for l in LABELS}
    for ls in label_sets:
        for l in ls:
            label_counts[l] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Dataset distribution", fontsize=13, fontweight="bold")

    # ─ Occurrences per label ─
    labels_sorted = sorted(label_counts, key=label_counts.get, reverse=True)
    counts_sorted = [label_counts[l] for l in labels_sorted]
    colors = ["#A32D2D" if c < 200 else "#BA7517" if c < 300 else "#1D9E75"
              for c in counts_sorted]
    short = [l[:12] for l in labels_sorted]

    axes[0].bar(short, counts_sorted, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Occurrences per label", fontsize=11)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts_sorted):
        axes[0].text(i, v + 5, str(v), ha="center", va="bottom", fontsize=10)

    # ─ Number of labels per sample ─
    nlabel_dist = Counter(len(ls) for ls in label_sets)
    ks = sorted(nlabel_dist.keys())
    vs = [nlabel_dist[k] for k in ks]
    axes[1].bar([str(k) for k in ks], vs, color="#185FA5", edgecolor="white")
    axes[1].set_title("Labels per sample", fontsize=11)
    axes[1].set_xlabel("Number of labels")
    axes[1].set_ylabel("Samples")
    for i, v in enumerate(vs):
        axes[1].text(i, v + 2, str(v), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "label_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Distribution saved to: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Label co-occurrence analysis of the report dataset")
    parser.add_argument("--input",  required=True, help="Label Studio JSON export")
    parser.add_argument("--output", default="./output", help="Output folder for plots and CSVs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"[INFO] Loading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    total = len(raw)

    label_sets, missing = load_dataset(args.input)
    matrix = build_cooccurrence_matrix(label_sets)
    combo_counter = count_combinations(label_sets)

    print_report(label_sets, matrix, combo_counter, missing, total)
    plot_cooccurrence(matrix, args.output)
    plot_label_distribution(label_sets, args.output)

    # ── Matrix as CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output, "cooccurrence_matrix.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(LABELS) + "\n")
        for i, label in enumerate(LABELS):
            row = ",".join(str(matrix[i][j]) for j in range(len(LABELS)))
            f.write(f"{label},{row}\n")
    print(f"[OK] Matrix CSV saved to: {csv_path}")

    # ── Combinations as CSV ───────────────────────────────────────────────────
    combo_path = os.path.join(args.output, "label_combinations.csv")
    with open(combo_path, "w", encoding="utf-8") as f:
        f.write("combination,count,percent\n")
        for combo, count in combo_counter.most_common():
            pct = count / len(label_sets) * 100
            f.write(f"\"{' + '.join(combo)}\",{count},{pct:.2f}\n")
    print(f"[OK] Combinations CSV saved to: {combo_path}")


if __name__ == "__main__":
    main()