#!/usr/bin/env python3
"""
multiclass_analysis.py
──────────────────────────────────────────────────────────────────────────────
Complete analysis pipeline for the MULTICLASS (multilabel) task logs.
Produces publication-quality PNG figures and structured CSV/JSON outputs.

Labels: Neoformazione, Vascolare, Infiammazione o Infezione,
        Neurodegenerativo, Altro  (Label Powerset formulation)

Analyses:
  A1  Metrics per config: macro/micro F1, exact match, per-label P/R/F1
  A2  Learning curves: macro F1 vs shot count (mean ± std across sets)
  A3  Prompt comparison
  A4  Model comparison
  A5  Per-label aggregated analysis (hardest label per model)
  A6  Per-label majority voting vs mean individual set
  B3  Double confidence filter (human_conf=5 AND model_conf=5)
  C1  LP error taxonomy (FN-partial, FP-partial, mixed, complete)
  C2  Per-label confusion: which labels are confused with which
  C4  Systematically wrong report IDs
  C6  Error type change: 0-shot vs few-shot
  D1  Stratified reasoning sample per error type
  E1  Kruskal-Wallis omnibus + pairwise Mann-Whitney U + FDR
  E2  Chi-squared / Fisher exact: do models make different types of errors?

Usage:
    python analysis/multiclass_analysis.py --root_dir ./results_multiclass --output_dir ./figures/multiclass
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import random
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODELS      = ["gpt", "mistral", "gemma", "qwen"]
MODEL_DISPLAY_NAMES = {
    "gpt":     "GPT-OSS",
    "mistral": "Magistral-S",
    "gemma":   "Gemma-4",
    "qwen":    "Qwen3.5",
}
PROMPTS     = ["A", "B", "C", "D"]
PROMPT_DISPLAY = {"A": "1", "B": "2", "C": "3", "D": "4"}
SHOT_COUNTS = [0, 5, 10, 20]

VALID_LABELS = [
    "Neoformazione",
    "Vascolare",
    "Infiammazione o Infezione",
    "Neurodegenerativo",
    "Altro",
]
VALID_LABELS_SET = set(VALID_LABELS)

# English display names for labels (for publication figures)
LABEL_EN = {
    "Neoformazione":            "Neoformation",
    "Vascolare":                "Vascular",
    "Infiammazione o Infezione":"Inflammation or Infection",
    "Neurodegenerativo":        "Neurodegenerative",
    "Altro":                    "Other",
}

# LP error taxonomy categories (for error entries only)
ERROR_TYPES = ["fn_partial_only", "fp_partial_only", "mixed", "complete_error"]
ERROR_LABELS = {
    "fn_partial_only": "FN-partial only",
    "fp_partial_only": "FP-partial only",
    "mixed":           "Mixed (FN+FP)",
    "complete_error":  "Complete error",
}

N_MOTIVATION_SAMPLES = 5

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

MODEL_COLORS = {"gpt": "#DD8452", "mistral": "#55A868", "gemma": "#4C72B0", "qwen": "#C44E52"}
SHOT_COLORS  = {0: "#5B9BD5", 5: "#ED7D31", 10: "#70AD47", 20: "#C44E52"}
PROMPT_COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52"}
LABEL_COLORS  = {
    "Neoformazione":            "#E74C3C",
    "Vascolare":                "#3498DB",
    "Infiammazione o Infezione":"#2ECC71",
    "Neurodegenerativo":        "#9B59B6",
    "Altro":                    "#F39C12",
}
ERROR_COLORS = {
    "fn_partial_only": "#3498DB",
    "fp_partial_only": "#E74C3C",
    "mixed":           "#8E44AD",
    "complete_error":  "#2C3E50",
}

def disp(model: str) -> str:
    """Return the publication display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model, model.upper())



# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def parse_label_set(raw) -> frozenset:
    """Parse comma-separated label string or list into a frozenset of valid labels."""
    if not raw:
        return frozenset()
    if isinstance(raw, list):
        return frozenset(s.strip() for s in raw if s.strip() in VALID_LABELS_SET)
    return frozenset(s.strip() for s in str(raw).split(",") if s.strip() in VALID_LABELS_SET)


def load_all_logs(root_dir: Path) -> pd.DataFrame:
    """
    Walk MULTICLASS directory tree and load every successful log entry.
    Structure: results_multiclass/{model}/prompt_{X}/{N}-shot(s)/set_{K}/logs/logs_*.json
    """
    records = []
    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir() or not prompt_dir.name.startswith("prompt_"):
                continue
            prompt = prompt_dir.name.removeprefix("prompt_")
            for shot_dir in sorted(prompt_dir.iterdir()):
                if not shot_dir.is_dir():
                    continue
                name = shot_dir.name
                if name == "0-shot":
                    shot_count = 0
                elif name.endswith("-shots"):
                    try:
                        shot_count = int(name.removesuffix("-shots"))
                    except ValueError:
                        continue
                else:
                    continue
                for set_dir in sorted(shot_dir.iterdir()):
                    if not set_dir.is_dir() or not set_dir.name.startswith("set_"):
                        continue
                    set_id   = set_dir.name
                    logs_dir = set_dir / "logs"
                    if not logs_dir.exists():
                        continue
                    # Accept any json file in logs dir
                    for log_file in sorted(logs_dir.glob("*.json")):
                        try:
                            with open(log_file, encoding="utf-8") as f:
                                entries = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            continue
                        for e in entries:
                            if e.get("status") != "success":
                                continue
                            true_set = parse_label_set(e.get("true_label", ""))
                            pred_set = parse_label_set(e.get("predicted_label", ""))
                            if not true_set:
                                continue
                            # confidence is int in multiclass (not string)
                            hc = e.get("confidence")
                            try:
                                human_conf = int(hc) if hc is not None else None
                            except (ValueError, TypeError):
                                human_conf = None
                            records.append({
                                "model":            model,
                                "prompt":           prompt,
                                "shot_count":       shot_count,
                                "set_id":           set_id,
                                "report_id":        e["id"],
                                "true_label":       e.get("true_label", ""),
                                "predicted_label":  e.get("predicted_label", ""),
                                "true_set":         true_set,
                                "pred_set":         pred_set,
                                "human_confidence": human_conf,
                                "model_confidence": e.get("model_confidence"),
                                "model_motivation": (e.get("model_motivation") or "" if isinstance(e.get("model_motivation"), str) else ""),
                                "label_evidence":   json.dumps(e.get("label_evidence") or {}),
                                "correct":          true_set == pred_set,
                                "error_type":       classify_error(true_set, pred_set),
                            })
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} entries")
    print(f"  Models     : {sorted(df['model'].unique())}")
    print(f"  Prompts    : {sorted(df['prompt'].unique())}")
    print(f"  Shot counts: {sorted(df['shot_count'].unique())}")
    print(f"  Unique IDs : {df['report_id'].nunique()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MULTILABEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def multilabel_metrics(sub: pd.DataFrame) -> dict:
    """
    Compute multilabel metrics from a sub-DataFrame with true_set / pred_set columns.
    Returns: exact_match, hamming_score, micro P/R/F1, macro P/R/F1, per_label.
    """
    if sub.empty:
        return {}
    n = len(sub)

    exact = int((sub["true_set"] == sub["pred_set"]).sum())

    # Jaccard-based Hamming score per sample
    def jaccard(t, p):
        union = t | p
        return len(t & p) / len(union) if union else 1.0

    hamming = float(np.mean([jaccard(r["true_set"], r["pred_set"])
                              for _, r in sub.iterrows()]))

    # Per-label TP/FP/FN
    tp = {l: 0 for l in VALID_LABELS}
    fp = {l: 0 for l in VALID_LABELS}
    fn = {l: 0 for l in VALID_LABELS}
    for _, row in sub.iterrows():
        t, p = row["true_set"], row["pred_set"]
        for l in VALID_LABELS:
            if   l in t and l in p: tp[l] += 1
            elif l in p:            fp[l] += 1
            elif l in t:            fn[l] += 1

    # Micro
    tp_m = sum(tp.values()); fp_m = sum(fp.values()); fn_m = sum(fn.values())
    pr_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) else 0.0
    re_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) else 0.0
    f1_m = 2*pr_m*re_m / (pr_m+re_m) if (pr_m+re_m) else 0.0

    # Macro + per-label
    per_label = {}
    macro_p = macro_r = macro_f1 = 0.0
    for l in VALID_LABELS:
        p = tp[l] / (tp[l]+fp[l]) if (tp[l]+fp[l]) else 0.0
        r = tp[l] / (tp[l]+fn[l]) if (tp[l]+fn[l]) else 0.0
        f = 2*p*r / (p+r) if (p+r) else 0.0
        per_label[l] = {"precision": p, "recall": r, "f1": f,
                         "support": tp[l]+fn[l], "tp": tp[l], "fp": fp[l], "fn": fn[l]}
        macro_p  += p; macro_r  += r; macro_f1 += f

    k = len(VALID_LABELS)
    return {
        "n":               n,
        "exact_match":     exact / n,
        "hamming_score":   hamming,
        "precision_micro": pr_m,
        "recall_micro":    re_m,
        "f1_micro":        f1_m,
        "precision_macro": macro_p / k,
        "recall_macro":    macro_r / k,
        "f1_macro":        macro_f1 / k,
        "per_label":       per_label,
    }


def classify_error(true_set: frozenset, pred_set: frozenset) -> str:
    """
    Classify a prediction into the LP error taxonomy.
    Returns one of: correct, fn_partial_only, fp_partial_only, mixed, complete_error.
    """
    if true_set == pred_set:
        return "correct"
    fn = true_set - pred_set  # labels missed
    fp = pred_set - true_set  # labels added
    if not (true_set & pred_set):
        return "complete_error"
    elif fn and fp:
        return "mixed"
    elif fn:
        return "fn_partial_only"
    else:
        return "fp_partial_only"


def per_label_majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, prompt, shot_count, report_id):
      For each VALID_LABEL: count sets predicting it.
      Include in MV prediction if count > n_sets / 2 (strict majority).
      Tie-breaking: exclude label (conservative).
    Returns one row per (model, prompt, shot_count, report_id).
    """
    rows = []
    for (model, prompt, sc), g in df.groupby(["model", "prompt", "shot_count"]):
        sets      = g["set_id"].unique()
        n_sets    = len(sets)
        threshold = n_sets / 2  # strict majority: need > threshold votes

        for rid, rg in g.groupby("report_id"):
            true_set = rg["true_set"].iloc[0]
            # Count per-label votes across sets
            label_votes = {l: 0 for l in VALID_LABELS}
            for _, row in rg.iterrows():
                for l in row["pred_set"]:
                    if l in label_votes:
                        label_votes[l] += 1
            mv_pred = frozenset(l for l, cnt in label_votes.items() if cnt > threshold)
            rows.append({
                "model":         model,
                "prompt":        prompt,
                "shot_count":    sc,
                "report_id":     rid,
                "true_set":      true_set,
                "pred_set":      mv_pred,
                "n_sets":        n_sets,
                "correct":       mv_pred == true_set,
                "error_type":    classify_error(true_set, mv_pred),
            })
    return pd.DataFrame(rows)


def save_fig(fig: plt.Figure, name: str, output_dir: Path) -> None:
    path = output_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.name}")



# ══════════════════════════════════════════════════════════════════════════════
#  A1 — Compute metrics dataframe (no plots)
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a1(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Macro/micro F1, exact match, hamming, per-label P/R/F1 for every
    (model x prompt x shot_count x set_id) combination.
    """
    print("\n-- A1: Metrics per configuration --")
    rows = []
    for (model, prompt, sc, sid), g in df.groupby(
        ["model", "prompt", "shot_count", "set_id"]
    ):
        m = multilabel_metrics(g)
        if m:
            flat = {"model": model, "prompt": prompt, "shot_count": sc, "set_id": sid}
            for k, v in m.items():
                if k == "per_label":
                    for lbl, lm in v.items():
                        for metric, val in lm.items():
                            flat[f"{lbl}__{metric}"] = val
                else:
                    flat[k] = v
            rows.append(flat)

    a1 = pd.DataFrame(rows)
    a1.to_csv(output_dir / "a1_metrics_per_config.csv", index=False)
    print(f"  Saved a1_metrics_per_config.csv ({len(a1)} rows)")


    return a1



# ══════════════════════════════════════════════════════════════════════════════
#  A2 — Learning curves
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a2(a1: pd.DataFrame, output_dir: Path) -> None:
    """Macro F1 vs shot count (mean ± std), one subplot per (model x prompt)."""
    print("\n-- A2: Learning curves --")
    models_p  = [m for m in MODELS  if m in a1["model"].unique()]
    prompts_p = [p for p in PROMPTS if p in a1["prompt"].unique()]
    nrows, ncols = len(models_p), len(prompts_p)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5*ncols, 4.8*nrows),
                              sharey=True, sharex=True)
    if nrows == 1 and ncols == 1: axes = [[axes]]
    elif nrows == 1: axes = [list(axes)]
    elif ncols == 1: axes = [[ax] for ax in axes]

    for i, model in enumerate(models_p):
        for j, prompt in enumerate(prompts_p):
            ax  = axes[i][j]
            sub = a1[(a1["model"] == model) & (a1["prompt"] == prompt)]
            agg = (sub.groupby("shot_count")["f1_macro"]
                      .agg(mean="mean", std="std").reset_index().fillna({"std": 0}))
            color = MODEL_COLORS.get(model, "#333")
            ax.plot(agg["shot_count"], agg["mean"], color=color, marker="o", lw=3, ms=11)
            ax.fill_between(agg["shot_count"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            color=color, alpha=0.15)
            ax.set_xticks(SHOT_COUNTS); ax.set_ylim(0.4, 0.9)
            ax.tick_params(axis="both", labelsize=22)
            ax.set_title(f"{disp(model)} — Prompt M{PROMPT_DISPLAY.get(prompt, prompt)}",
                         fontsize=26)
            if j == 0: ax.set_ylabel("Macro F1", fontsize=24)
            if i == nrows-1: ax.set_xlabel("Shot Count", fontsize=24)

    fig.tight_layout()
    save_fig(fig, "a2_learning_curves.png", output_dir)



# ══════════════════════════════════════════════════════════════════════════════
#  A5 — Per-label aggregated analysis (heatmap only)
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a5(a1: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Per-label aggregated analysis, exported as a table (not a heatmap).

    For every (model x label) the table reports the mean F1 together with the
    standard deviation of that F1 measured ACROSS the experimental conditions
    (prompts, shot counts, and sets) — i.e. the run-to-run variability of the
    label's score. It also adds, per model, an overall Macro F1 column computed
    across all five labels (mean of the per-label mean-F1 values) with its own
    across-condition standard deviation.

    Two CSVs are written:
      * a5_per_label_table.csv  — long format (one row per model x label) plus
                                  the overall 'Macro F1 (all labels)' rows.
      * a5_per_label_table_wide.csv — wide, publication-ready layout with
                                  'mean ± std' cells, one row per model.
    """
    print("\n-- A5: Per-label aggregated table --")
    models_p = [m for m in MODELS if m in a1["model"].unique()]

    # ── Long-format records ───────────────────────────────────────────────────
    long_rows = []
    # macro_per_config[model] -> per-config macro F1 over the 5 labels,
    # used to compute the overall mean and its across-condition std.
    for model in models_p:
        sub = a1[a1["model"] == model]

        # Collect the per-config (row-wise) per-label F1 matrix to derive an
        # honest macro-F1-per-config series for the overall column.
        label_cols = [f"{lbl}__f1" for lbl in VALID_LABELS if f"{lbl}__f1" in sub.columns]

        for label in VALID_LABELS:
            f1c = f"{label}__f1"
            pc  = f"{label}__precision"
            rc  = f"{label}__recall"
            if f1c not in sub.columns:
                continue
            long_rows.append({
                "model":          disp(model),
                "label":          LABEL_EN.get(label, label),
                "mean_f1":        sub[f1c].mean(),
                "std_f1":         sub[f1c].std(ddof=1),
                "mean_precision": sub[pc].mean() if pc in sub.columns else np.nan,
                "std_precision":  sub[pc].std(ddof=1) if pc in sub.columns else np.nan,
                "mean_recall":    sub[rc].mean() if rc in sub.columns else np.nan,
                "std_recall":     sub[rc].std(ddof=1) if rc in sub.columns else np.nan,
                "n_conditions":   int(sub[f1c].notna().sum()),
            })

        # Overall Macro F1 (all labels) per model:
        # macro F1 of each configuration = mean across the 5 per-label F1 values,
        # then take mean and std of that series across all conditions.
        if label_cols:
            macro_per_config = sub[label_cols].mean(axis=1)
            long_rows.append({
                "model":          disp(model),
                "label":          "Macro F1 (all labels)",
                "mean_f1":        macro_per_config.mean(),
                "std_f1":         macro_per_config.std(ddof=1),
                "mean_precision": np.nan, "std_precision": np.nan,
                "mean_recall":    np.nan, "std_recall":    np.nan,
                "n_conditions":   int(macro_per_config.notna().sum()),
            })

    result = pd.DataFrame(long_rows)
    result.to_csv(output_dir / "a5_per_label_table.csv", index=False)
    print(f"  Saved a5_per_label_table.csv ({len(result)} rows)")

    # ── Wide, publication-ready layout: 'mean ± std' cells ────────────────────
    def cell(mean, std):
        if pd.isna(mean):
            return ""
        if pd.isna(std):
            return f"{mean:.3f}"
        return f"{mean:.3f} ± {std:.3f}"

    col_order = [LABEL_EN.get(l, l) for l in VALID_LABELS] + ["Macro F1 (all labels)"]
    wide_rows = []
    for model in models_p:
        dmodel = disp(model)
        sub = result[result["model"] == dmodel]
        row = {"Model": dmodel}
        for col in col_order:
            cell_df = sub[sub["label"] == col]
            if cell_df.empty:
                row[col] = ""
            else:
                r0 = cell_df.iloc[0]
                row[col] = cell(r0["mean_f1"], r0["std_f1"])
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows, columns=["Model"] + col_order)
    wide.to_csv(output_dir / "a5_per_label_table_wide.csv", index=False)
    print(f"  Saved a5_per_label_table_wide.csv  (std = variability across "
          f"prompts, shot counts, and sets)")

    return result



def _saturate(hex_color: str, factor: float = 0.78) -> str:
    """Return a more saturated / darker variant of a hex colour for accents."""
    import matplotlib.colors as mcolors
    import colorsys
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, sat = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, l * factor)
    sat = min(1.0, sat * 1.25)
    r, g, b = colorsys.hls_to_rgb(h, l, sat)
    return mcolors.to_hex((r, g, b))


# ══════════════════════════════════════════════════════════════════════════════
#  A6 — Per-label majority voting
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a6(df: pd.DataFrame, a1: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Per-label majority vote: label included in MV pred if >50% of sets predict it.
    Tie-breaking: exclude (conservative).
    Compare MV macro F1 vs mean individual set macro F1 per model.
    """
    print("\n-- A6: Per-label majority voting --")
    mv_df = per_label_majority_vote(df)

    mv_rows = []
    for (model, prompt, sc), g in mv_df.groupby(["model", "prompt", "shot_count"]):
        m = multilabel_metrics(g)
        if m:
            mv_rows.append({"model": model, "prompt": prompt,
                             "shot_count": sc, "f1_macro_mv": m["f1_macro"],
                             "exact_match_mv": m["exact_match"]})
    mv_metrics = pd.DataFrame(mv_rows)

    ind_agg = (a1.groupby(["model", "prompt", "shot_count"])["f1_macro"]
                 .agg(f1_mean="mean", f1_std="std").reset_index())
    merged = mv_metrics.merge(ind_agg, on=["model", "prompt", "shot_count"])
    merged.to_csv(output_dir / "a6_majority_vote_metrics.csv", index=False)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    models_p = [m for m in MODELS if m in merged["model"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), sharey=True)
    axes = axes.flatten()

    for ax, model in zip(axes, models_p):
        color = MODEL_COLORS.get(model, "#666")
        box_data, positions, mv_points = [], [], []
        for k, sc in enumerate(SHOT_COUNTS):
            ind_vals = a1[(a1["model"] == model) &
                          (a1["shot_count"] == sc)]["f1_macro"].dropna().values
            if len(ind_vals) == 0:
                continue
            box_data.append(ind_vals)
            positions.append(k)
            mv_sc = merged[(merged["model"] == model) &
                           (merged["shot_count"] == sc)]["f1_macro_mv"]
            mv_points.append((k, mv_sc.mean() if len(mv_sc) else np.nan))

        bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#333333", lw=1.8),
                        whiskerprops=dict(color="#666666", lw=1.2),
                        capprops=dict(color="#666666", lw=1.2),
                        boxprops=dict(edgecolor="#888888", lw=1.0))
        for patch in bp["boxes"]:
            patch.set_facecolor(color); patch.set_alpha(0.30)
        mv_color = _saturate(color)
        for (k, mv) in mv_points:
            ax.scatter([k], [mv], color=mv_color, edgecolor="white",
                       linewidth=1.2, s=90, zorder=5, marker="D")

        ax.set_title(f"{disp(model)}", fontsize=26)
        ax.set_xticks(range(len(SHOT_COUNTS)))
        ax.set_xticklabels([f"{'0' if s==0 else str(s)}-shot{'s' if s>0 else ''}"
                            for s in SHOT_COUNTS], rotation=15, fontsize=22)
        ax.tick_params(axis="y", labelsize=22)
        ax.set_ylabel("Macro F1", fontsize=24); ax.set_ylim(0.1, 1.05)
        legend_handles = [
            Patch(facecolor=color, alpha=0.30, edgecolor="#888888",
                  label="Individual sets (distribution)"),
            Line2D([0], [0], marker="D", color="w",
                   markerfacecolor=_saturate(color), markeredgecolor="white",
                   markersize=11, label="Majority vote"),
        ]
        ax.legend(handles=legend_handles, fontsize=18)

    for ax in axes[len(models_p):]: ax.set_visible(False)
    fig.tight_layout()
    save_fig(fig, "a6_majority_voting.png", output_dir)

    _variance_benefit_figure(a1, merged, output_dir,
                             fname="a6_variance_vs_benefit.png")
    return mv_metrics


def _variance_benefit_figure(a1: pd.DataFrame, merged: pd.DataFrame,
                             output_dir: Path, fname: str) -> None:
    """
    Scatter of across-set variance (x) vs majority-voting benefit (y),
    one point per (model x shot count). A positive trend indicates that
    models with higher run-to-run variance gain more from ensemble voting.
    """
    rows = []
    for model in [m for m in MODELS if m in merged["model"].unique()]:
        for sc in SHOT_COUNTS:
            ind_vals = a1[(a1["model"] == model) &
                          (a1["shot_count"] == sc)]["f1_macro"].dropna().values
            if len(ind_vals) < 2:
                continue
            mv_sc = merged[(merged["model"] == model) &
                           (merged["shot_count"] == sc)]["f1_macro_mv"]
            if len(mv_sc) == 0:
                continue
            rows.append({"model": model, "shot_count": sc,
                         "variance": float(np.std(ind_vals)),
                         "benefit": float(mv_sc.mean() - np.mean(ind_vals))})
    vb = pd.DataFrame(rows)
    if vb.empty:
        print("  [variance-vs-benefit] no data — skipping"); return
    vb.to_csv(output_dir / fname.replace(".png", ".csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 9))
    for model in [m for m in MODELS if m in vb["model"].unique()]:
        sub = vb[vb["model"] == model]
        ax.scatter(sub["variance"], sub["benefit"],
                   color=MODEL_COLORS.get(model, "#666"), s=320, alpha=0.85,
                   edgecolor="black", linewidth=1.5, label=disp(model), zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(f"{int(r['shot_count'])}",
                        (r["variance"], r["benefit"]),
                        textcoords="offset points", xytext=(0, 14),
                        ha="center", fontsize=15, color="#222")

    if len(vb) >= 2 and vb["variance"].nunique() >= 2:
        z  = np.polyfit(vb["variance"], vb["benefit"], 1)
        xs = np.linspace(vb["variance"].min(), vb["variance"].max(), 100)
        r  = np.corrcoef(vb["variance"], vb["benefit"])[0, 1]
        ax.plot(xs, z[0] * xs + z[1], ls="--", color="gray", lw=2.5,
                label=f"trend (slope={z[0]:.2f}, r={r:.2f})", zorder=2)

    ax.axhline(0, color="black", lw=1.2, alpha=0.5)
    ax.set_xlabel("Across-set variance (std of individual-set Macro F1)", fontsize=23)
    ax.set_ylabel("Majority-voting benefit\n(MV F1 - mean individual F1)", fontsize=23)
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_fig(fig, fname, output_dir)



def export_always_wrong(df: pd.DataFrame, output_dir: Path) -> None:
    """Export full log entries for reports misclassified in 100% of configurations."""
    print("\n-- EXPORT: Reports wrong 100% of the time --")
    total_per  = df.groupby("report_id").size()
    errors_per = df[df["error_type"] != "correct"].groupby("report_id").size()
    ids_100 = [rid for rid in total_per.index
               if errors_per.get(rid, 0) == total_per[rid]]
    print(f"  Found {len(ids_100)} reports wrong in 100% of configurations")
    result = []
    for rid in sorted(ids_100):
        sub = df[df["report_id"] == rid]
        result.append({
            "report_id":     int(rid),
            "true_label":    sub["true_label"].iloc[0],
            "total_configs": int(total_per[rid]),
            "error_types_distribution": sub["error_type"].value_counts().to_dict(),
            "entries": [
                {
                    "model":            row["model"],
                    "model_display":    disp(row["model"]),
                    "prompt":           row["prompt"],
                    "prompt_display":   PROMPT_DISPLAY.get(row["prompt"], row["prompt"]),
                    "shot_count":       int(row["shot_count"]),
                    "set_id":           row["set_id"],
                    "predicted_label":  row["predicted_label"],
                    "error_type":       row["error_type"],
                    "model_confidence": row["model_confidence"],
                    "model_motivation": row["model_motivation"],
                    "label_evidence":   json.loads(row["label_evidence"])
                                        if "label_evidence" in row and row["label_evidence"] else {},
                }
                for _, row in sub.iterrows()
            ],
        })
    path = output_dir / "always_wrong_multiclass.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path.name}  ({len(result)} reports)")


def export_error_type_samples(df: pd.DataFrame, output_dir: Path,
                               n_samples: int = 5) -> None:
    """Export n_samples random entries per LP error type (4 types x 5 = 20 entries)."""
    print("\n-- EXPORT: Error type samples (5 per type) --")
    errors = df[df["error_type"] != "correct"].copy()
    result = {}
    for et in ERROR_TYPES:
        subset  = errors[errors["error_type"] == et]
        n       = min(n_samples, len(subset))
        sampled = subset.sample(n=n, random_state=RANDOM_SEED) if n > 0 else subset
        result[et] = [
            {
                "report_id":        int(row["report_id"]),
                "true_label":       row["true_label"],
                "predicted_label":  row["predicted_label"],
                "error_type":       row["error_type"],
                "model":            row["model"],
                "model_display":    disp(row["model"]),
                "prompt":           row["prompt"],
                "prompt_display":   PROMPT_DISPLAY.get(row["prompt"], row["prompt"]),
                "shot_count":       int(row["shot_count"]),
                "set_id":           row["set_id"],
                "model_confidence": row["model_confidence"],
                "model_motivation": row["model_motivation"],
                "label_evidence":   json.loads(row["label_evidence"])
                                    if "label_evidence" in row and row["label_evidence"] else {},
            }
            for _, row in sampled.iterrows()
        ]
        print(f"  {ERROR_LABELS[et]}: {len(subset):,} total -> sampled {len(result[et])}")
    path = output_dir / "error_samples_by_type_multiclass.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path.name}  (4 types x {n_samples} = {sum(len(v) for v in result.values())} entries)")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiclass task - analysis pipeline")
    p.add_argument("--root_dir",   default="./results_multiclass",
                   help="Root directory containing multiclass task logs")
    p.add_argument("--output_dir", default="./figures/multiclass",
                   help="Output directory for plots and JSON exports")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    root_dir   = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MULTICLASS TASK - Analysis Pipeline")
    print(f"  Root   : {root_dir}")
    print(f"  Output : {output_dir}")
    print("=" * 60)

    df = load_all_logs(root_dir)
    if df.empty:
        print("ERROR: no data loaded - check --root_dir path.")
        return

    a1_df = analysis_a1(df, output_dir)
    analysis_a2(a1_df, output_dir)
    analysis_a5(a1_df, output_dir)
    analysis_a6(df, a1_df, output_dir)

    print("\n" + "-"*60)
    print("  EXPORT")
    print("-"*60)
    export_always_wrong(df, output_dir)
    export_error_type_samples(df, output_dir)

    print("\n" + "="*60)
    print("  COMPLETE")
    print(f"  Outputs: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()