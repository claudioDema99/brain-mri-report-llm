#!/usr/bin/env python3
"""
binary_analysis.py
──────────────────────────────────────────────────────────────────────────────
Complete analysis pipeline for the BINARY classification task logs.
Produces publication-quality PNG figures and structured CSV/JSON outputs.

Analyses:
  A1  Metrics per configuration (model × prompt × shot_count × set)
  A2  Learning curves: Macro F1 vs shot count (mean ± std across sets)
  A3  Prompt comparison at fixed model and shot count
  A4  Model comparison at fixed prompt and shot count
  A5  Majority voting vs mean individual set F1
  B3  Double confidence filter (human_conf = 5 AND model_conf = 5)
  C1  False Positive vs False Negative rate per model/configuration
  C3  Systematically wrong report IDs
  C4  Error overlap: 0-shot vs few-shot
  D1  Stratified reasoning sample (FP / FN motivations for manual review)
  E1  Kruskal-Wallis omnibus + pairwise Mann-Whitney U + FDR correction

Usage:
    python analysis/binary_analysis.py --root_dir ./results_binary --output_dir ./figures/binary
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
POS_LABEL   = "positivo"
NEG_LABEL   = "negativo"

N_MOTIVATION_SAMPLES = 5  # random motivations extracted per error type in D1

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

MODEL_COLORS  = {"gpt": "#DD8452", "mistral": "#55A868", "gemma": "#4C72B0", "qwen": "#C44E52"}
SHOT_COLORS   = {0: "#5B9BD5", 5: "#ED7D31", 10: "#70AD47", 20: "#C44E52"}
PROMPT_COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52"}

def disp(model: str) -> str:
    """Return the publication display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model, model.upper())


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_logs(root_dir: Path) -> pd.DataFrame:
    """
    Walk the BINARY directory tree and load every successful log entry.
    Structure: results_binary/{model}/prompt_{X}/{N}-shot(s)/set_{K}/logs/logs_binary_*.json
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
                    for log_file in sorted(logs_dir.glob("logs_binary_*.json")):
                        try:
                            with open(log_file, encoding="utf-8") as f:
                                entries = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            continue
                        for e in entries:
                            if e.get("status") != "success":
                                continue
                            true_lbl = (e.get("true_label") or "").strip().lower()
                            pred_lbl = (e.get("predicted_label") or "").strip().lower()
                            if not true_lbl or not pred_lbl:
                                continue
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
                                "true_label":       true_lbl,
                                "predicted_label":  pred_lbl,
                                "human_confidence": human_conf,
                                "model_confidence": e.get("model_confidence"),
                                "model_motivation": (e.get("model_motivation") or "").strip(),
                                "correct":          true_lbl == pred_lbl,
                            })
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} entries")
    print(f"  Models     : {sorted(df['model'].unique())}")
    print(f"  Prompts    : {sorted(df['prompt'].unique())}")
    print(f"  Shot counts: {sorted(df['shot_count'].unique())}")
    print(f"  Unique IDs : {df['report_id'].nunique()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def metrics_from_df(sub: pd.DataFrame,
                    true_col: str = "true_label",
                    pred_col: str = "predicted_label") -> dict:
    """Compute standard binary classification metrics from a sub-DataFrame."""
    if sub.empty:
        return {}
    t = sub[true_col].tolist()
    p = sub[pred_col].tolist()
    return {
        "n":             len(t),
        "accuracy":      accuracy_score(t, p),
        "f1_macro":      f1_score(t, p, average="macro",   zero_division=0),
        "f1_positivo":   f1_score(t, p, pos_label=POS_LABEL, average="binary", zero_division=0),
        "f1_negativo":   f1_score(t, p, pos_label=NEG_LABEL, average="binary", zero_division=0),
        "precision_pos": precision_score(t, p, pos_label=POS_LABEL, average="binary", zero_division=0),
        "recall_pos":    recall_score(t, p, pos_label=POS_LABEL, average="binary", zero_division=0),
        "precision_neg": precision_score(t, p, pos_label=NEG_LABEL, average="binary", zero_division=0),
        "recall_neg":    recall_score(t, p, pos_label=NEG_LABEL, average="binary", zero_division=0),
    }


def majority_vote(preds: list, tie_label: str = NEG_LABEL) -> str:
    """Plurality winner; ties broken by tie_label (conservative for medical context)."""
    counts    = Counter(preds)
    max_count = max(counts.values())
    winners   = [k for k, v in counts.items() if v == max_count]
    return winners[0] if len(winners) == 1 else tie_label


def compute_mv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute majority vote prediction per (model, prompt, shot_count, report_id)
    across all sets. Returns one row per combination.
    """
    rows = []
    for (model, prompt, sc), g in df.groupby(["model", "prompt", "shot_count"]):
        for rid, rg in g.groupby("report_id"):
            mv = majority_vote(rg["predicted_label"].tolist())
            rows.append({
                "model":             model,
                "prompt":            prompt,
                "shot_count":        sc,
                "report_id":         rid,
                "true_label":        rg["true_label"].iloc[0],
                "predicted_label":   mv,
                "n_sets":            len(rg),
                "correct":           mv == rg["true_label"].iloc[0],
            })
    return pd.DataFrame(rows)


def save_fig(fig: plt.Figure, name: str, output_dir: Path) -> None:
    path = output_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  A1 — Metrics per configuration
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a1(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Compute accuracy, precision, recall, F1 for every
    (model x prompt x shot_count x set_id) combination.
    Saves CSV + heatmap of macro F1 averaged over prompts and sets.
    """
    print("\n-- A1: Metrics per configuration --")
    rows = []
    for (model, prompt, sc, sid), g in df.groupby(
        ["model", "prompt", "shot_count", "set_id"]
    ):
        m = metrics_from_df(g)
        if m:
            rows.append({"model": model, "prompt": prompt,
                         "shot_count": sc, "set_id": sid, **m})
    a1 = pd.DataFrame(rows)
    a1.to_csv(output_dir / "a1_metrics_per_config.csv", index=False)
    print(f"  Saved a1_metrics_per_config.csv ({len(a1)} rows)")

    # Heatmap: mean macro F1 per (model x shot_count), averaged over prompts+sets
    pivot = (a1.groupby(["model", "shot_count"])["f1_macro"]
               .mean()
               .unstack("shot_count")
               .reindex(index=[m for m in MODELS if m in a1["model"].unique()])
               .reindex(columns=[s for s in SHOT_COUNTS if s in a1["shot_count"].unique()]))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Macro F1"},
    )
    ax.set_title("A1 - Mean Macro F1  (averaged over prompts and sets)")
    ax.set_xlabel("Shot Count")
    ax.set_ylabel("Model")
    ax.set_yticklabels([disp(m) for m in pivot.index], rotation=0)
    fig.tight_layout()
    save_fig(fig, "a1_f1_heatmap.png", output_dir)
    return a1


# ══════════════════════════════════════════════════════════════════════════════
#  A2 — Learning curves
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a2(a1: pd.DataFrame, output_dir: Path) -> None:
    """
    One subplot per (model x prompt): macro F1 vs shot count,
    mean +/- std across sets. Shaded band = +/- 1 std.
    """
    print("\n-- A2: Learning curves --")
    models_present  = [m for m in MODELS  if m in a1["model"].unique()]
    prompts_present = [p for p in PROMPTS if p in a1["prompt"].unique()]
    nrows, ncols    = len(models_present), len(prompts_present)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.8 * nrows),
                              sharey=True, sharex=True)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for i, model in enumerate(models_present):
        for j, prompt in enumerate(prompts_present):
            ax  = axes[i][j]
            sub = a1[(a1["model"] == model) & (a1["prompt"] == prompt)]
            agg = (sub.groupby("shot_count")["f1_macro"]
                      .agg(mean="mean", std="std")
                      .reset_index()
                      .fillna({"std": 0}))
            color = MODEL_COLORS.get(model, "#333333")
            ax.plot(agg["shot_count"], agg["mean"],
                    color=color, marker="o", lw=3, ms=11, zorder=3)
            ax.fill_between(agg["shot_count"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            color=color, alpha=0.15)
            ax.set_xticks(SHOT_COUNTS)
            ax.set_ylim(0.4, 0.9)
            ax.tick_params(axis="both", labelsize=22)
            ax.set_title(f"{disp(model)} — Prompt B{PROMPT_DISPLAY.get(prompt, prompt)}",
                         fontsize=26)
            if j == 0:
                ax.set_ylabel("Macro F1", fontsize=24)
            if i == nrows - 1:
                ax.set_xlabel("Shot Count", fontsize=24)

    fig.tight_layout()
    save_fig(fig, "a2_learning_curves.png", output_dir)


def _saturate(hex_color: str, factor: float = 0.78) -> str:
    """Return a more saturated / darker variant of a hex colour for accents."""
    import matplotlib.colors as mcolors
    import colorsys
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, sat = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, l * factor)          # darker
    sat = min(1.0, sat * 1.25)        # more saturated
    r, g, b = colorsys.hls_to_rgb(h, l, sat)
    return mcolors.to_hex((r, g, b))


# ══════════════════════════════════════════════════════════════════════════════
#  A5 — Majority voting
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a5(df: pd.DataFrame, a1: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    For each (model x prompt x shot_count): compute majority vote prediction
    across all sets. Compare majority vote macro F1 vs mean individual set F1.
    Ties broken by NEG_LABEL (conservative for clinical context).
    """
    print("\n-- A5: Majority voting --")
    mv_df = compute_mv_df(df)

    mv_rows = []
    for (model, prompt, sc), g in mv_df.groupby(["model", "prompt", "shot_count"]):
        m = metrics_from_df(g)
        if m:
            mv_rows.append({"model": model, "prompt": prompt,
                             "shot_count": sc, "f1_macro_mv": m["f1_macro"]})
    mv_metrics = pd.DataFrame(mv_rows)

    ind_agg = (a1.groupby(["model", "prompt", "shot_count"])["f1_macro"]
                 .agg(f1_mean="mean", f1_std="std").reset_index())
    merged = mv_metrics.merge(ind_agg, on=["model", "prompt", "shot_count"])
    merged.to_csv(output_dir / "a5_majority_vote_metrics.csv", index=False)

    # ── Boxplots: distribution of individual-set macro F1 across sets+prompts,
    #    with the majority-vote macro F1 overlaid as a marker, per shot count ──
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    models_present = [m for m in MODELS if m in merged["model"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), sharey=True)
    axes = axes.flatten()

    for ax, model in zip(axes, models_present):
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
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="#333333", lw=1.8),
                        whiskerprops=dict(color="#666666", lw=1.2),
                        capprops=dict(color="#666666", lw=1.2),
                        boxprops=dict(edgecolor="#888888", lw=1.0),
                        flierprops=dict(marker="o", ms=4, alpha=0.25,
                                        markerfacecolor=color,
                                        markeredgecolor="none"))
        for patch in bp["boxes"]:
            patch.set_facecolor(color); patch.set_alpha(0.30)
        mv_color = _saturate(color)
        for (k, mv) in mv_points:
            ax.scatter([k], [mv], color=mv_color, edgecolor="white",
                       linewidth=1.2, s=90, zorder=5, marker="D")

        ax.set_title(f"{disp(model)}", fontsize=26)
        ax.set_xticks(range(len(SHOT_COUNTS)))
        ax.set_xticklabels([f"{'0' if s == 0 else str(s)}-shot{'s' if s > 0 else ''}"
                            for s in SHOT_COUNTS], rotation=15, fontsize=22)
        ax.tick_params(axis="y", labelsize=22)
        ax.set_ylabel("Macro F1", fontsize=24)
        ax.set_ylim(0.3, 1.05)
        legend_handles = [
            Patch(facecolor=color, alpha=0.30, edgecolor="#888888",
                  label="Individual sets (distribution)"),
            Line2D([0], [0], marker="D", color="w",
                   markerfacecolor=_saturate(color), markeredgecolor="white",
                   markersize=11, label="Majority vote"),
        ]
        ax.legend(handles=legend_handles, fontsize=18)

    for ax in axes[len(models_present):]:
        ax.set_visible(False)

    fig.tight_layout()
    save_fig(fig, "a5_majority_voting.png", output_dir)

    # ── Variance vs Benefit: does higher across-set variance translate into a
    #    larger gain from majority voting? (one point per model x shot count) ──
    _variance_benefit_figure(a1, merged, output_dir,
                             fname="a5_variance_vs_benefit.png")
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



# ══════════════════════════════════════════════════════════════════════════════
#  C3 — Systematically wrong reports
# ══════════════════════════════════════════════════════════════════════════════

def analysis_c3(df: pd.DataFrame, output_dir: Path, top_n: int = 30) -> pd.DataFrame:
    """
    For each report_id: fraction of all (model x prompt x shot x set)
    configurations where it was misclassified.
    Reports wrong in >= 50% of configurations are flagged as systematic errors.
    """
    print("\n-- C3: Systematically wrong reports --")
    total_per  = df.groupby("report_id").size().rename("total_configs")
    errors_per = df[~df["correct"]].groupby("report_id").size().rename("n_errors")

    summary = (pd.concat([total_per, errors_per], axis=1)
                 .fillna({"n_errors": 0})
                 .astype({"n_errors": int})
                 .assign(error_rate=lambda x: x["n_errors"] / x["total_configs"])
                 .sort_values("error_rate", ascending=False)
                 .reset_index())

    systematic = summary[summary["error_rate"] >= 0.5]
    print(f"  Reports wrong in >=50% of configs: {len(systematic)}")
    print(f"  Top 5:")
    print(systematic.head(5)[["report_id", "n_errors", "total_configs", "error_rate"]]
          .to_string(index=False))
    summary.to_csv(output_dir / "c3_systematic_errors.csv", index=False)

    top  = summary.head(top_n)
    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = ["#E74C3C" if r >= 0.5 else "#E67E22" if r >= 0.3 else "#F1C40F"
                  for r in top["error_rate"]]
    ax.bar(range(len(top)), top["error_rate"], color=bar_colors, alpha=0.85)
    ax.axhline(0.5, color="red",    ls="--", lw=1.5, label="50% threshold (systematic)")
    ax.axhline(0.3, color="orange", ls=":",  lw=1.2, label="30% threshold")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([str(rid) for rid in top["report_id"]], rotation=90, fontsize=8)
    ax.set_xlabel("Report ID")
    ax.set_ylabel("Error Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"C3 - Top {top_n} Most Consistently Misclassified Reports\n"
                 "(error rate = fraction of model x prompt x shot x set configs where report was wrong)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "c3_systematic_errors.png", output_dir)
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  C4 — Error overlap: 0-shot vs few-shot
# ══════════════════════════════════════════════════════════════════════════════

def analysis_c4(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    For each (model x prompt): compare reports wrong at 0-shot
    vs wrong at 5/10/20-shot (majority vote across sets).
    Four categories: only-0-shot wrong, only-few-shot wrong, both wrong, both correct.
    """
    print("\n-- C4: Error overlap - 0-shot vs few-shot --")
    mv_all = compute_mv_df(df)
    rows   = []

    for (model, prompt), g in df.groupby(["model", "prompt"]):
        zero = g[g["shot_count"] == 0]
        if zero.empty:
            continue
        zero_errors = set(zero[~zero["correct"]]["report_id"])

        for sc in [5, 10, 20]:
            mv_sub = mv_all[(mv_all["model"] == model) &
                            (mv_all["prompt"] == prompt) &
                            (mv_all["shot_count"] == sc)]
            if mv_sub.empty:
                continue
            few_errors  = set(mv_sub[~mv_sub["correct"]]["report_id"])
            all_reports = set(zero["report_id"]) & set(mv_sub["report_id"])
            ze = zero_errors & all_reports
            fe = few_errors  & all_reports
            rows.append({
                "model": model, "prompt": prompt, "shot_count": sc,
                "n_reports":       len(all_reports),
                "only_zero_wrong": len(ze - fe),
                "only_few_wrong":  len(fe - ze),
                "both_wrong":      len(ze & fe),
                "both_correct":    len(all_reports - ze - fe),
            })

    c4 = pd.DataFrame(rows)
    if c4.empty:
        print("  No data for C4 - skipping plot")
        return c4
    c4.to_csv(output_dir / "c4_error_overlap.csv", index=False)

    agg       = (c4.groupby(["model", "shot_count"])
                   [["only_zero_wrong", "only_few_wrong", "both_wrong", "both_correct"]]
                   .mean().reset_index())
    shot_opts = [s for s in [5, 10, 20] if s in agg["shot_count"].unique()]
    if not shot_opts:
        return c4

    cat_colors = {"both_correct": "#2ECC71", "only_zero_wrong": "#E74C3C",
                  "only_few_wrong": "#3498DB", "both_wrong": "#8E44AD"}
    cat_labels = {"both_correct": "Both correct", "only_zero_wrong": "0-shot wrong only",
                  "only_few_wrong": "Few-shot wrong only", "both_wrong": "Both wrong"}

    fig, axes = plt.subplots(1, len(shot_opts),
                              figsize=(5 * len(shot_opts), 5), sharey=True)
    if len(shot_opts) == 1:
        axes = [axes]

    for ax, shot in zip(axes, shot_opts):
        sub     = agg[agg["shot_count"] == shot]
        present = [m for m in MODELS if m in sub["model"].values]
        x       = np.arange(len(present))
        bottom  = np.zeros(len(present))
        for cat in ["both_correct", "only_zero_wrong", "only_few_wrong", "both_wrong"]:
            vals = [sub[sub["model"] == m][cat].values[0]
                    if m in sub["model"].values else 0 for m in present]
            ax.bar(x, vals, bottom=bottom,
                   color=cat_colors[cat], alpha=0.85, label=cat_labels[cat])
            bottom += np.array(vals, dtype=float)
        ax.set_xticks(x)
        ax.set_xticklabels([disp(m) for m in present])
        ax.set_title(f"{shot}-shots")
        ax.set_ylabel("Mean report count (over prompts)")
        if shot == shot_opts[0]:
            ax.legend(fontsize=9)

    fig.suptitle("C4 - Error Overlap: 0-shot vs Few-shot\n"
                 "(0-shot = set_1; few-shot = majority vote; averaged over prompts)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "c4_error_overlap.png", output_dir)
    return c4


# ══════════════════════════════════════════════════════════════════════════════
#  D1 — Qualitative reasoning sample
# ══════════════════════════════════════════════════════════════════════════════

def analysis_d1(df: pd.DataFrame, output_dir: Path,
                n_samples: int = N_MOTIVATION_SAMPLES) -> None:
    """
    Stratified sampling: extract n_samples random entries with non-empty
    model_motivation for FP and FN separately.
    Saved to d1_reasoning_samples.json for manual qualitative review.
    """
    print("\n-- D1: Qualitative reasoning sample --")
    errors = df[~df["correct"]].copy()
    fp     = errors[(errors["true_label"] == NEG_LABEL) & (errors["predicted_label"] == POS_LABEL)]
    fn     = errors[(errors["true_label"] == POS_LABEL) & (errors["predicted_label"] == NEG_LABEL)]

    output = {}
    for label, subset in [("false_positive", fp), ("false_negative", fn)]:
        has_motiv = subset[subset["model_motivation"].str.len() > 0]
        n         = min(n_samples, len(has_motiv))
        sampled   = has_motiv.sample(n=n, random_state=RANDOM_SEED) if n > 0 else has_motiv
        output[label] = [
            {
                "report_id":        int(row["report_id"]),
                "model":            row["model"],
                "prompt":           row["prompt"],
                "shot_count":       int(row["shot_count"]),
                "set_id":           row["set_id"],
                "true_label":       row["true_label"],
                "predicted_label":  row["predicted_label"],
                "human_confidence": row["human_confidence"],
                "model_confidence": row["model_confidence"],
                "model_motivation": row["model_motivation"],
            }
            for _, row in sampled.iterrows()
        ]
        print(f"  {label}: {len(subset):,} total -> sampled {len(output[label])}")

    path = output_dir / "d1_reasoning_samples.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path.name}")



# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT — Reports wrong 100% of the time
# ══════════════════════════════════════════════════════════════════════════════

def export_always_wrong(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export all log entries for reports misclassified in 100% of configurations.
    Groups entries by report_id. Saved to always_wrong_binary.json.
    """
    print("\n-- EXPORT: Reports wrong 100% of the time --")
    total_per  = df.groupby("report_id").size()
    errors_per = df[~df["correct"]].groupby("report_id").size()
    always_wrong_ids = [rid for rid in total_per.index
                        if errors_per.get(rid, 0) == total_per[rid]]
    print(f"  Found {len(always_wrong_ids)} reports wrong in 100% of configurations")

    result = []
    for rid in sorted(always_wrong_ids):
        entries = df[df["report_id"] == rid].copy()
        result.append({
            "report_id":     int(rid),
            "true_label":    entries["true_label"].iloc[0],
            "total_configs": int(total_per[rid]),
            "error_types_distribution": {"incorrect": int(total_per[rid])},
            "entries": [
                {
                    "model":            row["model"],
                    "model_display":    disp(row["model"]),
                    "prompt":           row["prompt"],
                    "prompt_display":   PROMPT_DISPLAY.get(row["prompt"], row["prompt"]),
                    "shot_count":       int(row["shot_count"]),
                    "set_id":           row["set_id"],
                    "predicted_label":  row["predicted_label"],
                    "human_confidence": row["human_confidence"],
                    "model_confidence": row["model_confidence"],
                    "model_motivation": row["model_motivation"],
                }
                for _, row in entries.iterrows()
            ],
        })

    path = output_dir / "always_wrong_binary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path.name}  ({len(result)} reports, "
          f"{sum(len(r['entries']) for r in result)} total entries)")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary task - complete analysis pipeline")
    p.add_argument("--root_dir",   default="./results_binary",
                   help="Root directory containing binary task logs")
    p.add_argument("--output_dir", default="./figures/binary",
                   help="Output directory for plots, CSVs and JSON")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    root_dir   = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BINARY TASK - Complete Analysis Pipeline")
    print(f"  Root   : {root_dir}")
    print(f"  Output : {output_dir}")
    print("=" * 60)

    df = load_all_logs(root_dir)
    if df.empty:
        print("ERROR: no data loaded - check --root_dir path.")
        return

    print("\n" + "-" * 60)
    print("  A. FULL RESULTS (UNFILTERED)")
    print("-" * 60)
    a1_df = analysis_a1(df, output_dir)
    analysis_a2(a1_df, output_dir)
    analysis_a5(df, a1_df, output_dir)

    print("\n" + "-" * 60)
    print("  C. ERROR ANALYSIS")
    print("-" * 60)
    analysis_c3(df, output_dir)
    analysis_c4(df, output_dir)

    print("\n" + "-" * 60)
    print("  D. QUALITATIVE REASONING")
    print("-" * 60)
    analysis_d1(df, output_dir)

    print("\n" + "-"*60)
    print("  EXPORT")
    print("-"*60)
    export_always_wrong(df, output_dir)


if __name__ == "__main__":
    main()