#!/usr/bin/env python3
"""
peft_analysis.py
──────────────────────────────────────────────────────────────────────────────
Complete analysis pipeline for PEFT (QLoRA fine-tuned) task logs.
Handles all 10 LoRA configurations and produces comparison analyses
against zero-shot and few-shot Qwen baselines.

PEFT log structure (confidence=null, model_motivation=null by design):
  results_multiclass/qwen_finetuned/config_{N:02d}/prompt_dataset/finetuned/set_1/logs/logs_multiclass_*.json

Analyses:
  A1  Metrics per config: macro/micro F1, exact match, per-label P/R/F1
  A2  Comparison across 10 LoRA configurations
  A3  Hyperparameter ablation: rank, target_modules, layers, dropout
  A4  PEFT best config vs best zero-shot Qwen (same report IDs)
  A5  PEFT best config vs best few-shot Qwen ensemble (same report IDs)
  B1  LP error taxonomy per configuration
  B2  Per-label confusion matrix for best configuration
  B4  Reports consistently wrong across all 10 configurations
  B5  Error type comparison: PEFT vs zero-shot Qwen
  E   McNemar per-report test + bootstrap CI on macro F1 delta

Usage:
    # PEFT only (A1, A2, A3, B1, B2, B4)
    python analysis/peft_analysis.py --peft_dir ./results_multiclass/qwen_finetuned --output_dir ./figures/peft

    # PEFT + Qwen baseline comparisons (all analyses)
    python analysis/peft_analysis.py --peft_dir ./results_multiclass/qwen_finetuned --output_dir ./figures/peft \
                            --multiclass_dir ./results_multiclass
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

RANDOM_SEED   = 42
N_BOOTSTRAP   = 10_000
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

ERROR_TYPES = ["fn_partial_only", "fp_partial_only", "mixed", "complete_error"]
ERROR_LABELS = {
    "fn_partial_only": "FN-partial only",
    "fp_partial_only": "FP-partial only",
    "mixed":           "Mixed (FN+FP)",
    "complete_error":  "Complete error",
}

# ── LoRA configuration metadata ───────────────────────────────────────────────
LORA_CONFIGS = {
    1:  {"short": "QV r=8",           "modules": "q,v",              "r": 8,  "lr": "1e-4", "layers": "all",   "dropout": 0.05},
    2:  {"short": "QV r=16",          "modules": "q,v",              "r": 16, "lr": "1e-4", "layers": "all",   "dropout": 0.05},
    3:  {"short": "QKVO r=16",        "modules": "q,k,v,o",         "r": 16, "lr": "8e-5", "layers": "all",   "dropout": 0.05},
    4:  {"short": "QKVO r=32",        "modules": "q,k,v,o",         "r": 32, "lr": "8e-5", "layers": "all",   "dropout": 0.05},
    5:  {"short": "QKVO r=16 L28",    "modules": "q,k,v,o",         "r": 16, "lr": "8e-5", "layers": "28-39", "dropout": 0.05},
    6:  {"short": "QKVO r=16 L20",    "modules": "q,k,v,o",         "r": 16, "lr": "8e-5", "layers": "20-39", "dropout": 0.05},
    7:  {"short": "QKVO+gate r=16",   "modules": "q,k,v,o,gate",   "r": 16, "lr": "6e-5", "layers": "all",   "dropout": 0.05},
    8:  {"short": "QKVO+FFN d=0.1",  "modules": "q,k,v,o,gate,up,dn", "r": 16, "lr": "5e-5", "layers": "all",   "dropout": 0.10},
    9:  {"short": "QKVO+FFN L28",    "modules": "q,k,v,o,gate,up,dn", "r": 16, "lr": "5e-5", "layers": "28-39", "dropout": 0.05},
    10: {"short": "QKVO r=64",        "modules": "q,k,v,o",         "r": 64, "lr": "5e-5", "layers": "all",   "dropout": 0.05},
}

# ── Hyperparameter ablation groups ─────────────────────────────────────────────
HYPERPARAM_GROUPS = {
    "rank_QV":    {"title": "Rank effect  (QV modules, all layers)",
                   "configs": [1, 2],
                   "label_fn": lambda c: f"r={LORA_CONFIGS[c]['r']}"},
    "rank_QKVO":  {"title": "Rank effect  (QKVO modules, all layers)",
                   "configs": [3, 4, 10],
                   "label_fn": lambda c: f"r={LORA_CONFIGS[c]['r']}"},
    "modules":    {"title": "Target modules effect  (r=16, all layers)",
                   "configs": [2, 3, 7, 8],
                   "label_fn": lambda c: LORA_CONFIGS[c]["modules"].replace(",","\n")},
    "layers":     {"title": "Layers to transform effect  (QKVO r=16)",
                   "configs": [3, 5, 6],
                   "label_fn": lambda c: LORA_CONFIGS[c]["layers"]},
    "dropout":    {"title": "Dropout effect  (QKVO+FFN)",
                   "configs": [8, 9],
                   "label_fn": lambda c: f"d={LORA_CONFIGS[c]['dropout']}"},
}

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

CONFIG_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
CONFIG_COLOR  = {i+1: CONFIG_COLORS[i] for i in range(10)}
ERROR_COLORS  = {
    "fn_partial_only": "#3498DB",
    "fp_partial_only": "#E74C3C",
    "mixed":           "#8E44AD",
    "complete_error":  "#2C3E50",
}
LABEL_COLORS = {
    "Neoformazione":            "#E74C3C",
    "Vascolare":                "#3498DB",
    "Infiammazione o Infezione":"#2ECC71",
    "Neurodegenerativo":        "#9B59B6",
    "Altro":                    "#F39C12",
}

# Model display names (consistent with binary and multiclass scripts)
MODEL_DISPLAY_NAMES = {
    "gpt":     "GPT-OSS",
    "mistral": "Magistral-S",
    "gemma":   "Gemma-4",
    "qwen":    "Qwen3.5-A3B",
}
MODEL_COLORS = {"gpt": "#DD8452", "mistral": "#55A868", "gemma": "#4C72B0", "qwen": "#C44E52"}
PROMPT_DISPLAY = {"A": "1", "B": "2", "C": "3", "D": "4"}

def disp(model: str) -> str:
    """Return the publication display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model, model.upper())



# ══════════════════════════════════════════════════════════════════════════════
#  MULTILABEL UTILITIES  (mirrors multiclass_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def parse_label_set(raw) -> frozenset:
    if not raw:
        return frozenset()
    if isinstance(raw, list):
        return frozenset(s.strip() for s in raw if s.strip() in VALID_LABELS_SET)
    return frozenset(s.strip() for s in str(raw).split(",") if s.strip() in VALID_LABELS_SET)


def multilabel_metrics(sub: pd.DataFrame,
                        true_col: str = "true_set",
                        pred_col: str = "pred_set") -> dict:
    if sub.empty:
        return {}
    n = len(sub)
    exact  = int((sub[true_col] == sub[pred_col]).sum())
    hamming = float(np.mean([
        len(r[true_col] & r[pred_col]) / len(r[true_col] | r[pred_col])
        if (r[true_col] | r[pred_col]) else 1.0
        for _, r in sub.iterrows()
    ]))
    tp = {l: 0 for l in VALID_LABELS}
    fp = {l: 0 for l in VALID_LABELS}
    fn = {l: 0 for l in VALID_LABELS}
    for _, row in sub.iterrows():
        t, p = row[true_col], row[pred_col]
        for l in VALID_LABELS:
            if   l in t and l in p: tp[l] += 1
            elif l in p:            fp[l] += 1
            elif l in t:            fn[l] += 1
    tp_m = sum(tp.values()); fp_m = sum(fp.values()); fn_m = sum(fn.values())
    pr_m = tp_m/(tp_m+fp_m) if (tp_m+fp_m) else 0.0
    re_m = tp_m/(tp_m+fn_m) if (tp_m+fn_m) else 0.0
    f1_m = 2*pr_m*re_m/(pr_m+re_m) if (pr_m+re_m) else 0.0
    per_label = {}
    macro_p = macro_r = macro_f1 = 0.0
    for l in VALID_LABELS:
        p_ = tp[l]/(tp[l]+fp[l]) if (tp[l]+fp[l]) else 0.0
        r_ = tp[l]/(tp[l]+fn[l]) if (tp[l]+fn[l]) else 0.0
        f_ = 2*p_*r_/(p_+r_) if (p_+r_) else 0.0
        per_label[l] = {"precision": p_, "recall": r_, "f1": f_,
                         "support": tp[l]+fn[l], "tp": tp[l], "fp": fp[l], "fn": fn[l]}
        macro_p += p_; macro_r += r_; macro_f1 += f_
    k = len(VALID_LABELS)
    return {
        "n": n, "exact_match": exact/n, "hamming_score": hamming,
        "precision_micro": pr_m, "recall_micro": re_m, "f1_micro": f1_m,
        "precision_macro": macro_p/k, "recall_macro": macro_r/k, "f1_macro": macro_f1/k,
        "per_label": per_label,
    }


def classify_error(true_set: frozenset, pred_set: frozenset) -> str:
    if true_set == pred_set: return "correct"
    fn = true_set - pred_set; fp = pred_set - true_set
    if not (true_set & pred_set): return "complete_error"
    elif fn and fp:               return "mixed"
    elif fn:                      return "fn_partial_only"
    else:                         return "fp_partial_only"


def save_fig(fig: plt.Figure, name: str, output_dir: Path) -> None:
    path = output_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.name}")

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING — PEFT
# ══════════════════════════════════════════════════════════════════════════════

def load_peft_logs(peft_dir: Path) -> pd.DataFrame:
    """
    Load PEFT logs from:
      results_multiclass/qwen_finetuned/config_{N:02d}/prompt_dataset/finetuned/set_1/logs/logs_*.json
    Returns one row per (config_num, report_id).
    """
    records = []
    for config_dir in sorted(peft_dir.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        try:
            config_num = int(config_dir.name.removeprefix("config_"))
        except ValueError:
            continue

        # Walk: prompt_dataset/finetuned/set_1/logs/
        for prompt_dir in sorted(config_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            for mode_dir in sorted(prompt_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                for set_dir in sorted(mode_dir.iterdir()):
                    if not set_dir.is_dir():
                        continue
                    logs_dir = set_dir / "logs"
                    if not logs_dir.exists():
                        continue
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
                            records.append({
                                "config_num":      config_num,
                                "report_id":       e["id"],
                                "true_label":      e.get("true_label", ""),
                                "predicted_label": e.get("predicted_label", ""),
                                "true_set":        true_set,
                                "pred_set":        pred_set,
                                "model_confidence": e.get("model_confidence"),
                                "correct":         true_set == pred_set,
                                "error_type":      classify_error(true_set, pred_set),
                            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    print(f"Loaded PEFT: {len(df):,} entries, "
          f"{df['config_num'].nunique()} configs, "
          f"{df['report_id'].nunique()} unique report IDs")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING — QWEN BASELINE  (from MULTICLASS directory)
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen_logs(multiclass_dir: Path, target_ids: set) -> pd.DataFrame:
    """
    Load Qwen logs from results_multiclass/qwen/, filtered to target_ids (PEFT test set).
    Returns one row per (prompt, shot_count, set_id, report_id).
    """
    qwen_dir = multiclass_dir / "qwen"
    if not qwen_dir.exists():
        print(f"  WARNING: {qwen_dir} not found — skipping baseline comparisons")
        return pd.DataFrame()

    records = []
    for prompt_dir in sorted(qwen_dir.iterdir()):
        if not prompt_dir.is_dir() or not prompt_dir.name.startswith("prompt_"):
            continue
        prompt = prompt_dir.name.removeprefix("prompt_")
        for shot_dir in sorted(prompt_dir.iterdir()):
            if not shot_dir.is_dir():
                continue
            name = shot_dir.name
            if name == "0-shot":      shot_count = 0
            elif name.endswith("-shots"):
                try: shot_count = int(name.removesuffix("-shots"))
                except ValueError: continue
            else: continue
            for set_dir in sorted(shot_dir.iterdir()):
                if not set_dir.is_dir() or not set_dir.name.startswith("set_"):
                    continue
                logs_dir = set_dir / "logs"
                if not logs_dir.exists():
                    continue
                for log_file in sorted(logs_dir.glob("*.json")):
                    try:
                        with open(log_file, encoding="utf-8") as f:
                            entries = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        continue
                    for e in entries:
                        if e.get("status") != "success":
                            continue
                        rid = e["id"]
                        if rid not in target_ids:
                            continue
                        true_set = parse_label_set(e.get("true_label", ""))
                        pred_set = parse_label_set(e.get("predicted_label", ""))
                        if not true_set:
                            continue
                        records.append({
                            "prompt":          prompt,
                            "shot_count":      shot_count,
                            "set_id":          set_dir.name,
                            "report_id":       rid,
                            "true_set":        true_set,
                            "pred_set":        pred_set,
                            "correct":         true_set == pred_set,
                            "error_type":      classify_error(true_set, pred_set),
                        })

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"  Loaded Qwen baseline: {len(df):,} entries, "
              f"{df['report_id'].nunique()} unique IDs")
    return df


def find_best_zeroshot(qwen_df: pd.DataFrame) -> tuple:
    """
    Returns (prompt, DataFrame) for the zero-shot Qwen configuration
    with the highest macro F1 on the PEFT test IDs.
    """
    best_prompt, best_f1, best_df = None, -1.0, None
    for prompt, g in qwen_df[qwen_df["shot_count"] == 0].groupby("prompt"):
        # Use set_1 (or first available set) for zero-shot
        set_df = g[g["set_id"] == g["set_id"].min()]
        m = multilabel_metrics(set_df)
        if m and m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]; best_prompt = prompt; best_df = set_df.copy()
    return best_prompt, best_df


def find_best_fewshot(qwen_df: pd.DataFrame) -> tuple:
    """
    Returns (prompt, shot_count, majority_vote_DataFrame) for the Qwen few-shot
    configuration (majority vote across sets) with the highest macro F1.
    """
    best_key, best_f1, best_mv = None, -1.0, None
    for (prompt, sc), g in qwen_df[qwen_df["shot_count"] > 0].groupby(["prompt", "shot_count"]):
        # Majority vote per label across sets
        n_sets = g["set_id"].nunique()
        threshold = n_sets / 2
        mv_rows = []
        for rid, rg in g.groupby("report_id"):
            true_set   = rg["true_set"].iloc[0]
            label_votes = {l: 0 for l in VALID_LABELS}
            for _, row in rg.iterrows():
                for l in row["pred_set"]:
                    if l in label_votes: label_votes[l] += 1
            mv_pred = frozenset(l for l, cnt in label_votes.items() if cnt > threshold)
            mv_rows.append({"report_id": rid, "true_set": true_set,
                             "pred_set": mv_pred, "correct": mv_pred == true_set,
                             "error_type": classify_error(true_set, mv_pred)})
        mv_df = pd.DataFrame(mv_rows)
        m = multilabel_metrics(mv_df)
        if m and m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_key = (prompt, sc)
            best_mv  = mv_df.copy()
    return best_key, best_mv


# ══════════════════════════════════════════════════════════════════════════════
#  A1 — Compute metrics dataframe (no plots)
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a1(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Macro/micro F1, exact match, per-label P/R/F1 for each of the 10 configs."""
    print("\n-- A1: Metrics per configuration --")
    rows = []
    for config_num, g in df.groupby("config_num"):
        m = multilabel_metrics(g)
        if not m: continue
        flat = {"config_num": config_num,
                "config_short": LORA_CONFIGS.get(config_num, {}).get("short", f"cfg{config_num}")}
        for k, v in m.items():
            if k == "per_label":
                for lbl, lm in v.items():
                    for metric, val in lm.items():
                        flat[f"{lbl}__{metric}"] = val
            else:
                flat[k] = v
        rows.append(flat)

    a1 = pd.DataFrame(rows).sort_values("config_num")
    a1.to_csv(output_dir / "a1_metrics_per_config.csv", index=False)
    print(f"  Saved a1_metrics_per_config.csv ({len(a1)} rows)")

    return a1



# ══════════════════════════════════════════════════════════════════════════════
#  A2 — LoRA configuration comparison
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a2(a1: pd.DataFrame, output_dir: Path) -> int:
    """
    2x2 grid of macro F1, micro F1, exact match, and macro recall for all 10 configs.
    Returns the config_num of the best configuration (highest macro F1).
    """
    print("\n-- A2: Config comparison --")
    df_sorted = a1.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best_config = int(df_sorted.iloc[0]["config_num"])
    print(f"  Best config: {best_config} ({LORA_CONFIGS[best_config]['short']})  "
          f"macro F1 = {df_sorted.iloc[0]['f1_macro']:.4f}")

    x      = np.arange(len(df_sorted))
    labels = [f"{int(r['config_num'])}"
              for _, r in df_sorted.iterrows()]
    colors = ["#E74C3C" if int(r["config_num"]) == best_config else "#5B9BD5"
              for _, r in df_sorted.iterrows()]

    panels = [
        ("f1_macro",        "Macro F1",          "Macro F1 per Config (sorted)"),
        ("f1_micro",        "Micro F1",          "Micro F1 per Config (sorted)"),
        ("exact_match",     "Exact Match Acc.",  "Exact Match Acc. per Config (sorted)"),
        ("recall_macro",    "Macro Recall",      "Macro Recall per Config (sorted)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 13), sharey=False)
    axes = axes.flatten()
    for ax, (metric, ylabel, subtitle) in zip(axes, panels):
        # Each panel is sorted independently by its own metric (descending),
        # while the best-config bar stays highlighted in red wherever it lands.
        order = df_sorted[metric].values.argsort()[::-1]
        vals_sorted   = df_sorted[metric].values[order]
        labels_sorted = [labels[i] for i in order]
        colors_sorted = [colors[i] for i in order]

        bars = ax.bar(x, vals_sorted, color=colors_sorted, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels_sorted, fontsize=20)
        ax.set_xlabel("cfg", fontsize=24)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_ylabel(ylabel, fontsize=24); ax.set_ylim(0, 1.0)
        ax.set_title(subtitle, fontsize=26)
        for bar, val in zip(bars, vals_sorted):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=17)
        ax.axhline(vals_sorted.mean(), color="gray", ls="--", lw=1.5,
                   label=f"mean={vals_sorted.mean():.3f}")
        ax.legend(fontsize=18)

    fig.suptitle(f"LoRA Configuration Comparison (best: cfg{best_config} = {LORA_CONFIGS[best_config]['short']}, highlighted in red)",
                 fontsize=30, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "a2_config_comparison.png", output_dir)
    return best_config



# ══════════════════════════════════════════════════════════════════════════════
#  A4 — PEFT best config vs best zero-shot Qwen
# ══════════════════════════════════════════════════════════════════════════════

def analysis_a4(df: pd.DataFrame, a1: pd.DataFrame, best_config: int,
                qwen_df: pd.DataFrame, output_dir: Path) -> tuple:
    """
    Three-way comparison of the best PEFT config against:
      (1) the best zero-shot Qwen, and
      (2) the best few-shot Qwen (20-shot majority-vote ensemble),
    on the shared set of report IDs. Produces a single combined figure
    (per-label F1 + aggregate metrics) and returns (peft_cmp, zs_cmp)
    for downstream use in the statistical-test analysis (E).
    """
    print("\n-- A4: PEFT best config vs Qwen 0-shot and 20-shot MV --")

    # ── Zero-shot baseline ────────────────────────────────────────────────────
    best_prompt, zs_df = find_best_zeroshot(qwen_df)
    if zs_df is None or zs_df.empty:
        print("  No zero-shot data available — skipping A4"); return None, None

    peft_sub = df[df["config_num"] == best_config].copy()

    # ── Few-shot ensemble baseline ────────────────────────────────────────────
    best_key, fs_mv = find_best_fewshot(qwen_df)
    has_fewshot = fs_mv is not None and not fs_mv.empty

    # ── Common report IDs ─────────────────────────────────────────────────────
    common = set(peft_sub["report_id"]) & set(zs_df["report_id"])
    if has_fewshot:
        common &= set(fs_mv["report_id"])
    if not common:
        print("  No common report IDs — skipping A4"); return None, None

    peft_cmp = peft_sub[peft_sub["report_id"].isin(common)]
    zs_cmp   = zs_df[zs_df["report_id"].isin(common)]
    m_peft   = multilabel_metrics(peft_cmp)
    m_zs     = multilabel_metrics(zs_cmp)

    m_fs = None
    if has_fewshot:
        fs_cmp = fs_mv[fs_mv["report_id"].isin(common)]
        m_fs   = multilabel_metrics(fs_cmp)
        best_fs_prompt, best_sc = best_key

    print(f"  Common reports: {len(common)}")
    print(f"  Qwen 0-shot (prompt_{best_prompt})  macro F1 = {m_zs['f1_macro']:.4f}")
    if m_fs:
        print(f"  Qwen {best_sc}-shot MV               macro F1 = {m_fs['f1_macro']:.4f}")
    print(f"  PEFT cfg{best_config}                       macro F1 = {m_peft['f1_macro']:.4f}")

    # ── Combined figure: per-label F1 (left) + aggregate metrics (right) ──────
    C_PEFT, C_ZS, C_FS = "#E74C3C", "#3498DB", "#F39C12"
    n_series = 3 if m_fs else 2
    w = 0.8 / n_series

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    # Left: per-label F1
    ax = axes[0]
    x  = np.arange(len(VALID_LABELS))
    peft_f1s = [m_peft["per_label"][l]["f1"] for l in VALID_LABELS]
    zs_f1s   = [m_zs["per_label"][l]["f1"]   for l in VALID_LABELS]
    if m_fs:
        fs_f1s = [m_fs["per_label"][l]["f1"] for l in VALID_LABELS]
        ax.bar(x - w,   peft_f1s, width=w, color=C_PEFT, alpha=0.9, label="PEFT Best Config")
        ax.bar(x,       zs_f1s,   width=w, color=C_ZS,   alpha=0.9, label="Qwen3.5 0-shot")
        ax.bar(x + w,   fs_f1s,   width=w, color=C_FS,   alpha=0.9, label="Qwen3.5 20-shot MV")
    else:
        ax.bar(x - w/2, peft_f1s, width=w, color=C_PEFT, alpha=0.9, label="PEFT Best Config")
        ax.bar(x + w/2, zs_f1s,   width=w, color=C_ZS,   alpha=0.9, label="Qwen3.5 0-shot")
    ax.set_xticks(x); ax.set_xticklabels([LABEL_EN.get(l, l) for l in VALID_LABELS],
                                          rotation=20, ha="right", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel("F1", fontsize=24); ax.set_ylim(0, 1.05)
    ax.set_title("Per-label F1", fontsize=26)
    ax.legend(fontsize=19)

    # Right: aggregate metrics
    ax2 = axes[1]
    agg_metrics = ["f1_macro", "f1_micro", "exact_match", "precision_macro", "recall_macro"]
    agg_labels  = ["Macro F1", "Micro F1", "Exact Match", "Macro P", "Macro R"]
    x2 = np.arange(len(agg_metrics))
    peft_agg = [m_peft[k] for k in agg_metrics]
    zs_agg   = [m_zs[k]   for k in agg_metrics]
    if m_fs:
        fs_agg = [m_fs[k] for k in agg_metrics]
        ax2.bar(x2 - w, peft_agg, width=w, color=C_PEFT, alpha=0.9, label="PEFT Best Config")
        ax2.bar(x2,     zs_agg,   width=w, color=C_ZS,   alpha=0.9, label="Qwen3.5 0-shot")
        ax2.bar(x2 + w, fs_agg,   width=w, color=C_FS,   alpha=0.9, label="Qwen3.5 20-shot MV")
    else:
        ax2.bar(x2 - w/2, peft_agg, width=w, color=C_PEFT, alpha=0.9, label="PEFT Best Config")
        ax2.bar(x2 + w/2, zs_agg,   width=w, color=C_ZS,   alpha=0.9, label="Qwen3.5 0-shot")
    ax2.set_xticks(x2); ax2.set_xticklabels(agg_labels, rotation=15, fontsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    ax2.set_ylabel("Score", fontsize=24); ax2.set_ylim(0, 1.15)
    ax2.set_title("Aggregate Metrics", fontsize=26)
    ax2.legend(fontsize=19)

    fig.tight_layout()
    save_fig(fig, "a4_peft_vs_baselines.png", output_dir)
    return peft_cmp, zs_cmp


# ══════════════════════════════════════════════════════════════════════════════
#  E — Statistical tests: McNemar + Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

def analysis_e(peft_cmp: pd.DataFrame, zs_cmp: pd.DataFrame,
               best_config: int, best_prompt: str,
               output_dir: Path) -> pd.DataFrame:
    """
    E1 — McNemar per-report test: PEFT vs zero-shot Qwen.
        Tests whether the pattern of errors (which specific reports are wrong)
        differs significantly between the two approaches.

    E2 — Bootstrap CI on macro F1 delta: PEFT - zero-shot.
        Estimates uncertainty around the point estimate of improvement
        given the fixed single test partition.
    """
    print("\n-- E: McNemar + Bootstrap CI --")
    # Keep true_set only from peft side (ground truth is identical for both)
    merged = (peft_cmp[["report_id","correct","true_set","pred_set"]]
              .rename(columns={"correct":"peft_correct",
                               "pred_set":"peft_pred_set"})
              .merge(
                  zs_cmp[["report_id","correct","pred_set"]]
                  .rename(columns={"correct":"zs_correct",
                                   "pred_set":"zs_pred_set"}),
                  on="report_id"))
    n = len(merged)
    stat_rows = []

    # ── McNemar ──────────────────────────────────────────────────────────────
    both_correct   = int(( merged["peft_correct"] &  merged["zs_correct"]).sum())
    peft_only      = int(( merged["peft_correct"] & ~merged["zs_correct"]).sum())
    zs_only        = int((~merged["peft_correct"] &  merged["zs_correct"]).sum())
    both_wrong     = int((~merged["peft_correct"] & ~merged["zs_correct"]).sum())

    print(f"  n reports     : {n}")
    print(f"  Both correct  : {both_correct}")
    print(f"  PEFT only     : {peft_only}   (PEFT wins)")
    print(f"  Zero-shot only: {zs_only}   (zero-shot wins)")
    print(f"  Both wrong    : {both_wrong}")

    table = np.array([[both_correct, peft_only], [zs_only, both_wrong]])
    discordant = peft_only + zs_only

    if discordant < 5:
        print(f"  WARNING: discordant pairs = {discordant} — McNemar not reliable")
        mcnemar_stat, mcnemar_p = np.nan, np.nan
    else:
        # McNemar with continuity correction
        mcnemar_stat = (abs(peft_only - zs_only) - 1)**2 / discordant
        mcnemar_p    = 1 - stats.chi2.cdf(mcnemar_stat, df=1)

    sig_str = ("SIGNIFICANT" if (not np.isnan(mcnemar_p) and mcnemar_p < 0.05)
               else "not significant" if not np.isnan(mcnemar_p) else "N/A")
    print(f"  McNemar: chi2={mcnemar_stat:.4f}  p={mcnemar_p:.4f}  {sig_str}")
    stat_rows.append({
        "test": "mcnemar", "statistic": mcnemar_stat, "p_value": mcnemar_p,
        "significant": mcnemar_p < 0.05 if not np.isnan(mcnemar_p) else None,
        "peft_only_correct": peft_only, "zs_only_correct": zs_only,
        "both_correct": both_correct, "both_wrong": both_wrong, "n": n,
        "note": f"discordant={discordant}" + (" (<5: unreliable)" if discordant < 5 else ""),
    })

    # ── Bootstrap CI on macro F1 delta ────────────────────────────────────────
    # Prepare aligned DataFrames indexed by position for clean bootstrap resampling
    peft_records = (merged[["true_set","peft_pred_set"]]
                    .rename(columns={"peft_pred_set":"pred_set"})
                    .reset_index(drop=True))
    zs_records   = (merged[["true_set","zs_pred_set"]]
                    .rename(columns={"zs_pred_set":"pred_set"})
                    .reset_index(drop=True))

    obs_diff = (multilabel_metrics(peft_records)["f1_macro"] -
                multilabel_metrics(zs_records)["f1_macro"])

    rng      = np.random.RandomState(RANDOM_SEED)
    n_ids    = len(peft_records)
    bs_diffs = []

    for _ in range(N_BOOTSTRAP):
        idx     = rng.randint(0, n_ids, size=n_ids)
        peft_f1 = multilabel_metrics(peft_records.iloc[idx])["f1_macro"]
        zs_f1   = multilabel_metrics(zs_records.iloc[idx])["f1_macro"]
        bs_diffs.append(peft_f1 - zs_f1)

    bs_diffs = np.array(bs_diffs)
    ci_lo = float(np.percentile(bs_diffs, 2.5))
    ci_hi = float(np.percentile(bs_diffs, 97.5))
    print(f"  Bootstrap delta: {obs_diff:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    stat_rows.append({
        "test": "bootstrap_f1_delta", "statistic": obs_diff,
        "ci_low_95": ci_lo, "ci_high_95": ci_hi, "n_bootstrap": N_BOOTSTRAP,
        "significant": (ci_lo > 0 or ci_hi < 0), "n": n,
        "note": f"95% CI excludes 0: {'yes' if (ci_lo>0 or ci_hi<0) else 'no'}",
    })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(output_dir / "e_statistical_tests.csv", index=False)

    # ── Plot: Bootstrap CI on the Macro F1 delta (single panel, no titles) ────
    fig, ax2 = plt.subplots(figsize=(12, 8))
    ax2.hist(bs_diffs, bins=60, color="#5B9BD5", alpha=0.75, edgecolor="white")
    ax2.axvline(obs_diff, color="#E74C3C", lw=3, label=f"Observed delta:\n{obs_diff:+.4f}")
    ax2.axvline(ci_lo,    color="#888",    lw=2, ls="--", label=f"95% CI:\n[{ci_lo:+.3f}, {ci_hi:+.3f}]")
    ax2.axvline(ci_hi,    color="#888",    lw=2, ls="--")
    ax2.axvline(0,        color="black",   lw=2, ls="-", alpha=0.5, label="No difference (0)")
    ax2.fill_betweenx([0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1],
                       ci_lo, ci_hi, alpha=0.1, color="#888")
    ax2.set_xlabel("Macro F1 delta (PEFT - 0-shot Qwen3.5)", fontsize=24)
    ax2.set_ylabel("Bootstrap frequency", fontsize=24)
    ax2.tick_params(axis="both", labelsize=20)
    ax2.legend(fontsize=18)

    fig.tight_layout()
    save_fig(fig, "e_statistical_tests.png", output_dir)
    return stat_df



def export_always_wrong(df: pd.DataFrame, output_dir: Path) -> None:
    """Export all log entries for reports misclassified in all 10 LoRA configurations."""
    print("\n-- EXPORT: Reports wrong 100% of the time --")
    total_per  = df.groupby("report_id").size()
    errors_per = df[df["error_type"] != "correct"].groupby("report_id").size()
    ids_100 = [rid for rid in total_per.index
               if errors_per.get(rid, 0) == total_per[rid]]
    n_configs = df["config_num"].nunique()
    print(f"  Found {len(ids_100)} reports wrong in ALL {n_configs} configs")
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
                    "config_num":       int(row["config_num"]),
                    "config_short":     LORA_CONFIGS.get(int(row["config_num"]), {}).get("short", ""),
                    "predicted_label":  row["predicted_label"],
                    "error_type":       row["error_type"],
                    "model_confidence": row["model_confidence"],
                }
                for _, row in sub.iterrows()
            ],
        })
    path = output_dir / "always_wrong_peft.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path.name}  ({len(result)} reports)")


def export_error_type_samples(df: pd.DataFrame, output_dir: Path,
                               n_samples: int = 5) -> None:
    """Export n_samples random entries per LP error type (4 types x 5 = 20 entries).
    Note: model_motivation is null in PEFT logs by design."""
    print("\n-- EXPORT: Error type samples (5 per type) --")
    errors = df[df["error_type"] != "correct"].copy()
    result = {}
    for et in ERROR_TYPES:
        subset  = errors[errors["error_type"] == et]
        n       = min(n_samples, len(subset))
        sampled = subset.sample(n=n, random_state=RANDOM_SEED) if n > 0 else subset
        result[et] = [
            {
                "report_id":       int(row["report_id"]),
                "true_label":      row["true_label"],
                "predicted_label": row["predicted_label"],
                "error_type":      row["error_type"],
                "config_num":      int(row["config_num"]),
                "config_short":    LORA_CONFIGS.get(int(row["config_num"]), {}).get("short", ""),
                "model_confidence": row["model_confidence"],
            }
            for _, row in sampled.iterrows()
        ]
        print(f"  {ERROR_LABELS[et]}: {len(subset):,} total -> sampled {len(result[et])}")
    path = output_dir / "error_samples_by_type_peft.json"
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
    p = argparse.ArgumentParser(description="PEFT task - analysis pipeline")
    p.add_argument("--peft_dir",       required=True,
                   help="Root PEFT directory (contains config_01/ ... config_10/)")
    p.add_argument("--output_dir",     default="./figures/peft")
    p.add_argument("--multiclass_dir", default="./results_multiclass",
                   help="MULTICLASS root dir for A4/A5/E comparisons (default: ./results_multiclass)")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    peft_dir   = Path(args.peft_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    multiclass_dir = Path(args.multiclass_dir)

    print("=" * 60)
    print("  PEFT TASK - Analysis Pipeline")
    print(f"  PEFT dir       : {peft_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Multiclass dir : {multiclass_dir}")
    print("=" * 60)

    # ── Load PEFT logs ────────────────────────────────────────────────────────
    df = load_peft_logs(peft_dir)
    if df.empty:
        print("ERROR: no PEFT data loaded - check --peft_dir path."); return

    # ── A2: LoRA configuration comparison ────────────────────────────────────
    a1_df       = analysis_a1(df, output_dir)
    best_config = analysis_a2(a1_df, output_dir)

    # ── Export: always-wrong + error type samples ─────────────────────────────
    print("\n" + "-"*60)
    print("  EXPORT")
    print("-"*60)
    export_always_wrong(df, output_dir)
    export_error_type_samples(df, output_dir)

    # ── A4 / A5 / E: PEFT vs Qwen comparisons ────────────────────────────────
    print("\n" + "-"*60)
    print("  A4, A5, E — PEFT vs Qwen baseline")
    print("-"*60)

    if not multiclass_dir.exists():
        print(f"  ERROR: --multiclass_dir '{multiclass_dir}' not found.")
        print("  Cannot generate a4_peft_vs_zeroshot.png, a5_peft_vs_fewshot_ensemble.png,")
        print("  or e_statistical_tests.png without the MULTICLASS logs.")
        print("  Run with: --multiclass_dir /path/to/results_multiclass")
    else:
        target_ids = set(df["report_id"].unique())
        qwen_df    = load_qwen_logs(multiclass_dir, target_ids)

        if qwen_df.empty:
            print("  ERROR: no Qwen logs found in the MULTICLASS directory matching "
                  "the PEFT test report IDs.")
        else:
            best_prompt_str, _ = find_best_zeroshot(qwen_df)
            peft_cmp, zs_cmp   = analysis_a4(df, a1_df, best_config, qwen_df, output_dir)

            if peft_cmp is not None and zs_cmp is not None and not peft_cmp.empty:
                analysis_e(peft_cmp, zs_cmp, best_config, best_prompt_str, output_dir)
            else:
                print("  WARNING: A4 returned no common reports — "
                      "e_statistical_tests.png cannot be generated.")

    print("\n" + "="*60)
    print("  COMPLETE")
    print(f"  Outputs: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()