"""
qwen_evaluation.py
──────────────────────────────────────────────────────────────────────────────
Evaluation pipeline for QLoRA fine-tuned Qwen3.5-35B-A3B adapters.
Loads the test split from the HF DatasetDict and runs inference with each
specified adapter × each specified prompt, saving logs and statistics in the
same structure as pipeline_multiclass.py.

Output structure:
  results_multiclass/
    qwen_finetuned/
      config_{N}/
        prompt_{X}/
          finetuned/
            set_1/
              logs/logs_multiclass_{timestamp}.json
              results/results_multiclass_{timestamp}.json

Usage:
    # Test configs 1, 2, 3 with all prompts
    python finetuning/qwen_evaluation.py \
        --configs 1 2 3 \
        --output_base_dir ./output \
        --dataset_dir ./dataset/hf_dataset \
        --reports_file ./filtered_reports.json

    # Test specific adapter paths explicitly
    python finetuning/qwen_evaluation.py \
        --adapter_paths ./output/config_01/final_adapter ./output/config_02/final_adapter \
        --dataset_dir ./dataset/hf_dataset

    # Only test prompt A and B (skip original)
    python finetuning/qwen_evaluation.py \
        --configs 1 2 \
        --output_base_dir ./output \
        --dataset_dir ./dataset/hf_dataset \
        --prompts A B
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import gc
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS: dict[str, str] = {
    "A": (
        "You are an expert neuroradiologist. Your task is to classify pathological brain MRI "
        "radiology reports into one or more of the following categories:\n\n"
        "  - Neoformazione\n"
        "  - Vascolare\n"
        "  - Infiammazione o Infezione\n"
        "  - Neurodegenerativo\n"
        "  - Altro\n\n"
        "Classification rules:\n"
        "  - A single report can be assigned MULTIPLE categories simultaneously. "
        "Identify ALL relevant categories present.\n"
        "  - \"Neoformazione\" must only be assigned for intracranial findings.\n"
        "  - \"Vascolare\" must only be assigned for intracranial findings.\n"
        "  - \"Infiammazione o Infezione\" must only be assigned for findings involving "
        "the central nervous system.\n"
        "  - \"Neurodegenerativo\" must only be assigned for findings involving "
        "the central nervous system.\n"
        "  - \"Altro\" covers all pathologies not fitting the above classes, including "
        "extracranial lesions.\n"
        "  - Do not infer or assume pathologies not explicitly stated or clearly implied "
        "in the report.\n\n"
        'ALWAYS respond in the following exact JSON format:\n'
        '{"labels": "category1, category2, ...", "confidence": <integer 1-5>}\n\n'
        "Confidence scale: 1 = very uncertain, 2 = uncertain, 3 = moderate, "
        "4 = fairly confident, 5 = very confident.\n"
        "Every report must be assigned at least one label. If no finding clearly fits "
        "the defined categories, assign 'Altro' as the default label."
    ),
    "B": (
        "You are an expert neuroradiologist. Your task is to classify pathological brain MRI "
        "radiology reports into one or more of the following categories:\n\n"
        "  - Neoformazione \u2014 Intracranial neoplasms, cysts, and tumor-like lesions.\n"
        "  - Vascolare \u2014 Spontaneous (non-traumatic) intracranial hemorrhages and "
        "intracranial vascular lesions (aneurysms, vascular malformations, stroke, "
        "venous occlusions, vasculopathies).\n"
        "  - Infiammazione o Infezione \u2014 Infectious and non-infectious inflammatory "
        "diseases of the central nervous system.\n"
        "  - Neurodegenerativo \u2014 Diseases characterized by progressive neuronal loss.\n"
        "  - Altro \u2014 Pathologies of other nature (e.g., trauma, malformations, "
        "toxic-metabolic conditions), including extracranial lesions.\n\n"
        "Classification rules:\n"
        "  - A single report can be assigned MULTIPLE categories simultaneously. "
        "Identify ALL relevant categories present.\n"
        "  - \"Neoformazione\" must only be assigned for intracranial findings.\n"
        "  - \"Vascolare\" must only be assigned for intracranial findings.\n"
        "  - \"Infiammazione o Infezione\" must only be assigned for findings involving "
        "the central nervous system.\n"
        "  - \"Neurodegenerativo\" must only be assigned for findings involving "
        "the central nervous system.\n"
        "  - \"Altro\" covers all pathologies not fitting the above classes, including "
        "extracranial lesions.\n"
        "  - Do not infer or assume pathologies not explicitly stated or clearly implied "
        "in the report.\n\n"
        'ALWAYS respond in the following exact JSON format:\n'
        '{"labels": "category1, category2, ...", "confidence": <integer 1-5>}\n\n'
        "Confidence scale: 1 = very uncertain, 2 = uncertain, 3 = moderate, "
        "4 = fairly confident, 5 = very confident.\n"
        "Every report must be assigned at least one label. If no finding clearly fits "
        "the defined categories, assign 'Altro' as the default label."
    ),
    "original": (
        "You are a system specialized in classifying brain MRI reports. "
        "Classify the report into one or more of the following categories: "
        "Neoformazione, Vascolare, Infiammazione o Infezione, Neurodegenerativo, Altro. "
        "Use multiple labels if applicable. "
        "Assign categories only to findings involving the intracranial/central nervous system; "
        "otherwise classify them as Altro. Do not infer unstated pathologies. "
        "Confidence scale: 1 = very uncertain, 2 = uncertain, 3 = moderate, "
        "4 = fairly confident, 5 = very confident. "
        'Output strict JSON only: {"labels": "category1, category2, ...", "confidence": <integer 1-5>}'
    ),
}

# Sentinel: use the prompt already stored in the dataset text column
PROMPTS["dataset"] = "__use_dataset_text__"

VALID_LABELS = {
    "Neoformazione",
    "Vascolare",
    "Infiammazione o Infezione",
    "Neurodegenerativo",
    "Altro",
}

NUMBER_TO_LABEL = {
    "1": "Neoformazione",
    "2": "Vascolare",
    "3": "Infiammazione o Infezione",
    "4": "Neurodegenerativo",
    "5": "Altro",
}

MAX_NEW_TOKENS = 1024


# ══════════════════════════════════════════════════════════════════════════════
#  LOG ENTRY
# ══════════════════════════════════════════════════════════════════════════════

class LogEntry:
    def __init__(self, id, report, true_label, confidence=None, id_refertante=None,
                 status="failed", task="multiclass", set_id=1, shot_ids=None,
                 num_shot=0, predicted_label=None, model_confidence=None,
                 model_motivation=None, response=None, label_evidence=None,
                 error=None, prompt=None, created_at=None):
        self.id               = id
        self.report           = report
        self.true_label       = true_label
        self.confidence       = confidence
        self.id_refertante    = id_refertante
        self.status           = status
        self.task             = task
        self.set_id           = set_id
        self.shot_ids         = shot_ids or []
        self.num_shot         = num_shot
        self.predicted_label  = predicted_label
        self.model_confidence = model_confidence
        self.model_motivation = model_motivation
        self.response         = response
        self.label_evidence   = label_evidence or {}
        self.error            = error
        self.prompt           = prompt
        self.created_at       = created_at or time.strftime("%d-%m-%Y %H:%M")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "report": self.report, "true_label": self.true_label,
            "confidence": self.confidence, "id_refertante": self.id_refertante,
            "status": self.status, "task": self.task, "set_id": self.set_id,
            "shot_ids": self.shot_ids, "num_shot": self.num_shot,
            "predicted_label": self.predicted_label,
            "model_confidence": self.model_confidence,
            "model_motivation": self.model_motivation,
            "response": self.response, "label_evidence": self.label_evidence,
            "error": self.error, "prompt": self.prompt, "created_at": self.created_at,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_statistics_multiclass(logs: list[dict]) -> dict:
    from collections import defaultdict
    successful = [e for e in logs if e.get("status") == "success"]
    if not successful:
        return {"error": "no successful predictions"}

    LABELS = list(VALID_LABELS)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for entry in successful:
        true_set = {l.strip() for l in (entry.get("true_label") or "").split(",")
                    if l.strip() in VALID_LABELS}
        pred_set = {l.strip() for l in (entry.get("predicted_label") or "").split(",")
                    if l.strip() in VALID_LABELS}
        for label in LABELS:
            if label in true_set and label in pred_set:   tp[label] += 1
            elif label not in true_set and label in pred_set: fp[label] += 1
            elif label in true_set and label not in pred_set: fn[label] += 1

    per_label = {}
    macro_p = macro_r = macro_f1 = 0.0
    for label in LABELS:
        p  = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r  = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[label] = {"precision": round(p,4), "recall": round(r,4),
                            "f1": round(f1,4), "tp": tp[label],
                            "fp": fp[label], "fn": fn[label]}
        macro_p += p; macro_r += r; macro_f1 += f1

    n = len(LABELS)
    exact = sum(
        1 for e in successful
        if {l.strip() for l in (e.get("true_label") or "").split(",") if l.strip() in VALID_LABELS}
        == {l.strip() for l in (e.get("predicted_label") or "").split(",") if l.strip() in VALID_LABELS}
    )
    return {
        "per_label":         per_label,
        "macro_precision":   round(macro_p  / n, 4),
        "macro_recall":      round(macro_r  / n, 4),
        "macro_f1":          round(macro_f1 / n, 4),
        "exact_match":       round(exact / len(successful), 4),
        "exact_match_count": exact,
        "total_evaluated":   len(successful),
    }


def save_logs_as_json(logs: list[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
#  PARSING
# ══════════════════════════════════════════════════════════════════════════════

def extract_report_text_from_prompt(text: str) -> str:
    """Extracts raw report from the formatted chat template stored in the dataset."""
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if match:
        # Strip /no_think suffix that was added during dataset preparation
        return match.group(1).replace("/no_think", "").strip()
    return text.strip()


def parse_predicted_labels(raw: str) -> tuple:
    """
    Parses JSON model output into (labels_string, confidence, reasoning).
    Handles <think> tokens, truncated JSON, and numbered label fallback.
    """
    import json as _json

    # Strip thinking tokens
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"</think>", "", cleaned)
    cleaned = cleaned.replace("/no_think", "").strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip("` \n")

    def _resolve_label(token: str):
        t = token.strip()
        if t in VALID_LABELS:          return t
        if t in NUMBER_TO_LABEL:       return NUMBER_TO_LABEL[t]
        for label in VALID_LABELS:
            if label.lower() in t.lower() or t.lower() in label.lower():
                return label
        return None

    def _extract_labels(labels_raw: str) -> str:
        parts    = [p.strip() for p in str(labels_raw).split(",")]
        resolved = [_resolve_label(p) for p in parts]
        valid    = [r for r in resolved if r is not None]
        return ", ".join(valid) if valid else str(labels_raw).strip()

    def _safe_int(v):
        try:    return int(v)
        except: return None

    # Try extraction between first { and last }
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = _json.loads(cleaned[start:end + 1])
            return (
                _extract_labels(str(obj.get("labels", ""))),
                _safe_int(obj.get("confidence")),
                obj.get("reasoning", None),
            )
        except (_json.JSONDecodeError, AttributeError):
            pass

    # Regex field extraction — handles truncated JSON
    labels_match = re.search(r'"labels"\s*:\s*"([^"]+)"', cleaned)
    conf_match   = re.search(r'"confidence"\s*:\s*["\']?(\d+)["\']?', cleaned)
    if labels_match:
        return (
            _extract_labels(labels_match.group(1)),
            _safe_int(conf_match.group(1)) if conf_match else None,
            None,
        )

    # Last resort: plain comma-separated
    parts = [p.strip() for p in cleaned.split(",")]
    valid = [_resolve_label(p) for p in parts]
    valid = [v for v in valid if v is not None]
    return ", ".join(valid) if valid else cleaned, None, None


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING / ADAPTER HOT-SWAP
#
#  Strategy: load the base model ONCE, then hot-swap adapters between configs.
#
#  Flow:
#    base_model, tokenizer = load_base_model(model_id)   ← once
#    for each adapter:
#        model = attach_adapter(base_model, adapter_path) ← adds LoRA modules
#        run inference
#        base_model = detach_adapter(model)               ← strips LoRA, returns base
#    free_base_model(base_model, tokenizer)               ← once at the end
#
#  PEFT's unload() removes adapter modules in-place and returns the unwrapped
#  base model, so the next attach_adapter() starts from a clean state without
#  any re-downloading or re-quantizing the 71GB weights.
# ══════════════════════════════════════════════════════════════════════════════

def load_base_model(model_id: str, first_adapter_path: str):
    """
    Loads the base model (4-bit NF4) and tokenizer once.
    Tokenizer is loaded from the first adapter directory because it was saved
    there by qwen_finetune.py and is guaranteed to match the training config.
    Returns (base_model, tokenizer) — no adapter attached yet.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading tokenizer from {first_adapter_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(first_adapter_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model (once): {model_id} ...")
    t0   = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    print(f"  Base model loaded in {time.time()-t0:.1f}s")
    base.config.use_cache = False

    n_params = sum(p.numel() for p in base.parameters()) / 1e9
    print(f"  Parameters: {n_params:.2f}B")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f} GB allocated")

    return base, tokenizer


def attach_adapter(base_model, adapter_path: str):
    """
    Wraps the base model with a LoRA adapter loaded from adapter_path.
    Returns the PeftModel ready for inference.
    """
    from peft import PeftModel
    print(f"  Attaching adapter: {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def detach_adapter(peft_model):
    """
    Removes LoRA adapter modules from the model graph and returns the clean
    base model. Uses PEFT's unload() which strips adapters without merging
    weights, leaving the base model identical to its pre-attachment state.
    GPU memory used by adapter weights is freed immediately.
    """
    import gc, torch
    print("  Detaching adapter ...")
    base_model = peft_model.unload()   # strips LoRA, returns unwrapped base
    gc.collect()
    torch.cuda.empty_cache()
    return base_model


def free_base_model(base_model, tokenizer):
    """Frees the base model after all adapters have been evaluated."""
    import gc, torch
    print("\nFreeing base model ...")
    del base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("GPU memory fully released.")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def infer(model, tokenizer, report_text: str, system_prompt: str,
          prebuilt_prompt: str = None) -> tuple:
    import torch

    if prebuilt_prompt is not None:
        # Use the training prompt stored in the dataset text column verbatim.
        # Strip the assistant answer turn and append the generation marker.
        # This is byte-for-byte identical to what the model saw during training.
        cut    = prebuilt_prompt.find("<|im_start|>assistant")
        prompt = prebuilt_prompt[:cut] + "<|im_start|>assistant\n"
        # /no_think is already embedded in the stored user turn — do not add again
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": report_text.strip() + " /no_think"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw        = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"    Raw model output:\n\n{raw}")
    labels, confidence, reasoning = parse_predicted_labels(raw)
    return labels, confidence, reasoning, raw


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE PROMPT EVALUATION (one adapter × one prompt)
# ══════════════════════════════════════════════════════════════════════════════

def run_prompt_eval(model, tokenizer, test, metadata_by_id: dict,
                    system_prompt: str, prompt_key: str,
                    config_label: str, adapter_path: str) -> None:
    """
    Runs inference on the full test split for one (adapter, prompt) pair
    and saves logs + statistics to disk.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_path   = f"./results_multiclass/qwen_finetuned/{config_label}/prompt_{prompt_key}/finetuned/set_1"
    logs_dir    = f"{base_path}/logs"
    results_dir = f"{base_path}/results"
    os.makedirs(logs_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    logs_fp  = f"{logs_dir}/logs_multiclass_{timestamp}.json"
    stats_fp = f"{results_dir}/results_multiclass_{timestamp}.json"

    n    = len(test)
    logs = []

    for i in range(n):
        row      = test[i]
        rid      = row["id"]
        true_labels_list = row.get("label_names", row.get("labels", []))
        true_label = (
            ", ".join(true_labels_list)
            if isinstance(true_labels_list, list)
            else str(true_labels_list)
        )

        meta           = metadata_by_id.get(rid, {})
        report_text    = meta.get("report_text") or extract_report_text_from_prompt(row["text"])
        label_evidence = meta.get("label_evidence", {})
        confidence     = meta.get("confidence", None)
        id_refertante  = meta.get("id_refertante", None)

        pred = raw = model_conf = model_mot = None
        status = "failed"
        err    = None

        try:
            prebuilt = row["text"] if prompt_key == "dataset" else None
            pred, model_conf, model_mot, raw = infer(
                model, tokenizer, report_text, system_prompt,
                prebuilt_prompt=prebuilt,
            )
            status = "success"
            print(f"  [{i+1:04d}/{n}] ID:{rid} | GT:{true_label} | PRED:{pred} | CONF:{model_conf}")
        except Exception:
            err = traceback.format_exc().splitlines()[-1]
            print(f"  [{i+1:04d}/{n}] FAIL ID:{rid} | {err}")

        logs.append(LogEntry(
            id=rid, report=report_text, true_label=true_label,
            label_evidence=label_evidence, confidence=confidence,
            id_refertante=id_refertante, status=status,
            task="multiclass", set_id=1, shot_ids=[], num_shot=0,
            predicted_label=pred, model_confidence=model_conf,
            model_motivation=model_mot, response=raw,
            error=err, prompt=prompt_key,
            created_at=time.strftime("%d-%m-%Y %H:%M"),
        ).to_dict())

    # Save logs
    save_logs_as_json(logs, logs_fp)

    # Compute and save stats
    stats = calculate_statistics_multiclass(logs)
    evaluated = sum(1 for e in logs if e.get("status") == "success")
    stats.update({
        "evaluated":    evaluated,
        "failed":       n - evaluated,
        "adapter_path": adapter_path,
        "config_label": config_label,
        "prompt":       prompt_key,
        "model":        "qwen_finetuned",
    })
    with open(stats_fp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n  {'─'*50}")
    print(f"  {config_label} | Prompt {prompt_key} | {evaluated}/{n} evaluated")
    print(f"  Exact match : {stats.get('exact_match', 0):.4f}  ({stats.get('exact_match_count', 0)}/{evaluated})")
    print(f"  Macro F1    : {stats.get('macro_f1', 0):.4f}")
    print(f"  Macro P     : {stats.get('macro_precision', 0):.4f}")
    print(f"  Macro R     : {stats.get('macro_recall', 0):.4f}")
    for label, s in stats.get("per_label", {}).items():
        print(f"    {label:<35} F1={s['f1']:.4f}  P={s['precision']:.4f}  R={s['recall']:.4f}")
    print(f"  Logs   → {logs_fp}")
    print(f"  Stats  → {stats_fp}")
    print(f"  {'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTER RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def resolve_adapters(args) -> list[tuple[str, str]]:
    """
    Returns a list of (adapter_path, config_label) pairs.
    Sources (in priority order):
      1. --adapter_paths  : explicit list of adapter directory paths
      2. --configs        : config numbers resolved via --output_base_dir
    """
    if args.adapter_paths:
        return [
            (p, Path(p).parent.parent.name)   # config_label = grandparent dir name
            for p in args.adapter_paths
        ]

    if args.configs:
        pairs = []
        for n in args.configs:
            label        = f"config_{n:02d}"
            adapter_path = Path(args.output_base_dir) / label / "final_adapter"
            if not adapter_path.exists():
                print(f"  WARNING: adapter not found for config {n}: {adapter_path} — skipping")
                continue
            pairs.append((str(adapter_path), label))
        return pairs

    raise ValueError("Provide either --configs or --adapter_paths.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(args):
    from datasets import load_from_disk

    # ── Load test split ───────────────────────────────────────────────────────
    print(f"Loading test split from {args.dataset_dir} ...")
    ds   = load_from_disk(args.dataset_dir)
    test = ds["test"]
    print(f"  Test samples: {len(test)}")
    print(f"  Columns     : {test.column_names}")

    # ── Load optional metadata ────────────────────────────────────────────────
    metadata_by_id: dict[int, dict] = {}
    if args.reports_file and Path(args.reports_file).exists():
        print(f"Loading metadata from {args.reports_file} ...")
        with open(args.reports_file, encoding="utf-8") as f:
            for r in json.load(f):
                metadata_by_id[r["id"]] = r
        print(f"  Loaded metadata for {len(metadata_by_id)} reports")
    else:
        print("  No --reports_file — report text extracted from dataset text column")

    # ── Resolve adapters ──────────────────────────────────────────────────────
    adapters = resolve_adapters(args)
    prompts  = args.prompts

    print(f"\nAdapters to evaluate : {len(adapters)}")
    for path, label in adapters:
        print(f"  {label} → {path}")
    print(f"Prompts to test      : {prompts}")
    print(f"Total runs           : {len(adapters)} × {len(prompts)} = {len(adapters)*len(prompts)}\n")

    global_start = time.time()

    # ── Load base model ONCE ──────────────────────────────────────────────────
    # The tokenizer is taken from the first adapter directory (all adapters
    # share the same tokenizer since they were trained from the same base).
    base_model, tokenizer = load_base_model(args.model_id, adapters[0][0])

    # ── Main loop: hot-swap adapters, base model stays in GPU memory ──────────
    for adapter_idx, (adapter_path, config_label) in enumerate(adapters, 1):
        print(f"\n{'═'*60}")
        print(f"  Adapter {adapter_idx}/{len(adapters)}: {config_label}")
        print(f"  Path: {adapter_path}")
        print(f"{'═'*60}")

        model = attach_adapter(base_model, adapter_path)

        for prompt_key in prompts:
            print(f"\n  ── Prompt {prompt_key} ──")
            run_prompt_eval(
                model=model,
                tokenizer=tokenizer,
                test=test,
                metadata_by_id=metadata_by_id,
                system_prompt=PROMPTS[prompt_key],
                prompt_key=prompt_key,
                config_label=config_label,
                adapter_path=adapter_path,
            )

        # Strip adapter, recover base model for next iteration
        base_model = detach_adapter(model)

    # ── Free everything once all adapters are done ────────────────────────────
    free_base_model(base_model, tokenizer)

    elapsed = time.time() - global_start
    print(f"\n{'═'*60}")
    print(f"  ALL EVALUATIONS COMPLETE")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results in: ./results_multiclass/qwen_finetuned/")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate one or more QLoRA adapters on the HF test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run configs 1, 2, 3 with all prompts
  python finetuning/qwen_evaluation.py --configs 1 2 3 --output_base_dir ./output --dataset_dir ./dataset/hf_dataset

  # Run all 10 configs with prompts A and B only
  python finetuning/qwen_evaluation.py --configs $(seq 1 10) --output_base_dir ./output --dataset_dir ./dataset/hf_dataset --prompts A B

  # Explicit adapter paths
  python finetuning/qwen_evaluation.py --adapter_paths ./output/config_01/final_adapter ./output/config_02/final_adapter --dataset_dir ./dataset/hf_dataset
        """,
    )
    # ── Adapter source (one of the two is required) ───────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--configs", nargs="+", type=int,
                     help="Config numbers to evaluate (e.g. 1 2 3). "
                          "Adapter path is resolved as {output_base_dir}/config_{N:02d}/final_adapter")
    src.add_argument("--adapter_paths", nargs="+",
                     help="Explicit paths to adapter directories.")

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--dataset_dir", required=True,
                   help="Path to HF DatasetDict (must contain a 'test' split)")

    # ── Optional ──────────────────────────────────────────────────────────────
    p.add_argument("--output_base_dir", default="./output",
                   help="Base directory where config folders live (default: ./output). "
                        "Used only with --configs.")
    p.add_argument("--reports_file", default=None,
                   help="Path to filtered_reports.json for enriched metadata "
                        "(report_text, label_evidence, confidence, id_refertante). "
                        "If omitted, report text is extracted from the dataset text column.")
    p.add_argument("--prompts", nargs="+", default=["A", "B", "original"],
                   choices=list(PROMPTS.keys()),
                   help="Prompt variants to test (default: A B original dataset). "
                        "'dataset' uses the system prompt stored in the test split text column.")
    p.add_argument("--model_id", default="Qwen/Qwen3.5-35B-A3B",
                   help="Base model HuggingFace ID")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)