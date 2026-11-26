"""
qwen_finetune.py
──────────────────────────────────────────────────────────────────────────────
Production PEFT QLoRA fine-tuning for Qwen/Qwen3.5-35B-A3B.
Supports multiple pre-defined LoRA configurations selectable via --config.

All 10 configurations share the same dataset, base model, and training
hyperparameters — only the LoRA architecture and learning rate vary.
Each config should be run with a different --output_dir so adapters
are saved separately and can be compared.

Run:
    python finetuning/qwen_finetune.py \
        --config 1 \
        --dataset_dir ./dataset/hf_dataset \
        --weights_file ./dataset/class_weights.json \
        --output_dir ./output/config_01

    # List all available configs without running:
    python finetuning/qwen_finetune.py --list_configs

    # Run all configs sequentially (bash):
    for i in $(seq 1 10); do
        python finetuning/qwen_finetune.py \
            --config $i \
            --dataset_dir ./dataset/hf_dataset \
            --weights_file ./dataset/class_weights.json \
            --output_dir ./output/config_$(printf '%02d' $i)
    done
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED SETTINGS — identical across every configuration
# ══════════════════════════════════════════════════════════════════════════════

SHARED = {
    "model_id":                "Qwen/Qwen3.5-35B-A3B",
    "max_length":              2048,
    "batch_size":              1,
    "grad_accum":              8,        # effective_batch = 8 x num_gpus
    "weight_decay":            0.01,
    "warmup_steps":            20,
    "max_grad_norm":           1.0,
    "weight_smoothing":        "log",
    "early_stopping_patience": 2,
    "save_steps":              50,
    "eval_steps":              50,
    "seed":                    42,
    "report_to":               "none",
}


# ══════════════════════════════════════════════════════════════════════════════
#  LORA CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoraConfig:
    target_modules:      list
    r:                   int
    lora_alpha:          int
    num_epochs:          int
    learning_rate:       float
    lora_dropout:        float          = 0.05
    layers_to_transform: Optional[list] = None
    description:         str            = ""


CONFIGS: dict[int, LoraConfig] = {

    1: LoraConfig(
        description="Attention QV | r=8 | baseline minimal",
        target_modules=["q_proj", "v_proj"],
        r=8, lora_alpha=16,
        num_epochs=3, learning_rate=1e-4,
    ),

    2: LoraConfig(
        description="Attention QV | r=16 | standard attention-only",
        target_modules=["q_proj", "v_proj"],
        r=16, lora_alpha=32,
        num_epochs=3, learning_rate=1e-4,
    ),

    3: LoraConfig(
        description="Attention QKVO | r=16 | full attention",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=16, lora_alpha=32,
        num_epochs=3, learning_rate=8e-5,
    ),

    4: LoraConfig(
        description="Attention QKVO | r=32 | high-rank full attention",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=32, lora_alpha=64,
        num_epochs=4, learning_rate=8e-5,
    ),

    5: LoraConfig(
        description="Attention QKVO | r=16 | layers 28-39 (last 12)",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=16, lora_alpha=32,
        num_epochs=3, learning_rate=8e-5,
        layers_to_transform=list(range(28, 40)),
    ),

    6: LoraConfig(
        description="Attention QKVO | r=16 | layers 20-39 (last 20)",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=16, lora_alpha=32,
        num_epochs=3, learning_rate=8e-5,
        layers_to_transform=list(range(20, 40)),
    ),

    7: LoraConfig(
        description="Attention QKVO + gate_proj | r=16 | partial FFN",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        r=16, lora_alpha=32,
        num_epochs=3, learning_rate=6e-5,
    ),

    8: LoraConfig(
        description="Attention QKVO + full FFN | r=16 | dropout=0.1",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        r=16, lora_alpha=32, lora_dropout=0.1,
        num_epochs=3, learning_rate=5e-5,
    ),

    9: LoraConfig(
        description="Attention QKVO + full FFN | r=16 | layers 28-39",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        r=16, lora_alpha=32,
        num_epochs=3, learning_rate=5e-5,
        layers_to_transform=list(range(28, 40)),
    ),

    10: LoraConfig(
        description="Attention QKVO | r=64 | very high rank, short run",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=64, lora_alpha=128,
        num_epochs=2, learning_rate=5e-5,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning — select one of 10 pre-defined configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=int, choices=list(CONFIGS.keys()),
                   help="Configuration index 1-10.")
    p.add_argument("--list_configs", action="store_true",
                   help="Print all configurations and exit.")
    p.add_argument("--dataset_dir",  default=None)
    p.add_argument("--weights_file", default=None)
    p.add_argument("--output_dir",   default=None)
    return p.parse_args()


def list_configs():
    print("\n  Available QLoRA configurations")
    print("  " + "─" * 70)
    for idx, cfg in CONFIGS.items():
        layers = (
            f"layers {cfg.layers_to_transform[0]}-{cfg.layers_to_transform[-1]}"
            if cfg.layers_to_transform else "all layers"
        )
        print(f"\n  Config {idx:2d} — {cfg.description}")
        print(f"    target_modules : {cfg.target_modules}")
        print(f"    r / alpha      : {cfg.r} / {cfg.lora_alpha}")
        print(f"    dropout        : {cfg.lora_dropout}")
        print(f"    epochs / lr    : {cfg.num_epochs} / {cfg.learning_rate}")
        print(f"    layers         : {layers}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    log = logging.getLogger("qwen_finetune")
    log.info(f"Log file: {log_file}")
    return log


# ══════════════════════════════════════════════════════════════════════════════
#  GPU SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def gpu_summary(log):
    import torch
    n = torch.cuda.device_count()
    log.info(f"GPUs: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        log.info(f"  GPU {i}: {p.name}  total={total/1e9:.1f}GB  free={free/1e9:.1f}GB")
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHT SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def smooth_weights(raw_weights: dict, method: str, log) -> dict:
    import math as _math
    smoothed = {}
    log.info(f"\nClass weight smoothing (method={method}):")
    log.info(f"  {'Labelset':<50} {'raw':>8}  {'smoothed':>10}")
    for key, w in raw_weights.items():
        if method == "log":
            s = round(_math.log1p(w), 4)
        elif method == "sqrt":
            s = round(_math.sqrt(w), 4)
        else:
            s = w
        smoothed[key] = s
        log.info(f"  {key:<50} {w:>8.4f}  {s:>10.4f}")
    return smoothed


# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHTED LOSS TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def make_weighted_trainer_class(label_token_ids, class_weights, log):
    import torch
    from trl import SFTTrainer

    class WeightedLossTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels   = inputs.get("labels")
            outputs  = model(**inputs)
            logits   = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            weight_mask = torch.ones_like(token_losses)
            for labelset_key, token_ids in label_token_ids.items():
                w = class_weights.get(labelset_key, 1.0)
                for tid in token_ids:
                    weight_mask[shift_labels == tid] = w
            non_pad = (shift_labels != -100).float()
            weighted_loss = (
                (token_losses * weight_mask * non_pad).sum()
                / non_pad.sum().clamp(min=1)
            )
            return (weighted_loss, outputs) if return_outputs else weighted_loss

    return WeightedLossTrainer


def get_label_token_ids(tokenizer, class_weights, log):
    label_token_ids = {}
    log.info("\nLabel token ID mapping:")
    for key in class_weights.keys():
        ids    = tokenizer.encode(key, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        label_token_ids[key] = ids
        log.info(f"  {key:<50} → {tokens}")
    return label_token_ids


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_dir: str, log):
    from datasets import load_from_disk
    log.info(f"\nLoading dataset from {dataset_dir} ...")
    ds = load_from_disk(dataset_dir)
    log.info(f"  Train : {len(ds['train'])} | Val : {len(ds['val'])} | Test : {len(ds['test'])}")
    sample = ds["train"][0]
    log.info(f"  Sample id={sample['id']}  key={sample['labelset_key']}  w={sample['class_weight']}")
    log.info(f"  text[:200]: {sample['text'][:200]}")
    return ds


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL + TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

def load_model_tokenizer(log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = SHARED["model_id"]
    bnb_cfg  = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    log.info(f"\nLoading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    log.info(f"Loading model (4-bit NF4): {model_id}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    log.info(f"  Loaded in {time.time()-t0:.1f}s")
    model.config.use_cache      = False
    model.config.pretraining_tp = 1
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"  Parameters: {n_params:.2f}B")
    for i in range(torch.cuda.device_count()):
        log.info(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f} GB allocated")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  LORA INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def inject_lora(model, cfg: LoraConfig, log):
    from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    lora_kwargs = dict(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.target_modules,
    )
    if cfg.layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = cfg.layers_to_transform

    model = get_peft_model(model, PeftLoraConfig(**lora_kwargs))

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"\nLoRA injected:")
    log.info(f"  r={cfg.r}  alpha={cfg.lora_alpha}  dropout={cfg.lora_dropout}")
    log.info(f"  target_modules={cfg.target_modules}")
    if cfg.layers_to_transform:
        log.info(f"  layers_to_transform: {cfg.layers_to_transform[0]}…{cfg.layers_to_transform[-1]}")
    log.info(f"  Trainable: {trainable/1e6:.2f}M / {total/1e9:.2f}B  ({100*trainable/total:.4f}%)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def run_training(model, tokenizer, ds, class_weights, label_token_ids,
                 cfg: LoraConfig, output_dir: Path, log):
    from trl import SFTConfig
    from transformers import EarlyStoppingCallback

    WeightedLossTrainer = make_weighted_trainer_class(label_token_ids, class_weights, log)

    sft_cfg = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=SHARED["batch_size"],
        per_device_eval_batch_size=SHARED["batch_size"],
        gradient_accumulation_steps=SHARED["grad_accum"],
        max_length=SHARED["max_length"],
        dataset_text_field="text",
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=SHARED["warmup_steps"],
        weight_decay=SHARED["weight_decay"],
        max_grad_norm=SHARED["max_grad_norm"],
        eval_strategy="steps",
        eval_steps=SHARED["eval_steps"],
        eval_on_start=True,
        save_strategy="steps",
        save_steps=SHARED["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        logging_first_step=True,
        report_to=SHARED["report_to"],
        seed=SHARED["seed"],
        data_seed=SHARED["seed"],
        remove_unused_columns=True,
        dataloader_num_workers=2,
        packing=False,
    )

    train_ds = ds["train"].rename_column("labels", "label_names")
    val_ds   = ds["val"].rename_column("labels", "label_names")

    trainer = WeightedLossTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=SHARED["early_stopping_patience"],
            early_stopping_threshold=0.001,
        )],
    )

    import torch
    n_gpus = torch.cuda.device_count()
    effective_batch = SHARED["batch_size"] * SHARED["grad_accum"] * n_gpus
    log.info("\n" + "═" * 60)
    log.info("  TRAINING START")
    log.info(f"  Epochs         : {cfg.num_epochs}")
    log.info(f"  LR             : {cfg.learning_rate}")
    log.info(f"  Effective batch: {SHARED['batch_size']} x {SHARED['grad_accum']} x {n_gpus} GPUs = {effective_batch}")
    log.info("═" * 60)

    t0           = time.time()
    train_result = trainer.train()
    elapsed      = time.time() - t0

    log.info(f"\nCompleted in {elapsed/60:.1f} min | loss={train_result.training_loss:.6f} | steps={train_result.global_step}")
    for i in range(torch.cuda.device_count()):
        log.info(f"  GPU {i} peak VRAM: {torch.cuda.max_memory_allocated(i)/1e9:.2f} GB")

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    log.info(f"  History → {history_path}")
    return trainer


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_adapter(trainer, output_dir: Path, log):
    adapter_path = output_dir / "final_adapter"
    log.info(f"\nSaving adapter → {adapter_path}")
    trainer.model.save_pretrained(str(adapter_path))
    trainer.processing_class.save_pretrained(str(adapter_path))
    return str(adapter_path)


def save_run_summary(config_idx, cfg, adapter_path, trainer, output_dir, log):
    history      = trainer.state.log_history
    val_entries  = [h for h in history if "eval_loss" in h]
    train_losses = [h["loss"] for h in history if "loss" in h and "eval_loss" not in h]
    best_val     = min((h["eval_loss"] for h in val_entries), default=None)
    final_val    = val_entries[-1]["eval_loss"] if val_entries else None

    summary = {
        "run_date": datetime.now().isoformat(),
        "config_index": config_idx,
        "config_description": cfg.description,
        **SHARED,
        "target_modules": cfg.target_modules,
        "r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "layers_to_transform": cfg.layers_to_transform,
        "num_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "adapter_path": adapter_path,
        "total_steps": trainer.state.global_step,
        "best_val_loss": best_val,
        "final_val_loss": final_val,
        "final_train_loss": train_losses[-1] if train_losses else None,
    }
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary → {summary_path}  |  best_val={best_val}  final_val={final_val}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.list_configs:
        list_configs()
        sys.exit(0)

    if not all([args.config, args.dataset_dir, args.weights_file, args.output_dir]):
        print("Error: --config, --dataset_dir, --weights_file, --output_dir are all required.")
        print("Run with --list_configs to see available configurations.")
        sys.exit(1)

    cfg = CONFIGS[args.config]
    out = Path(args.output_dir)
    log = setup_logging(out)

    log.info("═" * 60)
    log.info(f"  Config {args.config} — {cfg.description}")
    log.info(f"  Dataset : {args.dataset_dir}")
    log.info(f"  Output  : {args.output_dir}")
    log.info("═" * 60)

    # Save full config snapshot immediately
    with open(out / "config_snapshot.json", "w") as f:
        json.dump({
            "config_index": args.config,
            "config_description": cfg.description,
            "target_modules": cfg.target_modules,
            "r": cfg.r, "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "layers_to_transform": cfg.layers_to_transform,
            "num_epochs": cfg.num_epochs,
            "learning_rate": cfg.learning_rate,
            **SHARED,
        }, f, indent=2)

    gpu_summary(log)

    with open(args.weights_file) as f:
        weights_data = json.load(f)
    class_weights = smooth_weights(weights_data["weights"], SHARED["weight_smoothing"], log)

    ds              = load_dataset(args.dataset_dir, log)
    model, tokenizer = load_model_tokenizer(log)
    label_token_ids  = get_label_token_ids(tokenizer, class_weights, log)
    model            = inject_lora(model, cfg, log)

    try:
        trainer = run_training(model, tokenizer, ds, class_weights,
                               label_token_ids, cfg, out, log)
    except Exception:
        log.error("Training failed:")
        log.error(traceback.format_exc())
        sys.exit(1)

    adapter_path = save_adapter(trainer, out, log)
    save_run_summary(args.config, cfg, adapter_path, trainer, out, log)

    log.info("\n" + "═" * 60)
    log.info(f"  CONFIG {args.config} COMPLETE — {cfg.description}")
    log.info(f"  Adapter : {adapter_path}")
    log.info(f"  Evaluate: python finetuning/qwen_evaluation.py --adapter_paths {adapter_path} --dataset_dir <hf_dataset> --prompts dataset")
    log.info("═" * 60)


if __name__ == "__main__":
    main()