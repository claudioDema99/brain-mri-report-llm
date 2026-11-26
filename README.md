# Open-weight LLMs for brain MRI report classification

Code for the paper:

> Demaria C, Crema C, Verdicchio M, Brancato V, Alfano V, Basso L, Pontillo G, Archetti D, Fostinelli S, De Francesco S, Pievani M, Baronio CM, Boccali A, Toschi N, Brunetti A, Redolfi A, Aiello M.
> **Comparative Evaluation of Open-Weight Large Language Models for Pathological Brain MRI Reports Classification: From Prompts to a Fine-Tuned Model.**
> *Academic Radiology*, 2026. [DOI: 10.1016/j.acra.2026.09.053](https://doi.org/10.1016/j.acra.2026.09.053) · [PubMed: 42875268](https://pubmed.ncbi.nlm.nih.gov/42875268/)

Results, figures and discussion are in the paper. This repository contains the code that produced them.

## What the study does

Brain MRI reports written in Italian are classified in two tasks:

| Task | Labels |
|---|---|
| Binary | `positivo` (pathological) / `negativo` |
| Multiclass (multi-label, pathological reports only) | `Neoformazione`, `Vascolare`, `Infiammazione o Infezione`, `Neurodegenerativo`, `Altro` |

Two approaches are compared:

1. **Prompting** four open-weight models served with vLLM, with four prompt versions (A to D) and 0, 5, 10 or 20 few-shot examples:
   `openai/gpt-oss-20b`, `mistralai/Magistral-Small-2509`, `google/gemma-4-31B-it`, `Qwen/Qwen3.5-35B-A3B`.
2. **QLoRA fine-tuning** of `Qwen/Qwen3.5-35B-A3B` on the multiclass task, with 10 LoRA configurations (target modules, rank, layers, learning rate).

## Repository layout

| Path | Contents |
|---|---|
| `prompting/` | Zero- and few-shot pipeline (`python -m prompting.main`): builds prompts and few-shot blocks, calls the vLLM endpoints, parses the JSON answers, saves per-report logs and per-set metrics |
| `prompts/binary/`, `prompts/multiclass/` | The four prompt versions per task (system and user templates, with and without examples) |
| `data/shot_sets_*.json` | IDs of the reports used as few-shot examples: 11 sets of 5, 5 sets of 10, 3 sets of 20, plus zero-shot. They are excluded from the evaluated reports. |
| `data/split_index.json` | Train / validation / test IDs of the fine-tuning split (922 / 117 / 114), seed and forced IDs |
| `data/example_reports.json` | Three invented reports in the expected input format |
| `finetuning/` | `filter_reports.py` → `label_powerset.py` → `training_data.py` → `qwen_finetune.py` → `qwen_evaluation.py` |
| `analysis/` | Metrics, figures and statistical tests (`binary_analysis.py`, `multiclass_analysis.py`, `peft_analysis.py`) and label co-occurrence statistics |
| `scripts/` | Shell entry points for the three stages |

## Data

The reports are clinical data and are not distributed. The pipelines read a [Label Studio](https://labelstud.io/) JSON export with this structure (see `data/example_reports.json`):

| Field | Meaning |
|---|---|
| `id` | task id, used by the shot sets and the split |
| `data.TestoRefertoStringa` | report text |
| `data.IdRefertante` | annotator id |
| `annotations[].result[]` with `from_name`: | the most recent annotation is used |
| `district_type` | `Encefalo` / `Non Encefalo`; only brain reports are used |
| `status_class` | `Positivo` / `Negativo` (binary ground truth) |
| `category_class` | one or more of the five multiclass labels |
| `confidence_rating` | annotator confidence, 1 to 5 |
| `label_vasc`, `label-neo`, `label_inf`, `label_neur`, `label_altro` | highlighted evidence span per label, shown in multiclass few-shot examples |

The binary task runs on all brain reports. The multiclass task and the fine-tuning use the brain reports not labelled `Negativo` (`finetuning/filter_reports.py`).

## Setup

Python 3.11 or newer. Each stage has its own requirements:

```bash
pip install -r requirements-prompting.txt    # prompting
pip install -r requirements-finetuning.txt   # QLoRA, needs CUDA GPUs
pip install -r requirements-analysis.txt     # metrics and figures
```

## Running

### 1. Prompting

Serve each model with vLLM (OpenAI-compatible API) and point the pipeline to it:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1          # same server for every model
# or one per model: VLLM_BASE_URL_GPT, VLLM_BASE_URL_MISTRAL, VLLM_BASE_URL_GEMMA, VLLM_BASE_URL_QWEN
export VLLM_API_KEY=...                                # only if the server requires one

# one model, one prompt, all shot sets
python -m prompting.main reports.json binary qwen A
# every prompt in parallel
scripts/run_prompting.sh reports.json multiclass gemma
```

Decoding is greedy (`temperature=0`). Outputs go to `results_<task>/<model>/prompt_<X>/<N>-shot(s)/set_<K>/{logs,results}/`.

Quick check on the example data (zero-shot only, since the shot-set IDs refer to the original dataset):

```bash
python -m prompting.main data/example_reports.json multiclass qwen D --active-sets none
```

### 2. Fine-tuning

```bash
scripts/run_finetuning.sh reports.json ./work
```

The script filters the reports, rebuilds the train / validation / test split, trains the 10 LoRA configurations (`python finetuning/qwen_finetune.py --list_configs` lists them) and evaluates each adapter on the test split. The split uses seed 42 and the forced IDs stored in `data/split_index.json`. Run on the original export, the preparation steps reproduce the paper's training data exactly: same split, same training texts, same weights.

Training texts follow the Qwen3.5 chat template with thinking disabled: the system prompt in `training_data.py`, the report as the user turn, and `{"labels": ..., "confidence": ...}` as the answer. The base model is loaded in 4-bit (bitsandbytes NF4). The loss is weighted by label combination, with balanced weights computed on the training split and log-smoothed in `qwen_finetune.py`.

### 3. Analysis

```bash
scripts/run_analysis.sh
```

This writes the metrics tables, figures and statistical tests (McNemar test and bootstrap confidence intervals for the fine-tuned model against the prompting baselines) to `figures/`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT, see [LICENSE](LICENSE). The model weights are not part of this repository and keep their own licenses.
