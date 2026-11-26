import argparse
import json
import sys

from prompting.models import set_model_and_prompt
from prompting.pipeline_binary import run_binary
from prompting.pipeline_multiclass import run_multiclass

# ── Argument parsing ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prompting.main",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Brain MRI Report Classifier
============================
Runs an LLM-based classification pipeline over brain MRI reports.

Required Arguments:
  1. JSON_FILE   : Path to the Label Studio export.
  2. TASK        : 'binary' or 'multiclass'.
  3. MODEL       : gpt, mistral, gemma, qwen, or all (one process per model).
  4. PROMPT      : Prompt version to use (A, B, C, or D).

Examples
--------
  # Basic usage with mandatory arguments:
  python -m prompting.main data/reports.json binary gemma B

  # Usage with optional shot sets:
  python -m prompting.main data/reports.json binary gpt A --active-sets B5_1 B5_2
""",
    )

    # --- Mandatory Positional Arguments ---
    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help="Path to the Label Studio JSON export file.",
    )
    parser.add_argument(
        "task",
        metavar="TASK",
        choices=["binary", "multiclass"],
        help="Classification task: 'binary' or 'multiclass'.",
    )
    parser.add_argument(
        "model",
        metavar="MODEL",
        choices=["gpt", "mistral", "gemma", "qwen", "all"],
        help="Model to use. Choices: %(choices)s",
    )
    parser.add_argument(
        "prompt",
        metavar="PROMPT",
        choices=["A", "B", "C", "D"],
        help="Prompt version. Choices: %(choices)s",
    )

    # --- Optional Arguments ---
    parser.add_argument(
        "--shot-sets",
        metavar="FILE",
        default=None,
        help=(
            "Path to the JSON file defining few-shot ID sets. "
            "Defaults to shot_sets_binary.json or shot_sets_multiclass.json based on task."
        ),
    )
    parser.add_argument(
        "--active-sets",
        metavar="SET_NAME",
        nargs="+",
        default=None,
        help=(
            "One or more set names to use (e.g., set_1 set_2). "
            "If omitted, ALL sets in the file are used."
        ),
    )

    parser.add_argument(
        "--max-concurrency",
        metavar="MAX_CONCURRENCY",
        type=int,
        default=8,
        help="Maximum number of concurrent requests to the model server.",
    )

    return parser

# ── Shot-set loader ───────────────────────────────────────────────────────────

def load_shot_sets(filepath: str, active_keys: list[str] | None = None) -> list[list[int]]:
    """
    Load few-shot ID sets from a JSON file.

    Args:
        filepath    (str)            : path to the shot-sets JSON file
        active_keys (list[str]|None) : set names to load; if None, all sets are loaded

    Returns:
        list[list[int]]: ordered list of active shot sets (each a list of IDs)

    Raises:
        SystemExit: if the file is missing, malformed, or references unknown set names
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: shot-sets file '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: '{filepath}' is not valid JSON — {e}")
        sys.exit(1)

    all_sets = config.get("sets", {})

    # Fall back to all sets if caller did not specify any
    if active_keys is None:
        active_keys = list(all_sets.keys())

    missing = [k for k in active_keys if k not in all_sets]
    if missing:
        print(
            f"Error: --active-sets references unknown set name(s): {missing}\n"
            f"Available sets: {list(all_sets.keys())}"
        )
        sys.exit(1)

    active = [all_sets[k] for k in active_keys]
    print(f"Loaded {len(active)} active shot set(s) from '{filepath}':")
    for k, s in zip(active_keys, active):
        print(f"  {k}: {len(s)} IDs → {s}")
    return active

# ── Entry point ───────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ["gemma", "gpt", "qwen", "mistral"]

def main(args):
    """Parse data and shot sets, then dispatch to the appropriate pipeline."""

    shot_sets_file = args.shot_sets or (
        "data/shot_sets_binary.json" if args.task == "binary"
        else "data/shot_sets_multiclass.json"
    )

    set_model_and_prompt(args.model, args.task, args.prompt)

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file '{args.json_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{args.json_file}' is not valid JSON.")
        sys.exit(1)

    # ── Load shot sets ────────────────────────────────────────────────────────
    sets = load_shot_sets(shot_sets_file, active_keys=args.active_sets)
    print(f"Data length: {len(data)} — New IDs to process: {len(data)}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.task == "multiclass":
        run_multiclass(
            args=args,
            data=data,
            sets=sets,
            max_concurrency=args.max_concurrency,
            batch_size=getattr(args, "batch_size", 256),
        )
    elif args.task == "binary":
        run_binary(args=args, data=data, sets=sets)

def run_single_model(model_name: str, args_namespace):
    """Run the whole pipeline for one model, prefixing its output with the model name."""
    import copy
    args = copy.deepcopy(args_namespace)
    args.model = model_name
    
    import sys
    import io

    class PrefixedWriter(io.TextIOBase):
        def __init__(self, prefix, original):
            self.prefix = prefix
            self.original = original
            self._buf = ""

        def write(self, s):
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self.original.write(f"[{self.prefix}] {line}\n")
            return len(s)

        def flush(self):
            if self._buf:
                self.original.write(f"[{self.prefix}] {self._buf}")
                self._buf = ""
            self.original.flush()

    sys.stdout = PrefixedWriter(model_name.upper(), sys.__stdout__)

    try:
        main(args)
    finally:
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # ── "--model all": one process per model ────────────────────────────────────────────────
    if args.model == "all":
        import multiprocessing as mp

        print(f"Mode 'all': starting {len(SUPPORTED_MODELS)} parallel processes.")
        print(f"Models: {', '.join(SUPPORTED_MODELS)}\n")

        processes = []
        for model_name in SUPPORTED_MODELS:
            p = mp.Process(
                target=run_single_model,
                args=(model_name, args),
                name=f"worker-{model_name}",
            )
            p.start()
            processes.append((model_name, p))
            print(f"Started [{model_name}], PID {p.pid}")

        print()

        exit_codes = {}
        for model_name, p in processes:
            p.join()
            exit_codes[model_name] = p.exitcode

        print("\n" + "=" * 50)
        print("PARALLEL RUN SUMMARY")
        print("=" * 50)
        for model_name, code in exit_codes.items():
            status = "✓ OK" if code == 0 else f"✗ FAILED (exit code {code})"
            print(f"  [{model_name:10s}] {status}")

        if any(c != 0 for c in exit_codes.values()):
            sys.exit(1)

    else:
        if args.model not in SUPPORTED_MODELS:
            print(f"Error: model '{args.model}' is not supported. Choose from: {SUPPORTED_MODELS + ['all']}")
            sys.exit(1)

        main(args)