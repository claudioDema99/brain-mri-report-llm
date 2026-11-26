import asyncio
import json
import os
import time
from datetime import datetime

from prompting.models import call_model_binary
from prompting.utils import (
    LogEntry,
    save_logs_as_json,
    filter_reports,
    create_shots,
    get_latest_annotation,
    calculate_statistics_full,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int(value):
    """Convert value to int, returning None on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunked(seq, size):
    """Yield successive fixed-size chunks from seq."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ── Binary pipeline ───────────────────────────────────────────────────────────

def run_binary(args, data: list, sets: list) -> None:
    """
    Run the full binary classification pipeline.

    Filters Encefalo reports, runs async LLM inference for every (report, shot-set)
    combination, then saves logs and statistics to the expected output paths.

    Args:
        args : parsed argparse.Namespace (carries model, prompt, etc.)
        data : full list of Label Studio task dicts
        sets : list of shot-ID lists, one per set
    """

    # ── 1) Filter only "Encefalo" reports ────────────────────────────────────
    reports = []
    for r in data:
        try:
            results = get_latest_annotation(r)["result"]
            district_dict = next(
                (x for x in results if x.get("from_name") == "district_type"),
                None
            )
            if district_dict and district_dict["value"]["choices"][0] == "Encefalo":
                reports.append(r)
        except (ValueError, KeyError, IndexError, TypeError):
            continue

    print(f"Brain (Encefalo) reports: {len(reports)}")

    filtered_reports = filter_reports(data=reports, id_sets=sets)
    print(f"Reports to classify (few-shot examples excluded): {len(filtered_reports)}")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    max_concurrency = max(1, int(getattr(args, "max_concurrency", 16)))
    batch_size      = max(1, int(getattr(args, "batch_size", 256)))

    # ── 2) Build per-set output paths ─────────────────────────────────────────
    set_configs = []
    for set_idx, shot_set in enumerate(sets):
        num_shots   = len(shot_set)
        shot_folder = f"{num_shots}-shot" if num_shots == 0 else f"{num_shots}-shots"
        base_path   = f"./results_binary/{args.model}/prompt_{args.prompt}/{shot_folder}/set_{set_idx + 1}"
        logs_dir    = f"{base_path}/logs"
        results_dir = f"{base_path}/results"
        os.makedirs(logs_dir,    exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        set_configs.append({
            "set_idx":        set_idx,
            "shot_set":       shot_set,
            "num_shots":      num_shots,
            "shot_folder":    shot_folder,
            "logs_filepath":  f"{logs_dir}/logs_binary_{timestamp}.json",
            "stats_filepath": f"{results_dir}/results_binary_{timestamp}.json",
        })

    # ── 3) Cache shots by exact shot-set content ──────────────────────────────
    shot_cache = {}
    for cfg in set_configs:
        shot_key = tuple(cfg["shot_set"])
        if shot_key not in shot_cache:
            shot_cache[shot_key] = create_shots(data, cfg["shot_set"])

    # ── 4) Pre-extract report metadata ───────────────────────────────────────
    report_records = []
    for report in filtered_reports:
        report_id         = report.get("id")
        report_text       = report.get("data", {}).get("TestoRefertoStringa")
        raw_id_refertante = report.get("data", {}).get("IdRefertante")

        try:
            id_refertante = _safe_int(raw_id_refertante)
            if id_refertante is None:
                raise ValueError("IdRefertante non valido")

            results = get_latest_annotation(report)["result"]

            status_dict = next(
                (x for x in results if x.get("from_name") == "status_class"),
                None
            )
            if status_dict is None:
                raise KeyError("status_class not found")
            ground_truth_classes = str(status_dict["value"]["choices"][0]).lower()

            confidence_dict = next(
                (x for x in results if x.get("from_name") == "confidence_rating"),
                None
            )
            if confidence_dict is None:
                raise KeyError("confidence_rating not found")
            confidence = str(confidence_dict["value"]["choices"][0]).lower()

            report_records.append({
                "ok":                  True,
                "report_id":           report_id,
                "report_text":         report_text,
                "id_refertante":       id_refertante,
                "ground_truth_classes": ground_truth_classes,
                "confidence":          confidence,
            })

        except (KeyError, IndexError, TypeError, ValueError) as e:
            report_records.append({
                "ok":           False,
                "report_id":    report_id,
                "report_text":  report_text,
                "id_refertante": _safe_int(raw_id_refertante),
                "error":        type(e).__name__,
            })

    # ── 5) Prepare logs container ─────────────────────────────────────────────
    logs_by_set = {
        cfg["set_idx"]: [None] * len(report_records)
        for cfg in set_configs
    }

    # Fill failed-metadata entries immediately
    for cfg in set_configs:
        set_idx = cfg["set_idx"]
        for report_idx, record in enumerate(report_records):
            if record["ok"]:
                continue
            log = LogEntry(
                id=record["report_id"],
                report=record["report_text"],
                true_label=None,
                confidence=None,
                id_refertante=record["id_refertante"],
                status="failed",
                task="binary",
                set_id=set_idx + 1,
                shot_ids=cfg["shot_set"],
                num_shot=cfg["num_shots"],
                prompt=args.prompt,
                error=f"Campi mancanti ({record['error']})",
                created_at=time.strftime("%d-%m-%Y %H:%M"),
            )
            logs_by_set[set_idx][report_idx] = log.to_dict()

    # ── 6) Build valid-task list ──────────────────────────────────────────────
    valid_tasks = []
    for cfg in set_configs:
        set_idx = cfg["set_idx"]
        for report_idx, record in enumerate(report_records):
            if record["ok"]:
                valid_tasks.append((set_idx, report_idx, record))

    # ── 7) Async inference ────────────────────────────────────────────────────
    semaphore = asyncio.Semaphore(max_concurrency)

    async def infer_one(set_idx, report_idx, record):
        cfg      = set_configs[set_idx]
        shot_key = tuple(cfg["shot_set"])
        shots    = shot_cache[shot_key]

        # Pre-initialise all variables so LogEntry construction is always safe,
        # even if an exception is raised at any point inside call_model_binary.
        json_response    = None
        response         = None
        valutazione_ita  = None
        model_confidence = None
        model_motivation = None
        status_info      = "failed"
        error_msg        = None

        async with semaphore:
            try:
                json_response, _, response = await asyncio.to_thread(
                    call_model_binary,
                    shots=shots,
                    report=record["report_text"],
                    model=args.model,
                )

                if json_response is None:
                    error_msg = "Chiamata al modello non riuscita"
                else:
                    status_info      = "success"
                    valutazione      = json_response.get("classification").strip()
                    valutazione_ita  = "positivo" if valutazione == "positive" else "negativo"
                    model_confidence = json_response.get("confidence")
                    model_motivation = json_response.get("reasoning")

            except Exception as e:
                error_msg = f"Inference error ({type(e).__name__})"

        log = LogEntry(
            id=record["report_id"],
            report=record["report_text"],
            true_label=record["ground_truth_classes"],
            confidence=record["confidence"],
            id_refertante=record["id_refertante"],
            status=status_info,
            task="binary",
            set_id=set_idx + 1,
            shot_ids=cfg["shot_set"],
            num_shot=cfg["num_shots"],
            predicted_label=valutazione_ita,
            model_confidence=model_confidence,
            model_motivation=model_motivation,
            response=response,
            error=error_msg,
            prompt=args.prompt,
            created_at=time.strftime("%d-%m-%Y %H:%M"),
        )

        if status_info == "success":
            print(f"[SET {set_idx + 1}] ID: {record['report_id']} | LLM: {valutazione_ita}")
        else:
            print(f"[SET {set_idx + 1}] SKIP/FAIL - ID: {record['report_id']} | {error_msg}")

        return set_idx, report_idx, log.to_dict()

    async def run_all_valid_tasks():
        for batch in _chunked(valid_tasks, batch_size):
            batch_results = await asyncio.gather(
                *(infer_one(set_idx, report_idx, record)
                  for set_idx, report_idx, record in batch)
            )
            for set_idx, report_idx, log_dict in batch_results:
                logs_by_set[set_idx][report_idx] = log_dict

    asyncio.run(run_all_valid_tasks())

    # ── 8) Save logs and statistics per set ───────────────────────────────────
    for cfg in set_configs:
        set_idx = cfg["set_idx"]
        logs    = [entry for entry in logs_by_set[set_idx] if entry is not None]

        save_logs_as_json(logs, cfg["logs_filepath"])

        stats_binary = calculate_statistics_full(logs)

        evaluated = sum(1 for entry in logs if entry.get("status") == "success")
        failed    = len(logs) - evaluated

        for label_key in stats_binary:
            stats_binary[label_key]["evaluated"] = evaluated
            stats_binary[label_key]["failed"]    = failed
            assert evaluated + failed == stats_binary[label_key]["total"], (
                f"Counter mismatch in set {set_idx + 1}: "
                f"evaluated({evaluated}) + failed({failed}) "
                f"!= total({stats_binary[label_key]['total']})"
            )

        with open(cfg["stats_filepath"], "w", encoding="utf-8") as f:
            json.dump(stats_binary, f, indent=2, ensure_ascii=False)

        print(
            f"\nCompletato {cfg['num_shots']}-shots. "
            f"evaluated={evaluated}  failed={failed}  "
            f"Statistiche salvate in: {cfg['stats_filepath']}"
        )

    print(f"\nTotal time: {time.time() - start_time:.2f} s")
    print("\n" + "=" * 50)
    print("ALL RUNS COMPLETED")
    print("=" * 50)
