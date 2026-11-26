import asyncio
import json
import os
import time
from datetime import datetime

from prompting.models import call_model_multiclass
from prompting.utils import (
    LogEntry,
    save_logs_as_json,
    filter_reports,
    create_shots_multiclass,
    get_latest_annotation,
    calculate_statistics_multiclass,
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


# ── Multiclass pipeline ───────────────────────────────────────────────────────

def run_multiclass(
    args,
    data: list,
    sets: list,
    max_concurrency: int = 16,
    batch_size: int = 256,
) -> None:
    """
    Run the full multiclass classification pipeline.

    Filters Encefalo+Patologico reports, runs async LLM inference for every
    (report, shot-set) combination, then saves logs and statistics.

    Args:
        args           : parsed argparse.Namespace (carries model, prompt, etc.)
        data           : full list of Label Studio task dicts
        sets           : list of shot-ID lists, one per set
        max_concurrency: max parallel API calls (semaphore bound)
        batch_size     : reports per asyncio.gather batch
    """

    # ── 1) Filter Encefalo ────────────────────────────────────────────────────
    reports = []
    for report in data:
        try:
            results = get_latest_annotation(report)["result"]
        except (ValueError, KeyError, TypeError):
            continue

        district_dict = next(
            (r for r in results if r.get("from_name") == "district_type"),
            None
        )
        if district_dict and district_dict.get("value", {}).get("choices", [None])[0] == "Encefalo":
            reports.append(report)

    print(f"Brain (Encefalo) reports: {len(reports)}")

    # ── 2) Filter Patologico (exclude Negativo) ───────────────────────────────
    positive_reports = []
    for report in reports:
        try:
            results = get_latest_annotation(report)["result"]
        except (ValueError, KeyError, TypeError):
            continue

        status_dict = next(
            (r for r in results if r.get("from_name") == "status_class"),
            None
        )
        if status_dict and status_dict.get("value", {}).get("choices", [None])[0] == "Negativo":
            continue

        positive_reports.append(report)

    print(f"Brain reports labelled pathological: {len(positive_reports)}")

    filtered_reports = filter_reports(data=positive_reports, id_sets=sets)
    print(f"Reports to classify (few-shot examples excluded): {len(filtered_reports)}")
    
    start_time = time.time()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 3) Build per-set output paths ─────────────────────────────────────────
    set_configs = []
    for set_idx, shot_set in enumerate(sets):
        num_shots   = len(shot_set)
        shot_folder = f"{num_shots}-shot" if num_shots == 0 else f"{num_shots}-shots"
        base_path   = f"./results_multiclass/{args.model}/prompt_{args.prompt}/{shot_folder}/set_{set_idx + 1}"
        logs_dir    = f"{base_path}/logs"
        results_dir = f"{base_path}/results"
        os.makedirs(logs_dir,    exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        set_configs.append({
            "set_idx":        set_idx,
            "shot_set":       shot_set,
            "num_shots":      num_shots,
            "logs_filepath":  f"{logs_dir}/logs_multiclass_{timestamp}.json",
            "stats_filepath": f"{results_dir}/results_multiclass_{timestamp}.json",
        })

    # ── 4) Cache shots by exact shot-set content ──────────────────────────────
    shot_cache = {}
    for cfg in set_configs:
        key = tuple(cfg["shot_set"])
        if key not in shot_cache:
            shot_cache[key] = create_shots_multiclass(data, cfg["shot_set"], include_evidence=True)

    LABEL_FROM_NAMES = {"label_vasc", "label_altro", "label_neur", "label_inf", "label-neo"}

    # ── 5) Pre-extract report metadata ───────────────────────────────────────
    report_records = []
    for report in filtered_reports:
        report_id         = report.get("id")
        report_text       = report.get("data", {}).get("TestoRefertoStringa")
        raw_id_refertante = report.get("data", {}).get("IdRefertante")

        try:
            id_refertante = _safe_int(raw_id_refertante)
            if id_refertante is None:
                raise ValueError

            results = get_latest_annotation(report)["result"]

            category_dict = next(
                (r for r in results if r.get("from_name") == "category_class"),
                None
            )
            if category_dict is None:
                raise KeyError

            ground_truth = ", ".join(category_dict["value"]["choices"]).strip()
            if not ground_truth:
                raise ValueError

            confidence_dict = next(
                (r for r in results if r.get("from_name") == "confidence_rating"),
                None
            )
            if confidence_dict is None:
                raise KeyError

            confidence = int(confidence_dict["value"]["choices"][0])

            if not report_text:
                raise ValueError

            label_evidence = {
                r["value"]["labels"][0]: r["value"]["text"]
                for r in results
                if r.get("from_name") in LABEL_FROM_NAMES
                and r.get("type") == "labels"
            }

            report_records.append({
                "ok":           True,
                "report_id":    report_id,
                "report_text":  report_text,
                "id_refertante": id_refertante,
                "ground_truth": ground_truth,
                "confidence":   confidence,
                "label_evidence": label_evidence,
            })

        except Exception as e:
            report_records.append({
                "ok":           False,
                "report_id":    report_id,
                "report_text":  report_text,
                "id_refertante": _safe_int(raw_id_refertante),
                "error":        str(type(e).__name__),
            })

    # ── 6) Prepare logs container ─────────────────────────────────────────────
    logs_by_set = {
        cfg["set_idx"]: [None] * len(report_records)
        for cfg in set_configs
    }

    # Fill failed-metadata entries immediately
    for cfg in set_configs:
        for i, record in enumerate(report_records):
            if record["ok"]:
                continue
            logs_by_set[cfg["set_idx"]][i] = LogEntry(
                id=record["report_id"],
                report=record["report_text"],
                true_label=None,
                confidence=None,
                id_refertante=record["id_refertante"],
                status="failed",
                task="multiclass",
                set_id=cfg["set_idx"] + 1,
                shot_ids=cfg["shot_set"],
                num_shot=cfg["num_shots"],
                prompt=args.prompt,
                error=record["error"],
                created_at=time.strftime("%d-%m-%Y %H:%M"),
            ).to_dict()

    # ── 7) Build valid-task list ──────────────────────────────────────────────
    valid_tasks = []
    for cfg in set_configs:
        for i, record in enumerate(report_records):
            if record["ok"]:
                valid_tasks.append((cfg["set_idx"], i, record))

    # ── 8) Async inference ────────────────────────────────────────────────────
    semaphore = asyncio.Semaphore(max_concurrency)

    async def infer_one(set_idx, report_idx, record):
        cfg   = set_configs[set_idx]
        shots = shot_cache[tuple(cfg["shot_set"])]

        # Pre-initialise all variables so LogEntry construction is always safe,
        # even if an exception is raised at any point inside call_model_multiclass.
        json_response    = None
        response         = None
        model_motivation = None
        model_confidence = None
        pred             = None
        status           = "failed"
        err              = None

        async with semaphore:
            try:
                json_response, _, response = await asyncio.to_thread(
                    call_model_multiclass,
                    shots=shots,
                    report=record["report_text"],
                    model=args.model,
                )

                if json_response is None:
                    err = "Model returned None"
                else:
                    status           = "success"
                    pred             = json_response.get("labels")
                    model_motivation = json_response.get("reasoning")
                    model_confidence = json_response.get("confidence")

            except Exception as e:
                err = str(e)

        log = LogEntry(
            id=record["report_id"],
            report=record["report_text"],
            true_label=record["ground_truth"],
            label_evidence=record["label_evidence"],
            confidence=record["confidence"],
            id_refertante=record["id_refertante"],
            status=status,
            task="multiclass",
            set_id=set_idx + 1,
            shot_ids=cfg["shot_set"],
            num_shot=cfg["num_shots"],
            predicted_label=pred,
            model_confidence=model_confidence,
            model_motivation=model_motivation,
            response=response,
            error=err,
            prompt=args.prompt,
            created_at=time.strftime("%d-%m-%Y %H:%M"),
        )

        if status == "success":
            print(f"[SET {set_idx + 1}] ID: {record['report_id']} | LLM: {pred}")
        else:
            print(f"[SET {set_idx + 1}] SKIP/FAIL - ID: {record['report_id']} | {err}")

        return set_idx, report_idx, log.to_dict()

    async def run():
        for batch in _chunked(valid_tasks, batch_size):
            results = await asyncio.gather(
                *(infer_one(s, i, r) for s, i, r in batch)
            )
            for s, i, log in results:
                logs_by_set[s][i] = log

    asyncio.run(run())

    # ── 9) Save logs and statistics per set ───────────────────────────────────
    for cfg in set_configs:
        logs = [x for x in logs_by_set[cfg["set_idx"]] if x is not None]

        save_logs_as_json(logs, cfg["logs_filepath"])

        stats = calculate_statistics_multiclass(logs)

        evaluated = sum(1 for entry in logs if entry.get("status") == "success")
        failed    = len(logs) - evaluated
        stats["evaluated"] = evaluated
        stats["failed"]    = failed

        with open(cfg["stats_filepath"], "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Done: {cfg['num_shots']}-shot")
        print(f"Logs:  {cfg['logs_filepath']}")
        print(f"Stats: {cfg['stats_filepath']}")

    print(f"\nTotal time: {time.time() - start_time:.2f} s")
