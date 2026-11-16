"""FP32 fallback scoring for a trained run whose pinned FP16 scoring overflowed.

NOT the scoring protocol. The pinned scorer casts the adapter model to FP16. For an
adapter whose activations exceed the FP16 range (65,504) every later layer becomes
NaN and the pinned scorer stops with "Non-finite answer likelihood". This script
scores such a run with the unquantized weights (LoRA, embeddings, norms, vision
tower, output head) in FP32 and the 4-bit base weights unchanged. Everything else
is the pinned procedure: the same 4-bit base, the same adapter, and the unmodified
score_full_answers() from paper_reproduction.py (same prompt, same five candidate
answers, prompt masking, total answer log likelihood, FP16 pixel values that the
vision patch embedding casts to its FP32 weights).

Every result records the precision used, the reason, the failed FP16 attempt and the
diagnosis, plus the same hashes and identity as score_cohort.py. Before the fallback,
the pinned FP16 path is tried on the first image so each result also carries its own
evidence that FP16 fails (or not) on that cohort.
"""

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import score_cohort as pinned  # noqa: E402  Hash-pinned by finished score jobs; imported, never edited.

NONFINITE = "Non-finite answer likelihood"
PRECISION = "FP32 fallback"
PRECISION_DETAIL = ("Unquantized weights (LoRA, embeddings, norms, vision tower, output head) in FP32 via "
                    "model.float(); 4-bit quantized base weights unchanged; pixel values cast to FP16 by the "
                    "pinned helper and to FP32 by the vision patch embedding.")


def failure_evidence(log_path):
    """The failed pinned FP16 attempt: its log must show the non-finite likelihood error."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if NONFINITE in line]
    if not lines:
        raise ValueError(f"{log_path} does not show a '{NONFINITE}' failure; the fallback does not apply.")
    return {"log": str(Path(log_path).resolve()), "log_sha256": pinned.sha(log_path), "error": lines[-1]}


def check_train_run(run, adapter_record, cohort_csv):
    """The adapter is the run's saved best adapter, and an APTOS cohort is the run's own test list."""
    run = Path(run).resolve()
    config = json.loads((run / "pilot_configuration.json").read_text(encoding="utf-8"))
    trained = json.loads((run / "train_result.json").read_text(encoding="utf-8"))
    if (run / "score_result.json").is_file():
        raise ValueError(f"{run} already has a protocol score; no fallback is needed.")
    saved = trained["adapter"]
    if (Path(saved["directory"]).resolve() != Path(adapter_record["directory"]).resolve()
            or saved["weights_sha256"] != adapter_record["weights_sha256"]
            or saved["configuration_sha256"] != adapter_record["configuration_sha256"]):
        raise ValueError("The adapter differs from the one the training run saved")
    record = {"run": str(run), "configuration_identity": config.get("identity"),
              "train_result_sha256": pinned.sha(run / "train_result.json"),
              "validation_loss": trained.get("validation_loss"), "global_step": trained.get("global_step"),
              "helper_sha256_at_training": config.get("source_sha256", {}).get("helper")}
    if Path(cohort_csv).resolve() == (run / "aptos_test.csv").resolve():
        if pinned.sha(cohort_csv) != config["split_sha256"]["test"]:
            raise ValueError("The run's test list changed")
        images = run / "input_images.csv"
        if pinned.sha(images) != config["input_images_sha256"]:
            raise ValueError("Saved image fingerprints changed")
        checked = 0
        with images.open(encoding="utf-8", newline="") as stream:
            import csv
            for row in csv.DictReader(stream):
                if row["split"] == "test":
                    if pinned.sha(row["image_path"]) != row["sha256"]:
                        raise ValueError(f"Image changed: {row['image_path']}")
                    checked += 1
        record["test_images_fingerprint_checked"] = checked
    return record


def precision_census(model, torch):
    """What the loaded model actually holds: parameter dtypes and the 4-bit layers' compute dtype."""
    dtypes = {}
    for _, parameter in model.named_parameters():
        key = str(parameter.dtype).replace("torch.", "")
        dtypes[key] = dtypes.get(key, 0) + parameter.numel()
    compute = set()
    for module in model.modules():
        if type(module).__name__ == "Linear4bit":
            compute.add(str(getattr(module, "compute_dtype", None)).replace("torch.", ""))
            state = getattr(module.weight, "quant_state", None)
            if state is not None and getattr(state, "dtype", None) is not None:
                compute.add("quant_state:" + str(state.dtype).replace("torch.", ""))
    lora = {str(p.dtype).replace("torch.", "") for name, p in model.named_parameters() if "lora_" in name}
    return {"parameter_elements_by_dtype": dtypes, "linear4bit_compute_dtypes": sorted(compute),
            "lora_dtypes": sorted(lora)}


def load_fp32(args, adapter, torch, vision_model):
    model, processor = vision_model.from_pretrained(model_name=str(args.base), load_in_4bit=True,
                                                    local_files_only=True)
    if not getattr(model, "is_loaded_in_4bit", False) or getattr(model, "peft_config", None):
        raise ValueError("Expected a fresh 4-bit base")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    vision_model.for_inference(model)
    model = model.float()  # the only difference from the pinned path, which calls model.half()
    model.eval()
    return model, processor


def probe_fp16(args, adapter, torch, vision_model, helper, row):
    """Try the pinned FP16 path (score_cohort.load_model) on one image and record what happens."""
    model, processor = pinned.load_model(
        argparse.Namespace(base=args.base, base_precision="4bit", offload_layers=0, tokenizer=None),
        adapter, torch, vision_model)
    record = {"image_id": row["id_code"], "loader": "score_cohort.load_model (4-bit base, adapter, model.half())"}
    try:
        scores = [float(v) for v in helper.score_full_answers(model, processor, row["image_path"])]
        record.update(outcome="finite", scores=scores)
    except ValueError as error:
        if NONFINITE not in str(error):
            raise
        record.update(outcome="non-finite answer likelihood (FP16 failure reproduced)")
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score",))
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, required=True, help="The training run whose FP16 score failed")
    parser.add_argument("--failure-log", type=Path, required=True, help="score.log of the failed FP16 attempt")
    parser.add_argument("--overflow-layer", required=True, help="Module whose FP16 output overflowed")
    parser.add_argument("--diagnosis-detail", default="", help="One line of diagnosis numbers")
    parser.add_argument("--diagnosis-evidence", type=Path, action="append", default=[])
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--cohort-name", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-fp16-probe", action="store_true")
    parser.add_argument("--diagnostic-predictions", type=Path,
                        help="Earlier diagnostic FP32 predictions for the same adapter and cohort, to compare")
    parser.add_argument("--helper", type=Path, default=pinned.HELPER)
    args = parser.parse_args()

    helper = pinned.load_helper(args.helper.resolve(strict=True))
    output = args.output_dir.resolve()
    train_run = args.train_run.resolve()
    if output == train_run or train_run in output.parents:
        raise ValueError("The fallback never writes into the training run folder")
    rows = pinned.cohort_rows(helper, args.cohort_csv, args.limit)
    adapter, adapter_record = pinned.adapter_files(helper, args.adapter)
    run_record = check_train_run(train_run, adapter_record, args.cohort_csv)
    failure = failure_evidence(args.failure_log)
    identity = {
        "name": args.name, "scorer_sha256": pinned.sha(SCRIPT), "score_cohort_sha256": pinned.sha(pinned.SCRIPT),
        "helper_sha256": pinned.sha(args.helper),
        "cohort": args.cohort_name, "cohort_csv": str(args.cohort_csv.resolve()),
        "cohort_csv_sha256": pinned.sha(args.cohort_csv), "images": len(rows), "limit": args.limit,
        "base": str(args.base.resolve()), "base_config_sha256": pinned.sha(args.base / "config.json"),
        "base_precision": "4bit", "offload_layers": 0, "adapter": adapter_record, "generation_images": 0,
        "scoring_precision": PRECISION, "protocol_scoring_precision": "FP16",
        "fallback_for": run_record["run"],
    }
    reason = {
        "summary": f"FP16 overflow: the output of {args.overflow_layer} exceeds the FP16 maximum (65,504), "
                   "so every later layer is NaN and the pinned FP16 scorer stops with a non-finite likelihood.",
        "overflow_layer": args.overflow_layer, "detail": args.diagnosis_detail,
        "failed_fp16_attempt": failure,
        "diagnosis_evidence": [{"path": str(path.resolve()), "sha256": pinned.sha(path)}
                               for path in args.diagnosis_evidence if path.is_file()],
    }
    summary_path = output / "summary.json"
    predictions_path = output / "predictions.csv"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError(f"{output} holds a different scoring job. Choose a fresh output folder.")
        if saved["predictions_sha256"] != pinned.sha(predictions_path):
            raise ValueError("Saved predictions changed.")
        print(f"Already scored: {output}", flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    helper.configure_process_caches(output, evaluation=True)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    scratch = output / "cache" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    os.environ.update(TEMP=str(scratch), TMP=str(scratch))
    import pandas  # noqa: F401  Load before Unsloth, as the pinned runner does on Windows.
    from unsloth import FastVisionModel
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Scoring requires a CUDA GPU.")
    probe = None
    if not args.skip_fp16_probe:
        probe = probe_fp16(args, adapter, torch, FastVisionModel, helper, rows[0])
        print(f"FP16 probe on {probe['image_id']}: {probe['outcome']}", flush=True)
        if probe["outcome"] == "finite":
            print("Note: FP16 was finite on the probe image. Scoring continues in FP32 so this run's "
                  "cohorts share one precision; the probe result is recorded.", flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, processor = load_fp32(args, adapter, torch, FastVisionModel)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    census = precision_census(model, torch)
    print(f"Loaded 4-bit base with adapter, unquantized weights in FP32, in {load_seconds:.1f} s. "
          f"Parameter elements by dtype: {census['parameter_elements_by_dtype']}. "
          f"Scoring {len(rows)} {args.cohort_name} images (FP32 FALLBACK, not the FP16 protocol).", flush=True)

    predictions, timings = [], []
    for index, row in enumerate(rows, 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        scores = [float(value) for value in helper.score_full_answers(model, processor, row["image_path"])]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        if len(scores) != 5 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores) or abs(sum(scores) - 1) > 1e-6:
            raise ValueError(f"Invalid five-grade scores for {row['id_code']}")
        predictions.append({"image_id": row["id_code"], "image_path": row["image_path"],
                            "y_true": int(row["label"]), "y_pred": max(range(5), key=scores.__getitem__),
                            **{f"p{grade}": value for grade, value in enumerate(scores)}})
        timings.append({"image_id": row["id_code"], "full_answer_seconds": elapsed})
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)} images.", flush=True)

    metrics = helper.calculate_metrics(predictions)
    for item, value in zip(metrics["per_class"], pinned.specificity(metrics["confusion_matrix"])):
        item["sensitivity"] = item["recall"]
        item["specificity"] = value
    helper.write_csv(predictions_path, predictions)
    helper.write_csv(output / "timing.csv", timings, ["image_id", "full_answer_seconds"])
    summary = {
        "identity": identity, "fallback_reason": reason, "fp16_probe": probe,
        "precision_detail": {"description": PRECISION_DETAIL, **census},
        "training_run": run_record,
        "metrics": metrics, "predictions_sha256": pinned.sha(predictions_path),
        "model_load_seconds": load_seconds,
        "full_answer_timing": {**pinned.timing_summary([row["full_answer_seconds"] for row in timings]),
                               "forward_passes_per_image": len(helper.CANDIDATES)},
        "generation": None,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
        "packages": helper.package_versions(),
    }
    if args.diagnostic_predictions and args.diagnostic_predictions.is_file():
        summary["agreement_with_diagnostic_predictions"] = pinned.compare_with_reference(
            helper, args.diagnostic_predictions, predictions)
    helper.write_json(summary_path, summary)
    qwk = "undefined" if metrics["qwk"] is None else f"{metrics['qwk']:.4f}"
    print(f"{args.name} (FP32 fallback): accuracy {metrics['accuracy']:.2%}, QWK {qwk}, "
          f"median {summary['full_answer_timing']['median_seconds']:.3f} s per image.", flush=True)


if __name__ == "__main__":
    main()
