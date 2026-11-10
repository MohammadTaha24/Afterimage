"""Shared, explicit workflow for the two paper-reproduction notebooks.

Importing this module does not import Torch/Unsloth or load a GPU model.
GPU work happens only in the evaluate/train CLI subcommands, in fresh processes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
BASE_REVISION = "b5b904c3fcdc7541adf2a2bb219b0ed95288c794"
ORIGINAL_SPLIT_SHA256 = "0ca0f29b1ed2e4f74a226d333e296298f72bf35539c2507077026c103ff4875d"
APTOS_COUNTS = {
    "train": [1444, 296, 799, 154, 236],
    "val": [180, 37, 100, 20, 29],
    "test": [181, 37, 100, 19, 30],
}
MESSIDOR_COUNTS = [468, 207, 290, 71, 21]
LABEL_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
INSTRUCTION = (
    "You are a medical imaging assistant.\n"
    "Grade diabetic retinopathy severity from this retinal fundus image.\n\n"
    "Scale:\n0 - No DR\n1 - Mild\n2 - Moderate\n3 - Severe\n4 - Proliferative DR\n\n"
    "Return exactly this format:\n<answer>X - LABEL</answer>"
)
CANDIDATES = [f"<answer>{i} - {name}</answer>" for i, name in enumerate(LABEL_NAMES)]
# Reported training values, plus explicit original Trainer defaults where the
# paper does not specify them. No offline augmentation or dataset re-splitting.
PAPER_TRAINING = {
    "epochs": 50, "batch_size": 4, "grad_accum": 2, "lr": 5e-4,
    "warmup_steps": 20, "max_length": 2048, "weight_decay": 0.001,
    "optim": "adamw_8bit", "scheduler": "linear", "lora_r": 16,
    "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none",
    "gradient_checkpointing": True, "img_size": 512, "load_in_4bit": True,
    "eval_strategy": "epoch", "save_strategy": "epoch", "save_total_limit": 2,
    "early_stopping_patience": 2, "early_stopping_threshold": 0.0,
    "metric_for_best_model": "eval_loss", "logging_steps": 5,
    "trainer_seed": 42, "eval_batch_size": 8,
    "lora_seed_mode": "original_unsloth_default", "expected_lora_default_seed": 3407,
}
WORKFLOW_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None, immutable=False):
    import io
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing an empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if immutable and path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Existing export differs; do not overwrite it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def package_versions():
    from importlib.metadata import version
    return {name: version(name) for name in (
        "torch", "torchvision", "unsloth", "unsloth_zoo", "transformers", "trl",
        "peft", "datasets", "numpy", "pandas", "bitsandbytes",
    )}


def validate_adapter(path):
    path = Path(path).resolve()
    if not (path / "adapter_config.json").is_file():
        path = path / "best_lora"
    config_file = path / "adapter_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"No adapter_config.json in {path}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or not config.get("base_model_name_or_path"):
        raise ValueError(f"Invalid LoRA configuration: {config_file}")
    weights = next((path / name for name in ("adapter_model.safetensors", "adapter_model.bin")
                    if (path / name).is_file() and (path / name).stat().st_size), None)
    if weights is None:
        raise FileNotFoundError(f"Missing non-empty adapter weights: {path}")
    return path, weights


def resolve_base_snapshot(model_id=BASE_MODEL, revision=BASE_REVISION):
    # Resolve the exact cached revision before redirecting model-process caches.
    from huggingface_hub import snapshot_download
    sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
    from model_assets import validate_snapshot
    path = Path(snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)).resolve()
    validate_snapshot(path, "qwen3")
    return path


def _grade(value):
    text = str(value).strip()
    if text not in {"0", "1", "2", "3", "4"}:
        raise ValueError(f"Invalid grade: {value!r}")
    return int(text)


def match_original_split(original_rows, mapping_rows, image_dir):
    """Match by original image ID; never borrow split assignments from mapping_rows."""
    image_dir = Path(image_dir).resolve()
    originals, mappings = {}, {}
    for row in original_rows:
        key = Path(row["image_file"].replace("\\", "/")).stem
        if not key or key in originals:
            raise ValueError(f"Duplicate/empty original image ID: {key}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid original split: {row['split']}")
        _grade(row["label"])
        originals[key] = row
    for row in mapping_rows:
        if row["source_dataset"] != "aptos":
            continue
        patient_id = row["patient_id"]
        if not patient_id.startswith("aptos_"):
            raise ValueError(f"Unexpected APTOS identifier: {patient_id}")
        key = patient_id.removeprefix("aptos_")
        if key in mappings:
            raise ValueError(f"Duplicate mapped APTOS image ID: {key}")
        mappings[key] = row
    if set(originals) != set(mappings):
        raise ValueError("Original split and image mapping contain different APTOS image IDs.")
    result = []
    seen_paths = set()
    for key, old in originals.items():
        current = mappings[key]
        if _grade(old["label"]) != _grade(current["label"]):
            raise ValueError(f"Label mismatch for {key}")
        path = (image_dir / current["image_path"].replace("\\", "/")).resolve()
        path_key = os.path.normcase(str(path))
        if path_key in seen_paths or not path.is_file():
            raise ValueError(f"Duplicate or missing image file: {path}")
        seen_paths.add(path_key)
        result.append({
            "id_code": key, "filename": path.name, "file_path": str(path), "image_path": str(path),
            "label": _grade(old["label"]), "source_dataset": "aptos", "split": old["split"],
            "patient_id": current["patient_id"],
        })
    return result


def prepare_data(original_csv, mapping_csv, image_dir, output_root, verify_images=True):
    original_csv, mapping_csv = Path(original_csv).resolve(), Path(mapping_csv).resolve()
    if sha256_file(original_csv) != ORIGINAL_SPLIT_SHA256:
        raise ValueError("The original split CSV differs from the recovered paper split. Review it before running.")
    mapping_rows = read_csv(mapping_csv)
    matched = match_original_split(read_csv(original_csv), mapping_rows, image_dir)
    frames = {split: [row for row in matched if row["split"] == split] for split in APTOS_COUNTS}
    for split, expected in APTOS_COUNTS.items():
        counts = Counter(row["label"] for row in frames[split])
        if [counts[i] for i in range(5)] != expected:
            raise ValueError(f"Unexpected original {split} counts: {counts}")
    external = []
    for row in mapping_rows:
        if row["source_dataset"] != "messidor":
            continue
        path = (Path(image_dir) / row["image_path"].replace("\\", "/")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        external.append({
            "id_code": Path(row["image_path"]).stem, "filename": path.name,
            "file_path": str(path), "image_path": str(path), "label": _grade(row["label"]),
            "source_dataset": "messidor", "split": "test", "patient_id": row["patient_id"],
        })
    external_counts = Counter(row["label"] for row in external)
    if [external_counts[i] for i in range(5)] != MESSIDOR_COUNTS:
        raise ValueError(f"Unexpected Messidor external cohort: {external_counts}")
    paths = [row["image_path"] for row in matched + external]
    if len(set(map(os.path.normcase, paths))) != len(paths):
        raise ValueError("Duplicate image paths across cohorts.")
    if verify_images:
        from PIL import Image
        for index, path in enumerate(paths, 1):
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise ValueError(f"Expected 512x512 pixels: {path}, found {image.size}")
                image.verify()
            if index % 1000 == 0 or index == len(paths):
                print(f"Verified {index}/{len(paths)} image files", flush=True)
    identity = {
        "original_split_sha256": sha256_file(original_csv), "mapping_sha256": sha256_file(mapping_csv),
        "image_dir": str(Path(image_dir).resolve()), "workflow_version": WORKFLOW_VERSION,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    export_dir = Path(output_root).resolve() / "data" / fingerprint
    csv_paths = {}
    for name, rows in {**frames, "messidor_external": external}.items():
        path = export_dir / f"{name}.csv"
        write_csv(path, rows, immutable=True)
        csv_paths[name] = str(path)
    manifest = {
        **identity, "original_split_csv": str(original_csv), "image_mapping_csv": str(mapping_csv),
        "split_policy": "recovered_original_aptos_split; all_messidor_external_only",
        "counts": {name: len(read_csv(path)) for name, path in csv_paths.items()},
        "aptos_class_counts": APTOS_COUNTS, "messidor_class_counts": MESSIDOR_COUNTS,
        "csv_paths": csv_paths, "csv_sha256": {name: sha256_file(path) for name, path in csv_paths.items()},
        "images_verified": verify_images, "same_aptos_ids_and_labels": True,
        "note": "Image identity/labels match the recovered split. Original historical preprocessed image bytes are unavailable for comparison.",
    }
    write_json(export_dir / "dataset_manifest.json", manifest)
    return export_dir / "dataset_manifest.json"


def configure_process_caches(output_dir, evaluation=False):
    cache = Path(output_dir).resolve() / "cache"
    for name, suffix in (
        ("HF_HOME", "hf"), ("HF_HUB_CACHE", "hf/hub"), ("HF_XET_CACHE", "hf/xet"),
        ("TORCH_HOME", "torch"), ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("UNSLOTH_COMPILE_LOCATION", "unsloth"),
    ):
        path = cache / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if evaluation:
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
        os.environ["TORCHINDUCTOR_DISABLE"] = "1"


def run_live(command, log_path):
    command = [str(part) for part in command]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(log_path.with_suffix(".command.json"), command)
    print(subprocess.list2cmdline(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {utc_now()} ===\n")
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, command)


def roc_auc_binary(truth, scores):
    import numpy as np
    truth, scores = np.asarray(truth), np.asarray(scores)
    positives = truth.sum()
    if positives == 0 or positives == len(truth):
        return None
    order = np.argsort(-scores, kind="stable")
    truth, scores = truth[order], scores[order]
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(truth)[last]
    fp = 1 + last - tp
    return float(np.trapezoid(np.r_[0, tp] / positives, np.r_[0, fp] / (len(truth) - positives)))


def calculate_metrics(rows):
    import numpy as np
    if not rows:
        raise ValueError("No predictions to evaluate.")
    truth = np.array([_grade(row["y_true"]) for row in rows])
    pred = np.array([_grade(row["y_pred"]) for row in rows])
    scores = np.array([[float(row[f"p{i}"]) for i in range(5)] for row in rows])
    if not np.isfinite(scores).all() or (scores < 0).any() or not np.allclose(scores.sum(1), 1, atol=1e-6):
        raise ValueError("Invalid candidate scores.")
    if not np.array_equal(pred, scores.argmax(1)):
        raise ValueError("Predicted grades do not match the saved score argmax.")
    cm = np.zeros((5, 5), dtype=int)
    np.add.at(cm, (truth, pred), 1)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(5), where=predicted != 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) != 0)
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2 / 16
    expected = np.outer(support, predicted) / len(rows)
    denominator = float((weights * expected).sum())
    aucs = [roc_auc_binary(truth == i, scores[:, i]) for i in range(5)]
    valid_aucs = [value for value in aucs if value is not None]
    return {
        "samples": len(rows), "accuracy": float((truth == pred).mean()),
        "qwk": float(1 - (weights * cm).sum() / denominator) if denominator else None,
        "macro_ovr_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
        "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[support > 0].mean()),
        "confusion_matrix": cm.tolist(),
        "per_class": [{"grade": i, "label": LABEL_NAMES[i], "support": int(support[i]),
                       "precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "auc": aucs[i]} for i in range(5)],
    }


def score_full_answers(model, processor, image_path):
    """Original notebook full-answer log likelihood, not text generation.

    Retains text-before-image ordering, FP16 pixel values, prompt masking, and
    total (not length-normalized) answer log likelihood from the reference.
    """
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    prompt = [{"role": "user", "content": [
        {"type": "text", "text": INSTRUCTION}, {"type": "image", "image": image},
    ]}]
    prompt_text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    prompt_length = inputs["input_ids"].shape[1]
    scores = []
    with torch.no_grad():
        for candidate in CANDIDATES:
            messages = prompt + [{"role": "assistant", "content": [{"type": "text", "text": candidate}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            batch = processor(images=image, text=text, return_tensors="pt").to(model.device)
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.float16)
            labels = batch["input_ids"].clone()
            labels[:, :prompt_length] = -100
            count = int((labels != -100).sum().item())
            if count == 0:
                raise ValueError("No answer tokens remain after prompt masking.")
            output = model(**batch, labels=labels)
            score = -float(output.loss.item()) * count
            if not math.isfinite(score):
                raise ValueError(f"Non-finite answer likelihood for {image_path}")
            scores.append(score)
    scores = np.asarray(scores, dtype=np.float64)
    scores -= scores.max()
    return np.exp(scores) / np.exp(scores).sum()


def load_test_rows(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty test CSV: {path}")
    expected = []
    seen = set()
    for row in rows:
        if row.get("split") != "test":
            raise ValueError(f"Evaluation CSV contains a non-test row: {path}")
        image_path = str(Path(row["image_path"]).resolve())
        if image_path in seen or not Path(image_path).is_file():
            raise ValueError(f"Missing/duplicate evaluation image: {image_path}")
        seen.add(image_path)
        expected.append({"image_id": row["id_code"], "image_path": image_path, "y_true": _grade(row["label"])})
    return expected


def validate_evaluation_cohort(path, source, class_counts):
    records = read_csv(path)
    if any(row.get("source_dataset") != source for row in records):
        raise ValueError(f"Expected only {source} test images: {path}")
    counts = Counter(_grade(row["label"]) for row in records)
    if [counts[i] for i in range(5)] != class_counts:
        raise ValueError(f"Incorrect {source} test cohort: {counts}")
    return load_test_rows(path)


def validate_prediction_prefix(rows, expected):
    if len(rows) > len(expected):
        raise ValueError("Prediction file contains too many rows.")
    for row, target in zip(rows, expected):
        if row["image_id"] != target["image_id"] or row["image_path"] != target["image_path"] or _grade(row["y_true"]) != target["y_true"]:
            raise ValueError("Saved predictions are not a prefix of the selected test set.")
    if rows:
        calculate_metrics(rows)


def _evaluate_subset(model, processor, csv_path, output_dir, name, identity):
    import numpy as np
    output_dir = Path(output_dir)
    expected = load_test_rows(csv_path)
    predictions_path = output_dir / f"{name}.predictions.csv"
    fields = ["image_id", "image_path", "y_true", "y_pred"] + [f"p{i}" for i in range(5)]
    has_header = predictions_path.exists() and predictions_path.stat().st_size > 0
    if has_header:
        with predictions_path.open(encoding="utf-8", newline="") as stream:
            if csv.DictReader(stream).fieldnames != fields:
                raise ValueError("Unexpected prediction CSV header; preserve it and choose a new output directory.")
    rows = read_csv(predictions_path) if predictions_path.exists() else []
    validate_prediction_prefix(rows, expected)
    started = time.monotonic()
    with predictions_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        if not has_header:
            writer.writeheader()
        completed_before_start = len(rows)
        for index in range(len(rows), len(expected)):
            row = expected[index]
            scores = score_full_answers(model, processor, row["image_path"])
            result = {**row, "y_pred": int(np.argmax(scores)), **{f"p{i}": float(scores[i]) for i in range(5)}}
            writer.writerow(result)
            stream.flush()
            rows.append(result)
            if len(rows) == 1 or len(rows) % 10 == 0 or len(rows) == len(expected):
                print(f"[{name}] {len(rows)}/{len(expected)} | {time.monotonic() - started:.1f}s this session", flush=True)
    metrics = {**identity, "name": name, "test_csv": str(csv_path), "test_csv_sha256": sha256_file(csv_path),
               "resumed_rows": completed_before_start, **calculate_metrics(rows)}
    write_json(output_dir / f"{name}.summary.json", metrics)
    write_csv(output_dir / f"{name}.per_class.csv", metrics["per_class"])
    matrix_rows = [{"true_grade": i, **{str(j): metrics["confusion_matrix"][i][j] for j in range(5)}} for i in range(5)]
    write_csv(output_dir / f"{name}.confusion_matrix.csv", matrix_rows)
    print(json.dumps({key: metrics[key] for key in ("name", "samples", "accuracy", "qwk", "macro_ovr_auc", "macro_f1")}, indent=2), flush=True)
    return metrics


def evaluate(adapter, base_snapshot, test_csv, output_dir, external_csv=None):
    adapter, weights = validate_adapter(adapter)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets = {"aptos_test": Path(test_csv).resolve()}
    if external_csv:
        subsets["messidor_external"] = Path(external_csv).resolve()
    validate_evaluation_cohort(subsets["aptos_test"], "aptos", APTOS_COUNTS["test"])
    if external_csv:
        validate_evaluation_cohort(subsets["messidor_external"], "messidor", MESSIDOR_COUNTS)
    identity = {
        "workflow_version": WORKFLOW_VERSION, "adapter_dir": str(adapter), "adapter_sha256": sha256_file(weights),
        "base_snapshot": str(Path(base_snapshot).resolve()), "prediction_method": "full_answer_logprob_argmax",
        "evaluation_precision": "FP16", "helper_sha256": sha256_file(__file__),
        "csv_sha256": {name: sha256_file(path) for name, path in subsets.items()},
        "package_versions": package_versions(),
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Evaluation identity changed. Choose a new output directory.")
    write_json(manifest_path, identity)
    if all((output_dir / f"{name}.summary.json").is_file() for name in subsets):
        for name, path in subsets.items():
            predictions = read_csv(output_dir / f"{name}.predictions.csv")
            expected = load_test_rows(path)
            validate_prediction_prefix(predictions, expected)
            if len(predictions) != len(expected):
                raise ValueError("A completed evaluation has incomplete predictions.")
        print("Evaluation already complete and identity matches; keeping existing results.", flush=True)
        return
    configure_process_caches(output_dir, evaluation=True)
    import pandas  # Expose any first binary import failure before Unsloth probes.
    from unsloth import FastVisionModel
    from peft import PeftModel
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(base_snapshot), load_in_4bit=True, local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    FastVisionModel.for_inference(model)
    model = model.half()
    model.eval()
    for name, path in subsets.items():
        _evaluate_subset(model, processor, path, output_dir, name, identity)


def make_training_config(manifest_path, base_snapshot):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "workflow_version": WORKFLOW_VERSION, "model_name": BASE_MODEL, "base_revision": BASE_REVISION,
        "base_snapshot": str(Path(base_snapshot).resolve()), "dataset_manifest": str(manifest_path),
        "dataset_csv_sha256": manifest["csv_sha256"], "training": dict(PAPER_TRAINING),
        "helper_sha256": sha256_file(__file__), "trainer_module_sha256": sha256_file(Path(__file__).with_name("train_dr_qwen3vl_runsave.py")),
        "package_versions": package_versions(),
        "seed_note": "Original trainer defaults to 42; original LoRA call omits random_state. Preserve that call and assert the current default is 3407. The paper does not explicitly record its historical LoRA seed.",
    }


def _validate_training_config(config):
    if config["training"] != PAPER_TRAINING:
        raise ValueError("Training settings differ from the reviewed paper reproduction configuration.")
    if config["helper_sha256"] != sha256_file(__file__):
        raise ValueError("Helper source changed since this batch was prepared.")
    if config["trainer_module_sha256"] != sha256_file(Path(__file__).with_name("train_dr_qwen3vl_runsave.py")):
        raise ValueError("Shared trainer definitions changed since this batch was prepared.")
    if config["package_versions"] != package_versions():
        raise ValueError("Installed package versions changed since this batch was prepared.")
    manifest = json.loads(Path(config["dataset_manifest"]).read_text(encoding="utf-8"))
    for name, path in manifest["csv_paths"].items():
        if sha256_file(path) != config["dataset_csv_sha256"][name]:
            raise ValueError(f"Dataset CSV changed: {path}")
    return manifest


def make_sft_config(config, output_dir, sft_config_class):
    p = config["training"]
    return sft_config_class(
        output_dir=str(output_dir), per_device_train_batch_size=p["batch_size"],
        per_device_eval_batch_size=p["eval_batch_size"], gradient_accumulation_steps=p["grad_accum"],
        num_train_epochs=p["epochs"], max_steps=-1, learning_rate=p["lr"], warmup_steps=p["warmup_steps"],
        weight_decay=p["weight_decay"], optim=p["optim"], lr_scheduler_type=p["scheduler"],
        logging_steps=p["logging_steps"], seed=p["trainer_seed"], bf16=True, fp16=False,
        eval_strategy=p["eval_strategy"], save_strategy=p["save_strategy"], save_total_limit=p["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model=p["metric_for_best_model"], greater_is_better=False,
        max_length=p["max_length"], report_to="none", remove_unused_columns=False,
        dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=0, restore_callback_states_from_checkpoint=True,
    )


def resolve_lora_default_seed(fast_vision_model):
    method = fast_vision_model.get_peft_model
    parameter = inspect.signature(method).parameters.get("random_state")
    if parameter is None:
        # Current FastVisionModel aliases FastModel, whose wrapper forwards
        # ordinary VLMs to FastBaseModel and exposes only *args/**kwargs.
        from unsloth.models.vision import FastBaseModel
        parameter = inspect.signature(FastBaseModel.get_peft_model).parameters.get("random_state")
    if parameter is None or not isinstance(parameter.default, int):
        raise RuntimeError("Cannot resolve the installed vision LoRA default seed; review the library before training.")
    return parameter.default


def completed_training(run_dir, config):
    marker = Path(run_dir) / "training_complete.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved["configuration"] != config:
        raise ValueError("A completed run has different settings. Choose a new batch directory.")
    _, weights = validate_adapter(Path(run_dir) / "best_lora")
    if sha256_file(weights) != saved["adapter_sha256"]:
        raise ValueError("Completed adapter weights differ from the completion record.")
    return True


def train_one(config_path, run_dir):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    manifest = _validate_training_config(config)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if completed_training(run_dir, config):
        print(f"Training already complete; preserving adapter: {run_dir / 'best_lora'}", flush=True)
        return
    config_record = run_dir / "configuration.json"
    if config_record.exists() and json.loads(config_record.read_text(encoding="utf-8")) != config:
        raise ValueError("Run settings changed. Choose a new run directory.")
    write_json(config_record, config)
    configure_process_caches(run_dir)
    import pandas
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback
    from train_dr_qwen3vl_runsave import BalancedSFTTrainer, BestEpochTrackerCallback, build_datasets
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This reproduction configuration requires the verified CUDA/BF16 GPU environment.")
    p = config["training"]
    lora_seed = resolve_lora_default_seed(FastVisionModel)
    if lora_seed != p["expected_lora_default_seed"]:
        raise ValueError(f"Unsloth LoRA default seed changed: expected 3407, found {lora_seed}")
    print(f"Trainer seed: {p['trainer_seed']} | original-style Unsloth default seed: {lora_seed}", flush=True)
    print("Starting from the cached BASE model, never a previous run's adapter.", flush=True)
    train_ds, val_ds, test_ds = build_datasets(
        csv_path=manifest["csv_paths"]["train"], val_csv=manifest["csv_paths"]["val"],
        test_csv=manifest["csv_paths"]["test"], image_dir=manifest["image_dir"], ext=".png",
        test_size=0.1, seed=42, img_size="512", cache_dir=str(run_dir / "dataset_cache"),
        preproc_num_proc=0,
    )
    if [len(train_ds), len(val_ds), len(test_ds)] != [2929, 366, 367]:
        raise ValueError("Incorrect original APTOS split sizes.")
    checkpoint_dir = run_dir / "checkpoints"
    args = make_sft_config(config, checkpoint_dir, SFTConfig)
    model, processor = FastVisionModel.from_pretrained(
        model_name=config["base_snapshot"], load_in_4bit=True, local_files_only=True,
    )
    model = FastVisionModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        use_gradient_checkpointing=True,
        temporary_location=str(run_dir / "cache/saved_buffers"),
        # Deliberately omit random_state, as in the original script.
    )
    # Keep the exported adapter portable; snapshot provenance is saved separately.
    for adapter_config in model.peft_config.values():
        adapter_config.base_model_name_or_path = BASE_MODEL
    FastVisionModel.for_training(model)
    tracker = BestEpochTrackerCallback()
    resume_checkpoint = None
    if checkpoint_dir.exists():
        candidates = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
        for candidate in reversed(candidates):
            if all((candidate / name).is_file() for name in ("trainer_state.json", "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")):
                try:
                    validate_adapter(candidate)
                except (ValueError, FileNotFoundError):
                    continue
                resume_checkpoint = candidate
                break
        if candidates and resume_checkpoint is None:
            raise ValueError("Only incomplete checkpoints were found. Preserve them and select a new batch directory.")
    if resume_checkpoint:
        state = json.loads((resume_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        for event in state.get("log_history", []):
            if "eval_loss" in event:
                entry = {"step": event["step"], "epoch": event.get("epoch"), "eval_loss": event["eval_loss"]}
                tracker.history.append(entry)
                if tracker.best_eval_loss is None or event["eval_loss"] < tracker.best_eval_loss:
                    tracker.best_eval_loss, tracker.best_epoch, tracker.best_step = event["eval_loss"], event.get("epoch"), event["step"]
        print(f"Resuming this same run from {resume_checkpoint}", flush=True)
    else:
        print("Fresh training run.", flush=True)
    processor_key = "tokenizer" if "tokenizer" in inspect.signature(SFTTrainer.__init__).parameters else "processing_class"
    trainer = BalancedSFTTrainer(
        model=model, **{processor_key: processor}, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=UnslothVisionDataCollator(model, processor, resize=(512, 512), max_seq_length=2048),
        callbacks=[tracker, EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
    )
    write_json(run_dir / "resolved_training_arguments.json", args.to_dict())
    started = time.monotonic()
    statistics = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("No best validation checkpoint was recorded; refusing to label the final model as best.")
    destination = run_dir / "best_lora"
    trainer.save_model(str(destination))  # load_best_model_at_end has restored the validation winner.
    processor.save_pretrained(str(destination))
    _, weights = validate_adapter(destination)
    info = {
        "created_at": utc_now(), "configuration": config, "trainer_seed": 42,
        "resolved_lora_default_seed": lora_seed, "best_epoch": tracker.best_epoch,
        "best_step": tracker.best_step, "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint, "final_lora_dir": str(destination),
        "adapter_sha256": sha256_file(weights), "validation_history": tracker.history,
        "trainer_log_history": trainer.state.log_history, "train_metrics": statistics.metrics,
        "epochs_completed": trainer.state.epoch,
        "elapsed_this_session_seconds": time.monotonic() - started,
        "selection_rule": "minimum APTOS validation loss; test metrics never select checkpoints",
    }
    write_json(run_dir / "run_info.json", info)
    write_json(run_dir / "training_complete.json", info)
    print(f"Saved independent best adapter: {destination}", flush=True)


def run_three(config_path, batch_dir, include_external=True):
    """Sequential subprocesses; completed training is never overwritten on restart."""
    config_path, batch_dir = Path(config_path).resolve(), Path(batch_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = _validate_training_config(config)
    results = []
    for index in range(1, 4):
        run_dir = batch_dir / f"run_{index:02d}"
        entry = {"run": index, "run_dir": str(run_dir), "adapter_dir": str(run_dir / "best_lora"), "status": "running"}
        print(f"\n===== RUN {index}/3: same seeds, fresh base, original APTOS split =====", flush=True)
        try:
            run_live([sys.executable, "-u", __file__, "train", "--config", config_path, "--run-dir", run_dir], run_dir / "training.log")
            command = [sys.executable, "-u", __file__, "evaluate", "--adapter", run_dir / "best_lora",
                       "--base-snapshot", config["base_snapshot"], "--test-csv", manifest["csv_paths"]["test"],
                       "--output-dir", run_dir / "evaluation"]
            if include_external:
                command.extend(["--external-csv", manifest["csv_paths"]["messidor_external"]])
            run_live(command, run_dir / "evaluation.log")
            metrics = json.loads((run_dir / "evaluation/aptos_test.summary.json").read_text(encoding="utf-8"))
            info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
            entry.update(status="complete", best_epoch=info["best_epoch"], best_eval_loss=info["best_eval_loss"],
                         adapter_sha256=info["adapter_sha256"], epochs_completed=info["epochs_completed"],
                         **{key: metrics[key] for key in ("accuracy", "qwk", "macro_ovr_auc", "macro_f1", "balanced_accuracy")})
            if include_external:
                external = json.loads((run_dir / "evaluation/messidor_external.summary.json").read_text(encoding="utf-8"))
                entry.update({f"messidor_{key}": external[key] for key in ("accuracy", "qwk", "macro_ovr_auc")})
        except Exception as exc:
            # An interrupted notebook still stops the queue (KeyboardInterrupt is not caught).
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(run_dir / "failure.json", {**entry, "time": utc_now()})
            print(f"Run {index} failed; its files are preserved. Continuing to the next fresh run. {exc}", flush=True)
        results.append(entry)
        write_json(batch_dir / "batch_results.json", results)
        fields = sorted(set().union(*(row.keys() for row in results)))
        write_csv(batch_dir / "batch_results.csv", results, fields=fields)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    for name in ("adapter", "base-snapshot", "test-csv", "output-dir"):
        evaluate_parser.add_argument("--" + name, required=True)
    evaluate_parser.add_argument("--external-csv")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate(args.adapter, args.base_snapshot, args.test_csv, args.output_dir, args.external_csv)
    else:
        train_one(args.config, args.run_dir)


if __name__ == "__main__":
    main()
