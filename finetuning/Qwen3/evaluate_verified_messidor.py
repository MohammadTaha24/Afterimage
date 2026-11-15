"""Evaluate a saved Qwen adapter on an independently verified Messidor cohort."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import evaluate_clahe_bf16
import paper_reproduction as original


def verified_rows(csv_path):
    rows = original.read_csv(csv_path)
    if not rows or {row["source_dataset"] for row in rows} != {"messidor"}:
        raise ValueError("The cohort must contain only Messidor rows.")
    keys = [(row["id_code"], original._grade(row["label"]), row["patient_id"]) for row in rows]
    if any(not ident.strip() or not patient.strip() for ident, _, patient in keys):
        raise ValueError("Every row needs an image ID and a verified patient or pairing ID.")
    if len({ident for ident, _, _ in keys}) != len(keys) or {grade for _, grade, _ in keys} != set(range(5)):
        raise ValueError("Require unique image IDs and examples of all five grades.")
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_absolute() or path.suffix.lower() != ".png":
            raise ValueError(f"Use absolute paths to prepared PNG images: {path}")
    original.load_test_rows(csv_path)
    return sorted(keys)


def evaluate(adapter, base_snapshot, csv_path, output_dir, precision, base_precision, tokenizer=None):
    verified_rows(csv_path)
    adapter, weights = original.validate_adapter(adapter)
    identity = {
        "adapter_dir": str(adapter),
        "adapter_sha256": original.sha256_file(weights),
        "base_snapshot": str(base_snapshot.resolve(strict=True)),
        "base_precision": base_precision,
        "tokenizer": str(tokenizer.resolve(strict=True)) if tokenizer else None,
        "prediction_method": "full_answer_logprob_argmax",
        "evaluation_precision": precision.upper(),
        "helper_sha256": original.sha256_file(original.__file__),
        "worker_sha256": original.sha256_file(__file__),
        "bf16_scorer_sha256": original.sha256_file(evaluate_clahe_bf16.__file__) if precision == "bf16" else None,
        "csv_sha256": {"messidor_external": original.sha256_file(csv_path)},
        "package_versions": original.package_versions(),
    }
    # Bind each score to the prepared image bytes as well as the CSV.
    digest = hashlib.sha256()
    for row in original.read_csv(csv_path):
        digest.update(bytes.fromhex(original.sha256_file(row["image_path"])))
    identity["images_sha256"] = digest.hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "evaluation_manifest.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != identity:
        raise ValueError("Inputs changed. Use a fresh output tag to preserve the existing evaluation.")
    original.write_json(marker, identity)
    prefix = output_dir / "messidor_external"
    if prefix.with_suffix(".summary.json").is_file():
        rows = original.read_csv(prefix.with_suffix(".predictions.csv"))
        expected = original.load_test_rows(csv_path)
        original.validate_prediction_prefix(rows, expected)
        if len(rows) != len(expected):
            raise ValueError("A saved evaluation has incomplete predictions.")
        print("External evaluation already complete.", flush=True)
        return

    original.configure_process_caches(output_dir, evaluation=True)
    import pandas
    from unsloth import FastVisionModel
    import torch
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA hardware is required for this evaluation.")
    quantized = base_precision == "4bit"
    kwargs = {} if quantized else {"dtype": torch.bfloat16, "load_in_16bit": True,
                                   "load_in_8bit": False, "device_map": {"": 0}}
    if tokenizer:
        kwargs["tokenizer_name"] = str(tokenizer)
    model, processor = FastVisionModel.from_pretrained(model_name=str(base_snapshot),
                                                       load_in_4bit=quantized, local_files_only=True, **kwargs)
    if bool(getattr(model, "is_loaded_in_4bit", False)) != quantized:
        raise RuntimeError("The loaded base does not match the selected weight precision.")
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    FastVisionModel.for_inference(model)
    if precision == "bf16":
        model.to(dtype=torch.bfloat16)
        score = evaluate_clahe_bf16.score_full_answers
    else:
        model.half()
        score = original.score_full_answers
    model.eval()
    with patch.object(original, "score_full_answers", score):
        original._evaluate_subset(model, processor, csv_path, output_dir, "messidor_external", identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate",))
    for name in ("adapter", "base-snapshot", "csv-path", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--precision", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--base-precision", choices=("4bit", "bf16"), required=True)
    parser.add_argument("--tokenizer", type=Path)
    args = vars(parser.parse_args())
    args.pop("stage")
    evaluate(**args)
