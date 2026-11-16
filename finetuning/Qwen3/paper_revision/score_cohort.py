"""Score a Qwen base model, with or without a LoRA adapter, on one cohort and time each image.

Grades come from the pinned five-answer likelihood scorer in paper_reproduction.py.
Optionally the same images are also answered by greedy text generation, which gives
the single-pass inference time and a count of outputs that cannot be parsed.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path


SCRIPT = Path(__file__).resolve()
HELPER = SCRIPT.parents[1] / "paper_reproduction.py"
ANSWER = re.compile(r"<answer>\s*([0-4])\s*-")
WARMUP_IMAGES = 3
GENERATION_TOKENS = 24


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_helper(path):
    spec = importlib.util.spec_from_file_location("qwen_reproduction", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def parse_grade(text):
    match = ANSWER.search(text)
    return int(match.group(1)) if match else None


def specificity(confusion):
    total = sum(map(sum, confusion))
    values = []
    for grade in range(len(confusion)):
        positives = sum(confusion[grade])
        predicted = sum(row[grade] for row in confusion)
        negatives = total - positives
        true_negatives = negatives - (predicted - confusion[grade][grade])
        values.append(true_negatives / negatives if negatives else None)
    return values


def timing_summary(seconds):
    measured = seconds[WARMUP_IMAGES:] if len(seconds) > WARMUP_IMAGES else seconds
    ordered = sorted(measured)
    return {
        "images_timed": len(measured), "warmup_images_excluded": len(seconds) - len(measured),
        "mean_seconds": statistics.fmean(measured), "median_seconds": statistics.median(measured),
        "p95_seconds": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "images_per_second": len(measured) / sum(measured),
    }


def cohort_rows(helper, path, limit):
    rows = helper.read_csv(path)
    seen = set()
    for row in rows:
        if row["label"] not in {"0", "1", "2", "3", "4"} or row["id_code"] in seen:
            raise ValueError(f"Invalid or repeated cohort row: {row['id_code']}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        seen.add(row["id_code"])
    if limit:
        # Take images of every grade in turn so a short pilot still sees each class.
        by_grade = {grade: [row for row in rows if row["label"] == str(grade)] for grade in range(5)}
        picked = []
        while len(picked) < min(limit, len(rows)):
            for grade in range(5):
                if by_grade[grade] and len(picked) < limit:
                    picked.append(by_grade[grade].pop(0))
        rows = picked
    return rows


def adapter_files(helper, adapter):
    if adapter is None:
        return None, None
    path, weights = helper.validate_adapter(adapter)
    return path, {"directory": str(path), "weights_sha256": sha(weights),
                  "configuration_sha256": sha(path / "adapter_config.json")}


def load_model(args, adapter, torch, vision_model):
    offload = args.offload_layers or 0
    if args.base_precision == "4bit":
        model, processor = vision_model.from_pretrained(
            model_name=str(args.base), load_in_4bit=True, local_files_only=True)
        if not getattr(model, "is_loaded_in_4bit", False) or getattr(model, "peft_config", None):
            raise ValueError("Expected a fresh 4-bit base")
    else:
        # Same call as the BF16 control's scoring stage: FP16 weights, frozen layers in system RAM.
        model, processor = vision_model.from_pretrained(
            model_name=str(args.base), dtype=torch.float16 if offload else torch.bfloat16,
            load_in_16bit=True, load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
            offload_embedding=False, offload_layers=offload, local_files_only=True,
            device_map={"": 0}, tokenizer_name=str(args.tokenizer))
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_quantized", False):
            raise ValueError("Expected an unquantized base")
        if offload and len(getattr(model, "_unsloth_block_swap").indices) != offload:
            raise ValueError("The layer offload setting did not initialize")
    if adapter is not None:
        from peft import PeftModel
        if args.base_precision == "bf16" and offload:
            model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True,
                                              autocast_adapter_dtype=False)
        else:
            model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    vision_model.for_inference(model)
    if args.base_precision == "bf16" and offload:
        for name, param in model.named_parameters():
            if "lora_" in name and param.is_floating_point():
                param.data = param.data.to(dtype=torch.float16)
    elif adapter is not None:
        model = model.half()
    else:
        # A bare quantized model refuses .half(). The module-level cast is what the adapter
        # wrapper's .half() performs on the same base: non-quantized weights to FP16.
        model = torch.nn.Module.half(model)
    model.eval()
    return model, processor


def generate_answer(helper, model, processor, image_path, torch):
    from PIL import Image
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    prompt = [{"role": "user", "content": [
        {"type": "text", "text": helper.INSTRUCTION}, {"type": "image", "image": image}]}]
    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    prompt_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=GENERATION_TOKENS, do_sample=False, use_cache=True)
    new_tokens = output[0, prompt_tokens:]
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(new_tokens, skip_special_tokens=True), prompt_tokens, int(new_tokens.shape[0])


def compare_with_reference(helper, path, predictions):
    saved = {row["image_id"]: row for row in helper.read_csv(path)}
    grade_mismatches, largest = 0, 0.0
    for row in predictions:
        reference = saved.get(row["image_id"])
        if reference is None:
            raise ValueError(f"Reference predictions lack {row['image_id']}")
        grade_mismatches += int(reference["y_pred"]) != row["y_pred"]
        largest = max(largest, max(abs(float(reference[f"p{i}"]) - row[f"p{i}"]) for i in range(5)))
    return {"reference": str(path), "reference_sha256": sha(path), "images_compared": len(predictions),
            "grade_mismatches": grade_mismatches, "largest_probability_difference": largest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score",))
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-precision", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument("--offload-layers", type=int, default=0)
    parser.add_argument("--tokenizer", type=Path, help="Tokenizer source for a BF16 base (the 4-bit snapshot)")
    parser.add_argument("--adapter", type=Path, help="Omit to score the base model with no adapter")
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--cohort-name", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Score only this many images (pilot)")
    parser.add_argument("--generation-samples", type=int, default=0,
                        help="Images also answered by text generation; -1 for all")
    parser.add_argument("--reference-predictions", type=Path,
                        help="Saved predictions for the same adapter and cohort, to confirm agreement")
    parser.add_argument("--helper", type=Path, default=HELPER)
    args = parser.parse_args()
    if args.base_precision == "bf16" and args.tokenizer is None:
        parser.error("A BF16 base needs --tokenizer")

    helper = load_helper(args.helper.resolve(strict=True))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = cohort_rows(helper, args.cohort_csv, args.limit)
    adapter, adapter_record = adapter_files(helper, args.adapter)
    wanted = len(rows) if args.generation_samples < 0 else min(args.generation_samples, len(rows))
    identity = {
        "name": args.name, "scorer_sha256": sha(SCRIPT), "helper_sha256": sha(args.helper),
        "cohort": args.cohort_name, "cohort_csv": str(args.cohort_csv.resolve()),
        "cohort_csv_sha256": sha(args.cohort_csv), "images": len(rows), "limit": args.limit,
        "base": str(args.base.resolve()), "base_config_sha256": sha(args.base / "config.json"),
        "base_precision": args.base_precision, "offload_layers": args.offload_layers,
        "adapter": adapter_record, "generation_images": wanted, "scoring_precision": "FP16",
    }
    summary_path = output / "summary.json"
    predictions_path = output / "predictions.csv"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError(f"{output} holds a different scoring job. Choose a fresh output folder.")
        if saved["predictions_sha256"] != sha(predictions_path):
            raise ValueError("Saved predictions changed.")
        print(f"Already scored: {output}", flush=True)
        return

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
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, processor = load_model(args, adapter, torch, FastVisionModel)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    print(f"Loaded {args.base_precision} base{' with adapter' if adapter else ' (no adapter)'} "
          f"in {load_seconds:.1f} s. Scoring {len(rows)} {args.cohort_name} images.", flush=True)

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

    generation = None
    if wanted:
        outputs, prompt_tokens = [], None
        for index, (row, prediction, timing) in enumerate(zip(rows[:wanted], predictions, timings), 1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            text, prompt_tokens, new_tokens = generate_answer(helper, model, processor, row["image_path"], torch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            grade = parse_grade(text)
            timing.update(generation_seconds=elapsed, generation_new_tokens=new_tokens,
                          generated_grade="" if grade is None else grade, generated_text=text)
            outputs.append((prediction["y_true"], prediction["y_pred"], grade, elapsed))
            if index == 1 or index % 25 == 0 or index == wanted:
                print(f"Generated {index}/{wanted} answers.", flush=True)
        parsed = [item for item in outputs if item[2] is not None]
        generation = {
            "images": wanted, "max_new_tokens": GENERATION_TOKENS, "decoding": "greedy",
            "prompt_tokens": prompt_tokens,
            "unparseable_outputs": wanted - len(parsed),
            "accuracy_unparseable_counted_wrong": sum(t == g for t, _, g, _ in parsed) / wanted,
            "accuracy_unparseable_as_grade_0": sum(t == (0 if g is None else g) for t, _, g, _ in outputs) / wanted,
            "agreement_with_likelihood_grade": sum(p == g for _, p, g, _ in parsed) / wanted,
            "timing": timing_summary([item[3] for item in outputs]),
        }

    metrics = helper.calculate_metrics(predictions)
    for item, value in zip(metrics["per_class"], specificity(metrics["confusion_matrix"])):
        item["sensitivity"] = item["recall"]
        item["specificity"] = value
    helper.write_csv(predictions_path, predictions)
    fields = ["image_id", "full_answer_seconds", "generation_seconds", "generation_new_tokens",
              "generated_grade", "generated_text"]
    helper.write_csv(output / "timing.csv", [{field: row.get(field, "") for field in fields} for row in timings], fields)
    summary = {
        "identity": identity, "metrics": metrics, "predictions_sha256": sha(predictions_path),
        "model_load_seconds": load_seconds,
        "full_answer_timing": {**timing_summary([row["full_answer_seconds"] for row in timings]),
                               "forward_passes_per_image": len(helper.CANDIDATES)},
        "generation": generation,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
        "packages": helper.package_versions(),
    }
    if args.reference_predictions:
        summary["agreement_with_saved_predictions"] = compare_with_reference(
            helper, args.reference_predictions, predictions)
    helper.write_json(summary_path, summary)
    qwk = "undefined" if metrics["qwk"] is None else f"{metrics['qwk']:.4f}"
    print(f"{args.name}: accuracy {metrics['accuracy']:.2%}, QWK {qwk}, "
          f"median {summary['full_answer_timing']['median_seconds']:.3f} s per image.", flush=True)


if __name__ == "__main__":
    main()
