"""Score a CLAHE adapter with BF16 on the original 4-bit base."""

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import paper_reproduction as original


def score_full_answers(model, processor, image_path):
    """Use the archived prompt, masking and likelihood calculation with BF16 inputs."""
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    prompt = [{"role": "user", "content": [
        {"type": "text", "text": original.INSTRUCTION}, {"type": "image", "image": image},
    ]}]

    def encode(messages, generation_prompt):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
        batch = processor(images=image, text=text, return_tensors="pt").to(model.device)
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)
        return batch

    prompt_length = encode(prompt, True)["input_ids"].shape[1]
    scores = []
    with torch.no_grad():
        for candidate in original.CANDIDATES:
            messages = prompt + [{"role": "assistant", "content": [{"type": "text", "text": candidate}]}]
            batch = encode(messages, False)
            labels = batch["input_ids"].clone()
            labels[:, :prompt_length] = -100
            count = int((labels != -100).sum().item())
            if count == 0:
                raise ValueError("No answer tokens remain after prompt masking.")
            score = -float(model(**batch, labels=labels).loss.item()) * count
            if not math.isfinite(score):
                raise ValueError(f"Non-finite BF16 answer likelihood for {image_path}, candidate {candidate}")
            scores.append(score)
    weights = np.exp(np.asarray(scores, dtype=np.float64) - max(scores))
    return weights / weights.sum()


def evaluate(adapter, base_snapshot, test_csv, output_dir):
    # Reuse the original export, metric and resume logic with an explicit BF16 identity.
    adapter, weights = original.validate_adapter(adapter)
    original.validate_evaluation_cohort(test_csv, "aptos", original.APTOS_COUNTS["test"])
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "workflow_version": original.WORKFLOW_VERSION,
        "adapter_dir": str(adapter),
        "adapter_sha256": original.sha256_file(weights),
        "base_snapshot": str(base_snapshot.resolve()),
        "prediction_method": "full_answer_logprob_argmax",
        "evaluation_precision": "BF16",
        "helper_sha256": original.sha256_file(Path(original.__file__)),
        "scorer_sha256": original.sha256_file(__file__),
        "csv_sha256": {"aptos_test": original.sha256_file(test_csv)},
        "package_versions": original.package_versions(),
    }
    manifest = output_dir / "evaluation_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
        raise ValueError(f"Evaluation settings changed. Preserve the existing results in {output_dir}.")
    original.write_json(manifest, identity)
    if (output_dir / "aptos_test.summary.json").is_file():
        rows = original.read_csv(output_dir / "aptos_test.predictions.csv")
        expected = original.load_test_rows(test_csv)
        original.validate_prediction_prefix(rows, expected)
        if len(rows) != len(expected):
            raise ValueError("Completed BF16 evaluation has incomplete predictions.")
        print("BF16 evaluation already complete.", flush=True)
        return

    original.configure_process_caches(output_dir, evaluation=True)
    import pandas  # Match the historical loader's import order.
    from unsloth import FastVisionModel
    from peft import PeftModel
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This scorer requires a BF16-capable CUDA GPU.")
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(base_snapshot), load_in_4bit=True, local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    FastVisionModel.for_inference(model)
    model.to(dtype=torch.bfloat16)
    model.eval()
    print("Scoring with BF16 inference on the 4-bit base.", flush=True)
    with patch.object(original, "score_full_answers", score_full_answers):
        original._evaluate_subset(model, processor, test_csv, output_dir, "aptos_test", identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate",))
    for name in ("adapter", "base-snapshot", "test-csv", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    args.pop("stage")
    evaluate(**args)
