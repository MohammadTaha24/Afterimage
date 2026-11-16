"""Score one fine-tuned RetiZero checkpoint on one cohort, with timing and parameter counts.

Accepts the app's checkpoint (best_retizero_dr.pth) or a checkpoint written by train_retizero.py.
Writes predictions.csv (five probabilities per image), timing.csv and summary.json to the output
folder. The predictions come from a batched pass; a second pass at batch size 1 measures the time
per image, which is the figure comparable with Qwen's per-image time.

Normally started by retizero_runs.py from notebooks/02_retizero_runs.ipynb.
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retizero_reproduction as helper

WARM_UP = 3  # images left out of the timing statistics, as in the Qwen benchmark


def spread(values):
    ordered = sorted(values)
    return {"images": len(ordered), "median_seconds": statistics.median(ordered),
            "mean_seconds": statistics.fmean(ordered),
            "p95_seconds": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
            "min_seconds": ordered[0], "max_seconds": ordered[-1]}


def agreement(first, second):
    """Grade mismatches and the largest probability difference between two prediction lists."""
    if [(a["image_id"], int(a["y_true"])) for a in first] != [(b["image_id"], int(b["y_true"])) for b in second]:
        return {"comparable": False, "reason": "different image IDs, order or grades"}
    return {"comparable": True, "images": len(first),
            "grade_mismatches": sum(int(a["y_pred"]) != int(b["y_pred"]) for a, b in zip(first, second)),
            "largest_probability_difference": max(abs(float(a[f"p{i}"]) - float(b[f"p{i}"]))
                                                  for a, b in zip(first, second) for i in range(helper.GRADES))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("score",))
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cohort-csv", required=True, type=Path)
    parser.add_argument("--cohort-name", required=True)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--limit", default=0, type=int, help="score only the first N images (pilot)")
    parser.add_argument("--timing-images", default=0, type=int, help="images in the batch-1 timing pass; 0 = all")
    parser.add_argument("--reference-predictions", type=Path,
                        help="saved predictions for the same checkpoint and cohort, to compare with")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0 or args.timing_images < 0:
        parser.error("Use a positive batch size and non-negative limits.")
    output = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    rows = helper.cohort_rows(args.cohort_csv, args.limit)
    timing_rows = rows[:args.timing_images] if args.timing_images else rows
    identity = {
        "name": args.name, "checkpoint": str(checkpoint), "checkpoint_sha256": helper.sha(checkpoint),
        "cohort": args.cohort_name, "cohort_csv": str(args.cohort_csv.resolve()),
        "cohort_csv_sha256": helper.sha(args.cohort_csv), "images": len(rows),
        "images_sha256": helper.images_digest(rows), "limit": args.limit, "batch_size": args.batch_size,
        "timing_images": len(timing_rows), "precision": "CUDA automatic mixed precision (FP16)",
        "code": helper.code_identity(__file__),
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    predictions_path = output / "predictions.csv"
    if summary_path.is_file():
        saved = helper.read_json(summary_path)
        if saved["identity"] != identity or saved["predictions_sha256"] != helper.sha(predictions_path):
            raise ValueError(f"Inputs changed since this folder was written. Use a fresh folder: {output}")
        print(f"Saved scores verified: {summary_path}")
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("RetiZero scoring needs a CUDA GPU.")
    backend = helper.configure_backend(torch)
    print(f"Scoring {len(rows)} {args.cohort_name} images with {checkpoint.name}.", flush=True)

    started = time.perf_counter()
    model = helper.build_model(torch)
    layout = helper.load_fine_tuned(model, checkpoint, torch)
    model.to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    counts = model.parameter_counts
    print(f"Model ready in {load_seconds:.1f} s: {counts['total']:,} parameters.", flush=True)
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        model(torch.zeros(2, 3, helper.IMAGE_SIZE, helper.IMAGE_SIZE, device="cuda"))  # untimed warm-up
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions, _ = helper.predict(model, rows, args.batch_size, torch,
                                    progress=lambda done: print(f"Scored {done}/{len(rows)}.", flush=True))
    torch.cuda.synchronize()
    batched_seconds = time.perf_counter() - started
    batched_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    transform = helper.eval_transform()
    timing, single = [], []
    with torch.inference_mode():
        for row in timing_rows:
            start = time.perf_counter()
            pixels = helper.load_image(row["image_path"], transform)
            loaded = time.perf_counter()
            with torch.amp.autocast("cuda"):
                logits = model(pixels.unsqueeze(0).to("cuda"))
            values = logits.float().softmax(dim=1)[0].cpu().tolist()
            torch.cuda.synchronize()
            end = time.perf_counter()
            timing.append({"image_id": row["id_code"], "read_and_transform_seconds": loaded - start,
                           "model_seconds": end - loaded, "total_seconds": end - start})
            single.append({"image_id": row["id_code"], "y_true": int(row["label"]),
                           "y_pred": max(range(helper.GRADES), key=values.__getitem__),
                           **{f"p{i}": value for i, value in enumerate(values)}})
    single_peak = torch.cuda.max_memory_allocated()
    timed = timing[WARM_UP:] if len(timing) > WARM_UP else timing

    metrics = helper.calculate_metrics(predictions)
    helper.write_csv(predictions_path, predictions, helper.PREDICTION_FIELDS)
    helper.write_csv(output / "timing.csv", timing)
    summary = {
        "identity": identity, "checkpoint_layout": layout, "backend": backend,
        "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "metrics": metrics, "parameters": counts, "model_load_seconds": load_seconds,
        "batched": {"batch_size": args.batch_size, "seconds": batched_seconds,
                    "images_per_second": len(rows) / batched_seconds, "torch_peak_allocated_bytes": batched_peak,
                    "includes": "reading and transforming the saved 224-pixel images"},
        "single_image_timing": {
            "warm_up_images_excluded": len(timing) - len(timed),
            "model": spread([item["model_seconds"] for item in timed]),
            "read_and_transform": spread([item["read_and_transform_seconds"] for item in timed]),
            "total": spread([item["total_seconds"] for item in timed]),
            "torch_peak_allocated_bytes": single_peak,
            "note": "Batch size 1. Model time covers transfer to the GPU, the forward pass and softmax. "
                    "Cropping and resizing the raw photograph happened when the dataset was built and is not timed.",
        },
        "agreement_batch1_vs_batched": agreement(single, predictions[:len(single)]),
        "predictions_sha256": helper.sha(predictions_path),
    }
    if args.reference_predictions is not None:
        reference = helper.read_csv(args.reference_predictions)
        summary["agreement_with_saved_predictions"] = {
            "reference": str(args.reference_predictions.resolve()),
            "reference_sha256": helper.sha(args.reference_predictions), **agreement(predictions, reference)}
    helper.write_json(summary_path, summary)
    auc = metrics["macro_ovr_auc"]
    print(f"{args.cohort_name}: accuracy {metrics['accuracy']:.4%}, QWK {metrics['qwk']}, macro AUC {auc}.")
    print(f"Batch 1: {1000 * summary['single_image_timing']['model']['median_seconds']:.1f} ms per image (median). "
          f"Batch {args.batch_size}: {summary['batched']['images_per_second']:.0f} images per second.")


if __name__ == "__main__":
    main()
