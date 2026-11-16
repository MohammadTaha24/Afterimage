"""Train and evaluate one RetiZero fine-tuning run: check, train, evaluate.

Follows the original notebook's recipe (legacy/ICT302_Retizero_Training_Unbalanced_Aptos.ipynb):
RetiZero vision encoder with every weight trainable plus a new five-grade head, AdamW, cosine
schedule, mixed precision, cross-entropy weighted by inverse class frequency, training-only
augmentation, selection on validation accuracy. The split, CLAHE branch, seed and batch settings
are arguments and are recorded in the run's configuration.

Normally started by retizero_runs.py from notebooks/02_retizero_runs.ipynb.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retizero_reproduction as helper

RANDOM = "random initialisation (pilot only, not a result)"


class Blocked(ValueError):
    """The pretrained weights or their pinned checksum are unavailable."""


def initial_weights(args):
    if args.random_init:
        return {"kind": RANDOM, "path": None, "sha256": None}
    if args.weights is None or not args.weights.is_file():
        raise Blocked(f"Put the official pretrained RetiZero.pth at {args.weights} "
                      "(download link: RetiZero/RetiZero/README.md) and set its SHA-256 in notebook Setup.")
    expected = (args.weights_sha256 or "").strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise Blocked("Set the pretrained checkpoint's SHA-256 in notebook Setup.")
    digest = helper.sha(args.weights)
    if digest != expected:
        raise Blocked("The pretrained checkpoint's SHA-256 differs from the pinned value.")
    return {"kind": "official pretrained RetiZero", "path": str(args.weights), "sha256": digest}


def configuration(args, dataset, initial):
    pilot = None
    if args.pilot:
        pilot = {"train_per_grade": args.pilot_train_per_grade, "val_per_grade": args.pilot_val_per_grade,
                 "test_per_grade": args.pilot_test_per_grade}
    return {
        "model": "RetiZero vision encoder (ViT-L/16 with LoRA rank 8 branches, every weight trainable) "
                 "and a new five-grade linear head",
        "initial_weights": initial,
        "code": helper.code_identity(__file__),
        "dataset": dataset,
        "pilot": pilot,
        "training": {
            "seed": args.seed, "effective_batch": args.effective_batch, "physical_batch": args.physical_batch,
            "eval_batch": args.eval_batch, "learning_rate": args.learning_rate,
            "weight_decay": helper.RECIPE["weight_decay"], "epochs": args.epochs, "patience": args.patience,
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR over the epoch limit, stepped once per epoch",
            "loss": "Cross-entropy weighted by inverse training class frequency, mean over each group",
            "precision": "CUDA automatic mixed precision (FP16)",
            "training_transforms": "RandomResizedCrop(224, scale=(0.8, 1.0)), horizontal flip, rotation up to "
                                   "180 degrees, ImageNet normalisation",
            "eval_transforms": "Resize(224, 224), ImageNet normalisation",
            "selection": "Highest validation accuracy; ties keep the earlier epoch",
            "loader_workers": 0,
        },
    }


def new_model(args, torch):
    """Seeded model with its starting weights, on the GPU."""
    helper.seed_everything(args.seed, torch)
    model = helper.build_model(torch)
    if not args.random_init:
        helper.load_pretrained(model, args.weights, torch)
    return model.to("cuda")


def evaluate(args, rows, config, torch):
    weights, digest = helper.verified_best(args.run_dir, config)
    output = args.run_dir / "evaluation"
    output.mkdir(exist_ok=True)
    identity = {"checkpoint_sha256": digest, "configuration": config, "samples": len(rows),
                "images_sha256": helper.images_digest(rows)}
    manifest_path = output / "evaluation_manifest.json"
    if manifest_path.is_file() and identity != helper.read_json(manifest_path):
        raise ValueError("Evaluation inputs changed. Use a fresh run folder.")
    helper.write_json(manifest_path, identity)
    predictions_path = output / "aptos_test.predictions.csv"
    summary_path = output / "aptos_test.summary.json"
    if summary_path.is_file():
        if helper.read_json(summary_path)["predictions_sha256"] != helper.sha(predictions_path):
            raise ValueError("Saved predictions changed.")
        print("Saved APTOS test evaluation verified.")
        return
    model = helper.build_model(torch)
    helper.load_fine_tuned(model, weights, torch)
    model.to("cuda")
    predictions, _ = helper.predict(model, rows, config["training"]["eval_batch"], torch,
                                    progress=lambda done: print(f"Scored {done}/{len(rows)} test images.", flush=True))
    metrics = helper.calculate_metrics(predictions)
    helper.write_csv(predictions_path, predictions, helper.PREDICTION_FIELDS)
    if config["pilot"]:
        helper.write_csv(output / "pilot_test.csv", rows)
    helper.write_json(summary_path, {**metrics, "predictions_sha256": helper.sha(predictions_path),
                                     "checkpoint_sha256": digest})
    print(f"APTOS test ({metrics['samples']} images): accuracy {metrics['accuracy']:.4%}, QWK {metrics['qwk']}, "
          f"macro AUC {metrics['macro_ovr_auc']}.")


def main():
    recipe = helper.RECIPE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "train", "evaluate"))
    parser.add_argument("--dataset", required=True, type=Path, help="manifest.json of a 224-pixel APTOS build")
    parser.add_argument("--branch", default="off", choices=("off", "on"), help="CLAHE branch of that build")
    parser.add_argument("--matched-dataset", type=Path,
                        help="512-pixel Qwen manifest of the same split; ordered IDs and grades must agree")
    parser.add_argument("--weights", type=Path, help="official pretrained RetiZero.pth")
    parser.add_argument("--weights-sha256")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--effective-batch", default=recipe["effective_batch"], type=int,
                        help="images per optimizer step (the original used 128)")
    parser.add_argument("--physical-batch", default=32, type=int, help="images per forward pass")
    parser.add_argument("--eval-batch", default=64, type=int)
    parser.add_argument("--learning-rate", default=recipe["learning_rate"], type=float)
    parser.add_argument("--epochs", default=recipe["epochs"], type=int)
    parser.add_argument("--patience", default=recipe["patience"], type=int)
    parser.add_argument("--pilot", action="store_true", help="train and score on a small balanced subset")
    parser.add_argument("--pilot-train-per-grade", default=32, type=int)
    parser.add_argument("--pilot-val-per-grade", default=4, type=int)
    parser.add_argument("--pilot-test-per-grade", default=3, type=int)
    parser.add_argument("--random-init", action="store_true",
                        help="pilot only: start from random weights when the pretrained checkpoint is not available")
    parser.add_argument("--stop-after-epoch", type=int, help="train only: stop after this epoch (resume test)")
    args = parser.parse_args()
    if min(args.effective_batch, args.physical_batch, args.eval_batch, args.epochs, args.patience) < 1:
        parser.error("Batch sizes, epochs and patience must be positive.")
    if not 0 <= args.seed < 2**32 or args.learning_rate <= 0:
        parser.error("Supply a seed from 0 through 2**32-1 and a positive learning rate.")
    if args.random_init and not args.pilot:
        parser.error("--random-init is only allowed with --pilot.")
    if args.stop_after_epoch is not None and args.action != "train":
        parser.error("--stop-after-epoch belongs to train.")
    for name in ("dataset", "matched_dataset", "weights", "run_dir"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())

    tables, dataset = helper.dataset_tables(args.dataset, args.branch, matched_manifest=args.matched_dataset)
    print(f"Verified APTOS {dataset['split_name']} (CLAHE {args.branch}): "
          f"{', '.join(f'{len(tables[s])} {s}' for s in helper.SPLITS)}.", flush=True)
    rows = tables
    if args.pilot:
        rows = {"train": helper.balanced_subset(tables["train"], args.pilot_train_per_grade),
                "val": helper.balanced_subset(tables["val"], args.pilot_val_per_grade),
                "test": helper.balanced_subset(tables["test"], args.pilot_test_per_grade)}
        print(f"Pilot subset: {', '.join(f'{len(rows[s])} {s}' for s in helper.SPLITS)}.", flush=True)
    try:
        initial = initial_weights(args)
    except Blocked as error:
        message = f"RetiZero training is blocked. {error}"
        if args.action == "check":
            print(message, flush=True)
            return
        raise SystemExit(message) from None
    if args.random_init:
        print(f"Starting weights: {RANDOM}.", flush=True)
    config = configuration(args, dataset, initial)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if args.action == "check":
        model = helper.build_model(torch)
        if not args.random_init:
            loaded = helper.load_pretrained(model, args.weights, torch)
            print(f"The pretrained checkpoint fits the encoder ({loaded['encoder_tensors_loaded']} tensors).")
        counts = model.parameter_counts
        print(f"Model: {counts['total']:,} parameters, {counts['trainable']:,} trained "
              f"({counts['lora_branches']:,} of them in LoRA branches).")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("RetiZero training and scoring need a CUDA GPU.")
    config["backend"] = helper.configure_backend(torch)
    config["gpu"] = torch.cuda.get_device_name(0)
    config["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "configuration.json"
    if config_path.is_file() and helper.read_json(config_path) != config:
        raise ValueError("The run configuration changed. Choose a fresh run folder.")
    if args.action == "train":
        helper.write_json(config_path, config)
        if (args.run_dir / "train_result.json").is_file():
            helper.verified_best(args.run_dir, config)
            print("Saved training result and best checkpoint verified.")
            return
        helper.train(args.run_dir, rows, config, new_model(args, torch), torch,
                     stop_after_epoch=args.stop_after_epoch)
    else:
        evaluate(args, rows["test"], config, torch)


if __name__ == "__main__":
    main()
