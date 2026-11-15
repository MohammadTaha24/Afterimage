"""Train and evaluate the RetiZero control on the matched APTOS split."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import runpy
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path


SPLITS = {"train": 2786, "val": 357, "test": 361}
EPOCHS = 15
PATIENCE = 4
LEARNING_RATE = 7.68094483149829e-5
WEIGHT_DECAY = 1e-4
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CheckpointPrerequisiteError(ValueError):
    """The pretrained weights or their pinned checksum are unavailable."""


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, data):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_split(record, split, source):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError(f"Changed {source} split CSV: {path}")
    rows = read_csv(path)
    if len(rows) != SPLITS[split] or record["rows"] != len(rows):
        raise ValueError(f"Wrong {source} split size: {split}")
    if any(row["split"] != split or row["source_dataset"] != "aptos" for row in rows):
        raise ValueError(f"Wrong cohort in {source} {split}")
    return rows


def matched_tables(dataset, qwen_dataset):
    manifest = read_json(dataset)
    qwen = read_json(qwen_dataset)
    if manifest["image_size"] != 224 or qwen["image_size"] != 512:
        raise ValueError("Expected a 224-pixel RetiZero dataset and the matched 512-pixel Qwen dataset.")
    if set(manifest["branches"]) != {"off"}:
        raise ValueError("The RetiZero dataset must use the standard, CLAHE-off build.")
    audit_path = dataset.with_name("pairing_audit.csv")
    if sha(audit_path) != manifest["pairing_audit_sha256"]:
        raise ValueError("RetiZero image audit changed.")
    audit_rows = read_csv(audit_path)
    audits = {row["image_id"]: row for row in audit_rows}
    if len(audit_rows) != sum(SPLITS.values()) or len(audits) != len(audit_rows):
        raise ValueError("Unexpected or repeated IDs in the RetiZero image audit.")
    seen = set()
    tables = {}
    for split in SPLITS:
        rows = read_split(manifest["branches"]["off"][split], split, "RetiZero")
        qwen_rows = read_split(qwen["branches"]["off"][split], split, "Qwen")
        if [(r["id_code"], r["label"]) for r in rows] != [(r["id_code"], r["label"]) for r in qwen_rows]:
            raise ValueError(f"The ordered {split} IDs and grades differ between RetiZero and Qwen.")
        for row in rows:
            ident = row["id_code"]
            binding = audits[ident]
            if (ident in seen or row["label"] not in {"0", "1", "2", "3", "4"}
                    or (row["split"], row["label"], row["image_path"])
                    != (binding["split"], binding["grade"], binding["off_path"])):
                raise ValueError(f"Invalid APTOS row: {ident}")
            if sha(row["image_path"]) != binding["off_sha256"]:
                raise ValueError(f"RetiZero image changed: {ident}")
            seen.add(ident)
        tables[split] = rows
    return tables


def external_table(path, expected_sha):
    if not expected_sha or sha(path) != expected_sha.lower():
        raise ValueError("Supply the verified external cohort CSV and its SHA-256.")
    rows = read_csv(path)
    if (not rows or any(row["source_dataset"] not in {"messidor", "messidor_external"}
                        or row["split"] != "test" or row["label"] not in {"0", "1", "2", "3", "4"}
                        or not row["id_code"] or not row["patient_id"] for row in rows)
            or len({row["id_code"] for row in rows}) != len(rows)):
        raise ValueError("The external CSV must contain unique graded Messidor test IDs with patient IDs.")
    return rows


def configuration(args):
    repo = args.repo_root
    package = repo / "RetiZero/RetiZero"
    paths = (
        "iden_modules/modeling/LoraRETFound.py",
        "iden_modules/modeling/LORA/lora_image_encoder.py",
        "iden_modules/modeling/models_vit.py",
        "iden_modules/modeling/timm_models/models/vision_transformer.py",
    )
    if not args.weights.is_file():
        raise CheckpointPrerequisiteError(
            f"Put the official pretrained RetiZero.pth at {args.weights} "
            "or set RETIZERO_WEIGHTS in notebook Setup."
        )
    expected_sha = args.weights_sha256.strip().lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise CheckpointPrerequisiteError("Set RETIZERO_WEIGHTS_SHA256 in notebook Setup to the checkpoint's SHA-256.")
    digest = sha(args.weights)
    if digest != expected_sha:
        raise CheckpointPrerequisiteError("The pretrained checkpoint SHA-256 differs from the pinned value.")
    return {
        "model": "Pretrained RetiZero vision encoder (LoRA rank 8) and fresh five-grade head",
        "base_weights": str(args.weights),
        "base_sha256": digest,
        "source_files": {str(package / path): sha(package / path) for path in paths},
        "runner_sha256": sha(__file__),
        "metric_source_sha256": sha(repo / "finetuning/Qwen3/paper_reproduction.py"),
        "datasets": {str(path): sha(path) for path in (args.dataset, args.qwen_dataset)},
        "packages": {name: version(name) for name in ("torch", "torchvision", "transformers", "numpy", "Pillow")},
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss with inverse training class frequencies",
        "precision": "CUDA AMP FP16",
        "training_transforms": "RandomResizedCrop(224, scale=(0.8, 1.0)), horizontal flip, rotation 180 degrees",
        "eval_transforms": "Resize(224,224), tensor conversion, then ImageNet normalization",
        "selection": "Highest APTOS validation accuracy. Ties keep the earlier epoch.",
    }


def build_model(args, torch):
    # The historical classifier calls this vision encoder directly. The text
    # encoder and projection head do not participate in its forward pass.
    sys.path.insert(0, str(args.repo_root / "RetiZero/RetiZero"))
    from iden_modules.modeling.LoraRETFound import lora

    nn = torch.nn
    pretrained = torch.load(args.weights, map_location="cpu", weights_only=True)
    if not isinstance(pretrained, dict) or any(k.startswith(("classifier.", "retizero.", "img_encoder.")) for k in pretrained):
        raise ValueError("Expected the original pretrained CLIPRModel state dict, without a trained DR classifier.")
    prefix = "vision_model.model."
    state = {k[len(prefix):]: v for k, v in pretrained.items() if k.startswith(prefix)}
    if not state:
        raise ValueError("No pretrained vision encoder tensors found.")
    backbone = lora(pretrained=False, R=8)
    backbone.load_state_dict(state, strict=True)
    backbone.requires_grad_(True)
    del pretrained, state
    return nn.Sequential(backbone, nn.Linear(1024, 5)).to("cuda")


def loader(rows, batch_size, train, torch):
    from PIL import Image
    from torchvision import transforms

    train_ops = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
    ]
    ops = train_ops if train else [transforms.Resize((224, 224))]
    transform = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    class FundusImages(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            with Image.open(row["image_path"]) as source:
                if source.size != (224, 224):
                    raise ValueError(f"Expected a 224 x 224 standard image: {row['image_path']}")
                pixels = transform(source.convert("RGB"))
            return pixels, int(row["label"])

    return torch.utils.data.DataLoader(
        FundusImages(), batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=True,
    )


def save_checkpoint(torch, path, value):
    temporary = path.with_suffix(".pth.partial")
    torch.save(value, temporary)
    temporary.replace(path)


def validation_accuracy(model, batches, torch):
    model.eval()
    correct = 0
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for images, grades in batches:
            predictions = model(images.to("cuda", non_blocking=True)).argmax(1).cpu()
            correct += int((predictions == grades).sum())
    return correct / len(batches.dataset)


def train(args, tables, config, torch):
    run = args.run_dir
    latest = run / "checkpoint.pth"
    if (run / "train_result.json").is_file():
        verified_best(run, config)
        print("Saved training result and best checkpoint verified.")
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = build_model(args, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    train_loader = loader(tables["train"], args.batch_size, True, torch)
    val_loader = loader(tables["val"], args.batch_size, False, torch)
    counts = Counter(int(row["label"]) for row in tables["train"])
    weights = torch.tensor([len(tables["train"]) / (5 * counts[g]) for g in range(5)], device="cuda")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    history, best, best_epoch, stalled = [], -1.0, 0, 0
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["configuration"] != config:
            raise ValueError("Cannot resume a checkpoint with different inputs or settings.")
        model.load_state_dict(saved.pop("model"))
        optimizer.load_state_dict(saved.pop("optimizer"))
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_gpu"])
        history, best, best_epoch, stalled = (saved[key] for key in ("history", "best", "best_epoch", "stalled"))
        best_weights = run / f"best_epoch_{best_epoch:02d}.pth"
        best_sha = sha(best_weights)
        if best_sha != saved["best_sha256"]:
            raise ValueError("Saved best epoch weights changed.")
        del saved
        print(f"Resuming after epoch {len(history)}.", flush=True)
    print(f"Training on {len(tables['train'])} images with seed {args.seed}, batch {args.batch_size}.", flush=True)

    for epoch in range(len(history) + 1, EPOCHS + 1):
        if stalled >= PATIENCE:
            break
        model.train()
        running_loss = 0.0
        for step, (images, grades) in enumerate(train_loader, 1):
            images = images.to("cuda", non_blocking=True)
            grades = grades.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), grades)
            value = loss.detach().item()
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}, step {step}.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += value * len(grades)
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{EPOCHS}, batch {step}/{len(train_loader)}, loss {value:.5f}", flush=True)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        accuracy = validation_accuracy(model, val_loader, torch)
        history.append({"epoch": epoch, "learning_rate": lr, "train_loss": running_loss / len(tables["train"]),
                        "validation_accuracy": accuracy})
        print(f"Epoch {epoch}: validation accuracy {accuracy:.4%}", flush=True)
        if accuracy > best:
            best, best_epoch, stalled = accuracy, epoch, 0
            best_weights = run / f"best_epoch_{epoch:02d}.pth"
            save_checkpoint(torch, best_weights, model.state_dict())
            best_sha = sha(best_weights)
        else:
            stalled += 1
        save_checkpoint(torch, latest, {
            "configuration": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "best": best, "best_epoch": best_epoch, "best_sha256": best_sha, "stalled": stalled,
            "rng_cpu": torch.get_rng_state(), "rng_gpu": torch.cuda.get_rng_state(),
        })
    write_json(run / "train_result.json", {"configuration": config, "epochs_completed": len(history), "history": history,
        "best_epoch": best_epoch, "best_validation_accuracy": best,
        "checkpoint": best_weights.name, "checkpoint_sha256": best_sha})
    print(f"Selected epoch {best_epoch} by validation accuracy {best:.4%}.", flush=True)


def verified_best(run, config):
    record = read_json(run / "train_result.json")
    if record["configuration"] != config:
        raise ValueError("Saved training settings differ from current inputs.")
    name = f"best_epoch_{record['best_epoch']:02d}.pth"
    path = run / name
    if record["checkpoint"] != name or sha(path) != record["checkpoint_sha256"]:
        raise ValueError("Saved best checkpoint path or hash changed.")
    return path, record["checkpoint_sha256"]


def evaluate(args, rows, config, torch):
    weights, digest = verified_best(args.run_dir, config)
    source = "messidor_external" if args.external_csv else "aptos_test"
    output = args.run_dir / ("evaluation_external" if args.external_csv else "evaluation")
    output.mkdir(exist_ok=True)
    manifest_path = output / "evaluation_manifest.json"
    data_sha = {row["id_code"]: sha(row["image_path"]) for row in rows} if args.external_csv else None
    identity = {"checkpoint_sha256": digest, "configuration": config, "samples": len(rows),
                "cohort_csv_sha256": args.external_sha256, "external_images_sha256": data_sha}
    if manifest_path.is_file() and identity != read_json(manifest_path):
        raise ValueError("Evaluation inputs changed. Use a fresh evaluation folder.")
    write_json(manifest_path, identity)
    predictions_path = output / f"{source}.predictions.csv"
    summary_path = output / f"{source}.summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary["predictions_sha256"] != sha(predictions_path):
            raise ValueError("Saved predictions changed.")
        print(f"Saved {source} evaluation verified.")
        return

    model = build_model(args, torch)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    batches = loader(rows, args.batch_size, False, torch)
    predictions = []
    with torch.inference_mode():
        for images, grades in batches:
            with torch.amp.autocast("cuda"):
                logits = model(images.to("cuda", non_blocking=True))
            scores = logits.float().softmax(dim=1).cpu()
            for grade, values in zip(grades.tolist(), scores.tolist()):
                source_row = rows[len(predictions)]
                predictions.append({"image_id": source_row["id_code"], "image_path": source_row["image_path"],
                    "y_true": grade, "y_pred": max(range(5), key=values.__getitem__),
                    **{f"p{i}": value for i, value in enumerate(values)}})
            print(f"Scored {len(predictions)}/{len(rows)} {source} images.", flush=True)
    metric_path = args.repo_root / "finetuning/Qwen3/paper_reproduction.py"
    metrics = runpy.run_path(str(metric_path))["calculate_metrics"](predictions)
    temporary = predictions_path.with_suffix(".csv.partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    temporary.replace(predictions_path)
    write_json(summary_path, {**metrics, "predictions_sha256": sha(predictions_path), "checkpoint_sha256": digest})
    print(f"{source}: accuracy {metrics['accuracy']:.4%}, QWK {metrics['qwk']}, AUC {metrics['macro_ovr_auc']}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train", "evaluate"))
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="224-pixel duplicate-excluded dataset manifest")
    parser.add_argument("--qwen-dataset", required=True, type=Path, help="512-pixel duplicate-excluded dataset manifest")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--external-csv", type=Path, help="Verified 224-pixel standard Messidor test CSV")
    parser.add_argument("--external-sha256")
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.seed < 2**32:
        parser.error("Supply a positive batch size and a seed from 0 through 2**32-1.")
    if bool(args.external_csv) != bool(args.external_sha256) or (args.external_csv and args.action != "evaluate"):
        parser.error("Use --external-csv and --external-sha256 together with evaluate.")
    for name in ("repo_root", "dataset", "qwen_dataset", "weights", "run_dir"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        config = configuration(args)
    except CheckpointPrerequisiteError as error:
        message = f"RetiZero control is blocked. {error}"
        if args.action == "check":
            print(message, flush=True)
            return
        raise SystemExit(message) from None
    tables = matched_tables(args.dataset, args.qwen_dataset)
    print(f"Verified matched APTOS train/val/test counts: {list(map(len, tables.values()))}.", flush=True)
    if args.action == "check":
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("RetiZero training and scoring require a CUDA GPU.")
    config["gpu"] = torch.cuda.get_device_name(0)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "configuration.json"
    if config_path.is_file() and read_json(config_path) != config:
        raise ValueError("Run configuration changed. Choose a fresh run folder.")
    if args.action == "train":
        write_json(config_path, config)
        train(args, tables, config, torch)
    else:
        rows = external_table(args.external_csv, args.external_sha256) if args.external_csv else tables["test"]
        evaluate(args, rows, config, torch)


if __name__ == "__main__":
    main()
