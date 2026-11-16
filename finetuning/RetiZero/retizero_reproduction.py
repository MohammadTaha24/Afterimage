"""Shared code for the RetiZero fine-tuning reproduction.

Model construction, checkpoint handling, dataset checks, transforms, the
training loop and scoring live here, so every worker in paper_revision/ uses the
same code. Metrics come from finetuning/Qwen3/paper_reproduction.py, the same
function that scores Qwen.

Once full runs exist, the hash of this file is part of their recorded identity.
Do not edit it after that; add a new file instead.
"""

import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDORED = ROOT / "RetiZero/RetiZero"
METRIC_SOURCE = ROOT / "finetuning/Qwen3/paper_reproduction.py"
SOURCE_FILES = (
    "iden_modules/modeling/LoraRETFound.py",
    "iden_modules/modeling/LORA/lora_image_encoder.py",
    "iden_modules/modeling/models_vit.py",
    "iden_modules/modeling/timm_models/models/vision_transformer.py",
)
PACKAGES = ("torch", "torchvision", "transformers", "numpy", "Pillow")

GRADES = 5
FEATURES = 1024
LORA_RANK = 8
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test")
KNOWN_SPLITS = {
    "duplicate_excluded": {"train": 2786, "val": 357, "test": 361},
    "previous_split": {"train": 2929, "val": 366, "test": 367},
}
# The original notebook's settings (legacy/ICT302_Retizero_Training_Unbalanced_Aptos.ipynb, trial 3).
RECIPE = {"epochs": 15, "patience": 4, "learning_rate": 7.68094483149829e-5, "weight_decay": 1e-4,
          "effective_batch": 128}
# What that notebook printed for best_retizero_dr.pth on its 367 test images.
REPORTED = {
    "samples": 367, "accuracy": 0.8229, "qwk": 0.9097, "macro_ovr_auc": 0.9451,
    "macro_precision": 0.6661, "macro_recall": 0.6796, "macro_f1": 0.6724,
    "support": [181, 37, 100, 19, 30],
    "precision": [0.98, 0.65, 0.78, 0.33, 0.58],
    "recall": [0.97, 0.70, 0.76, 0.37, 0.60],
    "f1": [0.97, 0.68, 0.77, 0.35, 0.59],
}
PREDICTION_FIELDS = ["image_id", "image_path", "y_true", "y_pred", "p0", "p1", "p2", "p3", "p4"]
_metric_module = None


# ---------------------------------------------------------------- files

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def code_identity(*workers):
    """Hashes of everything that defines how a run is computed."""
    return {
        "helper_sha256": sha(__file__),
        "worker_sha256": {Path(path).name: sha(path) for path in workers},
        "metric_source_sha256": sha(METRIC_SOURCE),
        "vendored_source_sha256": {name: sha(VENDORED / name) for name in SOURCE_FILES},
        "packages": {name: version(name) for name in PACKAGES},
    }


# ---------------------------------------------------------------- metrics

def calculate_metrics(predictions):
    """Accuracy, QWK, macro AUC and per-grade figures from the pinned Qwen metric function."""
    global _metric_module
    if _metric_module is None:
        spec = importlib.util.spec_from_file_location("pinned_paper_reproduction", METRIC_SOURCE)
        _metric_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_metric_module)
    metrics = _metric_module.calculate_metrics(predictions)
    present = [item for item in metrics["per_class"] if item["support"]]
    metrics["macro_precision"] = sum(item["precision"] for item in metrics["per_class"]) / GRADES
    metrics["macro_recall"] = sum(item["recall"] for item in present) / len(present)
    return metrics


def compare_with_reported(metrics):
    """Compare a replay on the original 367 test images with the original notebook's printout."""
    checks = []
    for key in ("samples", "accuracy", "qwk", "macro_ovr_auc", "macro_precision", "macro_recall", "macro_f1"):
        value = metrics[key]
        shown = value if key == "samples" else round(value, 4)
        checks.append({"item": key, "reported": REPORTED[key], "replay": value, "match": shown == REPORTED[key]})
    for grade, item in enumerate(metrics["per_class"]):
        checks.append({"item": f"support_grade_{grade}", "reported": REPORTED["support"][grade],
                       "replay": item["support"], "match": item["support"] == REPORTED["support"][grade]})
        for key in ("precision", "recall", "f1"):
            checks.append({"item": f"{key}_grade_{grade}", "reported": REPORTED[key][grade], "replay": item[key],
                           "match": round(item[key], 2) == REPORTED[key][grade]})
    return {"all_match": all(check["match"] for check in checks), "checks": checks,
            "note": "Overall figures are compared at four decimals and per-grade figures at two, "
                    "the precision the original notebook printed."}


# ---------------------------------------------------------------- data

def dataset_tables(manifest_path, branch="off", *, matched_manifest=None, verify_images=True):
    """Read and verify one branch of a 224-pixel APTOS build. Returns the three tables and their identity."""
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    if manifest["image_size"] != IMAGE_SIZE:
        raise ValueError(f"Expected a {IMAGE_SIZE}-pixel dataset: {manifest_path}")
    if branch not in manifest["branches"]:
        raise ValueError(f"The dataset has no CLAHE-{branch} branch: {manifest_path}")
    records = manifest["branches"][branch]
    sizes = {split: records[split]["rows"] for split in SPLITS}
    split_name = next((name for name, known in KNOWN_SPLITS.items() if known == sizes), None)
    if split_name is None:
        raise ValueError(f"Unexpected split sizes {sizes}. Known splits: {KNOWN_SPLITS}")
    audit_path = manifest_path.with_name("pairing_audit.csv")
    if sha(audit_path) != manifest["pairing_audit_sha256"]:
        raise ValueError(f"The image audit changed: {audit_path}")
    audit_rows = read_csv(audit_path)
    audits = {row["image_id"]: row for row in audit_rows}
    if len(audit_rows) != sum(sizes.values()) or len(audits) != len(audit_rows):
        raise ValueError("Unexpected or repeated IDs in the image audit.")
    matched = read_json(matched_manifest)["branches"]["off"] if matched_manifest else None
    seen, tables, hashes = set(), {}, {}
    for split in SPLITS:
        path = Path(records[split]["path"])
        hashes[split] = sha(path)
        if hashes[split] != records[split]["sha256"]:
            raise ValueError(f"Changed split CSV: {path}")
        rows = read_csv(path)
        if len(rows) != sizes[split]:
            raise ValueError(f"Wrong number of rows in {path}")
        if matched is not None:
            other = read_csv(matched[split]["path"])
            if [(r["id_code"], r["label"]) for r in rows] != [(r["id_code"], r["label"]) for r in other]:
                raise ValueError(f"The ordered {split} IDs and grades differ from {matched_manifest}")
        for row in rows:
            ident = row["id_code"]
            binding = audits.get(ident)
            if (binding is None or ident in seen or row["label"] not in set("01234")
                    or row["split"] != split or row["source_dataset"] != "aptos"
                    or (row["split"], row["label"], row["image_path"])
                    != (binding["split"], binding["grade"], binding[f"{branch}_path"])):
                raise ValueError(f"Invalid APTOS row: {ident}")
            if verify_images and sha(row["image_path"]) != binding[f"{branch}_sha256"]:
                raise ValueError(f"Image bytes changed: {ident}")
            seen.add(ident)
        tables[split] = rows
    identity = {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "branch": branch,
                "split_name": split_name, "rows": sizes, "csv_sha256": hashes,
                "matched_manifest": str(matched_manifest) if matched_manifest else None,
                "matched_manifest_sha256": sha(matched_manifest) if matched_manifest else None}
    return tables, identity


def cohort_rows(csv_path, limit=0):
    """Rows of any scoring cohort: id_code, image_path and a grade from 0 to 4."""
    rows = read_csv(csv_path)
    if not rows or any(not row.get("id_code") or not row.get("image_path") or row.get("label") not in set("01234")
                       for row in rows):
        raise ValueError(f"The cohort needs id_code, image_path and a grade from 0 to 4 on every row: {csv_path}")
    if len({row["id_code"] for row in rows}) != len(rows):
        raise ValueError(f"Repeated image IDs in {csv_path}")
    return rows[:limit] if limit else rows


def images_digest(rows):
    """One hash for the bytes of every image in a cohort, in order."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['id_code']},{sha(row['image_path'])}\n".encode())
    return digest.hexdigest()


def balanced_subset(rows, per_grade):
    """The first per_grade rows of each grade, in their original order (pilot runs)."""
    taken = Counter()
    subset = []
    for row in rows:
        if taken[row["label"]] < per_grade:
            taken[row["label"]] += 1
            subset.append(row)
    if len(taken) != GRADES or min(taken.values()) < per_grade:
        raise ValueError(f"Not enough images for {per_grade} per grade.")
    return subset


def eval_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def train_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def load_image(path, transform):
    from PIL import Image
    with Image.open(path) as source:
        if source.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected a {IMAGE_SIZE} x {IMAGE_SIZE} image: {path}")
        return transform(source.convert("RGB"))


def loader(rows, batch_size, transform, torch, order=None):
    """Batches of (images, grades) on the CPU. `order` fixes the sample order for one training epoch."""

    class FundusImages(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return load_image(rows[index]["image_path"], transform), int(rows[index]["label"])

    return torch.utils.data.DataLoader(FundusImages(), batch_size=batch_size, sampler=order, shuffle=False,
                                       num_workers=0, pin_memory=True)


# ---------------------------------------------------------------- model

def build_model(torch):
    """The original notebook's model: RetiZero vision encoder, every weight trainable, and a new linear head.

    The notebook builds the full CLIPRModel and then uses only vision_model.model, which upstream creates
    with this same lora() call. Building it directly avoids loading the text encoder.
    """
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))
    from iden_modules.modeling.LoraRETFound import lora

    encoder = lora(pretrained=False, R=LORA_RANK)
    lora_branches = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    encoder.requires_grad_(True)
    model = torch.nn.Sequential(encoder, torch.nn.Linear(FEATURES, GRADES))
    encoder_total = sum(p.numel() for p in encoder.parameters())
    head_total = sum(p.numel() for p in model[1].parameters())
    model.parameter_counts = {
        "encoder": encoder_total, "head": head_total, "total": encoder_total + head_total,
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "lora_branches": lora_branches,
        "note": "Every encoder weight is trained, as in the original notebook. The LoRA branches are a small "
                "part of the encoder, not the only trained weights.",
    }
    return model


def load_state(path, torch):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Expected a state dict: {path}")
    return state


def pretrained_state(state):
    """Vision encoder tensors of the official pretrained RetiZero checkpoint, keyed for build_model()."""
    if any(key.startswith(("classifier.", "retizero.", "img_encoder.", "0.", "1.")) for key in state):
        raise ValueError("This is a fine-tuned DR checkpoint. Training needs the official pretrained RetiZero.pth.")
    prefix = "vision_model.model."
    encoder = {"0." + key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not encoder:
        raise ValueError("No pretrained vision encoder tensors (vision_model.model.*) found.")
    return encoder


def load_pretrained(model, path, torch):
    """Load the pretrained encoder; the head keeps its fresh initialisation. Any key mismatch is an error."""
    state = pretrained_state(load_state(path, torch))
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("1.")]
    if missing or result.unexpected_keys:
        raise ValueError(f"The pretrained checkpoint does not fit the encoder. Missing {len(missing)}: "
                         f"{missing[:8]}; unexpected {len(result.unexpected_keys)}: {result.unexpected_keys[:8]}")
    return {"encoder_tensors_loaded": len(state)}


def fine_tuned_state(state, torch):
    """A fine-tuned checkpoint keyed for build_model(), from the app's format or from a run of this code."""
    if any(key.startswith("img_encoder.") for key in state):
        converted = {"0." + key[len("img_encoder."):]: value for key, value in state.items()
                     if key.startswith("img_encoder.")}
        converted.update({"1.weight": state["classifier.weight"], "1.bias": state["classifier.bias"]})
        mirror_prefix = "retizero.vision_model.model."
        mirror = {"0." + key[len(mirror_prefix):]: value for key, value in state.items()
                  if key.startswith(mirror_prefix)}
        identical = None
        if mirror:
            identical = (set(mirror) == set(converted) - {"1.weight", "1.bias"}
                         and all(torch.equal(mirror[key], converted[key]) for key in mirror))
        return converted, {"format": "app (img_encoder.*, classifier.*)", "encoder_copy_identical": identical}
    if all(key.startswith(("0.", "1.")) for key in state):
        return dict(state), {"format": "run (encoder then head)", "encoder_copy_identical": None}
    raise ValueError("Unrecognised fine-tuned checkpoint layout.")


def load_fine_tuned(model, path, torch):
    converted, info = fine_tuned_state(load_state(path, torch), torch)
    model.load_state_dict(converted, strict=True)
    return info


# ---------------------------------------------------------------- scoring

def predict(model, rows, batch_size, torch, class_weights=None, progress=None):
    """Score rows in order. Returns the predictions and, when class weights are given, the weighted loss."""
    functional = torch.nn.functional
    model.eval()
    predictions, loss_sum = [], 0.0
    with torch.inference_mode():
        for images, grades in loader(rows, batch_size, eval_transform(), torch):
            with torch.amp.autocast("cuda"):
                logits = model(images.to("cuda", non_blocking=True))
            logits = logits.float()
            if class_weights is not None:
                loss_sum += functional.cross_entropy(logits, grades.to("cuda"), weight=class_weights).item() * len(grades)
            for values in logits.softmax(dim=1).cpu().tolist():
                row = rows[len(predictions)]
                predictions.append({"image_id": row["id_code"], "image_path": row["image_path"],
                                    "y_true": int(row["label"]), "y_pred": max(range(GRADES), key=values.__getitem__),
                                    **{f"p{i}": value for i, value in enumerate(values)}})
            if progress:
                progress(len(predictions))
    return predictions, (loss_sum / len(rows) if class_weights is not None else None)


# ---------------------------------------------------------------- training

def configure_backend(torch):
    """Fixed numerical settings, recorded with every run. The original enabled TF32 on its A100."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"tf32": True, "cudnn_benchmark": False, "cudnn_deterministic": True}


def seed_everything(seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)


def class_weights_for(rows, torch, device):
    """Inverse class frequency, as in the original notebook: N / (5 * count)."""
    counts = Counter(int(row["label"]) for row in rows)
    if len(counts) != GRADES:
        raise ValueError("Every grade must appear in the training rows.")
    return torch.tensor([len(rows) / (GRADES * counts[grade]) for grade in range(GRADES)],
                        dtype=torch.float32, device=device)


def accumulated_backward(model, images, grades, class_weights, physical_batch, torch, scaler=None):
    """Gradient of the weighted mean cross-entropy over one group, computed in smaller pieces.

    The original trained with batches of 128 on a 40 GB GPU. Here a group of 128 is passed through the model
    `physical_batch` images at a time; each piece adds its weighted loss sum divided by the whole group's
    weight. The accumulated gradient equals the single-batch gradient (the model has no batch statistics).
    """
    functional = torch.nn.functional
    device = class_weights.device
    grades = grades.to(device)
    total_weight = class_weights[grades].sum()
    value = 0.0
    for start in range(0, len(grades), physical_batch):
        piece = images[start:start + physical_batch].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=scaler is not None):
            logits = model(piece)
        loss = functional.cross_entropy(logits.float(), grades[start:start + physical_batch],
                                        weight=class_weights, reduction="sum") / total_weight
        (scaler.scale(loss) if scaler is not None else loss).backward()
        value += loss.item()
    return value


def save_weights(torch, path, value):
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def best_name(epoch):
    return f"best_epoch_{epoch:02d}.pth"


def train(run, rows, config, model, torch, *, stop_after_epoch=None):
    """Train to the configured epoch limit or early stop. Returns False if stopped early on request.

    Resumes from run/checkpoint.pth, which is written after every epoch with the optimizer, scheduler,
    scaler and random-number states.
    """
    settings = config["training"]
    epochs, patience = settings["epochs"], settings["patience"]
    train_rows, val_rows = rows["train"], rows["val"]
    device = torch.device("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(settings["seed"])
    class_weights = class_weights_for(train_rows, torch, device)
    latest = run / "checkpoint.pth"
    history, best, best_epoch, stalled, best_sha = [], -1.0, 0, 0, None
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
        generator.set_state(saved["rng_order"])
        history, best, best_epoch, stalled, best_sha = (
            saved[key] for key in ("history", "best", "best_epoch", "stalled", "best_sha256"))
        if sha(run / best_name(best_epoch)) != best_sha:
            raise ValueError("The saved best-epoch weights changed.")
        del saved
        print(f"Resuming after epoch {len(history)}.", flush=True)
    print(f"Training on {len(train_rows)} images, validating on {len(val_rows)}. Seed {settings['seed']}, "
          f"groups of {settings['effective_batch']} in pieces of {settings['physical_batch']}.", flush=True)

    for epoch in range(len(history) + 1, epochs + 1):
        if stalled >= patience:
            print(f"Early stop: no improvement in {patience} epochs.", flush=True)
            break
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        order_sha = hashlib.sha256("\n".join(train_rows[i]["id_code"] for i in order).encode()).hexdigest()
        batches = loader(train_rows, settings["effective_batch"], train_transform(), torch, order=order)
        torch.cuda.reset_peak_memory_stats()
        started, running = time.perf_counter(), 0.0
        for step, (images, grades) in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            value = accumulated_backward(model, images, grades, class_weights, settings["physical_batch"],
                                         torch, scaler)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite training loss at epoch {epoch}, step {step}.")
            scaler.step(optimizer)
            scaler.update()
            running += value * len(grades)
            if step == 1 or step % 5 == 0 or step == len(batches):
                print(f"Epoch {epoch}/{epochs}, step {step}/{len(batches)}, loss {value:.5f}", flush=True)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        started = time.perf_counter()
        predictions, validation_loss = predict(model, val_rows, settings["eval_batch"], torch, class_weights)
        validation = calculate_metrics(predictions)
        accuracy = validation["accuracy"]
        history.append({
            "epoch": epoch, "learning_rate": learning_rate, "optimizer_steps": len(batches),
            "train_loss": running / len(train_rows), "validation_accuracy": accuracy,
            "validation_loss": validation_loss, "validation_qwk": validation["qwk"],
            "order_sha256": order_sha, "train_seconds": train_seconds,
            "train_images_per_second": len(train_rows) / train_seconds,
            "validation_seconds": time.perf_counter() - started,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        print(f"Epoch {epoch}: train loss {history[-1]['train_loss']:.5f}, validation accuracy {accuracy:.4%}, "
              f"validation loss {validation_loss:.5f}, {train_seconds:.0f} s.", flush=True)
        superseded = None
        if accuracy > best:  # ties keep the earlier epoch, as in the original
            superseded = run / best_name(best_epoch) if best_epoch else None
            best, best_epoch, stalled = accuracy, epoch, 0
            save_weights(torch, run / best_name(epoch), model.state_dict())
            best_sha = sha(run / best_name(epoch))
        else:
            stalled += 1
        save_weights(torch, latest, {
            "configuration": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "best": best, "best_epoch": best_epoch, "best_sha256": best_sha, "stalled": stalled,
            "rng_cpu": torch.get_rng_state(), "rng_gpu": torch.cuda.get_rng_state(),
            "rng_order": generator.get_state(),
        })
        if superseded is not None:
            superseded.unlink(missing_ok=True)
        if stop_after_epoch == epoch and epoch < epochs:
            print(f"Stopped after epoch {epoch} as requested. Run train again to resume.", flush=True)
            return False
    write_json(run / "train_result.json", {
        "configuration": config, "epochs_completed": len(history), "history": history,
        "best_epoch": best_epoch, "best_validation_accuracy": best,
        "checkpoint": best_name(best_epoch), "checkpoint_sha256": best_sha,
        "selection": "Highest validation accuracy; ties keep the earlier epoch.",
    })
    latest.unlink(missing_ok=True)  # the resume checkpoint is about 3.6 GB and no longer needed
    print(f"Selected epoch {best_epoch} with validation accuracy {best:.4%}.", flush=True)
    return True


def verified_best(run, config):
    """Path and hash of a finished run's selected checkpoint, after checking it against the saved result."""
    record = read_json(run / "train_result.json")
    if record["configuration"] != config:
        raise ValueError("The saved training settings differ from the current inputs.")
    path = run / best_name(record["best_epoch"])
    if record["checkpoint"] != path.name or sha(path) != record["checkpoint_sha256"]:
        raise ValueError("The saved best checkpoint path or hash changed.")
    return path, record["checkpoint_sha256"]
