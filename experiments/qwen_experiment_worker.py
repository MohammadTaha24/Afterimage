"""Four-condition Qwen experiment stages. Nothing runs on import.

NotebookOrchestrator supplies the notebook's shared, standard-library launch
plumbing. CLI stage workers never spawn children. GPU/model imports occur only
in explicitly launched stages. Importing this module starts no work and writes
no files. Existing reproduction code and input files are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

VERSION = 1
QUEUE = [("r16_clahe_on", 16, True), ("r8_clahe_on", 8, True),
         ("r32_clahe_on", 32, True), ("r16_clahe_off", 16, False)]
BASE_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
BASE_REVISION = "b5b904c3fcdc7541adf2a2bb219b0ed95288c794"
COUNTS = {"train": [1444, 296, 799, 154, 236], "val": [180, 37, 100, 20, 29],
          "test": [181, 37, 100, 19, 30], "messidor_external": [468, 207, 290, 71, 21]}
EXPECTED_TRAINING = {
    "epochs": 50, "batch_size": 4, "grad_accum": 2, "lr": 0.0005, "warmup_steps": 20,
    "max_length": 2048, "weight_decay": 0.001, "optim": "adamw_8bit", "scheduler": "linear",
    "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none",
    "gradient_checkpointing": True, "img_size": 512, "load_in_4bit": True,
    "eval_strategy": "epoch", "save_strategy": "epoch", "save_total_limit": 2,
    "early_stopping_patience": 2, "early_stopping_threshold": 0.0,
    "metric_for_best_model": "eval_loss", "logging_steps": 5, "trainer_seed": 42,
    "eval_batch_size": 8, "lora_seed_mode": "original_unsloth_default", "expected_lora_default_seed": 3407,
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class SharedFailure(RuntimeError):
    """Invalidates the shared comparison; the notebook must stop the queue."""


class ConditionFailure(RuntimeError):
    """Preserve this condition and continue other eligible conditions."""


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def emit(message):
    print(f"[{utc()}] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists() and read_json(path) != value:
        raise SharedFailure(f"Existing configuration differs: {path}; use a new batch directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_csv(path, allow_empty=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise SharedFailure(f"Missing/duplicate headers: {path}")
        rows = list(reader)
    if (not rows and not allow_empty) or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise SharedFailure(f"Empty or malformed CSV: {path}")
    return rows


def write_csv(path, rows, fields):
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8-sig") != content:
        raise SharedFailure(f"Refusing to replace a different frozen export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_output(root, batch):
    root, batch = Path(root).resolve(), Path(batch).resolve()
    allowed = root / "experiments" / "artifacts"
    if batch == root or (root in batch.parents and allowed not in batch.parents):
        raise SharedFailure(f"Outputs inside the reference project must be below {allowed}")
    return batch


def offline(cache):
    cache = Path(cache)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ.update(TOKENIZERS_PARALLELISM="false", WANDB_DISABLED="true", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    for key, directory in {"HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub", "TORCH_HOME": "torch",
                           "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                           "UNSLOTH_COMPILE_LOCATION": "unsloth"}.items():
        target = cache / directory
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)


def resources(batch, minimum_disk_gib, gpu=False, expected_gpu=None):
    free = shutil.disk_usage(batch).free / 2**30
    if free < minimum_disk_gib:
        raise SharedFailure(f"Only {free:.2f} GiB disk free at {batch}; require {minimum_disk_gib} GiB")
    result = {"disk_free_gib": free}
    if gpu:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise SharedFailure("CUDA and BF16 support are required; no CPU or precision fallback")
        if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise SharedFailure("Select exactly one visible GPU; distributed training would change effective batch size")
        available, total = torch.cuda.mem_get_info(0)
        if available < 10 * 2**30:
            raise SharedFailure(f"Only {available / 2**30:.2f} GiB GPU memory free; require 10 GiB before training")
        properties = torch.cuda.get_device_properties(0)
        identity = {"name": properties.name, "total_memory": properties.total_memory,
                    "capability": [properties.major, properties.minor], "uuid": str(getattr(properties, "uuid", "unavailable")),
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
        if expected_gpu is not None and identity != expected_gpu:
            raise SharedFailure(f"Selected GPU changed since preflight: expected={expected_gpu}, found={identity}")
        result.update(gpu=properties.name, gpu_identity=identity, gpu_free_gib=available / 2**30, gpu_total_gib=total / 2**30)
    return result


def source_files(root):
    root = Path(root)
    return {"helper": root / "finetuning/Qwen3/paper_reproduction.py",
            "trainer": root / "finetuning/Qwen3/train_dr_qwen3vl_runsave.py",
            "phase1": root / "audit/phase1_audit.py", "worker": Path(__file__).resolve(),
            "preprocessing": root / "Qwen3/preprocess_qwen.py"}


def packages(reviewed):
    actual = {key: importlib.metadata.version(key) for key in reviewed}
    if actual != reviewed:
        raise SharedFailure(f"Package versions differ from reviewed configuration: expected={reviewed}; found={actual}")
    return actual


def base_record(path, revision):
    path = Path(path).resolve()
    if path.name != revision or (path / "adapter_config.json").exists():
        raise SharedFailure(f"Base must be the exact cached revision snapshot, not an adapter: {path}")
    model_config = read_json(path / "config.json")
    if model_config.get("model_type") != "qwen3_vl" or "Qwen3VLForConditionalGeneration" not in model_config.get("architectures", []):
        raise SharedFailure("Cached base configuration is not the reviewed Qwen3-VL architecture")
    quantization = model_config.get("quantization_config", {})
    for key, value in {"_load_in_4bit": True, "_load_in_8bit": False, "bnb_4bit_quant_type": "nf4",
                       "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16"}.items():
        if quantization.get(key) != value:
            raise SharedFailure(f"Cached base quantization differs on {key}: {quantization.get(key)!r}")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    processors = [name for name in ("processor_config.json", "preprocessor_config.json") if (path / name).is_file()]
    if not processors:
        raise SharedFailure(f"Missing cached processor config: {path}")
    required.update(processors)
    required.update(p.name for p in path.iterdir() if p.is_file() and p.name in {
        "generation_config.json", "chat_template.jinja", "chat_template.json", "merges.txt",
        "vocab.json", "tokenizer.model", "special_tokens_map.json", "added_tokens.json"})
    indices = [name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json") if (path / name).is_file()]
    if indices:
        index = read_json(path / indices[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SharedFailure("Cached weight index has no weight_map")
        required.add(indices[0])
        for name in weight_map.values():
            if Path(name).name != name:
                raise SharedFailure(f"Invalid cached shard name: {name}")
            required.add(name)
    else:
        weights = [name for name in ("model.safetensors", "pytorch_model.bin") if (path / name).is_file()]
        if not weights:
            raise SharedFailure(f"No cached base weights: {path}")
        required.add(weights[0])
    result = {}
    for name in sorted(required):
        file = path / name
        if not file.is_file() or not file.stat().st_size:
            raise SharedFailure(f"Missing/empty cached asset: {file}")
        if file.suffix == ".json" and not isinstance(read_json(file), dict):
            raise SharedFailure(f"Cached JSON asset is not an object: {file}")
        result[name] = {"size": file.stat().st_size, "sha256": sha(file)}
        emit(f"Fingerprinted cached asset {name}")
    return {"path": str(path), "revision": revision, "quantization_configuration": quantization, "files": result}


def frozen_cohorts(manifest, reviewed, phase1):
    if manifest.get("counts") != {key: sum(value) for key, value in COUNTS.items()}:
        raise SharedFailure("Manifest cohort sizes differ from 2929/366/367/1057")
    if manifest.get("csv_sha256") != reviewed["dataset_csv_sha256"]:
        raise SharedFailure("Reviewed configuration and manifest CSV fingerprints disagree")
    if manifest.get("split_policy") != "recovered_original_aptos_split; all_messidor_external_only":
        raise SharedFailure("Recovered split policy is not the reviewed APTOS-only/external-Messidor policy")
    cohorts, ids, paths, patients = {}, {}, set(), {}
    for partition, expected in COUNTS.items():
        file = Path(manifest["csv_paths"][partition])
        if sha(file) != manifest["csv_sha256"][partition]:
            raise SharedFailure(f"Frozen CSV changed: {file}")
        rows = read_csv(file)
        source = "messidor" if partition == "messidor_external" else "aptos"
        seen, counts = set(), Counter()
        for row in rows:
            ident, label = str(row.get("id_code", "")).strip(), phase1.grade(row.get("label"))
            if not ident or ident in seen or label is None:
                raise SharedFailure(f"Duplicate/missing ID or invalid label in {file}: {ident}")
            seen.add(ident)
            counts[label] += 1
            if row.get("source_dataset") != source or row.get("split") != (partition if partition in {"train", "val"} else "test"):
                raise SharedFailure(f"Wrong dataset/split in {file}: {ident}")
            name = row.get("filename", "")
            if not name or Path(name).name != name or Path(row["image_path"]).name != name:
                raise SharedFailure(f"Image/filename binding is ambiguous: {ident}")
            key = (source, ident)
            if key in ids or name.casefold() in paths:
                raise SharedFailure(f"Image ID/path overlaps frozen partitions: {key}")
            ids[key], patients_key = partition, (source, row.get("patient_id", ""))
            if patients_key[1] and patients_key in patients and patients[patients_key] != partition:
                raise SharedFailure(f"Patient overlaps splits: {patients_key}")
            patients[patients_key] = partition
            paths.add(name.casefold())
            row["label"] = label
        if [counts[i] for i in range(5)] != expected:
            raise SharedFailure(f"Class counts differ in {file}: {counts}")
        cohorts[partition] = rows
    original_path = Path(manifest["original_split_csv"])
    if sha(original_path) != manifest["original_split_sha256"]:
        raise SharedFailure(f"Recovered original split hash differs: {original_path}")
    original = {}
    for row in read_csv(original_path):
        key = phase1.image_id(row.get("image_file"), "aptos")
        if not key or key in original or phase1.grade(row.get("label")) is None:
            raise SharedFailure("Recovered original split has duplicate/invalid records")
        original[key] = (row["split"], phase1.grade(row["label"]))
    current = {r["id_code"]: (p, r["label"]) for p in ("train", "val", "test") for r in cohorts[p]}
    if current != original:
        raise SharedFailure("APTOS IDs/labels/membership differ from the recovered original split")
    mapping_path = Path(manifest["image_mapping_csv"])
    if sha(mapping_path) != manifest["mapping_sha256"]:
        raise SharedFailure("Image mapping hash changed")
    mapping = {}
    for row in read_csv(mapping_path):
        name = Path(row["image_path"]).name
        if name in mapping:
            raise SharedFailure(f"Duplicate mapped image: {name}")
        mapping[name] = row
    for rows in cohorts.values():
        for row in rows:
            mapped = mapping.get(row["filename"], {})
            if (phase1.grade(mapped.get("label")) != row["label"] or mapped.get("source_dataset") != row["source_dataset"]
                    or mapped.get("patient_id") != row.get("patient_id")):
                raise SharedFailure(f"Mapped labels/identity disagree: {row['filename']}")
    # Mapping split assignments are deliberately ignored: they differ from the
    # recovered APTOS split and must never be used to re-split this experiment.
    return cohorts


def image_branch(folder, cohorts):
    from PIL import Image
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Missing image folder: {folder}")
    expected = {r["filename"] for rows in cohorts.values() for r in rows}
    found, resolved_paths = {}, set()
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in found:
                raise ValueError(f"Duplicate branch filename: {path.name}")
            resolved = path.resolve()
            if path.is_symlink() or folder not in resolved.parents or os.path.normcase(str(resolved)) in resolved_paths:
                raise ValueError(f"Image escapes branch or aliases another image: {path}")
            resolved_paths.add(os.path.normcase(str(resolved)))
            found[path.name] = resolved
    if set(found) != expected:
        raise ValueError(f"Branch ID coverage mismatch: {folder}; missing={sorted(expected - set(found))[:10]}; extra={sorted(set(found) - expected)[:10]}")
    facts = []
    for index, name in enumerate(sorted(found), 1):
        path = found[name]
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"Expected lossless RGB PNG 512x512: {path}; found {image.format}/{image.mode}/{image.size}")
            image.verify()
        with Image.open(path) as image:
            image.load()  # Decode fully; keep only this one image in memory.
        facts.append({"filename": name, "path": str(path), "sha256": sha(path), "size": path.stat().st_size})
        if index % 500 == 0:
            emit(f"Verified {index}/{len(found)} images in {folder.name}")
    return {"root": str(folder), "files": facts, "identity": fingerprint(facts)}


def clahe_once(bgr):
    import cv2
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((enhanced, a, b)), cv2.COLOR_LAB2BGR)


def standard_from_original(path):
    """Same explicit crop/pad/resize recipe, with CLAHE deliberately absent."""
    import cv2
    import numpy as np
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode original image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.sum():
        coordinates = np.argwhere(mask > 0)
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0) + 1
        rgb = rgb[y0:y1, x0:x1]
    height, width = rgb.shape[:2]
    difference = abs(height - width)
    padding = (difference // 2, difference - difference // 2, 0, 0) if height < width else (0, 0, difference // 2, difference - difference // 2)
    rgb = cv2.copyMakeBorder(rgb, *padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    interpolation = cv2.INTER_AREA if 512 < rgb.shape[0] else cv2.INTER_CUBIC
    return cv2.cvtColor(cv2.resize(rgb, (512, 512), interpolation=interpolation), cv2.COLOR_RGB2BGR)


def pairing(request, cohorts, on, off, batch):
    import cv2
    import numpy as np
    raw_index = request.get("original_image_index")
    originals, original_paths = {}, set()
    if raw_index:
        index_path = Path(raw_index).resolve()
        for row in read_csv(index_path):
            key = (row.get("source_dataset"), row.get("image_id", row.get("id_code")))
            path = Path(row.get("original_path", ""))
            path = (index_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if key in originals or str(row.get("is_original", "")).lower() not in {"true", "1", "yes"} or not path.is_file():
                raise ValueError(f"Invalid/duplicate/unattested original image record: {key}, {path}")
            if Path(on["root"]) in path.parents or Path(off["root"]) in path.parents:
                raise ValueError("Original paths must be independent of both processed branches")
            path_key = os.path.normcase(str(path))
            if path_key in original_paths:
                raise ValueError(f"Original path reused for multiple IDs: {path}")
            original_paths.add(path_key)
            originals[key] = path
        expected = {(r["source_dataset"], r["id_code"]) for rows in cohorts.values() for r in rows}
        if set(originals) != expected:
            raise ValueError("Original image index does not exactly cover the frozen 4,719 IDs")
    on_paths = {item["filename"]: Path(item["path"]) for item in on["files"]}
    off_paths = {item["filename"]: Path(item["path"]) for item in off["files"]}
    proof, mismatches = [], []
    for partition, rows in cohorts.items():
        for row in rows:
            name = row["filename"]
            on_pixels = cv2.imread(str(on_paths[name]), cv2.IMREAD_COLOR)
            off_pixels = cv2.imread(str(off_paths[name]), cv2.IMREAD_COLOR)
            relation = np.array_equal(clahe_once(off_pixels), on_pixels)
            key = (row["source_dataset"], row["id_code"])
            original_match = bool(originals) and np.array_equal(standard_from_original(originals[key]), off_pixels)
            if not relation or (originals and not original_match):
                mismatches.append(f"{key}: once={relation}, original_to_off={original_match}")
            proof.append({"partition": partition, "source_dataset": key[0], "image_id": key[1],
                          "filename": name, "single_clahe_pixel_match": relation,
                          "original_to_off_pixel_match": original_match,
                          "original_path": str(originals[key]) if originals else "",
                          "original_sha256": sha(originals[key]) if originals else ""})
        emit(f"Pairing checked for {partition}; {len(proof)} images so far")
    # Save the proof even for a blocked ablation. Never write processed images.
    proof_path = batch / "pairing_checks.csv"
    temporary = proof_path.with_suffix(".partial.csv")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(proof[0]))
        writer.writeheader()
        writer.writerows(proof)
    temporary.replace(proof_path)
    if mismatches:
        raise ValueError(f"Pairing/preprocessing mismatch on {len(mismatches)} images: {mismatches[:10]}; see {proof_path}")
    if not originals:
        raise ValueError("Single-CLAHE pixel relation checked, but originals are unverified. Supply ORIGINAL_IMAGE_INDEX to establish that off has no prior CLAHE; see pairing_checks.csv")
    return {"verified": True, "images": len(proof), "proof_sha256": sha(proof_path),
            "original_index": str(index_path), "original_index_sha256": sha(index_path),
            "recipe": "BGR read -> RGB Otsu crop/black square pad/512 resize -> BGR; off saved unchanged; on BGR->LAB, CLAHE L once (2.0,8x8), LAB->BGR",
            "augmentation": "none in either branch"}


def export_branch(batch, tag, cohorts, branch):
    paths = {item["filename"]: item["path"] for item in branch["files"]}
    outputs = {}
    fields = ["original_image_id", "id_code", "filename", "image_path", "label", "source_dataset", "split", "patient_id"]
    for partition, rows in cohorts.items():
        target = batch / "data" / tag / f"{partition}.csv"
        export = [{key: (paths[r["filename"]] if key == "image_path" else r["id_code"] if key == "original_image_id" else r.get(key, "")) for key in fields} for r in rows]
        write_csv(target, export, fields)
        outputs[partition] = {"path": str(target), "sha256": sha(target)}
    return outputs


def preflight(request, batch):
    result = {"started_at": utc(), "status": "running", "shared_blockers": [], "ablation_blockers": [], "checks": []}
    write_json(batch / "preflight.json", result)
    try:
        root = Path(request["project_root"]).resolve()
        safe_output(root, batch)
        include_off = request.get("include_clahe_off", True)
        if type(include_off) is not bool:
            raise SharedFailure("include_clahe_off must be true or false")
        result["excluded_runs"] = [] if include_off else ["r16_clahe_off"]
        if float(request["minimum_disk_gib"]) < 10:
            raise SharedFailure("minimum_disk_gib must be at least 10")
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") != "0":
            raise SharedFailure("UNSLOTH_ENABLE_FULL_FINETUNING must be disabled for fresh LoRA experiments")
        reviewed = read_json(request["reviewed_configuration"])
        manifest = read_json(request["dataset_manifest"])
        if reviewed.get("training") != EXPECTED_TRAINING or reviewed.get("model_name") != BASE_NAME or reviewed.get("base_revision") != BASE_REVISION:
            raise SharedFailure("Reviewed settings differ from the inspected paper configuration; review explicitly instead of changing this queue silently")
        sources = source_files(root)
        hashes = {key: sha(path) for key, path in sources.items()}
        if hashes["helper"] != reviewed["helper_sha256"] or hashes["trainer"] != reviewed["trainer_module_sha256"]:
            raise SharedFailure("Reproduction helper/trainer source differs from reviewed configuration")
        reference_adapter = read_json(request["target_adapter_configuration"])
        for key, value in {"r": 16, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "peft_type": "LORA", "base_model_name_or_path": BASE_NAME}.items():
            if reference_adapter.get(key) != value:
                raise SharedFailure(f"Reviewed adapter configuration disagrees on {key}")
        if reference_adapter.get("use_rslora") or reference_adapter.get("use_dora") or reference_adapter.get("rank_pattern") or reference_adapter.get("alpha_pattern"):
            raise SharedFailure("Reviewed adapter uses nonstandard scaling/rank patterns")
        target_modules = reference_adapter.get("target_modules")
        if not target_modules:
            raise SharedFailure("Reviewed adapter has no target_modules record")
        phase1 = local_module("qwen_phase1", sources["phase1"])
        cohorts = frozen_cohorts(manifest, reviewed, phase1)
        result["checks"].append("Frozen counts, class counts, unique IDs, recovered labels/membership, mapping and image/patient split overlap passed; Messidor is external only")
        if sys.version_info[:2] != (3, 11) or Path(sys.prefix).name.casefold() != "afterimage_env":
            raise SharedFailure(f"Use Python 3.11 (afterimage_env); found {sys.executable}, {sys.version.split()[0]}")
        package_versions = packages(reviewed["package_versions"])
        # Binary/import and API checks occur here, tomorrow, before any model is
        # loaded. The preflight process exits before the first training process.
        offline(batch / "preflight_cache")
        import pandas
        import cv2
        import numpy
        from PIL import Image
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTConfig
        helper = local_module("qwen_reproduction", sources["helper"])
        if helper.resolve_lora_default_seed(FastVisionModel) != 3407:
            raise SharedFailure("Installed Unsloth LoRA default seed no longer matches 3407")
        probe = helper.make_sft_config({"training": reviewed["training"]}, batch / "argument_probe", SFTConfig)
        write_json(batch / "preflight_training_arguments.json", probe.to_dict())
        result["resources"] = resources(batch, float(request["minimum_disk_gib"]), gpu=True)
        readers = {"pillow": importlib.metadata.version("pillow")}
        for name in ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"):
            try:
                readers[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass  # Conda may provide cv2 without a wheel distribution record.
        from unsloth.models.vision import FastBaseModel
        dependency_sources = {str(Path(inspect.getsourcefile(item)).resolve()): sha(inspect.getsourcefile(item))
                              for item in (FastBaseModel, UnslothVisionDataCollator)}
        result["checks"].append("No image augmentation in the new formatting/preprocessing paths; pinned collator source performs deterministic resizing/formatting, and weighted sampling applies to train only")
        base_path = request.get("base_snapshot") or reviewed["base_snapshot"]
        base = base_record(base_path, BASE_REVISION)
        on = image_branch(request["clahe_on_images"], cohorts)
        result["checks"].append("Shared CLAHE branch: exact 4,719 filenames, valid RGB 512x512 PNGs, byte fingerprints; no generated augmentation variants accepted")
        off, pair = None, None
        if include_off:
            try:
                off = image_branch(request["clahe_off_images"], cohorts)
                pair = pairing(request, cohorts, on, off, batch)
            except Exception as error:
                result["ablation_blockers"].append(f"{type(error).__name__}: {error}")
                emit("CLAHE ablation BLOCKED; rank comparison may continue: " + str(error))
        else:
            result["checks"].append("CLAHE-off explicitly excluded; off-branch pairing and original-image-index checks skipped")
            emit("CLAHE-off excluded by configuration; preparing the three CLAHE-on rank experiments only")
        on_csvs = export_branch(batch, "clahe_on", cohorts, on)
        off_csvs = export_branch(batch, "clahe_off", cohorts, off) if pair else None
        shared = {"version": VERSION, "project_root": str(root), "reviewed_configuration": str(Path(request["reviewed_configuration"]).resolve()),
                  "reviewed_configuration_sha256": sha(request["reviewed_configuration"]),
                  "dataset_manifest": str(Path(request["dataset_manifest"]).resolve()), "dataset_manifest_sha256": sha(request["dataset_manifest"]),
                  "frozen_csvs": {key: {"path": manifest["csv_paths"][key], "sha256": manifest["csv_sha256"][key]} for key in COUNTS},
                  "training": reviewed["training"], "base": base, "model_name": BASE_NAME,
                  "package_versions": package_versions, "image_reader_versions": {"opencv": cv2.__version__, "pillow": importlib.metadata.version("pillow")},
                  "reader_package_versions": readers,
                  "source_paths": {key: str(path) for key, path in sources.items()}, "source_sha256": hashes,
                  "target_modules": target_modules, "target_config_sha256": sha(request["target_adapter_configuration"]),
                  "gpu_identity": result["resources"]["gpu_identity"],
                  "target_config_path": str(Path(request["target_adapter_configuration"]).resolve()),
                  "dependency_source_sha256": dependency_sources,
                  "provenance_inputs": [{"path": manifest[key], "sha256": manifest[hash_key]} for key, hash_key in
                                        (("original_split_csv", "original_split_sha256"), ("image_mapping_csv", "mapping_sha256"))],
                  "clahe_on": on, "on_csvs": on_csvs, "python": str(Path(sys.executable).resolve()),
                  "augmentation": "none; deterministic formatting only; inverse-frequency sampler in train loader only",
                  "planning_hours": 12, "hard_deadline": False}
        shared["identity"] = fingerprint(shared)
        write_json(batch / "shared_configuration.json", shared, immutable=True)
        eligibility = []
        for run_id, rank, clahe in QUEUE:
            run_dir = batch / run_id
            run_dir.mkdir(exist_ok=True)
            selected = clahe or include_off
            eligible = selected and (clahe or bool(pair))
            config = {"version": VERSION, "shared_identity": shared["identity"], "run_id": run_id,
                      "rank": rank, "clahe": clahe, "selected": selected, "eligible": eligible,
                      "exclusion_reason": "" if selected else "Excluded by INCLUDE_CLAHE_OFF=False",
                      "training": {**reviewed["training"], "lora_r": rank}, "base_revision": BASE_REVISION,
                      "base_snapshot": base["path"], "checkpoint_selection": "minimum APTOS validation loss",
                      "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "use_rslora": False,
                      "effective_lora_scaling": 16 / rank, "lora_initialization_seed": 3407,
                      "target_modules": target_modules, "csvs": on_csvs if clahe else off_csvs,
                      "images": on if clahe else off, "pairing": None if clahe else pair,
                      "blockers": [] if eligible or not selected else result["ablation_blockers"]}
            config["identity"] = fingerprint(config)
            path = run_dir / "resolved_configuration.json"
            if path.exists() and read_json(path) != config:
                previous = read_json(path)
                if previous.get("eligible") or (run_dir / "checkpoints").exists() or (run_dir / "training_complete.json").exists():
                    raise SharedFailure(f"Run configuration changed: {path}; use a new batch directory")
            write_json(path, config)
            eligibility.append({"run_id": run_id, "selected": selected, "eligible": eligible, "rank": rank, "clahe": clahe,
                                "exclusion_reason": config["exclusion_reason"], "blockers": config["blockers"]})
        result.update(status="ready" if pair or not include_off else "ready_rank_comparison_ablation_blocked", shared_identity=shared["identity"], runs=eligibility)
        emit("Preflight ready; the notebook may launch eligible conditions in fixed order")
    except Exception as error:
        result.update(status="blocked_shared", shared_blockers=[f"{type(error).__name__}: {error}"])
        # Preserve any previously resolved configuration; a failed new preflight
        # is recorded separately, never used to bless an old completion marker.
        for run_id, rank, clahe in QUEUE:
            path = batch / run_id / "requested_configuration.json"
            write_json(path, {"run_id": run_id, "rank": rank, "clahe": clahe, "request": request,
                              "status": "blocked_shared", "blockers": result["shared_blockers"]})
            resolved = path.with_name("resolved_configuration.json")
            if not resolved.exists():
                write_json(resolved, {"run_id": run_id, "rank": rank, "clahe": clahe, "eligible": False,
                                      "resolution_status": "blocked_shared_partial", "training": reviewed.get("training") if isinstance(locals().get("reviewed"), dict) else None,
                                      "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "effective_lora_scaling": 16 / rank,
                                      "request": request, "blockers": result["shared_blockers"]})
        raise SharedFailure(result["shared_blockers"][0]) from error
    finally:
        result["ended_at"] = utc()
        write_json(batch / "preflight.json", result)


def _load_condition(batch, run_id):
    preflight_result = read_json(batch / "preflight.json")
    if preflight_result.get("status") not in {"ready", "ready_rank_comparison_ablation_blocked"}:
        raise SharedFailure("Current preflight has not approved this batch")
    shared = read_json(batch / "shared_configuration.json")
    if preflight_result.get("shared_identity") != shared.get("identity"):
        raise SharedFailure("Preflight approval belongs to a different shared configuration")
    config = read_json(batch / run_id / "resolved_configuration.json")
    if (run_id, config["rank"], config["clahe"]) not in QUEUE or not config["eligible"]:
        raise ConditionFailure(f"Condition is not eligible: {run_id}")
    for record in (shared, config):
        if fingerprint({k: v for k, v in record.items() if k != "identity"}) != record["identity"]:
            raise SharedFailure("Resolved configuration identity is invalid")
    if config["shared_identity"] != shared["identity"]:
        raise SharedFailure("Condition/shared configuration identity mismatch")
    if config["training"] != {**shared["training"], "lora_r": config["rank"]}:
        raise SharedFailure("Condition changed a fixed setting other than its designated LoRA rank")
    for key, path in shared["source_paths"].items():
        if sha(path) != shared["source_sha256"][key]:
            raise SharedFailure(f"Source changed after preflight: {path}")
    for path, expected in shared["dependency_source_sha256"].items():
        if sha(path) != expected:
            raise SharedFailure(f"Installed constructor/collator source changed: {path}")
    if sha(shared["target_config_path"]) != shared["target_config_sha256"]:
        raise SharedFailure("Reviewed target-module adapter configuration changed")
    if sha(shared["reviewed_configuration"]) != shared["reviewed_configuration_sha256"] or sha(shared["dataset_manifest"]) != shared["dataset_manifest_sha256"]:
        raise SharedFailure("Reviewed configuration or dataset manifest changed")
    packages(shared["package_versions"])
    packages(shared["reader_package_versions"])
    for record in list(shared["frozen_csvs"].values()) + list(config["csvs"].values()) + shared["provenance_inputs"]:
        if sha(record["path"]) != record["sha256"]:
            raise SharedFailure(f"Dataset CSV changed: {record['path']}")
    for image in config["images"]["files"]:
        if sha(image["path"]) != image["sha256"]:
            raise SharedFailure(f"Frozen image bytes changed: {image['path']}")
    if not config["clahe"]:
        pair = config["pairing"]
        if sha(pair["original_index"]) != pair["original_index_sha256"] or sha(batch / "pairing_checks.csv") != pair["proof_sha256"]:
            raise SharedFailure("CLAHE pairing/original index proof changed")
        for row in read_csv(batch / "pairing_checks.csv"):
            if sha(row["original_path"]) != row["original_sha256"]:
                raise SharedFailure(f"Original image changed after ablation verification: {row['original_path']}")
    # Check base file integrity in every stage, including resumed/completed ones.
    for name, record in shared["base"]["files"].items():
        if sha(Path(shared["base"]["path"]) / name) != record["sha256"]:
            raise SharedFailure(f"Cached base file changed: {name}")
    if str(Path(sys.executable).resolve()) != shared["python"]:
        raise SharedFailure("Python executable changed since preflight")
    resources(batch, 10)
    return shared, config


def load_condition(batch, run_id):
    try:
        emit(f"Rechecking source, configuration, frozen CSV/image and cached-base hashes: {run_id}")
        return _load_condition(batch, run_id)
    except (ConditionFailure, SharedFailure):
        raise
    except Exception as error:
        raise SharedFailure(f"Shared integrity/environment check failed: {type(error).__name__}: {error}") from error


def adapter_config_matches(actual, config):
    expected = {"r": config["rank"], "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "peft_type": "LORA",
                "base_model_name_or_path": BASE_NAME, "target_modules": config["target_modules"], "use_rslora": False}
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ConditionFailure(f"Adapter setting {key} must be {value!r}; found {actual.get(key)!r}")
    if actual.get("rank_pattern") or actual.get("alpha_pattern") or actual.get("use_dora"):
        raise ConditionFailure("Saved adapter has unexpected rank/alpha/DoRA scaling")


def adapter_record(path, config):
    path = Path(path)
    adapter_config_matches(read_json(path / "adapter_config.json"), config)
    weights = [path / name for name in ("adapter_model.safetensors", "adapter_model.bin") if (path / name).is_file() and (path / name).stat().st_size]
    if len(weights) != 1:
        raise ConditionFailure(f"Expected exactly one nonempty adapter weight file: {path}")
    return {"directory": str(path.resolve()), "configuration_sha256": sha(path / "adapter_config.json"),
            "weights": weights[0].name, "weights_sha256": sha(weights[0]), "rank": config["rank"],
            "effective_scaling": 16 / config["rank"]}


def training_complete(run_dir, config):
    path = run_dir / "training_complete.json"
    if not path.exists():
        return None
    marker = read_json(path)
    if marker.get("configuration_identity") != config["identity"]:
        raise ConditionFailure("Training marker belongs to a different condition/configuration")
    if adapter_record(run_dir / "best_lora", config) != marker["adapter"]:
        raise ConditionFailure("Saved best adapter no longer matches the completion record")
    if not marker.get("best_checkpoint") or not math.isfinite(marker["best_validation_loss"]):
        raise ConditionFailure("Training completion has no valid validation winner")
    for name, expected in marker["portable_processor_sha256"].items():
        if sha(run_dir / "best_lora" / name) != expected:
            raise ConditionFailure(f"Saved processor changed: {name}")
    return marker


def checkpoint_record(path, config):
    path = Path(path)
    required = {"trainer_state.json", "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"}
    if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
        raise ConditionFailure(f"Incomplete optimizer/scheduler/RNG/trainer checkpoint: {path}")
    adapter_record(path, config)
    state = read_json(path / "trainer_state.json")
    if int(path.name.split("-")[-1]) != state.get("global_step"):
        raise ConditionFailure(f"Checkpoint step does not match trainer state: {path}")
    callbacks = state.get("stateful_callbacks", {})
    if "EarlyStoppingCallback" not in callbacks:
        raise ConditionFailure(f"Early-stopping state missing from {path}")
    return {"configuration_identity": config["identity"], "global_step": state["global_step"],
            "files": {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*"))
                      if p.is_file() and p.name not in {"checkpoint_identity.json", "checkpoint_identity.json.partial"}}}


def resume_checkpoint(run_dir, config):
    directory = run_dir / "checkpoints"
    candidates = sorted((p for p in directory.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)),
                        key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    errors = []
    for path in candidates:
        try:
            saved = read_json(path / "checkpoint_identity.json")
            if checkpoint_record(path, config) != saved:
                raise ConditionFailure("Checkpoint hashes/condition identity differ")
            state = read_json(path / "trainer_state.json")
            winner = state.get("best_model_checkpoint")
            if winner:
                winner_path = Path(winner).resolve()
                if winner_path.parent != directory.resolve():
                    raise ConditionFailure("Checkpoint references a validation winner outside this run")
                adapter_record(winner_path, config)
                if checkpoint_record(winner_path, config) != read_json(winner_path / "checkpoint_identity.json"):
                    raise ConditionFailure("Validation winner checkpoint hashes are incompatible/incomplete")
            emit(f"Resuming compatible complete checkpoint: {path}; ignored candidates: {errors}")
            return path
        except (ValueError, OSError, KeyError, ConditionFailure) as error:
            errors.append(f"{path.name}: {error}")
    if candidates or (run_dir / "training_started.json").exists() or (run_dir / "best_lora").exists():
        raise ConditionFailure(f"No compatible complete checkpoint available; preserve this run and use a new batch. Details: {errors}")
    return None


def formatted_dataset(rows, helper):
    from datasets import Dataset
    return Dataset.from_list([{
        "messages": [{"role": "user", "content": [{"type": "text", "text": helper.INSTRUCTION},
                                                      {"type": "image", "image": r["image_path"]}]},
                     {"role": "assistant", "content": [{"type": "text", "text": helper.CANDIDATES[int(r["label"])]}]}],
        "label": int(r["label"]), "image_path": r["image_path"], "img_w": 512, "img_h": 512,
    } for r in rows])


def train(batch, run_id):
    shared, config = load_condition(batch, run_id)
    run_dir = batch / run_id
    if training_complete(run_dir, config):
        emit(f"Verified completed training; skipping {run_id}")
        return
    resume = resume_checkpoint(run_dir, config)
    offline(run_dir / "cache")
    import pandas
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from transformers import EarlyStoppingCallback, TrainerCallback
    from trl import SFTTrainer, SFTConfig
    helper = local_module("qwen_reproduction", shared["source_paths"]["helper"])
    trainer_module = local_module("qwen_training_definitions", shared["source_paths"]["trainer"])
    resources(batch, 10, gpu=True, expected_gpu=shared["gpu_identity"])
    p = config["training"]
    if helper.resolve_lora_default_seed(FastVisionModel) != p["expected_lora_default_seed"]:
        raise SharedFailure("LoRA initialization seed changed")
    # Only these two CSVs are used to construct training/selection datasets.
    train_ds = formatted_dataset(read_csv(config["csvs"]["train"]["path"]), helper)
    val_ds = formatted_dataset(read_csv(config["csvs"]["val"]["path"]), helper)
    args = helper.make_sft_config({"training": p}, run_dir / "checkpoints", SFTConfig)
    model, processor = FastVisionModel.from_pretrained(model_name=shared["base"]["path"], load_in_4bit=p["load_in_4bit"], local_files_only=True)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise SharedFailure("Base was not loaded in the reviewed 4-bit quantization mode")
    if getattr(model, "peft_config", None):
        raise SharedFailure("Base loader returned an existing adapter; refusing cross-run continuation")
    constructor = {"r": config["rank"], "lora_alpha": p["lora_alpha"], "lora_dropout": p["lora_dropout"],
                   "bias": p["bias"], "target_modules": config["target_modules"], "use_rslora": False,
                   "random_state": config["lora_initialization_seed"], "use_gradient_checkpointing": True,
                   "temporary_location": str(run_dir / "cache/saved_buffers")}
    write_json(run_dir / "adapter_constructor_arguments.json", constructor, immutable=True)
    # Explicit rank reaches the actual constructor. Never delegate to the old
    # train_one(), whose constructor hard-codes r=16.
    model = FastVisionModel.get_peft_model(model, **constructor)
    for peft_config in model.peft_config.values():
        peft_config.base_model_name_or_path = BASE_NAME
        adapter_config_matches(peft_config.to_dict(), config)
    scaling = {name: {key: float(value) for key, value in module.scaling.items()}
               for name, module in model.named_modules() if hasattr(module, "scaling") and isinstance(module.scaling, dict)}
    if not scaling or any(abs(value - config["effective_lora_scaling"]) > 1e-12 for values in scaling.values() for value in values.values()):
        raise ConditionFailure("In-memory LoRA scaling does not equal alpha/r")
    write_json(run_dir / "effective_adapter_scaling.json", scaling)
    FastVisionModel.for_training(model)

    class CompleteCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_json(path / "checkpoint_identity.json", checkpoint_record(path, config))
            emit(f"Durably recorded checkpoint {path.name}")

    processor_key = "tokenizer" if "tokenizer" in inspect.signature(SFTTrainer.__init__).parameters else "processing_class"
    trainer = trainer_module.BalancedSFTTrainer(
        model=model, **{processor_key: processor}, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=UnslothVisionDataCollator(model, processor, resize=(512, 512), max_seq_length=2048),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=p["early_stopping_patience"],
                                         early_stopping_threshold=p["early_stopping_threshold"]), CompleteCheckpoint()],
    )
    # Some Unsloth Trainer wrappers replace the default-valued eval batch (8)
    # with the training batch (4). Restore the explicit reviewed value before
    # any validation dataloader is built; record the correction for review.
    if trainer.args.per_device_eval_batch_size == p["batch_size"] and p["eval_batch_size"] != p["batch_size"]:
        write_json(run_dir / "library_argument_corrections.json", {
            "per_device_eval_batch_size": {"library_value": trainer.args.per_device_eval_batch_size,
                                           "restored_reviewed_value": p["eval_batch_size"]}})
        trainer.args.per_device_eval_batch_size = p["eval_batch_size"]
        emit("Restored the explicitly reviewed validation batch size to 8 before validation")
    # Refuse silent library adjustments to the reviewed experiment.
    argument_checks = {"per_device_train_batch_size": 4, "per_device_eval_batch_size": 8,
                       "gradient_accumulation_steps": 2, "learning_rate": 0.0005, "num_train_epochs": 50,
                       "seed": 42, "save_total_limit": 2, "load_best_model_at_end": True,
                       "metric_for_best_model": "eval_loss", "greater_is_better": False,
                       "bf16": True, "fp16": False, "auto_find_batch_size": False,
                       "warmup_steps": 20, "weight_decay": 0.001, "optim": "adamw_8bit",
                       "lr_scheduler_type": "linear", "max_length": 2048, "logging_steps": 5,
                       "eval_strategy": "epoch", "save_strategy": "epoch", "gradient_checkpointing": True,
                       "restore_callback_states_from_checkpoint": True}
    for key, value in argument_checks.items():
        if getattr(trainer.args, key) != value:
            raise SharedFailure(f"Library changed reviewed Trainer argument {key}: expected {value}, found {getattr(trainer.args, key)}")
    write_json(run_dir / "resolved_training_arguments.json", trainer.args.to_dict())
    started = time.monotonic()
    write_json(run_dir / "training_started.json", {"configuration_identity": config["identity"], "time": utc(), "resume_checkpoint": str(resume) if resume else None})
    statistics = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    winner = trainer.state.best_model_checkpoint
    if not winner or not math.isfinite(float(trainer.state.best_metric)):
        raise ConditionFailure("No finite APTOS validation-loss winner; training is not complete")
    if Path(winner).resolve().parent != (run_dir / "checkpoints").resolve():
        raise ConditionFailure("Trainer selected a checkpoint outside this condition")
    destination = run_dir / "best_lora"
    trainer.save_model(str(destination))
    processor.save_pretrained(str(destination))
    adapter = adapter_record(destination, config)
    portable = {p.name: sha(p) for p in destination.iterdir() if p.is_file() and p.name not in {"adapter_config.json", adapter["weights"]}}
    if not any("processor" in name for name in portable) or not any("tokenizer" in name for name in portable):
        raise ConditionFailure("Best adapter lacks portable processor/tokenizer files")
    info = {"configuration_identity": config["identity"], "ended_at": utc(), "adapter": adapter,
            "best_checkpoint": str(winner), "best_validation_loss": float(trainer.state.best_metric),
            "best_global_step": getattr(trainer.state, "best_global_step", None), "epochs_completed": trainer.state.epoch,
            "train_metrics": statistics.metrics, "trainer_log_history": trainer.state.log_history,
            "elapsed_this_attempt_seconds": time.monotonic() - started, "portable_processor_sha256": portable,
            "selection": "minimum APTOS validation loss; test and external metrics excluded"}
    write_json(run_dir / "run_info.json", info)
    write_json(run_dir / "training_complete.json", info)
    emit(f"Training complete and adapter rank/targets/scaling verified: {destination}")


def prediction_metrics(rows, expected, phase1):
    if len(rows) != len(expected):
        raise ConditionFailure("Prediction export has incomplete or excess test coverage")
    indexed = {}
    for row in rows:
        key = row["image_id"]
        if key in indexed:
            raise ConditionFailure(f"Duplicate prediction ID: {key}")
        indexed[key] = row
    truth, predictions, vectors = [], [], []
    for target in expected:
        ident = target["id_code"]
        if ident not in indexed:
            raise ConditionFailure(f"Missing test prediction ID: {ident}")
        row = indexed[ident]
        label, prediction = phase1.grade(row["y_true"]), phase1.grade(row["y_pred"])
        scores = [float(row[f"p{i}"]) for i in range(5)]
        if (row.get("original_image_id") != ident or label != int(target["label"]) or prediction is None or row["image_path"] != target["image_path"]
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in scores) or abs(sum(scores) - 1) > 1e-6
                or max(range(5), key=lambda i: scores[i]) != prediction):
            raise ConditionFailure(f"Invalid prediction/label/scores for {ident}")
        truth.append(label)
        predictions.append(prediction)
        vectors.append(scores)
    accuracy, qwk = phase1.Audit.metrics(truth, predictions)
    aucs = [phase1.Audit.binary_auc([t == i for t in truth], [v[i] for v in vectors]) for i in range(5)]
    if any(v is None for v in aucs):
        raise ConditionFailure("Five-class macro OVR AUC is undefined: absent positive/negative class")
    return {"samples": len(rows), "accuracy": accuracy, "qwk": qwk, "macro_ovr_auc": sum(aucs) / 5, "per_grade_auc": aucs}


def evaluation_complete(run_dir, shared, config, adapter, phase1):
    marker_path = run_dir / "evaluation_complete.json"
    if not marker_path.exists():
        return None
    marker = read_json(marker_path)
    if marker["configuration_identity"] != config["identity"] or marker["adapter"] != adapter:
        raise ConditionFailure("Completed evaluation identity changed")
    for subset, partition in (("aptos_test", "test"), ("messidor_external", "messidor_external")):
        file = run_dir / "evaluation" / f"{subset}.predictions.csv"
        if sha(file) != marker["prediction_sha256"][subset]:
            raise ConditionFailure(f"Completed prediction export changed: {file}")
        metrics = prediction_metrics(read_csv(file), read_csv(config["csvs"][partition]["path"]), phase1)
        if metrics != marker["metrics"][subset]:
            raise ConditionFailure("Completed metric record disagrees with recomputation")
    return marker


def evaluate(batch, run_id):
    shared, config = load_condition(batch, run_id)
    run_dir = batch / run_id
    trained = training_complete(run_dir, config)
    if not trained:
        raise ConditionFailure("Evaluation requires verified completed training")
    phase1 = local_module("qwen_phase1", shared["source_paths"]["phase1"])
    if evaluation_complete(run_dir, shared, config, trained["adapter"], phase1):
        emit(f"Verified completed evaluation; skipping {run_id}")
        return
    output = run_dir / "evaluation"
    output.mkdir(exist_ok=True)
    identity = {"configuration_identity": config["identity"], "adapter": trained["adapter"],
                "csvs": {key: config["csvs"][key] for key in ("test", "messidor_external")},
                "score_method": "full_answer_logprob_softmax_argmax", "class_order": [0, 1, 2, 3, 4],
                "candidate_scores": "genuine complete-answer conditional log likelihood, normalized across five grades",
                "precision": "FP16", "helper_sha256": shared["source_sha256"]["helper"]}
    write_json(output / "evaluation_manifest.json", identity, immutable=True)
    offline(output / "cache")
    os.environ.update(UNSLOTH_COMPILE_DISABLE="1", UNSLOTH_DISABLE_COMPILE="1", TORCHINDUCTOR_DISABLE="1")
    import pandas
    from unsloth import FastVisionModel
    from peft import PeftModel
    import torch
    resources(batch, 10, gpu=True, expected_gpu=shared["gpu_identity"])
    helper = local_module("qwen_reproduction", shared["source_paths"]["helper"])
    model, processor = FastVisionModel.from_pretrained(model_name=shared["base"]["path"], load_in_4bit=True, local_files_only=True)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise SharedFailure("Evaluation base did not load in 4-bit mode")
    model = PeftModel.from_pretrained(model, str(run_dir / "best_lora"), local_files_only=True)
    FastVisionModel.for_inference(model)
    model = model.half()
    model.eval()
    all_metrics, hashes = {}, {}
    fields = ["original_image_id", "image_id", "image_path", "y_true", "y_pred", "p0", "p1", "p2", "p3", "p4"]
    for subset, partition in (("aptos_test", "test"), ("messidor_external", "messidor_external")):
        expected = read_csv(config["csvs"][partition]["path"])
        file = output / f"{subset}.predictions.csv"
        try:
            saved = read_csv(file, allow_empty=True) if file.exists() and file.stat().st_size else []
        except (OSError, ValueError, SharedFailure) as error:
            raise ConditionFailure(f"Partial prediction file is malformed; adapter preserved: {file}: {error}") from error
        if file.exists() and file.stat().st_size:
            with file.open(encoding="utf-8-sig", newline="") as stream:
                if csv.DictReader(stream).fieldnames != fields:
                    raise ConditionFailure(f"Prediction header differs: {file}")
        if len(saved) > len(expected):
            raise ConditionFailure("Partial prediction export exceeds frozen cohort")
        if saved:
            # Same verified ID/label/score checks for a partial prefix, with AUC
            # deferred until the whole five-class cohort is present.
            for row, target in zip(saved, expected):
                if row["image_id"] != target["id_code"] or row.get("original_image_id") != target["id_code"] or row["image_path"] != target["image_path"] or int(row["y_true"]) != int(target["label"]):
                    raise ConditionFailure("Partial predictions are not an exact prefix of this condition/cohort")
                scores = [float(row[f"p{i}"]) for i in range(5)]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores) or abs(sum(scores) - 1) > 1e-6 or int(row["y_pred"]) != max(range(5), key=lambda i: scores[i]):
                    raise ConditionFailure("Partial predictions contain invalid scores")
        with file.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not stream.tell():
                writer.writeheader()
            for target in expected[len(saved):]:
                scores = list(map(float, helper.score_full_answers(model, processor, target["image_path"])))
                if len(scores) != 5 or not all(math.isfinite(v) and 0 <= v <= 1 for v in scores) or abs(sum(scores) - 1) > 1e-6:
                    raise ConditionFailure("Scorer returned invalid grade scores; no Grade-0 fallback is used")
                row = {"original_image_id": target["id_code"], "image_id": target["id_code"], "image_path": target["image_path"], "y_true": int(target["label"]),
                       "y_pred": max(range(5), key=lambda i: scores[i]), **{f"p{i}": scores[i] for i in range(5)}}
                writer.writerow(row)
                stream.flush()
                saved.append(row)
                if len(saved) % 10 == 0 or len(saved) == len(expected):
                    emit(f"{run_id} {subset}: {len(saved)}/{len(expected)}")
                    write_json(output / "progress.json", {"subset": subset, "rows": len(saved), "total": len(expected), "time": utc()})
        metrics = prediction_metrics(saved, expected, phase1)
        all_metrics[subset], hashes[subset] = metrics, sha(file)
        write_json(output / f"{subset}.summary.json", {**identity, **metrics, "prediction_sha256": hashes[subset]})
        write_json(Path(str(file) + ".audit.json"), {
            "prediction_export_sha256": hashes[subset], "dataset": "aptos" if subset == "aptos_test" else "messidor2",
            "split": "test" if subset == "aptos_test" else "external", "run_id": run_id,
            "label_file": config["csvs"][partition]["path"], "label_file_sha256": config["csvs"][partition]["sha256"],
            "score_columns": [f"p{i}" for i in range(5)], "class_order": list(range(5)), "score_kind": "probability",
            "score_origin": "Saved complete-answer conditional likelihood softmax from the hash-linked helper",
            "producer_file": shared["source_paths"]["helper"], "producer_file_sha256": shared["source_sha256"]["helper"],
            "adapter_dir": str(run_dir / "best_lora"), "experiment_worker_sha256": shared["source_sha256"]["worker"]})
    write_json(run_dir / "evaluation_complete.json", {"configuration_identity": config["identity"], "adapter": trained["adapter"],
                                                       "ended_at": utc(), "metrics": all_metrics, "prediction_sha256": hashes})
    emit(f"Both frozen cohorts evaluated: {run_id}; legacy fallback audit remains separate in audit/")


def record_stage(batch, run_id, stage, state, **details):
    directory = batch / run_id if run_id else batch
    path = directory / "stage_status.json"
    record = read_json(path) if path.exists() else {}
    entry = dict(record.get(stage, {}))
    if state == "running":
        attempts = list(entry.get("attempts", []))
        attempts.append({"started_at": utc(), "status": "running"})
        entry["attempts"] = attempts
    entry.update(status=state, updated_at=utc(), **details)
    if state != "running" and entry.get("attempts"):
        entry["attempts"][-1].update(status=state, ended_at=utc(), **details)
    record[stage] = entry
    write_json(path, record)


def summarize(batch):
    preflight_result = read_json(batch / "preflight.json") if (batch / "preflight.json").exists() else {}
    queue_state = read_json(batch / "queue_status.json") if (batch / "queue_status.json").exists() else {}
    shared = read_json(batch / "shared_configuration.json") if (batch / "shared_configuration.json").exists() else {}
    phase1 = None
    rows = []
    for run_id, rank, clahe in QUEUE:
        directory = batch / run_id
        config, trained, evaluated, stages, artifact_errors = {}, {}, {}, {}, []
        try:
            config = read_json(directory / "resolved_configuration.json") if (directory / "resolved_configuration.json").exists() else {}
            stages = read_json(directory / "stage_status.json") if (directory / "stage_status.json").exists() else {}
            if config.get("eligible"):
                if (fingerprint({k: v for k, v in config.items() if k != "identity"}) != config.get("identity")
                        or config.get("shared_identity") != shared.get("identity")):
                    raise ConditionFailure("Summary withheld: condition/shared identity mismatch")
                trained = training_complete(directory, config) or {}
                if (directory / "evaluation_complete.json").exists():
                    if not trained:
                        raise ConditionFailure("Evaluation completion has no verified adapter")
                    source = shared["source_paths"]["phase1"]
                    if sha(source) != shared["source_sha256"]["phase1"]:
                        raise ConditionFailure("Phase 1 metric implementation changed; summary cannot revalidate exports")
                    for key in ("test", "messidor_external"):
                        record = config["csvs"][key]
                        if sha(record["path"]) != record["sha256"]:
                            raise ConditionFailure("Frozen test CSV changed; metrics withheld")
                    if phase1 is None:
                        phase1 = local_module("qwen_phase1_summary", source)
                    evaluated = evaluation_complete(directory, shared, config, trained["adapter"], phase1) or {}
        except Exception as error:
            artifact_errors.append(f"Artifact verification: {type(error).__name__}: {error}")
            evaluated = {}
        status = "complete" if evaluated else "trained_evaluation_pending" if trained else "not_started"
        errors = artifact_errors + [f"{key}: {value.get('error', '')}" for key, value in stages.items() if value.get("status") in {"failed", "blocked_shared", "interrupted"}]
        if errors:
            status = "evaluation_failed_adapter_preserved" if trained else "failed_or_interrupted"
        if preflight_result.get("status") == "blocked_shared":
            status = "blocked_shared"
        elif config and not config.get("selected", True):
            status = "excluded_by_configuration"
        elif config and not config.get("eligible"):
            status = "blocked_ablation"
        elif queue_state.get("status") == "blocked_shared" and status == "not_started":
            status = "not_started_shared_stop"
        row = {"run_id": run_id, "rank": rank, "clahe": clahe, "lora_scaling": 16 / rank, "status": status,
               "run_dir": str(directory), "adapter": str(directory / "best_lora"), "evaluation": str(directory / "evaluation"),
               "errors": "; ".join(errors + preflight_result.get("shared_blockers", []) + config.get("blockers", [])),
               "exclusion_reason": config.get("exclusion_reason", ""),
               "best_validation_loss": trained.get("best_validation_loss"), "epochs_completed": trained.get("epochs_completed")}
        for stage in ("train", "verify", "evaluate"):
            attempts = stages.get(stage, {}).get("attempts", [])
            row[f"{stage}_seconds"] = sum(a.get("elapsed_seconds", 0) for a in attempts)
            row[f"{stage}_started_at"] = attempts[0].get("started_at", "") if attempts else ""
            row[f"{stage}_ended_at"] = attempts[-1].get("ended_at", "") if attempts else ""
        for subset in ("aptos_test", "messidor_external"):
            metrics = evaluated.get("metrics", {}).get(subset, {})
            for key in ("accuracy", "qwk", "macro_ovr_auc"):
                row[f"{subset}_{key}"] = metrics.get(key)
        rows.append(row)
    # Summaries are mutable progress products, unlike frozen inputs/configs.
    csv_path = batch / "batch_summary.csv"
    with csv_path.with_suffix(".partial.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix(".partial.csv").replace(csv_path)
    lines = ["# Qwen3-VL-8B four-condition experiment queue", "",
             f"Updated {utc()}. Queue state: {queue_state.get('status', 'unknown')}; preflight: {preflight_result.get('status', 'not run')}. "
             f"Current launch elapsed seconds: {queue_state.get('elapsed_seconds', 'in progress')}; "
             f"12-hour overrun seconds: {queue_state.get('planning_overrun_seconds', 0)}.", "",
             "Previously reported reference: **Qwen3-VL-8B Retraining 2 — APTOS accuracy 80.65%, QWK 0.8915, macro ROC-AUC 0.9362**. These are supplied figures, separate from new measurements.", "",
             "The fresh r16_clahe_on condition is the shared control for rank and CLAHE comparisons. All checkpoints use APTOS validation loss. Held-out APTOS and Messidor results never select configurations or checkpoints.", "",
             "| Run | Rank | CLAHE | alpha/r | Status | APTOS accuracy / QWK / AUC | Messidor accuracy / QWK / AUC | Train / eval seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    def display(value):
        return "—" if value is None else f"{value:.6f}"
    for row in rows:
        a = " / ".join(display(row[f"aptos_test_{key}"]) for key in ("accuracy", "qwk", "macro_ovr_auc"))
        m = " / ".join(display(row[f"messidor_external_{key}"]) for key in ("accuracy", "qwk", "macro_ovr_auc"))
        lines.append(f"| {row['run_id']} | {row['rank']} | {row['clahe']} | {row['lora_scaling']} | {row['status']} | {a} | {m} | {row['train_seconds']:.1f} / {row['evaluate_seconds']:.1f} |")
    lines += ["", "Planning window: roughly 12 hours, not a hard cutoff. Fixed validation-based stopping is preserved; actual durations and overruns are recorded. The queue stops when eligible work completes or a shared failure/interruption occurs.", "",
              "Preflight blockers: " + "; ".join(preflight_result.get("shared_blockers", []) + preflight_result.get("ablation_blockers", [])), "",
              "Excluded by configuration: " + ", ".join(preflight_result.get("excluded_runs", [])), "",
              "Legacy invalid-output/Grade-0 fallback analysis stays in audit/phase1_report.md. This queue scores five complete grade answers and never substitutes Grade 0 for invalid scores.", ""]
    for row in rows:
        lines += [f"- `{row['run_id']}`: `{row['run_dir']}`; adapter `{row['adapter']}`; predictions `{row['evaluation']}`; errors: {row['errors'] or 'none'}."]
    lines += ["", "Source/input/weight hashes and effective settings are in shared_configuration.json and each resolved_configuration.json. Check stage_status.json for attempts/start/end times, queue_status.json for interruption/shared stops, and preflight.json / pairing_checks.csv for input and channel-handling evidence.", "",
              "Summary metrics are revalidated against the saved adapter, prediction hashes, exact frozen IDs/labels and recalculated five-class metrics. Changed/corrupt artifacts are withheld and retained for review.", ""]
    target = batch / "batch_report.md"
    target.with_suffix(".partial.md").write_text("\n".join(lines), encoding="utf-8")
    target.with_suffix(".partial.md").replace(target)
    emit(f"Batch summary: {csv_path}; report: {target}")


def classify(error):
    if isinstance(error, SharedFailure):
        return 3
    text = str(error).lower()
    if isinstance(error, (ImportError, ModuleNotFoundError)) or (isinstance(error, OSError) and getattr(error, "errno", None) == 28):
        return 3
    if "out of memory" not in text and any(key in text for key in ("cuda error", "device-side assert", "illegal memory access", "driver version", "dll load failed")):
        return 3
    return 1


class NotebookOrchestrator:
    """Small notebook controller; models and scientific checks stay in workers.

    Construction and show_* methods never launch processes. Each explicit action
    acquires the batch lock, streams one child at a time, and releases the lock
    before returning control to a notebook cell.
    """

    def __init__(self, request, batch_dir, gpu_index="0"):
        self.request = dict(request)
        self.batch = Path(batch_dir).expanduser().resolve()
        self.worker = Path(__file__).resolve()
        self.gpu_index = str(gpu_index)
        self._busy = False
        self._active = None
        self._state = {}

    @staticmethod
    def _run_id(run_id):
        if run_id not in {item[0] for item in QUEUE}:
            raise ValueError(f"Unknown condition: {run_id}")
        return run_id

    def _validate_paths(self):
        safe_output(self.request["project_root"], self.batch)
        for key in ("clahe_on_images", "clahe_off_images", "base_snapshot"):
            if self.request.get(key):
                protected = Path(self.request[key]).expanduser().resolve()
                if self.batch == protected or protected in self.batch.parents:
                    raise ValueError(f"Batch output must be separate from {key}: {protected}")

    def _save_state(self):
        write_json(self.batch / "queue_status.json", self._state)

    def _launch(self, stage, run_id=None):
        import subprocess
        command = [sys.executable, "-u", str(self.worker), stage, "--batch-dir", str(self.batch)]
        if stage == "preflight":
            command += ["--request", str(self.batch / "launch_request.json")]
        if run_id:
            command += ["--run-id", self._run_id(run_id)]
        directory = self.batch / run_id if run_id else self.batch
        directory.mkdir(exist_ok=True)
        self._active = {"stage": stage, "run_id": run_id, "started_at": utc()}
        if stage != "summarize":
            self._state["status"] = "running"
        self._state["active"] = self._active
        self._save_state()
        write_json(directory / f"{stage}.command.json", command)
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": self.gpu_index,
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        child = None
        try:
            with (directory / f"{stage}.log").open("a", encoding="utf-8") as log:
                log.write(f"\n=== {utc()} ===\n")
                log.flush()
                child = subprocess.Popen(command, cwd=self.worker.parent, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                         errors="replace", bufsize=1, env=environment, shell=False)
                self._active["pid"] = child.pid
                self._save_state()
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
        except BaseException:
            if child is not None:
                if child.poll() is None:
                    try:
                        child.terminate()
                    except OSError:
                        pass  # The process may have exited after poll().
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                else:
                    child.wait()
            raise
        finally:
            if child is not None and child.stdout is not None:
                child.stdout.close()
            emit(f"{run_id or 'batch'} / {stage}: exit {child.returncode if child is not None else 'not launched'}; log={directory / (stage + '.log')}")
        if code == 130:
            raise KeyboardInterrupt("Worker interrupted; ending the current notebook action")
        self._state["last_stage"] = {**self._active, "exit_status": code, "ended_at": utc()}
        self._active = None
        self._state["active"] = None
        return code

    def _record_stop(self, error):
        self._state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else
                           "blocked_shared" if isinstance(error, SharedFailure) else "launcher_failed",
                           error=f"{type(error).__name__}: {error}", ended_at=utc(), active=self._active,
                           elapsed_seconds=time.monotonic() - self._started)
        self._state["planning_overrun_seconds"] = max(0.0, self._state["elapsed_seconds"] - 12 * 3600)
        self._save_state()
        if self._active is not None:
            action = self._active
            directory = self.batch / action["run_id"] if action["run_id"] else self.batch
            elapsed = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(action["started_at"])).total_seconds())
            path = directory / "stage_status.json"
            statuses = read_json(path) if path.exists() else {}
            entry = statuses.get(action["stage"], {})
            attempts = entry.setdefault("attempts", [])
            if not attempts or attempts[-1].get("started_at", "") < action["started_at"]:
                attempts.append({"started_at": action["started_at"]})
            entry.update(status=self._state["status"], error=self._state["error"], updated_at=utc(), elapsed_seconds=elapsed)
            attempts[-1].update(status=entry["status"], ended_at=utc(), elapsed_seconds=elapsed)
            statuses[action["stage"]] = entry
            write_json(path, statuses)
            summary_path = self.batch / "batch_summary.csv"
            if action["run_id"] and summary_path.exists():
                with summary_path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    if row["run_id"] == action["run_id"]:
                        row["status"], row["errors"] = entry["status"], entry["error"]
                        row[f"{action['stage']}_ended_at"] = utc()
                if rows:
                    temporary = summary_path.with_suffix(".partial.csv")
                    with temporary.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    temporary.replace(summary_path)
        with (self.batch / "batch_report.md").open("a", encoding="utf-8") as report:
            report.write(f"\nNotebook action ended: {self._state['status']} at {utc()}; see queue_status.json. Any active child was stopped and waited for.\n")

    @contextmanager
    def _session(self, mode):
        if self._busy:
            raise RuntimeError("A notebook action is already active")
        self._validate_paths()
        self.batch.mkdir(parents=True, exist_ok=True)
        lock = (self.batch / "queue.lock").open("a+b")
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                lock.seek(0, 2)
                if not lock.tell():
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = self._busy = True
            self._active = None
            self._started = time.monotonic()
            path = self.batch / "queue_status.json"
            previous = read_json(path) if path.exists() else {}
            self._state = {"status": "starting", "mode": mode, "started_at": utc(), "active": None, "planning_hours": 12}
            if mode not in {"preflight", "all"} and previous.get("status") in {"blocked_shared", "interrupted", "launcher_failed"}:
                raise SharedFailure("Run the Preflight cell again before starting another stage after a shared failure/interruption")
            yield
            self._state.update(ended_at=utc(), active=None, elapsed_seconds=time.monotonic() - self._started)
            self._state["planning_overrun_seconds"] = max(0.0, self._state["elapsed_seconds"] - 12 * 3600)
            self._save_state()
            code = self._launch("summarize")
            self._state["summary_exit_status"] = code
            if code:
                self._state["status"] += "_summary_failed"
            self._save_state()
        except BaseException as error:
            if locked:
                self._record_stop(error)
            raise
        finally:
            try:
                if locked:
                    if os.name == "nt":
                        lock.seek(0)
                        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            finally:
                lock.close()
                self._busy = False
            if locked:
                emit(f"Notebook action ended: {self._state['status']}; report={self.batch / 'batch_report.md'}")

    def _preflight(self):
        write_json(self.batch / "launch_request.json", self.request)
        code = self._launch("preflight")
        if code:
            raise SharedFailure(f"Preflight exited {code}; inspect {self.batch / 'preflight.json'} and preflight.log")
        return read_json(self.batch / "preflight.json")

    def preflight(self):
        """Run only shared preflight; no training/evaluation is queued."""
        with self._session("preflight"):
            result = self._preflight()
            self._state["status"] = "preflight_ready"
        self.show_preflight()
        return result["status"]

    def _ready(self, run_id):
        self._run_id(run_id)
        path = self.batch / "preflight.json"
        if not path.exists():
            raise SharedFailure("Run the Preflight cell before launching an individual stage")
        if read_json(self.batch / "launch_request.json") != self.request:
            raise SharedFailure("Configured paths changed; run the Preflight cell again")
        result = read_json(path)
        if result.get("status") not in {"ready", "ready_rank_comparison_ablation_blocked"}:
            raise SharedFailure("Shared preflight is blocked; inspect the Preflight results cell")
        condition = next((item for item in result["runs"] if item["run_id"] == run_id), None)
        if not condition:
            raise SharedFailure(f"Preflight has no resolved condition: {run_id}")
        if not condition["eligible"]:
            emit(f"{run_id}: " + (condition.get("exclusion_reason") or f"blocked: {condition['blockers']}"))
            return False
        return True

    def _execute(self, stage, run_id):
        directory = self.batch / run_id
        if stage in {"verify", "evaluate"} and not (directory / "training_complete.json").is_file():
            emit(f"{run_id}: no training completion marker; inspect train.log and the Training status cell first")
            return 1
        if stage == "evaluate":
            # A single evaluation-cell retry always verifies the retained adapter
            # in a CPU process before opening the fresh evaluation process.
            code = self._launch("verify", run_id)
            if code == 3:
                raise SharedFailure(f"{run_id} / verify: shared failure; inspect stage_status.json")
            if code:
                return code
        code = self._launch(stage, run_id)
        if code == 3:
            raise SharedFailure(f"{run_id} / {stage}: shared failure; inspect stage_status.json")
        return code

    def run_stage(self, stage, run_id):
        """Launch one named stage for one run; never start the next condition."""
        if stage not in {"train", "verify", "evaluate"}:
            raise ValueError("Choose train, verify, or evaluate")
        self._run_id(run_id)
        with self._session(f"{run_id}/{stage}"):
            if not self._ready(run_id):
                self._state["status"] = "excluded_by_configuration" if run_id == "r16_clahe_off" and not self.request.get("include_clahe_off", True) else "blocked_ablation"
                return 2
            code = self._execute(stage, run_id)
            self._state["status"] = "finished_stage" if code == 0 else "isolated_stage_failure"
        return code

    def run_all(self):
        """Optional unattended route; retains the same per-run stage artifacts."""
        with self._session("all"):
            result = self._preflight()
            if [item["run_id"] for item in result["runs"]] != [item[0] for item in QUEUE]:
                raise SharedFailure("Preflight queue order differs from the four reviewed conditions")
            for run_id, _, _ in QUEUE:
                if not self._ready(run_id):
                    continue
                if self._execute("train", run_id):
                    emit(f"Isolated training failure in {run_id}; continuing next eligible condition")
                    continue
                if self._execute("evaluate", run_id):
                    emit(f"Evaluation/verification failed for {run_id}; adapter retained; continuing next eligible condition")
            self._state["status"] = "finished_eligible_work"
        self.show_batch_summary()

    def show_preflight(self):
        """Read saved preflight only; do not query the GPU or rescan images."""
        path = self.batch / "preflight.json"
        if not path.exists():
            print("No saved preflight. Run the Preflight cell when ready.")
            return
        print(json.dumps(read_json(path), indent=2, ensure_ascii=False))
        print(f"Preflight log: {self.batch / 'preflight.log'}")

    def show_configuration(self, run_id):
        directory = self.batch / self._run_id(run_id)
        path = directory / "resolved_configuration.json"
        if not path.exists():
            print(f"{run_id}: no resolved configuration yet; run Preflight first.")
            return
        config = read_json(path)
        view = {key: value for key, value in config.items() if key not in {"images", "pairing"}}
        if config.get("images"):
            view["image_branch"] = {"root": config["images"]["root"], "identity": config["images"]["identity"]}
        print(json.dumps(view, indent=2, ensure_ascii=False))
        print(f"Full configuration: {path}")

    def show_run(self, run_id):
        """Read saved status/metrics without loading images, tensors or models."""
        directory = self.batch / self._run_id(run_id)
        print(f"Run: {run_id}\nArtifacts: {directory}")
        config_path = directory / "resolved_configuration.json"
        if config_path.exists():
            config = read_json(config_path)
            print(f"Selected: {config.get('selected', True)}; eligible: {config.get('eligible')}; "
                  f"exclusion: {config.get('exclusion_reason', '')}; blockers: {config.get('blockers', [])}")
        status_path = directory / "stage_status.json"
        statuses = read_json(status_path) if status_path.exists() else {}
        for stage in ("train", "verify", "evaluate"):
            print(f"\n{stage}: {json.dumps(statuses.get(stage, {'status': 'not_started'}), indent=2)}")
        adapter_path = directory / "best_lora/adapter_config.json"
        if adapter_path.exists():
            adapter = read_json(adapter_path)
            print("\nSaved adapter: " + json.dumps({key: adapter.get(key) for key in
                  ("r", "lora_alpha", "lora_dropout", "bias", "use_rslora", "target_modules")}, indent=2))
        for subset in ("aptos_test", "messidor_external"):
            path = directory / "evaluation" / f"{subset}.summary.json"
            if path.exists():
                metrics = read_json(path)
                print(f"\n{subset}: " + json.dumps({key: metrics.get(key) for key in
                      ("samples", "accuracy", "qwk", "macro_ovr_auc", "per_grade_auc")}, indent=2))
        print("\nRead-only inspection of saved records. The Verify cell performs adapter integrity checks.")

    def show_log(self, run_id, stage, lines=40):
        if stage not in {"preflight", "train", "verify", "evaluate", "summarize"}:
            raise ValueError("Unknown log stage")
        directory = self.batch / self._run_id(run_id) if run_id else self.batch
        path = directory / f"{stage}.log"
        print(f"Log: {path}")
        if not path.exists():
            print("No saved log yet.")
            return
        with path.open(encoding="utf-8", errors="replace") as stream:
            print("".join(deque(stream, maxlen=max(1, int(lines)))), end="")

    def show_batch_summary(self):
        path = self.batch / "batch_summary.csv"
        if not path.exists():
            print("No saved batch summary yet.")
            return
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                print(f"{row['run_id']}: {row['status']}; APTOS acc/QWK/AUC="
                      f"{row.get('aptos_test_accuracy')}/{row.get('aptos_test_qwk')}/{row.get('aptos_test_macro_ovr_auc')}; "
                      f"Messidor acc/QWK/AUC={row.get('messidor_external_accuracy')}/{row.get('messidor_external_qwk')}/{row.get('messidor_external_macro_ovr_auc')}")
                if row.get("errors"):
                    print("  " + row["errors"])
        print(f"CSV: {path}\nReport: {self.batch / 'batch_report.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "train", "verify", "evaluate", "summarize", "interrupt"])
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--run-id", choices=[row[0] for row in QUEUE])
    parser.add_argument("--interrupted-stage", choices=["preflight", "train", "verify", "evaluate"])
    args = parser.parse_args(argv)
    batch = args.batch_dir.resolve()
    if not batch.is_dir():
        raise ValueError("The notebook must create and lock the batch directory first")
    sys.dont_write_bytecode = True
    started = time.monotonic()
    if args.stage == "interrupt":
        record_stage(batch, args.run_id, args.interrupted_stage or "preflight", "interrupted", error="Notebook interrupted; active child was stopped and waited for")
        summarize(batch)
        return 130
    if args.stage == "summarize":
        summarize(batch)
        return 0
    record_stage(batch, args.run_id, args.stage, "running")
    try:
        if args.stage == "preflight":
            preflight(read_json(args.request), batch)
        elif args.stage == "train":
            train(batch, args.run_id)
        elif args.stage == "evaluate":
            evaluate(batch, args.run_id)
        else:
            shared, config = load_condition(batch, args.run_id)
            if not training_complete(batch / args.run_id, config):
                raise ConditionFailure("Training has no verified completion record")
            emit(f"Saved adapter verified after training process exit: {args.run_id}")
        record_stage(batch, args.run_id, args.stage, "complete", elapsed_seconds=time.monotonic() - started)
        summarize(batch)
        return 0
    except KeyboardInterrupt:
        record_stage(batch, args.run_id, args.stage, "interrupted", elapsed_seconds=time.monotonic() - started)
        summarize(batch)
        return 130
    except Exception as error:
        code = classify(error)
        record_stage(batch, args.run_id, args.stage, "blocked_shared" if code == 3 else "failed",
                     error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        traceback.print_exc()
        emit(f"Stage failed, exit {code}: {error}")
        summarize(batch)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
