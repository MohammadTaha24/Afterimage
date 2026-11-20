"""Run the Qwen smoke test or one fresh APTOS experiment condition."""

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import psutil


def write_json(path, record):
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_worker(path):
    spec = importlib.util.spec_from_file_location("afterimage_worker", path)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    return worker


def candidate_tables(manifest_path, branch, recorded_config, worker, historical=False):
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("image_size", 512) != 512:
        raise ValueError("Qwen requires the 512-pixel APTOS build")
    if (Path(manifest["source_config"]).resolve() != recorded_config.resolve()
            or manifest["source_config_sha256"] != worker.sha(recorded_config)):
        raise ValueError("Candidate source differs from the recorded Qwen batch")
    audit_path = manifest_path.with_name("pairing_audit.csv")
    if worker.sha(audit_path) != manifest["pairing_audit_sha256"]:
        raise ValueError("Candidate pairing audit changed")
    audit_rows = worker.read_csv(audit_path)
    audit = {row["image_id"]: row for row in audit_rows}
    if len(audit) != len(audit_rows) or len(audit) != manifest["rows"]:
        raise ValueError("Candidate pairing audit contains repeated or missing IDs")
    if not historical:
        for field in ("original_sha256", f"{branch}_pixels_sha256"):
            if len({row[field] for row in audit_rows}) != len(audit):
                raise ValueError(f"Candidate has repeated content for {field}")
    tables = {}
    seen = set()
    for split in ("train", "val", "test"):
        record = manifest["branches"][branch][split]
        path = Path(record["path"])
        if worker.sha(path) != record["sha256"]:
            raise ValueError(f"Candidate list changed: {path}")
        rows = worker.read_csv(path)
        if len(rows) != record["rows"]:
            raise ValueError(f"Unexpected number of candidate rows in {path}")
        for row in rows:
            image_id = row["id_code"]
            binding = audit[image_id]
            if (image_id in seen or row["split"] != split or binding["split"] != split
                    or row["source_dataset"] != "aptos" or row["label"] != binding["grade"]
                    or row["image_path"] != binding[f"{branch}_path"]):
                raise ValueError(f"Candidate row differs from pairing audit: {image_id}")
            seen.add(image_id)
            if worker.sha(Path(row["image_path"])) != binding[f"{branch}_sha256"]:
                raise ValueError(f"Candidate image changed: {image_id}")
        tables[split] = rows
    if seen != audit.keys():
        raise ValueError("Candidate lists do not cover the pairing audit")
    return tables


def full_historical_tables(args, recorded, recorded_path, worker):
    tables = candidate_tables(args.candidate, args.clahe, recorded_path, worker, historical=True)
    clean = candidate_tables(args.comparison_manifest, args.clahe, recorded_path, worker)
    for split, rows in tables.items():
        archived = Path(recorded["csvs"][split]["path"])
        if worker.sha(archived) != recorded["csvs"][split]["sha256"]:
            raise ValueError(f"Archived split list changed: {archived}")
        identity = lambda table: {(row["id_code"], row["label"], row["split"], row["filename"]) for row in table}
        if identity(rows) != identity(worker.read_csv(archived)):
            raise ValueError(f"Historical comparison requires the complete archived {split} membership")
    original = {row["id_code"]: row for row in tables["test"]}
    common = []
    for row in clean["test"]:
        match = original.get(row["id_code"])
        if (match is None or row["label"] != match["label"]
                or worker.sha(Path(row["image_path"])) != worker.sha(Path(match["image_path"]))):
            raise ValueError(f"Historical and duplicate-excluded test inputs differ: {row['id_code']}")
        common.append(match)
    tables["test"] = common
    return tables


def verify_base(path, files, worker):
    for name, record in files.items():
        item = path / name
        if item.stat().st_size != record["size"] or worker.sha(item) != record["sha256"]:
            raise ValueError(f"Pinned model file differs: {item}")


def identity(config):
    body = {key: value for key, value in config.items() if key != "identity"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(args, output):
    project = args.project.resolve(strict=True)
    batch = project / "experiments/artifacts/qwen_four_run_001"
    shared = json.loads((batch / "shared_configuration.json").read_text(encoding="utf-8"))
    recorded_path = batch / "r16_clahe_on/resolved_configuration.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    worker = load_worker(project / "experiments/qwen_experiment_worker.py")
    worker.safe_output(project, output)
    for name in ("worker", "helper", "trainer", "phase1"):
        path = Path(shared["source_paths"][name])
        if worker.sha(path) != shared["source_sha256"][name]:
            raise ValueError(f"Source differs from the saved run: {path}")
    worker.packages(shared["package_versions"])
    base = args.base_snapshot.resolve(strict=True) if args.base_snapshot else Path(shared["base"]["path"])
    verify_base(base, shared["base"]["files"], worker)
    if args.historical_split:
        selected = full_historical_tables(args, recorded, recorded_path, worker)
    elif args.candidate:
        available = candidate_tables(args.candidate, args.clahe, recorded_path, worker)
        selected = available if args.full else balanced_subset(available)
    else:
        selected = archived_pilot_subset(recorded, worker)
    ids = [row["id_code"] for rows in selected.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Pilot images repeat an ID")
    output.mkdir(parents=True, exist_ok=False)
    prefix = "aptos" if args.full else "pilot"
    for split, rows in selected.items():
        write_csv(output / f"{prefix}_{split}.csv", rows)
    image_records = [
        {"split": split, "image_path": row["image_path"], "sha256": worker.sha(Path(row["image_path"]))}
        for split, rows in selected.items() for row in rows
    ]
    write_csv(output / "input_images.csv", image_records)
    config = {
        "purpose": "Fresh APTOS comparison" if args.full else "Training and scoring smoke test on a small APTOS subset",
        "mode": "full" if args.full else "pilot",
        "source_batch": str(batch), "worker": shared["source_paths"]["worker"],
        "helper": shared["source_paths"]["helper"], "trainer": shared["source_paths"]["trainer"],
        "phase1": shared["source_paths"]["phase1"],
        "source_sha256": shared["source_sha256"], "base": str(base), "base_files": shared["base"]["files"],
        "runner_sha256": worker.sha(Path(__file__).resolve()),
        "package_versions": shared["package_versions"], "rank": args.rank, "clahe": args.clahe == "on",
        "target_modules": recorded["target_modules"], "steps": None if args.full else args.steps,
        "training": {**shared["training"], "lora_r": args.rank, "trainer_seed": args.trainer_seed,
                     "warmup_steps": shared["training"]["warmup_steps"] if args.full else 0},
        "adapter_seed": args.adapter_seed,
        "candidate_manifest": str(args.candidate.resolve()) if args.candidate else None,
        "candidate_manifest_sha256": worker.sha(args.candidate) if args.candidate else None,
        "historical_split": args.historical_split,
        "comparison_manifest": str(args.comparison_manifest.resolve()) if args.comparison_manifest else None,
        "comparison_manifest_sha256": worker.sha(args.comparison_manifest) if args.comparison_manifest else None,
        "input_images_sha256": worker.sha(output / "input_images.csv"),
        "rows": {split: len(rows) for split, rows in selected.items()},
        "split_prefix": prefix,
        "split_sha256": {split: worker.sha(output / f"{prefix}_{split}.csv") for split in selected},
    }
    config["identity"] = identity(config)
    write_json(output / "pilot_configuration.json", config)
    return config


def archived_pilot_subset(recorded, worker):
    by_name = {entry["filename"]: entry for entry in recorded["images"]["files"]}
    counts = Counter(entry["sha256"] for entry in by_name.values())
    available = {}
    for split in ("train", "val"):
        csv_record = recorded["csvs"][split]
        path = Path(csv_record["path"])
        if worker.sha(path) != csv_record["sha256"]:
            raise ValueError(f"Photo list differs from the saved run: {path}")
        available[split] = [
            row for row in worker.read_csv(path)
            if row["source_dataset"] == "aptos" and row["split"] == split
            and counts[by_name[row["filename"]]["sha256"]] == 1
        ]
    selected = balanced_subset(available)
    for rows in selected.values():
        for row in rows:
            image_path = Path(row["image_path"])
            if worker.sha(image_path) != by_name[row["filename"]]["sha256"]:
                raise ValueError(f"Image differs from the saved run: {image_path}")
    return selected


def balanced_subset(tables):
    selected = {}
    for split, per_grade in (("train", 16), ("val", 2)):
        subset = []
        for grade in range(5):
            group = [row for row in tables[split] if int(row["label"]) == grade][:per_grade]
            if len(group) != per_grade:
                raise ValueError(f"Too few unique images for {split}, grade {grade}")
            subset.extend(group)
        selected[split] = subset
    return selected


def run_model_stage(output, stage, resume=None):
    cfg = json.loads((output / "pilot_configuration.json").read_text(encoding="utf-8"))
    if "identity" in cfg and identity(cfg) != cfg["identity"]:
        raise ValueError("Run configuration changed")
    worker = load_worker(Path(cfg["worker"]))
    if cfg.get("runner_sha256") and worker.sha(Path(__file__).resolve()) != cfg["runner_sha256"]:
        raise ValueError("The runner differs from the saved setup")
    names = ("worker", "helper", "trainer", "phase1") if cfg.get("mode") == "full" else ("worker", "helper", "trainer")
    for name in names:
        if worker.sha(Path(cfg[name])) != cfg["source_sha256"][name]:
            raise ValueError(f"Source changed after pilot setup: {name}")
    if resume and checked_resume(output, cfg, worker) != resume.resolve():
        raise ValueError("The requested resume checkpoint is no longer the verified checkpoint")
    cache = output / "cache" / stage
    worker.offline(cache)
    compiler_temp = cache / "tmp"
    compiler_temp.mkdir(exist_ok=True)
    tempfile.tempdir = str(compiler_temp)
    os.environ.update(TEMP=str(compiler_temp), TMP=str(compiler_temp))
    if stage == "score":
        os.environ.update(UNSLOTH_COMPILE_DISABLE="1", UNSLOTH_DISABLE_COMPILE="1",
                          TORCHINDUCTOR_DISABLE="1")
    import pandas  # Load before Unsloth to match the recorded Windows setup.
    from unsloth import FastVisionModel
    import torch

    worker.resources(output, 10, gpu=True)
    worker.packages(cfg["package_versions"])
    helper = worker.local_module("qwen_reproduction", cfg["helper"])
    score_split = "test" if cfg.get("mode") == "full" else "val"
    tables = {}
    prefix = cfg.get("split_prefix", "pilot")
    for split in (("train", "val") if stage == "train" else (score_split,)):
        path = output / f"{prefix}_{split}.csv"
        if worker.sha(path) != cfg["split_sha256"][split]:
            raise ValueError(f"Pilot photo list changed: {path}")
        tables[split] = worker.read_csv(path)
    if "input_images_sha256" in cfg:
        path = output / "input_images.csv"
        if worker.sha(path) != cfg["input_images_sha256"]:
            raise ValueError("Saved image fingerprints changed")
        for row in worker.read_csv(path):
            if row["split"] in tables and worker.sha(Path(row["image_path"])) != row["sha256"]:
                raise ValueError(f"Image changed: {row['image_path']}")
    torch.cuda.reset_peak_memory_stats()
    model, processor = FastVisionModel.from_pretrained(
        model_name=cfg["base"], load_in_4bit=True, local_files_only=True)
    if not getattr(model, "is_loaded_in_4bit", False) or getattr(model, "peft_config", None):
        raise ValueError("Expected a fresh 4-bit base")
    if stage == "train":
        result = train(output, cfg, worker, helper, model, processor, tables, FastVisionModel, torch, resume)
    else:
        result = score(output, cfg, worker, helper, model, processor, tables[score_split], FastVisionModel, score_split)
    result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["cuda_runtime"] = torch.version.cuda
    result["configuration_identity"] = cfg.get("identity")
    worker.write_json(output / f"{stage}_result.json", result)


def train(output, cfg, worker, helper, model, processor, tables, vision_model, torch, resume=None):
    from transformers import EarlyStoppingCallback, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    definitions = worker.local_module("qwen_training_definitions", cfg["trainer"])
    sampler_rng = torch.Generator().manual_seed(cfg["training"]["trainer_seed"])
    if resume:
        sampler_rng.set_state(torch.load(resume / "sampler_rng.pth", map_location="cpu", weights_only=True))

    class PilotTrainer(definitions.BalancedSFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            sampler = super()._get_train_sampler(train_dataset)
            if not isinstance(sampler, torch.utils.data.WeightedRandomSampler):
                raise ValueError("Pilot requires the existing weighted sampler")
            sampler.generator = sampler_rng
            return sampler

    class SaveVerifiedCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            torch.save(sampler_rng.get_state(), checkpoint / "sampler_rng.pth")
            worker.write_json(checkpoint / "checkpoint_identity.json", worker.checkpoint_record(checkpoint, cfg))
            print(f"Verified checkpoint: {checkpoint}", flush=True)

    training = cfg["training"]
    full = cfg.get("mode") == "full"
    adapter_args = {
        "r": cfg["rank"], "lora_alpha": training["lora_alpha"], "lora_dropout": training["lora_dropout"],
        "bias": training["bias"], "target_modules": cfg["target_modules"], "use_rslora": False,
        "random_state": cfg["adapter_seed"], "use_gradient_checkpointing": True,
        "temporary_location": str(output / "cache/train/saved_buffers"),
    }
    write_json(output / "adapter_constructor_arguments.json", adapter_args)
    model = vision_model.get_peft_model(model, **adapter_args)
    for peft_config in model.peft_config.values():
        peft_config.base_model_name_or_path = worker.BASE_NAME
        worker.adapter_config_matches(peft_config.to_dict(), cfg)
    vision_model.for_training(model)

    def short_config(**kwargs):
        return SFTConfig(**{**kwargs, "max_steps": cfg["steps"], "logging_steps": 1,
                            "eval_strategy": "steps", "save_strategy": "steps",
                            "eval_steps": cfg["steps"], "save_steps": cfg["steps"]})

    args = helper.make_sft_config(cfg, output / "checkpoints", SFTConfig if full else short_config)
    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=training["early_stopping_patience"],
        early_stopping_threshold=training["early_stopping_threshold"],
    ), SaveVerifiedCheckpoint()] if full else []
    processor_key = "tokenizer" if "tokenizer" in inspect.signature(SFTTrainer.__init__).parameters else "processing_class"
    trainer = PilotTrainer(
        model=model, args=args, **{processor_key: processor},
        train_dataset=worker.formatted_dataset(tables["train"], helper),
        eval_dataset=worker.formatted_dataset(tables["val"], helper),
        data_collator=UnslothVisionDataCollator(model, processor, resize=(512, 512), max_seq_length=2048),
        callbacks=callbacks,
    )
    if trainer.args.per_device_eval_batch_size == training["batch_size"]:
        trainer.args.per_device_eval_batch_size = training["eval_batch_size"]
    expected = {"seed": training["trainer_seed"], "max_steps": -1 if full else cfg["steps"],
                "per_device_train_batch_size": training["batch_size"],
                "per_device_eval_batch_size": training["eval_batch_size"],
                "gradient_accumulation_steps": training["grad_accum"],
                "num_train_epochs": training["epochs"], "eval_strategy": "epoch" if full else "steps",
                "save_strategy": "epoch" if full else "steps", "warmup_steps": training["warmup_steps"],
                "metric_for_best_model": "eval_loss", "load_best_model_at_end": True}
    actual = {key: getattr(trainer.args, key) for key in expected}
    if actual != expected:
        raise ValueError(f"Trainer settings differ: expected {expected}, got {actual}")
    write_json(output / "resolved_training_arguments.json", actual)
    if full:
        worker.write_json(output / "training_started.json", {"configuration_identity": cfg["identity"], "time": worker.utc()})
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    winner = trainer.state.best_model_checkpoint
    loss = trainer.state.best_metric
    if (not full and trainer.state.global_step != cfg["steps"]) or not winner or loss is None or not math.isfinite(loss):
        raise ValueError("Training did not finish with a validation-selected checkpoint")
    adapter = output / "best_lora"
    trainer.save_model(str(adapter))
    processor.save_pretrained(str(adapter))
    return {"global_step": trainer.state.global_step, "checkpoint": winner, "validation_loss": loss,
            "adapter": worker.adapter_record(adapter, cfg), "history": trainer.state.log_history}


def score(output, cfg, worker, helper, model, processor, rows, vision_model, split):
    from peft import PeftModel

    adapter = output / "best_lora"
    expected = json.loads((output / "train_result.json").read_text(encoding="utf-8"))
    if (expected.get("configuration_identity") != cfg.get("identity")
            or worker.adapter_record(adapter, cfg) != expected["adapter"]):
        raise ValueError("Saved adapter changed before reload")
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    vision_model.for_inference(model)
    model = model.half()
    model.eval()
    predictions = []
    for row in rows:
        scores = [float(value) for value in helper.score_full_answers(model, processor, row["image_path"])]
        if len(scores) != 5 or not all(math.isfinite(v) and 0 <= v <= 1 for v in scores) or abs(sum(scores) - 1) > 1e-6:
            raise ValueError(f"Invalid five-grade scores for {row['id_code']}")
        predictions.append({
            "original_image_id": row["id_code"], "image_id": row["id_code"], "image_path": row["image_path"],
            "y_true": int(row["label"]), "y_pred": max(range(5), key=scores.__getitem__),
            **{f"p{grade}": value for grade, value in enumerate(scores)},
        })
    if split == "val":
        write_csv(output / "pilot_validation.predictions.csv", predictions)
        return {"validation_photos_scored": len(predictions), "adapter": expected["adapter"]}
    phase1 = worker.local_module("qwen_phase1_metrics", cfg["phase1"])
    metrics = worker.prediction_metrics(predictions, rows, phase1)
    path = output / "aptos_test.predictions.csv"
    temporary = path.with_suffix(".csv.partial")
    write_csv(temporary, predictions)
    temporary.replace(path)
    return {"metrics": metrics, "predictions_sha256": worker.sha(path), "adapter": expected["adapter"]}


def gpu_query(fields):
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        timeout=5, text=True, stderr=subprocess.STDOUT,
    ).strip()


def system_specs():
    return {"os": platform.platform(), "python": sys.version, "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
            "gpu": gpu_query("index,name,memory.total,driver_version").splitlines(),
            "gpu_fields": ["index", "name", "total_vram_mib", "driver"]}


def measure(stage):
    query = gpu_query("index,utilization.gpu,memory.used")
    samples = []
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().used
    now = datetime.now(timezone.utc).isoformat()
    for row in csv.reader(query.splitlines(), skipinitialspace=True):
        index, gpu, vram = row
        samples.append({"utc": now, "stage": stage, "gpu_index": index,
                        "gpu_percent": float(gpu), "vram_mib": float(vram), "cpu_percent": cpu, "ram_bytes": ram})
    return samples


@contextmanager
def gpu_lock(directory):
    with (directory / ".gpu.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another Qwen launch owns the GPU lock in {directory}") from error
        yield


def checked_resume(output, config, worker):
    if config.get("mode") != "full" or not config.get("identity"):
        raise ValueError("Resume needs a full run with saved checkpoint identities")
    checkpoint = worker.resume_checkpoint(output, config)
    if checkpoint is None:
        raise ValueError("There is no completed checkpoint in this run directory")
    state = worker.read_json(checkpoint / "trainer_state.json")
    epoch = float(state["epoch"])
    if not epoch.is_integer() or not (checkpoint / "sampler_rng.pth").is_file():
        raise ValueError("Resume needs a complete epoch checkpoint with the saved sampler state")
    return checkpoint


def trained_adapter(output, config, worker):
    record = worker.read_json(output / "train_result.json")
    if (record.get("configuration_identity") != config.get("identity")
            or worker.adapter_record(output / "best_lora", config) != record["adapter"]):
        raise ValueError("Saved adapter or run identity differs from the training result")
    return record


def run_logged(output, config, stages_to_run, resume=None):
    stages, samples, errors = {}, [], []
    status = "failed"
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    attempt = Path(tempfile.mkdtemp(prefix=stamp, dir=attempts))
    write_json(output / "latest_attempt.json", {"path": str(attempt), "status": "running"})
    with (attempt / "utilization.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["utc", "stage", "gpu_index", "gpu_percent", "vram_mib", "cpu_percent", "ram_bytes"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        def sample(stage):
            try:
                readings = measure(stage)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append(f"{stage}: {error}")
            else:
                samples.extend(readings)
                writer.writerows(readings)
                stream.flush()

        try:
            write_json(attempt / "system_specs.json", system_specs())
            psutil.cpu_percent()
            for stage in stages_to_run:
                print(f"Starting {stage}. Log: {attempt / (stage + '.log')}", flush=True)
                started = time.monotonic()
                with (attempt / f"{stage}.log").open("x", encoding="utf-8") as log:
                    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                               "--output", str(output), "--stage", stage]
                    if stage == "train" and resume:
                        command += ["--resume-from", str(resume)]
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, cwd=output)
                    try:
                        next_update = started + 30
                        while True:
                            sample(stage)
                            now = time.monotonic()
                            if now >= next_update:
                                print(f"{stage}: {now - started:.0f} s elapsed; utilization logged in {attempt.name}.", flush=True)
                                next_update = now + 30
                            try:
                                code = process.wait(timeout=5)
                                break
                            except subprocess.TimeoutExpired:
                                continue
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait()
                    stages[stage] = {"exit_code": code, "seconds": round(time.monotonic() - started, 2)}
                if code != 0:
                    raise RuntimeError(f"{stage} exited with code {code}. See {attempt / (stage + '.log')}")
                if stage == "score":
                    predictions = "aptos_test.predictions.csv" if config.get("mode") == "full" else "pilot_validation.predictions.csv"
                    for name in (predictions, "score_result.json"):
                        shutil.copyfile(output / name, attempt / name)
            if any(not any(row["stage"] == stage for row in samples) for stage in stages):
                raise RuntimeError("Utilization sampling did not cover both stages")
            status = "passed"
        except BaseException as error:
            errors.append(str(error) or type(error).__name__)
            raise
        finally:
            scope = ", ".join(f"{count} {split}" for split, count in config["rows"].items())
            lines = ["# Qwen run report", "", f"Status: **{status}**.", "",
                     f"{config['purpose']}. APTOS photos: {scope}.",
                     f"Rank {config['rank']}, CLAHE {'on' if config['clahe'] else 'off'}, "
                     f"trainer seed {config['training']['trainer_seed']}, adapter seed {config['adapter_seed']}.",
                     f"Attempt files: {attempt}", ""]
            if resume:
                lines.append(f"Resumed checkpoint: {resume}")
            for stage, record in stages.items():
                lines.append(f"- {stage}: exit {record['exit_code']}, {record['seconds']} s")
                rows = [row for row in samples if row["stage"] == stage]
                if rows:
                    lines.append(f"- {stage}: {len(rows)} samples, mean GPU {mean(row['gpu_percent'] for row in rows):.1f}%, "
                                 f"mean CPU {mean(row['cpu_percent'] for row in rows):.1f}%, "
                                 f"peak device VRAM {max(row['vram_mib'] for row in rows):.0f} MiB, "
                                 f"peak system RAM {max(row['ram_bytes'] for row in rows) / 2**30:.2f} GiB")
            lines.extend(["", "Device and host samples are in utilization.csv. Framework peaks are in the stage result JSONs.", ""])
            score_result = attempt / "score_result.json"
            if score_result.is_file():
                result = json.loads(score_result.read_text(encoding="utf-8"))
                if "metrics" in result:
                    metrics = result["metrics"]
                    lines.append(f"APTOS test: {metrics['accuracy']:.2%} accuracy, {metrics['qwk']:.4f} QWK, "
                                 f"{metrics['macro_ovr_auc']:.4f} macro one-vs-rest ROC AUC.")
                else:
                    lines.append(f"Validation photos scored after reload: {result['validation_photos_scored']}.")
            lines.extend(f"- Issue: {error}" for error in errors)
            report = "\n".join(lines) + "\n"
            (attempt / "run_report.md").write_text(report, encoding="utf-8")
            (output / "run_report.md").write_text(report, encoding="utf-8")
            write_json(output / "latest_attempt.json", {"path": str(attempt), "status": status})
            print(f"Report: {attempt / 'run_report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--trainer-seed", type=int, default=43)
    parser.add_argument("--adapter-seed", type=int, default=3408)
    parser.add_argument("--candidate", type=Path, help="Manifest for the checked APTOS candidate")
    parser.add_argument("--base-snapshot", type=Path, help="Local copy of the pinned base model")
    parser.add_argument("--historical-split", action="store_true", help="Train on the original split and score the common test IDs")
    parser.add_argument("--comparison-manifest", type=Path, help="Duplicate-excluded manifest supplying the common test set")
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--clahe", choices=("on", "off"), default="on")
    parser.add_argument("--full", action="store_true", help="Train one complete condition and score its APTOS test set")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--resume", action="store_true", help="Resume an interrupted full run or finish its scoring")
    action.add_argument("--score-only", action="store_true", help="Score the verified adapter in an existing run")
    parser.add_argument("--stage", choices=("train", "score"), help=argparse.SUPPRESS)
    parser.add_argument("--resume-from", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage:
        run_model_stage(output, args.stage, args.resume_from)
        return
    if args.resume or args.score_only:
        config = json.loads((output / "pilot_configuration.json").read_text(encoding="utf-8"))
        if config.get("identity") and identity(config) != config["identity"]:
            raise ValueError("Run configuration changed")
        worker = load_worker(Path(config["worker"]))
        if config.get("base_files"):
            verify_base(Path(config["base"]), config["base_files"], worker)
        with gpu_lock(output.parent):
            if args.score_only or (output / "train_result.json").is_file():
                trained_adapter(output, config, worker)
                run_logged(output, config, ("score",))
            else:
                resume = checked_resume(output, config, worker)
                run_logged(output, config, ("train", "score"), resume)
        return
    if args.project is None or not 1 <= args.steps <= 10:
        parser.error("--project is required and --steps must be between 1 and 10")
    if not args.candidate and (args.full or args.clahe != "on"):
        parser.error("Full runs and CLAHE-off require --candidate")
    if args.historical_split and not (args.full and args.comparison_manifest):
        parser.error("Historical split runs require --full and --comparison-manifest")
    config = prepare(args, output)
    print(f"Prepared {config['rows']} APTOS images in {output}", flush=True)
    if not args.prepare_only:
        with gpu_lock(output.parent):
            run_logged(output, config, ("train", "score"))


if __name__ == "__main__":
    main()
