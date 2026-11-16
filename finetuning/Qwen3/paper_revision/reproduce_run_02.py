"""Notebook support for the original Qwen training and evaluation commands."""

import csv
import json
import os
import re
import runpy
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

# paper_reproduction.py and evaluate_clahe_bf16.py stay in finetuning/Qwen3 (hash-pinned).
REVISION = Path(__file__).resolve().parent
if str(REVISION.parent) not in sys.path:
    sys.path.append(str(REVISION.parent))

import paper_reproduction as original

ROOT = original.ROOT
QWEN = ROOT / "finetuning/Qwen3"
ARTIFACTS = QWEN / "artifacts/paper_reproduction"
REFERENCE = ARTIFACTS / "reference_run_02"
FROZEN = ARTIFACTS / "data/frozen_run_02"
EXPERIMENTS = ROOT.with_name("Afterimage_Experiments")
BASE = EXPERIMENTS / "assets/qwen3-vl-8b-4bit"
RAW = ROOT.with_name("aptos2019-blindness-detection")
PREPROCESS = ROOT / "Qwen3/preprocess_qwen.py"
SCRIPT = QWEN / "paper_reproduction.py"
CLAHE_SCORER = QWEN / "evaluate_clahe_bf16.py"
SPLITS = ("train", "val", "test")
METRICS = ("accuracy", "qwk", "macro_ovr_auc")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return original.sha256_file(path)


def runtime():
    code_dir = str(EXPERIMENTS / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from notebook_runner import run_command
    monitor = runpy.run_path(str(EXPERIMENTS / "code/qwen_pilot.py"))
    return run_command, monitor


def folder(kind, tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag):
        raise ValueError("Use letters, digits, underscores or hyphens for the tag.")
    parent = ARTIFACTS / kind
    target = parent / tag
    if target.resolve().parent != parent.resolve():
        raise ValueError(f"Unexpected output location: {target}")
    return target


def verify_images(records, workers=4):
    def check(record):
        path, digest = record
        if sha(Path(path)) != digest:
            raise ValueError(f"Image bytes changed: {path}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(check, records.items()):
            pass


def check_inputs():
    config = read(FROZEN / "paper_training_configuration.json")
    manifest = original._validate_training_config(config)
    provenance = read(REFERENCE / "provenance.json")
    for name, record in provenance["base_files"].items():
        path = BASE / name
        if path.stat().st_size != record["size"] or sha(path) != record["sha256"]:
            raise ValueError(f"Base snapshot file changed: {path}")
    if sha(REFERENCE / "best_lora/adapter_model.safetensors") != provenance["adapter_sha256"]:
        raise ValueError("Archived adapter weights changed.")
    for name, digest in provenance["reference_files"].items():
        if sha(REFERENCE / name) != digest:
            raise ValueError(f"Archived reference file changed: {name}")
    verify_images(manifest["image_sha256"])
    print(f"Verified archived source, packages, model, adapter, CSVs and {sum(manifest['counts'].values())} images.")
    print("APTOS: 2,929 train, 366 validation, 367 test. Archived Messidor: 1,057.")


def build_clahe(branch, tag, workers=4):
    if branch not in {"on", "off"} or workers < 1:
        raise ValueError("Choose on or off and at least one worker.")
    root = folder("data", tag) / f"clahe_{branch}"
    if root.exists():
        raise FileExistsError(f"Choose a new BUILD_TAG or use the finished build: {root}")
    config = read(FROZEN / "paper_training_configuration.json")
    manifest = original._validate_training_config(config)
    labels = {r["id_code"]: r["diagnosis"] for r in original.read_csv(RAW / "train.csv")}
    table, jobs = {}, []
    for split in SPLITS:
        table[split] = original.read_csv(Path(manifest["csv_paths"][split]))
        for row in table[split]:
            ident = row["id_code"]
            src = RAW / "train_images" / f"{ident}.png"
            if labels.get(ident) != row["label"] or not src.is_file():
                raise ValueError(f"Missing original or mismatched grade: {ident}")
            dst = root / "images" / row["filename"]
            jobs.append({"input_path": str(src), "output_path": str(dst)})
            row.update(file_path=str(dst), image_path=str(dst))
    root.mkdir(parents=True)
    job_file = root / "preprocessing_jobs.csv"
    original.write_csv(job_file, jobs, list(jobs[0]))
    raw_hashes = {job["input_path"]: sha(Path(job["input_path"])) for job in jobs}
    script_hash = sha(PREPROCESS)
    run_command, _ = runtime()
    run_command(PREPROCESS, ["--manifest", job_file, "--clahe", branch, "--workers", workers],
                root / "preprocessing.log", keep_pc_awake=True)
    manifest.update(image_dir=str(root / "images"), image_sha256={}, clahe=branch,
                    preprocessing_sha256=script_hash, raw_image_sha256=raw_hashes, workers=workers)
    for split, rows in table.items():
        path = root / f"{split}.csv"
        original.write_csv(path, rows, list(rows[0]))
        manifest["csv_paths"][split] = str(path)
        manifest["csv_sha256"][split] = sha(path)
        manifest["image_sha256"].update({r["image_path"]: sha(Path(r["image_path"])) for r in rows})
    destination = root / "dataset_manifest.json"
    original.write_json(destination, manifest)
    config.update(dataset_manifest=str(destination), dataset_csv_sha256=manifest["csv_sha256"])
    original.write_json(root / "paper_training_configuration.json", config)
    print(f"CLAHE {branch}: {len(jobs)} APTOS images verified and saved to {root}")


def matched_config(branch, build_tag):
    configs, manifests = {}, {}
    frozen = read(FROZEN / "dataset_manifest.json")
    for mode in ("on", "off"):
        root = folder("data", build_tag) / f"clahe_{mode}"
        path = root / "paper_training_configuration.json"
        configs[mode] = path
        manifest = original._validate_training_config(read(path))
        if manifest["clahe"] != mode:
            raise ValueError("The CLAHE labels do not match the dataset branches.")
        manifests[mode] = manifest
        for split in SPLITS:
            def membership(record):
                return [{k: v for k, v in row.items() if k not in {"file_path", "image_path"}}
                        for row in original.read_csv(Path(record["csv_paths"][split]))]
            if membership(manifest) != membership(frozen):
                raise ValueError(f"{mode} {split} membership differs from the original split.")
        verify_images(manifest["image_sha256"])
    for key in ("raw_image_sha256", "preprocessing_sha256"):
        if manifests["on"][key] != manifests["off"][key]:
            raise ValueError(f"CLAHE builds differ in {key}. Rebuild both with a new tag.")
    return configs[branch]


def run_stage(stage, arguments, run, *, script=None):
    script = SCRIPT if script is None else script
    run_command, monitor = runtime()
    attempt = Path(tempfile.mkdtemp(prefix=f"{stage}-", dir=run / "attempts"))
    temp = attempt / "temp"
    temp.mkdir()
    original.write_json(attempt / "command.json", [sys.executable, "-B", "-u", str(script), stage,
                                                    *map(str, arguments)])
    original.write_json(attempt / "system_specs.json", monitor["system_specs"]())
    stop = threading.Event()
    samples, errors = [], []
    fields = ["utc", "stage", "gpu_index", "gpu_percent", "vram_mib", "cpu_percent", "ram_bytes"]

    def sample(stream):
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        while not stop.is_set():
            try:
                for row in monitor["measure"](stage):
                    writer.writerow(row)
                    samples.append(row)
            except (OSError, ValueError, monitor["subprocess"].SubprocessError) as error:
                errors.append(str(error))
            stream.flush()
            stop.wait(5)

    started = time.monotonic()
    status = "interrupted or failed"
    with (attempt / "utilization.csv").open("w", encoding="utf-8", newline="") as stream:
        thread = threading.Thread(target=sample, args=(stream,))
        thread.start()
        try:
            environment = {key: str(temp) for key in ("TEMP", "TMP", "TMPDIR")}
            environment["PYTHONIOENCODING"] = "utf-8"
            with patch.dict(os.environ, environment):
                run_command(script, [stage, *arguments], attempt / f"{stage}.log", keep_pc_awake=True)
            status = "complete"
        finally:
            stop.set()
            thread.join()
            report = {"stage": stage, "status": status, "elapsed_seconds": time.monotonic() - started,
                      "samples": len(samples), "sampling_errors": sorted(set(errors))}
            if samples:
                report.update(mean_gpu_percent=statistics.mean(row["gpu_percent"] for row in samples),
                              mean_cpu_percent=statistics.mean(row["cpu_percent"] for row in samples),
                              peak_vram_mib=max(row["vram_mib"] for row in samples),
                              peak_ram_bytes=max(row["ram_bytes"] for row in samples))
            original.write_json(attempt / "run_report.json", report)
            print(f"{stage} report: {attempt / 'run_report.json'}", flush=True)


def score(run, adapter, config_path, external, *, clahe=False):
    config = read(config_path)
    manifest = original._validate_training_config(config)
    arguments = ["--adapter", adapter, "--base-snapshot", config["base_snapshot"],
                 "--test-csv", manifest["csv_paths"]["test"],
                 "--output-dir", run / ("evaluation_bf16" if clahe else "evaluation")]
    if external:
        arguments.extend(["--external-csv", manifest["csv_paths"]["messidor_external"]])
    run_stage("evaluate", arguments, run, script=CLAHE_SCORER if clahe else SCRIPT)


def compare_replay(run):
    result = {}
    for cohort in ("aptos_test", "messidor_external"):
        saved = REFERENCE / cohort
        replay = run / "evaluation" / cohort
        old, new = [original.read_csv(p.with_suffix(".predictions.csv")) for p in (saved, replay)]
        identity = lambda rows: [(r["image_id"], r["y_true"]) for r in rows]
        if identity(old) != identity(new):
            raise ValueError(f"Replay cohort differs from archive: {cohort}")
        differences = [{"image_id": a["image_id"], "archived": a["y_pred"], "replay": b["y_pred"]}
                       for a, b in zip(old, new) if a["y_pred"] != b["y_pred"]]
        a, b = [read(p.with_suffix(".summary.json")) for p in (saved, replay)]
        result[cohort] = {"archived": {k: a[k] for k in METRICS}, "replay": {k: b[k] for k in METRICS},
                          "max_score_difference": max(abs(float(x[f"p{i}"]) - float(y[f"p{i}"]))
                                                      for x, y in zip(old, new) for i in range(5)),
                          "grade_mismatches": differences}
        print(f"{cohort}: {len(differences)} grade mismatches out of {len(new)}. Accuracy {b['accuracy']:.2%}.")
    original.write_json(run / "replay_comparison.json", result)
    if any(c["grade_mismatches"] for c in result.values()):
        raise RuntimeError(f"Review the mismatches in {run / 'replay_comparison.json'} before retraining.")


def replay(tag):
    run = folder("training_batches", tag) / "replay_run_02"
    check_inputs()
    _, monitor = runtime()
    with monitor["gpu_lock"](EXPERIMENTS / "runs"):
        (run / "attempts").mkdir(parents=True, exist_ok=True)
        score(run, REFERENCE / "best_lora", FROZEN / "paper_training_configuration.json", external=True)
        compare_replay(run)


def train_repeat(tag, index, *, branch=None, build_tag=None, reviewed_replay=False):
    if index not in (1, 2, 3):
        raise ValueError("Choose repeat 1, 2 or 3.")
    batch = folder("training_batches", tag)
    replay_path = batch / "replay_run_02/replay_comparison.json"
    results = read(replay_path)
    if any(result["grade_mismatches"] for result in results.values()) and not reviewed_replay:
        raise RuntimeError(f"Review replay differences first: {replay_path}")
    historical = branch is None
    config_path = FROZEN / "paper_training_configuration.json" if historical else matched_config(branch, build_tag)
    config = read(config_path)
    manifest = original._validate_training_config(config)
    if historical:
        verify_images(manifest["image_sha256"])
    run = batch / (f"historical_{index:02}" if historical else f"clahe_{branch}_{index:02}")
    _, monitor = runtime()
    with monitor["gpu_lock"](EXPERIMENTS / "runs"):
        (run / "attempts").mkdir(parents=True, exist_ok=True)
        if not original.completed_training(run, config):
            checkpoint_dir = run / "checkpoints"
            candidates = list(checkpoint_dir.glob("checkpoint-*"))
            required = ("trainer_state.json", "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
            def usable(path):
                if not all((path / name).is_file() for name in required):
                    return False
                try:
                    original.validate_adapter(path)
                except (ValueError, FileNotFoundError):
                    return False
                return True
            if candidates and not any(map(usable, candidates)):
                saved_config = run / "configuration.json"
                if not saved_config.is_file() or read(saved_config) != config:
                    raise ValueError("Cannot recover checkpoints for a different or unknown configuration.")
                recovery = Path(tempfile.mkdtemp(prefix="incomplete-", dir=run / "attempts"))
                checkpoint_dir.rename(recovery / "checkpoints")
                print(f"Preserved incomplete checkpoints in {recovery}. Starting from the base.")
            run_stage("train", ["--config", config_path, "--run-dir", run], run)
        arguments = read(run / "resolved_training_arguments.json")
        if arguments["per_device_eval_batch_size"] != 4:
            raise ValueError("The effective validation batch differs from the archived value of 4.")
        score(run, run / "best_lora", config_path, external=historical, clahe=not historical)


def summarize(tag):
    batch = folder("training_batches", tag)
    if not batch.is_dir():
        print(f"No runs saved for {tag}.")
        return
    values, groups, paired, external = {}, {}, {}, {}
    for kind in ("historical", "clahe_on", "clahe_off"):
        for i in range(1, 4):
            evaluation = "evaluation" if kind == "historical" else "evaluation_bf16"
            path = batch / f"{kind}_{i:02}" / evaluation / "aptos_test.summary.json"
            if path.is_file():
                result = read(path)
                values[kind, i] = result
                print(kind, i, {**{k: round(result[k], 5) for k in METRICS},
                                "grade_recall": [round(r["recall"], 4) for r in result["per_class"]]})
                messidor = path.with_name("messidor_external.summary.json")
                if kind == "historical" and messidor.is_file():
                    external_result = read(messidor)
                    external[str(i)] = {k: external_result[k] for k in METRICS}
        group = [v for (k, _), v in values.items() if k == kind]
        if group:
            groups[kind] = {"completed": len(group), "metrics": {
                k: {"mean": statistics.mean(r[k] for r in group),
                    "range": [min(r[k] for r in group), max(r[k] for r in group)],
                    "sample_sd": statistics.stdev(r[k] for r in group) if len(group) > 1 else None}
                for k in METRICS}}
            print(kind, f"{len(group)}/3 complete", groups[kind]["metrics"])
    print("Historical reference (three archived runs):")
    historical = original.read_csv(REFERENCE / "aptos_run_comparison.csv")
    for row in historical:
        print(row["run"], {k: row[k] for k in METRICS})
    for i in range(1, 4):
        if ("clahe_off", i) in values and ("clahe_on", i) in values:
            off, on = values["clahe_off", i], values["clahe_on", i]
            paired[str(i)] = {k: round(off[k] - on[k], 5) for k in METRICS}
            print(f"Pair {i}, CLAHE off minus on:", paired[str(i)])
    original.write_json(batch / "results_comparison.json", {
        "aptos_repeats": {f"{kind}_{i:02}": {**{k: result[k] for k in METRICS},
                                            "per_class": result["per_class"]}
                          for (kind, i), result in values.items()},
        "aptos_groups": groups, "clahe_off_minus_on": paired,
        "historical_reference": historical, "messidor_historical_repeats": external,
    })
    print(f"Saved comparison: {batch / 'results_comparison.json'}")
