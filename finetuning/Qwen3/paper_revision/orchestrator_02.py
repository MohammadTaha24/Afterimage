"""Training queue for the learning-rate 2e-4 grid (final_02), used by notebooks/04_training_orchestrator.ipynb.

Phase A: CLAHE on, rank/alpha 8/16, 32/16, 8/8, 32/32. Phase B: CLAHE off, 8/16, 16/16, 32/16,
8/8, 32/32. Seeds 42/3407, 43/3408, 44/3409 as in final_01. Phase C (optional): timing-only
re-scoring of earlier adapters that have no warm per-image timing.

Every GPU job goes through the frozen launchers, imported and never edited: qwen_variant.py
trains (pinned runner, validation-loss checkpoint selection, FP16 score of the 361-image test),
score_cohort.py scores a cohort and times each image, and score_fp32_fallback.py is the labelled
FP32 fallback when the pinned FP16 score overflows. This module only orders the jobs and adds
the logs the paper needs: per attempt command.json, system_specs.json, the stage log,
utilization.csv every 5 s and run_report.json; per queue a GPU power log every 5 s, job
start/end events, an idle-power baseline and energy per job.
"""

import bisect
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import final_runs as final
import remaining_runs as remaining
import reproduce_run_02 as reproduction

REVISION = final.REVISION
FINAL = final.FINAL
RUNS = final.RUNS
SEEDS = final.SEEDS
COHORTS = final.COHORTS
LR = 2e-4
PROTECTED_TAGS = {"final_01"}
# Archived 512-pixel export without CLAHE; the CLAHE export is the one in COHORTS["messidor_archived"].
MESSIDOR_OFF_IMAGES = Path(r"D:\recovered\DR_Aptos_Messidor_Only_Full_Dataset\Res_512\Aptos_messidor_external")
PILOT_LIMIT = 12
PILOT_GENERATION = 6
AUTO_DIAGNOSIS = {
    "layer": "not located (automatic fallback; the pinned FP16 score stopped with a non-finite likelihood)",
    "detail": ("Automatic FP32 fallback queued by orchestrator_02 because the pinned FP16 score stopped with "
               "'Non-finite answer likelihood'. The overflowing layer was not diagnosed."),
    "evidence": [],
}
STATE_LABELS = {"done": "done", "pending": "pending", "waiting": "pending", "failed": "failed",
                "interrupted": "failed", "running": "running", "passed": "failed",
                "fp16_overflow": "done (FP16 failed, FP32 fallback queued)",
                "fp16_nonfinite": "failed (FP16 non-finite, no fallback)"}
# Default minutes when no logged durations exist yet.
DEFAULT_MINUTES = {"train": 90, "messidor": 15, "timing": 16, "aptos_fp32": 7, "messidor_fp32": 30,
                   "pilot_train": 6, "pilot_score": 4}
POWER_FIELDS = ["utc", "gpu_index", "power_draw_w", "power_draw_instant_w", "power_limit_w", "gpu_percent",
                "vram_mib", "temperature_c", "sm_clock_mhz", "pstate"]
ENERGY_FIELDS = ["job", "attempt", "start_utc", "end_utc", "outcome", "wall_s", "samples", "mean_w", "energy_wh",
                 "idle_baseline_w", "energy_above_idle_wh", "max_sample_gap_s", "fully_covered"]
ENERGY_NOTE = ("Whole-card GPU power from nvidia-smi power.draw, sampled every 5 s and integrated with the "
               "trapezoidal rule. It includes the display and any other process on the card; CPU, RAM and "
               "the rest of the PC are not included. energy_above_idle_wh subtracts the idle baseline "
               "measured at the start of the queue session.")


def default_settings():
    return {
        "tag": "final_02", "pilot_tag": "pilot_01", "lr": LR,
        "phases": {"A": True, "B": True, "C": False},
        "seed_pairs": [1, 2, 3],
        "phase_a": [(8, 16), (32, 16), (8, 8), (32, 32)],
        "phase_b": [(8, 16), (16, 16), (32, 16), (8, 8), (32, 32)],
        # Images also answered by greedy generation in the timing job (-1: all), as final_01's benchmark.
        "timing_generation_samples": -1,
        # Messidor for CLAHE-off adapters: "archived_off" (same 1,057 images from the archived export
        # without CLAHE), "archived_on" (the CLAHE export, a preprocessing mismatch) or None (skip).
        "messidor_off": "archived_off",
        "idle_baseline_seconds": 60, "power_interval_seconds": 5, "free_gb_warning": 30,
        "auto_fp32_fallback": True, "fp16_overflow_diagnoses": {},
        "linked_tag": "final_01",
        "phase_c": phase_c_items(),
    }


def phase_c_items():
    """The seven configurations GAPS.md lists without warm per-image timing; seed pair 1 each."""
    def old(condition, split="duplicate_excluded"):
        return RUNS / remaining.qwen_run_name(condition, 1, split, "01")
    linked = FINAL / "final_01" / "train"
    return [
        {"label": "r16_off_lr0p0005_t42_a3407", "run": old("r16_off"), "clahe": "off"},
        {"label": "r8_off_lr0p0005_t42_a3407", "run": old("r8_off"), "clahe": "off"},
        {"label": "r32_off_lr0p0005_t42_a3407", "run": old("r32_off"), "clahe": "off"},
        {"label": "r16_on_lr0p0005_previous_split_t42_a3407", "run": old("r16_on", "previous_split"),
         "clahe": "on", "own_test": True},
        {"label": "r16_on_lr0p0001_t42_a3407", "run": linked / "r16_on_lr0p0001_t42_a3407", "clahe": "on"},
        {"label": "r16_on_lr0p0002_t42_a3407", "run": linked / "r16_on_lr0p0002_t42_a3407", "clahe": "on"},
        {"label": "r16_on_alpha8_t42_a3407", "run": linked / "r16_on_alpha8_t42_a3407", "clahe": "on"},
    ]


def linked_runs(settings):
    """The rank 16, alpha 16, CLAHE-on runs at 2e-4 that final_01 already holds (read only)."""
    where = FINAL / remaining.valid_tag(settings["linked_tag"])
    return [where / "train" / final.variant_name(16, "on", pair, {"lr": settings["lr"]}) for pair in SEEDS]


def root(settings, pilot=False):
    tag = remaining.valid_tag(settings["tag"])
    if tag in PROTECTED_TAGS:
        raise ValueError(f"{tag} holds finished results. Choose a new tag.")
    return FINAL / (f"{tag}__{remaining.valid_tag(settings['pilot_tag'])}" if pilot else tag)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    reproduction.original.write_json(path, value)


def utc_now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- queue


def messidor_cohort(settings, clahe, where):
    """(cohort name, CSV) for archived Messidor matching the adapter's CLAHE branch, or None."""
    if clahe == "on" or settings["messidor_off"] == "archived_on":
        return "messidor_archived", COHORTS["messidor_archived"]
    if settings["messidor_off"] is None:
        return None
    if settings["messidor_off"] != "archived_off":
        raise ValueError("messidor_off must be 'archived_off', 'archived_on' or None.")
    return "messidor_archived_off", where / "cohorts" / "messidor_archived_off.csv"


def build_messidor_off(destination, source=None, images=None):
    """Write the archived Messidor cohort with each image taken from the export without CLAHE.

    Same rows, order, IDs and grades as the CLAHE cohort; only the image paths change. A
    provenance JSON beside the CSV records the source and every image's SHA-256.
    """
    source = Path(source or COHORTS["messidor_archived"])
    images = Path(images or MESSIDOR_OFF_IMAGES)
    destination = Path(destination)
    rows = reproduction.original.read_csv(source)
    for row in rows:
        path = images / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"No CLAHE-off export of {row['filename']} in {images}")
        row["file_path"] = row["image_path"] = str(path)
    if destination.is_file():
        if reproduction.original.read_csv(destination) != rows:
            raise ValueError(f"{destination} differs from a fresh build. Remove it only if no job used it.")
        return destination
    hashes = {row["filename"]: reproduction.sha(Path(row["image_path"])) for row in rows}
    reproduction.original.write_csv(destination, rows, list(rows[0]))
    write_json(destination.with_suffix(".json"), {
        "built_utc": utc_now().isoformat(), "rows": len(rows), "source_csv": str(source),
        "source_csv_sha256": reproduction.sha(source), "image_folder": str(images),
        "csv_sha256": reproduction.sha(destination), "image_sha256": hashes,
        "note": "Same images and grades as the archived CLAHE Messidor cohort, read from the export without CLAHE."})
    return destination


def minutes_for(category, estimates, pilot):
    if pilot:
        return DEFAULT_MINUTES["pilot_train" if category == "train" else "pilot_score"]
    return round(estimates.get(category, DEFAULT_MINUTES[category]))


def score_job(stage, name, cohort, cohort_csv, adapter, *, where, minutes, generation=0, limit=0,
              reference=None, needs=None, base=None, extra=None, note=""):
    output = where / "scores" / name
    arguments = ["--name", name, "--output-dir", output, "--base", base or reproduction.BASE,
                 "--base-precision", "4bit", "--cohort-csv", cohort_csv, "--cohort-name", cohort,
                 "--generation-samples", generation, "--adapter", adapter]
    if reference is not None:
        arguments += ["--reference-predictions", reference]
    if limit:
        arguments += ["--limit", limit]
    return final.Job(stage, name, "score", output, arguments, minutes, needs=needs, note=note,
                     extra={"precision": "FP16", "cohort": cohort, **(extra or {})})


def fallback_job(stage, train, cohort, cohort_csv, *, where, diagnosis, minutes, needs=None, extra=None):
    """Labelled FP32 fallback (score_fp32_fallback.py) for a run whose pinned FP16 score failed."""
    name = final.fallback_name(train.name, cohort)
    output = where / "scores" / name
    arguments = ["--name", name, "--output-dir", output, "--base", reproduction.BASE,
                 "--adapter", train.output / "best_lora", "--train-run", train.output,
                 "--failure-log", final.latest_score_log(train.output), "--overflow-layer", diagnosis["layer"],
                 "--diagnosis-detail", diagnosis.get("detail", ""),
                 "--cohort-csv", cohort_csv, "--cohort-name", cohort]
    for path in diagnosis.get("evidence", []):
        arguments += ["--diagnosis-evidence", path]
    return final.Job(stage, name, "score", output, arguments, minutes, needs=needs,
                     note=f"{final.FALLBACK_LABEL}: FP16 scoring of {train.name} failed",
                     extra={"script": final.FALLBACK_SCRIPT, "fallback_for": train.name,
                            "precision": final.FALLBACK_LABEL, "cohort": cohort, **(extra or {})})


def run_jobs(settings, where, phase, rank, alpha, clahe, pair, estimates, pilot=False):
    """One training run and the jobs that follow it: Messidor, then timing (or the FP32 fallbacks)."""
    stage = f"phase_{phase}"
    train = final.train_job(stage, rank, clahe, pair, {"lr": settings["lr"], "lora_alpha": alpha},
                            where=where, pilot=pilot)
    train.minutes = minutes_for("train", estimates, pilot)
    train.extra.update(phase=phase, alpha=alpha, role="train")
    base = {"phase": phase, "rank": rank, "alpha": alpha, "clahe": clahe, "seed_pair": pair, "train": train.name}
    messidor = messidor_cohort(settings, clahe, where)
    aptos = f"aptos361_{clahe}"
    jobs = [train]
    diagnosis = settings["fp16_overflow_diagnoses"].get(train.name)
    if not pilot and diagnosis is None and settings["auto_fp32_fallback"] and final.fp16_score_failed(train.output):
        diagnosis = dict(AUTO_DIAGNOSIS, automatic=True)
    if diagnosis and not pilot:
        train.extra["fp16_overflow"] = diagnosis
    if train.status() == "fp16_overflow":
        first = fallback_job(stage, train, aptos, train.output / "aptos_test.csv", where=where, diagnosis=diagnosis,
                             minutes=minutes_for("aptos_fp32", estimates, pilot), extra={**base, "role": "aptos_fp32"})
        train.note = f"{final.STATUS_LABELS['fp16_overflow']}; {final.FALLBACK_LABEL}: {first.name}"
        jobs.append(first)
        if messidor:
            jobs.append(fallback_job(stage, train, *messidor, where=where, diagnosis=diagnosis,
                                     minutes=minutes_for("messidor_fp32", estimates, pilot),
                                     needs=first.output / "summary.json", extra={**base, "role": "messidor_fp32"}))
        return jobs
    needs = train.output / "score_result.json"
    adapter = train.output / "best_lora"
    limit = PILOT_LIMIT if pilot else 0
    if messidor:
        jobs.append(score_job(stage, f"{train.name}__{messidor[0]}", *messidor, adapter, where=where,
                              minutes=minutes_for("messidor", estimates, pilot), limit=limit, needs=needs,
                              extra={**base, "role": "messidor"}, note="runs after its training job"))
    jobs.append(score_job(stage, f"{train.name}__{aptos}", aptos, COHORTS[aptos], adapter, where=where,
                          minutes=minutes_for("timing", estimates, pilot), limit=limit, needs=needs,
                          generation=PILOT_GENERATION if pilot else settings["timing_generation_samples"],
                          reference=None if pilot else train.output / "aptos_test.predictions.csv",
                          extra={**base, "role": "timing"}, note="warm per-image timing on the 361 test images"))
    return jobs


def phase_c_job(settings, where, item, estimates, pilot=False):
    run = Path(item["run"])
    if item.get("own_test"):
        cohort, cohort_csv = f"aptos361_previous_split_{item['clahe']}", run / "aptos_test.csv"
    else:
        cohort, cohort_csv = f"aptos361_{item['clahe']}", COHORTS[f"aptos361_{item['clahe']}"]
    config = run / "pilot_configuration.json"
    base = Path(read_json(config)["base"]) if config.is_file() else reproduction.BASE
    return score_job("phase_C", f"timing_{item['label']}__{cohort}", cohort, cohort_csv, run / "best_lora",
                     where=where, minutes=minutes_for("timing", estimates, pilot), base=base,
                     generation=PILOT_GENERATION if pilot else settings["timing_generation_samples"],
                     limit=PILOT_LIMIT if pilot else 0,
                     reference=None if pilot else run / "aptos_test.predictions.csv",
                     needs=run / "best_lora" / "adapter_config.json",
                     extra={"phase": "C", "role": "timing", "run": str(run), "label": item["label"],
                            "clahe": item["clahe"]},
                     note="timing only, no training")


def planned_runs(settings, pilot=False):
    """(phase, rank, alpha, clahe, seed_pair) in queue order: phase, then config, then seed."""
    phases = settings["phases"]
    configs = [("A", r, a, "on") for r, a in settings["phase_a"]] if phases["A"] else []
    configs += [("B", r, a, "off") for r, a in settings["phase_b"]] if phases["B"] else []
    if pilot:  # one variant per phase, first seed only
        first = {}
        for item in configs:
            first.setdefault(item[0], item)
        return [(*item, settings["seed_pairs"][0]) for item in first.values()]
    return [(*item, pair) for item in configs for pair in settings["seed_pairs"]]


def build_queue(settings, pilot=False, estimates=None):
    where = root(settings, pilot)
    estimates = job_estimates(settings) if estimates is None else estimates
    jobs = []
    for phase, rank, alpha, clahe, pair in planned_runs(settings, pilot):
        jobs += run_jobs(settings, where, phase, rank, alpha, clahe, pair, estimates, pilot)
    if settings["phases"]["C"]:
        items = settings["phase_c"][:1] if pilot else settings["phase_c"]
        jobs += [phase_c_job(settings, where, item, estimates, pilot) for item in items]
    return jobs


def category(name):
    if name.startswith(("zeroshot", "bf16_", "original_paper")) or re.match(r"r\d+_on_p\d", name):
        return "benchmark" if name.endswith(("aptos361_on", "aptos361_off")) else "other"
    if name.endswith("_fp32"):
        return "aptos_fp32" if "__aptos" in name else "messidor_fp32"
    if "__messidor_archived" in name:
        return "messidor"
    if "__aptos361" in name:
        return "timing"
    return "train" if "__" not in name else "other"


def job_estimates(settings):
    """Median minutes per kind of job from the logged queue histories (final_01 and this tag)."""
    durations = {}
    sources = [FINAL / remaining.valid_tag(settings["linked_tag"]) / "queue_state.json",
               FINAL / remaining.valid_tag(settings["tag"]) / "queue_state.json"]
    for path in sources:
        if not path.is_file():
            continue
        for record in read_json(path):
            if record.get("outcome") != "passed":
                continue
            minutes = (datetime.fromisoformat(record["ended"]) - datetime.fromisoformat(record["started"])).total_seconds() / 60
            kind = category(record["job"])
            if kind == "benchmark":  # final_01's benchmark is the same job as the timing job here
                kind = "timing"
            durations.setdefault(kind, []).append(minutes)
    return {kind: statistics.median(values) for kind, values in durations.items()}


# ---------------------------------------------------------------- status


def stop_file(settings, pilot=False):
    return root(settings, pilot) / "STOP"


def request_stop(settings, pilot=False):
    path = stop_file(settings, pilot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Stop requested {utc_now().isoformat()}\n", encoding="utf-8")
    print(f"The queue stops after the current job. Delete {path} to cancel.")


def current_job(where):
    """The job a live queue process is running now, or None."""
    marker = where / "queue_current.json"
    if not marker.is_file():
        return None
    try:
        record = read_json(marker)
    except (OSError, ValueError):
        return None
    import psutil
    return record if psutil.pid_exists(record.get("pid", -1)) else None


def history(where):
    path = where / "queue_state.json"
    return read_json(path) if path.is_file() else []


def job_state(job, where, current=None, records=None):
    """done / pending / waiting / failed / interrupted / running / fp16_overflow / fp16_nonfinite."""
    if current and current.get("job") == job.name:
        return "running"
    state = job.status()
    if state in ("pending", "waiting"):
        tries = [record for record in (records or []) if record["job"] == job.name]
        if tries and not tries[-1]["outcome"].startswith("passed"):
            return "failed"
    return state


def show_status(settings, pilot=False):
    """Print the queue with each job's state and time estimate. Starts nothing."""
    where = root(settings, pilot)
    jobs = build_queue(settings, pilot)
    current, records = current_job(where), history(where)
    print(f"{'Pilot' if pilot else 'Queue'} in {where}  (learning rate {settings['lr']:g})")
    if stop_file(settings, pilot).is_file():
        print(f"STOP file present: {stop_file(settings, pilot)} (the next start removes it)")
    counts, left, phase = {}, 0, None
    for job in jobs:
        if job.stage != phase:
            phase = job.stage
            print(f"\n{phase_title(phase.split('_')[1])}")
        state = job_state(job, where, current, records)
        label = STATE_LABELS.get(state, state)
        counts[label.split(" ")[0]] = counts.get(label.split(" ")[0], 0) + 1
        if state not in ("done", "fp16_overflow", "fp16_nonfinite"):
            left += job.minutes
        note = f"  -> {job.note}" if state == "fp16_overflow" or "fallback_for" in job.extra else ""
        print(f"  [{label[:8]:<8}] {job.kind:<5} {job.name}  (~{job.minutes} min){note}")
    if not pilot:
        print(f"\nLinked from {settings['linked_tag']} (r16 alpha16 CLAHE on, not rerun):")
        for run in linked_runs(settings):
            print(f"  [{'done' if (run / 'score_result.json').is_file() else 'missing':<8}] {run}")
    trains = sum(job.kind == "train" for job in jobs)
    print(f"\n{len(jobs)} jobs ({trains} training runs): " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
          + f". Estimated GPU time left: about {left / 60:.1f} h (from logged durations of earlier jobs).")
    print(f"Free space on {where.anchor}: {free_gb(where):.0f} GB.")
    return jobs


def phase_title(phase):
    return {"A": "Phase A: CLAHE on", "B": "Phase B: CLAHE off", "C": "Phase C: timing only"}.get(phase, phase)


def free_gb(path):
    return shutil.disk_usage(Path(path).anchor).free / 1e9


def disk_warning(path, threshold_gb):
    free = free_gb(path)
    if free < threshold_gb:
        return f"Warning: only {free:.0f} GB free on {Path(path).anchor} (below {threshold_gb} GB)."
    return None


# ---------------------------------------------------------------- logging


_STATIC_SPECS = {}


def static_specs():
    """Versions that do not change during a queue session (looked up once)."""
    if _STATIC_SPECS:
        return dict(_STATIC_SPECS)
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ("torch", "torchvision", "transformers", "unsloth", "unsloth_zoo", "peft", "trl", "bitsandbytes",
                 "accelerate", "datasets", "numpy", "pandas", "psutil"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        torch_cuda = subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.version.cuda)"], text=True, timeout=120,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        torch_cuda = None
    try:
        header = subprocess.check_output(["nvidia-smi"], text=True, timeout=10, stderr=subprocess.STDOUT)
        match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", header)
        driver_cuda = match.group(1) if match else None
    except (OSError, subprocess.SubprocessError):
        driver_cuda = None
    cpu = platform.processor()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    _STATIC_SPECS.update(packages=packages, torch_cuda_runtime=torch_cuda, driver_cuda_version=driver_cuda,
                         cpu_name=cpu, python_executable=sys.executable)
    return dict(_STATIC_SPECS)


def system_specs(monitor, where):
    specs = monitor["system_specs"]()
    specs.update(static_specs())
    specs.update(recorded_utc=utc_now().isoformat(), disk_free_bytes=shutil.disk_usage(where.anchor).free,
                 disk=where.anchor)
    return specs


def gpu_power_query(fields):
    return subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                   text=True, timeout=10, stderr=subprocess.STDOUT).strip()


def number_or_none(text):
    try:
        return float(text)
    except ValueError:
        return None


def read_power():
    """One nvidia-smi reading per GPU: whole-card power, limit, utilisation, memory, temperature."""
    fields = "index,power.draw,power.draw.instant,power.limit,utilization.gpu,memory.used,temperature.gpu,clocks.sm,pstate"
    try:
        text, instant = gpu_power_query(fields), True
    except subprocess.CalledProcessError:
        text, instant = gpu_power_query(fields.replace("power.draw.instant,", "")), False
    now = utc_now().isoformat()
    rows = []
    for values in csv.reader(text.splitlines(), skipinitialspace=True):
        if not instant:
            values.insert(2, "")
        index, draw, burst, limit, gpu, vram, temp, clock, pstate = values
        rows.append({"utc": now, "gpu_index": index, "power_draw_w": number_or_none(draw),
                     "power_draw_instant_w": number_or_none(burst), "power_limit_w": number_or_none(limit),
                     "gpu_percent": number_or_none(gpu), "vram_mib": number_or_none(vram),
                     "temperature_c": number_or_none(temp), "sm_clock_mhz": number_or_none(clock), "pstate": pstate})
    return rows


class PowerLogger:
    """Append a whole-card power reading to power_log.csv every interval seconds, in a thread."""

    def __init__(self, path, interval=5, reader=read_power):
        self.path, self.interval, self.reader = Path(path), interval, reader
        self.stop_event, self.thread, self.errors, self.rows = threading.Event(), None, [], 0

    def _loop(self):
        new = not self.path.is_file()
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, POWER_FIELDS, lineterminator="\n")
            if new:
                writer.writeheader()
            while True:
                stopping = self.stop_event.is_set()  # one last reading after the stop, so the last job is covered
                try:
                    readings = self.reader()
                    writer.writerows(readings)
                    self.rows += len(readings)
                    stream.flush()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    self.errors.append(str(error))
                if stopping:
                    break
                self.stop_event.wait(self.interval)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.stop_event.is_set():
            return
        time.sleep(1)  # let the end event fall strictly before the final reading
        self.stop_event.set()
        self.thread.join(timeout=30)


def power_event(where, job, event, attempt=""):
    path = where / "power_events.csv"
    new = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        if new:
            writer.writerow(["utc", "job", "event", "attempt"])
        writer.writerow([utc_now().isoformat(), job, event, attempt])


def idle_baseline(where, monitor, seconds, reader=read_power):
    """Measure whole-card power for `seconds` with the GPU lock held and nothing queued (1 Hz)."""
    samples, apps = [], None
    with monitor["gpu_lock"](RUNS):
        try:
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                text=True, timeout=10).strip().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            apps = [f"query failed: {error}"]
        power_event(where, "idle_baseline", "start")
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            try:
                samples += reader()
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            time.sleep(1)
        power_event(where, "idle_baseline", "end")
    watts = [row["power_draw_w"] for row in samples if row["power_draw_w"] is not None]
    utilisation = [row["gpu_percent"] for row in samples if row["gpu_percent"] is not None]
    record = {"start_utc": samples[0]["utc"] if samples else None, "end_utc": utc_now().isoformat(),
              "seconds": seconds, "samples": len(watts),
              "mean_w": statistics.fmean(watts) if watts else None,
              "median_w": statistics.median(watts) if watts else None,
              "min_w": min(watts, default=None), "max_w": max(watts, default=None),
              "mean_gpu_percent": statistics.fmean(utilisation) if utilisation else None,
              "max_vram_mib": max((row["vram_mib"] for row in samples if row["vram_mib"] is not None), default=None),
              "compute_apps": apps, "note": "Whole-card idle power with the GPU lock held and no job running; "
                                            "the display and desktop apps still draw from the card."}
    path = where / "idle_baseline.json"
    records = read_json(path) if path.is_file() else []
    records.append(record)
    write_json(path, records)
    if record["mean_gpu_percent"] is not None and record["mean_gpu_percent"] > 10:
        print(f"Warning: the GPU was {record['mean_gpu_percent']:.0f}% busy during the idle baseline.", flush=True)
    print(f"Idle baseline: {record['mean_w'] or float('nan'):.1f} W mean over {len(watts)} samples.", flush=True)
    return record


def parse_utc(text):
    when = datetime.fromisoformat(text)
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def energy_per_job(where):
    """Integrate power_log.csv over each job attempt window in power_events.csv; write energy_per_job.csv."""
    log, events = where / "power_log.csv", where / "power_events.csv"
    if not log.is_file() or not events.is_file():
        return []
    samples = []
    for row in reproduction.original.read_csv(log):
        watts = number_or_none(row.get("power_draw_w") or "")
        if watts is not None:
            samples.append((parse_utc(row["utc"]), watts))
    samples.sort()
    times = [t for t, _ in samples]
    baselines = []
    path = where / "idle_baseline.json"
    if path.is_file():
        baselines = sorted((parse_utc(item["end_utc"]), item["mean_w"]) for item in read_json(path)
                           if item.get("mean_w") is not None)

    def power_at(moment):
        i = bisect.bisect_left(times, moment)
        if i < len(times) and times[i] == moment:
            return samples[i][1]
        if i == 0 or i == len(times):
            return None
        (t0, p0), (t1, p1) = samples[i - 1], samples[i]
        return p0 + (p1 - p0) * (moment - t0).total_seconds() / (t1 - t0).total_seconds()

    rows, opened = [], {}
    for event in reproduction.original.read_csv(events):
        key = (event["job"], event["attempt"])
        if event["event"] == "start":
            opened[key] = parse_utc(event["utc"])
            continue
        if key not in opened:
            continue
        start, end = opened.pop(key), parse_utc(event["utc"])
        inside = samples[bisect.bisect_right(times, start):bisect.bisect_left(times, end)]
        points = [(t, p) for t, p in [(start, power_at(start)), *inside, (end, power_at(end))] if p is not None]
        joules = sum((t1 - t0).total_seconds() * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(points, points[1:]))
        seconds = (end - start).total_seconds()
        gaps = [(t1 - t0).total_seconds() for (t0, _), (t1, _) in zip(points, points[1:])]
        idle = next((w for t, w in reversed(baselines) if t <= start), baselines[0][1] if baselines else None)
        covered = bool(points) and points[0][0] == start and points[-1][0] == end
        rows.append({"job": event["job"], "attempt": event["attempt"], "start_utc": start.isoformat(),
                     "end_utc": end.isoformat(), "outcome": event["event"], "wall_s": round(seconds, 1),
                     "samples": len(inside), "mean_w": round(joules / seconds, 1) if seconds and covered else "",
                     "energy_wh": round(joules / 3600, 3) if covered else "",
                     "idle_baseline_w": round(idle, 1) if idle is not None else "",
                     "energy_above_idle_wh": round((joules - idle * seconds) / 3600, 3) if covered and idle is not None else "",
                     "max_sample_gap_s": round(max(gaps), 1) if gaps else "", "fully_covered": covered})
    for (job, attempt), start in opened.items():
        rows.append({"job": job, "attempt": attempt, "start_utc": start.isoformat(), "outcome": "no end event"})
    with (where / "energy_per_job.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ENERGY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (where / "energy_note.txt").write_text(ENERGY_NOTE + "\n", encoding="utf-8")
    return rows


def training_numbers(run):
    """Wall time, images per second and peaks of a finished training run (None if not trained)."""
    run = Path(run)
    if not (run / "train_result.json").is_file():
        return None
    result = read_json(run / "train_result.json")
    arguments = read_json(run / "resolved_training_arguments.json")
    per_step = arguments["per_device_train_batch_size"] * arguments["gradient_accumulation_steps"]
    images = result["global_step"] * per_step
    stage_seconds = {"train": 0.0, "score": 0.0}
    for report in sorted((run / "attempts").glob("*/run_report.md")):
        for stage, seconds in re.findall(r"^- (train|score): exit 0, ([\d.]+) s$", report.read_text(encoding="utf-8"), re.M):
            stage_seconds[stage] += float(seconds)
    summary = next((entry for entry in reversed(result["history"]) if "train_runtime" in entry), {})
    epochs = max((entry.get("epoch", 0) for entry in result["history"]), default=None)
    runtime = summary.get("train_runtime")
    return {"global_step": result["global_step"], "epochs_run": epochs, "best_checkpoint": result["checkpoint"],
            "best_validation_loss": result["validation_loss"], "images_processed": images,
            "train_stage_seconds": stage_seconds["train"] or None, "score_stage_seconds": stage_seconds["score"] or None,
            "trainer_runtime_seconds": runtime,
            # Trainer runtime covers optimisation and the per-epoch validation, not model loading. A resumed
            # run's last trainer call covers only the resumed part, so the stage total is used then.
            "train_images_per_second": (images / runtime if runtime and stage_seconds["train"] and
                                        len(list((run / "attempts").glob("*/train.log"))) == 1
                                        else images / stage_seconds["train"] if stage_seconds["train"] else None),
            "hf_train_samples_per_second_inflated": summary.get("train_samples_per_second"),
            "train_torch_peak_allocated_bytes": result.get("torch_peak_allocated_bytes"),
            "train_torch_peak_reserved_bytes": result.get("torch_peak_reserved_bytes"),
            "cuda_runtime": result.get("cuda_runtime")}


def run_logged(job, where, settings, monitor, launch):
    """Run one job in a new attempt folder with command, specs, stage log, utilisation and report."""
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ-")
    attempts = where / "logs" / job.name / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=stamp, dir=attempts))
    temp = attempt / "temp"
    temp.mkdir()
    stage = job.kind
    if job.kind == "train":
        script, arguments = REVISION / "qwen_variant.py", list(job.arguments)
    else:
        script, arguments = REVISION / job.extra.get("script", "score_cohort.py"), ["score", *job.arguments]
    write_json(attempt / "command.json", {
        "argv": [sys.executable, "-B", "-u", str(script), *map(str, arguments)], "cwd": str(script.parent),
        "environment": {"PYTHONIOENCODING": "utf-8", "TEMP": str(temp), "TMP": str(temp), "TMPDIR": str(temp)},
        "job": job.name, "phase": job.stage, "kind": job.kind, "role": job.extra.get("role"),
        "precision": job.extra.get("precision", "FP16" if job.kind == "train" else None),
        "gpu_lock": str(RUNS / ".gpu.lock")})
    write_json(attempt / "system_specs.json", system_specs(monitor, where))
    previous_inner = read_json(job.output / "latest_attempt.json")["path"] if (job.output / "latest_attempt.json").is_file() else None
    warning = disk_warning(where, settings["free_gb_warning"]) if job.kind == "train" else None
    if warning:
        print(warning, flush=True)
    stop, samples, errors = threading.Event(), [], []
    fields = ["utc", "stage", "gpu_index", "gpu_percent", "vram_mib", "cpu_percent", "ram_bytes"]

    def sample(stream):
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        while not stop.is_set():
            try:
                for row in monitor["measure"](stage):
                    writer.writerow(row)
                    samples.append(row)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append(str(error))
            stream.flush()
            stop.wait(5)

    started_utc, started = utc_now(), time.monotonic()
    status, failure = "interrupted or failed", None
    power_event(where, job.name, "start", attempt.name)
    with (attempt / "utilization.csv").open("w", encoding="utf-8", newline="") as stream:
        thread = threading.Thread(target=sample, args=(stream,), daemon=True)
        thread.start()
        try:
            environment = {key: str(temp) for key in ("TEMP", "TMP", "TMPDIR")}
            environment["PYTHONIOENCODING"] = "utf-8"
            with patch.dict(os.environ, environment):
                if job.kind == "train":  # qwen_variant takes the shared GPU lock itself
                    launch(script, arguments, attempt / "launch.log", keep_pc_awake=True)
                else:
                    with monitor["gpu_lock"](RUNS):
                        launch(script, arguments, attempt / f"{stage}.log", keep_pc_awake=True)
            status = "complete"
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            stop.set()
            thread.join(timeout=30)
            power_event(where, job.name, "end" if status == "complete" else "end_error", attempt.name)
            report = {"job": job.name, "phase": job.stage, "kind": job.kind, "role": job.extra.get("role"),
                      "status": status, "error": failure, "started_utc": started_utc.isoformat(),
                      "ended_utc": utc_now().isoformat(), "wall_seconds": round(time.monotonic() - started, 2),
                      "samples": len(samples), "sampling_errors": sorted(set(errors)), "disk_warning": warning,
                      "disk_free_gb_at_end": round(free_gb(where), 1), "output": str(job.output)}
            if samples:
                report.update(mean_gpu_percent=statistics.fmean(row["gpu_percent"] for row in samples),
                              mean_cpu_percent=statistics.fmean(row["cpu_percent"] for row in samples),
                              peak_vram_mib=max(row["vram_mib"] for row in samples),
                              peak_ram_bytes=max(row["ram_bytes"] for row in samples))
            try:
                report.update(attempt_results(job, attempt, previous_inner))
            except Exception as error:  # the report must still be written
                report["result_read_error"] = str(error)
            write_json(attempt / "run_report.json", report)
            print(f"{job.name}: {status}. Report: {attempt / 'run_report.json'}", flush=True)
    return attempt


def attempt_results(job, attempt, previous_inner):
    """Framework peaks, timing and training speed for the run report; copies the inner stage logs."""
    extra = {}
    if job.kind == "train":
        marker = job.output / "latest_attempt.json"
        if marker.is_file():
            inner = Path(read_json(marker)["path"])
            if str(inner) != previous_inner:
                extra["inner_attempt"] = str(inner)
                for name in ("train.log", "score.log", "run_report.md", "system_specs.json", "utilization.csv"):
                    if (inner / name).is_file():
                        shutil.copyfile(inner / name, attempt / (name if name.endswith(".log") else f"inner_{name}"))
        numbers = training_numbers(job.output)
        if numbers:
            extra["training"] = numbers
            extra["torch_peak_allocated_bytes"] = numbers["train_torch_peak_allocated_bytes"]
        if (job.output / "score_result.json").is_file():
            extra["score_torch_peak_allocated_bytes"] = read_json(job.output / "score_result.json").get(
                "torch_peak_allocated_bytes")
        return extra
    summary = job.output / "summary.json"
    if summary.is_file():
        data = read_json(summary)
        extra.update(torch_peak_allocated_bytes=data.get("torch_peak_allocated_bytes"),
                     torch_peak_reserved_bytes=data.get("torch_peak_reserved_bytes"),
                     model_load_seconds=data.get("model_load_seconds"), timing=data.get("full_answer_timing"),
                     generation_timing=(data.get("generation") or {}).get("timing"),
                     scoring_precision=data["identity"].get("scoring_precision"))
    return extra


# ---------------------------------------------------------------- run


def run_queue(settings, pilot=False):
    """Start or resume the queue. Finished jobs are skipped; interrupted training resumes."""
    where = root(settings, pilot)
    where.mkdir(parents=True, exist_ok=True)
    launch, monitor = reproduction.runtime()
    stop = stop_file(settings, pilot)
    if stop.is_file():
        stop.unlink()
        print(f"Removed an old STOP file: {stop}", flush=True)
    if current_job(where):
        print("Another queue process is already running this folder. Nothing started.")
        return {}
    if not final.lock_is_free():
        print(f"Another job holds the GPU lock ({RUNS / '.gpu.lock'}). Nothing started.")
        return {}
    for clahe in sorted({item[3] for item in planned_runs(settings, pilot)}):
        cohort = messidor_cohort(settings, clahe, where)
        if cohort and cohort[0] == "messidor_archived_off":
            build_messidor_off(cohort[1])
    records = history(where)
    outcomes, tried = {}, set()
    power_event(where, "queue", "session_start", f"pid {os.getpid()}")
    power = PowerLogger(where / "power_log.csv", settings["power_interval_seconds"])
    with power:
        try:
            idle_baseline(where, monitor, settings["idle_baseline_seconds"])
            while True:
                if stop.is_file():
                    print(f"STOP file found ({stop}). Stopping before the next job.", flush=True)
                    outcomes["(queue)"] = "stopped by STOP file"
                    break
                jobs = build_queue(settings, pilot)
                job = next((job for job in jobs if job.name not in tried), None)
                if job is None:
                    break
                tried.add(job.name)
                position = f"[{jobs.index(job) + 1}/{len(jobs)}]"
                state = job.status()
                if state == "done":
                    outcomes[job.name] = "done earlier"
                    continue
                if state in final.STATUS_LABELS:
                    outcomes[job.name] = job.note or final.STATUS_LABELS[state]
                    print(f"{position} {job.name}: {outcomes[job.name]}", flush=True)
                    continue
                if state == "waiting":
                    outcomes[job.name] = f"skipped: needs {job.needs}"
                    print(f"{position} Skipping {job.name}: {job.needs} is missing.", flush=True)
                    continue
                if not final.lock_is_free():
                    outcomes[job.name] = "not started: GPU lock is held by another job"
                    print("Another job holds the GPU lock. Stopping the queue here.", flush=True)
                    break
                print(f"\n{position} {phase_title(job.stage.split('_')[1])}: {job.name}", flush=True)
                record = {"job": job.name, "stage": job.stage, "started": utc_now().isoformat()}
                write_json(where / "queue_current.json", {"job": job.name, "pid": os.getpid(),
                                                          "started_utc": record["started"]})
                try:
                    attempt = run_logged(job, where, settings, monitor, launch)
                    record["attempt"] = str(attempt)
                    outcome = "passed" if job.status() in ("done", "fp16_overflow") else "finished without a completion marker"
                except KeyboardInterrupt:
                    record.update(ended=utc_now().isoformat(), outcome="interrupted by user")
                    records.append(record)
                    write_json(where / "queue_state.json", records)
                    raise
                except Exception as error:  # keep the queue moving
                    outcome = f"failed: {error}"
                    if job.kind == "train" and final.fp16_score_failed(job.output):
                        outcome = "trained; FP16 score non-finite, FP32 fallback follows"
                    print(traceback.format_exc(), flush=True)
                finally:
                    (where / "queue_current.json").unlink(missing_ok=True)
                record.update(ended=utc_now().isoformat(), outcome=outcome)
                records.append(record)
                write_json(where / "queue_state.json", records)
                outcomes[job.name] = outcome
        finally:
            power_event(where, "queue", "session_end", f"pid {os.getpid()}")
            power.__exit__(None, None, None)
            energy_per_job(where)  # also after an interruption
    if power.errors:
        print(f"Power sampling errors: {sorted(set(power.errors))[:3]}", flush=True)
    print("\nQueue finished." if "(queue)" not in outcomes else "\nQueue stopped.")
    for name, outcome in outcomes.items():
        print(f"  {outcome:<28} {name}")
    summarize(settings, pilot)
    if pilot:
        check_pilot(settings)
    return outcomes


def run_pilot(settings):
    return run_queue(settings, pilot=True)


ATTEMPT_FILES = ("command.json", "system_specs.json", "utilization.csv", "run_report.json")


def check_attempt(attempt):
    """Missing log files in one attempt folder (empty list when complete)."""
    missing = [name for name in ATTEMPT_FILES if not (attempt / name).is_file()]
    if not any(attempt.glob("*.log")):
        missing.append("stage log")
    utilisation = attempt / "utilization.csv"
    if utilisation.is_file() and len(utilisation.read_text(encoding="utf-8").splitlines()) < 2:
        missing.append("utilization rows")
    if (attempt / "system_specs.json").is_file():
        specs = read_json(attempt / "system_specs.json")
        for key in ("torch", "transformers", "unsloth", "peft"):
            if not specs.get("packages", {}).get(key):
                missing.append(f"{key} version")
        if not specs.get("gpu") or not specs.get("torch_cuda_runtime"):
            missing.append("GPU/CUDA specs")
    return missing


def check_pilot(settings):
    """Print 'Pilot passed' only when every pilot job finished and left every required log."""
    where = root(settings, pilot=True)
    problems = []
    for job in build_queue(settings, pilot=True):
        if job.status() != "done":
            problems.append(f"{job.name}: {job.status()}")
            continue
        attempts = sorted((where / "logs" / job.name / "attempts").glob("*"))
        if not attempts:
            problems.append(f"{job.name}: no attempt folder")
            continue
        missing = check_attempt(attempts[-1])
        if missing:
            problems.append(f"{job.name}: missing {', '.join(missing)}")
        if job.kind == "score" and not (job.output / "predictions.csv").is_file():
            problems.append(f"{job.name}: no predictions.csv")
        if not (where / "logs" / job.name / "metrics.json").is_file():
            problems.append(f"{job.name}: no metrics.json")
    for name in ("power_log.csv", "power_events.csv", "energy_per_job.csv", "idle_baseline.json"):
        path = where / name
        if not path.is_file() or (name.endswith(".csv") and len(path.read_text(encoding="utf-8").splitlines()) < 2):
            problems.append(f"{name} missing or empty")
    if problems:
        print(f"Pilot incomplete: {len(problems)} problem(s).")
        for problem in problems:
            print("  " + problem)
        return False
    print("Pilot passed. The full queue can be started.")
    return True


# ---------------------------------------------------------------- summary


def scored(metrics):
    counts = [sum(column) for column in zip(*metrics["confusion_matrix"])]
    return {"images": metrics["samples"], "accuracy": metrics["accuracy"], "qwk": metrics["qwk"],
            "macro_f1": metrics["macro_f1"], "macro_auc": metrics["macro_ovr_auc"],
            "collapsed": max(counts) == metrics["samples"], "predicted_per_grade": " ".join(map(str, counts))}


def prediction_metrics(path):
    metrics = reproduction.original.calculate_metrics(reproduction.original.read_csv(path))
    import score_cohort
    for item, value in zip(metrics["per_class"], score_cohort.specificity(metrics["confusion_matrix"])):
        item["sensitivity"], item["specificity"] = item["recall"], value
    return metrics


def energy_by_job(where):
    path = where / "energy_per_job.csv"
    totals = {}
    if path.is_file():
        for row in reproduction.original.read_csv(path):
            if row.get("energy_wh"):
                item = totals.setdefault(row["job"], {"energy_wh": 0.0, "energy_above_idle_wh": 0.0, "attempts": 0})
                item["energy_wh"] += float(row["energy_wh"])
                item["energy_above_idle_wh"] += float(row["energy_above_idle_wh"] or 0)
                item["attempts"] += 1
    return totals


def score_fields(output, prefix):
    summary = read_json(output / "summary.json")
    timing = summary["full_answer_timing"]
    generation = summary.get("generation") or {}
    fields = {f"{prefix}_{key}": value for key, value in scored(summary["metrics"]).items()}
    fields.update({f"{prefix}_scoring_precision": summary["identity"].get("scoring_precision", "FP16"),
                   f"{prefix}_cohort": summary["identity"]["cohort"],
                   f"{prefix}_median_seconds_per_image": timing["median_seconds"],
                   f"{prefix}_p95_seconds_per_image": timing["p95_seconds"],
                   f"{prefix}_images_per_second": timing["images_per_second"],
                   f"{prefix}_model_load_seconds": summary["model_load_seconds"],
                   f"{prefix}_torch_peak_gib": summary["torch_peak_allocated_bytes"] / 2**30,
                   f"{prefix}_predictions": str(output / "predictions.csv")})
    if generation:
        fields.update({f"{prefix}_generation_unparseable": generation["unparseable_outputs"],
                       f"{prefix}_generation_images": generation["images"],
                       f"{prefix}_generation_median_seconds": generation["timing"]["median_seconds"]})
    agreement = summary.get("agreement_with_saved_predictions")
    if agreement:
        fields[f"{prefix}_grade_mismatches_vs_training_score"] = agreement["grade_mismatches"]
    return fields, summary


def run_row(run, *, phase, source, rank, alpha, clahe, pair, scores, energy=None):
    """One summary row for a training run: APTOS 361, Messidor, timing, training and energy."""
    trainer, adapter = SEEDS[pair]
    row = {"phase": phase, "source": source, "run": run.name, "rank": rank, "alpha": alpha,
           "clahe": clahe, "lr": None, "seed_pair": pair, "trainer_seed": trainer, "adapter_seed": adapter}
    config = run / "pilot_configuration.json"
    if config.is_file():
        row["lr"] = read_json(config)["training"]["lr"]
    numbers = training_numbers(run)
    if numbers:
        row.update(best_validation_loss=numbers["best_validation_loss"], epochs_run=numbers["epochs_run"],
                   global_step=numbers["global_step"], train_stage_seconds=numbers["train_stage_seconds"],
                   train_images_per_second=numbers["train_images_per_second"],
                   train_torch_peak_gib=(numbers["train_torch_peak_allocated_bytes"] or 0) / 2**30 or None)
    predictions = run / "aptos_test.predictions.csv"
    if (run / "score_result.json").is_file() and predictions.is_file():
        row.update({f"aptos_{key}": value for key, value in scored(prediction_metrics(predictions)).items()},
                   aptos_scoring_precision="FP16", aptos_predictions=str(predictions))
        row["status"] = "scored"
    elif scores.get("aptos_fp32") is not None and (scores["aptos_fp32"] / "summary.json").is_file():
        fields, _ = score_fields(scores["aptos_fp32"], "aptos")
        row.update(fields, status="FP16 scoring failed; FP32 fallback")
    else:
        row["status"] = "trained, not scored" if numbers else "not trained"
    for role, prefix in (("messidor", "messidor"), ("messidor_fp32", "messidor"), ("timing", "timing")):
        output = scores.get(role)
        if output is not None and (output / "summary.json").is_file():
            fields, _ = score_fields(output, prefix)
            if prefix == "timing":
                fields = {key: value for key, value in fields.items()
                          if not key.startswith(("timing_accuracy", "timing_qwk", "timing_macro", "timing_collapsed",
                                                 "timing_predicted", "timing_images"))
                          or key == "timing_images_per_second"}
            row.update(fields)
    if energy is not None:
        row.update(energy)
    return row


def grid_lines(rows):
    """Mean over seeds per (CLAHE, rank, alpha) at the queue's learning rate."""
    groups = {}
    for row in rows:
        if row.get("aptos_accuracy") is not None and row["phase"] in ("A", "B", "linked"):
            groups.setdefault((row["clahe"], row["rank"], row["alpha"]), []).append(row)

    def mean(values, style, spread=False):
        values = [value for value in values if value is not None]
        if not values:
            return ""
        text = format(statistics.fmean(values), style)
        if spread and len(values) > 1:
            text += " ± " + format(statistics.stdev(values), style)
        return text

    lines = ["| CLAHE | Rank | Alpha | Source | Seeds | Scoring | Accuracy % | QWK | Macro-F1 | Macro AUC | Collapsed | "
             "Messidor accuracy % | Messidor QWK | Median s/image | Train h | Energy Wh (all jobs) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (clahe, rank, alpha), members in sorted(groups.items(), key=lambda item: (item[0][0] == "off", item[0][1], item[0][2])):
        fallback = sum(row.get("aptos_scoring_precision") == final.FALLBACK_LABEL for row in members)
        scoring = "FP16" if not fallback else f"FP16 + {fallback} {final.FALLBACK_LABEL}"
        lines.append(
            f"| {clahe} | {rank} | {alpha} | {', '.join(sorted({row['source'] for row in members}))} | {len(members)} | "
            f"{scoring} | {mean([100 * row['aptos_accuracy'] for row in members], '.2f', True)} | "
            f"{mean([row['aptos_qwk'] for row in members], '.4f', True)} | "
            f"{mean([row['aptos_macro_f1'] for row in members], '.4f')} | "
            f"{mean([row['aptos_macro_auc'] for row in members], '.4f')} | "
            f"{sum(bool(row.get('aptos_collapsed')) for row in members)} | "
            f"{mean([100 * row['messidor_accuracy'] for row in members if row.get('messidor_accuracy') is not None], '.2f')} | "
            f"{mean([row.get('messidor_qwk') for row in members], '.4f')} | "
            f"{mean([row.get('timing_median_seconds_per_image') or row.get('aptos_median_seconds_per_image') for row in members], '.3f')} | "
            f"{mean([row['train_stage_seconds'] / 3600 if row.get('train_stage_seconds') else None for row in members], '.2f')} | "
            f"{mean([row.get('energy_wh_total') for row in members], '.0f')} |")
    return lines


def summarize(settings, pilot=False):
    """Write summary.csv and summary.md (per-run rows and the combined 2e-4 grid) plus metrics.json per job."""
    where = root(settings, pilot)
    jobs = build_queue(settings, pilot)
    energy = energy_by_job(where)
    for job in jobs:  # metrics.json beside each job's attempt logs
        target = where / "logs" / job.name / "metrics.json"
        if job.status() != "done":
            continue
        if job.kind == "train" and (job.output / "aptos_test.predictions.csv").is_file():
            write_json(target, {"job": job.name, "cohort": "aptos361_" + job.extra["clahe"], "scoring_precision": "FP16",
                                "predictions": str(job.output / "aptos_test.predictions.csv"),
                                "metrics": prediction_metrics(job.output / "aptos_test.predictions.csv"),
                                "training": training_numbers(job.output)})
        elif job.kind == "train":
            result = read_json(job.output / "score_result.json")
            write_json(target, {"job": job.name, "note": "pilot: validation subset only",
                                "validation_photos_scored": result.get("validation_photos_scored"),
                                "training": training_numbers(job.output)})
        else:
            summary = read_json(job.output / "summary.json")
            write_json(target, {"job": job.name, "cohort": summary["identity"]["cohort"],
                                "scoring_precision": summary["identity"].get("scoring_precision", "FP16"),
                                "predictions": str(job.output / "predictions.csv"), "metrics": summary["metrics"],
                                "full_answer_timing": summary["full_answer_timing"],
                                "generation": summary.get("generation"),
                                "torch_peak_allocated_bytes": summary["torch_peak_allocated_bytes"]})
    rows = []
    trains = [job for job in jobs if job.kind == "train"]
    for train in trains:
        followers = {job.extra["role"]: job.output for job in jobs if job.extra.get("train") == train.name}
        names = [train.name] + [job.name for job in jobs if job.extra.get("train") == train.name]
        spent = {"energy_wh_train": energy.get(train.name, {}).get("energy_wh"),
                 "energy_wh_scores": sum(energy.get(name, {}).get("energy_wh", 0) for name in names[1:]) or None,
                 "energy_wh_total": sum(energy.get(name, {}).get("energy_wh", 0) for name in names) or None,
                 "energy_above_idle_wh_total": sum(energy.get(name, {}).get("energy_above_idle_wh", 0) for name in names) or None}
        rows.append(run_row(train.output, phase=train.extra["phase"], source=where.name, rank=train.extra["rank"],
                            alpha=train.extra["alpha"], clahe=train.extra["clahe"], pair=train.extra["seed_pair"],
                            scores=followers, energy=spent))
    timing_c = {job.extra["label"]: job for job in jobs if job.extra.get("phase") == "C"}
    if not pilot:
        linked = FINAL / settings["linked_tag"]
        for pair, run in zip(SEEDS, linked_runs(settings)):
            timing = timing_c.get(run.name)
            scores = {"messidor": linked / "scores" / f"{run.name}__messidor_archived",
                      "timing": timing.output if timing else None}
            rows.append(run_row(run, phase="linked", source=settings["linked_tag"], rank=16, alpha=16, clahe="on",
                                pair=pair, scores=scores))
    for label, job in timing_c.items():
        if job.status() == "done":
            fields, _ = score_fields(job.output, "timing")
            rows.append({"phase": "C", "source": where.name, "run": label, "clahe": job.extra["clahe"],
                         "status": "timing only", **fields,
                         "energy_wh_total": energy.get(job.name, {}).get("energy_wh")})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    where.mkdir(parents=True, exist_ok=True)
    with (where / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    def cell(value, style):
        return "" if value is None or value == "" else format(value, style)

    lines = [f"# Training queue summary ({where.name})", "",
             f"Learning rate {settings['lr']:g}, 4-bit base, split 2,786/357/361, checkpoint chosen by validation loss. "
             "Accuracy, QWK, macro-F1 and macro AUC come from the pinned five-answer scorer on the 361 clean test "
             "images. Collapsed means every test image got the same grade. Timing excludes model loading and the "
             "first three images. Energy is whole-card GPU power (see energy_note.txt).", "",
             "## Grid at this learning rate", "",
             f"Rank 16, alpha 16, CLAHE on comes from {settings['linked_tag']} (not rerun).", "", *grid_lines(rows), "",
             "## Runs", "",
             "| Phase | Run | Status | Scoring | Accuracy | QWK | Macro-F1 | AUC | Collapsed | Messidor acc | Median s/image | Train h | Train img/s | Energy Wh |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["phase"] == "C":
            continue
        lines.append(f"| {row['phase']} | {row['run']} | {row['status']} | {row.get('aptos_scoring_precision', '')} | "
                     f"{cell(row.get('aptos_accuracy'), '.2%')} | {cell(row.get('aptos_qwk'), '.4f')} | "
                     f"{cell(row.get('aptos_macro_f1'), '.4f')} | {cell(row.get('aptos_macro_auc'), '.4f')} | "
                     f"{'yes' if row.get('aptos_collapsed') else ''} | {cell(row.get('messidor_accuracy'), '.2%')} | "
                     f"{cell(row.get('timing_median_seconds_per_image') or row.get('aptos_median_seconds_per_image'), '.3f')} | "
                     f"{cell(row['train_stage_seconds'] / 3600 if row.get('train_stage_seconds') else None, '.2f')} | "
                     f"{cell(row.get('train_images_per_second'), '.2f')} | {cell(row.get('energy_wh_total'), '.0f')} |")
    phase_c = [row for row in rows if row["phase"] == "C"]
    if phase_c:
        lines += ["", "## Phase C: timing of earlier adapters", "",
                  "| Run | Cohort | Median s/image | p95 s/image | Images/s | Load s | Energy Wh |", "|---|---|---|---|---|---|---|"]
        for row in phase_c:
            lines.append(f"| {row['run']} | {row['timing_cohort']} | {cell(row['timing_median_seconds_per_image'], '.3f')} | "
                         f"{cell(row['timing_p95_seconds_per_image'], '.3f')} | {cell(row['timing_images_per_second'], '.2f')} | "
                         f"{cell(row['timing_model_load_seconds'], '.1f')} | {cell(row.get('energy_wh_total'), '.1f')} |")
    lines += ["", "Messidor for CLAHE-off adapters: "
              + {"archived_off": "the same 1,057 archived images, read from the export without CLAHE "
                                 "(cohorts/messidor_archived_off.csv).",
                 "archived_on": "the archived CLAHE export (preprocessing differs from training).",
                 None: "not scored."}[settings["messidor_off"]]]
    (where / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary of {len(rows)} row(s): {where / 'summary.md'}")
    return rows
