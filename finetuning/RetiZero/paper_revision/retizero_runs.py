"""Queue for the RetiZero reproduction: checks, pilot, replay, training runs, scoring and summary.

Used by notebooks/02_retizero_runs.ipynb. It mirrors finetuning/Qwen3/paper_revision/final_runs.py and
uses the same launcher, so every GPU job keeps a log, the hardware specs and a 5-second utilization
record, and takes the same GPU lock as the Qwen jobs. Results live under
D:/Afterimage_Experiments/retizero_runs/<tag>, apart from the Qwen runs.
"""

import csv
import re
import statistics
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RETIZERO = HERE.parent
for directory in (RETIZERO, RETIZERO.parent / "Qwen3/paper_revision"):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

import retizero_reproduction as helper
import reproduce_run_02 as reproduction  # launcher, utilization monitor and GPU lock shared with Qwen

ROOT = helper.ROOT
EXPERIMENTS = reproduction.EXPERIMENTS
DATA = EXPERIMENTS / "datasets"
LOCK_DIR = EXPERIMENTS / "runs"  # holds the GPU lock shared with the Qwen jobs
RESULTS = EXPERIMENTS / "retizero_runs"
TRAIN = HERE / "train_retizero.py"
SCORE = HERE / "score_retizero.py"
FINE_TUNED_SHA256 = "b254d11db55df03ad96e7538b5a2b410492301301cf3952a7af9c099206871f0"
# stage -> (dataset key, CLAHE branch, test cohort)
TRAIN_STAGES = {"historical": ("original", "off", "aptos367_original"),
                "matched": ("clean", "off", "aptos361_clean"),
                "clahe": ("clean_clahe", "on", "aptos361_clean_clahe")}
# Minutes per job. Scoring is measured; training is an estimate from the pilot's throughput.
MINUTES = {"score": 1, "score_messidor": 2, "train": 12, "pilot_train": 2, "pilot_score": 1}
PILOT_EPOCHS = 2
EXPECTED_PARAMETERS = 304_093_189


def default_settings():
    standard = DATA / "aptos_224_standard"
    return {
        "tag": "retizero_01", "pilot_tag": "pilot_01",
        "fine_tuned": ROOT / "RetiZero/best_retizero_dr.pth", "fine_tuned_sha256": FINE_TUNED_SHA256,
        "pretrained": EXPERIMENTS / "assets/RetiZero.pth", "pretrained_sha256": "",
        "seeds": [42, 43, 44],
        "effective_batch": helper.RECIPE["effective_batch"], "physical_batch": 32, "eval_batch": 64,
        "learning_rate": helper.RECIPE["learning_rate"],
        "epochs": helper.RECIPE["epochs"], "patience": helper.RECIPE["patience"],
        "stages": {"replay": True, "historical": True, "matched": True, "clahe": True,
                   "benchmark": True, "messidor_archived": True},
        "datasets": {"original": standard / "previous_split/manifest.json",
                     "clean": standard / "duplicate_excluded/manifest.json",
                     "clean_clahe": DATA / "aptos_224_clahe/duplicate_excluded/manifest.json"},
        "qwen_datasets": {"original": DATA / "aptos_512/previous_split/manifest.json",
                          "clean": DATA / "aptos_512/duplicate_excluded/manifest.json",
                          "clean_clahe": DATA / "aptos_512/duplicate_excluded/manifest.json"},
        "cohorts": {"aptos367_original": standard / "previous_split/clahe_off/test.csv",
                    "aptos361_clean": standard / "duplicate_excluded/clahe_off/test.csv",
                    "aptos361_clean_clahe": DATA / "aptos_224_clahe/duplicate_excluded/clahe_on/test.csv",
                    "messidor_archived": DATA / "messidor_224_archived/clahe_off/test.csv"},
    }


@dataclass
class Job:
    stage: str
    name: str
    kind: str                      # "score" or "train"
    output: Path
    arguments: list
    minutes: int
    needs: list = field(default_factory=list)  # files that must exist before this job can run
    blocked: str = ""              # a setting that must be filled in before this job can run
    note: str = ""
    extra: dict = field(default_factory=dict)

    def done(self):
        marker = "summary.json" if self.kind == "score" else "evaluation/aptos_test.summary.json"
        return (self.output / marker).is_file()

    def missing(self):
        return ([self.blocked] if self.blocked else []) + [str(path) for path in self.needs if not Path(path).exists()]

    def status(self):
        if self.done():
            return "done"
        if self.missing():
            return "waiting"
        if self.kind == "train" and (self.output / "configuration.json").is_file():
            return "interrupted"
        return "pending"


def valid_tag(tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag):
        raise ValueError("Use a tag with letters, digits, hyphens or underscores.")
    return tag


def root(settings, pilot=False):
    tag = valid_tag(settings["tag"])
    return RESULTS / (f"{tag}__{valid_tag(settings['pilot_tag'])}" if pilot else tag)


def pretrained_ready(settings):
    return Path(settings["pretrained"]).is_file() and len(settings["pretrained_sha256"].strip()) == 64


def score_job(stage, label, cohort, checkpoint, *, where, settings, cohort_csv=None, reference=None,
              limit=0, needs=(), note=""):
    name = f"{label}__{cohort}"
    output = where / "scores" / name
    cohort_csv = Path(cohort_csv or settings["cohorts"][cohort])
    arguments = ["--name", name, "--output-dir", output, "--checkpoint", checkpoint,
                 "--cohort-csv", cohort_csv, "--cohort-name", cohort, "--batch-size", settings["eval_batch"]]
    if reference is not None:
        arguments += ["--reference-predictions", reference]
    if limit:
        arguments += ["--limit", limit]
    minutes = MINUTES["pilot_score" if limit else "score_messidor" if cohort.startswith("messidor") else "score"]
    return Job(stage, name, "score", output, arguments, minutes,
               needs=[checkpoint, cohort_csv, *needs], note=note)


def train_job(stage, seed, dataset, branch, *, where, settings, name=None, pilot=False, stop_after=None):
    name = name or f"{stage}_t{seed}"
    output = where / "train" / name
    manifest = Path(settings["datasets"][dataset])
    arguments = ["--dataset", manifest, "--branch", branch,
                 "--matched-dataset", settings["qwen_datasets"][dataset],
                 "--run-dir", output, "--seed", seed,
                 "--effective-batch", settings["effective_batch"], "--physical-batch", settings["physical_batch"],
                 "--eval-batch", settings["eval_batch"], "--learning-rate", repr(settings["learning_rate"]),
                 "--patience", settings["patience"]]
    needs = [manifest]
    random_init = pilot and not pretrained_ready(settings)
    if random_init:
        arguments += ["--random-init"]
    else:
        arguments += ["--weights", settings["pretrained"], "--weights-sha256", settings["pretrained_sha256"]]
        needs.append(Path(settings["pretrained"]))
    arguments += ["--pilot", "--epochs", PILOT_EPOCHS] if pilot else ["--epochs", settings["epochs"]]
    note = "starts from random weights: the pretrained checkpoint is not here yet" if random_init else ""
    blocked = ""
    if not pilot and not pretrained_ready(settings):
        note = "needs the pretrained checkpoint and its SHA-256"
        if Path(settings["pretrained"]).is_file():
            blocked = "pretrained_sha256 is not set in Setup"
    return Job(stage, name, "train", output, arguments, MINUTES["pilot_train" if pilot else "train"],
               needs=needs, blocked=blocked, note=note,
               extra={"seed": seed, "dataset": dataset, "branch": branch, "stop_after": stop_after,
                      "random_init": random_init})


def selected_checkpoint(run):
    """The selected checkpoint of a training run; a placeholder path until the run has finished."""
    result = run / "train_result.json"
    return run / (helper.read_json(result)["checkpoint"] if result.is_file() else "best_epoch_(pending).pth")


def build_queue(settings, pilot=False):
    where = root(settings, pilot)
    stages = settings["stages"]
    fine_tuned = Path(settings["fine_tuned"])
    seeds = settings["seeds"]
    jobs = []
    if pilot:
        jobs.append(score_job("replay", "saved_checkpoint", "aptos367_original", fine_tuned, where=where,
                              settings=settings, limit=12))
        straight = train_job("pilot_training", seeds[0], "clean", "off", where=where, settings=settings,
                             name=f"pilot_straight_t{seeds[0]}", pilot=True)
        resumed = train_job("pilot_training", seeds[0], "clean", "off", where=where, settings=settings,
                            name=f"pilot_resumed_t{seeds[0]}", pilot=True, stop_after=1)
        jobs += [straight, resumed]
        jobs.append(score_job("benchmark", straight.name, "pilot_test", selected_checkpoint(straight.output),
                              where=where, settings=settings, cohort_csv=straight.output / "evaluation/pilot_test.csv",
                              reference=straight.output / "evaluation/aptos_test.predictions.csv",
                              note="runs after its training job"))
        return jobs

    if stages["replay"]:
        jobs.append(score_job("replay", "saved_checkpoint", "aptos367_original", fine_tuned, where=where,
                              settings=settings, note="compared with the original notebook's report"))
        jobs.append(score_job("replay", "saved_checkpoint", "aptos361_clean", fine_tuned, where=where,
                              settings=settings,
                              note="not leak-free: this checkpoint was trained on the original split"))
    trained = {}
    for stage, (dataset, branch, cohort) in TRAIN_STAGES.items():
        if stages[stage]:
            trained[stage] = [train_job(stage, seed, dataset, branch, where=where, settings=settings)
                              for seed in seeds]
            jobs += trained[stage]
    if stages["benchmark"]:
        for job in trained.get("matched", []):
            jobs.append(score_job("benchmark", job.name, "aptos361_clean", selected_checkpoint(job.output),
                                  where=where, settings=settings,
                                  reference=job.output / "evaluation/aptos_test.predictions.csv",
                                  note="runs after its training job"))
    if stages["messidor_archived"]:
        jobs.append(score_job("messidor_archived", "saved_checkpoint", "messidor_archived", fine_tuned,
                              where=where, settings=settings, note="needs the 224-pixel Messidor build"))
        for job in trained.get("matched", []):
            jobs.append(score_job("messidor_archived", job.name, "messidor_archived",
                                  selected_checkpoint(job.output), where=where, settings=settings,
                                  note="runs after its training job; needs the 224-pixel Messidor build"))
    return jobs


def lock_is_free():
    _, monitor = reproduction.runtime()
    try:
        with monitor["gpu_lock"](LOCK_DIR):
            return True
    except RuntimeError:
        return False


def checks(settings):
    """Read-only check of every input. No GPU."""
    lines = []

    def report(ok, text):
        lines.append((ok, text))
        print(f"  [{'ok' if ok else '--'}] {text}")

    print("Inputs")
    fine_tuned = Path(settings["fine_tuned"])
    if fine_tuned.is_file():
        same = helper.sha(fine_tuned) == settings["fine_tuned_sha256"]
        report(same, f"Fine-tuned checkpoint {fine_tuned} " + ("matches its pinned SHA-256" if same else "has a different SHA-256"))
    else:
        report(False, f"Fine-tuned checkpoint missing: {fine_tuned}")
    pretrained = Path(settings["pretrained"])
    if not pretrained.is_file():
        report(False, f"Pretrained RetiZero.pth missing: {pretrained} (needed for every training run)")
    elif not pretrained_ready(settings):
        report(False, f"Pretrained checkpoint found. Set SETTINGS['pretrained_sha256'] = '{helper.sha(pretrained)}'")
    else:
        same = helper.sha(pretrained) == settings["pretrained_sha256"].strip().lower()
        report(same, "Pretrained checkpoint " + ("matches its pinned SHA-256" if same else "differs from its pinned SHA-256"))
    for key, manifest in settings["datasets"].items():
        manifest = Path(manifest)
        if not manifest.is_file():
            report(False, f"Dataset '{key}' not built: {manifest}")
            continue
        branch = "on" if key.endswith("clahe") else "off"
        _, identity = helper.dataset_tables(manifest, branch, matched_manifest=settings["qwen_datasets"][key],
                                            verify_images=False)
        counts = " / ".join(str(identity["rows"][split]) for split in helper.SPLITS)
        report(True, f"Dataset '{key}': {counts} images, same ordered IDs and grades as the Qwen build")
    for name, path in settings["cohorts"].items():
        present = Path(path).is_file()
        report(present, f"Cohort '{name}': " + (f"{len(helper.read_csv(path))} images" if present else f"not built ({path})"))
    free = lock_is_free()
    report(free, "GPU lock is free" if free else "GPU lock is held by another job")
    return lines


def show_plan(settings, pilot=False):
    jobs = build_queue(settings, pilot)
    print(f"{'Pilot' if pilot else 'RetiZero'} queue in {root(settings, pilot)}")
    left = 0
    for job in jobs:
        state = job.status()
        if state != "done":
            left += job.minutes
        note = f"  [{job.note}]" if job.note and state != "done" else ""
        print(f"  [{state:<11}] {job.stage:<18} {job.name}  (~{job.minutes} min){note}")
    done = sum(job.status() == "done" for job in jobs)
    waiting = sum(job.status() == "waiting" for job in jobs)
    print(f"{done} of {len(jobs)} jobs done, {waiting} waiting for a missing input. "
          f"Estimated GPU time for everything not done: about {left / 60:.1f} h.")
    return jobs


def run_job(job):
    _, monitor = reproduction.runtime()
    (job.output / "attempts").mkdir(parents=True, exist_ok=True)
    with monitor["gpu_lock"](LOCK_DIR):
        if job.kind == "score":
            reproduction.run_stage("score", job.arguments, job.output, script=SCORE)
            return
        fresh = not any((job.output / name).exists() for name in ("checkpoint.pth", "train_result.json"))
        if job.extra["stop_after"] and fresh:  # resume test: stop on purpose, then continue below
            reproduction.run_stage("train", [*job.arguments, "--stop-after-epoch", job.extra["stop_after"]],
                                   job.output, script=TRAIN)
        reproduction.run_stage("train", job.arguments, job.output, script=TRAIN)
        reproduction.run_stage("evaluate", job.arguments, job.output, script=TRAIN)


def run_queue(settings, pilot=False, only=None):
    """Run every unfinished job in order. A failed job is recorded and the queue continues."""
    where = root(settings, pilot)
    where.mkdir(parents=True, exist_ok=True)
    state_path = where / "queue_state.json"
    history = helper.read_json(state_path) if state_path.is_file() else []
    names = [job.name for job in build_queue(settings, pilot) if only is None or job.stage in only]
    outcomes = {}
    for position, name in enumerate(names, 1):
        job = next(item for item in build_queue(settings, pilot) if item.name == name)  # fresh paths and status
        state = job.status()
        if state == "done":
            outcomes[name] = "done earlier"
            continue
        if state == "waiting":
            outcomes[name] = "waiting for an input"
            print(f"[{position}/{len(names)}] Skipping {name}. Missing: {'; '.join(job.missing())}", flush=True)
            continue
        if not lock_is_free():
            outcomes[name] = "not started: GPU lock is held by another job"
            print("Another job holds the GPU lock. Stopping the queue here.", flush=True)
            break
        print(f"\n[{position}/{len(names)}] {job.stage}: {name}", flush=True)
        record = {"job": name, "stage": job.stage, "started": datetime.now(timezone.utc).isoformat()}
        try:
            run_job(job)
            outcome = "passed" if job.done() else "finished without a completion marker"
        except KeyboardInterrupt:
            record.update(ended=datetime.now(timezone.utc).isoformat(), outcome="interrupted by user")
            history.append(record)
            helper.write_json(state_path, history)
            raise
        except Exception as error:  # keep the queue moving
            outcome = f"failed: {error}"
            print(traceback.format_exc(), flush=True)
        record.update(ended=datetime.now(timezone.utc).isoformat(), outcome=outcome)
        history.append(record)
        helper.write_json(state_path, history)
        outcomes[name] = outcome
    print("\nQueue finished.")
    for name, outcome in outcomes.items():
        print(f"  {outcome:<28} {name}")
    summarize(settings, pilot)
    if pilot:
        problems = [name for name, outcome in outcomes.items() if not outcome.startswith(("passed", "done earlier"))]
        passed = pilot_checks(settings) and not problems
        print("Pilot passed. The full queue can be started." if passed
              else "Pilot incomplete: review the lines marked [--] above before starting the full queue.")
    return outcomes


def pilot_checks(settings):
    """Everything the pilot must show before the full queue is trusted. Prints and saves each check."""
    where = root(settings, pilot=True)
    jobs = {job.name: job for job in build_queue(settings, pilot=True)}
    results = []

    def check(name, passed, detail):
        results.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'ok' if passed else '--'}] {name}: {detail}")

    print("Pilot checks")
    unfinished = [name for name, job in jobs.items() if not job.done()]
    check("Every pilot job finished", not unfinished, "all done" if not unfinished else f"not done: {unfinished}")
    if unfinished:
        helper.write_json(where / "pilot_checks.json", results)
        return False
    replay = helper.read_json(next(job for job in jobs.values() if job.stage == "replay").output / "summary.json")
    benchmark = helper.read_json(next(job for job in jobs.values() if job.stage == "benchmark").output / "summary.json")
    check("Saved checkpoint loads; its two encoder copies are identical",
          replay["checkpoint_layout"]["encoder_copy_identical"] is True, replay["checkpoint_layout"]["format"])
    check("Model size", replay["parameters"]["total"] == EXPECTED_PARAMETERS == benchmark["parameters"]["total"],
          f"{replay['parameters']['total']:,} parameters, all trained in fine-tuning")
    for label, summary in (("saved checkpoint", replay), ("pilot checkpoint", benchmark)):
        same = summary["agreement_batch1_vs_batched"]
        check(f"Batch 1 and batched scoring agree ({label})",
              same["comparable"] and same["grade_mismatches"] == 0 and same["largest_probability_difference"] < 0.02,
              f"{same.get('grade_mismatches')} grade mismatches, largest probability difference "
              f"{same.get('largest_probability_difference', float('nan')):.2e}")
    timing = replay["single_image_timing"]["model"]
    check("Timing recorded", timing["images"] > 0 and timing["median_seconds"] > 0,
          f"{1000 * timing['median_seconds']:.1f} ms per image at batch 1 on {timing['images']} images")
    saved = benchmark["agreement_with_saved_predictions"]
    check("Scoring worker reproduces the training worker's test predictions",
          saved["comparable"] and saved["grade_mismatches"] == 0,
          f"{saved.get('grade_mismatches')} grade mismatches, largest probability difference "
          f"{saved.get('largest_probability_difference', float('nan')):.2e}")
    runs = {name: helper.read_json(job.output / "train_result.json") for name, job in jobs.items() if job.kind == "train"}
    straight, resumed = (next(run for name, run in runs.items() if word in name) for word in ("straight", "resumed"))
    epochs = [len(run["history"]) for run in (straight, resumed)]
    check("Both pilot runs trained every pilot epoch", epochs == [PILOT_EPOCHS] * 2, f"epochs completed: {epochs}")
    resumed_job = next(job for name, job in jobs.items() if "resumed" in name)
    attempts = len(list((resumed_job.output / "attempts").glob("train-*")))
    check("The resumed run really stopped and resumed", attempts >= 2, f"{attempts} training attempts recorded")
    orders = [[entry["order_sha256"] for entry in run["history"]] for run in (straight, resumed)]
    check("Same image order with and without the interruption", orders[0] == orders[1],
          "identical in every epoch" if orders[0] == orders[1] else "the order differs")
    difference = max(abs(a[key] - b[key]) for a, b in zip(straight["history"], resumed["history"])
                     for key in ("train_loss", "validation_loss"))
    check("Same losses with and without the interruption", difference < 0.02,
          f"largest difference in train or validation loss: {difference:.2e}")
    peak = max(entry["torch_peak_reserved_bytes"] for run in (straight, resumed) for entry in run["history"])
    total = straight["configuration"]["gpu_total_bytes"]
    check("Training fits in GPU memory", peak < 0.9 * total,
          f"peak {peak / 2**30:.1f} GiB of {total / 2**30:.1f} GiB at {settings['physical_batch']} images per pass")
    rate = max(entry["train_images_per_second"] for entry in straight["history"])
    print(f"  Training speed after warm-up: {rate:.0f} images per second "
          f"(about {2786 / rate:.0f} s per full epoch on the clean split, before validation and saving).")
    if any(job.extra.get("random_init") for job in jobs.values()):
        print("  Note: the pilot trained from random weights. Repeat it once the pretrained checkpoint is here; "
              "that also checks that the checkpoint fits the encoder.")
    helper.write_json(where / "pilot_checks.json", results)
    return all(item["passed"] for item in results)


def parameter_counts(settings):
    """Exact counts from the saved checkpoint's tensors. No GPU."""
    import torch
    state = helper.load_state(settings["fine_tuned"], torch)
    groups = {"encoder": 0, "lora_branches": 0, "head": 0, "unused_copy_and_text_model": 0}
    for key, tensor in state.items():
        if key.startswith("img_encoder."):
            groups["encoder"] += tensor.numel()
            if ".linear_a_" in key or ".linear_b_" in key:
                groups["lora_branches"] += tensor.numel()
        elif key.startswith("classifier."):
            groups["head"] += tensor.numel()
        else:
            groups["unused_copy_and_text_model"] += tensor.numel()
    counts = {"method": "product of tensor shapes in the fine-tuned checkpoint",
              "counted_from": str(settings["fine_tuned"]), **groups,
              "model_total": groups["encoder"] + groups["head"],
              "trained_in_fine_tuning": groups["encoder"] + groups["head"],
              "note": "The original notebook set every encoder weight trainable, so fine-tuning trains the whole "
                      "model. The checkpoint file also stores a second encoder copy and the text model, which "
                      "the classifier does not use."}
    where = root(settings)
    where.mkdir(parents=True, exist_ok=True)
    helper.write_json(where / "parameter_counts.json", counts)
    print(f"RetiZero classifier: {counts['model_total']:,} parameters "
          f"({groups['encoder']:,} encoder + {groups['head']:,} head), all trained in fine-tuning.")
    print(f"  LoRA branches inside the encoder: {groups['lora_branches']:,}.")
    print(f"  Also stored in the file but unused by the classifier: {groups['unused_copy_and_text_model']:,}.")
    return counts


def per_class(metrics):
    cells = {}
    confusion = metrics["confusion_matrix"]
    total = sum(map(sum, confusion))
    for grade, item in enumerate(metrics["per_class"]):
        negatives = total - item["support"]
        false_positives = sum(row[grade] for row in confusion) - confusion[grade][grade]
        cells[f"sensitivity_{grade}"] = item["recall"]
        cells[f"specificity_{grade}"] = (negatives - false_positives) / negatives if negatives else None
    return cells


def telemetry(run, stage):
    """Wall time and utilization over every attempt of one stage of a job."""
    reports = [helper.read_json(path) for path in sorted((run / "attempts").glob(f"{stage}-*/run_report.json"))]
    sampled = [report for report in reports if report.get("samples")]
    cells = {"attempts": len(reports), "wall_minutes": sum(report["elapsed_seconds"] for report in reports) / 60}
    if sampled:
        samples = sum(report["samples"] for report in sampled)
        cells.update(mean_gpu_percent=sum(r["mean_gpu_percent"] * r["samples"] for r in sampled) / samples,
                     mean_cpu_percent=sum(r["mean_cpu_percent"] * r["samples"] for r in sampled) / samples,
                     peak_vram_mib=max(r["peak_vram_mib"] for r in sampled),
                     peak_ram_gib=max(r["peak_ram_bytes"] for r in sampled) / 2**30)
    return cells


def summarize(settings, pilot=False):
    """Collect every finished job into summary.csv and summary.md."""
    where = root(settings, pilot)
    rows, comparison = [], None
    for job in build_queue(settings, pilot):
        if not job.done():
            continue
        row = {"stage": job.stage, "job": job.name, "kind": job.kind}
        if job.kind == "score":
            summary = helper.read_json(job.output / "summary.json")
            metrics, identity, single = summary["metrics"], summary["identity"], summary["single_image_timing"]
            saved = summary.get("agreement_with_saved_predictions", {})
            row.update(cohort=identity["cohort"], checkpoint=Path(identity["checkpoint"]).name,
                       median_ms_per_image=1000 * single["model"]["median_seconds"],
                       p95_ms_per_image=1000 * single["model"]["p95_seconds"],
                       median_ms_read_and_transform=1000 * single["read_and_transform"]["median_seconds"],
                       images_per_second_batched=summary["batched"]["images_per_second"],
                       model_load_seconds=summary["model_load_seconds"],
                       torch_peak_gib_batch1=single["torch_peak_allocated_bytes"] / 2**30,
                       torch_peak_gib_batched=summary["batched"]["torch_peak_allocated_bytes"] / 2**30,
                       parameters=summary["parameters"]["total"],
                       batch1_vs_batched_mismatches=summary["agreement_batch1_vs_batched"].get("grade_mismatches"),
                       saved_prediction_grade_mismatches=saved.get("grade_mismatches"),
                       saved_prediction_largest_difference=saved.get("largest_probability_difference"),
                       **telemetry(job.output, "score"))
            if job.stage == "replay" and identity["cohort"] == "aptos367_original" and not identity["limit"]:
                comparison = helper.compare_with_reported(metrics)
                comparison.update(checkpoint_sha256=identity["checkpoint_sha256"],
                                  predictions_sha256=summary["predictions_sha256"])
        else:
            training = helper.read_json(job.output / "train_result.json")
            metrics = helper.read_json(job.output / "evaluation/aptos_test.summary.json")
            config, history = training["configuration"], training["history"]
            row.update(cohort=TRAIN_STAGES.get(job.stage, (None, None, "pilot_test"))[2],
                       split=config["dataset"]["split_name"], clahe=config["dataset"]["branch"],
                       seed=config["training"]["seed"], initial_weights=config["initial_weights"]["kind"],
                       effective_batch=config["training"]["effective_batch"],
                       physical_batch=config["training"]["physical_batch"],
                       learning_rate=config["training"]["learning_rate"],
                       epochs_run=training["epochs_completed"], best_epoch=training["best_epoch"],
                       best_validation_accuracy=training["best_validation_accuracy"],
                       train_images_per_second=statistics.fmean(e["train_images_per_second"] for e in history),
                       torch_peak_gib_training=max(e["torch_peak_allocated_bytes"] for e in history) / 2**30,
                       **telemetry(job.output, "train"))
        counts = [sum(column) for column in zip(*metrics["confusion_matrix"])]
        row.update(images=metrics["samples"], accuracy=metrics["accuracy"], qwk=metrics["qwk"],
                   macro_auc=metrics["macro_ovr_auc"], macro_f1=metrics["macro_f1"],
                   macro_precision=metrics["macro_precision"], macro_recall=metrics["macro_recall"],
                   predicted_per_grade=" ".join(map(str, counts)),
                   constant_predictor=max(counts) == metrics["samples"], **per_class(metrics))
        rows.append(row)
    if not rows:
        print("No finished jobs to summarise yet.")
        return []
    fields = list(dict.fromkeys(key for row in rows for key in row))
    where.mkdir(parents=True, exist_ok=True)
    with (where / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    def cell(value, style):
        return "" if value is None else format(value, style)

    lines = [f"# RetiZero runs summary ({where.name})", "",
             "Accuracy, QWK and macro AUC are computed by the same function that scores Qwen "
             "(`finetuning/Qwen3/paper_reproduction.py`).", ""]
    if pilot:
        lines += ["**Pilot results. Small subsets; not results for the paper.**", ""]
    if comparison is not None:
        helper.write_json(where / "replay_comparison.json", comparison)
        overall = [check for check in comparison["checks"] if "grade" not in check["item"]]
        lines += ["## Replay of the saved checkpoint against the original notebook's report", "",
                  "Every reported figure is reproduced." if comparison["all_match"]
                  else "**Some reported figures are not reproduced; see `replay_comparison.json`.**", "",
                  "| Figure | Reported | Replay | Match |", "|---|---|---|---|"]
        lines += [f"| {check['item']} | {check['reported']} | {cell(check['replay'], '.4f') if check['item'] != 'samples' else check['replay']} "
                  f"| {'yes' if check['match'] else 'NO'} |" for check in overall]
        grades = sum(check["match"] for check in comparison["checks"] if "grade" in check["item"])
        lines += ["", f"Per-grade support, precision, recall and F1: {grades} of "
                      f"{len(comparison['checks']) - len(overall)} figures match at the printed precision.", ""]
    scored = [row for row in rows if row["kind"] == "score"]
    if scored:
        lines += ["## Scoring jobs", "",
                  "| Stage | Job | Images | Accuracy | QWK | Macro AUC | Macro F1 | ms/image (batch 1) | Images/s (batched) | Peak GiB |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {row['stage']} | {row['job']} | {row['images']} | {cell(row['accuracy'], '.2%')} | "
                  f"{cell(row['qwk'], '.4f')} | {cell(row['macro_auc'], '.4f')} | {cell(row['macro_f1'], '.4f')} | "
                  f"{row['median_ms_per_image']:.1f} | {row['images_per_second_batched']:.0f} | "
                  f"{row['torch_peak_gib_batched']:.2f} |" for row in scored]
        lines += ["", "Time per image is the median at batch size 1 after three warm-up images, on the saved "
                      "224-pixel images. Peak GiB is PyTorch's own peak during batched scoring.", ""]
    trained = [row for row in rows if row["kind"] == "train"]
    if trained:
        lines += ["## Training runs", "",
                  "| Stage | Job | Split | CLAHE | Seed | Epochs | Best epoch | Best val. accuracy | Test images | Accuracy | QWK | Macro AUC | Wall min | Mean GPU % | Peak VRAM MiB |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {row['stage']} | {row['job']} | {row['split']} | {row['clahe']} | {row['seed']} | "
                  f"{row['epochs_run']} | {row['best_epoch']} | {row['best_validation_accuracy']:.2%} | {row['images']} | "
                  f"{cell(row['accuracy'], '.2%')} | {cell(row['qwk'], '.4f')} | {cell(row['macro_auc'], '.4f')} | "
                  f"{row['wall_minutes']:.1f} | {cell(row.get('mean_gpu_percent'), '.0f')} | "
                  f"{cell(row.get('peak_vram_mib'), '.0f')} |" for row in trained]
        groups = {}
        for row in trained:
            groups.setdefault((row["stage"], row["split"], row["clahe"]), []).append(row)
        if not pilot:
            lines += ["", "## Mean over seeds", "",
                      "| Stage | Split | CLAHE | Seeds | Accuracy mean ± SD | QWK mean | Macro AUC mean |",
                      "|---|---|---|---|---|---|---|"]
            for key, members in groups.items():
                accuracy = [100 * row["accuracy"] for row in members]
                deviation = f" ± {statistics.stdev(accuracy):.2f}" if len(accuracy) > 1 else ""
                lines.append("| " + " | ".join(key) + f" | {len(members)} | {statistics.fmean(accuracy):.2f}{deviation} | "
                             f"{cell(statistics.fmean(q) if (q := [row['qwk'] for row in members if row['qwk'] is not None]) else None, '.4f')} | "
                             f"{statistics.fmean(row['macro_auc'] for row in members):.4f} |")
    (where / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary of {len(rows)} finished job(s): {where / 'summary.md'}")
    return rows
