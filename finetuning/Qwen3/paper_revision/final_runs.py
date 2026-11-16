"""Queue for the final paper runs: pilot, scoring, training variants and summary.

Used by notebooks/03_final_runs.ipynb. Every GPU job goes through the existing
launchers, so each one keeps a log, hardware specs and a 5-second utilization
record. New results live under D:/Afterimage_Experiments/final_runs/<tag>, apart
from the 24 verified runs in D:/Afterimage_Experiments/runs.
"""

import csv
import json
import statistics
import struct
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import reproduce_run_02 as reproduction
import remaining_runs as remaining

REVISION = reproduction.REVISION
EXPERIMENTS = reproduction.EXPERIMENTS
RUNS = remaining.RUNS
DATA = remaining.DATA
FINAL = EXPERIMENTS / "final_runs"
SEEDS = remaining.SEEDS
COHORTS = {
    "aptos361_on": DATA / "aptos_512/duplicate_excluded/clahe_on/test.csv",
    "aptos361_off": DATA / "aptos_512/duplicate_excluded/clahe_off/test.csv",
    "messidor_archived": reproduction.FROZEN / "messidor_external.csv",
    "aptos367_original": reproduction.FROZEN / "test.csv",
}
# Minutes per job, from the logged times of the finished runs. Estimates only.
MINUTES = {"score_aptos": 8, "score_aptos_bf16": 14, "score_messidor": 18, "score_messidor_bf16": 30,
           "score_aptos_fp32": 11, "score_messidor_fp32": 30,
           "train": 85, "pilot_train": 6, "pilot_score": 4}
# A training job whose pinned FP16 score stopped with a non-finite likelihood. With a recorded
# diagnosis it is shown as an FP16 overflow and scored by the labelled FP32 fallback instead.
NONFINITE = "Non-finite answer likelihood"
STATUS_LABELS = {"fp16_overflow": "FP16 scoring failed: overflow",
                 "fp16_nonfinite": "FP16 scoring failed: non-finite, not diagnosed"}
FALLBACK_SCRIPT = "score_fp32_fallback.py"
FALLBACK_LABEL = "FP32 fallback"
VERIFICATION = REVISION.parents[2] / ".verification" / "final_runs"
FP16_OVERFLOW_DIAGNOSES = {
    "r16_on_alpha32_t43_a3408": {
        "layer": "language_model.layers.2.mlp.down_proj",
        "detail": ("FP16 output of language_model.layers.2.mlp.down_proj exceeds the FP16 maximum of 65,504 (largest "
                   "finite value 6.4e4; residual stream already 2.4e4 after layer 1); every later layer is NaN. Adapter "
                   "weights are finite. BF16 and FP32 give finite, near-constant likelihoods."),
        "evidence": [VERIFICATION / "diagnose_alpha32.py", VERIFICATION / "diagnose_alpha32_best.log",
                     VERIFICATION / "diagnose_alpha32_t42.log", VERIFICATION / "diagnose_alpha32_fp32.py",
                     VERIFICATION / "diagnose_alpha32_fp32.log"],
        "diagnostic_predictions": {"aptos361_on": VERIFICATION / "alpha32_t43_fp32_predictions.csv"},
    },
}
OVERRIDE_FLAGS = {"lr": "--lr", "lora_alpha": "--lora-alpha", "lora_dropout": "--lora-dropout",
                  "early_stopping_patience": "--early-stopping-patience"}


def default_settings():
    return {
        "tag": "final_01", "pilot_tag": "pilot_01",
        "run_tag": "01", "control_tag": "paper_controls_01",
        "bf16_base": EXPERIMENTS / "assets/qwen3-vl-8b-bf16", "offload_layers": 16,
        "stages": {"zero_shot": True, "benchmark": True, "original_adapter_audit": True,
                   "messidor_archived": True, "lr_study": True, "alpha_study": True,
                   "extra_variants": True, "main_lr_grid": False, "messidor_for_new_runs": True},
        "zero_shot_bf16": True,
        "generation_samples": -1,
        "benchmark_runs": [("r16_on", 1), ("r16_on", 2), ("r16_on", 3), ("r8_on", 3), ("r32_on", 1), ("bf16", 1)],
        "messidor_runs": [("r16_on", 1), ("r16_on", 2), ("r16_on", 3), ("bf16", 1), ("bf16", 2), ("bf16", 3)],
        "study_rank": 16, "study_clahe": "on", "seed_pairs": [1, 2, 3],
        "lr_values": [1e-4, 2e-4], "alpha_values": [8, 32],
        "extra_variants": [],
        "main_lr": None, "main_lr_ranks": [8, 16, 32], "main_lr_clahe": ["on", "off"],
        "original_adapter": remaining.LEGACY / "Qwen3/lr0.0005_epoch4_batchsize8/best_lora",
        "fp16_overflow_diagnoses": dict(FP16_OVERFLOW_DIAGNOSES),
    }


@dataclass
class Job:
    stage: str
    name: str
    kind: str                      # "score" or "train"
    output: Path
    arguments: list
    minutes: int
    needs: Path | None = None      # a file that must exist before this job can run
    note: str = ""
    extra: dict = field(default_factory=dict)

    def status(self):
        if self.kind == "score":
            return "done" if (self.output / "summary.json").is_file() else self._waiting()
        marker = self.output / "latest_attempt.json"
        if marker.is_file():
            state = reproduction.read(marker)["status"]
            if state == "passed" and (self.output / "score_result.json").is_file():
                return "done"
            if state == "failed" and fp16_score_failed(self.output):
                return "fp16_overflow" if self.extra.get("fp16_overflow") else "fp16_nonfinite"
            return "interrupted" if state == "running" else state
        return self._waiting()

    def _waiting(self):
        return "waiting" if self.needs is not None and not self.needs.exists() else "pending"

    def label(self):
        state = self.status()
        return STATUS_LABELS.get(state, state)


def fp16_score_failed(run):
    """Training finished, but the latest attempt's pinned FP16 score stopped with a non-finite likelihood."""
    run = Path(run)
    marker = run / "latest_attempt.json"
    if not (run / "train_result.json").is_file() or (run / "score_result.json").is_file() or not marker.is_file():
        return False
    log = Path(reproduction.read(marker)["path"]) / "score.log"
    return log.is_file() and NONFINITE in log.read_text(encoding="utf-8", errors="replace")


def latest_score_log(run):
    return Path(reproduction.read(Path(run) / "latest_attempt.json")["path"]) / "score.log"


def fallback_name(run_name, cohort):
    """Folder name of an FP32 fallback score: '<run>__aptos_test_fp32' or '<run>__<cohort>_fp32'."""
    return f"{run_name}__{'aptos_test' if cohort.startswith('aptos361') else cohort}_fp32"


def root(settings, pilot=False):
    tag = remaining.valid_tag(settings["pilot_tag"] if pilot else settings["tag"])
    return FINAL / (f"{settings['tag']}__{tag}" if pilot else tag)


def number(value):
    return f"{value:g}".replace("-0", "-").replace("+0", "").replace(".", "p")


def variant_name(rank, clahe, seed_pair, overrides):
    trainer, adapter = SEEDS[seed_pair]
    short = {"lr": "lr", "lora_alpha": "alpha", "lora_dropout": "drop", "early_stopping_patience": "pat"}
    parts = [f"{short[key]}{number(value)}" for key, value in sorted(overrides.items())]
    return "_".join([f"r{rank}", clahe, *parts, f"t{trainer}", f"a{adapter}"])


def existing_run(settings, condition, seed_pair):
    """A finished run in runs/: its folder, base, precision and tokenizer source."""
    if condition == "bf16":
        run = RUNS / remaining.bf16_run_name(seed_pair, settings["control_tag"])
        config = reproduction.read(run / "pilot_configuration.json")
        return {"label": f"bf16_r16_on_p{seed_pair}", "run": run, "base": Path(config["base"]),
                "precision": "bf16", "tokenizer": Path(config["reference_base"]),
                "offload": config["offload_layers"], "clahe": "on"}
    run = RUNS / remaining.qwen_run_name(condition, seed_pair, "duplicate_excluded", settings["run_tag"])
    config = reproduction.read(run / "pilot_configuration.json")
    return {"label": f"{condition}_p{seed_pair}", "run": run, "base": Path(config["base"]),
            "precision": "4bit", "tokenizer": None, "offload": 0, "clahe": condition.split("_")[1]}


def score_job(stage, label, cohort, base, *, where, precision="4bit", adapter=None, tokenizer=None,
              offload=0, generation=0, reference=None, limit=0, needs=None, note=""):
    name = f"{label}__{cohort}"
    output = where / "scores" / name
    arguments = ["--name", name, "--output-dir", output, "--base", base, "--base-precision", precision,
                 "--cohort-csv", COHORTS[cohort], "--cohort-name", cohort, "--generation-samples", generation]
    if precision == "bf16":
        arguments += ["--tokenizer", tokenizer, "--offload-layers", offload]
    if adapter is not None:
        arguments += ["--adapter", adapter]
    if reference is not None:
        arguments += ["--reference-predictions", reference]
    if limit:
        arguments += ["--limit", limit]
    bf16 = "_bf16" if precision == "bf16" else ""
    minutes = MINUTES["pilot_score"] if limit else MINUTES[("score_messidor" if cohort.startswith("messidor") else "score_aptos") + bf16]
    return Job(stage, name, "score", output, arguments, minutes, needs=needs, note=note)


def fallback_score_job(stage, train, cohort, *, where, diagnosis, needs=None):
    """Labelled FP32 fallback score of a trained run whose pinned FP16 score overflowed."""
    name = fallback_name(train.name, cohort)
    output = where / "scores" / name
    # APTOS: the run's own test list, exactly what the pinned score stage would have scored.
    cohort_csv = train.output / "aptos_test.csv" if cohort.startswith("aptos361") else COHORTS[cohort]
    arguments = ["--name", name, "--output-dir", output, "--base", reproduction.BASE,
                 "--adapter", train.output / "best_lora", "--train-run", train.output,
                 "--failure-log", latest_score_log(train.output), "--overflow-layer", diagnosis["layer"],
                 "--diagnosis-detail", diagnosis.get("detail", ""),
                 "--cohort-csv", cohort_csv, "--cohort-name", cohort]
    for path in diagnosis.get("evidence", []):
        arguments += ["--diagnosis-evidence", path]
    reference = diagnosis.get("diagnostic_predictions", {}).get(cohort)
    if reference is not None:
        arguments += ["--diagnostic-predictions", reference]
    minutes = MINUTES["score_messidor_fp32" if cohort.startswith("messidor") else "score_aptos_fp32"]
    return Job(stage, name, "score", output, arguments, minutes, needs=needs,
               note=f"{FALLBACK_LABEL}: FP16 scoring of {train.name} overflowed in {diagnosis['layer']}",
               extra={"script": FALLBACK_SCRIPT, "fallback_for": train.name, "precision": FALLBACK_LABEL})


def train_job(stage, rank, clahe, seed_pair, overrides, *, where, pilot=False):
    trainer, adapter = SEEDS[seed_pair]
    name = variant_name(rank, clahe, seed_pair, overrides)
    output = where / "train" / name
    arguments = ["--output", output, "--lock-dir", RUNS, "--project", remaining.LEGACY,
                 "--candidate", DATA / "aptos_512/duplicate_excluded/manifest.json",
                 "--base-snapshot", reproduction.BASE, "--rank", rank, "--clahe", clahe,
                 "--trainer-seed", trainer, "--adapter-seed", adapter, "--variant-name", name]
    arguments += ["--steps", 10] if pilot else ["--full"]
    for key, value in sorted(overrides.items()):
        arguments += [OVERRIDE_FLAGS[key], value]
    return Job(stage, name, "train", output, arguments, MINUTES["pilot_train" if pilot else "train"],
               extra={"rank": rank, "clahe": clahe, "seed_pair": seed_pair, "overrides": dict(overrides)})


def variants(settings):
    """(stage, rank, clahe, seed_pair, overrides) for every enabled training variant, without repeats."""
    stages, rank, clahe = settings["stages"], settings["study_rank"], settings["study_clahe"]
    wanted = []
    if stages["lr_study"]:
        wanted += [("lr_study", rank, clahe, pair, {"lr": value})
                   for value in settings["lr_values"] for pair in settings["seed_pairs"]]
    if stages["alpha_study"]:
        wanted += [("alpha_study", rank, clahe, pair, {"lora_alpha": value})
                   for value in settings["alpha_values"] for pair in settings["seed_pairs"]]
    if stages["extra_variants"]:
        for item in settings["extra_variants"]:
            item = dict(item)
            wanted.append(("extra_variants", item.pop("rank", rank), item.pop("clahe", clahe),
                           item.pop("seed_pair"), item))
    if stages["main_lr_grid"]:
        if settings["main_lr"] is None:
            raise ValueError("Set main_lr before enabling main_lr_grid.")
        wanted += [("main_lr_grid", r, c, pair, {"lr": settings["main_lr"]})
                   for r in settings["main_lr_ranks"] for c in settings["main_lr_clahe"]
                   for pair in settings["seed_pairs"]]
    seen, unique = set(), []
    for item in wanted:
        key = variant_name(*item[1:])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def build_queue(settings, pilot=False):
    where = root(settings, pilot)
    stages = settings["stages"]
    base = reproduction.BASE
    limit = 12 if pilot else 0
    generation = 6 if pilot else settings["generation_samples"]
    jobs = []

    def first(items):
        return items[:1] if pilot else items

    if stages["zero_shot"]:
        for cohort in first(["aptos361_on", "aptos361_off"]):
            jobs.append(score_job("zero_shot", "zeroshot_4bit", cohort, base, where=where,
                                  generation=generation, limit=limit))
        if settings["zero_shot_bf16"] and not pilot:
            reference = existing_run(settings, "bf16", 1)
            jobs.append(score_job("zero_shot", "zeroshot_bf16", "aptos361_on", settings["bf16_base"], where=where,
                                  precision="bf16", tokenizer=reference["tokenizer"],
                                  offload=settings["offload_layers"], generation=generation))
    if stages["benchmark"]:
        chosen = settings["benchmark_runs"]
        if pilot:
            chosen = [item for item in chosen if item[0] != "bf16"][:1] + [item for item in chosen if item[0] == "bf16"][:1]
        for condition, pair in chosen:
            run = existing_run(settings, condition, pair)
            jobs.append(score_job("benchmark", run["label"], f"aptos361_{run['clahe']}", run["base"], where=where,
                                  precision=run["precision"], adapter=run["run"] / "best_lora",
                                  tokenizer=run["tokenizer"], offload=run["offload"], generation=generation,
                                  reference=run["run"] / "aptos_test.predictions.csv", limit=limit))
    if stages["original_adapter_audit"] and not pilot:
        adapter = Path(settings["original_adapter"])
        jobs.append(score_job("original_adapter_audit", "original_paper_adapter", "aptos367_original", base,
                              where=where, adapter=adapter, generation=-1, needs=adapter / "adapter_config.json",
                              note="skipped unless the original adapter folder exists"))
    if stages["messidor_archived"]:
        jobs.append(score_job("messidor_archived", "zeroshot_4bit", "messidor_archived", base, where=where,
                              generation=0, limit=limit))
        for condition, pair in ([] if pilot else settings["messidor_runs"]):
            run = existing_run(settings, condition, pair)
            jobs.append(score_job("messidor_archived", run["label"], "messidor_archived", run["base"], where=where,
                                  precision=run["precision"], adapter=run["run"] / "best_lora",
                                  tokenizer=run["tokenizer"], offload=run["offload"]))
    planned = variants(settings)
    if pilot:
        kept, seen = [], set()
        for item in planned:  # one pilot per kind of override
            kind = tuple(sorted(item[4]))
            if kind not in seen:
                seen.add(kind)
                kept.append(item)
        planned = kept
    for stage, rank, clahe, pair, overrides in planned:
        job = train_job(stage, rank, clahe, pair, overrides, where=where, pilot=pilot)
        jobs.append(job)
        diagnosis = settings.get("fp16_overflow_diagnoses", {}).get(job.name)
        if diagnosis and not pilot:
            job.extra["fp16_overflow"] = diagnosis
        if job.status() == "fp16_overflow":
            aptos = fallback_score_job(stage, job, f"aptos361_{clahe}", where=where, diagnosis=diagnosis)
            job.note = f"{STATUS_LABELS['fp16_overflow']}; {FALLBACK_LABEL}: {aptos.name}"
            jobs.append(aptos)
            if stages["messidor_for_new_runs"] and clahe == "on":
                jobs.append(fallback_score_job(stage, job, "messidor_archived", where=where, diagnosis=diagnosis,
                                               needs=aptos.output / "summary.json"))
            continue
        if stages["messidor_for_new_runs"] and clahe == "on" and not pilot:
            adapter = job.output / "best_lora"
            jobs.append(score_job(stage, job.name, "messidor_archived", base, where=where, adapter=adapter,
                                  needs=job.output / "score_result.json",
                                  note="runs after its training job"))
    return jobs


def show_plan(settings, pilot=False):
    jobs = build_queue(settings, pilot)
    print(f"{'Pilot' if pilot else 'Final'} queue in {root(settings, pilot)}")
    left = 0
    for job in jobs:
        state = job.status()
        if state not in ("done", *STATUS_LABELS):
            left += job.minutes
        note = ""
        if state in STATUS_LABELS:
            note = f"  -> {job.note}" if state == "fp16_overflow" else "  -> record a diagnosis to queue a fallback"
        elif "fallback_for" in job.extra:
            note = f"  ({FALLBACK_LABEL}, not the FP16 protocol)"
        print(f"  [{job.label():<11}] {job.stage:<22} {job.name}  (~{job.minutes} min){note}")
    done = sum(job.status() == "done" for job in jobs)
    failed_fp16 = sum(job.status() in STATUS_LABELS for job in jobs)
    extra = f" {failed_fp16} trained run(s) with FP16 scoring failed (shown above)." if failed_fp16 else ""
    print(f"{done} of {len(jobs)} jobs done.{extra} Estimated GPU time left: about {left / 60:.1f} h. "
          "Training times are estimates from earlier runs.")
    return jobs


def lock_is_free():
    _, monitor = reproduction.runtime()
    try:
        with monitor["gpu_lock"](RUNS):
            return True
    except RuntimeError:
        return False


def run_job(job):
    launch, monitor = reproduction.runtime()
    if job.kind == "train":
        launch(REVISION / "qwen_variant.py", job.arguments,
               job.output.parent.parent / "logs" / job.name / "launch.log", keep_pc_awake=True)
        return
    (job.output / "attempts").mkdir(parents=True, exist_ok=True)
    with monitor["gpu_lock"](RUNS):
        reproduction.run_stage("score", job.arguments, job.output,
                               script=REVISION / job.extra.get("script", "score_cohort.py"))


def run_queue(settings, pilot=False, only=None):
    """Run every unfinished job in order. A failed job is recorded and the queue continues."""
    where = root(settings, pilot)
    where.mkdir(parents=True, exist_ok=True)
    state_path = where / "queue_state.json"
    history = reproduction.read(state_path) if state_path.is_file() else []
    jobs = [job for job in build_queue(settings, pilot) if only is None or job.stage in only]
    outcomes = {}
    for position, job in enumerate(jobs, 1):
        state = job.status()
        if state == "done":
            outcomes[job.name] = "done earlier"
            continue
        if state in STATUS_LABELS:  # training finished; rerunning would only repeat the FP16 failure
            outcomes[job.name] = (job.note if state == "fp16_overflow"
                                  else f"{job.label()}: record a diagnosis to queue a fallback")
            print(f"[{position}/{len(jobs)}] {job.name}: {outcomes[job.name]}", flush=True)
            continue
        if state == "waiting":
            outcomes[job.name] = f"skipped: needs {job.needs}"
            print(f"[{position}/{len(jobs)}] Skipping {job.name}: {job.needs} is missing.", flush=True)
            continue
        if not lock_is_free():
            outcomes[job.name] = "not started: GPU lock is held by another job"
            print("Another job holds the GPU lock. Stopping the queue here.", flush=True)
            break
        print(f"\n[{position}/{len(jobs)}] {job.stage}: {job.name}", flush=True)
        record = {"job": job.name, "stage": job.stage, "started": datetime.now(timezone.utc).isoformat()}
        try:
            run_job(job)
            outcome = "passed" if job.status() == "done" else "finished without a completion marker"
        except KeyboardInterrupt:
            record.update(ended=datetime.now(timezone.utc).isoformat(), outcome="interrupted by user")
            history.append(record)
            reproduction.original.write_json(state_path, history)
            raise
        except Exception as error:  # keep the overnight queue moving
            outcome = f"failed: {error}"
            print(traceback.format_exc(), flush=True)
        record.update(ended=datetime.now(timezone.utc).isoformat(), outcome=outcome)
        history.append(record)
        reproduction.original.write_json(state_path, history)
        outcomes[job.name] = outcome
    print("\nQueue finished.")
    for name, outcome in outcomes.items():
        print(f"  {outcome:<28} {name}")
    problems = [name for name, outcome in outcomes.items()
                if not outcome.startswith(("passed", "done earlier", STATUS_LABELS["fp16_overflow"]))]
    if pilot:
        print("Pilot passed. The full queue can be started." if not problems
              else f"Pilot incomplete: {len(problems)} job(s) need attention before the full queue.")
    summarize(settings, pilot)
    return outcomes


def safetensors_parameters(path):
    """Count parameters from a safetensors header, without loading the weights."""
    with Path(path).open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
    total = 0
    for key, tensor in header.items():
        if key != "__metadata__":
            count = 1
            for size in tensor["shape"]:
                count *= size
            total += count
    return total


def parameter_counts(settings):
    """Exact counts from the weight files: the BF16 base and one adapter per rank."""
    base = Path(settings["bf16_base"])
    index = base / "model.safetensors.index.json"
    shards = sorted(set(reproduction.read(index)["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    counts = {"method": "product of tensor shapes in the safetensors headers",
              "base_model": str(base), "base_parameters": sum(safetensors_parameters(base / name) for name in shards),
              "adapters": {}}
    for rank in (8, 16, 32):
        for pair in SEEDS:
            run = RUNS / remaining.qwen_run_name(f"r{rank}_on", pair, "duplicate_excluded", settings["run_tag"])
            weights = run / "best_lora/adapter_model.safetensors"
            if weights.is_file():
                trainable = safetensors_parameters(weights)
                counts["adapters"][f"rank_{rank}"] = {
                    "trainable_parameters": trainable, "adapter_file_bytes": weights.stat().st_size,
                    "percent_of_base": 100 * trainable / counts["base_parameters"], "counted_from": str(weights)}
                break
    where = root(settings)
    where.mkdir(parents=True, exist_ok=True)
    reproduction.original.write_json(where / "parameter_counts.json", counts)
    print(f"Base model: {counts['base_parameters']:,} parameters.")
    for name, item in counts["adapters"].items():
        print(f"  LoRA {name}: {item['trainable_parameters']:,} trainable "
              f"({item['percent_of_base']:.3f}% of the base), {item['adapter_file_bytes'] / 2**20:.0f} MiB on disk.")
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




def training_fields(job):
    config = reproduction.read(job.output / "pilot_configuration.json")
    training = reproduction.read(job.output / "train_result.json")
    return dict(cohort="aptos361_" + job.extra["clahe"], base="4bit", adapter="yes",
                rank=config["rank"], seed_pair=job.extra["seed_pair"],
                **{key: config["training"][key] for key in OVERRIDE_FLAGS},
                best_validation_loss=training["validation_loss"],
                epochs_run=max((entry.get("epoch", 0) for entry in training["history"]), default=None))


def group_key(row):
    return (row["stage"], row["rank"], row["cohort"], *(row[name] for name in OVERRIDE_FLAGS))


def summarize(settings, pilot=False):
    """Collect every finished job into summary.csv and summary.md in the final-runs folder.

    A trained run whose FP16 score failed gets a row with no metrics; its FP32 fallback scores
    are separate rows flagged in scoring_precision. Group means use FP16 (protocol) seeds only,
    with a separate, labelled line that adds the fallback seeds.
    """
    where = root(settings, pilot)
    rows = []
    jobs = build_queue(settings, pilot)
    for job in jobs:
        state = job.status()
        if job.kind == "train" and state in STATUS_LABELS:
            rows.append({"stage": job.stage, "job": job.name, "kind": job.kind, **training_fields(job),
                         "scoring_precision": "FP16", "scoring_status": job.label(),
                         "fallback_jobs": " ".join(other.name for other in jobs
                                                   if other.extra.get("fallback_for") == job.name)})
            continue
        if state != "done":
            continue
        row = {"stage": job.stage, "job": job.name, "kind": job.kind}
        if job.kind == "score":
            summary = reproduction.read(job.output / "summary.json")
            metrics, identity = summary["metrics"], summary["identity"]
            timing, generation = summary["full_answer_timing"], summary["generation"] or {}
            agreement = summary.get("agreement_with_saved_predictions", {})
            row.update(cohort=identity["cohort"], images=metrics["samples"], base=identity["base_precision"],
                       adapter="yes" if identity["adapter"] else "none",
                       median_seconds_per_image=timing["median_seconds"], p95_seconds_per_image=timing["p95_seconds"],
                       model_load_seconds=summary["model_load_seconds"],
                       generation_median_seconds=generation.get("timing", {}).get("median_seconds"),
                       generation_unparseable=generation.get("unparseable_outputs"),
                       generation_images=generation.get("images"),
                       generation_accuracy_strict=generation.get("accuracy_unparseable_counted_wrong"),
                       generation_accuracy_grade0_fallback=generation.get("accuracy_unparseable_as_grade_0"),
                       prompt_tokens=generation.get("prompt_tokens"),
                       torch_peak_gib=summary["torch_peak_allocated_bytes"] / 2**30,
                       saved_prediction_grade_mismatches=agreement.get("grade_mismatches"),
                       saved_prediction_largest_difference=agreement.get("largest_probability_difference"))
            precision = identity.get("scoring_precision", "FP16")
        else:
            result = reproduction.read(job.output / "score_result.json")
            row.update(training_fields(job))
            precision = "FP16"
            if "metrics" not in result:  # pilot: validation subset only
                row.update(images=result["validation_photos_scored"], scoring_precision=precision)
                rows.append(row)
                continue
            predictions = reproduction.original.read_csv(job.output / "aptos_test.predictions.csv")
            metrics = reproduction.original.calculate_metrics(predictions)
            row.update(images=metrics["samples"])
        counts = [sum(column) for column in zip(*metrics["confusion_matrix"])]
        row.update(accuracy=metrics["accuracy"], qwk=metrics["qwk"], macro_auc=metrics["macro_ovr_auc"],
                   predicted_per_grade=" ".join(map(str, counts)),
                   constant_predictor=max(counts) == metrics["samples"], **per_class(metrics),
                   scoring_precision=precision)
        if "fallback_for" in job.extra:
            reason = summary.get("fallback_reason", {})
            agreement = summary.get("agreement_with_diagnostic_predictions", {})
            row.update(fallback_for=job.extra["fallback_for"], fallback_reason=reason.get("summary"),
                       fp16_probe=(summary.get("fp16_probe") or {}).get("outcome"),
                       diagnostic_grade_mismatches=agreement.get("grade_mismatches"),
                       diagnostic_largest_difference=agreement.get("largest_probability_difference"))
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

    def scoring(row):
        return f"**{FALLBACK_LABEL}**" if row.get("fallback_for") else row.get("scoring_precision", "FP16")

    failed = [row for row in rows if row.get("scoring_status")]
    lines = [f"# Final runs summary ({where.name})", "",
             "Accuracy, QWK and macro AUC come from the pinned five-answer scorer. Times exclude model loading "
             "and the first three images.", ""]
    if failed:
        lines += ["Rows marked **FP32 fallback** are not the FP16 scoring protocol: the run's FP16 score "
                  "overflowed, so the same scorer was run with the unquantized weights in FP32 (4-bit base "
                  "unchanged). See *FP16 scoring failures* below.", ""]
    lines += ["| Stage | Job | Cohort | Images | Scoring | Accuracy | QWK | Macro AUC | Median s/image | Unparseable generations |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("scoring_status"):
            lines.append(f"| {row['stage']} | {row['job']} | {row['cohort']} |  | {row['scoring_status']} | "
                         f"see {row['fallback_jobs'] or 'no fallback yet'} |  |  |  |  |")
            continue
        if "accuracy" not in row:
            continue
        unparseable = "" if row.get("generation_images") is None else f"{row['generation_unparseable']} of {row['generation_images']}"
        lines.append(f"| {row['stage']} | {row['job']} | {row['cohort']} | {row['images']} | {scoring(row)} | "
                     f"{cell(row['accuracy'], '.2%')} | {cell(row['qwk'], '.4f')} | {cell(row['macro_auc'], '.4f')} | "
                     f"{cell(row.get('median_seconds_per_image'), '.3f')} | {unparseable} |")
    trained = {row["job"]: row for row in rows if row["kind"] == "train"}
    groups, fallbacks, excluded = {}, {}, {}
    for row in rows:
        if row["kind"] == "train" and "accuracy" in row:
            groups.setdefault(group_key(row), []).append(row)
        elif row["kind"] == "train" and row.get("scoring_status"):
            excluded.setdefault(group_key(row), []).append(row)
        elif row.get("fallback_for") in trained and row["cohort"] == trained[row["fallback_for"]]["cohort"]:
            fallbacks.setdefault(group_key(trained[row["fallback_for"]]), []).append(row)
    for key in list(excluded) + list(fallbacks):
        groups.setdefault(key, [])
    if groups:
        lines += ["", "## Training variants, mean over seeds", "",
                  "Means use the seeds scored with the FP16 protocol. Where a seed's FP16 score failed, a second "
                  "line adds its FP32 fallback result; that line mixes precisions and is labelled as such.", "",
                  "| Stage | Rank | Cohort | lr | alpha | dropout | patience | Scoring | Seeds | Accuracy mean ± SD | QWK mean | Macro AUC mean | Constant predictors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

        def line(key, members, label, seeds):
            if not members:
                return "| " + " | ".join(map(str, key)) + f" | {label} | {seeds} |  |  |  |  |"
            accuracy = [100 * row["accuracy"] for row in members]
            spread = f" ± {statistics.stdev(accuracy):.2f}" if len(accuracy) > 1 else ""
            defined = [row["qwk"] for row in members if row["qwk"] is not None]
            return ("| " + " | ".join(map(str, key)) + f" | {label} | {seeds} | {statistics.fmean(accuracy):.2f}{spread} | "
                    f"{cell(statistics.fmean(defined) if defined else None, '.4f')} | "
                    f"{statistics.fmean(row['macro_auc'] for row in members):.4f} | "
                    f"{sum(row['constant_predictor'] for row in members)} |")

        for key, members in groups.items():
            missing = excluded.get(key, [])
            if not missing:
                lines.append(line(key, members, "FP16", len(members)))
                continue
            lines.append(line(key, members, "FP16 (protocol)",
                              f"{len(members)} ({len(missing)} excluded: FP16 scoring failed)"))
            added = fallbacks.get(key, [])
            if added:
                lines.append(line(key, members + added, f"FP16 + {FALLBACK_LABEL} (mixed)",
                                  f"{len(members) + len(added)} (incl. {len(added)} {FALLBACK_LABEL})"))
    if failed:
        lines += ["", "## FP16 scoring failures", ""]
        for row in failed:
            reason = next((other.get("fallback_reason") for other in rows
                           if other.get("fallback_for") == row["job"] and other.get("fallback_reason")), None)
            lines.append(f"- **{row['job']}**: {row['scoring_status']}. "
                         + (reason + " " if reason else "")
                         + (f"{FALLBACK_LABEL} jobs: {row['fallback_jobs']}." if row["fallback_jobs"]
                            else "No fallback queued."))
            for other in rows:
                if other.get("fallback_for") == row["job"] and other.get("diagnostic_grade_mismatches") is not None:
                    lines.append(f"  - {other['job']}: compared with the earlier diagnostic FP32 predictions, "
                                 f"{other['diagnostic_grade_mismatches']} grade mismatches, largest probability "
                                 f"difference {other['diagnostic_largest_difference']:.2e}.")
                if other.get("fallback_for") == row["job"] and other.get("fp16_probe"):
                    lines.append(f"  - {other['job']}: FP16 probe on its first image: {other['fp16_probe']}.")
    (where / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary of {len(rows)} finished job(s): {where / 'summary.md'}")
    return rows
