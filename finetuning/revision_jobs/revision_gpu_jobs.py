"""One command for the remaining GPU work of the paper revision.

    python -B finetuning/revision_jobs/revision_gpu_jobs.py all --messidor-package <Messidor-2_for_teammate folder>

`all` runs, in order, and stops at the first step that fails:
  check           inputs, environment and GPU lock (reads files only)
  build-messidor  the verified Messidor-2 cohort from the raw images (CPU, about 5 minutes)
  pilot           every kind of job on a few images (about 15 minutes)
  run             the full queue; finished jobs are skipped, so rerunning resumes
  summarize       status table, energy per job, then the CPU analysis tables

Each step can also be run on its own: `... revision_gpu_jobs.py status`, `... run`, and so on.

GPU jobs (all scoring-only unless --rank-sweep is given):
  messidor2   verified Messidor-2 (1,744 gradable images, adjudicated grades, raw -> model input in one step)
              Qwen: r16 CLAHE-on lr2e-4 x3, r16 CLAHE-on lr5e-4 x3, r16 CLAHE-off lr5e-4 x3, zero-shot on/off
              RetiZero: matched x3, CLAHE x3, saved app checkpoint                            about 6 h
  timing      warm per-image timing, greedy-generation check and power for the 7 Qwen
              configurations that have no timing record (361 images each)                        about 2 h
  rank_sweep  OPTIONAL (--rank-sweep): ranks 8/16/32 x CLAHE on/off at lr 2e-4, plus rank 8 and 32 with
              alpha = rank (fixed alpha/r), three seeds each; trains 21 new adapters            about 30 h

Every job goes through the existing launchers in finetuning/Qwen3/paper_revision/final_runs.py and
finetuning/RetiZero/paper_revision/retizero_runs.py, so each keeps its log, hardware specs, 5-second utilisation
record and takes the shared GPU lock. GPU board power is sampled every 5 s by nvidia-smi for the whole queue.
Results: D:/Afterimage_Experiments/final_runs/final_02 and D:/Afterimage_Experiments/retizero_runs/retizero_02.
Nothing in final_01, retizero_01 or runs/ is written to.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINETUNING = HERE.parent
REPO = FINETUNING.parent
for folder in (FINETUNING / "Qwen3", FINETUNING / "Qwen3/paper_revision", FINETUNING / "RetiZero/paper_revision"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import final_runs as final  # noqa: E402
import remaining_runs as remaining  # noqa: E402
import retizero_runs as rz  # noqa: E402

EXPERIMENTS = final.EXPERIMENTS
MESSIDOR2 = EXPERIMENTS / "datasets" / "messidor2_verified"
QWEN_TAG, RETIZERO_TAG, PILOT_TAG = "final_02", "retizero_02", "pilot_01"
LR_SELECTED = 2e-4           # chosen on validation loss (revision_analysis/lr_selection_by_validation_loss.csv)
PAIRS = (1, 2, 3)            # seed pairs 42/3407, 43/3408, 44/3409
MESSIDOR2_MINUTES = 30
COHORT_CSV = {
    "messidor2_on": MESSIDOR2 / "qwen_512/clahe_on/test.csv",
    "messidor2_off": MESSIDOR2 / "qwen_512/clahe_off/test.csv",
    "messidor2_224_off": MESSIDOR2 / "retizero_224/clahe_off/test.csv",
    "messidor2_224_on": MESSIDOR2 / "retizero_224/clahe_on/test.csv",
}
POWER_FIELDS = "timestamp,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu,clocks.sm"


def qwen_root(pilot=False):
    return final.FINAL / (f"{QWEN_TAG}__{PILOT_TAG}" if pilot else QWEN_TAG)


def retizero_root(pilot=False):
    return rz.RESULTS / (f"{RETIZERO_TAG}__{PILOT_TAG}" if pilot else RETIZERO_TAG)


def final01_train(rank, clahe, pair, overrides):
    return final.FINAL / "final_01" / "train" / final.variant_name(rank, clahe, pair, overrides)


# --------------------------------------------------------------------------- queue

def build_jobs(pilot=False, rank_sweep=False):
    """[(queue, job)] in run order. queue is 'qwen' or 'retizero'."""
    final.COHORTS.update({k: v for k, v in COHORT_CSV.items() if k.startswith("messidor2_o")})
    qs = final.default_settings()
    rs = rz.default_settings()
    wq, wr = qwen_root(pilot), retizero_root(pilot)
    base = final.reproduction.BASE
    limit = 12 if pilot else 0
    jobs = []

    # RetiZero on verified Messidor-2 (minutes in total).
    trained = rz.RESULTS / "retizero_01" / "train"
    retizero = [("saved_checkpoint", Path(rs["fine_tuned"]), "messidor2_224_off")]
    retizero += [(f"matched_t{s}", rz.selected_checkpoint(trained / f"matched_t{s}"), "messidor2_224_off") for s in rs["seeds"]]
    retizero += [(f"clahe_t{s}", rz.selected_checkpoint(trained / f"clahe_t{s}"), "messidor2_224_on") for s in rs["seeds"]]
    for label, checkpoint, cohort in (retizero[1:2] if pilot else retizero):
        jobs.append(("retizero", rz.score_job("messidor2", label, cohort, checkpoint, where=wr, settings=rs,
                                              cohort_csv=COHORT_CSV[cohort], limit=limit)))

    # Qwen on verified Messidor-2.
    qwen = []
    for pair in PAIRS:
        run = final01_train(16, "on", pair, {"lr": LR_SELECTED})
        qwen.append((run.name, run / "best_lora", "messidor2_on", run / "aptos_test.predictions.csv"))
    for condition, cohort in (("r16_on", "messidor2_on"), ("r16_off", "messidor2_off")):
        for pair in PAIRS:
            run = final.RUNS / remaining.qwen_run_name(condition, pair, "duplicate_excluded", qs["run_tag"])
            qwen.append((f"{condition}_p{pair}", run / "best_lora", cohort, None))
    qwen += [("zeroshot_4bit", None, "messidor2_on", None), ("zeroshot_4bit", None, "messidor2_off", None)]
    for label, adapter, cohort, _ in (qwen[:1] + qwen[-1:] if pilot else qwen):
        job = final.score_job("messidor2", label, cohort, base, where=wq, adapter=adapter, limit=limit,
                              needs=None if adapter is None else adapter / "adapter_config.json")
        job.minutes = final.MINUTES["pilot_score"] if pilot else MESSIDOR2_MINUTES
        jobs.append(("qwen", job))

    # Qwen timing for the configurations with no warm per-image record (GAPS.md), one seed each.
    timing = []
    for condition in ("r16_off", "r8_off", "r32_off"):
        run = final.RUNS / remaining.qwen_run_name(condition, 1, "duplicate_excluded", qs["run_tag"])
        timing.append((f"timing_{condition}_p1", run, f"aptos361_{condition.split('_')[1]}"))
    run = final.RUNS / remaining.qwen_run_name("r16_on", 1, "previous_split", qs["run_tag"])
    timing.append(("timing_r16_on_previous_split_p1", run, "aptos361_on"))
    for overrides in ({"lr": 1e-4}, {"lr": 2e-4}, {"lora_alpha": 8}):
        run = final01_train(16, "on", 1, overrides)
        timing.append((f"timing_{run.name}", run, "aptos361_on"))
    for label, run, cohort in (timing[:1] if pilot else timing):
        job = final.score_job("timing", label, cohort, base, where=wq, adapter=run / "best_lora",
                              generation=6 if pilot else -1, reference=run / "aptos_test.predictions.csv",
                              limit=limit, needs=run / "best_lora" / "adapter_config.json",
                              note="per-image timing; must reproduce the run's saved predictions")
        job.minutes = final.MINUTES["pilot_score"] if pilot else 16
        jobs.append(("qwen", job))

    # Optional rank sweep at the validation-selected learning rate.
    if rank_sweep:
        sweep = [(r, c, p, {"lr": LR_SELECTED}) for r in (8, 16, 32) for c in ("on", "off") for p in PAIRS
                 if not (r == 16 and c == "on")]                      # rank 16 CLAHE-on lr2e-4 exists in final_01
        sweep += [(r, "on", p, {"lr": LR_SELECTED, "lora_alpha": r}) for r in (8, 32) for p in PAIRS]
        if pilot:
            sweep = sweep[-1:]
        for rank, clahe, pair, overrides in sweep:
            jobs.append(("qwen", final.train_job("rank_sweep", rank, clahe, pair, overrides, where=wq, pilot=pilot)))
    return jobs


INPUT_FLAGS = ("--cohort-csv", "--adapter", "--checkpoint", "--reference-predictions", "--base")


def missing_inputs(job) -> list[str]:
    """Every input file or folder a job names on its command line that does not exist yet."""
    args = [str(a) for a in job.arguments]
    named = [args[i + 1] for i, a in enumerate(args[:-1]) if a in INPUT_FLAGS]
    if job.kind == "train":
        named += [args[args.index("--candidate") + 1]] if "--candidate" in args else []
    extra = [str(job.needs)] if getattr(job, "needs", None) and not isinstance(job.needs, list) else []
    return [p for p in named + extra if not Path(p).exists()]


def state_of(job) -> str:
    state = job.status()
    if state in ("pending", "interrupted") and missing_inputs(job):
        return "waiting"
    return state


def show(jobs):
    left = 0
    for queue, job in jobs:
        state = state_of(job)
        if state != "done":
            left += job.minutes
        missing = ""
        if state == "waiting":
            needs = (job.missing() if queue == "retizero" else []) or missing_inputs(job)
            missing = f"  (needs {needs[0]})"
        print(f"  [{state:<11}] {queue:<8} {job.stage:<10} {job.name}  ~{job.minutes} min{missing}")
    done = sum(job.status() == "done" for _, job in jobs)
    print(f"{done} of {len(jobs)} jobs done. GPU time left: about {left / 60:.1f} h (estimate).")
    return done, len(jobs)


class PowerLog:
    """nvidia-smi samples of the whole card every 5 s for as long as the queue runs."""

    def __init__(self, path: Path):
        self.path, self.process = path, None

    def __enter__(self):
        exe = shutil.which("nvidia-smi")
        if exe is None:
            print("nvidia-smi not found: power is not recorded.", flush=True)
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_name(f"{self.path.stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv")
        self.process = subprocess.Popen([exe, f"--query-gpu={POWER_FIELDS}", "--format=csv", "-l", "5", "-f", str(part)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Power log: {part}", flush=True)
        return self

    def __exit__(self, *exc):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def run_jobs(jobs, state_path: Path):
    history = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else []
    outcomes = {}
    with PowerLog(state_path.with_name("power_log.csv")):
        for position, (queue, job) in enumerate(jobs, 1):
            state = state_of(job)
            if state == "done":
                outcomes[job.name] = "done earlier"
                continue
            if state == "waiting":
                outcomes[job.name] = "skipped: an input is missing"
                print(f"[{position}/{len(jobs)}] Skipping {job.name}: an input is missing.", flush=True)
                continue
            if not final.lock_is_free():
                outcomes[job.name] = "not started: GPU lock held by another job"
                print("Another job holds the GPU lock. Stopping here; rerun when it has finished.", flush=True)
                break
            print(f"\n[{position}/{len(jobs)}] {queue} {job.stage}: {job.name}", flush=True)
            record = {"queue": queue, "job": job.name, "stage": job.stage, "output": str(job.output),
                      "started": datetime.now(timezone.utc).isoformat()}
            try:
                (rz.run_job if queue == "retizero" else final.run_job)(job)
                outcome = "passed" if job.status() == "done" else "finished without a completion marker"
            except KeyboardInterrupt:
                record.update(ended=datetime.now(timezone.utc).isoformat(), outcome="interrupted by user")
                history.append(record)
                state_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
                raise
            except Exception as error:  # keep the queue moving; the failure is recorded
                outcome = f"failed: {error}"
                print(traceback.format_exc(), flush=True)
            record.update(ended=datetime.now(timezone.utc).isoformat(), outcome=outcome)
            history.append(record)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            outcomes[job.name] = outcome
    print("\nQueue finished.")
    for name, outcome in outcomes.items():
        print(f"  {outcome:<34} {name}")
    return outcomes


# --------------------------------------------------------------------------- steps

def check(args) -> bool:
    ok = True

    def need(label, condition, hint=""):
        nonlocal ok
        print(f"  [{'ok' if condition else 'MISSING':<7}] {label}{'' if condition else '  -> ' + hint}")
        ok &= bool(condition)

    print(f"Repository: {REPO}\nExperiments folder: {EXPERIMENTS}")
    need("Python 3.11", sys.version_info[:2] == (3, 11), "activate afterimage_env")
    need("experiments folder", EXPERIMENTS.is_dir(), "run from the checkout next to D:\\Afterimage_Experiments")
    need("launcher code (experiments\\code)", (EXPERIMENTS / "code/qwen_pilot.py").is_file()
         and (EXPERIMENTS / "code/notebook_runner.py").is_file(), "copy external/Afterimage_Experiments/code")
    need("Qwen 4-bit base", (final.reproduction.BASE / "config.json").is_file(), "see README prerequisites")
    need("nvidia-smi on PATH", shutil.which("nvidia-smi") is not None, "install the NVIDIA driver")
    for name in ("scipy", "torch", "unsloth"):
        need(f"package {name}", importlib.util.find_spec(name) is not None, f"python -m pip install {name}")
    try:
        lock_free = final.lock_is_free()
    except Exception as error:  # the launcher code is missing; reported above
        lock_free = False
        print(f"    (could not test the GPU lock: {error})")
    need("GPU lock free", lock_free, "wait for the running job to finish")
    if args.rank_sweep:
        need("legacy project for 4-bit training (C: copy)", remaining.LEGACY.is_dir(), str(remaining.LEGACY))
    free = shutil.disk_usage(EXPERIMENTS.anchor if EXPERIMENTS.exists() else REPO.anchor).free / 2**30
    need(f"free disk {free:.0f} GiB (need 10, or 40 with --rank-sweep)", free > (40 if args.rank_sweep else 10), "free some space")
    built = all(p.is_file() for p in COHORT_CSV.values())
    print(f"  [{'ok' if built else 'later':<7}] verified Messidor-2 cohort at {MESSIDOR2}"
          + ("" if built else "  -> built by the build-messidor step"))
    print("\nPlanned jobs:")
    try:
        show(build_jobs(rank_sweep=args.rank_sweep))
    except Exception as error:
        need("queue can be built", False, str(error))
    print("\nCheck passed." if ok else "\nCheck FAILED: fix the items marked MISSING, then run again.")
    return ok


def build_messidor(args) -> bool:
    if all(p.is_file() for p in COHORT_CSV.values()) and args.messidor_package is None:
        print(f"Verified Messidor-2 cohort already built at {MESSIDOR2}.")
        return True
    if args.messidor_package is None:
        print("Give --messidor-package <folder that holds 01_raw_images, 02_adjudicated_grades, 03_eye_pairing>.")
        return False
    package = args.messidor_package
    command = [sys.executable, "-B", str(HERE / "build_messidor2_cohort.py"),
               "--raw-dir", str(package / "01_raw_images"),
               "--grades", str(package / "02_adjudicated_grades/messidor_data.csv"),
               "--pairs", str(package / "03_eye_pairing/pairs.csv"), "--out", str(MESSIDOR2)]
    archive = Path(r"D:\recovered\DR_Aptos_Messidor_Only_Full_Dataset\Res_512\Aptos_messidor_external")
    if archive.is_dir():
        command += ["--archive-dir", str(archive)]
    return subprocess.run(command).returncode == 0


def pilot(args) -> bool:
    jobs = build_jobs(pilot=True, rank_sweep=args.rank_sweep)
    print("Pilot queue:")
    show(jobs)
    run_jobs(jobs, qwen_root(pilot=True) / "revision_queue_state.json")
    done, total = show(jobs)
    passed = done == total
    if passed:
        for queue, job in jobs:  # the timing pilot must reproduce the saved predictions of its run
            if job.stage == "timing":
                agreement = json.loads((job.output / "summary.json").read_text(encoding="utf-8")).get("agreement_with_saved_predictions", {})
                if agreement.get("grade_mismatches"):
                    print(f"Pilot: {job.name} disagrees with the saved predictions on {agreement['grade_mismatches']} images.")
                    passed = False
    print("\nPilot passed. The full queue can be started." if passed else "\nPilot FAILED: read the log above before the full run.")
    return passed


def run(args) -> bool:
    jobs = build_jobs(rank_sweep=args.rank_sweep)
    print("Full queue:")
    show(jobs)
    outcomes = run_jobs(jobs, qwen_root() / "revision_queue_state.json")
    return all(o.startswith(("passed", "done earlier")) for o in outcomes.values())


def energy_per_job(state_path: Path) -> Path | None:
    logs = sorted(state_path.parent.glob("power_log_*.csv"))
    if not state_path.is_file() or not logs:
        return None
    import pandas as pd
    samples = []
    for path in logs:
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [c.split(" [")[0].strip() for c in frame.columns]
        frame["t"] = pd.to_datetime(frame["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
        frame["w"] = pd.to_numeric(frame["power.draw"].astype(str).str.replace(" W", "", regex=False), errors="coerce")
        samples.append(frame[["t", "w"]])
    power = pd.concat(samples).dropna().sort_values("t")
    local_offset = datetime.now().astimezone().utcoffset()
    rows = []
    for record in json.loads(state_path.read_text(encoding="utf-8")):
        if "ended" not in record:
            continue
        start = pd.Timestamp(record["started"]).tz_convert(None) + local_offset  # nvidia-smi writes local time
        end = pd.Timestamp(record["ended"]).tz_convert(None) + local_offset
        window = power[(power["t"] >= start) & (power["t"] <= end)]
        seconds = (end - start).total_seconds()
        rows.append({"queue": record["queue"], "job": record["job"], "outcome": record["outcome"], "seconds": seconds,
                     "samples": len(window), "mean_w": window["w"].mean() if len(window) else None,
                     "energy_wh": window["w"].mean() * seconds / 3600 if len(window) else None})
    out = state_path.with_name("energy_per_job.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def summarize(args) -> bool:
    jobs = build_jobs(rank_sweep=args.rank_sweep)
    show(jobs)
    energy = energy_per_job(qwen_root() / "revision_queue_state.json")
    if energy:
        print(f"Energy per job: {energy}")
    if args.messidor_package is None:
        print("\nAdd --messidor-package to also rebuild the analysis tables with the Messidor-2 grades.")
        return True
    package = args.messidor_package
    tables = FINETUNING / "paper_tables"
    audits = tables / "output/revision_analysis/image_audits"
    command = [sys.executable, "-I", "-B", str(tables / "build_revision_analysis.py"),
               "--experiments-root", str(EXPERIMENTS),
               "--messidor-grades", str(package / "02_adjudicated_grades/messidor_data.csv"),
               "--messidor-pairs", str(package / "03_eye_pairing/pairs.csv"),
               "--messidor-duplicates", str(audits / "messidor_duplicate_pairs.csv"),
               "--near-duplicates", str(audits / "near_duplicate_test_ids.csv"),
               "--original-split", str(EXPERIMENTS / "reference/aptos_split")]
    print("\nRebuilding the analysis tables:\n  " + " ".join(command))
    return subprocess.run(command).returncode == 0


def status(args) -> bool:
    print("Pilot:")
    show(build_jobs(pilot=True, rank_sweep=args.rank_sweep))
    print("\nFull queue:")
    show(build_jobs(rank_sweep=args.rank_sweep))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("step", choices=("all", "check", "status", "build-messidor", "pilot", "run", "summarize"))
    ap.add_argument("--messidor-package", type=Path, default=None,
                    help="the unzipped Messidor-2_for_teammate folder (01_raw_images, 02_adjudicated_grades, 03_eye_pairing)")
    ap.add_argument("--rank-sweep", action="store_true", help="also train the optional rank sweep at lr 2e-4 (about 30 h)")
    ap.add_argument("--skip-pilot", action="store_true", help="for `all`: skip the pilot (only if it already passed)")
    args = ap.parse_args(argv)
    steps = {"check": check, "status": status, "build-messidor": build_messidor, "pilot": pilot, "run": run, "summarize": summarize}
    if args.step != "all":
        return 0 if steps[args.step](args) else 1
    order = ["check", "build-messidor"] + ([] if args.skip_pilot else ["pilot"]) + ["run", "summarize"]
    for name in order:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        if not steps[name](args):  # check tolerates a missing Messidor-2 build: the next step makes it
            print(f"Stopping after `{name}`. Fix the problem above and run the same command again; finished jobs are kept.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
