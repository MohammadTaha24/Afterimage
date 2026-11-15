"""Entry points for the remaining paper experiments."""

from pathlib import Path
import re
import tempfile

import reproduce_run_02 as reproduction

ROOT = reproduction.ROOT
EXPERIMENTS = reproduction.EXPERIMENTS
RUNS = EXPERIMENTS / "runs"
DATA = EXPERIMENTS / "datasets"
LEGACY = Path(r"C:\Users\moht2\Documents\Afterimage")
SEEDS = {1: (42, 3407), 2: (43, 3408), 3: (44, 3409)}
QWEN_CASES = (("r8_off", "duplicate_excluded"), ("r32_off", "duplicate_excluded"),
              ("r16_on", "previous_split"))


def valid_tag(tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag):
        raise ValueError("Use a tag with letters, digits, hyphens or underscores.")
    return tag


def qwen_run_name(condition, seed_pair, split, tag):
    trainer, adapter = SEEDS[seed_pair]
    return f"full_{condition}_{split}_t{trainer}_a{adapter}_{valid_tag(tag)}"


def bf16_run_name(seed_pair, control_tag):
    trainer, adapter = SEEDS[seed_pair]
    return f"full_bf16_r16_on_duplicate_excluded_t{trainer}_a{adapter}_{valid_tag(control_tag)}"


def qwen_remaining(condition, seed_pair, split, tag):
    if (condition, split) not in QWEN_CASES:
        raise ValueError("Choose a remaining Qwen case.")
    launch, _ = reproduction.runtime()
    name = qwen_run_name(condition, seed_pair, split, tag)
    run = RUNS / name
    logs = EXPERIMENTS / "logs/training" / name
    trainer, adapter = SEEDS[seed_pair]
    rank, clahe = condition.split("_")
    rank = int(rank[1:])
    manifest = DATA / f"aptos_512/{split}/manifest.json"
    historical = split == "previous_split"
    comparison = DATA / "aptos_512/duplicate_excluded/manifest.json"
    print(f"Run folder: {run}", flush=True)
    if run.exists():
        cfg = reproduction.read(run / "pilot_configuration.json")
        expected = ("full", rank, clahe == "on", trainer, adapter, manifest.resolve(),
                    historical, comparison.resolve() if historical else None, reproduction.BASE.resolve())
        actual = (cfg["mode"], cfg["rank"], cfg["clahe"], cfg["training"]["trainer_seed"],
                  cfg["adapter_seed"], Path(cfg["candidate_manifest"]).resolve(), cfg["historical_split"],
                  Path(cfg["comparison_manifest"]).resolve() if cfg["comparison_manifest"] else None,
                  Path(cfg["base"]).resolve())
        if actual != expected:
            raise ValueError(f"Saved configuration differs: {run}")
        launch(EXPERIMENTS / "code/recover_qwen.py", ["--output", run], logs / "resume.log",
               keep_pc_awake=True)
        return

    args = ["--project", LEGACY, "--candidate", manifest, "--base-snapshot", reproduction.BASE,
            "--rank", rank, "--clahe", clahe, "--trainer-seed", trainer, "--adapter-seed", adapter,
            "--full", "--output", run]
    if historical:
        args += ["--historical-split", "--comparison-manifest", comparison]
    launch(EXPERIMENTS / "code/qwen_pilot.py", args, logs / "launch.log", keep_pc_awake=True)


def score_clahe_on_pair1(tag, build_tag):
    """Recover the remaining BF16 score only after verifying saved training."""
    run = reproduction.folder("training_batches", tag) / "clahe_on_01"
    config = reproduction.matched_config("on", build_tag)
    _, monitor = reproduction.runtime()
    with monitor["gpu_lock"](RUNS):
        if not reproduction.original.completed_training(run, reproduction.read(config)):
            raise ValueError(f"Missing completed training: {run}")
        (run / "attempts").mkdir(parents=True, exist_ok=True)
        reproduction.score(run, run / "best_lora", config, external=False, clahe=True)


def qwen_bf16(mode, seed_pair, run_tag, base, *, control_tag, offload_layers=0):
    if mode not in {"check", "run"}:
        raise ValueError("Choose check or run.")
    launch, _ = reproduction.runtime()
    reference = RUNS / qwen_run_name("r16_on", seed_pair, "duplicate_excluded", run_tag)
    name = bf16_run_name(seed_pair, control_tag)
    output = RUNS / name
    entry = reproduction.QWEN / "train_bf16_control.py"
    if mode == "check":
        args = ["check", "--reference-run", reference]
    elif output.exists() and (output / "pilot_configuration.json").is_file():
        args = ["resume", "--output", output]
    else:
        args = ["train", "--reference-run", reference, "--output", output]
    if base and args[0] in {"check", "train"}:
        args += ["--base-snapshot", base]
    args += ["--offload-layers", offload_layers]
    if mode == "run":
        args.append("--report-blocked")
    launch(entry, args, EXPERIMENTS / "logs/training" / name / f"{args[0]}.log",
           keep_pc_awake=mode == "run")


def bf16_shared_memory_pilot(base, run_tag, *, offload_layers=0):
    base = Path(base)
    index = base / "model.safetensors.index.json"
    if not index.is_file():
        print(f"BF16 pilot is blocked. The official model snapshot is needed at {base}.")
        return
    shards = set(reproduction.read(index)["weight_map"].values())
    missing = sorted(name for name in shards if not (base / name).is_file())
    if missing:
        print(f"BF16 pilot is waiting for {len(missing)} model weight shard(s) in {base}.")
        return
    reference = RUNS / qwen_run_name("r16_on", 1, "duplicate_excluded", run_tag)
    _, monitor = reproduction.runtime()
    with monitor["gpu_lock"](RUNS):
        output = Path(tempfile.mkdtemp(prefix="bf16-shared-memory-", dir=EXPERIMENTS / "checks"))
        (output / "attempts").mkdir()
        print(f"Feasibility pilot (two updates). Output: {output}", flush=True)
        reproduction.run_stage("train", ["--base-snapshot", base, "--reference-run", reference, "--output", output,
                                          "--offload-layers", offload_layers],
                               output, script=reproduction.QWEN / "probe_bf16_training.py")


def retizero(mode, seed_pair, control_tag, weights, weights_sha256, *, external_csv=None):
    if mode not in {"check", "run", "external"}:
        raise ValueError("Choose check, run or external.")
    seed, _ = SEEDS[seed_pair]
    entry = reproduction.QWEN / "train_retizero_control.py"
    run = reproduction.folder("training_batches", control_tag) / f"retizero_t{seed}"
    args = ["--repo-root", ROOT, "--dataset", DATA / "aptos_224_standard/duplicate_excluded/manifest.json",
            "--qwen-dataset", DATA / "aptos_512/duplicate_excluded/manifest.json",
            "--weights", weights, "--weights-sha256", weights_sha256, "--run-dir", run, "--seed", seed]
    launch, monitor = reproduction.runtime()
    if mode == "check":
        launch(entry, ["check", *args], reproduction.ARTIFACTS / "checks/retizero.log")
        return
    if mode == "external":
        args += ["--external-csv", external_csv, "--external-sha256", reproduction.sha(external_csv)]
    with monitor["gpu_lock"](RUNS):
        if mode == "external" and not run.is_dir():
            raise FileNotFoundError(f"Train the RetiZero adapter first: {run}")
        (run / "attempts").mkdir(parents=True, exist_ok=True)
        if mode == "run":
            reproduction.run_stage("train", args, run, script=entry)
        reproduction.run_stage("evaluate", args, run, script=entry)


def check_messidor(manifests):
    from evaluate_verified_messidor import verified_rows
    for key, csv_path in manifests.items():
        if not csv_path.is_file():
            raise FileNotFoundError(f"Prepare the verified {key} cohort CSV first: {csv_path}")
    on, off = (verified_rows(manifests[k]) for k in ("512_on", "512_off"))
    if on != off:
        raise ValueError("The Messidor CLAHE branches differ in image IDs, grades, or pairing IDs.")
    if "224_off" in manifests and on != verified_rows(manifests["224_off"]):
        raise ValueError("The 224-pixel Messidor cohort differs from the Qwen cohort.")
    print(f"Matched {len(on)} Messidor image IDs, grades and pairing IDs.")


def score_messidor_qwen(series, condition, index, manifests, run_tag, historical_tag,
                        control_tag, messidor_tag):
    """Evaluate one saved adapter using its matching image branch."""
    check_messidor({k: manifests[k] for k in ("512_on", "512_off")})
    base = reproduction.BASE
    base_precision = "4bit"
    if series == "original":
        if condition not in {"r16_on", "r16_off"}:
            raise ValueError("The original-split CLAHE series uses rank 16.")
        run = reproduction.folder("training_batches", historical_tag) / f"clahe_{condition.split('_')[1]}_{index:02}"
        cfg = reproduction.read(run / "configuration.json")
        if not reproduction.original.completed_training(run, cfg):
            raise ValueError(f"Incomplete adapter training: {run}")
        precision = "bf16"
    elif series in {"duplicate_excluded", "previous_split", "bf16_base"}:
        if condition not in {f"r{rank}_{mode}" for rank in (8, 16, 32) for mode in ("on", "off")}:
            raise ValueError("Choose a rank and CLAHE branch.")
        if series == "bf16_base":
            if condition != "r16_on":
                raise ValueError("The BF16-base control uses rank 16 and CLAHE on.")
            run = RUNS / bf16_run_name(index, control_tag)
            base_precision = "bf16"
        else:
            run = RUNS / qwen_run_name(condition, index, series, run_tag)
        cfg = reproduction.read(run / "pilot_configuration.json")
        _, pinned = reproduction.runtime()
        worker = pinned["load_worker"](Path(cfg["worker"]))
        if series == "bf16_base":
            worker.BASE_NAME = cfg["bf16_model_name"]
        pinned["trained_adapter"](run, cfg, worker)
        base = Path(cfg["base"])
        precision = "fp16"
    else:
        raise ValueError("Choose original, duplicate_excluded, previous_split, or bf16_base.")
    eval_run = reproduction.folder("external_evaluation", messidor_tag) / run.name
    cohort = manifests["512_" + condition.split("_")[1]]
    arguments = ["--adapter", run / "best_lora", "--base-snapshot", base, "--csv-path", cohort,
                 "--precision", precision, "--base-precision", base_precision,
                 "--output-dir", eval_run / "evaluation"]
    if base_precision == "bf16":
        arguments += ["--tokenizer", cfg["reference_base"]]
    _, monitor = reproduction.runtime()
    with monitor["gpu_lock"](RUNS):
        (eval_run / "attempts").mkdir(parents=True, exist_ok=True)
        reproduction.run_stage("evaluate", arguments, eval_run,
                               script=reproduction.QWEN / "evaluate_verified_messidor.py")


def show_checklist(run_tag, historical_tag, control_tag):
    batch = reproduction.folder("training_batches", historical_tag)
    control = reproduction.folder("training_batches", control_tag)

    def evaluation_status(path):
        return "scored" if path.is_file() else "pending"

    score = batch / "clahe_on_01/evaluation_bf16/aptos_test.summary.json"
    print(f"Pair 1 CLAHE-on, BF16 scoring: {evaluation_status(score)}")
    for condition, split in QWEN_CASES:
        for seed_pair in SEEDS:
            name = qwen_run_name(condition, seed_pair, split, run_tag)
            run = RUNS / name
            status = "pending"
            marker = run / "latest_attempt.json"
            if marker.is_file():
                status = reproduction.read(marker)["status"]
                if status == "running":
                    status += " (last saved marker; check live logs)"
                if status == "passed":
                    score_path = run / "score_result.json"
                    if score_path.is_file():
                        status = format_metrics(reproduction.read(score_path)["metrics"])
                    else:
                        status = "completion marker present, score missing"
            print(f"{condition} {split}, seed pair {seed_pair}: {status}")
    for seed_pair, (trainer, _) in SEEDS.items():
        prefix = RUNS / bf16_run_name(seed_pair, control_tag)
        print(f"BF16 base, seed pair {seed_pair}: {evaluation_status(prefix / 'score_result.json')}")
        print(f"RetiZero, seed {trainer}: {evaluation_status(control / f'retizero_t{trainer}/evaluation/aptos_test.summary.json')}")
        i = seed_pair
        print(f"Deferred historical repeat {i}: {evaluation_status(batch / f'historical_{i:02}/evaluation/aptos_test.summary.json')}")


def format_metrics(metrics):
    return (f"scored | accuracy {metrics['accuracy']:.2%}, QWK {metrics['qwk']:.4f}, "
            f"AUC {metrics['macro_ovr_auc']:.4f}")
