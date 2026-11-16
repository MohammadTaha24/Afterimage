"""Train and score one 4-bit Qwen variant through the pinned runner.

The pinned runner (D:/Afterimage_Experiments/code/qwen_pilot.py) and worker are
used unchanged. This wrapper only edits the saved run configuration before the
first stage starts, records each override, and relaxes the worker's adapter
check for the two settings it hard-codes (alpha 16, dropout 0.05) so that a
declared alpha or dropout variant can pass. Nothing else differs from a run
started by the pinned runner.
"""

import argparse
import hashlib
import importlib.util
import json
import runpy
from pathlib import Path
from types import FunctionType, SimpleNamespace


SCRIPT = Path(__file__).resolve()
DEFAULT_RUNNER = Path(r"D:\Afterimage_Experiments\code\qwen_pilot.py")
OVERRIDES = ("lr", "lora_alpha", "lora_dropout", "early_stopping_patience")
PINNED_ADAPTER = {"lora_alpha": 16, "lora_dropout": 0.05}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def variant_adapter_check(pinned_check, failure):
    """Check declared alpha/dropout here, then let the pinned check do the rest."""

    def check(actual, config):
        shown = dict(actual)
        for key, pinned in PINNED_ADAPTER.items():
            declared = config["training"][key]
            if actual.get(key) != declared:
                raise failure(f"Adapter setting {key} must be {declared!r}; found {actual.get(key)!r}")
            shown[key] = pinned
        pinned_check(shown, config)

    return check


def load_runner(path):
    spec = importlib.util.spec_from_file_location("afterimage_pinned_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    pinned_loader = runner.load_worker

    def load_worker(worker_path):
        worker = pinned_loader(worker_path)
        worker.adapter_config_matches = variant_adapter_check(worker.adapter_config_matches,
                                                              worker.ConditionFailure)
        return worker

    runner.load_worker = load_worker
    return runner


def apply_overrides(config, overrides, name, runner):
    training = config["training"]
    unknown = sorted(set(overrides) - set(OVERRIDES))
    if unknown or any(key not in training for key in overrides):
        raise ValueError(f"Unsupported override: {unknown or sorted(overrides)}")
    config["variant"] = {
        "name": name,
        "overrides": dict(overrides),
        "pinned_values": {key: training[key] for key in overrides},
        "variant_runner_sha256": sha(SCRIPT),
    }
    training.update(overrides)
    if overrides:
        config["purpose"] = f"{config['purpose']} (variant {name})"
    config["identity"] = runner.identity(config)
    return config


def logged_runner(runner):
    # Same reporter and checkpoint protocol; stage subprocesses come back to this file.
    pinned = runner.run_logged
    return FunctionType(pinned.__code__, {**vars(runner), "__file__": str(SCRIPT)},
                        pinned.__name__, pinned.__defaults__)


def load_run(output, runner, runner_path):
    config = json.loads((output / "pilot_configuration.json").read_text(encoding="utf-8"))
    if runner.identity(config) != config.get("identity"):
        raise ValueError("Run configuration changed or lacks a saved identity")
    if config.get("variant", {}).get("variant_runner_sha256") != sha(SCRIPT):
        raise ValueError("The variant runner differs from the one that prepared this run")
    worker = runner.load_worker(Path(config["worker"]))
    if worker.sha(runner_path) != config["runner_sha256"]:
        raise ValueError("The pinned runner differs from the saved setup")
    runner.verify_base(Path(config["base"]), config["base_files"], worker)
    return config, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--lock-dir", type=Path, help="Folder holding the shared .gpu.lock")
    parser.add_argument("--project", type=Path, help="Reference project read by the pinned runner")
    parser.add_argument("--candidate", type=Path, help="512-pixel APTOS dataset manifest")
    parser.add_argument("--base-snapshot", type=Path)
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--clahe", choices=("on", "off"), default="on")
    parser.add_argument("--trainer-seed", type=int)
    parser.add_argument("--adapter-seed", type=int)
    parser.add_argument("--full", action="store_true", help="Train the whole condition and score the APTOS test set")
    parser.add_argument("--steps", type=int, default=10, help="Optimizer steps for a pilot (1-10)")
    parser.add_argument("--variant-name", default="pinned")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--stage", choices=("train", "score"), help=argparse.SUPPRESS)
    parser.add_argument("--resume-from", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    runner_path = args.runner.resolve(strict=True)
    runner = load_runner(runner_path)
    output = args.output.resolve()
    if args.stage:
        runner.run_model_stage(output, args.stage, args.resume_from)
        return

    lock_dir = (args.lock_dir or output.parent).resolve()
    lock_dir.mkdir(parents=True, exist_ok=True)
    logged = logged_runner(runner)
    with runner.gpu_lock(lock_dir):
        if (output / "pilot_configuration.json").is_file():
            config, worker = load_run(output, runner, runner_path)
            recover = runpy.run_path(str(runner_path.with_name("recover_qwen.py")))["recover"]
            recover(output, config, worker, {**vars(runner), "run_logged": logged})
            return
        if output.exists():
            parser.error(f"{output} exists without a saved configuration. Choose a fresh folder.")
        missing = [name for name in ("project", "candidate", "trainer_seed", "adapter_seed")
                   if getattr(args, name) is None]
        if missing or not 1 <= args.steps <= 10:
            parser.error(f"A new run needs --{', --'.join(missing) or 'steps between 1 and 10'}")
        overrides = {key: value for key, value in (
            ("lr", args.lr), ("lora_alpha", args.lora_alpha), ("lora_dropout", args.lora_dropout),
            ("early_stopping_patience", args.early_stopping_patience)) if value is not None}
        request = SimpleNamespace(
            project=args.project, candidate=args.candidate, base_snapshot=args.base_snapshot,
            historical_split=False, comparison_manifest=None, rank=args.rank, clahe=args.clahe,
            full=args.full, steps=args.steps, trainer_seed=args.trainer_seed, adapter_seed=args.adapter_seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        config = runner.prepare(request, output)
        config = apply_overrides(config, overrides, args.variant_name, runner)
        runner.write_json(output / "pilot_configuration.json", config)
        print(f"Prepared {config['rows']} APTOS images in {output}", flush=True)
        print(f"Variant {args.variant_name}: overrides {overrides or 'none'}", flush=True)
        logged(output, config, ("train", "score"))


if __name__ == "__main__":
    main()
