"""Recover a saved Qwen run using the runner that created its configuration."""

import argparse
import json
import runpy
import tempfile
from pathlib import Path


def preserve_partial(output, paths):
    if not paths:
        return
    archive = output / "recovery_artifacts"
    if archive.resolve().parent != output:
        raise ValueError("Recovery archive must be inside the run folder")
    if any(path.resolve().parent != output for path in paths):
        raise ValueError("Partial artifacts must be inside the run folder")
    archive.mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="partial-", dir=archive))
    for path in paths:
        path.rename(destination / path.name)
    print(f"Preserved partial files in {destination}", flush=True)


def recover(output, config, worker, runner):
    if (output / "train_result.json").is_file():
        trained = runner["trained_adapter"](output, config, worker)
        scores = output / "score_result.json"
        full = config["mode"] == "full"
        predictions = output / ("aptos_test.predictions.csv" if full else "pilot_validation.predictions.csv")
        latest = output / "latest_attempt.json"
        if all(path.is_file() for path in (scores, predictions, latest)):
            result = worker.read_json(scores)
            attempt = worker.read_json(latest)
            if (result.get("configuration_identity") == config.get("identity")
                    and result.get("adapter") == trained["adapter"]
                    and attempt.get("status") == "passed"):
                if full:
                    valid = result.get("predictions_sha256") == worker.sha(predictions)
                else:
                    saved = Path(attempt["path"]).resolve()
                    archived = saved / predictions.name
                    valid = (saved.parent == (output / "attempts").resolve()
                             and archived.is_file() and worker.sha(archived) == worker.sha(predictions))
                if valid:
                    print(f"Training and scoring already verified: {output}", flush=True)
                    return
        print("Training is saved. Starting scoring in a new attempt.", flush=True)
        runner["run_logged"](output, config, ("score",))
        return

    checkpoints = output / "checkpoints"
    adapter = output / "best_lora"
    if config["mode"] == "full" and list(checkpoints.glob("checkpoint-*/checkpoint_identity.json")):
        checkpoint = runner["checked_resume"](output, config, worker)
        if adapter.exists():
            preserve_partial(output, [adapter])
        runner["run_logged"](output, config, ("train", "score"), checkpoint)
        return

    if (output / "score_result.json").exists():
        raise ValueError("Scores exist without a matching training result")
    preserve_partial(output, [path for path in (checkpoints, adapter) if path.exists()])
    print("No resumable epoch checkpoint. Starting again from the base in a new attempt.", flush=True)
    runner["run_logged"](output, config, ("train", "score"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    script = Path(__file__).resolve().with_name("qwen_pilot.py")
    runner = runpy.run_path(str(script))
    with runner["gpu_lock"](output.parent):
        config_path = output / "pilot_configuration.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if runner["identity"](config) != config.get("identity"):
            raise ValueError("Run configuration changed or lacks a saved identity")
        worker = runner["load_worker"](Path(config["worker"]))
        if worker.sha(script) != config["runner_sha256"]:
            raise ValueError("The runner differs from the saved setup")
        runner["verify_base"](Path(config["base"]), config["base_files"], worker)
        recover(output, config, worker, runner)


if __name__ == "__main__":
    main()
