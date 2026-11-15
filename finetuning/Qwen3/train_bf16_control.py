"""Train or score rank-16 LoRA on an unquantized Qwen3-VL-8B-Instruct base."""

import argparse
import csv
import hashlib
import json
import math
import os
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import FunctionType


SCRIPT = Path(__file__).resolve()
DEFAULT_RUNNER = Path(r"D:\Afterimage_Experiments\code\qwen_pilot.py")
RUNNER_ENV = "AFTERIMAGE_BF16_REFERENCE_RUNNER"
SEED_PAIRS = {(42, 3407), (43, 3408), (44, 3409)}
GIB = 2**30


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_file(path, digest):
    if sha(path) != digest:
        raise ValueError(f"File differs from the recorded comparison: {path}")


def read_reference(reference, runner_path, runner):
    reference = reference.resolve(strict=True)
    config = read(reference / "pilot_configuration.json")
    if config["identity"] != runner["identity"](config):
        raise ValueError("The 4-bit reference configuration changed.")
    if (config["mode"] != "full" or config["rank"] != 16 or not config["clahe"]
            or config["historical_split"] or config["rows"] != {"train": 2786, "val": 357, "test": 361}
            or not config["training"]["load_in_4bit"]):
        raise ValueError("Choose a full rank-16 CLAHE-on 4-bit run with the duplicate-excluded APTOS split.")
    if (config["training"]["trainer_seed"], config["adapter_seed"]) not in SEED_PAIRS:
        raise ValueError("The reference's training and adapter seeds must match one of the three planned pairs.")
    same_file(runner_path, config["runner_sha256"])
    for name in ("worker", "helper", "trainer", "phase1"):
        same_file(Path(config[name]), config["source_sha256"][name])
    worker = runner["load_worker"](Path(config["worker"]))
    worker.packages(config["package_versions"])
    trained = runner["trained_adapter"](reference, config, worker)
    score = read(reference / "score_result.json")
    if (read(reference / "latest_attempt.json")["status"] != "passed"
            or score["configuration_identity"] != config["identity"]
            or score["adapter"] != trained["adapter"] or score["metrics"]["samples"] != 361):
        raise ValueError("The matching 4-bit reference must have completed training and APTOS scoring.")
    same_file(reference / "aptos_test.predictions.csv", score["predictions_sha256"])
    return config


def bf16_snapshot(path, quantized_config):
    path = path.resolve(strict=True)
    config = read(path / "config.json")
    if config.get("quantization_config") or config.get("model_type") != "qwen3_vl":
        raise ValueError("Provide an unquantized Qwen3-VL-8B-Instruct safetensors snapshot.")
    for key in ("text_config", "vision_config"):
        old = quantized_config[key]
        new = config[key]
        for field in ("hidden_size", "num_hidden_layers", "depth", "vocab_size", "intermediate_size"):
            if new.get(field) != old.get(field):
                raise ValueError(f"Base model architectures differ at {key}.{field}.")
    index = path / "model.safetensors.index.json"
    files = {"model.safetensors"}
    if index.is_file():
        files = set(read(index)["weight_map"].values())
    if not files or any(Path(name).name != name or not name.endswith(".safetensors") for name in files):
        raise ValueError("Invalid model weight shards in the snapshot.")
    bf16_bytes = 0
    for name in sorted(files):
        with (path / name).open("rb") as stream:
            header_length, = struct.unpack("<Q", stream.read(8))
            if not 0 < header_length <= 100_000_000:
                raise ValueError(f"Invalid safetensors header: {name}")
            header = json.loads(stream.read(header_length))
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            if tensor["dtype"] == "BF16":
                start, end = tensor["data_offsets"]
                bf16_bytes += end - start
            if "quant_state" in key or tensor["dtype"] in {"I8", "U8"}:
                raise ValueError(f"Found quantized weights in {name}.")
    if bf16_bytes < 16 * GIB:
        raise ValueError("The snapshot does not contain a complete 8B-class BF16 weight set.")
    records = {}
    for file in sorted(item for item in path.iterdir() if item.is_file() and not item.name.startswith(".")):
        records[file.name] = {"size": file.stat().st_size, "sha256": sha(file)}
    return path, bf16_bytes, records


def gpu_requirements(runner, weight_bytes, offload_layers=0):
    query = runner["gpu_query"]("index,name,memory.total,memory.free")
    rows = list(csv.reader(query.splitlines(), skipinitialspace=True))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and len(rows) > 1:
        rows = [row for row in rows if row[0] == visible]
    if offload_layers:
        # The tested decoder blocks occupy 368 MiB each. Leave room for
        # the prefetch pool, adapters and activations.
        enough = len(rows) == 1 and int(rows[0][3]) >= max(
            10 * 1024, math.ceil(weight_bytes / 2**20) - 368 * offload_layers + 4 * 1024
        )
    else:
        enough = len(rows) == 1 and int(rows[0][2]) * 2**20 >= weight_bytes + 4 * GIB
    return [f"GPU {index}: {name}, {total} MiB total / {free} MiB free"
            for index, name, total, free in rows], enough


def verify_split(run, config, splits):
    paths = {run / "input_images.csv": config["input_images_sha256"]}
    for split in splits:
        paths[run / f"aptos_{split}.csv"] = config["split_sha256"][split]
    for path, digest in paths.items():
        same_file(path, digest)
    with (run / "input_images.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in splits:
                same_file(Path(row["image_path"]), row["sha256"])


def create_run(output, reference, config, snapshot, weight_bytes, files, runner, offload_layers):
    output.mkdir()
    for name in ("input_images.csv", "aptos_train.csv", "aptos_val.csv", "aptos_test.csv"):
        shutil.copyfile(reference / name, output / name)
    config.update(
        purpose="BF16-base LoRA matched to the duplicate-excluded 4-bit Qwen control",
        reference_run=str(reference.resolve()),
        reference_identity=config["identity"],
        reference_base=config["base"],
        reference_base_files=config["base_files"],
        base=str(snapshot),
        base_files=files,
        bf16_model_name="Qwen/Qwen3-VL-8B-Instruct",
        bf16_weight_bytes=weight_bytes,
        offload_layers=offload_layers,
        bf16_worker_sha256=sha(SCRIPT),
    )
    config["training"]["load_in_4bit"] = False
    config["identity"] = runner["identity"](config)
    runner["write_json"](output / "pilot_configuration.json", config)


def load_run(output, runner_path, runner):
    config = read(output / "pilot_configuration.json")
    if (config["identity"] != runner["identity"](config)
            or config.get("bf16_worker_sha256") != sha(SCRIPT)):
        raise ValueError("Run identity or BF16 worker changed.")
    same_file(runner_path, config["runner_sha256"])
    for name in ("worker", "helper", "trainer", "phase1"):
        same_file(Path(config[name]), config["source_sha256"][name])
    worker = runner["load_worker"](Path(config["worker"]))
    worker.BASE_NAME = config["bf16_model_name"]
    worker.packages(config["package_versions"])
    return config, worker


def run_stage(output, stage, resume, runner_path, runner):
    config, worker = load_run(output, runner_path, runner)
    cache = output / "cache" / stage
    worker.offline(cache)
    compiler_temp = cache / "tmp"
    compiler_temp.mkdir(exist_ok=True)
    tempfile.tempdir = str(compiler_temp)
    os.environ.update(TEMP=str(compiler_temp), TMP=str(compiler_temp))
    if stage == "score":
        os.environ.update(UNSLOTH_COMPILE_DISABLE="1", UNSLOTH_DISABLE_COMPILE="1",
                          TORCHINDUCTOR_DISABLE="1")
    import pandas  # Match the 4-bit launcher's Windows import order.
    from unsloth import FastVisionModel
    import torch

    worker.resources(output, 10, gpu=True)
    if resume and runner["checked_resume"](output, config, worker) != resume.resolve():
        raise ValueError("The requested epoch checkpoint is no longer verified.")
    splits = ("train", "val") if stage == "train" else ("test",)
    verify_split(output, config, splits)
    tables = {split: worker.read_csv(output / f"aptos_{split}.csv") for split in splits}
    helper = worker.local_module("qwen_reproduction", config["helper"])
    offload_layers = config["offload_layers"]
    dtype = torch.float16 if stage == "score" and offload_layers else torch.bfloat16
    torch.cuda.reset_peak_memory_stats()
    model, processor = FastVisionModel.from_pretrained(
        model_name=config["base"], dtype=dtype, load_in_16bit=True,
        load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
        offload_embedding=False, offload_layers=offload_layers,
        local_files_only=True, device_map={"": 0}, tokenizer_name=config["reference_base"],
    )
    if (getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_quantized", False) or getattr(model, "peft_config", None)):
        raise ValueError("Expected a fresh unquantized base without an attached adapter.")
    parked = {}
    if offload_layers:
        swapper = getattr(model, "_unsloth_block_swap", None)
        if swapper is None or len(swapper.indices) != offload_layers:
            raise ValueError("The saved layer offload setting did not initialize.")
        parked = {id(param): host for block in swapper.blocks
                  for param, host in zip(block.params, block.host, strict=True)}
        print(f"Offloaded {offload_layers} layers using {swapper.host_bytes() / GIB:.2f} GiB of RAM.", flush=True)
    resident = [param for param in model.parameters() if id(param) not in parked]
    loaded = sum(param.numel() for param in (*resident, *parked.values()) if param.dtype == dtype)
    if loaded * 2 < 16 * GIB or any(param.device.type != "cuda" for param in resident):
        raise ValueError("Expected the complete unquantized base with remaining weights mapped to CUDA.")
    print(f"Verified unquantized base, {loaded:,} {dtype} parameters; starting {stage}.", flush=True)
    if stage == "train":
        result = runner["train"](output, config, worker, helper, model, processor, tables, FastVisionModel, torch, resume)
    elif offload_layers:
        result = score_offloaded(output, config, worker, helper, model, processor, tables["test"], FastVisionModel, torch, runner)
    else:
        result = runner["score"](output, config, worker, helper, model, processor, tables["test"], FastVisionModel, "test")
    result.update(
        base_weights="unquantized BF16",
        offload_layers=offload_layers,
        scoring_precision="FP16",  # Same full-answer scorer and inference dtype as the matched 4-bit reference.
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cuda_runtime=torch.version.cuda,
        configuration_identity=config["identity"],
    )
    worker.write_json(output / f"{stage}_result.json", result)


def score_offloaded(output, cfg, worker, helper, model, processor, rows, vision_model, torch, runner):
    from peft import PeftModel

    trained = runner["trained_adapter"](output, cfg, worker)
    model = PeftModel.from_pretrained(model, str(output / "best_lora"), local_files_only=True,
                                     autocast_adapter_dtype=False)
    vision_model.for_inference(model)
    model.eval()
    # The swap pools were created in FP16 at load. Cast adapter tensors only.
    for name, param in model.named_parameters():
        if "lora_" in name and param.is_floating_point():
            param.data = param.data.to(dtype=torch.float16)
    predictions = []
    for i, row in enumerate(rows, 1):
        scores = [float(value) for value in helper.score_full_answers(model, processor, row["image_path"])]
        if len(scores) != 5 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores) or abs(sum(scores) - 1) > 1e-6:
            raise ValueError(f"Invalid five-grade scores for {row['id_code']}")
        predictions.append({
            "original_image_id": row["id_code"], "image_id": row["id_code"], "image_path": row["image_path"],
            "y_true": int(row["label"]), "y_pred": max(range(5), key=scores.__getitem__),
            **{f"p{grade}": value for grade, value in enumerate(scores)},
        })
        if i == 1 or i % 25 == 0 or i == len(rows):
            print(f"Scored {i}/{len(rows)} APTOS images.", flush=True)
    phase1 = worker.local_module("qwen_phase1_metrics", cfg["phase1"])
    metrics = worker.prediction_metrics(predictions, rows, phase1)
    path = output / "aptos_test.predictions.csv"
    temporary = path.with_suffix(".csv.partial")
    runner["write_csv"](temporary, predictions)
    temporary.replace(path)
    return {"metrics": metrics, "predictions_sha256": worker.sha(path), "adapter": trained["adapter"]}


def launch_logged(output, config, worker, args, runner_path, runner):
    # This copy uses the pinned reporter/checkpoint protocol, with this file as its stage subprocess.
    run_logged = runner["run_logged"]
    logged = FunctionType(run_logged.__code__, {**run_logged.__globals__, "__file__": str(SCRIPT)},
                          run_logged.__name__, run_logged.__defaults__)
    runner["verify_base"](Path(config["base"]), config["base_files"], worker)
    runner["verify_base"](Path(config["reference_base"]), config["reference_base_files"], worker)
    if args.action == "train":
        logged(output, config, ("train", "score"))
    elif args.action == "score":
        runner["trained_adapter"](output, config, worker)
        logged(output, config, ("score",))
    else:
        recover = runpy.run_path(str(runner_path.with_name("recover_qwen.py")))["recover"]
        recover(output, config, worker, {**runner, "run_logged": logged})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train", "resume", "score"), nargs="?")
    parser.add_argument("--base-snapshot", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offload-layers", type=int, choices=range(37), metavar="0-36",
                        help="Frozen decoder layers kept in system RAM. Resume uses the saved setting.")
    parser.add_argument("--runner", type=Path, default=Path(os.environ.get(RUNNER_ENV, DEFAULT_RUNNER)))
    parser.add_argument("--stage", choices=("train", "score"), help=argparse.SUPPRESS)
    parser.add_argument("--resume-from", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--logged", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report-blocked", action="store_true",
                        help="Print missing prerequisites and return control to the notebook.")
    args = parser.parse_args()
    runner_path = args.runner.resolve(strict=True)
    runner = runpy.run_path(str(runner_path))
    output = args.output.resolve() if args.output else None
    if args.stage:
        run_stage(output, args.stage, args.resume_from, runner_path, runner)
        return
    if args.action is None:
        parser.error("Choose check, train, resume or score.")
    if args.action != "check":
        runs = runner_path.parent.parent / "runs"
        if output is None or output.parent != runs.resolve():
            parser.error(f"Choose a run folder directly under {runs}.")

    if args.logged:
        with runner["gpu_lock"](output.parent):
            if args.action == "train":
                config = read_reference(args.reference_run, runner_path, runner)
                quantized = read(Path(config["base"]) / "config.json")
                snapshot, weight_bytes, files = bf16_snapshot(args.base_snapshot, quantized)
                verify_split(args.reference_run, config, ("train", "val", "test"))
                create_run(output, args.reference_run, config, snapshot, weight_bytes, files, runner,
                           args.offload_layers or 0)
            config, worker = load_run(output, runner_path, runner)
            launch_logged(output, config, worker, args, runner_path, runner)
        return

    config = None
    if args.action in {"check", "train"}:
        if args.reference_run is None:
            parser.error("Supply one completed --reference-run for the seed pair.")
        config = read_reference(args.reference_run, runner_path, runner)
    weight_bytes = 2 * 8_818_470_640
    offload_layers = args.offload_layers or 0
    if args.action in {"resume", "score"}:
        config, _ = load_run(output, runner_path, runner)
        weight_bytes = config["bf16_weight_bytes"]
        offload_layers = config["offload_layers"]
        if args.offload_layers is not None and args.offload_layers != offload_layers:
            parser.error("The run already has a different saved offload setting. Choose a new run tag.")
    missing = []
    try:
        gpu, enough = gpu_requirements(runner, weight_bytes, offload_layers)
    except (OSError, subprocess.SubprocessError):
        gpu, enough = [], False
    if gpu:
        print("\n".join(gpu), flush=True)
    if not enough:
        if offload_layers:
            missing.append(f"Free dedicated GPU memory is too low for the {offload_layers}-layer offload setting.")
        else:
            missing.append("Use one visible CUDA GPU with room for the 16.4 GiB BF16 base plus activations. "
                           "Start with 24 GiB or more and confirm capacity in the matched run.")
    if offload_layers:
        import psutil
        available = psutil.virtual_memory().available
        print(f"CPU offload: {offload_layers} decoder layers. Free system RAM: {available / GIB:.1f} GiB.", flush=True)
        if available < 24 * GIB:
            missing.append("Free at least 24 GiB of system RAM before using layer offload.")
    if args.action in {"check", "train"}:
        if args.base_snapshot is None or not args.base_snapshot.is_dir():
            missing.append("Point --base-snapshot at the local unquantized BF16 Qwen3-VL-8B-Instruct weights.")
        elif read(args.base_snapshot / "config.json").get("quantization_config"):
            missing.append("The selected base is quantized. Provide the unquantized BF16 snapshot.")
    if missing:
        message = "BF16-base control is blocked:\n" + "\n".join(f"- {reason}" for reason in missing)
        if args.action == "check" or args.report_blocked:
            print(message, flush=True)
            if args.action != "check":
                print("No BF16 work was started. Rerun this cell after meeting the prerequisites.", flush=True)
            return
        raise SystemExit(message)
    if args.action == "check":
        quantized = read(Path(config["base"]) / "config.json")
        snapshot, weight_bytes, _ = bf16_snapshot(args.base_snapshot, quantized)
        verify_split(args.reference_run, config, ("train", "val", "test"))
        print(f"Ready with {snapshot}. BF16 weights: {weight_bytes / GIB:.2f} GiB. CPU-offloaded layers: {offload_layers}.", flush=True)
        return
    if output.exists() and args.action == "train":
        parser.error(f"Run exists: {output}. Use resume to recover it or choose a fresh run folder.")
    sys.path.insert(0, str(runner_path.parent))
    from notebook_runner import stream_qwen
    command = [sys.executable, "-B", "-u", str(SCRIPT), *sys.argv[1:], "--logged"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8", RUNNER_ENV: str(runner_path)}) as child:
        try:
            stream_qwen(child, output, lambda line: print(line, end="", flush=True))
            code = child.wait()
        finally:
            if child.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
                if child.poll() is None:
                    child.terminate()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
