"""Run two BF16 LoRA updates to test CUDA system-memory fallback."""

import argparse
import os
import runpy
from pathlib import Path

import psutil

import train_bf16_control as control


def train(base_snapshot, reference_run, output, offload_layers):
    runner_path = control.DEFAULT_RUNNER.resolve(strict=True)
    runner = runpy.run_path(str(runner_path))
    cfg = control.read_reference(reference_run, runner_path, runner)
    quantized = control.read(Path(cfg["base"]) / "config.json")
    base, weights, fingerprints = control.bf16_snapshot(base_snapshot, quantized)
    control.verify_split(reference_run, cfg, ("train", "val"))
    worker = runner["load_worker"](Path(cfg["worker"]))
    worker.BASE_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    tables = runner["balanced_subset"]({
        split: worker.read_csv(reference_run / f"aptos_{split}.csv")
        for split in ("train", "val")
    })
    reference = cfg["base"]
    cfg.update(mode="pilot", steps=2, purpose="BF16 shared-memory feasibility only",
               base=str(base), base_files=fingerprints, reference_base=reference,
               offload_layers=offload_layers,
               rows={split: len(rows) for split, rows in tables.items()})
    cfg["training"].update(load_in_4bit=False, warmup_steps=0)
    cfg["split_sha256"] = {}
    for split, rows in tables.items():
        csv_path = output / f"aptos_{split}.csv"
        runner["write_csv"](csv_path, rows)
        cfg["split_sha256"][split] = control.sha(csv_path)
    selected = {r["image_path"] for rows in tables.values() for r in rows}
    image_csv = output / "input_images.csv"
    runner["write_csv"](image_csv, [row for row in worker.read_csv(reference_run / "input_images.csv")
                                    if row["image_path"] in selected])
    cfg["input_images_sha256"] = control.sha(image_csv)
    cfg["pilot_worker_sha256"] = control.sha(Path(__file__))
    cfg["reference_run"] = str(reference_run)
    cfg["identity"] = runner["identity"](cfg)
    worker.write_json(output / "pilot_configuration.json", cfg)
    worker.offline(output / "cache")
    temp = output / "cache/tmp"
    temp.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp))

    import pandas  # Match the pinned Windows import order.
    from unsloth import FastVisionModel
    import torch

    worker.resources(output, 10, gpu=True)
    available = psutil.virtual_memory().available
    if available < 24 * 2**30:
        raise RuntimeError(f"Need 24 GiB of available system RAM for the fallback test, have {available / 2**30:.1f} GiB.")
    helper = worker.local_module("qwen_reproduction", cfg["helper"])
    torch.cuda.reset_peak_memory_stats()
    print(f"Pilot only: loading {weights / 2**30:.2f} GiB of BF16 weights, then running two updates.", flush=True)
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(base), dtype=torch.bfloat16, load_in_16bit=True,
        load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
        offload_embedding=False, offload_layers=offload_layers,
        local_files_only=True, device_map={"": 0}, tokenizer_name=reference,
    )
    if any(getattr(model, flag, False) for flag in ("is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized", "peft_config")):
        raise RuntimeError("Expected the fresh unquantized base without an adapter.")
    swapper = getattr(model, "_unsloth_block_swap", None)
    parked = {}
    if offload_layers:
        if swapper is None or len(swapper.indices) != offload_layers:
            raise RuntimeError("The requested training layer offload did not initialize.")
        parked = {id(param): host for block in swapper.blocks
                  for param, host in zip(block.params, block.host, strict=True)}
        print(f"Offloaded {offload_layers} layers using {swapper.host_bytes() / 2**30:.2f} GiB of RAM.", flush=True)
    resident = [p for p in model.parameters() if id(p) not in parked]
    bf16_bytes = 2 * sum(t.numel() for t in (*resident, *parked.values()) if t.dtype == torch.bfloat16)
    if bf16_bytes < 16 * 2**30 or any(p.device.type != "cuda" for p in resident):
        raise RuntimeError("Expected the complete BF16 base with remaining weights mapped to CUDA.")
    print(f"Loaded {bf16_bytes / 2**30:.2f} GiB of BF16 weights.", flush=True)
    result = runner["train"](output, cfg, worker, helper, model, processor, tables, FastVisionModel, torch)
    result.update(feasibility_only=True, configuration_identity=cfg["identity"],
                  torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    worker.write_json(output / "train_result.json", result)
    print(f"Two updates and a small validation pass finished. Pilot result: {output / 'train_result.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train",))
    for name in ("base-snapshot", "reference-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--offload-layers", type=int, choices=range(37), default=0, metavar="0-36",
                        help="Number of Qwen's 36 frozen decoder layers stored in system RAM.")
    args = parser.parse_args()
    train(args.base_snapshot.resolve(strict=True), args.reference_run.resolve(strict=True),
          args.output.resolve(strict=True), args.offload_layers)
