"""S14 efficiency tables (Qwen3-VL-8B + LoRA vs RetiZero), computed only from saved run records.

Reads, read-only, under D:\\Afterimage_Experiments:
  runs\\<run>                              Qwen runs (training + inline scoring), attempts\\<id>\\{utilization.csv,system_specs.json,train.log,run_report.md}
  final_runs\\final_01\\train\\<job>         Qwen queue training jobs (same layout)
  final_runs\\final_01\\scores\\<job>        Qwen scoring jobs (summary.json, timing.csv, attempts\\<id>\\run_report.json, utilization.csv)
  final_runs\\final_01\\parameter_counts.json
  retizero_runs\\{retizero_01, retizero_01__pilot_01}   RetiZero scoring jobs, pilot training, parameter counts

Writes (distinct names from the older qwen_efficiency_*.csv):
  output/efficiency/efficiency_{training_runs,training_by_config,scoring_jobs,scoring_by_config,models,
  hardware_software,comparison}.csv, EFFICIENCY.md and GAPS.md.

Run from the repo root:
  C:\\Users\\moht2\\.conda\\envs\\afterimage_env\\python.exe -X utf8 -B finetuning\\paper_tables\\build_efficiency_tables.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

WARMUP_IMAGES = 3
HERE = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS = Path(r"D:\Afterimage_Experiments")
DEFAULT_OUT = HERE / "output" / "efficiency"
GIB = 2 ** 30

REPORT_STAGE = re.compile(r"^- (train|score): exit (\S+), ([\d.]+) s", re.M)
REPORT_UTIL = re.compile(r"^- (train|score): (\d+) samples, mean GPU ([\d.]+)%, mean CPU ([\d.]+)%, "
                         r"peak device VRAM ([\d.]+) MiB, peak system RAM ([\d.]+) GiB", re.M)
TRAINABLE = re.compile(r"Trainable parameters = ([\d,]+) of ([\d,]+)")
TRAIN_RUNTIME = re.compile(r"'train_runtime': '([\d.]+)'")


class BuildError(RuntimeError):
    pass


# --------------------------------------------------------------------------- small helpers

def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_sd(values) -> tuple[float | None, float | None]:
    """Mean and sample SD (n-1). SD is None for one value, mean None for none."""
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def timing_stats(seconds, warmup: int = WARMUP_IMAGES) -> dict:
    """Warm per-image timing: the first `warmup` images are left out (same rule as score_cohort.py)."""
    seconds = [float(v) for v in seconds]
    measured = seconds[warmup:] if len(seconds) > warmup else seconds
    ordered = sorted(measured)
    return {"images_timed": len(measured), "mean_s": statistics.fmean(measured), "median_s": statistics.median(measured),
            "p95_s": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
            "images_per_s": len(measured) / sum(measured)}


def close(a, b, rel=1e-6, absolute=1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=absolute)


def utilization_summary(rows: list[dict], stage: str | None = None) -> dict | None:
    """Per-stage statistics of one utilization.csv (5 s samples of nvidia-smi / psutil, whole machine)."""
    rows = [r for r in rows if stage is None or r["stage"] == stage]
    if not rows:
        return None
    times = [datetime.fromisoformat(r["utc"]) for r in rows]
    gpu = [float(r["gpu_percent"]) for r in rows]
    cpu = [float(r["cpu_percent"]) for r in rows]
    vram = [float(r["vram_mib"]) for r in rows]
    ram = [float(r["ram_bytes"]) for r in rows]
    return {"samples": len(rows), "span_s": (max(times) - min(times)).total_seconds(),
            "gpu_mean_pct": statistics.fmean(gpu), "gpu_peak_pct": max(gpu), "cpu_mean_pct": statistics.fmean(cpu),
            "card_peak_vram_mib": max(vram), "card_baseline_vram_mib": vram[0],
            "ram_peak_gib": max(ram) / GIB, "ram_baseline_gib": ram[0] / GIB}


def parse_run_report_md(text: str) -> dict:
    """{'status', 'wall': {stage: (seconds, exit)}, 'util': {stage: (samples, gpu, cpu, vram, ram)}} from run_report.md."""
    status = re.search(r"Status: \*\*(\w+)\*\*", text)
    return {"status": status.group(1) if status else None,
            "wall": {m[0]: (float(m[2]), m[1]) for m in REPORT_STAGE.findall(text)},
            "util": {m[0]: tuple(float(x) for x in m[1:]) for m in REPORT_UTIL.findall(text)}}


def parse_train_log(text: str) -> dict:
    trainable = TRAINABLE.search(text)
    runtimes = TRAIN_RUNTIME.findall(text)
    return {"trainable": int(trainable.group(1).replace(",", "")) if trainable else None,
            "total": int(trainable.group(2).replace(",", "")) if trainable else None,
            "trainer_runtime_s": float(runtimes[-1]) if runtimes else None}


def read_util(attempt: Path) -> list[dict]:
    path = attempt / "utilization.csv"
    return read_csv(path) if path.is_file() else []


def attempt_stage_wall(attempt: Path, stage: str) -> dict | None:
    """Wall time of one stage in one attempt: the recorded figure when there is one, else the span of utilization.csv.

    The span is one sampling interval (about 5 s) short of the true wall time; `source` says which was used.
    """
    report_md = attempt / "run_report.md"
    if report_md.is_file():
        parsed = parse_run_report_md(report_md.read_text(encoding="utf-8"))
        if stage in parsed["wall"]:
            return {"seconds": parsed["wall"][stage][0], "source": "run_report.md", "exit": parsed["wall"][stage][1]}
    report_json = attempt / "run_report.json"
    if report_json.is_file():
        data = read_json(report_json)
        if data.get("stage") == stage and "elapsed_seconds" in data:
            return {"seconds": float(data["elapsed_seconds"]), "source": "run_report.json", "exit": "0"}
    summary = utilization_summary(read_util(attempt), stage)
    if summary:
        return {"seconds": summary["span_s"], "source": "utilization.csv span (no recorded wall time)", "exit": None}
    return None


def sorted_attempts(folder: Path) -> list[Path]:
    root = folder / "attempts"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


# --------------------------------------------------------------------------- Qwen configuration

def config_label(cfg: dict) -> str:
    training = cfg["training"]
    base = "4-bit" if training["load_in_4bit"] else "BF16"
    offload = int(cfg.get("offload_layers") or 0)
    label = (f"{base}{f' (+{offload} layers CPU-offloaded)' if offload else ''} r{int(cfg['rank'])} "
             f"CLAHE-{'on' if cfg['clahe'] else 'off'} lr {float(training['lr']):g} alpha {float(training['lora_alpha']):g}")
    return label + (" previous split" if cfg.get("historical_split") else "")


def base_bytes(cfg: dict) -> int | None:
    files = cfg.get("base_files") or {}
    sizes = [v["size"] for k, v in files.items() if k.endswith(".safetensors")]
    return int(sum(sizes)) if sizes else None


def qwen_training_record(folder: Path, source: str) -> dict:
    cfg = read_json(folder / "pilot_configuration.json")
    result = read_json(folder / "train_result.json")
    attempts = sorted_attempts(folder)
    train_walls = []
    for attempt in attempts:
        wall = attempt_stage_wall(attempt, "train")
        if wall:
            train_walls.append({"attempt": attempt, **wall})
    if not train_walls:
        raise BuildError(f"{folder.name}: no training stage in any attempt")
    completing = [w for w in train_walls if w["exit"] == "0"]
    done = completing[-1] if completing else train_walls[-1]
    done_dir = done["attempt"]
    util_all = utilization_summary(read_util(done_dir), "train")
    report = parse_run_report_md((done_dir / "run_report.md").read_text(encoding="utf-8")) \
        if (done_dir / "run_report.md").is_file() else None
    if report and "train" in report["util"] and util_all:   # my statistics must reproduce the report's own line
        samples, gpu, cpu, vram, ram = report["util"]["train"]
        if int(samples) != util_all["samples"] or abs(gpu - util_all["gpu_mean_pct"]) > 0.06 \
                or abs(vram - util_all["card_peak_vram_mib"]) > 0.5 or abs(ram - util_all["ram_peak_gib"]) > 0.006:
            raise BuildError(f"{folder.name}: utilization.csv statistics differ from run_report.md")
    log_path = done_dir / "train.log"
    log = parse_train_log(log_path.read_text(encoding="utf-8", errors="replace")) if log_path.is_file() \
        else {"trainable": None, "total": None, "trainer_runtime_s": None}
    if report is not None and log["trainer_runtime_s"] is not None and log["trainer_runtime_s"] > done["seconds"] + 1:
        raise BuildError(f"{folder.name}: trainer runtime exceeds the stage wall time")
    history = result["history"]
    epochs = max(float(h.get("epoch", 0)) for h in history)
    train_images = int(cfg["rows"]["train"])
    score_walls = [w for w in (attempt_stage_wall(a, "score") for a in attempts) if w]
    inline = next(((a, attempt_stage_wall(a, "score")) for a in reversed(attempts)
                   if attempt_stage_wall(a, "score") and attempt_stage_wall(a, "score")["exit"] == "0"), None)
    inline_util = utilization_summary(read_util(inline[0]), "score") if inline else None
    score_result = read_json(folder / "score_result.json") if (folder / "score_result.json").is_file() else {}
    runtime = log["trainer_runtime_s"]
    training = cfg["training"]
    return {
        "source": source, "run": folder.name, "config": config_label(cfg),
        "base": "4-bit" if training["load_in_4bit"] else "BF16", "rank": int(cfg["rank"]),
        "clahe": "on" if cfg["clahe"] else "off", "lr": float(training["lr"]), "alpha": float(training["lora_alpha"]),
        "split": "previous" if cfg.get("historical_split") else "duplicate_excluded",
        "offload_layers": int(cfg.get("offload_layers") or 0),
        "trainer_seed": int(training["trainer_seed"]), "lora_seed": int(cfg["adapter_seed"]),
        "is_pilot": folder.name.startswith("pilot_"), "train_images": train_images,
        "epochs_run": epochs, "global_step": int(result["global_step"]),
        "attempts": len(attempts), "train_attempts": len(train_walls),
        "train_wall_all_attempts_s": sum(w["seconds"] for w in train_walls),
        "train_wall_completing_attempt_s": done["seconds"],
        "train_wall_other_attempts_s": sum(w["seconds"] for w in train_walls) - done["seconds"],
        "wall_source_other_attempts": "; ".join(sorted({w["source"] for w in train_walls if w is not done})),
        "attempts_timed_from_utilization_span": sum(1 for w in train_walls if w["source"].startswith("utilization")),
        "trainer_runtime_s": runtime,
        "s_per_epoch": runtime / epochs if runtime else None,
        "images_per_s": train_images * epochs / runtime if runtime else None,
        "torch_peak_alloc_gib": result.get("torch_peak_allocated_bytes", float("nan")) / GIB
        if result.get("torch_peak_allocated_bytes") else None,
        "torch_peak_reserved_gib": result["torch_peak_reserved_bytes"] / GIB if result.get("torch_peak_reserved_bytes") else None,
        "card_peak_vram_mib": util_all["card_peak_vram_mib"] if util_all else None,
        "card_baseline_vram_mib": util_all["card_baseline_vram_mib"] if util_all else None,
        "gpu_mean_pct": util_all["gpu_mean_pct"] if util_all else None,
        "gpu_peak_pct": util_all["gpu_peak_pct"] if util_all else None,
        "cpu_mean_pct": util_all["cpu_mean_pct"] if util_all else None,
        "ram_peak_gib": util_all["ram_peak_gib"] if util_all else None,
        "ram_baseline_gib": util_all["ram_baseline_gib"] if util_all else None,
        "inline_score_images": cfg["rows"].get("test"),
        "inline_score_wall_all_attempts_s": sum(w["seconds"] for w in score_walls) if score_walls else None,
        "inline_score_wall_s": inline[1]["seconds"] if inline else None,
        "inline_score_card_peak_vram_mib": inline_util["card_peak_vram_mib"] if inline_util else None,
        "inline_score_torch_peak_alloc_gib": score_result["torch_peak_allocated_bytes"] / GIB
        if score_result.get("torch_peak_allocated_bytes") else None,
        "trainable_params_logged": log["trainable"], "total_params_logged": log["total"],
        "_cfg": cfg, "_folder": folder,
    }


# --------------------------------------------------------------------------- RetiZero pilot training

def retizero_pilot_record(folder: Path) -> dict:
    result = read_json(folder / "train_result.json")
    attempts = [a for a in sorted_attempts(folder) if a.name.startswith("train-")]
    walls = [attempt_stage_wall(a, "train") for a in attempts]
    walls = [w for w in walls if w]
    history = result["history"]
    seconds = sum(h["train_seconds"] + h["validation_seconds"] for h in history)
    images = round(history[0]["train_seconds"] * history[0]["train_images_per_second"])
    util = utilization_summary([r for a in attempts for r in read_util(a)], "train")
    config = result["configuration"]
    return {
        "source": "retizero_01__pilot_01", "run": folder.name, "config": "RetiZero ViT-L/16 + LoRA r8, random init, 2-epoch 160-image pilot",
        "base": "FP16 autocast", "rank": 8, "clahe": "off", "lr": config["training"]["learning_rate"], "alpha": None,
        "split": "pilot subset of duplicate_excluded", "offload_layers": 0, "trainer_seed": config["training"]["seed"],
        "lora_seed": None, "is_pilot": True, "train_images": images, "epochs_run": float(result["epochs_completed"]),
        "global_step": sum(h["optimizer_steps"] for h in history), "attempts": len(sorted_attempts(folder)),
        "train_attempts": len(walls), "train_wall_all_attempts_s": sum(w["seconds"] for w in walls),
        "train_wall_completing_attempt_s": walls[-1]["seconds"] if walls else None,
        "train_wall_other_attempts_s": sum(w["seconds"] for w in walls[:-1]),
        "wall_source_other_attempts": "; ".join(sorted({w["source"] for w in walls[:-1]})),
        "attempts_timed_from_utilization_span": sum(1 for w in walls if w["source"].startswith("utilization")),
        "trainer_runtime_s": seconds, "s_per_epoch": seconds / result["epochs_completed"],
        "images_per_s": statistics.fmean(h["train_images_per_second"] for h in history),
        "torch_peak_alloc_gib": max(h["torch_peak_allocated_bytes"] for h in history) / GIB,
        "torch_peak_reserved_gib": max(h["torch_peak_reserved_bytes"] for h in history) / GIB,
        "card_peak_vram_mib": util["card_peak_vram_mib"] if util else None,
        "card_baseline_vram_mib": util["card_baseline_vram_mib"] if util else None,
        "gpu_mean_pct": util["gpu_mean_pct"] if util else None, "gpu_peak_pct": util["gpu_peak_pct"] if util else None,
        "cpu_mean_pct": util["cpu_mean_pct"] if util else None, "ram_peak_gib": util["ram_peak_gib"] if util else None,
        "ram_baseline_gib": util["ram_baseline_gib"] if util else None,
        "inline_score_images": None, "inline_score_wall_all_attempts_s": None, "inline_score_wall_s": None,
        "inline_score_card_peak_vram_mib": None, "inline_score_torch_peak_alloc_gib": None,
        "trainable_params_logged": None, "total_params_logged": None, "_cfg": None, "_folder": folder,
    }


# --------------------------------------------------------------------------- scoring jobs

def adapter_run_config(identity: dict, roots: list[Path]) -> tuple[str, dict | None, Path | None]:
    adapter = identity.get("adapter")
    base = {"4bit": "4-bit", "bf16": "BF16"}[identity["base_precision"]]
    if adapter is None:
        return f"zero-shot {base}", None, None
    run = Path(adapter["directory"]).parent
    for candidate in [run] + [root / run.name for root in roots]:
        if (candidate / "pilot_configuration.json").is_file():
            cfg = read_json(candidate / "pilot_configuration.json")
            return config_label(cfg), cfg, candidate
    return "original paper adapter (legacy best_lora, outside the experiment folders)", None, None


def qwen_scoring_record(folder: Path, roots: list[Path]) -> dict:
    summary = read_json(folder / "summary.json")
    identity = summary["identity"]
    label, cfg, run = adapter_run_config(identity, roots)
    timing = read_csv(folder / "timing.csv")
    if len(timing) != int(identity["images"]):
        raise BuildError(f"{folder.name}: timing.csv rows differ from the image count")
    warm = timing_stats([r["full_answer_seconds"] for r in timing], summary["full_answer_timing"]["warmup_images_excluded"])
    saved = summary["full_answer_timing"]
    for ours, theirs in (("median_s", "median_seconds"), ("mean_s", "mean_seconds"), ("images_per_s", "images_per_second")):
        if not close(warm[ours], saved[theirs]):
            raise BuildError(f"{folder.name}: recomputed {ours} differs from summary.json")
    gen = {}
    if int(identity["generation_images"]):
        seconds = [r["generation_seconds"] for r in timing[:int(identity["generation_images"])] if r["generation_seconds"] != ""]
        gen = timing_stats(seconds, summary["generation"]["timing"]["warmup_images_excluded"])
        if not close(gen["median_s"], summary["generation"]["timing"]["median_seconds"]):
            raise BuildError(f"{folder.name}: recomputed generation median differs from summary.json")
    attempts = sorted_attempts(folder)
    wall = sum(attempt_stage_wall(a, "score")["seconds"] for a in attempts)
    util = utilization_summary([r for a in attempts for r in read_util(a)], "score")
    precision = identity.get("scoring_precision", "FP16")
    return {
        "source": "final_01/scores", "job": folder.name, "model": label, "cohort": identity["cohort"], "images": int(identity["images"]),
        "scoring_precision": precision, "is_fallback": bool(identity.get("fallback_for")), "is_pilot": bool(identity.get("limit")),
        "attempts": len(attempts), "job_wall_s": wall, "model_load_s": float(summary["model_load_seconds"]),
        "warm_images": warm["images_timed"], "warm_median_s": warm["median_s"], "warm_mean_s": warm["mean_s"], "warm_p95_s": warm["p95_s"],
        "warm_images_per_s": warm["images_per_s"], "forward_passes_per_image": summary["full_answer_timing"]["forward_passes_per_image"],
        "gen_median_s": gen.get("median_s"), "gen_images_per_s": gen.get("images_per_s"),
        "batched_images_per_s": None, "torch_peak_alloc_gib": summary["torch_peak_allocated_bytes"] / GIB,
        "torch_peak_reserved_gib": summary["torch_peak_reserved_bytes"] / GIB, "torch_peak_batched_gib": None,
        "card_peak_vram_mib": util["card_peak_vram_mib"], "card_baseline_vram_mib": util["card_baseline_vram_mib"],
        "gpu_mean_pct": util["gpu_mean_pct"], "gpu_peak_pct": util["gpu_peak_pct"], "cpu_mean_pct": util["cpu_mean_pct"],
        "ram_peak_gib": util["ram_peak_gib"], "util_samples": util["samples"], "_run": run, "_cfg": cfg, "_folder": folder,
        "_packages": summary.get("packages", {}),
    }


def retizero_scoring_record(folder: Path, source: str) -> dict:
    summary = read_json(folder / "summary.json")
    identity = summary["identity"]
    timing = read_csv(folder / "timing.csv")
    single = summary["single_image_timing"]
    skip = int(single["warm_up_images_excluded"])
    warm = timing_stats([r["total_seconds"] for r in timing], skip)
    if not close(warm["median_s"], single["total"]["median_seconds"]) or not close(warm["mean_s"], single["total"]["mean_seconds"]):
        raise BuildError(f"{folder.name}: recomputed batch-1 timing differs from summary.json")
    attempts = sorted_attempts(folder)
    wall = sum(attempt_stage_wall(a, "score")["seconds"] for a in attempts)
    util = utilization_summary([r for a in attempts for r in read_util(a)], "score")
    return {
        "source": source, "job": folder.name, "model": "RetiZero ViT-L/16 + LoRA r8 (saved fine-tuned checkpoint)",
        "cohort": identity["cohort"], "images": int(identity["images"]), "scoring_precision": identity["precision"],
        "is_fallback": False, "is_pilot": bool(identity.get("limit")) or "pilot" in source, "attempts": len(attempts), "job_wall_s": wall,
        "model_load_s": float(summary["model_load_seconds"]), "warm_images": warm["images_timed"], "warm_median_s": warm["median_s"],
        "warm_mean_s": warm["mean_s"], "warm_p95_s": warm["p95_s"], "warm_images_per_s": warm["images_per_s"],
        "forward_passes_per_image": 1, "gen_median_s": None, "gen_images_per_s": None,
        "batched_images_per_s": summary["batched"]["images_per_second"],
        "torch_peak_alloc_gib": single["torch_peak_allocated_bytes"] / GIB, "torch_peak_reserved_gib": None,
        "torch_peak_batched_gib": summary["batched"]["torch_peak_allocated_bytes"] / GIB,
        "card_peak_vram_mib": util["card_peak_vram_mib"], "card_baseline_vram_mib": util["card_baseline_vram_mib"],
        "gpu_mean_pct": util["gpu_mean_pct"], "gpu_peak_pct": util["gpu_peak_pct"], "cpu_mean_pct": util["cpu_mean_pct"],
        "ram_peak_gib": util["ram_peak_gib"], "util_samples": util["samples"], "_run": None, "_cfg": None, "_folder": folder,
        "_packages": identity["code"]["packages"],
    }


# --------------------------------------------------------------------------- aggregation

TRAIN_METRICS = ["epochs_run", "train_wall_all_attempts_s", "trainer_runtime_s", "s_per_epoch", "images_per_s",
                 "torch_peak_alloc_gib", "torch_peak_reserved_gib", "card_peak_vram_mib", "gpu_mean_pct", "gpu_peak_pct",
                 "cpu_mean_pct", "ram_peak_gib", "inline_score_wall_s"]
SCORE_METRICS = ["job_wall_s", "model_load_s", "warm_median_s", "warm_images_per_s", "gen_median_s", "gen_images_per_s",
                 "batched_images_per_s", "torch_peak_alloc_gib", "torch_peak_reserved_gib", "torch_peak_batched_gib",
                 "card_peak_vram_mib", "gpu_mean_pct", "gpu_peak_pct", "cpu_mean_pct", "ram_peak_gib"]


def aggregate(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, members in groups.items():
        record = dict(zip(keys, key))
        record["n"] = len(members)
        for metric in metrics:
            mean, sd = mean_sd([m[metric] for m in members])
            record[f"{metric}_mean"], record[f"{metric}_sd"] = mean, sd
        out.append(record)
    return out


def fmt(value, digits=2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def fmt_mean_sd(mean, sd, digits=2, n=None) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:,.{digits}f}" if sd is None else f"{mean:,.{digits}f} ± {sd:,.{digits}f}"


def public(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else (f"{v:.10g}" if isinstance(v, float) else v)) for k, v in row.items()})


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------- hardware / software / models

def hardware_software(train_rows, score_rows) -> list[dict]:
    specs: dict[str, list[str]] = defaultdict(list)
    for row in list(train_rows) + list(score_rows):
        for attempt in sorted_attempts(row["_folder"]):
            path = attempt / "system_specs.json"
            if not path.is_file():
                continue
            data = read_json(path)
            gpu = dict(zip(data["gpu_fields"], [x.strip() for x in data["gpu"][0].split(",")]))
            key = json.dumps({**{k: data[k] for k in ("os", "python", "cpu", "logical_cpus", "physical_cores", "ram_bytes")},
                              "gpu_name": gpu["name"], "gpu_total_vram_mib": gpu["total_vram_mib"], "gpu_driver": gpu["driver"]})
            specs[key].append(str(path))
    rows = []
    for key, where in specs.items():
        data = json.loads(key)
        for name, value in data.items():
            rows.append({"kind": "hardware/OS", "item": name, "value": value, "attempt_files": len(where),
                         "note": "a driver update separates the two driver values" if name == "gpu_driver" else ""})
    versions: dict[tuple, int] = defaultdict(int)
    for row in train_rows:
        if row["_cfg"]:
            for pkg, version in row["_cfg"]["package_versions"].items():
                versions[("Qwen training", pkg, version)] += 1
    for row in score_rows:
        for pkg, version in row["_packages"].items():
            versions[("Qwen scoring" if row["source"] == "final_01/scores" else "RetiZero scoring", pkg, version)] += 1
    for (stack, pkg, version), count in sorted(versions.items()):
        rows.append({"kind": f"software ({stack})", "item": pkg, "value": version, "attempt_files": count, "note": ""})
    return rows


def model_rows(final_root: Path, counts: dict, train_rows: list[dict], retizero_counts: dict, retizero_paths: dict) -> list[dict]:
    base = counts["base_parameters"]
    rows = []
    for rank_key, info in counts["adapters"].items():
        rank = int(rank_key.split("_")[1])
        runs = [r for r in train_rows if r["rank"] == rank and r["base"] == "4-bit" and not r["is_pilot"]]
        if not runs:
            continue
        adapter_path = runs[0]["_folder"] / "best_lora" / "adapter_model.safetensors"
        size = adapter_path.stat().st_size
        if size != info["adapter_file_bytes"]:
            raise BuildError(f"rank {rank}: adapter size on disk differs from parameter_counts.json")
        logged = {r["trainable_params_logged"] for r in runs if r["trainable_params_logged"]}
        if logged != {info["trainable_parameters"]}:
            raise BuildError(f"rank {rank}: logged trainable parameters {logged} differ from parameter_counts.json")
        bf16 = [r for r in train_rows if r["rank"] == rank and r["base"] == "BF16" and not r["is_pilot"]]
        for label, subset in (("4-bit", runs), ("BF16", bf16)):
            if not subset:
                continue
            rows.append({"model": f"Qwen3-VL-8B-Instruct + LoRA r{rank} ({label} base)", "frozen_base_parameters": base,
                         "trainable_parameters": info["trainable_parameters"], "total_parameters": base + info["trainable_parameters"],
                         "trainable_percent_of_base": 100 * info["trainable_parameters"] / base,
                         "adapter_file_bytes": size, "frozen_base_weight_files_bytes": base_bytes(subset[0]["_cfg"]),
                         "note": "parameters: product of tensor shapes in the safetensors headers (parameter_counts.json); "
                                 "adapter size: stat of best_lora/adapter_model.safetensors"
                                 + ("; train.log total differs for BF16 (CPU-offloaded run), so the header count is used" if label == "BF16" else "")})
    rows.append({"model": "RetiZero ViT-L/16 + LoRA r8, all weights trained", "frozen_base_parameters": 0,
                 "trainable_parameters": retizero_counts["trained_in_fine_tuning"], "total_parameters": retizero_counts["model_total"],
                 "trainable_percent_of_base": 100.0, "adapter_file_bytes": retizero_paths["app_checkpoint_bytes"],
                 "frozen_base_weight_files_bytes": None,
                 "note": "parameters: parameter_counts.json (encoder 304,088,064 incl. 786,432 LoRA-branch weights + head 5,125); "
                         "file size: app-format checkpoint written by the pilot trainer (best_epoch_01.pth, same architecture); "
                         f"the original saved checkpoint file is {retizero_paths['original_checkpoint_bytes']:,} bytes because it also stores a "
                         "second encoder copy and the text model"})
    return rows


# --------------------------------------------------------------------------- build

def collect(experiments: Path, repo_root: Path) -> dict:
    runs_root, final_root = experiments / "runs", experiments / "final_runs" / "final_01"
    train = [qwen_training_record(p, "runs") for p in sorted(runs_root.iterdir()) if (p / "train_result.json").is_file()]
    train += [qwen_training_record(p, "final_01/train") for p in sorted((final_root / "train").iterdir()) if (p / "train_result.json").is_file()]
    pilot_root = experiments / "retizero_runs" / "retizero_01__pilot_01"
    train += [retizero_pilot_record(p) for p in sorted((pilot_root / "train").iterdir())]
    scoring = [qwen_scoring_record(p, [runs_root, final_root / "train"]) for p in sorted((final_root / "scores").iterdir())]
    for name in ("retizero_01", "retizero_01__pilot_01"):
        for p in sorted((experiments / "retizero_runs" / name / "scores").iterdir()):
            scoring.append(retizero_scoring_record(p, name))
    retizero_paths = {"app_checkpoint_bytes": (pilot_root / "train" / "pilot_straight_t42" / "best_epoch_01.pth").stat().st_size,
                      "original_checkpoint_bytes": (repo_root / "RetiZero" / "best_retizero_dr.pth").stat().st_size}
    return {"train": train, "scoring": scoring, "counts": read_json(final_root / "parameter_counts.json"),
            "retizero_counts": read_json(experiments / "retizero_runs" / "retizero_01" / "parameter_counts.json"),
            "retizero_paths": retizero_paths}


BASELINE_4BIT = "4-bit r16 CLAHE-on lr 0.0005 alpha 16"
BASELINE_BF16 = "BF16 (+16 layers CPU-offloaded) r16 CLAHE-on lr 0.0005 alpha 16"
ROWS_COMPARISON = [  # (quantity, unit, source table, column, digits)
    ("Total parameters", "count", "model", "total_parameters", 0), ("Trainable parameters", "count", "model", "trainable_parameters", 0),
    ("Trainable share of base", "%", "model", "trainable_percent_of_base", 3), ("Adapter / checkpoint file", "bytes", "model", "adapter_file_bytes", 0),
    ("Training epochs run (early stopping)", "epochs", "train", "epochs_run_mean", 1),
    ("Training wall time, all attempts", "s", "train", "train_wall_all_attempts_s_mean", 0),
    ("Training time per epoch (incl. validation)", "s/epoch", "train", "s_per_epoch_mean", 1),
    ("Training throughput", "images/s", "train", "images_per_s_mean", 2),
    ("Training peak GPU memory, PyTorch allocated", "GiB", "train", "torch_peak_alloc_gib_mean", 2),
    ("Training peak GPU memory, whole card", "MiB", "train", "card_peak_vram_mib_mean", 0),
    ("Training mean / peak GPU utilisation", "%", "train", "gpu_mean_pct_mean", 1),
    ("Scoring model load time", "s", "score", "model_load_s_mean", 1),
    ("Scoring warm median time per image (batch 1)", "s/image", "score", "warm_median_s_mean", 4),
    ("Scoring throughput (batch 1)", "images/s", "score", "warm_images_per_s_mean", 2),
    ("Qwen greedy generation, warm median", "s/image", "score", "gen_median_s_mean", 3),
    ("Scoring throughput (batched, 64)", "images/s", "score", "batched_images_per_s_mean", 1),
    ("Scoring peak GPU memory, PyTorch allocated (batch 1)", "GiB", "score", "torch_peak_alloc_gib_mean", 2),
    ("Scoring peak GPU memory, whole card", "MiB", "score", "card_peak_vram_mib_mean", 0),
    ("Scoring mean GPU utilisation", "%", "score", "gpu_mean_pct_mean", 1),
]


def comparison(models: list[dict], train_agg: list[dict], score_agg: list[dict]) -> tuple[list[dict], list[str], list[list]]:
    columns = [("Qwen3-VL-8B + LoRA r16, 4-bit, CLAHE-on", BASELINE_4BIT, "Qwen3-VL-8B-Instruct + LoRA r16 (4-bit base)", "aptos361_on"),
               ("Qwen3-VL-8B + LoRA r16, BF16, CLAHE-on", BASELINE_BF16, "Qwen3-VL-8B-Instruct + LoRA r16 (BF16 base)", "aptos361_on"),
               ("RetiZero ViT-L/16 + LoRA r8", None, "RetiZero ViT-L/16 + LoRA r8, all weights trained", "aptos361_clean")]
    long_rows, wide = [], []
    for quantity, unit, table, column, digits in ROWS_COMPARISON:
        wide_row = [quantity, unit]
        for title, config, model_name, cohort in columns:
            mean = sd = None
            n = 0
            note = ""
            if table == "model":
                row = next((m for m in models if m["model"] == model_name), None)
                mean, n = (row[column] if row else None), 1
            elif table == "train":
                row = next((a for a in train_agg if a["config"] == config and not a["is_pilot"]), None) if config else None
                if row:
                    mean, sd, n = row[column], row[column[:-5] + "_sd"], row["n"]
                elif config is None:
                    note = "not run yet"
            else:
                row = next((a for a in score_agg if a["cohort"] == cohort and not a["is_pilot"] and a["scoring_precision"] != "FP32"
                            and (a["model"] == config if config else a["model"].startswith("RetiZero"))), None)
                if row:
                    mean, sd, n = row[column], row[column[:-5] + "_sd"], row["n"]
            if column == "gpu_mean_pct_mean" and table == "train" and mean is not None:
                peak = next(a for a in train_agg if a["config"] == config and not a["is_pilot"])["gpu_peak_pct_mean"]
                cell = f"{mean:.1f} / {peak:.0f}"
            else:
                cell = note or fmt_mean_sd(mean, sd, digits)
            long_rows.append({"quantity": quantity, "unit": unit, "model": title, "n": n, "mean": mean, "sd": sd, "note": note})
            wide_row.append(cell + (f" (n={n})" if n > 1 and mean is not None and table != "model" else ""))
        wide.append(wide_row)
    return long_rows, ["Quantity", "Unit"] + [c[0] for c in columns], wide


def build_gaps(data: dict, train_agg, score_agg) -> str:
    train, scoring = data["train"], data["scoring"]
    qwen_train = [r for r in train if r["source"] != "retizero_01__pilot_01" and not r["is_pilot"]]
    scored = {(a["model"]) for a in score_agg if a["cohort"] == "aptos361_on" and not a["is_pilot"]}
    missing = sorted({r["config"] for r in qwen_train if r["config"] not in scored})
    missing_runs = sorted(r["run"] for r in qwen_train if r["config"] in missing)
    no_report = sorted(r["run"] for r in qwen_train if r["attempts_timed_from_utilization_span"])
    rows = [
        ("all training runs, all scoring jobs", "GPU power draw and energy (W, Wh per epoch / run / image); CPU power",
         "utilization.csv samples only gpu_percent, vram_mib, cpu_percent and ram_bytes; system_specs.json has no power field",
         "short measurement job: sample nvidia-smi power.draw at 1 Hz during one training epoch and one 361-image scoring pass, giving W and Wh/epoch, Wh/image; "
         "whole-run energy then follows by multiplying with the recorded epochs, or needs a full rerun for a measured value"),
        ("RetiZero (no full-split training run exists)", "training wall time, epochs to convergence, s/epoch and images/s on the 2,786-image split, peak training memory at the real recipe",
         "only retizero_01__pilot_01 exists: 2 epochs on 160 random-init images (indicative figures are in EFFICIENCY.md); the saved checkpoint came from the original notebook, which left no timing or utilisation record",
         "short measurement job for s/epoch, images/s and peak memory (1-2 epochs on the full train split with the pilot trainer); wall time to early-stopped convergence needs a full run"),
        ("RetiZero scoring jobs (retizero_01)", "SD across repeats; whole-card peak memory and GPU/CPU utilisation",
         "one job per cohort; the whole job lasts 18 s so utilization.csv holds 4 samples; PyTorch peaks are exact but the whole-card figure is coarse",
         "short measurement job: repeat the 361-image scoring 3-5 times and sample utilisation for the whole loop"),
        ("all Qwen scoring jobs", "batched / multi-image throughput", "scoring is batch 1, five forward passes per image (likelihood) or one greedy generation; no batched path was timed",
         "short measurement job (one or two batch sizes on 361 images)"),
        ("Qwen training runs with only an inline score stage" + (f" ({len(missing_runs)} runs, {len(missing)} configurations)" if missing else ""),
         "warm per-image scoring time, images/s, load time" + (": " + "; ".join(missing) if missing else ""),
         "runs/ stored only the stage wall time (includes model load and five passes per image); timing.csv exists only for the final_01 re-scoring jobs, which cover only: " + "; ".join(sorted(scored)),
         "short measurement job: re-score the 361-image test set once per configuration (about 15 minutes each)"),
        ("all Qwen and RetiZero runs", "per-process peak GPU and host memory",
         "utilization.csv is whole-machine (card and RAM include other processes: the first sample is a 1.4 GB card and about 28 GB RAM baseline); only PyTorch's own peak is process-level and is on the GPU side only",
         "short measurement job with per-process sampling (psutil RSS, nvidia-smi --query-compute-apps) for host memory; GPU memory is already covered by the PyTorch peaks"),
        ("original paper adapter (scored as original_paper_adapter__aptos367_original)", "training time, epochs, hardware, trainable parameters, adapter size",
         "its folder lies outside the experiment tree (original Windows path under C:), so no training record or file is read",
         "not derivable from this tree; only a rerun of the original recipe would measure them"),
        ("zero-shot rows", "training quantities", "no training takes place", "not applicable"),
    ]
    if no_report:
        rows.append((", ".join(no_report), "exact wall time of the interrupted training attempt",
                     "the attempt has no run_report (killed/interrupted), so its wall time is the span of utilization.csv, about one 5 s sampling interval short; the first attempt trained and was discarded; the later attempt is the one that produced the adapter",
                     "not fillable from records; the error is under 10 s per attempt, no rerun needed"))
    out = ["# Efficiency gaps", "",
           "Quantities that could not be derived from the saved run records. Every number in the tables comes from a file; "
           "blank or `n/a` cells appear only for the rows below.", "",
           "| Run(s) | Quantity | Why it is not derivable | Fill with |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--repo-root", type=Path, default=HERE.parent.parent)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out = args.out
    data = collect(args.experiments_root, args.repo_root)
    train, scoring = data["train"], data["scoring"]

    models = model_rows(args.experiments_root / "final_runs" / "final_01", data["counts"], train, data["retizero_counts"], data["retizero_paths"])
    hw = hardware_software(train, scoring)
    write_csv([public(r) for r in train], out / "efficiency_training_runs.csv")
    write_csv([public(r) for r in scoring], out / "efficiency_scoring_jobs.csv")
    write_csv(models, out / "efficiency_models.csv")
    write_csv(hw, out / "efficiency_hardware_software.csv")

    qwen_train = [r for r in train if not r["is_pilot"]]
    train_agg = aggregate(train, ["config", "base", "rank", "clahe", "lr", "alpha", "split", "is_pilot"], TRAIN_METRICS)
    for row in train_agg:
        members = [r for r in train if r["config"] == row["config"] and r["is_pilot"] == row["is_pilot"]]
        row["runs_with_extra_attempts"] = sum(1 for m in members if m["train_attempts"] > 1)
    train_agg.sort(key=lambda r: (r["is_pilot"], r["base"], r["split"], r["rank"], r["clahe"] == "off", r["lr"], r["alpha"]))
    score_agg = aggregate(scoring, ["model", "cohort", "scoring_precision", "is_pilot"], SCORE_METRICS)
    write_csv(train_agg, out / "efficiency_training_by_config.csv")
    write_csv(score_agg, out / "efficiency_scoring_by_config.csv")
    long_rows, headers, wide = comparison(models, train_agg, score_agg)
    write_csv(long_rows, out / "efficiency_comparison.csv")
    (out / "GAPS.md").write_text(build_gaps(data, train_agg, score_agg), encoding="utf-8")
    (out / "EFFICIENCY.md").write_text(render_markdown(data, models, hw, train_agg, score_agg, headers, wide), encoding="utf-8")
    print(f"{len(train)} training runs, {len(scoring)} scoring jobs, {len(train_agg)} training configs -> {out}")


def render_markdown(data, models, hw, train_agg, score_agg, headers, wide) -> str:
    L = ["# S14 efficiency: Qwen3-VL-8B + LoRA vs RetiZero", "",
         "Computed by `build_efficiency_tables.py` from saved run records only (no new GPU work). Mean ± SD is over seeds (sample SD); "
         "n is shown when above 1. Missing quantities are listed in [GAPS.md](GAPS.md).", "",
         "## Definitions", "",
         "- **Training wall time (all attempts)**: sum over every attempt of the training stage. Recorded stage times are used where they exist "
         "(`run_report.md`); an interrupted attempt has none, so the span of its `utilization.csv` is used (about 5 s short).",
         "- **s/epoch and images/s** use the Trainer's own `train_runtime` (training loop, per-epoch validation and checkpointing, without model loading) "
         "divided by the epochs run (maximum `epoch` in `train_result.json`). Epochs vary (3 to 9) because of early stopping, so totals are not comparable across runs, "
         "s/epoch is. The Trainer's logged `train_samples_per_second` is not used: it divides by the 50 planned epochs, not the epochs run.",
         "- **Whole-card memory / GPU % / CPU % / RAM** come from `utilization.csv` (5 s samples, whole machine, including the desktop; "
         "baseline = first sample). **PyTorch peak** is process-level.",
         "- **Warm per-image time** skips the first 3 images. Qwen: log-likelihood of all five grades (five forward passes), the path that produces the reported metrics; "
         "greedy generation (24 new tokens) is listed separately. RetiZero: one forward pass at 224 px, batch 1, transfer + forward + softmax + image read/transform.",
         "- Runs with extra attempts (interrupted or failed first attempts) are counted in full in wall time; GPU, memory and utilisation figures use the attempt that completed training "
         "(`train_wall_completing_attempt_s` and `train_wall_other_attempts_s` in the per-run CSV separate the two). Pilot runs are flagged and excluded from the comparison.", ""]
    L += ["## Comparison (APTOS 361-image test set)", "", md_table(headers, wide), "",
          "RetiZero training: **not run yet** (only a 2-epoch pilot on 160 random-initialised images exists; see GAPS.md). "]
    pilots = [r for r in data["train"] if r["source"] == "retizero_01__pilot_01"]
    if pilots:
        p = pilots[0]
        L += [f"Pilot indication, not a result: {p['images_per_s']:.1f} images/s, PyTorch peak {p['torch_peak_alloc_gib']:.1f} GiB, "
              f"whole-card peak {p['card_peak_vram_mib']:.0f} MiB (`{p['run']}`).", ""]
    L += ["Qwen scoring figures are for the same seeds' adapters re-scored in final_01 (`timing.csv`); the BF16 column has one scoring job (n=1). "
          "RetiZero scoring is one deterministic job (n=1).", ""]
    merged: dict[tuple, dict] = {}
    for h in hw:
        slot = merged.setdefault((h["kind"], h["item"]), {})
        slot[h["value"]] = slot.get(h["value"], 0) + h["attempt_files"]
    hw_rows = [[kind, item, "; ".join(f"{str(v).replace('|', '/')} ({n} attempt files)" for v, n in values.items())]
               for (kind, item), values in merged.items()]
    L += ["## Hardware and software", "", md_table(["Kind", "Item", "Value"], hw_rows), ""]
    L += ["## Parameters and checkpoint size", "",
          md_table(["Model", "Frozen base", "Trainable", "Total", "Trainable % of base", "Adapter/checkpoint bytes", "Base weight files bytes"],
                   [[m["model"], f"{m['frozen_base_parameters']:,}", f"{m['trainable_parameters']:,}", f"{m['total_parameters']:,}",
                     f"{m['trainable_percent_of_base']:.3f}", f"{m['adapter_file_bytes']:,}",
                     f"{m['frozen_base_weight_files_bytes']:,}" if m["frozen_base_weight_files_bytes"] else "n/a"] for m in models]), "",
          "Notes per row are in `efficiency_models.csv`.", ""]
    train_rows = []
    for a in train_agg:
        gpu = f"{a['gpu_mean_pct_mean']:.0f} / {a['gpu_peak_pct_mean']:.0f}" if a["gpu_mean_pct_mean"] is not None else "n/a"
        train_rows.append([a["config"] + (" (pilot)" if a["is_pilot"] else ""), a["n"], fmt_mean_sd(a["epochs_run_mean"], a["epochs_run_sd"], 1),
                           fmt_mean_sd(a["train_wall_all_attempts_s_mean"], a["train_wall_all_attempts_s_sd"], 0),
                           fmt_mean_sd(a["s_per_epoch_mean"], a["s_per_epoch_sd"], 0), fmt_mean_sd(a["images_per_s_mean"], a["images_per_s_sd"], 2),
                           fmt_mean_sd(a["torch_peak_alloc_gib_mean"], a["torch_peak_alloc_gib_sd"], 2),
                           fmt_mean_sd(a["card_peak_vram_mib_mean"], a["card_peak_vram_mib_sd"], 0), gpu,
                           fmt_mean_sd(a["cpu_mean_pct_mean"], a["cpu_mean_pct_sd"], 1), fmt_mean_sd(a["ram_peak_gib_mean"], a["ram_peak_gib_sd"], 1),
                           a["runs_with_extra_attempts"]])
    L += ["## Training by configuration", "",
          md_table(["Configuration", "n", "Epochs", "Wall s (all attempts)", "s/epoch", "Images/s", "PyTorch peak GiB", "Card peak MiB",
                    "GPU mean / peak %", "CPU mean %", "RAM peak GiB", "Runs with >1 attempt"], train_rows), ""]
    score_rows = []
    for a in score_agg:
        score_rows.append([a["model"] + (" (pilot)" if a["is_pilot"] else ""), a["cohort"], a["scoring_precision"], a["n"],
                           fmt_mean_sd(a["model_load_s_mean"], a["model_load_s_sd"], 1),
                           fmt_mean_sd(a["warm_median_s_mean"], a["warm_median_s_sd"], 4), fmt_mean_sd(a["warm_images_per_s_mean"], a["warm_images_per_s_sd"], 2),
                           fmt_mean_sd(a["gen_median_s_mean"], a["gen_median_s_sd"], 3), fmt_mean_sd(a["batched_images_per_s_mean"], a["batched_images_per_s_sd"], 1),
                           fmt_mean_sd(a["torch_peak_alloc_gib_mean"], a["torch_peak_alloc_gib_sd"], 2),
                           fmt_mean_sd(a["card_peak_vram_mib_mean"], a["card_peak_vram_mib_sd"], 0)])
    L += ["## Scoring by model and cohort", "",
          md_table(["Model", "Cohort", "Precision", "n jobs", "Load s", "Warm median s/image", "Images/s", "Generation median s", "Batched images/s",
                    "PyTorch peak GiB", "Card peak MiB"], score_rows), "",
          "FP32 rows are the fallback jobs for runs whose FP16 scoring overflowed; their times are not comparable to FP16. "
          "Per-run and per-job values: `efficiency_training_runs.csv`, `efficiency_scoring_jobs.csv`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
