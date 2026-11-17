"""Build the paper tables from saved Qwen3-VL prediction exports.

Read-only on every input. Writes only under --output-dir.

Result sets (never mixed in one table without a `result_set` column):
  A  duplicate-excluded APTOS suite, 4-bit base, 361 test images
  B  previous train/val split control, 4-bit base, same 361 test images
  C  BF16-base control, 361 test images
  D  original-split CLAHE pairs, 367 test images (same-seed repeats)
  E  Retraining 2 saved-adapter replay, 367 APTOS + 1,057 archived Messidor
  F  final queue final_01 (zero-shot, inference benchmark, original adapter audit, archived Messidor
     for all models, learning-rate and alpha studies); written to output/final_01 only

Usage (from the repo root):
  C:\\Users\\moht2\\.conda\\envs\\afterimage_env\\python.exe -I finetuning\\paper_tables\\build_paper_tables.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RUNS = Path(r"D:\Afterimage_Experiments\runs")
DEFAULT_ORIGINAL = REPO / "finetuning/Qwen3/artifacts/paper_reproduction/training_batches/run02_reproduction_01"
DEFAULT_REFERENCE = REPO / "finetuning/Qwen3/artifacts/paper_reproduction/reference_run_02"
DEFAULT_OUTPUT = HERE / "output"
DEFAULT_DATASETS = Path(r"D:\Afterimage_Experiments\datasets")
DEFAULT_FINAL = Path(r"D:\Afterimage_Experiments\final_runs\final_01")

GRADES = list(range(5))
LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
PROB_COLS = [f"p{g}" for g in GRADES]
TOL = 1e-9
BOOT_N = 2000
BOOT_SEED = 20261115
SET_NAMES = {
    "A": "A: duplicate-excluded suite (4-bit)",
    "B": "B: previous train/val split control (4-bit)",
    "C": "C: BF16-base control",
    "D": "D: original-split CLAHE pairs (same-seed repeats)",
    "E": "E: Retraining 2 replay",
}

INPUTS: dict[str, str] = {}  # path -> sha256 of every input file read


class BuildError(RuntimeError):
    pass


# --------------------------------------------------------------------------- I/O helpers

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def track(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise BuildError(f"Missing expected file: {path}")
    INPUTS[str(path)] = sha256(path)
    return path


def read_json(path: Path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(track(path), dtype={"image_id": str})
    needed = ["image_id", "y_true", "y_pred", *PROB_COLS]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise BuildError(f"{path}: missing columns {missing}")
    if frame.empty:
        raise BuildError(f"{path}: prediction file is empty")
    if frame["image_id"].duplicated().any():
        raise BuildError(f"{path}: duplicate image IDs")
    probs = frame[PROB_COLS].to_numpy(float)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1, atol=1e-6):
        raise BuildError(f"{path}: invalid probability vectors")
    if not np.array_equal(frame["y_pred"].to_numpy(int), probs.argmax(1)):
        raise BuildError(f"{path}: y_pred differs from the probability argmax")
    return frame


# --------------------------------------------------------------------------- metrics

def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def confusion(y_true, y_pred, n_classes: int = 5) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return matrix


def qwk_from_confusion(matrix) -> np.ndarray | float:
    """Quadratic weighted kappa. Accepts (K,K) or (B,K,K). NaN when undefined."""
    matrix = np.asarray(matrix, dtype=float)
    k = matrix.shape[-1]
    weights = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    total = matrix.sum(axis=(-2, -1))
    expected = matrix.sum(-1)[..., :, None] * matrix.sum(-2)[..., None, :] / total[..., None, None]
    denominator = (weights * expected).sum(axis=(-2, -1))
    numerator = (weights * matrix).sum(axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator > 0, 1 - numerator / denominator, np.nan)
    return float(result) if result.ndim == 0 else result


def qwk(y_true, y_pred, n_classes: int = 5) -> float:
    return qwk_from_confusion(confusion(y_true, y_pred, n_classes))


def weighted_auc(positive, scores, weights) -> np.ndarray:
    """One-vs-rest ROC-AUC with ties counted as one half (Mann-Whitney).

    positive: (n,) bool; scores: (n,); weights: (n,) or (B, n) case multiplicities.
    Returns AUC per weight row; NaN where positives or negatives are absent.
    """
    positive = np.asarray(positive, bool)
    scores = np.asarray(scores, float)
    weights = np.atleast_2d(np.asarray(weights, float))
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    pos_w = np.where(positive[order], weights[:, order], 0.0)
    neg_w = np.where(~positive[order], weights[:, order], 0.0)
    pos_g = np.add.reduceat(pos_w, starts, axis=1)
    neg_g = np.add.reduceat(neg_w, starts, axis=1)
    neg_below = np.cumsum(neg_g, axis=1) - neg_g
    numerator = (pos_g * (neg_below + 0.5 * neg_g)).sum(1)
    denominator = pos_g.sum(1) * neg_g.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def auc_ovr(y_true, scores, grade) -> float:
    y_true = np.asarray(y_true)
    return float(weighted_auc(y_true == grade, scores, np.ones(len(y_true)))[0])


def class_rows(matrix: np.ndarray, aucs) -> list[dict]:
    rows = []
    total = matrix.sum()
    for g in GRADES:
        tp = matrix[g, g]
        fn = matrix[g].sum() - tp
        fp = matrix[:, g].sum() - tp
        tn = total - tp - fn - fp
        support = tp + fn
        predicted = tp + fp
        rows.append({
            "grade": g, "label": LABELS[g], "support": int(support), "predicted": int(predicted),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "sensitivity": tp / support if support else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "precision": tp / predicted if predicted else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
            "auc_ovr": aucs[g],
        })
    return rows


def point_metrics(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(int)
    p = frame["y_pred"].to_numpy(int)
    probs = frame[PROB_COLS].to_numpy(float)
    aucs = [auc_ovr(y, probs[:, g], g) for g in GRADES]
    matrix = confusion(y, p)
    return {"n": len(y), "accuracy": accuracy(y, p), "qwk": qwk_from_confusion(matrix),
            "macro_auc": float(np.mean(aucs)) if not np.isnan(aucs).any() else np.nan,
            "aucs": aucs, "confusion": matrix}


def bootstrap(frame: pd.DataFrame, rng_seed: int = BOOT_SEED, n_boot: int = BOOT_N) -> dict:
    """Percentile 95% CIs. Same seed for every run, so resample indices repeat for equal n.

    A resample with an absent class has an undefined AUC for that class; its macro AUC is
    treated as undefined and excluded (the saved scorers also require all five classes).
    A resample with undefined QWK (zero expected weighted disagreement) is excluded from the QWK CI.
    """
    y = frame["y_true"].to_numpy(int)
    p = frame["y_pred"].to_numpy(int)
    probs = frame[PROB_COLS].to_numpy(float)
    n = len(y)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1)
    correct = (y == p).astype(float)
    acc = counts @ correct / n
    cell = y * 5 + p
    cm = np.zeros((n_boot, 25))
    np.add.at(cm, (np.repeat(np.arange(n_boot), n), cell[idx].ravel()), 1)
    kappa = qwk_from_confusion(cm.reshape(n_boot, 5, 5))
    aucs = np.stack([weighted_auc(y == g, probs[:, g], counts) for g in GRADES], axis=1)
    macro = np.where(np.isnan(aucs).any(1), np.nan, aucs.mean(1))

    def ci(values):
        values = values[~np.isnan(values)]
        if values.size == 0:
            return (np.nan, np.nan)
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    return {
        "accuracy_ci": ci(acc), "qwk_ci": ci(kappa), "macro_auc_ci": ci(macro),
        "boot_qwk_undefined": int(np.isnan(kappa).sum()), "boot_auc_undefined": int(np.isnan(macro).sum()),
    }


def check(name: str, what: str, ours, saved):
    if saved is None:
        raise BuildError(f"{name}: saved {what} is missing")
    if not math.isclose(float(ours), float(saved), rel_tol=0, abs_tol=TOL):
        raise BuildError(f"{name}: {what} mismatch: recomputed {ours!r}, saved {saved!r}")


# --------------------------------------------------------------------------- discovery

RUN_REPORT_STAGE = re.compile(r"^- (train|score): exit (\S+), ([\d.]+) s", re.M)
RUN_REPORT_PEAK = re.compile(r"^- (train|score): \d+ samples, .*peak device VRAM ([\d.]+) MiB, peak system RAM ([\d.]+) GiB", re.M)
TRAINABLE = re.compile(r"Trainable parameters = ([\d,]+) of ([\d,]+)")


def logged_parameters(attempt_dirs: list[Path]):
    for attempt in attempt_dirs:
        log = attempt / "train.log"
        if log.is_file():
            match = TRAINABLE.search(track(log).read_text(encoding="utf-8", errors="replace"))
            if match:
                return int(match.group(1).replace(",", "")), int(match.group(2).replace(",", "")), attempt.name
    return None, None, None


def best_epoch_from_history(train_result: dict):
    step = int(Path(train_result["checkpoint"]).name.split("-")[-1])
    for entry in train_result["history"]:
        if "eval_loss" in entry and int(entry.get("step", -1)) == step:
            return float(entry["epoch"]), step
    raise BuildError(f"Best checkpoint step {step} has no matching validation entry")


def discover_pilot_runs(runs_root: Path) -> list[dict]:
    runs = []
    folders = sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "pilot_configuration.json").exists())
    if not folders:
        raise BuildError(f"No run folders with pilot_configuration.json under {runs_root}")
    for run in folders:
        name = run.name
        cfg = read_json(run / "pilot_configuration.json")
        if cfg.get("mode") != "full":
            continue
        status = read_json(run / "latest_attempt.json").get("status")
        if status != "passed":
            raise BuildError(f"{name}: latest_attempt.json status is {status!r}, not 'passed'")
        training = cfg["training"]
        four_bit = bool(training["load_in_4bit"])
        historical = bool(cfg["historical_split"])
        result_set = "B" if historical else ("A" if four_bit else "C")
        if not four_bit and historical:
            raise BuildError(f"{name}: unexpected BF16 previous-split configuration")
        split = Path(cfg["candidate_manifest"]).parent.name
        score = read_json(run / "score_result.json")
        train = read_json(run / "train_result.json")
        pred_path = run / "aptos_test.predictions.csv"
        frame = read_predictions(pred_path)
        if INPUTS[str(pred_path)] != score["predictions_sha256"]:
            raise BuildError(f"{name}: prediction SHA-256 {INPUTS[str(pred_path)]} != recorded {score['predictions_sha256']}")
        expected_rows = cfg["rows"]["test"]
        if len(frame) != expected_rows:
            raise BuildError(f"{name}: {len(frame)} predictions, configuration expects {expected_rows}")
        best_epoch, best_step = best_epoch_from_history(train)
        attempts = sorted((run / "attempts").iterdir()) if (run / "attempts").is_dir() else []
        latest = Path(read_json(run / "latest_attempt.json")["path"]).name
        latest_dir = run / "attempts" / latest
        ordered = [latest_dir] + [a for a in reversed(attempts) if a != latest_dir]
        trainable, total, param_source = logged_parameters(ordered)
        report = track(run / "run_report.md").read_text(encoding="utf-8")
        times = {m[0]: float(m[2]) for m in RUN_REPORT_STAGE.findall(report)}
        peaks = {m[0]: (float(m[1]), float(m[2])) for m in RUN_REPORT_PEAK.findall(report)}
        if set(times) != {"train", "score"} or set(peaks) != {"train", "score"}:
            raise BuildError(f"{name}: run_report.md lacks train/score time or peak lines")
        precision = score.get("scoring_precision", "FP16")
        runs.append({
            "result_set": result_set, "run": name, "frame": frame,
            "rank": int(cfg["rank"]), "alpha": float(training["lora_alpha"]),
            "clahe": "on" if cfg["clahe"] else "off", "split": split,
            "base": "4-bit" if four_bit else "BF16",
            "scoring_precision": precision,
            "scoring_precision_source": "score_result.json" if "scoring_precision" in score
            else "not recorded; qwen_pilot.py score() casts model.half()",
            "trainer_seed": int(training["trainer_seed"]), "lora_seed": int(cfg["adapter_seed"]),
            "best_epoch": best_epoch, "best_step": best_step, "best_val_loss": float(train["validation_loss"]),
            "saved": score["metrics"], "saved_per_grade_auc": score["metrics"].get("per_grade_auc"),
            "efficiency": {
                "attempts": len(attempts), "final_attempt": latest,
                "trainable_params_logged": trainable, "total_params_logged": total, "param_log_attempt": param_source,
                "train_wall_s": times["train"], "score_wall_s": times["score"],
                "train_peak_device_vram_mib": peaks["train"][0], "score_peak_device_vram_mib": peaks["score"][0],
                "train_peak_system_ram_gib": peaks["train"][1], "score_peak_system_ram_gib": peaks["score"][1],
                "train_torch_peak_alloc_bytes": train.get("torch_peak_allocated_bytes"),
                "score_torch_peak_alloc_bytes": score.get("torch_peak_allocated_bytes"),
                "offload_layers": cfg.get("offload_layers"), "n_scored": len(frame),
            },
        })
    return runs


def attempt_for(run: Path, stage: str, output_suffix: str | None = None) -> Path | None:
    chosen = None
    for attempt in sorted((run / "attempts").glob(f"{stage}-*")):
        command = read_json(attempt / "command.json")
        report = attempt / "run_report.json"
        if not report.is_file() or read_json(report).get("status") != "complete":
            continue
        if output_suffix and not any(str(part).endswith(output_suffix) for part in command):
            continue
        chosen = attempt
    return chosen


def discover_original_split(root: Path) -> list[dict]:
    runs = []
    folders = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"clahe_(on|off)_\d\d", p.name))
    if not folders:
        raise BuildError(f"No clahe_on/off_NN folders under {root}")
    for run in folders:
        name = run.name
        cfg = read_json(run / "configuration.json")
        info = read_json(run / "run_info.json")
        read_json(run / "training_complete.json")
        manifest = read_json(Path(cfg["dataset_manifest"]))
        clahe = manifest.get("clahe")
        if clahe not in {"on", "off"}:
            raise BuildError(f"{name}: dataset manifest has no clahe on/off field")
        evaluation = run / "evaluation_bf16"
        summary = read_json(evaluation / "aptos_test.summary.json")
        frame = read_predictions(evaluation / "aptos_test.predictions.csv")
        if summary.get("samples") != len(frame):
            raise BuildError(f"{name}: summary samples {summary.get('samples')} != {len(frame)} predictions")
        training = cfg["training"]
        train_attempt = attempt_for(run, "train")
        eval_attempt = attempt_for(run, "evaluate", "evaluation_bf16")
        if train_attempt is None or eval_attempt is None:
            raise BuildError(f"{name}: no completed train or BF16 evaluate attempt")
        train_report, eval_report = read_json(train_attempt / "run_report.json"), read_json(eval_attempt / "run_report.json")
        trainable, total, param_source = logged_parameters([train_attempt])
        runs.append({
            "result_set": "D", "run": name, "frame": frame,
            "rank": int(training["lora_r"]), "alpha": float(training["lora_alpha"]), "clahe": clahe,
            "split": "original_2929_366_367", "base": "4-bit" if training["load_in_4bit"] else "BF16",
            "scoring_precision": summary["evaluation_precision"], "scoring_precision_source": "aptos_test.summary.json",
            "trainer_seed": int(info["trainer_seed"]), "lora_seed": int(info["resolved_lora_default_seed"]),
            "best_epoch": float(info["best_epoch"]), "best_step": info.get("best_step"),
            "best_val_loss": float(info["best_eval_loss"]),
            "saved": summary, "saved_per_grade_auc": [r["auc"] for r in summary["per_class"]],
            "efficiency": {
                "attempts": len(list((run / "attempts").iterdir())), "final_attempt": f"{train_attempt.name}; {eval_attempt.name}",
                "trainable_params_logged": trainable, "total_params_logged": total, "param_log_attempt": param_source,
                "train_wall_s": train_report["elapsed_seconds"], "score_wall_s": eval_report["elapsed_seconds"],
                "train_peak_device_vram_mib": train_report.get("peak_vram_mib"),
                "score_peak_device_vram_mib": eval_report.get("peak_vram_mib"),
                "train_peak_system_ram_gib": train_report.get("peak_ram_bytes", np.nan) / 2**30,
                "score_peak_system_ram_gib": eval_report.get("peak_ram_bytes", np.nan) / 2**30,
                "train_torch_peak_alloc_bytes": None, "score_torch_peak_alloc_bytes": None,
                "offload_layers": None, "n_scored": len(frame),
            },
        })
    return runs


def discover_replay(root: Path, reference: Path) -> tuple[dict, pd.DataFrame, dict]:
    replay = root / "replay_run_02"
    evaluation = replay / "evaluation"
    cfg = read_json(reference / "paper_training_configuration.json")
    command = read_json(reference / "evaluation.command.json")
    run_match = [re.search(r"run_(\d+)", str(part)) for part in command]
    run_ids = {int(m.group(1)) for m in run_match if m}
    if len(run_ids) != 1:
        raise BuildError(f"Cannot identify the archived run number from {reference / 'evaluation.command.json'}")
    run_id = run_ids.pop()
    comparison = pd.read_csv(track(reference / "aptos_run_comparison.csv"))
    row = comparison.loc[comparison["run"] == run_id]
    if len(row) != 1:
        raise BuildError(f"aptos_run_comparison.csv has no single row for run {run_id}")
    row = row.iloc[0]
    replay_check = read_json(replay / "replay_comparison.json")
    cohorts = {}
    for cohort in ("aptos_test", "messidor_external"):
        summary = read_json(evaluation / f"{cohort}.summary.json")
        archived_summary = read_json(reference / f"{cohort}.summary.json")
        frame = read_predictions(evaluation / f"{cohort}.predictions.csv")
        archived = read_predictions(reference / f"{cohort}.predictions.csv")
        if list(frame["image_id"]) != list(archived["image_id"]) or not np.array_equal(frame["y_pred"], archived["y_pred"]):
            raise BuildError(f"Replay {cohort}: IDs or grades differ from the archived export")
        if replay_check[cohort]["grade_mismatches"]:
            raise BuildError(f"Replay {cohort}: replay_comparison.json records grade mismatches")
        cohorts[cohort] = (frame, summary, archived_summary)
    attempt = attempt_for(replay, "evaluate")
    eval_report = read_json(attempt / "run_report.json")
    training = cfg["training"]
    aptos, aptos_summary, aptos_archived = cohorts["aptos_test"]
    run = {
        "result_set": "E", "run": f"retraining2_replay (archived run_{run_id:02})", "frame": aptos,
        "rank": int(training["lora_r"]), "alpha": float(training["lora_alpha"]), "clahe": "on (archived inputs)",
        "split": "original_2929_366_367", "base": "4-bit" if training["load_in_4bit"] else "BF16",
        "scoring_precision": aptos_summary["evaluation_precision"], "scoring_precision_source": "aptos_test.summary.json",
        "trainer_seed": int(training["trainer_seed"]), "lora_seed": int(training["expected_lora_default_seed"]),
        "best_epoch": float(row["best_epoch"]), "best_step": None, "best_val_loss": float(row["best_eval_loss"]),
        "saved": aptos_summary, "saved_archived": aptos_archived,
        "saved_per_grade_auc": [r["auc"] for r in aptos_summary["per_class"]],
        "efficiency": {
            "attempts": len(list((replay / "attempts").iterdir())), "final_attempt": attempt.name,
            "trainable_params_logged": None, "total_params_logged": None, "param_log_attempt": None,
            "train_wall_s": None, "score_wall_s": eval_report["elapsed_seconds"],
            "train_peak_device_vram_mib": None, "score_peak_device_vram_mib": eval_report.get("peak_vram_mib"),
            "train_peak_system_ram_gib": None, "score_peak_system_ram_gib": eval_report.get("peak_ram_bytes", np.nan) / 2**30,
            "train_torch_peak_alloc_bytes": None, "score_torch_peak_alloc_bytes": None, "offload_layers": None,
            "n_scored": len(aptos) + len(cohorts["messidor_external"][0]),
        },
    }
    return run, cohorts["messidor_external"][0], {"summary": cohorts["messidor_external"][1],
                                                  "archived": cohorts["messidor_external"][2]}


# --------------------------------------------------------------------------- per-run computation

def evaluate_run(run: dict) -> dict:
    metrics = point_metrics(run["frame"])
    name = f"{run['result_set']}/{run['run']}"
    saved = run["saved"]
    check(name, "accuracy", metrics["accuracy"], saved.get("accuracy"))
    check(name, "QWK", metrics["qwk"], saved.get("qwk"))
    check(name, "macro AUC", metrics["macro_auc"], saved.get("macro_ovr_auc"))
    check(name, "n", metrics["n"], saved.get("samples"))
    for g, value in enumerate(run["saved_per_grade_auc"] or []):
        check(name, f"grade {g} AUC", metrics["aucs"][g], value)
    if "saved_archived" in run:
        for key, saved_key in (("accuracy", "accuracy"), ("qwk", "qwk"), ("macro_auc", "macro_ovr_auc")):
            check(name + " vs archive", key, metrics[key], run["saved_archived"][saved_key])
    if "confusion_matrix" in saved:
        if not np.array_equal(metrics["confusion"], np.asarray(saved["confusion_matrix"])):
            raise BuildError(f"{name}: confusion matrix differs from the saved summary")
    metrics["boot"] = bootstrap(run["frame"])
    return metrics


def condition_of(run: dict) -> str:
    if run["result_set"] == "E":
        return "E r16 CLAHE-on (archived) 4-bit"
    return f"{run['result_set']} r{run['rank']} CLAHE-{run['clahe']} {run['base']}"


def per_run_row(run: dict, metrics: dict) -> dict:
    predicted = metrics["confusion"].sum(0)
    never = [g for g in GRADES if predicted[g] == 0]
    row = {
        "result_set": run["result_set"], "condition": condition_of(run), "run": run["run"],
        "rank": run["rank"], "alpha": run["alpha"], "alpha_over_r": run["alpha"] / run["rank"],
        "clahe": run["clahe"], "split": run["split"], "base": run["base"],
        "scoring_precision": run["scoring_precision"], "scoring_precision_source": run["scoring_precision_source"],
        "trainer_seed": run["trainer_seed"], "lora_seed": run["lora_seed"], "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "accuracy_ci_low": metrics["boot"]["accuracy_ci"][0], "accuracy_ci_high": metrics["boot"]["accuracy_ci"][1],
        "qwk": metrics["qwk"], "qwk_ci_low": metrics["boot"]["qwk_ci"][0], "qwk_ci_high": metrics["boot"]["qwk_ci"][1],
        "macro_auc": metrics["macro_auc"],
        "macro_auc_ci_low": metrics["boot"]["macro_auc_ci"][0], "macro_auc_ci_high": metrics["boot"]["macro_auc_ci"][1],
        "boot_qwk_undefined": metrics["boot"]["boot_qwk_undefined"], "boot_auc_undefined": metrics["boot"]["boot_auc_undefined"],
    }
    row.update({f"auc_g{g}": metrics["aucs"][g] for g in GRADES})
    row.update({f"n_pred_g{g}": int(predicted[g]) for g in GRADES})
    row.update({
        "n_grades_predicted": int((predicted > 0).sum()), "degenerate": bool((predicted > 0).sum() == 1),
        "grades_never_predicted": ";".join(map(str, never)),
        "best_epoch": run["best_epoch"], "best_val_loss": run["best_val_loss"],
    })
    return row


def classwise_frame(run_rows: list[tuple[dict, dict]]) -> pd.DataFrame:
    rows = []
    for run, metrics in run_rows:
        for item in class_rows(metrics["confusion"], metrics["aucs"]):
            rows.append({"result_set": run["result_set"], "condition": condition_of(run), "run": run["run"], **item})
    return pd.DataFrame(rows)


def classwise_mean(frame: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for (result_set, condition, grade, label), group in frame.groupby(["result_set", "condition", "grade", "label"], sort=True):
        row = {"result_set": result_set, "condition": condition, "grade": grade, "label": label, "n_runs": len(group)}
        for col in ("sensitivity", "specificity", "precision", "f1", "auc_ovr"):
            values = group[col].dropna()
            row[f"{col}_mean"] = values.mean() if len(values) else np.nan
            row[f"{col}_n_defined"] = len(values)
        stats.append(row)
    return pd.DataFrame(stats)


def confusion_long(run_rows: list[tuple[dict, dict]]) -> pd.DataFrame:
    rows = []
    for run, metrics in run_rows:
        for g in GRADES:
            rows.append({"result_set": run["result_set"], "condition": condition_of(run), "run": run["run"],
                         "true_grade": g, **{f"pred_{p}": int(metrics["confusion"][g, p]) for p in GRADES}})
    return pd.DataFrame(rows)


def pooled_confusion(run_rows) -> pd.DataFrame:
    pooled: dict[tuple, np.ndarray] = {}
    counts: dict[tuple, int] = {}
    for run, metrics in run_rows:
        key = (run["result_set"], condition_of(run))
        pooled[key] = pooled.get(key, 0) + metrics["confusion"]
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for key in sorted(pooled):
        for g in GRADES:
            rows.append({"result_set": key[0], "condition": key[1], "n_runs": counts[key], "true_grade": g,
                         **{f"pred_{p}": int(pooled[key][g, p]) for p in GRADES}})
    return pd.DataFrame(rows)


def condition_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["result_set", "condition", "rank", "alpha_over_r", "clahe", "split", "base", "scoring_precision", "n"]
    for key, group in per_run.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["n_runs"] = len(group)
        row["n_degenerate"] = int(group["degenerate"].sum())
        row["runs_missing_a_grade"] = int((group["n_grades_predicted"] < 5).sum())
        for metric in ("accuracy", "qwk", "macro_auc"):
            values = group[metric]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["result_set", "rank", "clahe", "base"], ascending=[True, True, False, True],
                             kind="stable").reset_index(drop=True)


def overlap_facts(datasets_root: Path) -> dict:
    """Exact-duplicate (identical original-file SHA-256) overlap counts from the saved pairing audits."""
    previous = pd.read_csv(track(datasets_root / "aptos_512/previous_split/pairing_audit.csv"), dtype={"image_id": str})
    clean = pd.read_csv(track(datasets_root / "aptos_512/duplicate_excluded/pairing_audit.csv"), dtype={"image_id": str})
    by_split = {s: set(previous.loc[previous["split"] == s, "original_sha256"]) for s in ("train", "val", "test")}
    old_test = previous[previous["split"] == "test"]
    clean_test_sha = previous.set_index("image_id").loc[clean.loc[clean["split"] == "test", "image_id"], "original_sha256"]
    return {
        "original_test_n": len(old_test),
        "original_test_identical_to_train": int(old_test["original_sha256"].isin(by_split["train"]).sum()),
        "original_test_identical_to_val": int(old_test["original_sha256"].isin(by_split["val"]).sum()),
        "clean_test_n": len(clean_test_sha),
        "clean_test_identical_to_previous_train": int(clean_test_sha.isin(by_split["train"]).sum()),
        "clean_test_identical_to_previous_val": int(clean_test_sha.isin(by_split["val"]).sum()),
    }


def paired(per_run: pd.DataFrame, left: dict, right: dict, pair_on: list[str], label: str) -> pd.DataFrame:
    def select(spec):
        mask = np.ones(len(per_run), bool)
        for key, value in spec.items():
            mask &= per_run[key] == value
        return per_run[mask]

    a, b = select(left), select(right)
    merged = a.merge(b, on=pair_on, suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise BuildError(f"{label}: unpaired runs between the two arms")
    rows = []
    for _, r in merged.sort_values(pair_on).iterrows():
        row = {"comparison": label, **{k: r[k] for k in pair_on},
               "left_run": r["run_left"], "right_run": r["run_right"],
               "left_degenerate": r["degenerate_left"], "right_degenerate": r["degenerate_right"]}
        for metric in ("accuracy", "qwk", "macro_auc"):
            row[f"{metric}_left"] = r[f"{metric}_left"]
            row[f"{metric}_right"] = r[f"{metric}_right"]
            row[f"{metric}_diff_right_minus_left"] = r[f"{metric}_right"] - r[f"{metric}_left"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- final queue (set F, final_01)

FALLBACK = "FP32 fallback"
NONFINITE = "Non-finite answer likelihood"   # final_runs.py: marker of a failed FP16 score
WARMUP_IMAGES = 3                             # score_cohort.py: first images left out of every timing summary
GROUP_FP16 = "FP16"
GROUP_FP16_ONLY = "FP16 only (FP32-fallback seed excluded)"
GROUP_MIXED = "FP16 + FP32 fallback (mixed)"
REFERENCE_SOURCE = "set A runs/ (lr 5e-4, alpha 16)"
FINAL_SOURCE = "final_01"
SCALING_NOTE = ("train_result.json effective_scaling = {recorded:g} (stored as 16/rank, record error); "
                "true alpha/r = {true:g} from adapter_config.json")


def same(ours, saved, tol=TOL) -> bool:
    """Equal within tol; None and NaN count as 'undefined' and only match each other."""
    a = np.nan if ours is None else float(ours)
    b = np.nan if saved is None else float(saved)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def check_same(name: str, what: str, ours, saved, tol=TOL):
    if not same(ours, saved, tol):
        raise BuildError(f"{name}: {what} mismatch: recomputed {ours!r}, saved {saved!r}")


def path_key(path) -> str:
    return str(Path(path)).rstrip("\\/").lower()


def timing_stats(seconds) -> dict:
    """Same summary as score_cohort.timing_summary: the first WARMUP_IMAGES are left out."""
    seconds = [float(v) for v in seconds]
    measured = seconds[WARMUP_IMAGES:] if len(seconds) > WARMUP_IMAGES else seconds
    ordered = sorted(measured)
    return {"images_timed": len(measured), "mean_seconds": statistics.fmean(measured),
            "median_seconds": statistics.median(measured),
            "p95_seconds": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
            "images_per_second": len(measured) / sum(measured)}


def lora_scaling(alpha: float, rank: int, recorded: float) -> tuple[float, str]:
    """True LoRA scaling alpha/r and a note when train_result.json records something else.

    train_result.json stores effective_scaling as 16/rank whatever the alpha, so it is wrong for
    alpha != 16. Any other disagreement is unexplained and stops the build.
    """
    true = alpha / rank
    if math.isclose(recorded, true, rel_tol=0, abs_tol=TOL):
        return true, ""
    if math.isclose(recorded, 16 / rank, rel_tol=0, abs_tol=TOL):
        return true, SCALING_NOTE.format(recorded=recorded, true=true)
    raise BuildError(f"effective_scaling {recorded!r} is neither alpha/r = {true!r} nor 16/r = {16 / rank!r}")


def training_info(run: Path) -> dict:
    """Training configuration of a saved run, with the LoRA scaling taken from adapter_config.json."""
    cfg = read_json(run / "pilot_configuration.json")
    train = read_json(run / "train_result.json")
    adapter_cfg = read_json(run / "best_lora" / "adapter_config.json")
    training = cfg["training"]
    rank, alpha = int(cfg["rank"]), float(training["lora_alpha"])
    if int(adapter_cfg["r"]) != rank or int(train["adapter"]["rank"]) != rank or int(training["lora_r"]) != rank:
        raise BuildError(f"{run.name}: rank differs between configuration, train_result.json and adapter_config.json")
    if float(adapter_cfg["lora_alpha"]) != alpha:
        raise BuildError(f"{run.name}: adapter_config.json lora_alpha {adapter_cfg['lora_alpha']} != configured {alpha}")
    if adapter_cfg.get("use_rslora") or adapter_cfg.get("alpha_pattern") or adapter_cfg.get("rank_pattern"):
        raise BuildError(f"{run.name}: adapter uses rsLoRA or per-module alpha/rank; alpha/r does not apply")
    recorded = float(train["adapter"]["effective_scaling"])
    try:
        scaling, note = lora_scaling(alpha, rank, recorded)
    except BuildError as error:
        raise BuildError(f"{run.name}: {error}") from None
    best_epoch, best_step = best_epoch_from_history(train)
    return {
        "source_run": run.name, "base": "4-bit" if training["load_in_4bit"] else "BF16", "rank": rank,
        "lr": float(training["lr"]), "alpha": alpha, "lora_dropout": float(training["lora_dropout"]),
        "patience": int(training["early_stopping_patience"]), "clahe_trained": "on" if cfg["clahe"] else "off",
        "trainer_seed": int(training["trainer_seed"]), "lora_seed": int(cfg["adapter_seed"]),
        "effective_scaling": scaling, "recorded_effective_scaling": recorded, "scaling_note": note,
        "best_epoch": best_epoch, "best_step": best_step, "best_val_loss": float(train["validation_loss"]),
        "epochs_run": max(float(entry.get("epoch", 0)) for entry in train["history"]),
        "adapter_weights_sha256": train["adapter"]["weights_sha256"],
        "adapter_config_sha256": INPUTS[str(run / "best_lora" / "adapter_config.json")],
        "train_result_sha256": INPUTS[str(run / "train_result.json")],
    }


def condition_label(info: dict) -> str:
    return f"{info['base']} r{info['rank']} CLAHE-{info['clahe_trained']} lr {info['lr']:g} alpha {info['alpha']:g}"


def check_saved_metrics(name: str, metrics: dict, saved: dict):
    for key in ("accuracy", "qwk", "macro_ovr_auc", "samples"):
        if key not in saved:
            raise BuildError(f"{name}: saved metrics lack {key}")
    check_same(name, "accuracy", metrics["accuracy"], saved["accuracy"])
    check_same(name, "QWK", metrics["qwk"], saved["qwk"])
    check_same(name, "macro AUC", metrics["macro_auc"], saved["macro_ovr_auc"])
    check(name, "n", metrics["n"], saved["samples"])
    aucs = saved.get("per_grade_auc") or [item["auc"] for item in saved.get("per_class", [])]
    if len(aucs) != 5:
        raise BuildError(f"{name}: saved metrics lack the five per-grade AUCs")
    for g, value in enumerate(aucs):
        check_same(name, f"grade {g} AUC", metrics["aucs"][g], value)
    if "confusion_matrix" in saved and not np.array_equal(metrics["confusion"], np.asarray(saved["confusion_matrix"])):
        raise BuildError(f"{name}: confusion matrix differs from the saved summary")


def compare_reference(name: str, frame: pd.DataFrame, record: dict) -> dict:
    """Recompute a saved 'agreement with reference predictions' block from the two prediction files."""
    path = Path(record["reference"])
    reference = pd.read_csv(track(path), dtype={"image_id": str})
    if INPUTS[str(path)] != record["reference_sha256"]:
        raise BuildError(f"{name}: reference {path} SHA-256 differs from the recorded one")
    reference = reference.set_index("image_id")
    if not set(frame["image_id"]) <= set(reference.index):
        raise BuildError(f"{name}: reference {path} lacks some image IDs")
    reference = reference.loc[frame["image_id"]]
    mismatches = int((reference["y_pred"].to_numpy(int) != frame["y_pred"].to_numpy(int)).sum())
    largest = float(np.abs(reference[PROB_COLS].to_numpy(float) - frame[PROB_COLS].to_numpy(float)).max())
    check(name, "reference images compared", len(frame), record["images_compared"])
    check(name, "reference grade mismatches", mismatches, record["grade_mismatches"])
    check(name, "reference largest probability difference", largest, record["largest_probability_difference"])
    return {"images": len(frame), "grade_mismatches": mismatches, "largest_probability_difference": largest}


def read_score_job(folder: Path, sources: set[str]) -> dict:
    """One scores/<job> folder: predictions, timing and generation recomputed and checked against summary.json."""
    name = folder.name
    summary = read_json(folder / "summary.json")
    identity = summary["identity"]
    if identity["name"] != name:
        raise BuildError(f"{name}: summary.json names job {identity['name']!r}")
    if identity.get("limit"):
        raise BuildError(f"{name}: limited (pilot) score")
    pred_path = folder / "predictions.csv"
    frame = read_predictions(pred_path)
    if INPUTS[str(pred_path)] != summary["predictions_sha256"]:
        raise BuildError(f"{name}: prediction SHA-256 differs from summary.json")
    check(name, "images", len(frame), identity["images"])
    metrics = point_metrics(frame)
    check_saved_metrics(name, metrics, summary["metrics"])

    timing = pd.read_csv(track(folder / "timing.csv"), dtype={"image_id": str, "generated_text": str})
    if list(timing["image_id"]) != list(frame["image_id"]):
        raise BuildError(f"{name}: timing.csv and predictions.csv list different images")
    full = timing_stats(timing["full_answer_seconds"])
    for key, value in full.items():
        check_same(name, f"likelihood timing {key}", value, summary["full_answer_timing"][key])
    times = {"likelihood_" + key.replace("_seconds", "_s"): value for key, value in full.items()}
    times.update(model_load_s=float(summary["model_load_seconds"]),
                 torch_peak_alloc_gib=summary["torch_peak_allocated_bytes"] / 2**30)

    wanted = int(identity["generation_images"])
    generation = {}
    saved_gen = summary.get("generation")
    if wanted:
        if not saved_gen or int(saved_gen["images"]) != wanted:
            raise BuildError(f"{name}: generation block missing or for a different image count")
        if timing["generation_seconds"].iloc[wanted:].notna().any():
            raise BuildError(f"{name}: generation timings beyond the generated images")
        part = timing.iloc[:wanted]
        parsed = part["generated_grade"].notna().to_numpy()
        grades = part["generated_grade"].fillna(-1).to_numpy(int)
        y_true = frame["y_true"].to_numpy(int)[:wanted]
        y_pred = frame["y_pred"].to_numpy(int)[:wanted]
        generation = {
            "generation_images": wanted, "unparseable": int(wanted - parsed.sum()),
            "generation_accuracy_strict": float((parsed & (grades == y_true)).sum() / wanted),
            "generation_accuracy_unparseable_as_0": float((np.where(parsed, grades, 0) == y_true).sum() / wanted),
            "agreement_with_likelihood_grade": float((parsed & (grades == y_pred)).sum() / wanted),
            "prompt_tokens": int(saved_gen["prompt_tokens"]),
        }
        check(name, "unparseable generations", generation["unparseable"], saved_gen["unparseable_outputs"])
        for ours, theirs in (("generation_accuracy_strict", "accuracy_unparseable_counted_wrong"),
                             ("generation_accuracy_unparseable_as_0", "accuracy_unparseable_as_grade_0"),
                             ("agreement_with_likelihood_grade", "agreement_with_likelihood_grade")):
            check_same(name, ours, generation[ours], saved_gen[theirs])
        gen_times = timing_stats(part["generation_seconds"])
        for key, value in gen_times.items():
            check_same(name, f"generation timing {key}", value, saved_gen["timing"][key])
            generation["generation_" + key.replace("_seconds", "_s")] = value
    elif saved_gen:
        raise BuildError(f"{name}: generation block present but no images were generated")

    agreement = {}
    for key, label in (("agreement_with_saved_predictions", "saved"), ("agreement_with_diagnostic_predictions", "diagnostic")):
        if key in summary:
            for item, value in compare_reference(name, frame, summary[key]).items():
                agreement[f"{label}_ref_{item}"] = value

    base = {"4bit": "4-bit", "bf16": "BF16"}[identity["base_precision"]]
    adapter = identity["adapter"]
    if adapter is None:
        model = {"condition": f"zero-shot {base}", "base": base, "adapter": "none"}
    else:
        run = Path(adapter["directory"]).parent
        if path_key(run.parent) in sources:
            info = training_info(run)
            if info["adapter_weights_sha256"] != adapter["weights_sha256"] or \
                    info["adapter_config_sha256"] != adapter["configuration_sha256"]:
                raise BuildError(f"{name}: scored adapter files differ from {run.name}'s saved adapter")
            if info["base"] != base:
                raise BuildError(f"{name}: scored on a {base} base, adapter trained on {info['base']}")
            model = {**info, "adapter": "yes", "condition": condition_label(info)}
        else:
            model = {"condition": "original paper adapter (legacy best_lora)", "base": base, "adapter": "yes",
                     "scaling_note": "legacy adapter outside the experiment folders; its files are not read here "
                                     "(identity hashes are in summary.json)"}
    model["offload_layers"] = int(identity["offload_layers"])
    precision = identity.get("scoring_precision", "FP16")
    fallback_for = Path(identity["fallback_for"]).name if identity.get("fallback_for") else ""
    if (precision == FALLBACK) != bool(fallback_for):
        raise BuildError(f"{name}: FP32 fallback label and fallback_for disagree")
    return {"job": name, "folder": folder, "kind": "score", "cohort": identity["cohort"], "frame": frame,
            "metrics": metrics, "precision": precision, "fallback_for": fallback_for, "model": model,
            "timing": times, "generation": generation, "agreement": agreement, "summary": summary}


def discover_final(final_root: Path, runs_root: Path) -> tuple[list[dict], dict]:
    """Every finished final_01 job. Returns (scored records, {train run whose FP16 score failed: info})."""
    history = read_json(final_root / "queue_state.json")
    last, stage_of = {}, {}
    for item in history:
        last[item["job"]], stage_of[item["job"]] = item["outcome"], item["stage"]
    score_dirs = sorted(p for p in (final_root / "scores").iterdir() if p.is_dir())
    train_dirs = sorted(p for p in (final_root / "train").iterdir() if p.is_dir())
    folders = {p.name for p in score_dirs} | {p.name for p in train_dirs}
    if set(last) != folders:
        raise BuildError(f"queue_state.json jobs and result folders differ: {sorted(set(last) ^ folders)}")
    sources = {path_key(runs_root), path_key(final_root / "train")}
    scores = {p.name: read_score_job(p, sources) for p in score_dirs}
    records, failures = [], {}
    for name, record in scores.items():
        if last[name] != "passed":
            raise BuildError(f"{name}: last queue outcome is {last[name]!r}")
        record["stage"] = stage_of[name]
        records.append(record)
    for run in train_dirs:
        name = run.name
        info = training_info(run)
        latest = read_json(run / "latest_attempt.json")
        model = {**info, "adapter": "yes", "condition": condition_label(info), "offload_layers": 0}
        if (run / "score_result.json").is_file():
            if last[name] != "passed" or latest["status"] != "passed":
                raise BuildError(f"{name}: score_result.json exists but the run did not pass")
            score = read_json(run / "score_result.json")
            cfg = read_json(run / "pilot_configuration.json")
            pred_path = run / "aptos_test.predictions.csv"
            frame = read_predictions(pred_path)
            if INPUTS[str(pred_path)] != score["predictions_sha256"]:
                raise BuildError(f"{name}: prediction SHA-256 differs from score_result.json")
            check(name, "test rows", len(frame), cfg["rows"]["test"])
            if score["adapter"]["weights_sha256"] != info["adapter_weights_sha256"]:
                raise BuildError(f"{name}: score_result.json adapter differs from train_result.json")
            check(name, "score_result.json effective_scaling", score["adapter"]["effective_scaling"],
                  info["recorded_effective_scaling"])
            metrics = point_metrics(frame)
            check_saved_metrics(name, metrics, score["metrics"])
            records.append({"job": name, "folder": run, "kind": "train", "stage": stage_of[name],
                            "cohort": f"aptos361_{info['clahe_trained']}", "frame": frame, "metrics": metrics,
                            "precision": "FP16", "fallback_for": "", "model": model,
                            "timing": {}, "generation": {}, "agreement": {}})
            continue
        # Training finished but the pinned FP16 score failed: the run must have a labelled FP32 fallback.
        if latest["status"] != "failed" or not last[name].startswith("failed"):
            raise BuildError(f"{name}: no score_result.json and the run is not recorded as failed")
        log = Path(latest["path"]) / "score.log"
        if NONFINITE not in track(log).read_text(encoding="utf-8", errors="replace"):
            raise BuildError(f"{name}: no score_result.json and {log} shows no non-finite FP16 likelihood")
        fallbacks = [r for r in scores.values() if r["fallback_for"] == name]
        if not any(r["cohort"] == f"aptos361_{info['clahe_trained']}" for r in fallbacks):
            raise BuildError(f"{name}: FP16 score failed and no APTOS FP32 fallback exists")
        for record in fallbacks:
            run_record = record["summary"]["training_run"]
            if run_record["train_result_sha256"] != info["train_result_sha256"]:
                raise BuildError(f"{record['job']}: fallback was made from a different train_result.json")
            check_same(record["job"], "fallback validation loss", run_record["validation_loss"], info["best_val_loss"])
        failures[name] = {**model, "stage": stage_of[name], "fallback_jobs": " ".join(r["job"] for r in fallbacks)}
    for record in records:
        record["in_fp16_mean"] = record["precision"] == "FP16"
        record["boot"] = bootstrap(record["frame"])
    return records, failures


def cross_check_queue_summary(final_root: Path, records: list[dict], failures: dict):
    """Every row of final_01/summary.csv must equal the recomputed values (no row may be missing)."""
    saved = pd.read_csv(track(final_root / "summary.csv"))
    by_job = {r["job"]: r for r in records}
    if set(saved["job"]) != set(by_job) | set(failures):
        raise BuildError(f"summary.csv jobs differ from the result folders: "
                         f"{sorted(set(saved['job']) ^ (set(by_job) | set(failures)))}")
    for row in saved.to_dict("records"):
        name = f"summary.csv {row['job']}"
        if row["job"] in failures:
            info = failures[row["job"]]
            if not str(row.get("scoring_status", "")).startswith("FP16 scoring failed"):
                raise BuildError(f"{name}: expected an FP16 scoring failure row")
            check_same(name, "best validation loss", info["best_val_loss"], row["best_validation_loss"])
            check_same(name, "epochs run", info["epochs_run"], row["epochs_run"])
            if set(str(row["fallback_jobs"]).split()) != set(info["fallback_jobs"].split()):
                raise BuildError(f"{name}: fallback jobs differ")
            continue
        record = by_job[row["job"]]
        metrics = record["metrics"]
        predicted = metrics["confusion"].sum(0)
        check(name, "images", metrics["n"], row["images"])
        check_same(name, "accuracy", metrics["accuracy"], row["accuracy"])
        check_same(name, "QWK", metrics["qwk"], row["qwk"])
        check_same(name, "macro AUC", metrics["macro_auc"], row["macro_auc"])
        if row["predicted_per_grade"] != " ".join(map(str, predicted)):
            raise BuildError(f"{name}: predicted_per_grade {row['predicted_per_grade']!r} != {predicted}")
        if str(row["constant_predictor"]) != str(bool((predicted > 0).sum() == 1)):
            raise BuildError(f"{name}: constant_predictor differs")
        if row["scoring_precision"] != record["precision"]:
            raise BuildError(f"{name}: scoring precision {row['scoring_precision']!r} != {record['precision']!r}")
        for item in class_rows(metrics["confusion"], metrics["aucs"]):
            check_same(name, f"sensitivity {item['grade']}", item["sensitivity"], row[f"sensitivity_{item['grade']}"])
            check_same(name, f"specificity {item['grade']}", item["specificity"], row[f"specificity_{item['grade']}"])
        if record["kind"] == "score":
            check_same(name, "median s/image", record["timing"]["likelihood_median_s"], row["median_seconds_per_image"])
            check_same(name, "p95 s/image", record["timing"]["likelihood_p95_s"], row["p95_seconds_per_image"])
            gen = record["generation"]
            check_same(name, "generation median", gen.get("generation_median_s"), row["generation_median_seconds"])
            check_same(name, "unparseable", gen.get("unparseable"), row["generation_unparseable"])
            check_same(name, "generation accuracy", gen.get("generation_accuracy_strict"), row["generation_accuracy_strict"])
            agreement = record["agreement"]
            check_same(name, "saved-prediction mismatches", agreement.get("saved_ref_grade_mismatches"),
                       row["saved_prediction_grade_mismatches"])
            check_same(name, "fallback diagnostic mismatches", agreement.get("diagnostic_ref_grade_mismatches"),
                       row.get("diagnostic_grade_mismatches"))
        else:
            model = record["model"]
            for ours, theirs in (("best_val_loss", "best_validation_loss"), ("epochs_run", "epochs_run"),
                                 ("lr", "lr"), ("alpha", "lora_alpha"), ("rank", "rank")):
                check_same(name, theirs, model[ours], row[theirs])


SUMMARY_MD_GROUP = re.compile(
    r"^\| (lr_study|alpha_study) \| (\d+) \| (\S+) \| ([\d.e-]+) \| ([\d.]+) \| [\d.]+ \| \d+ \| ([^|]+?) \| (\d+)[^|]* \| "
    r"([\d.]+)(?: (?:±|Â±) ([\d.]+))? \| ([\d.]*) \| ([\d.]+) \| (\d+) \|$", re.M)


def cross_check_queue_groups(final_root: Path, lr_summary: pd.DataFrame, alpha_summary: pd.DataFrame):
    """Group means in final_01/summary.md (rounded there) must match the recomputed group rows."""
    text = track(final_root / "summary.md").read_text(encoding="utf-8")
    matches = SUMMARY_MD_GROUP.findall(text)
    if not matches:
        raise BuildError("summary.md: no training-variant group rows found")
    groups = pd.concat([lr_summary.assign(stage="lr_study"), alpha_summary.assign(stage="alpha_study")])
    groups = groups[groups["source"] == FINAL_SOURCE]
    if len(matches) != len(groups):
        raise BuildError(f"summary.md has {len(matches)} group rows, recomputed {len(groups)}")
    for stage, _, _, lr, alpha, scoring, seeds, acc, sd, qwk_mean, auc_mean, constant in matches:
        wanted = GROUP_MIXED if "mixed" in scoring else (GROUP_FP16_ONLY if "protocol" in scoring else GROUP_FP16)
        row = groups[(groups["stage"] == stage) & np.isclose(groups["lr"], float(lr)) &
                     np.isclose(groups["alpha"], float(alpha)) & (groups["scoring_group"] == wanted)]
        if len(row) != 1:
            raise BuildError(f"summary.md group {stage} lr {lr} alpha {alpha} {scoring!r}: no single recomputed row")
        row = row.iloc[0]
        label = f"summary.md {stage} lr {lr} alpha {alpha} {wanted}"
        check(label, "seeds", row["n_runs"], int(seeds))
        check_same(label, "accuracy mean (pp, rounded)", 100 * row["accuracy_mean"], float(acc), tol=0.005 + 1e-9)
        if sd:
            check_same(label, "accuracy SD (pp, rounded)", 100 * row["accuracy_sd"], float(sd), tol=0.005 + 1e-9)
        if qwk_mean:
            check_same(label, "QWK mean (rounded)", row["qwk_mean"], float(qwk_mean), tol=5e-5 + 1e-12)
        check_same(label, "macro AUC mean (rounded)", row["macro_auc_mean"], float(auc_mean), tol=5e-5 + 1e-12)
        check(label, "constant predictors", row["n_constant_predictor"], int(constant))


def metric_columns(metrics: dict, boot: dict) -> dict:
    predicted = metrics["confusion"].sum(0)
    row = {"n": metrics["n"], "accuracy": metrics["accuracy"],
           "accuracy_ci_low": boot["accuracy_ci"][0], "accuracy_ci_high": boot["accuracy_ci"][1],
           "qwk": metrics["qwk"], "qwk_ci_low": boot["qwk_ci"][0], "qwk_ci_high": boot["qwk_ci"][1],
           "macro_auc": metrics["macro_auc"],
           "macro_auc_ci_low": boot["macro_auc_ci"][0], "macro_auc_ci_high": boot["macro_auc_ci"][1],
           "boot_qwk_undefined": boot["boot_qwk_undefined"], "boot_auc_undefined": boot["boot_auc_undefined"]}
    row.update({f"auc_g{g}": metrics["aucs"][g] for g in GRADES})
    row.update({f"n_pred_g{g}": int(predicted[g]) for g in GRADES})
    row.update({"n_grades_predicted": int((predicted > 0).sum()), "degenerate": bool((predicted > 0).sum() == 1),
                "grades_never_predicted": ";".join(str(g) for g in GRADES if predicted[g] == 0)})
    return row


MODEL_COLS = ["condition", "base", "offload_layers", "adapter", "source_run", "rank", "lr", "alpha", "lora_dropout",
              "patience", "effective_scaling", "recorded_effective_scaling", "scaling_note", "clahe_trained",
              "trainer_seed", "lora_seed"]
TRAIN_COLS = ["best_epoch", "best_step", "best_val_loss", "epochs_run"]
TIMING_COLS = ["likelihood_images_timed", "likelihood_mean_s", "likelihood_median_s", "likelihood_p95_s",
               "likelihood_images_per_second", "model_load_s", "torch_peak_alloc_gib"]
GENERATION_COLS = ["generation_images", "generation_images_timed", "generation_mean_s", "generation_median_s",
                   "generation_p95_s", "generation_images_per_second", "prompt_tokens", "unparseable",
                   "generation_accuracy_strict", "generation_accuracy_unparseable_as_0", "agreement_with_likelihood_grade"]
AGREEMENT_COLS = [f"{label}_ref_{item}" for label in ("saved", "diagnostic")
                  for item in ("images", "grade_mismatches", "largest_probability_difference")]
STRING_COLS = {"condition", "base", "adapter", "source_run", "scaling_note", "clahe_trained", "fallback_for"}


def final_row(record: dict, source: str = FINAL_SOURCE) -> dict:
    model = record["model"]
    row = {"result_set": "F" if source == FINAL_SOURCE else "A", "source": source, "stage": record["stage"],
           "job": record["job"], "kind": record["kind"], "cohort": record["cohort"]}
    row.update({col: model.get(col, "" if col in STRING_COLS else np.nan) for col in MODEL_COLS})
    row.update({"scoring_precision": record["precision"], "fallback_for": record["fallback_for"],
                "in_fp16_mean": record["in_fp16_mean"]})
    row.update(metric_columns(record["metrics"], record["boot"]))
    row.update({col: model.get(col, np.nan) for col in TRAIN_COLS})
    for cols, part in ((TIMING_COLS, "timing"), (GENERATION_COLS, "generation"), (AGREEMENT_COLS, "agreement")):
        row.update({col: record.get(part, {}).get(col, np.nan) for col in cols})
    return row


def study_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Group rows. Means use FP16-scored runs; a group with an FP32-fallback run gets a second, mixed row."""
    rows = []
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        excluded = int((~group["in_fp16_mean"]).sum())
        variants = [(GROUP_FP16, group)] if not excluded else [(GROUP_FP16_ONLY, group[group["in_fp16_mean"]]),
                                                                (GROUP_MIXED, group)]
        for label, members in variants:
            row = dict(zip(keys, key))
            row.update({"scoring_group": label, "n_runs": len(members),
                        "n_fallback_excluded": excluded if label == GROUP_FP16_ONLY else 0,
                        "n_fallback_included": int((~members["in_fp16_mean"]).sum()),
                        "n_constant_predictor": int(members["degenerate"].sum()),
                        "runs_missing_a_grade": int((members["n_grades_predicted"] < 5).sum()),
                        "n_images": ";".join(str(int(v)) for v in sorted(members["n"].unique()))})
            metrics = ["accuracy", "qwk", "macro_auc"] + (["best_val_loss"] if members["best_val_loss"].notna().any() else [])
            for metric in metrics:
                values = members[metric].dropna()
                row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
                row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
                row[f"{metric}_min"] = values.min() if len(values) else np.nan
                row[f"{metric}_max"] = values.max() if len(values) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def build_final(args, out: Path, evaluated, messidor_e: dict, written: dict):
    """Set F: tables from the finished final_01 queue, written to output/final_01 only."""
    folder = out / "final_01"
    records, failures = discover_final(args.final_root, args.runs_root)
    cross_check_queue_summary(args.final_root, records, failures)

    # Reference arm: the existing set A rank-16 CLAHE-on runs (lr 5e-4, alpha 16), same per-run values as Table 1.
    references = []
    for run, metrics in evaluated:
        if run["result_set"] == "A" and run["rank"] == 16 and run["clahe"] == "on":
            info = training_info(args.runs_root / run["run"])
            check_same(run["run"], "best validation loss", info["best_val_loss"], run["best_val_loss"])
            references.append({"job": run["run"], "kind": "train", "stage": "reference", "cohort": "aptos361_on",
                               "frame": run["frame"], "metrics": metrics, "boot": metrics["boot"],
                               "precision": run["scoring_precision"], "fallback_for": "", "in_fp16_mean": True,
                               "model": {**info, "adapter": "yes", "condition": condition_label(info), "offload_layers": 0}})
    if len(references) != 3:
        raise BuildError(f"Expected 3 set A rank-16 CLAHE-on reference runs, found {len(references)}")

    per_job = pd.DataFrame([final_row(r) for r in records])
    stage_order = {s: i for i, s in enumerate(["zero_shot", "benchmark", "original_adapter_audit", "messidor_archived",
                                               "lr_study", "alpha_study"])}
    per_job["_order"] = per_job["stage"].map(stage_order).fillna(len(stage_order))
    per_job = per_job.sort_values(["_order", "cohort", "job"], kind="stable").drop(columns="_order").reset_index(drop=True)
    write_csv(per_job, folder / "per_job_metrics.csv", written)
    order = {job: i for i, job in enumerate(per_job["job"])}
    ordered = sorted(records, key=lambda r: order[r["job"]])
    cm_rows, cw_rows = [], []
    for r in ordered:
        for g in GRADES:
            cm_rows.append({"stage": r["stage"], "job": r["job"], "cohort": r["cohort"], "condition": r["model"]["condition"],
                            "scoring_precision": r["precision"], "true_grade": g,
                            **{f"pred_{p}": int(r["metrics"]["confusion"][g, p]) for p in GRADES}})
        for item in class_rows(r["metrics"]["confusion"], r["metrics"]["aucs"]):
            cw_rows.append({"stage": r["stage"], "job": r["job"], "cohort": r["cohort"],
                            "condition": r["model"]["condition"], "scoring_precision": r["precision"], **item})
    write_csv(pd.DataFrame(cm_rows), folder / "confusion_matrices_per_job.csv", written)
    write_csv(pd.DataFrame(cw_rows), folder / "classwise_per_job.csv", written)

    zero = per_job[(per_job["stage"] == "zero_shot") & per_job["cohort"].str.startswith("aptos")].reset_index(drop=True)
    zero.insert(zero.columns.get_loc("cohort") + 1, "input_clahe", zero["cohort"].str.rsplit("_", n=1).str[1])
    write_csv(zero, folder / "zero_shot.csv", written)

    study_cols = ["source", "stage", "job", "cohort", "condition", "rank", "lr", "alpha", "effective_scaling",
                  "recorded_effective_scaling", "scaling_note", "trainer_seed", "lora_seed", "scoring_precision",
                  "fallback_for", "in_fp16_mean", "n", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "qwk",
                  "qwk_ci_low", "qwk_ci_high", "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high",
                  *[f"n_pred_g{g}" for g in GRADES], "n_grades_predicted", "degenerate", "grades_never_predicted",
                  "best_epoch", "best_step", "best_val_loss", "epochs_run"]
    reference_rows = pd.DataFrame([final_row(r, REFERENCE_SOURCE) for r in references])
    aptos = per_job[per_job["cohort"].str.startswith("aptos361")]
    studies = {}
    for stage, key in (("lr_study", "lr"), ("alpha_study", "alpha")):
        rows = pd.concat([reference_rows, aptos[aptos["stage"] == stage]], ignore_index=True)[study_cols]
        rows = rows.sort_values([key, "trainer_seed"], kind="stable").reset_index(drop=True)
        keys = ["source", "lr", "alpha", "effective_scaling"]
        summary = study_summary(rows, keys)
        studies[stage] = (rows, summary)
        write_csv(rows, folder / f"{stage}_per_run.csv", written)
        write_csv(summary, folder / f"{stage}_summary.csv", written)
    cross_check_queue_groups(args.final_root, studies["lr_study"][1], studies["alpha_study"][1])

    bench_cols = ["stage", "job", "cohort", "condition", "base", "offload_layers", "adapter", "scoring_precision", "n",
                  "accuracy", "qwk", "macro_auc", *TIMING_COLS, *GENERATION_COLS,
                  "saved_ref_images", "saved_ref_grade_mismatches", "saved_ref_largest_probability_difference"]
    bench = per_job[per_job["generation_images"].notna()][bench_cols].reset_index(drop=True)
    bench.insert(bench.columns.get_loc("generation_median_s") + 1, "generation_to_likelihood_median_ratio",
                 bench["generation_median_s"] / bench["likelihood_median_s"])
    write_csv(bench, folder / "inference_benchmark.csv", written)
    bench_groups = []
    for (base, adapter), group in bench.groupby(["base", "adapter"], sort=False):
        row = {"base": base, "adapter": adapter, "n_jobs": len(group), "jobs": ";".join(group["job"])}
        for col in ("likelihood_median_s", "generation_median_s", "generation_to_likelihood_median_ratio",
                    "agreement_with_likelihood_grade", "generation_accuracy_strict"):
            row.update({f"{col}_mean": group[col].mean(), f"{col}_min": group[col].min(), f"{col}_max": group[col].max()})
        row.update({"unparseable_total": int(group["unparseable"].sum()),
                    "generation_images_total": int(group["generation_images"].sum())})
        bench_groups.append(row)
    bench_groups = pd.DataFrame(bench_groups)
    write_csv(bench_groups, folder / "inference_benchmark_by_condition.csv", written)

    audit = per_job[per_job["stage"] == "original_adapter_audit"].reset_index(drop=True)
    if len(audit) != 1:
        raise BuildError(f"Expected one original adapter audit job, found {len(audit)}")
    write_csv(audit, folder / "original_adapter_audit.csv", written)

    # Archived Messidor: every final_01 model plus the set E replay (same 1,057 images and labels required).
    messidor = per_job[per_job["cohort"] == "messidor_archived"].copy()
    some = next(r for r in records if r["cohort"] == "messidor_archived")["frame"]
    e_frame = messidor_e["frame"]
    if sorted(zip(e_frame["image_id"], e_frame["y_true"])) != sorted(zip(some["image_id"], some["y_true"])):
        raise BuildError("Set E Messidor replay and final_01 Messidor scores cover different images or labels")
    e_record = {"job": "retraining2_replay/messidor_external", "kind": "score", "stage": "set E replay",
                "cohort": "messidor_archived", "frame": e_frame, "metrics": messidor_e["metrics"],
                "boot": messidor_e["boot"], "precision": messidor_e["precision"], "fallback_for": "",
                "in_fp16_mean": True, "model": {"condition": "Retraining 2 replay (set E, archived run_02)",
                                                "base": "4-bit", "adapter": "yes", "rank": 16, "clahe_trained": "on"}}
    messidor = pd.concat([messidor, pd.DataFrame([final_row(e_record, "set E replay")])], ignore_index=True)
    messidor_cols = ["source", "stage", "job", "condition", "base", "adapter", "rank", "lr", "alpha", "effective_scaling",
                     "scaling_note", "trainer_seed", "lora_seed", "scoring_precision", "fallback_for", "in_fp16_mean",
                     "n", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "qwk", "qwk_ci_low", "qwk_ci_high",
                     "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", *[f"n_pred_g{g}" for g in GRADES],
                     "n_grades_predicted", "degenerate", "grades_never_predicted", "likelihood_median_s"]
    messidor = messidor[messidor_cols]
    write_csv(messidor, folder / "messidor_all_models_per_run.csv", written)
    messidor_summary = study_summary(messidor.assign(best_val_loss=np.nan), ["condition"])
    write_csv(messidor_summary, folder / "messidor_all_models_summary.csv", written)

    write_final_summary(folder, zero, studies, bench, bench_groups, audit, messidor, messidor_summary, failures, written)


def write_final_summary(folder, zero, studies, bench, bench_groups, audit, messidor, messidor_summary, failures, written):
    def ci_pct(row):
        return f"{100 * row['accuracy']:.2f} [{100 * row['accuracy_ci_low']:.2f}–{100 * row['accuracy_ci_high']:.2f}]"

    def ci(row, metric):
        return f"{row[metric]:.4f} [{row[f'{metric}_ci_low']:.4f}–{row[f'{metric}_ci_high']:.4f}]"

    def run_table(frame, extra):
        table = frame[extra].copy()
        table["accuracy % [95% CI]"] = frame.apply(ci_pct, axis=1)
        table["QWK [95% CI]"] = frame.apply(lambda r: ci(r, "qwk"), axis=1)
        table["macro AUC [95% CI]"] = frame.apply(lambda r: ci(r, "macro_auc"), axis=1)
        table["predicted per grade"] = frame[[f"n_pred_g{g}" for g in GRADES]].astype(int).astype(str).agg(" ".join, axis=1)
        table["constant"] = frame["degenerate"]
        return table

    def group_table(summary, extra, val_loss=True):
        table = summary[extra + ["scoring_group", "n_runs", "n_constant_predictor", "runs_missing_a_grade"]].copy()
        table["accuracy % (mean ± SD)"] = mean_sd(summary, "accuracy", percent=True)
        table["QWK (mean ± SD)"] = mean_sd(summary, "qwk")
        table["macro AUC (mean ± SD)"] = mean_sd(summary, "macro_auc")
        if val_loss:
            table["best val loss (mean ± SD)"] = summary.apply(
                lambda r: f"{r.best_val_loss_mean:.5f}" + ("" if pd.isna(r.best_val_loss_sd) else f" ± {r.best_val_loss_sd:.5f}"),
                axis=1)
        return table

    num = {"lr": lambda v: f"{v:g}", "alpha": lambda v: f"{v:g}", "effective_scaling": lambda v: f"{v:g}",
           "recorded_effective_scaling": lambda v: f"{v:g}", "trainer_seed": lambda v: str(int(v)),
           "lora_seed": lambda v: str(int(v)), "rank": lambda v: str(int(v)), "n": lambda v: str(int(v)),
           "best_val_loss": lambda v: f"{v:.5f}", "epochs_run": lambda v: f"{v:g}", "best_epoch": lambda v: f"{v:g}"}
    lines = ["# Final queue final_01: summary", "",
             "Generated by `build_paper_tables.py` (set F) from the saved predictions, timing and training records in "
             "`D:\\Afterimage_Experiments\\final_runs\\final_01`. Every metric is recomputed from the prediction files "
             "and checked against the job's `summary.json` / `score_result.json` and against the queue's own "
             "`summary.csv` and `summary.md` (the build stops on any disagreement). Accuracy in percent; brackets are "
             "per-run 95% bootstrap CIs; ± is the sample SD across runs.", ""]

    lines += ["## Table F1. Zero-shot (no adapter), APTOS duplicate-excluded test, n = 361",
              "Likelihood (five-answer) scorer, FP16. The BF16 base is loaded as FP16 weights with 16 layers offloaded "
              "to CPU, as in the BF16 control's scoring. Generation: greedy, 24 new tokens, every image.", "",
              md_table(run_table(zero, ["condition", "input_clahe"]).assign(
                  **{"generation acc %": zero["generation_accuracy_strict"].map(pct),
                     "unparseable": zero["unparseable"].astype(int).astype(str) + " of " +
                     zero["generation_images"].astype(int).astype(str)})), "",
              "The zero-shot score on the archived Messidor export is in Table F6.", ""]

    for stage, title, key in (("lr_study", "F2. Learning-rate study", "lr"), ("alpha_study", "F3. LoRA alpha study", "alpha")):
        rows, summary = studies[stage]
        lines += [f"## Table {title} (rank 16, CLAHE on, 4-bit base, APTOS n = 361)",
                  "Three seed pairs (42/3407, 43/3408, 44/3409) per value. The lr 5e-4 / alpha 16 rows are the existing "
                  "set A rank-16 CLAHE-on runs (same per-run values as Table 1). `constant` = the run predicts a single "
                  "grade. Best val loss = validation loss of the saved (best) checkpoint.", "",
                  md_table(group_table(summary, ["source", "lr", "alpha", "effective_scaling"]), num), "",
                  md_table(run_table(rows, ["source", key, "effective_scaling", "trainer_seed", "lora_seed",
                                            "scoring_precision", "best_epoch", "epochs_run", "best_val_loss"]), num), ""]
    alpha_rows = studies["alpha_study"][0]
    noted = alpha_rows[alpha_rows["scaling_note"] != ""]
    lines += ["**Scaling record error.** `train_result.json` (and `score_result.json`) store `effective_scaling` as "
              "16/rank whatever the alpha, so they record 1.0 for every rank-16 run. The true LoRA scaling is alpha/r "
              "from each adapter's `adapter_config.json` (no rsLoRA): "
              + ", ".join(f"alpha {a:g} → {s:g}" for a, s in
                          sorted(set(zip(noted["alpha"], noted["effective_scaling"])))) +
              f". The `effective_scaling` column above uses adapter_config.json; {len(noted)} runs carry the note.", ""]
    for name, info in failures.items():
        lines += [f"**FP16 scoring failure.** `{name}` (alpha {info['alpha']:g}, seeds {info['trainer_seed']}/"
                  f"{info['lora_seed']}): the pinned FP16 score stopped with a non-finite likelihood (FP16 overflow). "
                  f"It is scored by the labelled FP32 fallback ({info['fallback_jobs']}), flagged "
                  f"`scoring_precision = {FALLBACK}`, left out of the FP16 group rows and included only in the rows "
                  f"labelled “{GROUP_MIXED}”.", ""]

    bench_md = bench[["stage", "job", "base", "adapter", "n", "likelihood_median_s", "likelihood_p95_s",
                      "generation_median_s", "generation_to_likelihood_median_ratio", "agreement_with_likelihood_grade",
                      "generation_accuracy_strict", "accuracy", "unparseable", "saved_ref_grade_mismatches"]].copy()
    bench_md["agreement_with_likelihood_grade"] = bench_md["agreement_with_likelihood_grade"].map(pct)
    bench_md["generation_accuracy_strict"] = bench_md["generation_accuracy_strict"].map(pct)
    bench_md["accuracy"] = bench_md["accuracy"].map(pct)
    sec = lambda v: f"{v:.3f}"  # noqa: E731
    lines += ["## Table F4. Inference benchmark: likelihood scorer vs text generation",
              "Seconds per image on one RTX 5070 Ti, excluding model load and the first three images. The likelihood "
              "scorer runs five forward passes per image; generation is greedy with up to 24 new tokens. "
              "`accuracy` = likelihood-scorer accuracy; generation accuracy counts unparseable outputs as wrong; "
              "agreement = generated grade equals the likelihood grade. `saved_ref_grade_mismatches` compares the "
              "re-scored grades with the run's saved predictions (benchmark jobs only).", "",
              md_table(bench_md, {"n": lambda v: str(int(v)), "likelihood_median_s": sec, "likelihood_p95_s": sec,
                                  "generation_median_s": sec, "generation_to_likelihood_median_ratio": lambda v: f"{v:.2f}",
                                  "unparseable": lambda v: str(int(v)), "saved_ref_grade_mismatches": lambda v: str(int(v))}), "",
              md_table(bench_groups[["base", "adapter", "n_jobs", "likelihood_median_s_mean", "generation_median_s_mean",
                                     "generation_to_likelihood_median_ratio_mean", "agreement_with_likelihood_grade_mean",
                                     "unparseable_total", "generation_images_total"]],
                       {"likelihood_median_s_mean": sec, "generation_median_s_mean": sec,
                        "generation_to_likelihood_median_ratio_mean": lambda v: f"{v:.2f}",
                        "agreement_with_likelihood_grade_mean": pct}), ""]

    a = audit.iloc[0]
    lines += ["## Table F5. Original paper adapter audit (original split test, n = 367)",
              "The original paper's saved adapter on the frozen 367-image original-split test list, 4-bit base, FP16 "
              "likelihood scoring, plus generation on every image.", "",
              md_table(run_table(audit, ["condition"]).assign(
                  **{"generation acc %": audit["generation_accuracy_strict"].map(pct),
                     "agreement with likelihood %": audit["agreement_with_likelihood_grade"].map(pct),
                     "unparseable": f"{int(a['unparseable'])} of {int(a['generation_images'])}"})), "",
              "The original split's test images overlap its training images (see the main SUMMARY.md caveats).", ""]

    lines += ["## Table F6. Archived Messidor export (n = 1,057), all models",
              "Archived 1,057-image Messidor export, not an independently verified cohort. FP16 likelihood scoring "
              f"except the row flagged `{FALLBACK}`. The set E replay row is the same frame as Table 8.", "",
              md_table(group_table(messidor_summary, ["condition"], val_loss=False)), "",
              md_table(run_table(messidor, ["condition", "trainer_seed", "scoring_precision"]), num), ""]

    lines += ["## Notes",
              "- The benchmark jobs re-score existing set A/C adapters; their likelihood grades match the saved "
              "set A/C predictions exactly (`saved_ref_grade_mismatches`).",
              "- Timing medians come from `timing.csv` and exclude the first three images, as in the scorer; "
              "they are not comparable with the upper-bound per-image times in Table 10.",
              "- Constant predictors and runs that never predict some grades are kept in every table and mean.",
              "- Full columns (per-grade AUCs, CIs, timing, agreement) are in the CSV files in this folder.", ""]
    path = folder / "SUMMARY.md"
    path.write_bytes("\n".join(lines).encode("utf-8"))
    written[str(path)] = len(lines)


# --------------------------------------------------------------------------- output

def write_csv(frame: pd.DataFrame, path: Path, written: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    path.write_bytes(buffer.getvalue().encode("utf-8"))
    written[str(path)] = len(frame)


def md_table(frame: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            fmt = formats.get(col)
            if isinstance(value, float) and math.isnan(value):
                cells.append("")
            elif fmt:
                cells.append(fmt(value))
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(value):
    return f"{100 * value:.2f}%"


def mean_sd(frame, metric, percent=False):
    def fmt(row):
        mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
        if percent:
            return f"{100 * mean:.2f}" + ("" if pd.isna(sd) else f" ± {100 * sd:.2f}")
        return f"{mean:.4f}" + ("" if pd.isna(sd) else f" ± {sd:.4f}")
    return frame.apply(fmt, axis=1)


def build(args) -> dict:
    out = args.output_dir
    written: dict[str, int] = {}

    pilot_runs = discover_pilot_runs(args.runs_root)
    original_runs = discover_original_split(args.original_split_root)
    replay_run, messidor, messidor_saved = discover_replay(args.original_split_root, args.reference_root)
    all_runs = pilot_runs + original_runs + [replay_run]
    for set_id in "ABCDE":
        if not any(r["result_set"] == set_id for r in all_runs):
            raise BuildError(f"No runs discovered for result set {set_id}")

    evaluated = [(run, evaluate_run(run)) for run in all_runs]
    per_run = pd.DataFrame([per_run_row(r, m) for r, m in evaluated])
    per_run = per_run.sort_values(["result_set", "rank", "clahe", "base", "trainer_seed", "run"], kind="stable").reset_index(drop=True)
    order = {r["run"]: i for i, r in enumerate(per_run.to_dict("records"))}
    evaluated.sort(key=lambda item: order[item[0]["run"]])

    groups = {
        "aptos_361_clean": [x for x in evaluated if x[0]["result_set"] in "ABC"],
        "aptos_367_original_split": [x for x in evaluated if x[0]["result_set"] == "D"],
        "retraining2_replay": [x for x in evaluated if x[0]["result_set"] == "E"],
    }
    summaries = {}
    for folder, items in groups.items():
        sets = {r["result_set"] for r, _ in items}
        subset = per_run[per_run["result_set"].isin(sets)].reset_index(drop=True)
        write_csv(subset, out / folder / "per_run_metrics.csv", written)
        summary = condition_summary(subset)
        summaries[folder] = summary
        write_csv(summary, out / folder / "condition_summary.csv", written)
        write_csv(confusion_long(items), out / folder / "confusion_matrices_per_run.csv", written)
        write_csv(pooled_confusion(items), out / folder / "confusion_matrices_pooled.csv", written)
        cw = classwise_frame(items)
        write_csv(cw, out / folder / "classwise_per_run.csv", written)
        write_csv(classwise_mean(cw), out / folder / "classwise_condition_mean.csv", written)
    write_csv(per_run, out / "all_sets_per_run_metrics.csv", written)

    # Table 3: rank sensitivity (set A)
    clean = summaries["aptos_361_clean"]
    eff = pd.DataFrame([{"run": r["run"], **r["efficiency"]} for r, _ in evaluated])
    params = per_run.merge(eff[["run", "trainable_params_logged"]], on="run")
    rank = clean[clean["result_set"] == "A"].copy()
    trainable = params[params["result_set"] == "A"].groupby("rank")["trainable_params_logged"].agg(lambda s: ";".join(sorted({str(int(v)) for v in s.dropna()})))
    rank["trainable_params_logged"] = rank["rank"].map(trainable)
    rank = rank[["rank", "alpha_over_r", "trainable_params_logged", "clahe", "n", "n_runs", "n_degenerate", "runs_missing_a_grade",
                 *[f"{m}_{s}" for m in ("accuracy", "qwk", "macro_auc") for s in ("mean", "sd", "min", "max")]]]
    rank = rank.sort_values(["rank", "clahe"], ascending=[True, False]).reset_index(drop=True)
    write_csv(rank, out / "aptos_361_clean" / "rank_sensitivity.csv", written)

    # Table 4: CLAHE ablation
    clahe_pairs = pd.concat([
        paired(per_run, {"result_set": "A", "rank": r, "clahe": "on"}, {"result_set": "A", "rank": r, "clahe": "off"},
               ["trainer_seed", "lora_seed"], f"A rank {r}: CLAHE off minus on")
        for r in sorted(per_run.loc[per_run["result_set"] == "A", "rank"].unique())
    ], ignore_index=True)
    write_csv(clahe_pairs, out / "aptos_361_clean" / "clahe_ablation_paired_by_seed.csv", written)
    original = per_run[per_run["result_set"] == "D"].copy()
    original["pair_index"] = original["run"].str.extract(r"_(\d\d)$")[0].astype(int)
    seeds = original[["trainer_seed", "lora_seed"]].drop_duplicates()
    original_pairs = paired(original, {"clahe": "on"}, {"clahe": "off"}, ["pair_index"],
                            "D original split: CLAHE off minus on (same-seed repeat)")
    original_pairs.insert(2, "trainer_seed", ";".join(map(str, sorted(seeds["trainer_seed"].unique()))))
    original_pairs.insert(3, "lora_seed", ";".join(map(str, sorted(seeds["lora_seed"].unique()))))
    original_pairs.insert(4, "pairing_note", "same trainer/LoRA seed in every pair; differences are not seed variation")
    write_csv(original_pairs, out / "aptos_367_original_split" / "clahe_pairs_same_seed_repeats.csv", written)

    # Table 5: 4-bit vs BF16
    bf16 = paired(per_run, {"result_set": "A", "rank": 16, "clahe": "on"}, {"result_set": "C", "rank": 16, "clahe": "on"},
                  ["trainer_seed", "lora_seed"], "BF16 base minus 4-bit base (rank 16, CLAHE on)")
    write_csv(bf16, out / "aptos_361_clean" / "bf16_vs_4bit_paired_by_seed.csv", written)

    # Table 8 and 9: Messidor archived cohort and misclassification lists
    m_metrics = point_metrics(messidor)
    for key, saved_key in (("accuracy", "accuracy"), ("qwk", "qwk"), ("macro_auc", "macro_ovr_auc")):
        check("E/messidor replay", key, m_metrics[key], messidor_saved["summary"][saved_key])
        check("E/messidor archive", key, m_metrics[key], messidor_saved["archived"][saved_key])
    if not np.array_equal(m_metrics["confusion"], np.asarray(messidor_saved["summary"]["confusion_matrix"])):
        raise BuildError("E/messidor: confusion matrix differs from the saved summary")
    m_boot = bootstrap(messidor)
    folder = out / "retraining2_replay"
    cohort_note = "archived 1,057-image Messidor export (not an independently verified cohort)"
    m_row = pd.DataFrame([{"cohort": cohort_note, "n": m_metrics["n"], "accuracy": m_metrics["accuracy"],
                           "accuracy_ci_low": m_boot["accuracy_ci"][0], "accuracy_ci_high": m_boot["accuracy_ci"][1],
                           "qwk": m_metrics["qwk"], "qwk_ci_low": m_boot["qwk_ci"][0], "qwk_ci_high": m_boot["qwk_ci"][1],
                           "macro_auc": m_metrics["macro_auc"], "macro_auc_ci_low": m_boot["macro_auc_ci"][0],
                           "macro_auc_ci_high": m_boot["macro_auc_ci"][1], "boot_qwk_undefined": m_boot["boot_qwk_undefined"],
                           "boot_auc_undefined": m_boot["boot_auc_undefined"], "scoring_precision": messidor_saved["summary"]["evaluation_precision"],
                           **{f"auc_g{g}": m_metrics["aucs"][g] for g in GRADES}}])
    write_csv(m_row, folder / "messidor_archived_metrics.csv", written)
    m_cm = pd.DataFrame([{"cohort": cohort_note, "true_grade": g, **{f"pred_{p}": int(m_metrics["confusion"][g, p]) for p in GRADES}}
                         for g in GRADES])
    write_csv(m_cm, folder / "messidor_archived_confusion_matrix.csv", written)
    m_cw = pd.DataFrame(class_rows(m_metrics["confusion"], m_metrics["aucs"]))
    m_cw.insert(0, "cohort", cohort_note)
    write_csv(m_cw, folder / "messidor_archived_classwise.csv", written)
    mild_total = int(m_metrics["confusion"][1].sum())
    mild_normal = int(m_metrics["confusion"][1, 0])
    mild = pd.DataFrame([{"cohort": cohort_note, "true_mild": mild_total, "mild_predicted_normal": mild_normal,
                          "rate": mild_normal / mild_total, "mild_predicted_mild": int(m_metrics["confusion"][1, 1])}])
    write_csv(mild, folder / "messidor_archived_mild_to_normal.csv", written)

    def misclassified(frame):
        wrong = frame[frame["y_true"] != frame["y_pred"]]
        return wrong[["image_id", "y_true", "y_pred", *PROB_COLS]].rename(
            columns={"y_true": "true_grade", "y_pred": "predicted_grade"}).sort_values("image_id", kind="stable").reset_index(drop=True)

    replay_frame = groups["retraining2_replay"][0][0]["frame"]
    write_csv(misclassified(replay_frame), folder / "misclassified_aptos_test.csv", written)
    m_wrong = misclassified(messidor)
    write_csv(m_wrong, folder / "misclassified_messidor_archived.csv", written)
    write_csv(m_wrong[(m_wrong["true_grade"] == 1) & (m_wrong["predicted_grade"] == 0)].reset_index(drop=True),
              folder / "messidor_archived_mild_predicted_normal.csv", written)

    # Table 10: efficiency
    eff_rows = []
    for run, metrics in evaluated:
        e = run["efficiency"]
        e_row = {"result_set": run["result_set"], "condition": condition_of(run), "run": run["run"], "base": run["base"],
                 "rank": run["rank"], "offload_layers": e["offload_layers"],
                 "trainable_params_logged": e["trainable_params_logged"], "total_params_logged": e["total_params_logged"],
                 "params_note": ("Unsloth train.log line; 4-bit and BF16 totals disagree (BF16 run used CPU layer offload); "
                                 "needs an independent count") if e["total_params_logged"] else "not logged",
                 "attempts": e["attempts"], "timed_attempt": e["final_attempt"],
                 "train_wall_s": e["train_wall_s"], "train_peak_device_vram_mib": e["train_peak_device_vram_mib"],
                 "train_torch_peak_alloc_bytes": e["train_torch_peak_alloc_bytes"],
                 "train_peak_system_ram_gib": e["train_peak_system_ram_gib"],
                 "score_wall_s": e["score_wall_s"], "score_peak_device_vram_mib": e["score_peak_device_vram_mib"],
                 "score_torch_peak_alloc_bytes": e["score_torch_peak_alloc_bytes"],
                 "score_peak_system_ram_gib": e["score_peak_system_ram_gib"], "n_scored": e["n_scored"],
                 "score_s_per_image_upper_bound": e["score_wall_s"] / e["n_scored"],
                 "timing_note": "final attempt only; scoring includes model load and five forward passes per image; "
                                "device VRAM is whole-GPU nvidia-smi"}
        if run["result_set"] == "E":
            e_row["timing_note"] += "; replay scored APTOS 367 + archived Messidor 1,057 in one stage; no training on this machine"
        eff_rows.append(e_row)
    efficiency = pd.DataFrame(eff_rows)
    for col in ("offload_layers", "trainable_params_logged", "total_params_logged", "train_wall_s", "train_peak_device_vram_mib",
                "train_torch_peak_alloc_bytes", "train_peak_system_ram_gib", "score_wall_s", "score_peak_device_vram_mib",
                "score_torch_peak_alloc_bytes", "score_peak_system_ram_gib"):
        efficiency[col] = pd.to_numeric(efficiency[col]).astype(float)
    write_csv(efficiency, out / "efficiency" / "qwen_efficiency_per_run.csv", written)
    eff_summary = efficiency.groupby(["result_set", "condition", "base", "rank"], sort=True).agg(
        n_runs=("run", "count"), trainable_params_logged=("trainable_params_logged", "first"),
        total_params_logged=("total_params_logged", "first"),
        train_wall_s_mean=("train_wall_s", "mean"), train_wall_s_min=("train_wall_s", "min"), train_wall_s_max=("train_wall_s", "max"),
        train_peak_device_vram_mib_max=("train_peak_device_vram_mib", "max"),
        train_torch_peak_alloc_gib_max=("train_torch_peak_alloc_bytes", lambda s: s.max() / 2**30),
        score_wall_s_mean=("score_wall_s", "mean"), score_peak_device_vram_mib_max=("score_peak_device_vram_mib", "max"),
        score_s_per_image_upper_bound_mean=("score_s_per_image_upper_bound", "mean")).reset_index()
    eff_summary = eff_summary.sort_values(["result_set", "rank", "condition"], ascending=[True, True, False],
                                          kind="stable").reset_index(drop=True)
    write_csv(eff_summary, out / "efficiency" / "qwen_efficiency_by_condition.csv", written)

    facts = overlap_facts(args.datasets_root)
    archived_runs = pd.read_csv(track(args.reference_root / "aptos_run_comparison.csv"))
    facts["archived_same_seed_accuracies"] = [float(v) for v in archived_runs.sort_values("run")["accuracy"]]
    write_csv(pd.DataFrame([{k: v for k, v in facts.items() if not isinstance(v, list)}]),
              out / "split_overlap_facts.csv", written)
    write_summary(out, summaries, rank, clahe_pairs, original_pairs, bf16, groups, m_row, m_cm, m_cw, mild, eff_summary,
                  facts, written)
    if args.final_root is not None:
        messidor_e = {"frame": messidor, "metrics": m_metrics, "boot": m_boot,
                      "precision": messidor_saved["summary"]["evaluation_precision"]}
        build_final(args, out, evaluated, messidor_e, written)
    return written


def write_summary(out, summaries, rank, clahe_pairs, original_pairs, bf16, groups, m_row, m_cm, m_cw, mild, eff_summary,
                  facts, written):
    def cond_table(summary):
        table = summary[["condition", "n", "n_runs", "n_degenerate", "runs_missing_a_grade"]].copy()
        table["accuracy % (mean ± SD)"] = mean_sd(summary, "accuracy", percent=True)
        table["acc min–max %"] = summary.apply(lambda r: f"{100*r.accuracy_min:.2f}–{100*r.accuracy_max:.2f}", axis=1)
        table["QWK (mean ± SD)"] = mean_sd(summary, "qwk")
        table["macro AUC (mean ± SD)"] = mean_sd(summary, "macro_auc")
        table["scoring"] = summary["scoring_precision"]
        return table

    def pair_table(frame, key_cols):
        table = frame[key_cols].copy()
        for metric, label in (("accuracy", "acc"), ("qwk", "QWK"), ("macro_auc", "AUC")):
            fmt = pct if metric == "accuracy" else (lambda v: f"{v:.4f}")
            table[f"{label} left"] = frame[f"{metric}_left"].map(fmt)
            table[f"{label} right"] = frame[f"{metric}_right"].map(fmt)
            diff = frame[f"{metric}_diff_right_minus_left"]
            table[f"{label} diff"] = diff.map(lambda v: f"{100*v:+.2f} pp") if metric == "accuracy" else diff.map(lambda v: f"{v:+.4f}")
        table["degenerate (left/right)"] = frame["left_degenerate"].astype(str) + "/" + frame["right_degenerate"].astype(str)
        return table

    def cm_table(frame, condition):
        sub = frame[frame["condition"] == condition]
        return md_table(sub[["true_grade", *[f"pred_{p}" for p in GRADES]]].reset_index(drop=True))

    lines = ["# Paper tables: summary", "",
             "Generated by `build_paper_tables.py` from saved prediction exports. Facts and numbers only. "
             "Accuracy in percent; SD is the sample SD across runs; ± never denotes a confidence interval here. "
             "Per-run 95% bootstrap CIs are in each folder's `per_run_metrics.csv`.", ""]

    clean = summaries["aptos_361_clean"]
    lines += ["## Table 2a. Condition summary, result sets A, B, C",
              "Sets A (duplicate-excluded, 4-bit), B (previous train/val split, 4-bit), C (BF16 base); APTOS test n = 361; FP16 scoring.", "",
              md_table(cond_table(clean)), ""]
    lines += ["## Table 2b. Condition summary, result set D",
              "Set D, original split 2929/366/367; APTOS test n = 367; BF16 scoring of 4-bit-trained adapters; three same-seed repeats per condition.", "",
              md_table(cond_table(summaries["aptos_367_original_split"])), ""]
    lines += ["## Table 2c. Retraining 2 replay, result set E",
              "Set E, original split; APTOS test n = 367; FP16 scoring; single saved adapter (archived run_02 of three same-seed runs).", "",
              md_table(cond_table(summaries["retraining2_replay"])), ""]

    rank_md = rank[["rank", "alpha_over_r", "trainable_params_logged", "clahe", "n_runs", "n_degenerate"]].copy()
    rank_md["accuracy % (mean ± SD)"] = mean_sd(rank, "accuracy", percent=True)
    rank_md["QWK (mean ± SD)"] = mean_sd(rank, "qwk")
    rank_md["macro AUC (mean ± SD)"] = mean_sd(rank, "macro_auc")
    lines += ["## Table 3. Rank sensitivity (alpha fixed at 16)",
              "Set A; APTOS test n = 361; duplicate-excluded split; 4-bit base; FP16 scoring; three seed pairs per cell.", "",
              md_table(rank_md, {"alpha_over_r": lambda v: f"{v:g}"}), ""]

    lines += ["## Table 4a. CLAHE ablation paired by seed (left = CLAHE on, right = CLAHE off)",
              "Set A; APTOS test n = 361; duplicate-excluded split; 4-bit base; FP16 scoring.", "",
              md_table(pair_table(clahe_pairs, ["comparison", "trainer_seed", "lora_seed"])), ""]
    lines += ["## Table 4b. Original-split CLAHE pairs (left = on, right = off), same-seed repeats",
              "Set D; APTOS test n = 367; original split; 4-bit-trained, BF16 scoring; every pair uses trainer seed 42 / LoRA seed 3407.", "",
              md_table(pair_table(original_pairs, ["pair_index", "trainer_seed", "lora_seed"])), ""]
    lines += ["## Table 5. 4-bit base (left) vs BF16 base (right), rank 16, CLAHE on",
              "Sets A and C; APTOS test n = 361; duplicate-excluded split; FP16 scoring for both; BF16 training offloaded 16 layers to CPU.", "",
              md_table(pair_table(bf16, ["trainer_seed", "lora_seed"])), ""]

    lines += ["## Table 6. Pooled confusion matrices (rows = true grade, columns = predicted grade; summed over runs)", ""]
    pooled_clean = pd.read_csv(out / "aptos_361_clean" / "confusion_matrices_pooled.csv")
    pooled_d = pd.read_csv(out / "aptos_367_original_split" / "confusion_matrices_pooled.csv")
    pooled_e = pd.read_csv(out / "retraining2_replay" / "confusion_matrices_pooled.csv")
    for frame, caption in ((pooled_clean, "n = 361 per run, FP16 scoring"), (pooled_d, "n = 367 per run, BF16 scoring"),
                           (pooled_e, "n = 367, FP16 scoring")):
        for condition in sorted(frame["condition"].unique()):
            if not any(tag in condition for tag in ("r16", "E ")):
                continue
            n_runs = int(frame.loc[frame["condition"] == condition, "n_runs"].iloc[0])
            lines += [f"**{condition}** ({n_runs} run(s), {caption})", "", cm_table(frame, condition), ""]
    lines += ["Rank 8 and rank 32 pooled matrices are in `aptos_361_clean/confusion_matrices_pooled.csv`.", ""]

    lines += ["## Table 8. Messidor, archived 1,057-image cohort (Retraining 2 replay only)",
              "Set E; archived Messidor export, not an independently verified cohort; n = 1,057; FP16 scoring.", "",
              md_table(m_row[["n", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "qwk", "qwk_ci_low", "qwk_ci_high",
                              "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high"]], {"n": lambda v: str(int(v))}), "",
              md_table(m_cm[["true_grade", *[f"pred_{p}" for p in GRADES]]]), "",
              md_table(m_cw[["grade", "label", "support", "sensitivity", "specificity", "precision", "f1", "auc_ovr"]]), "",
              f"Mild predicted as Normal: {int(mild.mild_predicted_normal[0])} of {int(mild.true_mild[0])} true Mild images "
              f"({100 * mild.rate[0]:.2f}%).", ""]

    eff_md = eff_summary.copy()
    for col in ("train_wall_s_mean", "train_wall_s_min", "train_wall_s_max", "score_wall_s_mean"):
        eff_md[col] = eff_md[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    eff_md["train_torch_peak_alloc_gib_max"] = eff_md["train_torch_peak_alloc_gib_max"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    eff_md["score_s_per_image_upper_bound_mean"] = eff_md["score_s_per_image_upper_bound_mean"].map(lambda v: f"{v:.3f}")
    for col in ("trainable_params_logged", "total_params_logged"):
        eff_md[col] = eff_md[col].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    for col in ("train_peak_device_vram_mib_max", "score_peak_device_vram_mib_max"):
        eff_md[col] = eff_md[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    lines += ["## Table 10. Qwen efficiency by condition",
              "Sets A–E; RTX 5070 Ti 16 GB; times are the final attempt only; per-image time is scoring wall time / n "
              "(upper bound: includes model load and five forward passes per image); VRAM is whole-device nvidia-smi peak; "
              "total parameters exactly as logged by Unsloth and inconsistent between 4-bit and BF16. "
              "RetiZero is not measured: no matched RetiZero run exists.", "",
              md_table(eff_md, {"rank": str}), ""]

    lines += ["## Caveats",
              "- Set D pairs all use trainer seed 42 / LoRA seed 3407: differences between pairs are run-to-run non-determinism, not seed variation.",
              f"- Original split (sets D, E): {facts['original_test_identical_to_train']} of {facts['original_test_n']} test images have "
              f"a byte-identical original file in training and {facts['original_test_identical_to_val']} in validation. "
              f"Previous-split control (set B): {facts['clean_test_identical_to_previous_train']} of its {facts['clean_test_n']} test "
              f"images have a byte-identical file in its training split and {facts['clean_test_identical_to_previous_val']} in validation "
              "(from the saved pairing audits; see `split_overlap_facts.csv`).",
              "- Alpha is fixed at 16 while rank changes, so alpha/r is 2, 1 and 0.5 for ranks 8, 16 and 32.",
              "- Scoring precision differs between sets: A, B, C and E use FP16; D uses BF16.",
              "- Constant predictors (degenerate = True) and runs that never predict some grades are kept in every table and mean.",
              "- Retraining 2 (set E) is one of three archived same-seed runs (APTOS accuracy "
              + ", ".join(f"{100 * v:.2f}%" for v in facts["archived_same_seed_accuracies"]) + ", from aptos_run_comparison.csv).",
              "- The Messidor figures use the archived 1,057-image export, not an independently verified cohort.", ""]
    path = out / "SUMMARY.md"
    path.write_bytes("\n".join(lines).encode("utf-8"))
    written[str(path)] = len(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--original-split-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL,
                        help="finished final_01 queue folder (set F, written to output/final_01)")
    parser.add_argument("--skip-final", dest="final_root", action="store_const", const=None,
                        help="build sets A-E only")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if not str(args.output_dir).lower().startswith(str(HERE).lower()):
        raise BuildError(f"Output must stay inside {HERE}")
    try:
        written = build(args)
    except BuildError as error:
        print(f"BUILD FAILED: {error}", file=sys.stderr)
        return 1
    try:
        import openpyxl  # noqa: F401
        workbook = "openpyxl available but workbook export not implemented"
    except ImportError:
        workbook = "skipped: openpyxl is not installed in this environment"
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "script_sha256": sha256(Path(__file__)),
        "test_sha256": sha256(HERE / "test_metrics.py") if (HERE / "test_metrics.py").exists() else None,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED, "method": "percentile 2.5/97.5"},
        "cross_check_tolerance": TOL,
        "workbook": workbook,
        "inputs": dict(sorted(INPUTS.items())),
        "outputs": {str(Path(p).relative_to(args.output_dir)).replace("\\", "/"): {"rows": n, "sha256": sha256(Path(p))}
                    for p, n in sorted(written.items())},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Cross-checks passed for all runs. Inputs read: {len(INPUTS)}. Files written: {len(written) + 1}.")
    print(f"Workbook: {workbook}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
