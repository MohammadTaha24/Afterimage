"""Revision analyses for the reviewer points, computed only from saved per-image predictions.

CPU only. Read-only on every input. Writes only under finetuning/paper_tables/output/revision_analysis.
Metric functions are imported from build_paper_tables.py (the pinned implementations), so every
number here is computed the same way as the existing paper tables.

What it adds (Dr. Noor's plan items and reviewer points):
  reproduction   recomputes accuracy, QWK and macro AUC for every run and checks them against the saved
                 summaries and the committed tables (1e-8); AUC re-derived independently with scipy
  per run        balanced accuracy, macro-F1, referable DR (grade >= 2), adjacent vs distant errors,
                 Mild -> Normal, calibration (ECE, Brier, NLL), bootstrap CIs
  studies        LoRA rank (item 1), learning rate chosen by validation loss (G2/S8), CLAHE (item 2),
                 adapter vs zero-shot (S7), Qwen vs RetiZero head-to-head (S12)
                 each with per-seed exact McNemar (Holm-adjusted) and paired / hierarchical bootstrap CIs
  class-wise     sensitivity, specificity, precision, F1, AUC per grade for every run (item 4)
  Messidor       adjudicated-grade audit, patient-clustered bootstrap CIs, Mild -> Normal analysis (item 4)
  S9 / S11       class counts per split and per epoch after balancing; misclassified-image lists
  efficiency     final Qwen vs RetiZero table from saved records, RetiZero training now included (item 3)

Usage (from the repo root):
  python -I -B finetuning/paper_tables/build_revision_analysis.py --experiments-root D:/Afterimage_Experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_paper_tables as bpt  # noqa: E402

DEFAULT_EXPERIMENTS = Path(r"D:\Afterimage_Experiments")
DEFAULT_OUT = HERE / "output" / "revision_analysis"
COMMITTED = HERE / "output"
GRADES = bpt.GRADES
PROB = bpt.PROB_COLS
BOOT_N = 2000
BOOT_SEED = 20261122
ECE_BINS = 15
TOL = 1e-8

# Headline condition names used by the studies below.
Q_ON_5E4 = "Qwen r16 CLAHE-on lr5e-4"
Q_OFF_5E4 = "Qwen r16 CLAHE-off lr5e-4"
Q_ON_2E4 = "Qwen r16 CLAHE-on lr2e-4"
Q_ON_1E4 = "Qwen r16 CLAHE-on lr1e-4"
R_OFF = "RetiZero CLAHE-off"
R_ON = "RetiZero CLAHE-on"


class AnalysisError(RuntimeError):
    pass


# --------------------------------------------------------------------------- discovery

PILOT_RE = re.compile(r"^full_(bf16_)?r(\d+)_(on|off)_(duplicate_excluded|previous_split)_t(\d+)_a(\d+)_(paper_controls_)?01$")
# Training variants named by final_runs.variant_name, e.g. r16_on_lr0p0002_t42_a3407 or r8_on_alpha8_lr0p0002_t42_a3407.
VARIANT_RE = re.compile(r"^r(\d+)_(on|off)_(?:alpha(\d+)_)?(?:lr(0p\d+)_)?t(\d+)_a(\d+)$")
QWEN_CAMPAIGNS = ("final_01", "final_02")      # final_02: verified Messidor-2, timing and the optional rank sweep
RETIZERO_CAMPAIGNS = ("retizero_01", "retizero_02")
MESSIDOR_COHORTS = {"messidor_archived": "messidor", "messidor_archived_fp32": "messidor",
                    "messidor2_on": "messidor2", "messidor2_off": "messidor2",
                    "messidor2_224_off": "messidor2", "messidor2_224_on": "messidor2"}


def lr_label(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def variant(name: str) -> dict | None:
    m = VARIANT_RE.match(name)
    if not m:
        return None
    rank, clahe, alpha, lr, seed = int(m.group(1)), m.group(2), int(m.group(3) or 16), m.group(4), int(m.group(5))
    lr = float(lr.replace("p", ".")) if lr else 5e-4
    cond = f"Qwen r{rank} CLAHE-{clahe} lr{lr_label(lr)}" + (f" alpha{alpha}" if alpha != 16 else "")
    return {"condition": cond, "seed": seed, "rank": rank, "clahe": clahe, "lr": lr, "alpha": alpha}


def run(model, condition, seed, cohort, path, source, **extra):
    return {"model": model, "condition": condition, "seed": seed, "cohort": cohort,
            "path": Path(path), "source": source, **extra}


def discover(root: Path) -> list[dict]:
    runs = []
    for d in sorted((root / "runs").iterdir()):
        m = PILOT_RE.match(d.name)
        if not m:
            continue
        bf16, rank, clahe, split, seed = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
        cond = f"Qwen r{rank} CLAHE-{clahe} lr5e-4"
        if bf16:
            cond = "Qwen BF16 r16 CLAHE-on lr5e-4"
        elif split == "previous_split":
            cond = "Qwen r16 CLAHE-on lr5e-4 previous-split control"
        val = bpt.read_json(d / "train_result.json")["validation_loss"]
        runs.append(run("Qwen", cond, seed, "aptos361", d / "aptos_test.predictions.csv", "runs", name=d.name,
                        val_loss=float(val), rank=rank, clahe=clahe, lr=5e-4, alpha=16))
    for campaign in QWEN_CAMPAIGNS:
        final = root / "final_runs" / campaign
        if not final.is_dir():
            continue
        for d in sorted((final / "train").iterdir()) if (final / "train").is_dir() else []:
            v = variant(d.name)
            if v is None or not (d / "train_result.json").is_file():
                continue  # not a variant, or training has not finished
            preds = d / "aptos_test.predictions.csv"
            job = d.name
            if not preds.is_file():  # the FP16-overflow run was scored with the labelled FP32 fallback
                job = f"{d.name}__aptos_test_fp32"
                preds = final / "scores" / job / "predictions.csv"
                if not preds.is_file():
                    continue
            val = bpt.read_json(d / "train_result.json")["validation_loss"]
            runs.append(run("Qwen", v.pop("condition"), v.pop("seed"), "aptos361", preds, campaign, name=job,
                            val_loss=float(val), **v))
        for d in sorted((final / "scores").iterdir()) if (final / "scores").is_dir() else []:
            name = d.name
            if not (d / "predictions.csv").is_file() or "__" not in name:
                continue
            m = re.match(r"^zeroshot_(4bit|bf16)__aptos361_(on|off)$", name)
            if m:
                runs.append(run("Qwen", f"Qwen zero-shot {m.group(1)} CLAHE-{m.group(2)}", None, "aptos361",
                                d / "predictions.csv", campaign, name=name, clahe=m.group(2)))
                continue
            if name.split("__")[1] in MESSIDOR_COHORTS:
                runs.append(qwen_messidor(name, d, campaign))
    for campaign in RETIZERO_CAMPAIGNS:
        rz_runs(root / "retizero_runs" / campaign, campaign, runs)
    return runs


def rz_runs(rz: Path, campaign: str, runs: list[dict]) -> None:
    if not rz.is_dir():
        return
    for d in sorted((rz / "train").iterdir()) if (rz / "train").is_dir() else []:
        stage, seed = d.name.rsplit("_t", 1)
        cond, cohort = {"matched": (R_OFF, "aptos361"), "clahe": (R_ON, "aptos361"),
                        "historical": ("RetiZero CLAHE-off original split", "aptos367")}[stage]
        result = bpt.read_json(d / "train_result.json")
        runs.append(run("RetiZero", cond, int(seed), cohort, d / "evaluation" / "aptos_test.predictions.csv",
                        campaign, name=d.name, val_acc=float(result["best_validation_accuracy"]),
                        clahe="on" if stage == "clahe" else "off",
                        summary=d / "evaluation" / "aptos_test.summary.json"))
    for d in sorted((rz / "scores").iterdir()) if (rz / "scores").is_dir() else []:
        name = d.name
        if not (d / "predictions.csv").is_file():
            continue
        model_part, cohort_part = name.split("__")
        if model_part.startswith("matched") and cohort_part == "aptos361_clean":
            continue  # benchmark rescoring; identical to the training run's own predictions (checked below)
        seed = int(model_part[-2:]) if model_part[:-2].endswith("_t") else None
        cond = {"matched": R_OFF, "clahe": R_ON}.get(model_part.rsplit("_t", 1)[0], "RetiZero saved app checkpoint")
        cohort = {"aptos361_clean": "aptos361", "aptos367_original": "aptos367", **MESSIDOR_COHORTS}[cohort_part]
        runs.append(run("RetiZero", cond, seed, cohort, d / "predictions.csv", campaign, name=name,
                        summary=d / "summary.json"))


def qwen_messidor(name: str, folder: Path, campaign: str) -> dict:
    stem, cohort_part = name.split("__")
    seed, cond = None, None
    m = re.match(r"^(bf16_)?r16_(on|off)_p(\d)$", stem)
    if m:
        seed = 41 + int(m.group(3))
        cond = "Qwen BF16 r16 CLAHE-on lr5e-4" if m.group(1) else f"Qwen r16 CLAHE-{m.group(2)} lr5e-4"
    v = variant(stem)
    if v:
        cond, seed = v["condition"], v["seed"]
    if stem == "zeroshot_4bit":
        cond = f"Qwen zero-shot 4bit CLAHE-{'off' if cohort_part.endswith('_off') else 'on'}"
    if cond is None:
        raise AnalysisError(f"Unrecognised Messidor job {name}")
    extra = {"summary": folder / "summary.json"} if (folder / "summary.json").is_file() else {}
    return run("Qwen", cond, seed, MESSIDOR_COHORTS[cohort_part], folder / "predictions.csv", campaign, name=name, **extra)


# --------------------------------------------------------------------------- metrics

def ece(probs: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def reliability_rows(probs: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> list[dict]:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    return [{"bin_low": edges[b], "bin_high": edges[b + 1], "n": int((which == b).sum()),
             "mean_confidence": float(conf[which == b].mean()) if (which == b).any() else np.nan,
             "accuracy": float(correct[which == b].mean()) if (which == b).any() else np.nan} for b in range(bins)]


def full_metrics(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(int)
    p = frame["y_pred"].to_numpy(int)
    probs = frame[PROB].to_numpy(float)
    base = bpt.point_metrics(frame)
    cm = base["confusion"]
    rows = bpt.class_rows(cm, base["aucs"])
    recalls = [r["sensitivity"] for r in rows if r["support"] > 0]
    f1s = [0.0 if np.isnan(r["f1"]) else r["f1"] for r in rows if r["support"] > 0]
    ref_true = y >= 2
    ref_score = probs[:, 2:].sum(1)
    ref_pred = p >= 2
    d = np.abs(y - p)
    onehot = np.eye(5)[y]
    p_true = probs[np.arange(len(y)), y]
    errors = int((d > 0).sum())
    return {
        "n": len(y), "accuracy": base["accuracy"], "qwk": base["qwk"], "macro_auc": base["macro_auc"],
        **{f"auc_g{g}": base["aucs"][g] for g in GRADES},
        "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)),
        "referable_auc": float(bpt.weighted_auc(ref_true, ref_score, np.ones(len(y)))[0]),
        "referable_sensitivity": float(ref_pred[ref_true].mean()),
        "referable_specificity": float((~ref_pred[~ref_true]).mean()),
        "exact_rate": float((d == 0).mean()), "adjacent_error_rate": float((d == 1).mean()),
        "distant_error_rate": float((d >= 2).mean()),
        "adjacent_share_of_errors": float((d == 1).sum() / errors) if errors else np.nan,
        "max_grade_distance": int(d.max()),
        "mild_n": int((y == 1).sum()), "mild_predicted_normal": int(((y == 1) & (p == 0)).sum()),
        "mild_recall": float(((y == 1) & (p == 1)).sum() / max((y == 1).sum(), 1)),
        "ece": ece(probs, y), "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "nll": float(-np.log(np.clip(p_true, 1e-12, None)).mean()),
        "mean_confidence": float(probs.max(1).mean()),
        **{f"n_pred_g{g}": int((p == g).sum()) for g in GRADES},
        "n_grades_predicted": int(len(set(p.tolist()))), "collapsed": bool(len(set(p.tolist())) <= 2),
        "grades_never_predicted": " ".join(str(g) for g in GRADES if g not in set(p.tolist())),
        "_rows": rows, "_cm": cm,
    }


# --------------------------------------------------------------------------- resampling

def resample_counts(n: int, groups: np.ndarray | None, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) multiplicity of each image. With groups, whole groups (patients) are resampled."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        counts = np.zeros((n_boot, n))
        np.add.at(counts, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1)
        return counts
    codes, uniq = pd.factorize(groups)
    g = len(uniq)
    pick = rng.integers(0, g, size=(n_boot, g))
    group_counts = np.zeros((n_boot, g))
    np.add.at(group_counts, (np.repeat(np.arange(n_boot), g), pick.ravel()), 1)
    return group_counts[:, codes]


def metrics_from_counts(frame: pd.DataFrame, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Accuracy, QWK and macro AUC for each bootstrap row of image multiplicities."""
    y = frame["y_true"].to_numpy(int)
    p = frame["y_pred"].to_numpy(int)
    probs = frame[PROB].to_numpy(float)
    b = counts.shape[0]
    acc = counts @ (y == p).astype(float) / counts.sum(1)
    cm = np.zeros((b, 25))
    for cell in np.unique(y * 5 + p):
        cm[:, cell] = counts[:, (y * 5 + p) == cell].sum(1)
    kappa = bpt.qwk_from_confusion(cm.reshape(b, 5, 5))
    aucs = np.stack([bpt.weighted_auc(y == g, probs[:, g], counts) for g in GRADES], axis=1)
    macro = np.where(np.isnan(aucs).any(1), np.nan, aucs.mean(1))
    return {"accuracy": acc, "qwk": kappa, "macro_auc": macro}


def ci(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def boot_p(diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a difference of zero (percentile method)."""
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return np.nan
    return float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    if a_only + b_only == 0:
        return a_only, b_only, 1.0
    return a_only, b_only, float(stats.binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue)


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


# --------------------------------------------------------------------------- loading and checks

def load_all(runs: list[dict]) -> None:
    for r in runs:
        frame = bpt.read_predictions(r["path"])
        frame["image_id"] = frame["image_id"].map(normalise_id)
        r["frame"] = frame
        r["metrics"] = full_metrics(frame)


def normalise_id(image_id: str) -> str:
    """Qwen and RetiZero name the same image differently in places; reduce to the source ID."""
    m = re.search(r"(\d{8}_\d+_\d{4}_PP)", image_id)
    if m:
        return m.group(1)
    m = re.match(r"^aptos_aptos_([0-9a-f]{12})_", image_id)
    return m.group(1) if m else image_id


def committed_lookup() -> dict[str, dict]:
    table = {}
    for path in (COMMITTED / "aptos_361_clean" / "per_run_metrics.csv", COMMITTED / "final_01" / "per_job_metrics.csv"):
        for row in csv.DictReader(open(bpt.track(path), encoding="utf-8")):
            key = row.get("run") or row.get("job")
            table[key] = row
    return table


def find_metric(obj, key):
    if isinstance(obj, dict):
        if key in obj and not isinstance(obj[key], (dict, list)):
            return obj[key]
        for v in obj.values():
            hit = find_metric(v, key)
            if hit is not None:
                return hit
    return None


def reproduction_table(runs: list[dict], experiments: Path) -> pd.DataFrame:
    committed = committed_lookup()
    rows = []
    for r in runs:
        m = r["metrics"]
        ref, source = None, None
        if r["name"] in committed:
            c = committed[r["name"]]
            ref = {"accuracy": float(c["accuracy"]), "qwk": float(c["qwk"]), "macro_auc": float(c["macro_auc"])}
            source = "committed paper table"
        elif r.get("summary"):
            s = bpt.read_json(r["summary"])
            s = s.get("metrics", s)  # score jobs nest the metrics; training evaluations keep them at the top
            ref = {"accuracy": find_metric(s, "accuracy"), "qwk": find_metric(s, "qwk"), "macro_auc": find_metric(s, "macro_ovr_auc")}
            source = "saved run summary"
        y = r["frame"]["y_true"].to_numpy(int)
        probs = r["frame"][PROB].to_numpy(float)
        scipy_auc = []
        for g in GRADES:
            pos, neg = probs[y == g, g], probs[y != g, g]
            scipy_auc.append(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)) if len(pos) and len(neg) else np.nan)
        row = {"model": r["model"], "condition": r["condition"], "seed": r["seed"], "cohort": r["cohort"], "job": r["name"],
               "n": m["n"], "reference": source, "auc_max_abs_diff_vs_scipy": float(np.nanmax(np.abs(np.array(scipy_auc) - [m[f"auc_g{g}"] for g in GRADES])))}
        ok = row["auc_max_abs_diff_vs_scipy"] < 1e-12
        for k in ("accuracy", "qwk", "macro_auc"):
            row[f"{k}_recomputed"] = m[k]
            row[f"{k}_reference"] = ref[k] if ref else np.nan
            if ref and ref[k] is not None:
                ok &= abs(m[k] - float(ref[k])) <= TOL
        row["match"] = bool(ok) if ref else None
        rows.append(row)
    frame = pd.DataFrame(rows)
    bad = frame[frame["match"] == False]  # noqa: E712
    if len(bad):
        raise AnalysisError(f"Recomputed metrics differ from saved values:\n{bad.to_string()}")
    # The RetiZero benchmark rescorings must equal the training runs' own predictions.
    rz = experiments / "retizero_runs" / "retizero_01"
    for seed in (42, 43, 44):
        a = pd.read_csv(bpt.track(rz / "train" / f"matched_t{seed}" / "evaluation" / "aptos_test.predictions.csv"))
        b = pd.read_csv(bpt.track(rz / "scores" / f"matched_t{seed}__aptos361_clean" / "predictions.csv"))
        if not (a["y_pred"].tolist() == b["y_pred"].tolist() and np.allclose(a[PROB], b[PROB], atol=1e-4)):
            raise AnalysisError(f"RetiZero matched_t{seed}: benchmark rescoring differs from training predictions")
    return frame


def check_alignment(runs: list[dict]) -> pd.DataFrame:
    """Every run on one cohort must score the same images, in the same order, with the same labels."""
    rows = []
    for cohort in sorted({r["cohort"] for r in runs}):
        group = [r for r in runs if r["cohort"] == cohort]
        ref = group[0]["frame"]
        for r in group:
            f = r["frame"]
            same_set = set(f["image_id"]) == set(ref["image_id"])
            same_order = f["image_id"].tolist() == ref["image_id"].tolist()
            aligned = f.set_index("image_id").loc[ref["image_id"]]
            same_labels = same_set and np.array_equal(aligned["y_true"].to_numpy(int), ref["y_true"].to_numpy(int))
            if not (same_set and same_labels):
                raise AnalysisError(f"{r['name']}: images or labels differ from {group[0]['name']} on {cohort}")
            rows.append({"cohort": cohort, "job": r["name"], "n": len(f), "same_ids": same_set,
                         "same_order": same_order, "same_labels": same_labels})
    return pd.DataFrame(rows)


def aligned(frame: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return frame.set_index("image_id").loc[ids].reset_index()


# --------------------------------------------------------------------------- tables

SCALAR = ["n", "accuracy", "qwk", "macro_auc", "balanced_accuracy", "macro_f1", "referable_auc",
          "referable_sensitivity", "referable_specificity", "exact_rate", "adjacent_error_rate",
          "distant_error_rate", "adjacent_share_of_errors", "max_grade_distance", "mild_n",
          "mild_predicted_normal", "mild_recall", "ece", "brier", "nll", "mean_confidence",
          *[f"auc_g{g}" for g in GRADES], *[f"n_pred_g{g}" for g in GRADES], "n_grades_predicted", "grades_never_predicted", "collapsed"]


def per_run_table(runs: list[dict], groups: dict[str, dict]) -> pd.DataFrame:
    """groups: cohort -> {image_id: cluster}; those cohorts get cluster (patient or exam) bootstrap CIs."""
    rows = []
    for r in runs:
        m = r["metrics"]
        f = r["frame"]
        g = None
        if r["cohort"] in groups:
            g = f["image_id"].map(groups[r["cohort"]])
            if g.isna().any():
                raise AnalysisError(f"{r['name']}: {int(g.isna().sum())} images have no patient or exam cluster")
            g = g.to_numpy()
        boots = metrics_from_counts(f, resample_counts(len(f), g, BOOT_N, BOOT_SEED))
        row = {"model": r["model"], "condition": r["condition"], "seed": r["seed"], "cohort": r["cohort"], "job": r["name"],
               "source": r["source"], "val_loss": r.get("val_loss"), "val_accuracy": r.get("val_acc"),
               "ci_method": "patient-clustered bootstrap" if g is not None else "image bootstrap",
               **{k: m[k] for k in SCALAR}}
        for k in ("accuracy", "qwk", "macro_auc"):
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = ci(boots[k])
        rows.append(row)
    return pd.DataFrame(rows)


def condition_table(per_run: pd.DataFrame) -> pd.DataFrame:
    keys = ["accuracy", "qwk", "macro_auc", "balanced_accuracy", "macro_f1", "referable_auc", "referable_sensitivity",
            "referable_specificity", "adjacent_share_of_errors", "distant_error_rate", "mild_recall", "ece", "brier", "val_loss", "val_accuracy"]
    rows = []
    for (model, cond, cohort), g in per_run.groupby(["model", "condition", "cohort"], sort=False):
        row = {"model": model, "condition": cond, "cohort": cohort, "n_runs": len(g), "n_collapsed": int(g["collapsed"].sum()),
               "seeds": " ".join(str(int(s)) for s in g["seed"].dropna())}
        for k in keys:
            v = pd.to_numeric(g[k], errors="coerce").dropna()
            row[f"{k}_mean"] = v.mean() if len(v) else np.nan
            row[f"{k}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            if k in ("accuracy", "qwk", "macro_auc"):
                row[f"{k}_min"], row[f"{k}_max"] = (v.min(), v.max()) if len(v) else (np.nan, np.nan)
        healthy = g[~g["collapsed"]]
        row["accuracy_mean_non_collapsed"] = healthy["accuracy"].mean() if len(healthy) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def classwise_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        for c in r["metrics"]["_rows"]:
            rows.append({"model": r["model"], "condition": r["condition"], "seed": r["seed"], "cohort": r["cohort"], "job": r["name"], **c})
    return pd.DataFrame(rows)


def classwise_summary(classwise: pd.DataFrame) -> pd.DataFrame:
    keys = ["sensitivity", "specificity", "precision", "f1", "auc_ovr"]
    out = classwise.groupby(["model", "condition", "cohort", "grade", "label"], sort=False)[keys].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def compare(runs: list[dict], family: str, label: str, cond_a: str, cond_b: str, cohort: str,
            pair_by_seed: bool, groups=None) -> tuple[list[dict], dict]:
    a_runs = sorted([r for r in runs if r["condition"] == cond_a and r["cohort"] == cohort], key=lambda r: (r["seed"] is None, r["seed"]))
    b_runs = sorted([r for r in runs if r["condition"] == cond_b and r["cohort"] == cohort], key=lambda r: (r["seed"] is None, r["seed"]))
    if not a_runs or not b_runs:
        raise AnalysisError(f"{label}: missing runs for {cond_a} or {cond_b} on {cohort}")
    ids = a_runs[0]["frame"]["image_id"].tolist()
    frames_a = [aligned(r["frame"], ids) for r in a_runs]
    frames_b = [aligned(r["frame"], ids) for r in b_runs]
    g = None if groups is None else pd.Series(ids).map(groups).to_numpy()
    counts = resample_counts(len(ids), g, BOOT_N, BOOT_SEED)
    boot_a = [metrics_from_counts(f, counts) for f in frames_a]
    boot_b = [metrics_from_counts(f, counts) for f in frames_b]
    pairs = []
    if pair_by_seed:
        seeds_b = {r["seed"]: i for i, r in enumerate(b_runs)}
        for i, r in enumerate(a_runs):
            j = seeds_b.get(r["seed"]) if r["seed"] is not None else None
            if j is None and len(b_runs) == 1:
                j = 0
            if j is None and len(a_runs) == 1:
                continue
            if j is not None:
                pairs.append((i, j))
        if len(a_runs) == 1:
            pairs = [(0, j) for j in range(len(b_runs))]
    pair_rows = []
    for i, j in pairs:
        fa, fb = frames_a[i], frames_b[j]
        ca = (fa["y_true"] == fa["y_pred"]).to_numpy()
        cb = (fb["y_true"] == fb["y_pred"]).to_numpy()
        a_only, b_only, p = mcnemar_exact(ca, cb)
        ma, mb = a_runs[i]["metrics"], b_runs[j]["metrics"]
        row = {"family": family, "comparison": label, "a": cond_a, "b": cond_b, "cohort": cohort,
               "seed_a": a_runs[i]["seed"], "seed_b": b_runs[j]["seed"], "n": len(ids),
               "correct_only_a": a_only, "correct_only_b": b_only, "mcnemar_exact_p": p}
        for k in ("accuracy", "qwk", "macro_auc"):
            diff = boot_a[i][k] - boot_b[j][k]
            row[f"{k}_a"], row[f"{k}_b"] = ma[k], mb[k]
            row[f"delta_{k}"] = ma[k] - mb[k]
            row[f"delta_{k}_ci_low"], row[f"delta_{k}_ci_high"] = ci(diff)
        pair_rows.append(row)
    # Hierarchical bootstrap: resample images (shared by both models) and, independently, each model's seeds.
    rng = np.random.default_rng(BOOT_SEED + 1)
    pick_a = rng.integers(0, len(a_runs), size=(BOOT_N, len(a_runs)))
    pick_b = rng.integers(0, len(b_runs), size=(BOOT_N, len(b_runs)))
    agg = {"family": family, "comparison": label, "a": cond_a, "b": cond_b, "cohort": cohort,
           "n_runs_a": len(a_runs), "n_runs_b": len(b_runs), "n_images": len(ids),
           "ci_method": "hierarchical bootstrap (images" + (" clustered by patient" if groups is not None else "") + " x seeds)"}
    for k in ("accuracy", "qwk", "macro_auc"):
        mat_a = np.stack([b[k] for b in boot_a], axis=1)
        mat_b = np.stack([b[k] for b in boot_b], axis=1)
        mean_a = np.take_along_axis(mat_a, pick_a, 1).mean(1)
        mean_b = np.take_along_axis(mat_b, pick_b, 1).mean(1)
        diff = mean_a - mean_b
        agg[f"{k}_mean_a"] = float(np.mean([r["metrics"][k] for r in a_runs]))
        agg[f"{k}_mean_b"] = float(np.mean([r["metrics"][k] for r in b_runs]))
        agg[f"delta_{k}"] = agg[f"{k}_mean_a"] - agg[f"{k}_mean_b"]
        agg[f"delta_{k}_ci_low"], agg[f"delta_{k}_ci_high"] = ci(diff)
        agg[f"delta_{k}_boot_p"] = boot_p(diff)
    return pair_rows, agg


def run_comparisons(runs: list[dict], specs: list[tuple], groups=None, optional=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """optional: leave out a comparison whose runs do not exist yet (results still to come from the GPU queue)."""
    pair_rows, agg_rows = [], []
    present = {(r["condition"], r["cohort"]) for r in runs}
    for spec in specs:
        if optional and not ((spec[2], spec[4]) in present and (spec[3], spec[4]) in present):
            continue
        pr, agg = compare(runs, *spec, groups=groups)
        pair_rows += pr
        agg_rows.append(agg)
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs["mcnemar_holm_p"] = np.nan
        for fam, idx in pairs.groupby("family").groups.items():
            pairs.loc[idx, "mcnemar_holm_p"] = holm(pairs.loc[idx, "mcnemar_exact_p"].tolist())
    return pairs, pd.DataFrame(agg_rows)


HEAD_TO_HEAD = [
    ("head_to_head", "H1 proposed primary: Qwen (lr chosen by validation loss) vs RetiZero, native preprocessing", Q_ON_2E4, R_OFF, "aptos361", True),
    ("head_to_head", "H2 Qwen original paper setting vs RetiZero, native preprocessing", Q_ON_5E4, R_OFF, "aptos361", True),
    ("head_to_head", "H3 both CLAHE-off", Q_OFF_5E4, R_OFF, "aptos361", True),
    ("head_to_head", "H4 both CLAHE-on", Q_ON_2E4, R_ON, "aptos361", True),
]
CLAHE = [
    ("clahe", "Qwen r8 CLAHE on vs off (lr5e-4)", "Qwen r8 CLAHE-on lr5e-4", "Qwen r8 CLAHE-off lr5e-4", "aptos361", True),
    ("clahe", "Qwen r16 CLAHE on vs off (lr5e-4)", Q_ON_5E4, Q_OFF_5E4, "aptos361", True),
    ("clahe", "Qwen r32 CLAHE on vs off (lr5e-4)", "Qwen r32 CLAHE-on lr5e-4", "Qwen r32 CLAHE-off lr5e-4", "aptos361", True),
    ("clahe", "Qwen zero-shot 4-bit CLAHE on vs off", "Qwen zero-shot 4bit CLAHE-on", "Qwen zero-shot 4bit CLAHE-off", "aptos361", True),
    ("clahe", "RetiZero CLAHE on vs off", R_ON, R_OFF, "aptos361", True),
]
RANK = [
    ("rank", "Qwen rank 8 vs 16, CLAHE-on (lr5e-4, alpha 16)", "Qwen r8 CLAHE-on lr5e-4", Q_ON_5E4, "aptos361", True),
    ("rank", "Qwen rank 32 vs 16, CLAHE-on (lr5e-4, alpha 16)", "Qwen r32 CLAHE-on lr5e-4", Q_ON_5E4, "aptos361", True),
    ("rank", "Qwen rank 8 vs 16, CLAHE-off (lr5e-4, alpha 16)", "Qwen r8 CLAHE-off lr5e-4", Q_OFF_5E4, "aptos361", True),
    ("rank", "Qwen rank 32 vs 16, CLAHE-off (lr5e-4, alpha 16)", "Qwen r32 CLAHE-off lr5e-4", Q_OFF_5E4, "aptos361", True),
]
TRAINING = [
    ("learning_rate", "Qwen r16 CLAHE-on lr2e-4 vs lr5e-4", Q_ON_2E4, Q_ON_5E4, "aptos361", True),
    ("learning_rate", "Qwen r16 CLAHE-on lr1e-4 vs lr5e-4", Q_ON_1E4, Q_ON_5E4, "aptos361", True),
    ("learning_rate", "Qwen r16 CLAHE-on lr2e-4 vs lr1e-4", Q_ON_2E4, Q_ON_1E4, "aptos361", True),
    ("adapter", "S7 Qwen LoRA adapter (lr2e-4) vs no adapter, 4-bit CLAHE-on", Q_ON_2E4, "Qwen zero-shot 4bit CLAHE-on", "aptos361", True),
    ("adapter", "S7 Qwen LoRA adapter (lr5e-4) vs no adapter, 4-bit CLAHE-on", Q_ON_5E4, "Qwen zero-shot 4bit CLAHE-on", "aptos361", True),
    ("adapter", "S7 Qwen LoRA adapter (lr5e-4) vs no adapter, 4-bit CLAHE-off", Q_OFF_5E4, "Qwen zero-shot 4bit CLAHE-off", "aptos361", True),
]
MESSIDOR = [
    ("messidor", "Messidor: Qwen lr2e-4 vs RetiZero, native preprocessing", Q_ON_2E4, R_OFF, "messidor", True),
    ("messidor", "Messidor: Qwen lr5e-4 vs RetiZero, native preprocessing", Q_ON_5E4, R_OFF, "messidor", True),
]
# Verified Messidor-2 (1,744 gradable images, built from the raw photos); present once the GPU queue has run.
MESSIDOR2 = [
    ("messidor2", "Messidor-2: Qwen lr2e-4 vs RetiZero, native preprocessing", Q_ON_2E4, R_OFF, "messidor2", True),
    ("messidor2", "Messidor-2: Qwen lr5e-4 vs RetiZero, native preprocessing", Q_ON_5E4, R_OFF, "messidor2", True),
    ("messidor2", "Messidor-2: both CLAHE-off", Q_OFF_5E4, R_OFF, "messidor2", True),
    ("messidor2", "Messidor-2: both CLAHE-on", Q_ON_2E4, R_ON, "messidor2", True),
    ("messidor2", "Messidor-2: Qwen adapter (lr2e-4) vs zero-shot, CLAHE-on", Q_ON_2E4, "Qwen zero-shot 4bit CLAHE-on", "messidor2", True),
    ("messidor2", "Messidor-2: RetiZero CLAHE on vs off", R_ON, R_OFF, "messidor2", True),
]
# Optional rank sweep at the validation-selected learning rate (revision_gpu_jobs.py --rank-sweep).
RANK_2E4 = [
    ("rank_lr2e-4", f"Qwen rank {r} vs 16, CLAHE-{c} (lr2e-4, alpha 16)", f"Qwen r{r} CLAHE-{c} lr2e-4", f"Qwen r16 CLAHE-{c} lr2e-4", "aptos361", True)
    for c in ("on", "off") for r in (8, 32)
] + [
    ("rank_lr2e-4", f"Qwen rank {r} vs 16 with alpha = rank (alpha/r = 1), CLAHE-on (lr2e-4)", f"Qwen r{r} CLAHE-on lr2e-4 alpha{r}", Q_ON_2E4, "aptos361", True)
    for r in (8, 32)
] + [
    ("rank_lr2e-4", "CLAHE on vs off, rank 16 (lr2e-4)", Q_ON_2E4, "Qwen r16 CLAHE-off lr2e-4", "aptos361", True),
]


def lr_selection(per_run: pd.DataFrame) -> pd.DataFrame:
    conds = [Q_ON_1E4, Q_ON_2E4, Q_ON_5E4]
    g = per_run[(per_run["condition"].isin(conds)) & (per_run["cohort"] == "aptos361")]
    out = g.groupby("condition").agg(n_runs=("job", "count"), val_loss_mean=("val_loss", "mean"), val_loss_sd=("val_loss", "std"),
                                     val_loss_max=("val_loss", "max"), n_collapsed=("collapsed", "sum"),
                                     test_accuracy_mean=("accuracy", "mean"), test_accuracy_sd=("accuracy", "std"),
                                     test_qwk_mean=("qwk", "mean"), test_macro_auc_mean=("macro_auc", "mean")).reindex(conds).reset_index()
    out["selected_by_validation_loss"] = out["val_loss_mean"] == out["val_loss_mean"].min()
    out["rank_by_validation_loss"] = out["val_loss_mean"].rank().astype(int)
    out["rank_by_test_accuracy"] = out["test_accuracy_mean"].rank(ascending=False).astype(int)
    return out


def collapse_vs_validation(per_run: pd.DataFrame) -> pd.DataFrame:
    """Would validation loss alone have flagged each collapsed Qwen run?"""
    q = per_run[(per_run["model"] == "Qwen") & per_run["val_loss"].notna() & (per_run["cohort"] == "aptos361")].copy()
    healthy = q.loc[~q["collapsed"], "val_loss"]
    threshold = healthy.mean() + 3 * healthy.std(ddof=1)
    q["val_loss_above_healthy_mean_plus_3sd"] = q["val_loss"] > threshold
    q["healthy_threshold"] = threshold
    return q[["condition", "seed", "job", "val_loss", "accuracy", "n_grades_predicted", "grades_never_predicted", "collapsed",
              "val_loss_above_healthy_mean_plus_3sd", "healthy_threshold"]].sort_values("val_loss", ascending=False)


def error_structure(per_run: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
    rows = []
    for (cond, cohort), g in per_run.groupby(["condition", "cohort"], sort=False):
        pooled = np.zeros((5, 5), int)
        for r in runs:
            if r["condition"] == cond and r["cohort"] == cohort:
                pooled += r["metrics"]["_cm"]
        dist = np.abs(np.subtract.outer(np.arange(5), np.arange(5)))
        total, errors = pooled.sum(), pooled[dist > 0].sum()
        rows.append({"condition": cond, "cohort": cohort, "n_runs": len(g), "pooled_images": int(total),
                     "exact": pooled[dist == 0].sum() / total, "off_by_1": pooled[dist == 1].sum() / total,
                     "off_by_2": pooled[dist == 2].sum() / total, "off_by_3_or_4": pooled[dist >= 3].sum() / total,
                     "share_of_errors_off_by_1": pooled[dist == 1].sum() / errors if errors else np.nan,
                     "qwk_mean": g["qwk"].mean(), "accuracy_mean": g["accuracy"].mean()})
    return pd.DataFrame(rows)


def misclassified(runs: list[dict], cond: str, cohort: str) -> pd.DataFrame:
    group = sorted([r for r in runs if r["condition"] == cond and r["cohort"] == cohort], key=lambda r: r["seed"])
    ids = group[0]["frame"]["image_id"].tolist()
    frame = pd.DataFrame({"image_id": ids, "y_true": group[0]["frame"]["y_true"].to_numpy(int)})
    preds, ptrue = [], []
    for r in group:
        f = aligned(r["frame"], ids)
        frame[f"pred_seed{r['seed']}"] = f["y_pred"].to_numpy(int)
        preds.append(f["y_pred"].to_numpy(int))
        ptrue.append(f[PROB].to_numpy(float)[np.arange(len(ids)), frame["y_true"].to_numpy(int)])
    preds = np.stack(preds, 1)
    frame["n_seeds"] = len(group)
    frame["n_seeds_wrong"] = (preds != frame["y_true"].to_numpy()[:, None]).sum(1)
    frame["modal_pred"] = stats.mode(preds, axis=1, keepdims=False).mode
    frame["mean_p_true_grade"] = np.mean(ptrue, 0)
    frame["max_grade_distance"] = np.abs(preds - frame["y_true"].to_numpy()[:, None]).max(1)
    frame.insert(0, "condition", cond)
    frame.insert(1, "cohort", cohort)
    return frame[frame["n_seeds_wrong"] > 0].sort_values(["n_seeds_wrong", "max_grade_distance", "mean_p_true_grade"],
                                                         ascending=[False, False, True])


# --------------------------------------------------------------------------- Messidor audit

def messidor_audit(runs: list[dict], grades_csv: Path | None, pairs_csv: Path | None) -> tuple[pd.DataFrame, dict]:
    m = [r for r in runs if r["cohort"] == "messidor"]
    frame = m[0]["frame"]
    patient = {}
    for r in m:
        for raw in pd.read_csv(r["path"], usecols=["image_id"])["image_id"]:
            hit = re.search(r"messidor_p_(\d+)_(\d{8}_\d+_\d{4}_PP)", raw)
            if hit:
                patient[hit.group(2)] = f"p{hit.group(1)}"
    if len(patient) != len(frame):
        raise AnalysisError("Could not read a patient ID for every Messidor image")
    facts = {"archived_images": len(frame), "archived_patients": len(set(patient.values()))}
    if grades_csv is None:
        facts["adjudicated_grades"] = "not supplied"
        return pd.DataFrame(), {**facts, "_patient_of": patient}
    adj = pd.read_csv(bpt.track(grades_csv))
    adj["key"] = adj["image_id"].str.replace(r"\.(png|jpg|JPG|jpeg)$", "", regex=True)
    png = adj[adj["image_id"].str.lower().str.endswith(".png")]
    joined = frame[["image_id", "y_true"]].merge(adj, left_on="image_id", right_on="key", how="left")
    facts.update({
        "adjudicated_rows": len(adj), "adjudicated_png_rows": len(png),
        "archived_matched_to_adjudicated": int(joined["adjudicated_dr_grade"].notna().sum()),
        "archived_label_equals_adjudicated": int((joined["y_true"] == joined["adjudicated_dr_grade"]).sum()),
        "png_not_in_archive": "; ".join(f"{k} (gradable={g})" for k, g in
                                        png.loc[~png["key"].isin(frame["image_id"]), ["image_id", "adjudicated_gradable"]].itertuples(index=False)),
        "archived_grade_counts": " ".join(str(int((frame["y_true"] == g).sum())) for g in GRADES),
        "adjudicated_png_gradable_counts": " ".join(str(int((png["adjudicated_dr_grade"] == g).sum())) for g in GRADES),
        "adjudicated_all_gradable_counts": " ".join(str(int((adj["adjudicated_dr_grade"] == g).sum())) for g in GRADES),
    })
    if facts["archived_label_equals_adjudicated"] != len(frame):
        raise AnalysisError("Archived Messidor labels differ from the adjudicated grades")
    if pairs_csv is not None:
        lines = [l.strip() for l in open(bpt.track(pairs_csv), encoding="utf-8").read().splitlines()[1:] if l.strip()]
        exams = [[x.strip().rsplit(".", 1)[0] for x in l.split(";")] for l in lines]
        png_exams = [e for e in exams if e[0] in patient or e[1] in patient or e[0].endswith("_PP")]
        both = [e for e in png_exams if e[0] in patient and e[1] in patient]
        facts.update({"pairs_exams_total": len(exams), "pairs_png_exams": len(png_exams),
                      "pairs_both_images_archived": len(both),
                      "pairs_same_patient_id": sum(patient[a] == patient[b] for a, b in both),
                      "pairs_one_image_archived": len(png_exams) - len(both)})
    joined["patient_id"] = joined["image_id_x"].map(patient)
    return joined, {**facts, "_patient_of": patient}


def mild_normal_analysis(runs: list[dict], adjudicated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dme = None
    if len(adjudicated):
        dme = adjudicated.set_index("image_id_x")["adjudicated_dme"]
    for r in runs:
        f = r["frame"]
        y, p = f["y_true"].to_numpy(int), f["y_pred"].to_numpy(int)
        probs = f[PROB].to_numpy(float)
        mild = y == 1
        row = {"model": r["model"], "condition": r["condition"], "seed": r["seed"], "cohort": r["cohort"], "job": r["name"],
               "true_grade_share": " ".join(f"{(y == g).mean():.3f}" for g in GRADES),
               "predicted_grade_share": " ".join(f"{(p == g).mean():.3f}" for g in GRADES),
               "mild_n": int(mild.sum()), "mild_predicted_normal": int((mild & (p == 0)).sum()),
               "mild_predicted_normal_rate": float((mild & (p == 0)).sum() / mild.sum()),
               "mild_recall": float((mild & (p == 1)).sum() / mild.sum()),
               "normal_predicted_mild_rate": float(((y == 0) & (p == 1)).sum() / (y == 0).sum()),
               "mean_p0_on_mild": float(probs[mild, 0].mean()), "mean_p1_on_mild": float(probs[mild, 1].mean()),
               "auc_mild_vs_normal": float(bpt.weighted_auc(y[(y <= 1)] == 1, probs[y <= 1, 1] / np.clip(probs[y <= 1, :2].sum(1), 1e-12, None),
                                                            np.ones(int((y <= 1).sum())))[0])}
        if dme is not None and r["cohort"] == "messidor":
            d = f["image_id"].map(dme).to_numpy()
            for flag in (0, 1):
                sel = mild & (d == flag)
                row[f"mild_dme{flag}_n"] = int(sel.sum())
                row[f"mild_dme{flag}_predicted_normal_rate"] = float((sel & (p == 0)).sum() / sel.sum()) if sel.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- class counts (S9)

def class_counts(experiments: Path, original_split: Path | None) -> pd.DataFrame:
    rows = []
    clean = experiments / "runs" / "full_r16_on_duplicate_excluded_t42_a3407_01"
    sources = [("clean (duplicate-excluded)", {s: clean / f"aptos_{s}.csv" for s in ("train", "val", "test")})]
    if original_split is not None:
        sources.append(("original 80/10/10", {s: original_split / f"{s}.csv" for s in ("train", "val", "test")}))
    for name, files in sources:
        for split, path in files.items():
            labels = pd.read_csv(bpt.track(path))["label"].astype(int)
            counts = labels.value_counts().reindex(GRADES, fill_value=0)
            n = int(counts.sum())
            row = {"split_set": name, "split": split, "n": n, **{f"g{g}": int(counts[g]) for g in GRADES}}
            rows.append({**row, "view": "images on disk (both models)"})
            if split == "train":
                rows.append({"split_set": name, "split": split, "n": n, "view": "Qwen: expected draws per epoch (inverse-frequency WeightedRandomSampler, with replacement, num_samples = n)",
                             **{f"g{g}": n / 5 for g in GRADES}})
                rows.append({"split_set": name, "split": split, "n": n, "view": "RetiZero: same images per epoch; loss weighted by inverse class frequency; online augmentation (RandomResizedCrop 0.8-1.0, flip, rotation 180)",
                             **{f"g{g}": int(counts[g]) for g in GRADES}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- efficiency (item 3)

def util_span(path: Path, stage: str) -> dict:
    u = pd.read_csv(bpt.track(path))
    u = u[u["stage"] == stage]
    if u.empty:
        return {}
    t = pd.to_datetime(u["utc"])
    return {"wall_s_from_samples": (t.max() - t.min()).total_seconds(), "peak_card_vram_mib": u["vram_mib"].max(),
            "mean_gpu_percent": u["gpu_percent"].mean(), "peak_ram_gib": u["ram_bytes"].max() / 2**30}


def efficiency_final(experiments: Path) -> pd.DataFrame:
    rows = []
    committed = pd.read_csv(bpt.track(COMMITTED / "efficiency" / "efficiency_comparison.csv"))
    keep = ["Total parameters", "Trainable parameters", "Scoring model load time", "Scoring warm median time per image (batch 1)",
            "Scoring throughput (batch 1)", "Scoring throughput (batched, 64)", "Scoring peak GPU memory, PyTorch allocated (batch 1)",
            "Scoring peak GPU memory, whole card", "Training epochs run (early stopping)", "Training wall time, all attempts",
            "Training time per epoch (incl. validation)", "Training peak GPU memory, PyTorch allocated", "Training peak GPU memory, whole card"]
    for _, r in committed[committed["quantity"].isin(keep)].iterrows():
        if r["model"].startswith("RetiZero") and r["quantity"].startswith("Training"):
            continue  # replaced below by the full retizero_01 training runs
        rows.append({"quantity": r["quantity"], "unit": r["unit"], "model": r["model"], "n": r["n"], "mean": r["mean"],
                     "sd": r["sd"], "source": "efficiency_comparison.csv (committed)"})
    rz = experiments / "retizero_runs" / "retizero_01" / "train"
    per = []
    for d in sorted(rz.iterdir()):
        if not d.name.startswith("matched"):
            continue
        result = bpt.read_json(d / "train_result.json")
        attempt = next(a for a in sorted((d / "attempts").iterdir()) if a.name.startswith("train-"))
        report = bpt.read_json(attempt / "run_report.json")
        hist = result["history"]
        per.append({"epochs": result["epochs_completed"], "wall": report["elapsed_seconds"],
                    "per_epoch": np.mean([h["train_seconds"] + h["validation_seconds"] for h in hist]),
                    "ips": np.mean([h["train_images_per_second"] for h in hist]),
                    "torch_peak": max(h["torch_peak_allocated_bytes"] for h in hist) / 2**30, "card_peak": report["peak_vram_mib"]})
    p = pd.DataFrame(per)
    model = "RetiZero ViT-L/16 + LoRA r8, all weights trained (matched, CLAHE-off)"
    for q, unit, col in [("Training epochs run (early stopping)", "epochs", "epochs"), ("Training wall time, all attempts", "s", "wall"),
                         ("Training time per epoch (incl. validation)", "s/epoch", "per_epoch"), ("Training throughput", "images/s", "ips"),
                         ("Training peak GPU memory, PyTorch allocated", "GiB", "torch_peak"), ("Training peak GPU memory, whole card", "MiB", "card_peak")]:
        rows.append({"quantity": q, "unit": unit, "model": model, "n": len(p), "mean": p[col].mean(), "sd": p[col].std(ddof=1),
                     "source": "retizero_01/train/matched_t*/ (train_result.json, run_report.json)"})
    final = experiments / "final_runs" / "final_01" / "train"
    per = []
    for d in sorted(final.glob("r16_on_lr0p0002_t*")):
        latest = Path(bpt.read_json(d / "latest_attempt.json")["path"]).name
        span = util_span(d / "attempts" / latest / "utilization.csv", "train")
        per.append({"wall": span["wall_s_from_samples"], "card_peak": span["peak_card_vram_mib"],
                    "steps": bpt.read_json(d / "train_result.json")["global_step"]})
    p = pd.DataFrame(per)
    model = "Qwen3-VL-8B + LoRA r16, 4-bit, CLAHE-on, lr2e-4"
    rows.append({"quantity": "Training wall time, final attempt (from 5 s utilisation samples)", "unit": "s", "model": model, "n": len(p),
                 "mean": p["wall"].mean(), "sd": p["wall"].std(ddof=1), "source": "final_01/train/r16_on_lr0p0002_t*/attempts/*/utilization.csv"})
    rows.append({"quantity": "Training peak GPU memory, whole card", "unit": "MiB", "model": model, "n": len(p),
                 "mean": p["card_peak"].mean(), "sd": p["card_peak"].std(ddof=1), "source": "same"})
    out = pd.DataFrame(rows)
    out["hardware"] = "NVIDIA RTX 5070 Ti 16 GB, same PC for every row (system_specs.json in each attempt)"
    return out


# --------------------------------------------------------------------------- near-duplicate sensitivity

def near_duplicate_sensitivity(runs: list[dict], flagged_csv: Path) -> pd.DataFrame:
    flagged = pd.read_csv(bpt.track(flagged_csv))
    rows = []
    for threshold, group in flagged.groupby("threshold"):
        drop = set(group["image_id"])
        for cond in (Q_ON_2E4, Q_ON_5E4, R_OFF, R_ON):
            for r in [r for r in runs if r["condition"] == cond and r["cohort"] == "aptos361"]:
                f = r["frame"]
                kept = f[~f["image_id"].isin(drop)]
                full, sub = r["metrics"], bpt.point_metrics(kept)
                removed = f[f["image_id"].isin(drop)]
                rows.append({"threshold": threshold, "excluded_test_images": len(drop & set(f["image_id"])),
                             "excluded_grade_counts": " ".join(str(int((removed["y_true"] == g).sum())) for g in GRADES),
                             "all_grade_counts": " ".join(str(int((f["y_true"] == g).sum())) for g in GRADES), "condition": cond,
                             "seed": r["seed"], "n_kept": len(kept), "accuracy_all": full["accuracy"], "accuracy_kept": sub["accuracy"],
                             "qwk_all": full["qwk"], "qwk_kept": sub["qwk"], "macro_auc_all": full["macro_auc"], "macro_auc_kept": sub["macro_auc"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- output

def write(frame: pd.DataFrame, path: Path, written: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    written[str(path)] = len(frame)


def fmt(v, pct=False, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{100 * v:.2f}%" if pct else f"{v:.{d}f}"


def write_summary(out: Path, cond: pd.DataFrame, lr: pd.DataFrame, collapse: pd.DataFrame, h2h_agg, h2h_pairs, clahe_agg,
                  rank_agg, train_agg, mess_facts, mild, errs, msd_agg, near, extra_sections=()):
    def mean_sd(c, k, cohort="aptos361", pct=True):
        r = cond[(cond["condition"] == c) & (cond["cohort"] == cohort)]
        if r.empty:
            return "n/a"
        r = r.iloc[0]
        sd = r[f"{k}_sd"]
        return f"{fmt(r[f'{k}_mean'], pct)} ± {fmt(sd, pct) if not np.isnan(sd) else 'n/a'}"

    def agg_lines(agg):
        lines = ["| Comparison | A | B | Δ accuracy [95% CI] | Δ QWK [95% CI] | Δ macro AUC [95% CI] |", "|---|---|---|---|---|---|"]
        for _, r in agg.iterrows():
            lines.append(f"| {r['comparison']} | {r['a']} | {r['b']} | {100*r['delta_accuracy']:+.2f} pt [{100*r['delta_accuracy_ci_low']:+.2f}, {100*r['delta_accuracy_ci_high']:+.2f}] "
                         f"| {r['delta_qwk']:+.3f} [{r['delta_qwk_ci_low']:+.3f}, {r['delta_qwk_ci_high']:+.3f}] "
                         f"| {r['delta_macro_auc']:+.3f} [{r['delta_macro_auc_ci_low']:+.3f}, {r['delta_macro_auc_ci_high']:+.3f}] |")
        return "\n".join(lines)

    s = ["# Revision analyses", "",
         "Built by `build_revision_analysis.py` from the saved per-image predictions only. CIs are 95% percentile bootstrap "
         f"({BOOT_N} resamples). Δ rows use a hierarchical bootstrap that resamples the images (shared by both arms) and each arm's seeds. "
         "Per-seed exact McNemar tests with Holm adjustment are in the `*_pairs.csv` files. Seed numbers are only a label across models, "
         "so the hierarchical interval is the one to quote for Qwen vs RetiZero.", "",
         "## Headline conditions (361 clean test images, mean ± SD over 3 seeds)", "",
         "| Condition | Accuracy | QWK | Macro AUC | Balanced acc. | Referable DR sens. / spec. | ECE | Collapsed runs (≤2 grades) |", "|---|---|---|---|---|---|---|---|"]
    for c in (Q_ON_2E4, Q_ON_1E4, Q_ON_5E4, Q_OFF_5E4, R_OFF, R_ON, "Qwen zero-shot 4bit CLAHE-on", "Qwen zero-shot 4bit CLAHE-off"):
        r = cond[(cond["condition"] == c) & (cond["cohort"] == "aptos361")]
        if r.empty:
            continue
        r = r.iloc[0]
        s.append(f"| {c} | {mean_sd(c, 'accuracy')} | {mean_sd(c, 'qwk', pct=False)} | {mean_sd(c, 'macro_auc', pct=False)} | {mean_sd(c, 'balanced_accuracy')} "
                 f"| {fmt(r['referable_sensitivity_mean'], True)} / {fmt(r['referable_specificity_mean'], True)} | {fmt(r['ece_mean'])} | {int(r['n_collapsed'])} of {int(r['n_runs'])} |")
    s += ["", "## Learning rate selected on validation loss only (G2, S8)", "",
          "| Condition | Mean validation loss | Max validation loss | Collapsed | Test accuracy | Rank by val. loss | Rank by test acc. |", "|---|---|---|---|---|---|---|"]
    for _, r in lr.iterrows():
        s.append(f"| {r['condition']} | {r['val_loss_mean']:.5f} | {r['val_loss_max']:.5f} | {int(r['n_collapsed'])} | {fmt(r['test_accuracy_mean'], True)} | {r['rank_by_validation_loss']} | {r['rank_by_test_accuracy']} |")
    failed = collapse[collapse["n_grades_predicted"] <= 3]
    working = collapse[collapse["n_grades_predicted"] >= 4]
    s += ["", "Validation loss alone ranks the learning rates in the same order as test accuracy.", "",
          f"Across all {len(collapse)} fine-tuned Qwen runs, the {len(failed)} runs that predict at most 3 of the 5 grades have validation loss "
          f"{failed['val_loss'].min():.5f} to {failed['val_loss'].max():.5f}; the {len(working)} runs that predict 4 or 5 grades have "
          f"{working['val_loss'].min():.5f} to {working['val_loss'].max():.5f}. "
          + ("The ranges do not overlap, so every failed run is detectable from validation data alone, without the test set."
             if failed["val_loss"].min() > working["val_loss"].max() else "The ranges overlap.")
          + " (`collapse_vs_validation.csv`)",
          "", "## Qwen vs RetiZero (S12)", "", agg_lines(h2h_agg), "",
          "## CLAHE (item 2, S3, S9)", "", agg_lines(clahe_agg), "",
          "## LoRA rank, alpha fixed at 16 (item 1, S6)", "", agg_lines(rank_agg), "",
          "Rank means at lr 5e-4 (accuracy, mean ± SD; collapsed runs included):", ""]
    for c in ("Qwen r8 CLAHE-on lr5e-4", Q_ON_5E4, "Qwen r32 CLAHE-on lr5e-4", "Qwen r8 CLAHE-off lr5e-4", Q_OFF_5E4, "Qwen r32 CLAHE-off lr5e-4"):
        r = cond[(cond["condition"] == c) & (cond["cohort"] == "aptos361")].iloc[0]
        s.append(f"- {c}: {mean_sd(c, 'accuracy')}, QWK {mean_sd(c, 'qwk', pct=False)}, {int(r['n_collapsed'])} of 3 collapsed, "
                 f"mean of non-collapsed runs {fmt(r['accuracy_mean_non_collapsed'], True)}")
    s += ["", "## Learning rate and adapter (S7, S8)", "", agg_lines(train_agg), "",
          "## Error structure (S13)", "", "| Condition | Cohort | Exact | Off by 1 | Off by 2 | Off by ≥3 | Off-by-1 share of errors |", "|---|---|---|---|---|---|---|"]
    for _, r in errs.iterrows():
        if r["condition"] in (Q_ON_2E4, Q_ON_5E4, R_OFF, R_ON):
            s.append(f"| {r['condition']} | {r['cohort']} | {fmt(r['exact'], True)} | {fmt(r['off_by_1'], True)} | {fmt(r['off_by_2'], True)} | {fmt(r['off_by_3_or_4'], True)} | {fmt(r['share_of_errors_off_by_1'], True)} |")
    s += ["", "## Messidor-2 (archived 1,057-image cohort)", ""]
    for k, v in mess_facts.items():
        if not k.startswith("_"):
            s.append(f"- {k}: {v}")
    s += ["", "Mild → Normal (mean over seeds):", "", "| Condition | Cohort | Mild predicted Normal | Mild recall | Normal predicted Mild | Mild-vs-Normal AUC |", "|---|---|---|---|---|---|"]
    mm = mild.groupby(["condition", "cohort"], sort=False)[["mild_predicted_normal_rate", "mild_recall", "normal_predicted_mild_rate", "auc_mild_vs_normal"]].mean().reset_index()
    for _, r in mm.iterrows():
        if r["condition"] in (Q_ON_2E4, Q_ON_5E4, R_OFF, "RetiZero saved app checkpoint", "Qwen zero-shot 4bit CLAHE-on"):
            s.append(f"| {r['condition']} | {r['cohort']} | {fmt(r['mild_predicted_normal_rate'], True)} | {fmt(r['mild_recall'], True)} | {fmt(r['normal_predicted_mild_rate'], True)} | {fmt(r['auc_mild_vs_normal'])} |")
    s += ["", "Messidor head-to-head (patient-clustered):", "", agg_lines(msd_agg)]
    for title, body in extra_sections:  # results that exist only after the final_02 / retizero_02 GPU queue
        if isinstance(body, pd.DataFrame) and "delta_accuracy" in body:
            s += ["", f"## {title}", "", agg_lines(body)]
        elif isinstance(body, pd.DataFrame) and len(body):
            s += ["", f"## {title}", "", bpt.md_table(body)]
    if near is not None and len(near):
        s += ["", "## Near-duplicate sensitivity (S10)", "",
              f"All 361 test images by grade 0-4: {near['all_grade_counts'].iloc[0]}. Removal is not random, so compare against the excluded grade mix.", "",
              "| pHash threshold | Excluded test images (grades 0-4) | Condition | Accuracy all → kept (mean) | QWK all → kept |", "|---|---|---|---|---|"]
        for (t, c), g in near.groupby(["threshold", "condition"], sort=False):
            s.append(f"| ≤{t} | {g['excluded_test_images'].iloc[0]} ({g['excluded_grade_counts'].iloc[0]}) | {c} | {fmt(g['accuracy_all'].mean(), True)} → {fmt(g['accuracy_kept'].mean(), True)} | {g['qwk_all'].mean():.3f} → {g['qwk_kept'].mean():.3f} |")
    (out / "SUMMARY.md").write_text("\n".join(s) + "\n", encoding="utf-8")


def messidor2_clusters(runs: list[dict], pairs_csv: Path | None, dups_csv: Path | None) -> dict:
    """image_id -> exam cluster for the verified Messidor-2 cohort (both eyes of an exam resample together).

    Exams whose photos are pixel-identical duplicates of another exam's (messidor_duplicate_pairs.csv) are merged.
    """
    ids = {i for r in runs if r["cohort"] == "messidor2" for i in r["frame"]["image_id"]}
    if not ids:
        return {}
    if pairs_csv is None:
        raise AnalysisError("--messidor-pairs is needed to cluster the Messidor-2 cohort by exam")
    exam = {}
    lines = [l.strip() for l in open(bpt.track(pairs_csv), encoding="utf-8-sig").read().splitlines()[1:] if l.strip()]
    for k, line in enumerate(lines, 1):
        for name in line.split(";"):
            exam[Path(name.strip()).stem.lower()] = f"exam_{k:04d}"
    missing = sorted(i for i in ids if i.lower() not in exam)
    if missing:
        raise AnalysisError(f"{len(missing)} Messidor-2 images are not in pairs.csv, e.g. {missing[:3]}")
    if dups_csv is not None and dups_csv.is_file():
        dups = pd.read_csv(bpt.track(dups_csv))
        alias = {}
        for _, g in dups.groupby("group"):
            exams = sorted(exam[i.lower()] for i in g["image"])
            alias.update({e: exams[0] for e in exams})
        for _ in range(len(alias)):  # transitive closure
            alias = {e: alias.get(a, a) for e, a in alias.items()}
        exam = {k: alias.get(v, v) for k, v in exam.items()}
    return {i: exam[i.lower()] for i in ids}


def archived_vs_verified(runs: list[dict]) -> pd.DataFrame:
    """The same model on the 1,057 archived images and on the same images rebuilt from the raw photos."""
    rows = []
    archived = {(r["condition"], r["seed"]): r for r in runs if r["cohort"] == "messidor"}
    for r in runs:
        if r["cohort"] != "messidor2" or (r["condition"], r["seed"]) not in archived:
            continue
        a = archived[(r["condition"], r["seed"])]
        ids = a["frame"]["image_id"].tolist()
        v = r["frame"].set_index("image_id")
        if not set(ids) <= set(v.index):
            raise AnalysisError(f"{r['name']}: the verified cohort lacks archived images")
        v = aligned(r["frame"], ids)
        if not np.array_equal(v["y_true"].to_numpy(), a["frame"]["y_true"].to_numpy()):
            raise AnalysisError(f"{r['name']}: verified grades differ from the archived grades on the shared images")
        sub = bpt.point_metrics(v)
        rows.append({"model": r["model"], "condition": r["condition"], "seed": r["seed"], "archived_job": a["name"],
                     "verified_job": r["name"], "shared_images": len(ids),
                     "same_predicted_grade": int((v["y_pred"].to_numpy() == a["frame"]["y_pred"].to_numpy()).sum()),
                     "max_abs_probability_difference": float(np.abs(v[PROB].to_numpy(float) - a["frame"][PROB].to_numpy(float)).max()),
                     "accuracy_archived": a["metrics"]["accuracy"], "accuracy_verified_same_images": sub["accuracy"],
                     "qwk_archived": a["metrics"]["qwk"], "qwk_verified_same_images": sub["qwk"],
                     "accuracy_verified_all_1744": r["metrics"]["accuracy"]})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["seed"] = frame["seed"].astype("Int64")
    return frame


def timing_table(experiments: Path) -> pd.DataFrame:
    """Warm per-image timing, generation check and energy of the final_02 timing jobs (revision_gpu_jobs.py)."""
    scores = experiments / "final_runs" / "final_02" / "scores"
    if not scores.is_dir():
        return pd.DataFrame()
    energy_path = experiments / "final_runs" / "final_02" / "energy_per_job.csv"
    energy = pd.read_csv(bpt.track(energy_path)).set_index("job") if energy_path.is_file() else pd.DataFrame()
    rows = []
    for d in sorted(scores.glob("*")):
        if not (d / "summary.json").is_file():
            continue
        s = bpt.read_json(d / "summary.json")
        timing, generation = s.get("full_answer_timing", {}), s.get("generation") or {}
        agreement = s.get("agreement_with_saved_predictions", {})
        row = {"job": d.name, "cohort": s["identity"]["cohort"], "images": s["metrics"]["samples"],
               "median_s_per_image": timing.get("median_seconds"), "p95_s_per_image": timing.get("p95_seconds"),
               "images_per_s": timing.get("images_per_second"), "model_load_s": s.get("model_load_seconds"),
               "torch_peak_gib": s.get("torch_peak_allocated_bytes", 0) / 2**30,
               "generation_median_s": generation.get("timing", {}).get("median_seconds"),
               "generation_unparseable": generation.get("unparseable_outputs"),
               "saved_prediction_grade_mismatches": agreement.get("grade_mismatches")}
        if d.name in energy.index:
            row.update(mean_board_w=energy.loc[d.name, "mean_w"], energy_wh=energy.loc[d.name, "energy_wh"],
                       energy_wh_per_image=energy.loc[d.name, "energy_wh"] / row["images"])
        rows.append(row)
    return pd.DataFrame(rows)


def build(args) -> dict:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    written: dict = {}
    runs = discover(args.experiments_root)
    load_all(runs)
    write(reproduction_table(runs, args.experiments_root), out / "reproduction_check.csv", written)
    write(check_alignment(runs), out / "image_alignment_check.csv", written)

    adjudicated, facts = messidor_audit(runs, args.messidor_grades, args.messidor_pairs)
    patient_of = facts["_patient_of"]
    if args.messidor_duplicates and args.messidor_duplicates.is_file():
        # Pixel-identical photos filed under two patient IDs: resample them as one patient.
        dups = pd.read_csv(bpt.track(args.messidor_duplicates))
        alias = {}
        for _, g in dups.groupby("group"):
            ids = sorted(g["patient_id"])
            alias.update({p: ids[0] for p in ids})
        merged = {}
        while alias != merged:  # transitive closure (p89-p91 and p90-p92 groups chain through shared IDs)
            merged = dict(alias)
            alias = {p: min(alias[p], alias.get(alias[p], alias[p])) for p in alias}
        patient_of = {img: alias.get(p, p) for img, p in patient_of.items()}
        facts["patient_clusters_after_merging_duplicates"] = len(set(patient_of.values()))
    # The per-image join is not written: it would re-publish the adjudicated grade table. The audit counts below are the record.
    write(pd.DataFrame([{k: v for k, v in facts.items() if not k.startswith("_")}]).T.reset_index()
          .rename(columns={"index": "fact", 0: "value"}), out / "messidor_label_audit.csv", written)

    exam_of = messidor2_clusters(runs, args.messidor_pairs, args.messidor_duplicates)
    groups = {"messidor": patient_of, **({"messidor2": exam_of} if exam_of else {})}
    per_run = per_run_table(runs, groups)
    write(per_run, out / "per_run_metrics.csv", written)
    cond = condition_table(per_run)
    write(cond, out / "condition_summary.csv", written)
    cw = classwise_table(runs)
    write(cw, out / "classwise_per_run.csv", written)
    write(classwise_summary(cw), out / "classwise_condition_mean.csv", written)
    lr = lr_selection(per_run)
    write(lr, out / "lr_selection_by_validation_loss.csv", written)
    collapse = collapse_vs_validation(per_run)
    write(collapse, out / "collapse_vs_validation.csv", written)

    results = {}
    for name, specs, cluster, optional in (("head_to_head", HEAD_TO_HEAD, None, False), ("clahe", CLAHE, None, False),
                                           ("rank", RANK, None, False), ("training", TRAINING, None, False),
                                           ("messidor", MESSIDOR, patient_of, False),
                                           ("messidor2", MESSIDOR2, exam_of, True), ("rank_lr2e-4", RANK_2E4, None, True)):
        pairs, agg = run_comparisons(runs, specs, cluster, optional=optional)
        if optional and agg.empty:
            results[name] = (pairs, agg)
            continue
        write(pairs, out / f"{name}_pairs.csv", written)
        write(agg, out / f"{name}_aggregate.csv", written)
        results[name] = (pairs, agg)

    errs = error_structure(per_run, runs)
    write(errs, out / "error_structure.csv", written)
    mild = mild_normal_analysis(runs, adjudicated)
    write(mild, out / "mild_vs_normal.csv", written)
    reliab = []
    for r in runs:
        if r["condition"] in (Q_ON_2E4, Q_ON_5E4, R_OFF, R_ON) and r["cohort"] == "aptos361":
            f = r["frame"]
            for row in reliability_rows(f[PROB].to_numpy(float), f["y_true"].to_numpy(int)):
                reliab.append({"condition": r["condition"], "seed": r["seed"], **row})
    write(pd.DataFrame(reliab), out / "calibration_reliability_bins.csv", written)
    archived_check = archived_vs_verified(runs)
    if len(archived_check):
        write(archived_check, out / "messidor_archived_vs_verified.csv", written)
    mis = pd.concat([misclassified(runs, c, coh) for c in (Q_ON_2E4, Q_ON_5E4, R_OFF) for coh in ("aptos361", "messidor", "messidor2")
                     if any(r["condition"] == c and r["cohort"] == coh for r in runs)], ignore_index=True)
    write(mis, out / "misclassified_headline.csv", written)
    write(class_counts(args.experiments_root, args.original_split), out / "class_counts.csv", written)
    write(efficiency_final(args.experiments_root), out / "efficiency_final.csv", written)
    timing = timing_table(args.experiments_root)
    if len(timing):
        write(timing, out / "timing_and_energy_final_02.csv", written)
    fallback = pd.read_csv(bpt.track(COMMITTED / "final_01" / "per_job_metrics.csv"))
    cols = ["job", "cohort", "generation_images", "unparseable", "generation_accuracy_strict", "generation_accuracy_unparseable_as_0", "accuracy"]
    fb = fallback.loc[fallback["generation_images"].notna(), cols].rename(columns={"accuracy": "likelihood_scorer_accuracy"})
    write(fb, out / "grade0_fallback_check.csv", written)

    near = None
    if args.near_duplicates and args.near_duplicates.is_file():
        near = near_duplicate_sensitivity(runs, args.near_duplicates)
        write(near, out / "near_duplicate_sensitivity.csv", written)

    write_summary(out, cond, lr, collapse, results["head_to_head"][1], results["head_to_head"][0], results["clahe"][1],
                  results["rank"][1], results["training"][1], {k: v for k, v in facts.items()}, mild, errs, results["messidor"][1], near,
                  extra_sections=[(title, results[key][1]) for key, title in
                                  (("messidor2", "Verified Messidor-2, 1,744 gradable images (exam-clustered)"),
                                   ("rank_lr2e-4", "LoRA rank at the validation-selected lr 2e-4"))
                                  if len(results[key][1])]
                                 + ([("Archived 1,057 vs the same images rebuilt from raw", archived_check)] if len(archived_check) else []))
    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "script_sha256": bpt.sha256(Path(__file__)),
        "metric_source_sha256": bpt.sha256(HERE / "build_paper_tables.py"),
        "arguments": {k: str(v) for k, v in vars(args).items()}, "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "ece_bins": ECE_BINS},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": stats.__name__ and __import__("scipy").__version__},
        "runs": len(runs), "inputs": dict(sorted(bpt.INPUTS.items())),
        "outputs": {str(Path(p).relative_to(out)).replace("\\", "/"): {"rows": n, "sha256": bpt.sha256(Path(p))} for p, n in sorted(written.items())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"runs": len(runs), "outputs": len(written)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS,
                        help="folder holding runs/, final_runs/final_01/ and retizero_runs/retizero_01/ (or the evidence folder)")
    parser.add_argument("--messidor-grades", type=Path, default=None, help="Krause 2018 adjudicated grades CSV (messidor_data.csv)")
    parser.add_argument("--messidor-pairs", type=Path, default=None, help="Messidor-2 eye pairing CSV (pairs.csv)")
    parser.add_argument("--messidor-duplicates", type=Path, default=None,
                        help="messidor_duplicate_pairs.csv from build_image_audits.py (merges patient IDs of identical photos)")
    parser.add_argument("--original-split", type=Path, default=None, help="folder with the original train/val/test CSVs")
    parser.add_argument("--near-duplicates", type=Path, default=None, help="near_duplicate_test_ids.csv from build_image_audits.py")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if not str(args.out).lower().startswith(str(HERE / "output").lower()):
        raise AnalysisError(f"Output must stay inside {HERE / 'output'}")
    try:
        result = build(args)
    except (AnalysisError, bpt.BuildError) as exc:
        print(f"ANALYSIS FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {result['runs']} runs, {result['outputs']} tables -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
