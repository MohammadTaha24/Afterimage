"""Build Afterimage_results.xlsx from the evidence folder.

Reads only files under the evidence folder (built by build_evidence_folder.py):
  D:\\Afterimage_Experiments\\deliverables_2026-11-17\\evidence
and writes
  D:\\Afterimage_Experiments\\deliverables_2026-11-17\\Afterimage_results.xlsx

Metrics come from the saved tables in finetuning/paper_tables/output (copied into the evidence
folder) and from the RetiZero queue summary. Every run is also recomputed from its saved
prediction file (accuracy, QWK, macro F1, macro one-vs-rest AUC, confusion matrix) and the two
are compared on the Checks sheet. Macro F1 is not in the saved tables for most runs, so it is
taken from the recomputation and compared with the metric file where one records it.

A quantity that no source records is written as "not recorded". "n/a" means the quantity does
not apply to that row (for example training time of a zero-shot model).

Usage:
  python -X utf8 -B finetuning\\paper_tables\\build_results_workbook.py [--evidence PATH] [--output PATH]
"""
import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DEFAULT_EVIDENCE = Path(r"D:\Afterimage_Experiments\deliverables_2026-11-17\evidence")
KRAUSE = "Krause et al. 2018, adjudicated Messidor-2 grades (Kaggle google-brain/messidor2-dr-grades)"
NR = "not recorded"
NA = "n/a"
TOL = 1e-8
GRADES = [0, 1, 2, 3, 4]
LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

T_CLEAN = "APTOS clean test (361)"
T_ORIG = "APTOS original test (367)"
T_MESS = "Messidor archived (1,057)"
S_CLEAN = "clean split (2,786 train / 357 val / 361 test)"
S_ORIG = "original split (2,929 train / 366 val / 367 test)"

EV = DEFAULT_EVIDENCE
PT = None  # paper_tables output folder inside the evidence folder
CHECKS = []


def rel(p):
    return Path(p).resolve().relative_to(EV.resolve()).as_posix()


def need(p):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(f"missing evidence file: {p}")
    return p


def check(area, what, got, expected, source, tol=TOL, note=""):
    """Record one cross-check. got / expected may be numbers, strings or bools."""
    ok = False
    try:
        if isinstance(got, (int, float, np.integer, np.floating)) and isinstance(expected, (int, float, np.integer, np.floating)):
            ok = (math.isnan(got) and math.isnan(expected)) or abs(float(got) - float(expected)) <= tol
        else:
            ok = got == expected
    except Exception:
        ok = False
    CHECKS.append({"area": area, "check": what, "workbook_or_recomputed_value": got,
                   "source_value": expected, "tolerance": tol if isinstance(expected, (int, float, np.floating, np.integer)) else "exact",
                   "result": "pass" if ok else "FAIL", "source": source, "note": note})
    return ok


# ---------------------------------------------------------------- metrics from predictions
def read_pred(path):
    df = pd.read_csv(need(path))
    return df


def confusion(y, p):
    cm = np.zeros((5, 5), dtype=int)
    for a, b in zip(y, p):
        cm[int(a), int(b)] += 1
    return cm


def qwk_from_cm(cm):
    n = cm.sum()
    w = np.array([[(i - j) ** 2 / 16.0 for j in range(5)] for i in range(5)])
    exp = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    den = (w * exp).sum()
    if den == 0:
        return float("nan")
    return 1.0 - (w * cm).sum() / (w * exp).sum()


def macro_f1_from_cm(cm):
    f1 = []
    for k in range(5):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        d = 2 * tp + fp + fn
        f1.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f1))


def auc_ovr(y, P):
    out = []
    y = np.asarray(y)
    for k in range(5):
        pos = y == k
        npos, nneg = pos.sum(), (~pos).sum()
        if npos == 0 or nneg == 0:
            out.append(float("nan"))
            continue
        r = pd.Series(P[:, k]).rank(method="average").to_numpy()
        out.append((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(out))


def recompute(df):
    y = df["y_true"].to_numpy()
    p = df["y_pred"].to_numpy()
    cm = confusion(y, p)
    res = {"n": len(df), "accuracy": float((y == p).mean()), "qwk": qwk_from_cm(cm), "macro_f1": macro_f1_from_cm(cm), "cm": cm}
    if all(f"p{k}" in df.columns for k in range(5)):
        res["macro_auc"] = auc_ovr(y, df[[f"p{k}" for k in range(5)]].to_numpy())
    else:
        res["macro_auc"] = float("nan")
    res["pred_counts"] = [int((p == k).sum()) for k in range(5)]
    return res


def cmp_run(area, name, rc, row, source, f1_recorded=None):
    """Compare recomputed metrics with the table row values (dict with accuracy, qwk, macro_auc, n)."""
    diffs = [abs(rc["accuracy"] - row["accuracy"]), abs(rc["qwk"] - row["qwk"])]
    if not math.isnan(rc["macro_auc"]):
        diffs.append(abs(rc["macro_auc"] - row["macro_auc"]))
    if f1_recorded is not None:
        diffs.append(abs(rc["macro_f1"] - f1_recorded))
    ok = (max(diffs) <= TOL) and rc["n"] == int(row["n"])
    CHECKS.append({"area": area, "check": f"{name}: accuracy, QWK, macro AUC{', macro F1' if f1_recorded is not None else ''} and n recomputed from the prediction file equal the saved values",
                   "workbook_or_recomputed_value": f"max abs difference {max(diffs):.2e}; n {rc['n']}",
                   "source_value": f"n {int(row['n'])}", "tolerance": TOL, "result": "pass" if ok else "FAIL", "source": source, "note": ""})


# ---------------------------------------------------------------- stats helpers
def msd(values):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def blank(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else x


# ---------------------------------------------------------------- Qwen rows
def build_qwen():
    tab_abc = pd.read_csv(need(PT / "aptos_361_clean" / "per_run_metrics.csv"))
    tab_d = pd.read_csv(need(PT / "aptos_367_original_split" / "per_run_metrics.csv"))
    tab_e = pd.read_csv(need(PT / "retraining2_replay" / "per_run_metrics.csv"))
    tab_f = pd.read_csv(need(PT / "final_01" / "per_job_metrics.csv"))
    eff = pd.read_csv(need(PT / "efficiency" / "efficiency_training_runs.csv"))
    eff_key = {(r["source"], r["run"]): r for _, r in eff.iterrows()}
    queue = {q["job"]: q for q in json.loads(need(EV / "final_runs" / "final_01" / "queue_state.json").read_text(encoding="utf-8"))}
    rows = []

    def status_text(base, degenerate, never, extra=""):
        s = base
        if degenerate:
            s += "; collapsed: predicts a single grade for every test image"
        elif isinstance(never, str) and never:
            s += "; grade(s) never predicted: " + ", ".join(never.split(";"))
        if extra:
            s += "; " + extra
        return s

    def ci(r):
        return [r["accuracy_ci_low"], r["accuracy_ci_high"], r["qwk_ci_low"], r["qwk_ci_high"], r["macro_auc_ci_low"], r["macro_auc_ci_high"]]

    # sets A, B, C: runs/
    for _, r in tab_abc.iterrows():
        run = r["run"]
        folder = EV / "runs" / run
        cfg = json.loads(need(folder / "pilot_configuration.json").read_text(encoding="utf-8"))
        latest = json.loads(need(folder / "latest_attempt.json").read_text(encoding="utf-8"))
        e = eff_key[("runs", run)]
        pred = folder / "aptos_test.predictions.csv"
        rc = recompute(read_pred(pred))
        cmp_run("Qwen per-run", run, rc, r, rel(pred))
        stage = {"A": "clean suite", "B": "previous-split training, scored on clean test", "C": "BF16 base control"}[r["result_set"]]
        base = "BF16 (16 layers CPU-offloaded)" if r["base"] == "BF16" else "4-bit"
        extra = []
        if int(e["attempts"]) > 1:
            extra.append(f"{int(e['attempts'])} attempts recorded; wall time sums all training attempts")
        if r["result_set"] == "B":
            extra.append("trained on the original split, so 16 of the 361 test images have exact copies in its training or validation data")
        rows.append(dict(
            run_id=run, condition=r["condition"], stage=stage, rank=int(r["rank"]), alpha=int(r["alpha"]), lr=cfg["training"]["lr"],
            clahe=r["clahe"], base_precision=base, trainer_seed=int(r["trainer_seed"]), lora_seed=int(r["lora_seed"]),
            train_split=S_ORIG if r["split"] == "previous_split" else S_CLEAN, test_set=T_CLEAN, n_test=int(r["n"]),
            accuracy=r["accuracy"], qwk=r["qwk"], macro_f1=rc["macro_f1"], macro_auc=r["macro_auc"],
            **dict(zip(["acc_ci_low", "acc_ci_high", "qwk_ci_low", "qwk_ci_high", "auc_ci_low", "auc_ci_high"], ci(r))),
            scoring_precision=r["scoring_precision"], status=status_text("completed" if latest["status"] == "passed" else latest["status"],
                                                                       bool(r["degenerate"]), r["grades_never_predicted"] if isinstance(r["grades_never_predicted"], str) else "", "; ".join(extra)),
            collapsed=bool(r["degenerate"]), missing_grades=r["grades_never_predicted"] if isinstance(r["grades_never_predicted"], str) else "",
            best_epoch=int(r["best_epoch"]), epochs_run=int(e["epochs_run"]), attempts=int(e["attempts"]), train_wall_s=float(e["train_wall_all_attempts_s"]),
            source=rel(pred), source_table=rel(PT / "aptos_361_clean" / "per_run_metrics.csv"), result_set=r["result_set"],
            pred_path=pred, cm=rc["cm"], pred_counts=rc["pred_counts"]))

    # set D: original split repeats (repo artifacts)
    base_d = EV / "repo" / "finetuning" / "Qwen3" / "artifacts" / "paper_reproduction" / "training_batches" / "run02_reproduction_01"
    for _, r in tab_d.iterrows():
        run = r["run"]
        folder = base_d / run
        cfg = json.loads(need(folder / "configuration.json").read_text(encoding="utf-8"))
        pred = folder / "evaluation_bf16" / "aptos_test.predictions.csv"
        summ = json.loads(need(folder / "evaluation_bf16" / "aptos_test.summary.json").read_text(encoding="utf-8"))
        rc = recompute(read_pred(pred))
        cmp_run("Qwen per-run", f"original-split repeat {run}", rc, r, rel(pred), f1_recorded=summ["macro_f1"])
        reports = sorted((folder / "attempts").glob("train-*/run_report.json"))
        wall = sum(json.loads(need(p).read_text(encoding="utf-8"))["elapsed_seconds"] for p in reports)
        info = json.loads(need(folder / "run_info.json").read_text(encoding="utf-8"))
        rows.append(dict(
            run_id="original_split/" + run, condition=r["condition"], stage="original split, same-seed repeat", rank=int(r["rank"]), alpha=int(r["alpha"]),
            lr=cfg["training"]["lr"], clahe=r["clahe"], base_precision="4-bit", trainer_seed=int(r["trainer_seed"]), lora_seed=int(r["lora_seed"]),
            train_split=S_ORIG, test_set=T_ORIG, n_test=int(r["n"]), accuracy=r["accuracy"], qwk=r["qwk"], macro_f1=rc["macro_f1"], macro_auc=r["macro_auc"],
            **dict(zip(["acc_ci_low", "acc_ci_high", "qwk_ci_low", "qwk_ci_high", "auc_ci_low", "auc_ci_high"], ci(r))),
            scoring_precision=r["scoring_precision"],
            status=status_text("completed", bool(r["degenerate"]), r["grades_never_predicted"] if isinstance(r["grades_never_predicted"], str) else "",
                               "scored in BF16; the original split has exact duplicates across train and test"),
            collapsed=bool(r["degenerate"]), missing_grades=r["grades_never_predicted"] if isinstance(r["grades_never_predicted"], str) else "",
            best_epoch=int(r["best_epoch"]), epochs_run=int(info["epochs_completed"]), attempts=len(reports), train_wall_s=float(wall),
            source=rel(pred), source_table=rel(PT / "aptos_367_original_split" / "per_run_metrics.csv"), result_set="D", pred_path=pred, cm=rc["cm"], pred_counts=rc["pred_counts"]))

    # set E: replay of the archived Retraining 2 adapter
    r = tab_e.iloc[0]
    pred = EV / "repo" / "finetuning" / "Qwen3" / "artifacts" / "paper_reproduction" / "training_batches" / "run02_reproduction_01" / "replay_run_02" / "evaluation" / "aptos_test.predictions.csv"
    summ = json.loads(need(pred.parent / "aptos_test.summary.json").read_text(encoding="utf-8"))
    rc = recompute(read_pred(pred))
    cmp_run("Qwen per-run", "Retraining 2 adapter replay", rc, r, rel(pred), f1_recorded=summ["macro_f1"])
    rows.append(dict(
        run_id="retraining2_replay", condition=r["condition"], stage="paper adapter replay (Retraining 2)", rank=16, alpha=16, lr=0.0005,
        clahe="on (archived inputs)", base_precision="4-bit", trainer_seed=42, lora_seed=3407, train_split=S_ORIG, test_set=T_ORIG, n_test=int(r["n"]),
        accuracy=r["accuracy"], qwk=r["qwk"], macro_f1=rc["macro_f1"], macro_auc=r["macro_auc"],
        **dict(zip(["acc_ci_low", "acc_ci_high", "qwk_ci_low", "qwk_ci_high", "auc_ci_low", "auc_ci_high"], ci(r))),
        scoring_precision=r["scoring_precision"], status="completed; archived adapter (run_02 of three same-seed runs) rescored, not retrained",
        collapsed=False, missing_grades="", best_epoch=int(r["best_epoch"]), epochs_run=NR, attempts=NA, train_wall_s=NR,
        source=rel(pred), source_table=rel(PT / "retraining2_replay" / "per_run_metrics.csv"), result_set="E", pred_path=pred, cm=rc["cm"], pred_counts=rc["pred_counts"]))

    # set F: final_01
    fr = tab_f.copy()
    sel = fr[(fr["stage"].isin(["zero_shot", "original_adapter_audit"])) |
             ((fr["kind"] == "train")) |
             ((fr["fallback_for"].notna()) & (fr["cohort"] == "aptos361_on"))]
    for _, r in sel.iterrows():
        job = r["job"]
        is_fp32 = isinstance(r["fallback_for"], str) and r["fallback_for"] != ""
        if r["kind"] == "train":
            folder = EV / "final_runs" / "final_01" / "train" / job
            pred = folder / "aptos_test.predictions.csv"
            sr = json.loads(need(folder / "score_result.json").read_text(encoding="utf-8"))["metrics"]
            f1_rec = None
            e = eff_key[("final_01/train", job)]
            latest = json.loads(need(folder / "latest_attempt.json").read_text(encoding="utf-8"))
            run_id = job
            wall, epochs, attempts = float(e["train_wall_all_attempts_s"]), int(e["epochs_run"]), int(e["attempts"])
            status = "completed" if latest["status"] == "passed" else latest["status"]
            stage = {"lr_study": "learning-rate study", "alpha_study": "alpha study"}[r["stage"]]
        elif is_fp32:
            folder = EV / "final_runs" / "final_01" / "scores" / job
            pred = folder / "predictions.csv"
            summ = json.loads(need(folder / "summary.json").read_text(encoding="utf-8"))
            f1_rec = summ["metrics"]["macro_f1"]
            run_id = r["fallback_for"]
            e = eff_key[("final_01/train", run_id)]
            wall, epochs, attempts = float(e["train_wall_all_attempts_s"]), int(e["epochs_run"]), int(e["attempts"])
            status = "training completed; FP16 scoring stopped with a non-finite likelihood (FP16 overflow); scored with the labelled FP32 fallback"
            stage = "alpha study (FP32 fallback score)"
        else:
            folder = EV / "final_runs" / "final_01" / "scores" / job
            pred = folder / "predictions.csv"
            summ = json.loads(need(folder / "summary.json").read_text(encoding="utf-8"))
            f1_rec = summ["metrics"]["macro_f1"]
            run_id = job
            wall, epochs, attempts = NA, NA, NA
            status = "completed"
            stage = {"zero_shot": "zero-shot (no adapter)", "original_adapter_audit": "original paper adapter audit"}[r["stage"]]
        rc = recompute(read_pred(pred))
        cmp_run("Qwen per-run", run_id + (" (FP32 fallback)" if is_fp32 else ""), rc, r, rel(pred), f1_recorded=f1_rec)
        zero = r["stage"] == "zero_shot"
        orig = r["stage"] == "original_adapter_audit"
        never = r["grades_never_predicted"] if isinstance(r["grades_never_predicted"], str) else ""
        extra = ""
        if r["kind"] == "train" and isinstance(r["scaling_note"], str):
            extra = "recorded effective_scaling in train_result.json is 16/rank; true scaling alpha/r is " + str(r["effective_scaling"])
        if orig:
            extra = "legacy adapter outside the experiment folders; no training record is read"
        rows.append(dict(
            run_id=run_id, condition=r["condition"], stage=stage,
            rank=NA if zero else (NR if orig else int(r["rank"])), alpha=NA if zero else (NR if orig else int(r["alpha"])),
            lr=NA if zero else (NR if orig else float(r["lr"])),
            clahe=(r["cohort"].split("_")[-1] if zero else ("not recorded" if orig else r["clahe_trained"])),
            base_precision=("BF16 (16 layers CPU-offloaded)" if r["base"] == "BF16" else "4-bit"),
            trainer_seed=NA if zero else (NR if orig else int(r["trainer_seed"])), lora_seed=NA if zero else (NR if orig else int(r["lora_seed"])),
            train_split=NA if zero else (NR if orig else S_CLEAN), test_set=T_ORIG if orig else T_CLEAN, n_test=int(r["n"]),
            accuracy=r["accuracy"], qwk=r["qwk"], macro_f1=rc["macro_f1"], macro_auc=r["macro_auc"],
            **dict(zip(["acc_ci_low", "acc_ci_high", "qwk_ci_low", "qwk_ci_high", "auc_ci_low", "auc_ci_high"], ci(r))),
            scoring_precision=("FP32 fallback" if is_fp32 else r["scoring_precision"]),
            status=status_text(status, bool(r["degenerate"]), never, extra), collapsed=bool(r["degenerate"]), missing_grades=never,
            best_epoch=(NA if (zero or orig) else int(r["best_epoch"])), epochs_run=(NR if orig else epochs), attempts=attempts, train_wall_s=(NA if zero else (NR if orig else wall)),
            source=rel(pred), source_table=rel(PT / "final_01" / "per_job_metrics.csv"), result_set="F", pred_path=pred, cm=rc["cm"], pred_counts=rc["pred_counts"]))

        # cross-check against the job's own metric file
        if r["kind"] == "train":
            for key, col in (("accuracy", "accuracy"), ("qwk", "qwk"), ("macro_ovr_auc", "macro_auc")):
                check("Qwen per-run", f"{run_id}: {col} equals score_result.json", float(r[col]), float(sr[key]), rel(folder / "score_result.json"))
        else:
            for key, col in (("accuracy", "accuracy"), ("qwk", "qwk"), ("macro_ovr_auc", "macro_auc")):
                check("Qwen per-run", f"{run_id}{' (FP32 fallback)' if is_fp32 else ''}: {col} equals summary.json", float(r[col]), float(summ["metrics"][key]), rel(folder / "summary.json"))
    return pd.DataFrame(rows)


QCOLS = ["run_id", "condition", "stage", "rank", "alpha", "lr", "clahe", "base_precision", "trainer_seed", "lora_seed", "train_split", "test_set", "n_test",
         "accuracy", "acc_ci_low", "acc_ci_high", "qwk", "qwk_ci_low", "qwk_ci_high", "macro_f1", "macro_auc", "auc_ci_low", "auc_ci_high",
         "scoring_precision", "status", "best_epoch", "epochs_run", "attempts", "train_wall_s", "source", "source_table"]


def condition_groups(q):
    """(group label, mask, published table, published row selector) for Qwen_summary."""
    g = []
    A = q["result_set"] == "A"

    def add(label, mask, note="", test_note=None):
        g.append((label, mask, note))

    zs = q["stage"] == "zero-shot (no adapter)"
    add("zero-shot 4-bit, CLAHE on", zs & (q["base_precision"] == "4-bit") & (q["clahe"] == "on"))
    add("zero-shot 4-bit, CLAHE off", zs & (q["base_precision"] == "4-bit") & (q["clahe"] == "off"))
    add("zero-shot BF16 (16 layers CPU-offloaded), CLAHE on", zs & (q["base_precision"] != "4-bit"))
    r16on = A & (q["rank"] == 16) & (q["clahe"] == "on")
    add("lr 5e-4, rank 16, alpha 16, CLAHE on (main setting)", r16on, "includes collapsed or partly collapsed seeds")
    add("lr 2e-4, rank 16, alpha 16, CLAHE on", q["stage"].eq("learning-rate study") & (q["lr"] == 0.0002))
    add("lr 1e-4, rank 16, alpha 16, CLAHE on", q["stage"].eq("learning-rate study") & (q["lr"] == 0.0001))
    for rk in (8, 16, 32):
        for cl in ("on", "off"):
            add(f"rank {rk}, alpha 16, lr 5e-4, CLAHE {cl}", A & (q["rank"] == rk) & (q["clahe"] == cl))
    alpha_f = q["stage"].str.startswith("alpha study")
    add("alpha 8, rank 16, lr 5e-4, CLAHE on", alpha_f & (q["alpha"] == 8))
    add("alpha 16, rank 16, lr 5e-4, CLAHE on (same runs as the main setting)", r16on)
    add("alpha 32, rank 16, lr 5e-4, CLAHE on, FP16-scored seeds only", alpha_f & (q["alpha"] == 32) & (q["scoring_precision"] == "FP16"),
        "the FP32-fallback seed is excluded")
    add("alpha 32, rank 16, lr 5e-4, CLAHE on, FP16 seeds plus the FP32-fallback seed", alpha_f & (q["alpha"] == 32), "mixed scoring precision")
    add("rank 16, lr 5e-4, CLAHE on, trained on the original split, scored on the 361 test", q["result_set"] == "B", "16 test images have exact copies in training or validation data")
    add("BF16 base (16 layers CPU-offloaded), rank 16, lr 5e-4, CLAHE on", q["result_set"] == "C")
    D = q["result_set"] == "D"
    add("original split, rank 16, CLAHE on, three same-seed repeats", D & (q["clahe"] == "on"))
    add("original split, rank 16, CLAHE off, three same-seed repeats", D & (q["clahe"] == "off"))
    add("paper adapter replay (Retraining 2, archived run_02)", q["result_set"] == "E", "single saved adapter rescored")
    add("original paper adapter audit (legacy best_lora)", q["stage"].eq("original paper adapter audit"), "single adapter; training record not read")
    return g


def build_qwen_summary(q):
    # published tables used for cross-checks
    pub = {}
    abc = pd.read_csv(PT / "aptos_361_clean" / "condition_summary.csv")
    d = pd.read_csv(PT / "aptos_367_original_split" / "condition_summary.csv")
    e = pd.read_csv(PT / "retraining2_replay" / "condition_summary.csv")
    lr = pd.read_csv(PT / "final_01" / "lr_study_summary.csv")
    al = pd.read_csv(PT / "final_01" / "alpha_study_summary.csv")
    zs = pd.read_csv(PT / "final_01" / "zero_shot.csv")
    oa = pd.read_csv(PT / "final_01" / "original_adapter_audit.csv")
    rows = []
    for label, mask, note in condition_groups(q):
        sub = q[mask]
        if len(sub) == 0:
            raise RuntimeError(f"empty Qwen group: {label}")
        am, asd = msd(sub["accuracy"])
        qm, qsd = msd(sub["qwk"])
        fm, fsd = msd(sub["macro_f1"])
        um, usd = msd(sub["macro_auc"])
        tests = sorted(sub["test_set"].unique())
        precisions = sorted(sub["scoring_precision"].unique())
        # published reference row
        ref = None
        src_tab = rel(PT / "aptos_361_clean" / "condition_summary.csv")
        first = sub.iloc[0]
        if first["result_set"] in ("A", "B", "C") and not label.startswith("alpha") and not label.startswith("lr 5e-4"):
            ref = abc[abc["condition"] == first["condition"]]
        elif label.startswith("lr 5e-4"):
            ref = abc[abc["condition"] == "A r16 CLAHE-on 4-bit"]
        elif first["result_set"] == "D":
            ref = d[d["condition"] == first["condition"]]
            src_tab = rel(PT / "aptos_367_original_split" / "condition_summary.csv")
        elif first["result_set"] == "E":
            ref = e
            src_tab = rel(PT / "retraining2_replay" / "condition_summary.csv")
        elif label.startswith("lr 2e-4") or label.startswith("lr 1e-4"):
            v = 0.0002 if label.startswith("lr 2e-4") else 0.0001
            ref = lr[np.isclose(lr["lr"], v)]
            src_tab = rel(PT / "final_01" / "lr_study_summary.csv")
        elif label.startswith("alpha 8"):
            ref = al[(al["alpha"] == 8)]
            src_tab = rel(PT / "final_01" / "alpha_study_summary.csv")
        elif label.startswith("alpha 16"):
            ref = al[(al["alpha"] == 16)]
            src_tab = rel(PT / "final_01" / "alpha_study_summary.csv")
        elif label.startswith("alpha 32, rank 16, lr 5e-4, CLAHE on, FP16-scored"):
            ref = al[(al["alpha"] == 32) & (al["n_runs"] == 2)]
            src_tab = rel(PT / "final_01" / "alpha_study_summary.csv")
        elif label.startswith("alpha 32"):
            ref = al[(al["alpha"] == 32) & (al["n_runs"] == 3)]
            src_tab = rel(PT / "final_01" / "alpha_study_summary.csv")
        elif label.startswith("zero-shot"):
            ref = zs[zs["job"] == sub.iloc[0]["run_id"]]
            src_tab = rel(PT / "final_01" / "zero_shot.csv")
        elif label.startswith("original paper adapter"):
            ref = oa
            src_tab = rel(PT / "final_01" / "original_adapter_audit.csv")
        if ref is not None and len(ref) == 1:
            ref = ref.iloc[0]
            if "accuracy_mean" in ref:
                check("Qwen summary", f"{label}: mean accuracy", am, ref["accuracy_mean"], src_tab)
                check("Qwen summary", f"{label}: mean QWK", qm, ref["qwk_mean"], src_tab)
                check("Qwen summary", f"{label}: mean macro AUC", um, ref["macro_auc_mean"], src_tab)
                if len(sub) > 1:
                    check("Qwen summary", f"{label}: SD of accuracy", asd, ref["accuracy_sd"], src_tab)
                    check("Qwen summary", f"{label}: SD of QWK", qsd, ref["qwk_sd"], src_tab)
            else:
                check("Qwen summary", f"{label}: accuracy", am, ref["accuracy"], src_tab)
                check("Qwen summary", f"{label}: QWK", qm, ref["qwk"], src_tab)
                check("Qwen summary", f"{label}: macro AUC", um, ref["macro_auc"], src_tab)
        else:
            check("Qwen summary", f"{label}: a single published reference row exists", 0 if ref is None else len(ref), 1, src_tab)
        rows.append(dict(
            condition=label, test_set="; ".join(tests), n_test=int(sub["n_test"].iloc[0]), n_runs=len(sub),
            accuracy_mean=am, accuracy_sd=asd, accuracy_min=float(sub["accuracy"].min()), accuracy_max=float(sub["accuracy"].max()),
            qwk_mean=qm, qwk_sd=qsd, macro_f1_mean=fm, macro_f1_sd=fsd, macro_auc_mean=um, macro_auc_sd=usd,
            n_collapsed=int(sub["collapsed"].sum()), n_missing_a_grade=int((sub["missing_grades"] != "").sum()),
            scoring_precision="; ".join(precisions), note=note + ("; " if note else "") + ("SD is the sample SD across runs" if len(sub) > 1 else "single run, no SD"),
            source=f"{src_tab}; per-run rows in sheet Qwen_runs", member_runs=" | ".join(sub["run_id"])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- RetiZero
def build_retizero():
    root = EV / "retizero_runs" / "retizero_01"
    s = pd.read_csv(need(root / "summary.csv"))
    energy = pd.read_csv(need(root / "energy_per_job.csv"))
    emap = {r["job"].split("/", 1)[1]: r for _, r in energy.iterrows() if r["job"].startswith("retizero_01/")}
    queue = {q["job"]: q for q in json.loads(need(root / "queue_state.json").read_text(encoding="utf-8"))}
    rows = []
    cohort_name = {"aptos367_original": T_ORIG, "aptos361_clean": T_CLEAN, "aptos361_clean_clahe": T_CLEAN, "messidor_archived": T_MESS}
    for _, r in s.iterrows():
        job = r["job"]
        stage = r["stage"]
        kind = r["kind"]
        if kind == "train":
            folder = root / "train" / job
            pred = folder / "evaluation" / "aptos_test.predictions.csv"
            msum = folder / "evaluation" / "aptos_test.summary.json"
            tr = json.loads(need(folder / "train_result.json").read_text(encoding="utf-8"))
            reports = sorted((folder / "attempts").glob("train-*/run_report.json"))
            wall_s = sum(json.loads(need(p).read_text(encoding="utf-8"))["elapsed_seconds"] for p in reports)
            check("RetiZero per-run", f"{job}: training wall time in summary.csv equals run_report.json", float(r["wall_minutes"]), wall_s / 60.0, rel(reports[0]), tol=1e-6)
            skip_nf = tr["skipped_steps_nonfinite_loss"]
            skip_ov = tr["skipped_steps_scaler_overflow"]
            check("RetiZero per-run", f"{job}: skipped steps summed over epochs equals the train_result.json totals",
                  sum(h["skipped_steps_nonfinite_loss"] for h in tr["history"]), skip_nf, rel(folder / "train_result.json"), tol=0)
            train_split = S_ORIG if r["split"] == "previous_split" else S_CLEAN
            clahe = r["clahe"]
            seed = int(r["seed"])
            ckpt = tr["checkpoint"]
        else:
            folder = root / "scores" / job
            pred = folder / "predictions.csv"
            msum = folder / "summary.json"
            skip_nf = skip_ov = NA
            train_split = S_ORIG if job.startswith("saved_checkpoint") else S_CLEAN
            clahe = NR if job.startswith("saved_checkpoint") else "off"
            m = re.search(r"_t(\d+)", job)
            seed = int(m.group(1)) if m else NA
            ckpt = r["checkpoint"]
            reports = sorted((folder / "attempts").glob("*/run_report.json")) if (folder / "attempts").exists() else []
            wall_s = float(r["wall_minutes"]) * 60.0
        sm = json.loads(need(msum).read_text(encoding="utf-8"))
        met = sm["metrics"] if "metrics" in sm else sm
        rc = recompute(read_pred(pred))
        cmp_run("RetiZero per-run", job, rc, dict(accuracy=r["accuracy"], qwk=r["qwk"], macro_auc=r["macro_auc"], n=r["images"]), rel(pred), f1_recorded=r["macro_f1"])
        for key, col in (("accuracy", "accuracy"), ("qwk", "qwk"), ("macro_ovr_auc", "macro_auc"), ("macro_f1", "macro_f1")):
            check("RetiZero per-run", f"{job}: {col} in summary.csv equals {rel(msum).split('/')[-1]}", float(r[col]), float(met[key]), rel(msum))
        saved = np.array(met["confusion_matrix"])
        check("RetiZero per-run", f"{job}: confusion matrix recomputed from predictions equals the saved matrix",
              bool((saved == rc["cm"]).all()), True, rel(msum), tol=0)
        e = emap.get(job)
        q = queue.get(job, {})
        no_energy = NR
        rows.append(dict(
            job=job, stage=stage, kind=kind, train_split=train_split, test_set=cohort_name[r["cohort"]], n_test=int(r["images"]),
            clahe_input=clahe, seed=seed, initial_weights=(r["initial_weights"] if kind == "train" else NA),
            effective_batch=(int(r["effective_batch"]) if kind == "train" else NA), physical_batch=(int(r["physical_batch"]) if kind == "train" else NA),
            lr=(float(r["learning_rate"]) if kind == "train" else NA), epochs_run=(int(r["epochs_run"]) if kind == "train" else NA),
            best_epoch=(int(r["best_epoch"]) if kind == "train" else NA), best_val_acc=(float(r["best_validation_accuracy"]) if kind == "train" else NA),
            accuracy=r["accuracy"], accuracy_ci=NR, qwk=r["qwk"], macro_f1=r["macro_f1"], macro_auc=r["macro_auc"],
            predicted_per_grade=r["predicted_per_grade"], constant_predictor=bool(r["constant_predictor"]),
            skipped_steps_nonfinite_loss=skip_nf, skipped_steps_scaler_overflow=skip_ov,
            stage_wall_s=wall_s, energy_wh=(float(e["energy_wh"]) if e is not None else no_energy), mean_power_w=(float(e["mean_w"]) if e is not None else no_energy),
            median_ms_per_image=(float(r["median_ms_per_image"]) if kind == "score" else NA), images_per_s_batched=(float(r["images_per_second_batched"]) if kind == "score" else NA),
            model_load_s=(float(r["model_load_seconds"]) if kind == "score" else NA),
            train_images_per_s=(float(r["train_images_per_second"]) if kind == "train" else NA),
            torch_peak_gib=(float(r["torch_peak_gib_training"]) if kind == "train" else float(r["torch_peak_gib_batch1"])),
            card_peak_vram_mib=float(r["peak_vram_mib"]), mean_gpu_pct=float(r["mean_gpu_percent"]) / 100.0,
            status=("completed (queue outcome: " + q.get("outcome", "not recorded") + ")"),
            checkpoint=ckpt, source=rel(pred), source_table=rel(root / "summary.csv"),
            cm=rc["cm"], pred_path=pred))
    df = pd.DataFrame(rows)
    # energy cross-check against the integrated power log
    plog = pd.read_csv(need(root / "power_log.csv"), skipinitialspace=True)
    plog.columns = [c.strip() for c in plog.columns]
    plog["t"] = pd.to_datetime(plog["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    plog["w"] = plog["power.draw [W]"].str.replace(" W", "", regex=False).astype(float)
    check("RetiZero energy", "power_log.csv has 5 s samples (median gap, s)", round(float(plog["t"].diff().dt.total_seconds().median())), 5, rel(root / "power_log.csv"), tol=0)
    for _, e in energy.iterrows():
        if not e["job"].startswith("retizero_01/"):
            continue
        expected = e["mean_w"] * e["wall_s"] / 3600.0
        check("RetiZero energy", f"{e['job'].split('/', 1)[1]}: energy_wh equals mean_w x wall_s / 3600 (rounding tolerance 0.01 Wh)", float(e["energy_wh"]), float(expected),
              rel(root / "energy_per_job.csv"), tol=0.01)
        check("RetiZero energy", f"{e['job'].split('/', 1)[1]}: samples cover the whole job window", bool(e["fully_covered"]), True, rel(root / "energy_per_job.csv"), tol=0)
    return df


RCOLS = ["job", "stage", "kind", "train_split", "test_set", "n_test", "clahe_input", "seed", "initial_weights", "effective_batch", "physical_batch", "lr",
         "epochs_run", "best_epoch", "best_val_acc", "accuracy", "accuracy_ci", "qwk", "macro_f1", "macro_auc", "predicted_per_grade", "constant_predictor",
         "skipped_steps_nonfinite_loss", "skipped_steps_scaler_overflow", "stage_wall_s", "energy_wh", "mean_power_w", "median_ms_per_image", "images_per_s_batched",
         "model_load_s", "train_images_per_s", "torch_peak_gib", "card_peak_vram_mib", "mean_gpu_pct", "status", "checkpoint", "source", "source_table"]


def build_retizero_summary(rz):
    rows = []

    def grp(label, mask, note, test=None):
        sub = rz[mask]
        if len(sub) == 0:
            raise RuntimeError("empty RetiZero group " + label)
        am, asd = msd(sub["accuracy"])
        qm, qsd = msd(sub["qwk"])
        fm, fsd = msd(sub["macro_f1"])
        um, usd = msd(sub["macro_auc"])
        wall = [x for x in sub["stage_wall_s"]]
        wm, wsd = msd(wall)
        en = [x for x in sub["energy_wh"] if not isinstance(x, str)]
        em, esd = msd(en)
        ms = [x for x in sub["median_ms_per_image"] if not isinstance(x, str)]
        mm, msd_ = msd(ms)
        skip = [x for x in sub["skipped_steps_nonfinite_loss"] if not isinstance(x, str)]
        rows.append(dict(
            group=label, test_set="; ".join(sorted(sub["test_set"].unique())), n_test=int(sub["n_test"].iloc[0]), n_jobs=len(sub),
            accuracy_mean=am, accuracy_sd=asd, accuracy_min=float(sub["accuracy"].min()), accuracy_max=float(sub["accuracy"].max()),
            qwk_mean=qm, qwk_sd=qsd, macro_f1_mean=fm, macro_f1_sd=fsd, macro_auc_mean=um, macro_auc_sd=usd,
            skipped_steps_nonfinite_total=(sum(skip) if skip else NA),
            skipped_steps_scaler_overflow_total=(sum(x for x in sub["skipped_steps_scaler_overflow"] if not isinstance(x, str)) if skip else NA),
            stage_wall_s_mean=wm, stage_wall_s_sd=wsd, energy_wh_mean=(em if en else NR), energy_wh_sd=(esd if len(en) > 1 else (NA if not en else "")),
            median_ms_per_image_mean=(mm if ms else NA), median_ms_per_image_sd=(msd_ if len(ms) > 1 else NA),
            note=note + ("; " if note else "") + ("SD is the sample SD across jobs" if len(sub) > 1 else "single job, no SD"),
            source="; ".join(sorted(set(sub["source_table"]))) + "; per-job rows in sheet RetiZero_runs", member_jobs=" | ".join(sub["job"])))
        return sub

    grp("replay: saved checkpoint, original-split test (367)", rz["job"] == "saved_checkpoint__aptos367_original", "app checkpoint trained on the original split; no training record in this queue; no energy figure (ran before power logging)")
    grp("replay: saved checkpoint, clean test (361)", rz["job"] == "saved_checkpoint__aptos361_clean", "trained on the original split, so 16 of the 361 test images have exact copies in its training or validation data; no energy figure")
    grp("historical: trained on original split, tested on 367 (three seeds)", rz["stage"] == "historical", "official pretrained start; selection on validation accuracy, patience 4")
    grp("matched: trained on clean split, CLAHE off, tested on 361 (three seeds)", rz["stage"] == "matched", "224 px; the main RetiZero result")
    grp("CLAHE: trained on clean split, CLAHE on at 224 px, tested on 361 (three seeds)", rz["stage"] == "clahe", "CLAHE grid gives 28 px tiles at 224 px (64 px tiles in the Qwen 512 px build)")
    grp("timing: benchmark rescoring of the matched checkpoints on 361", rz["stage"] == "benchmark", "batch size 1 for per-image time; energy covers the whole job window including model load")
    grp("Messidor: saved checkpoint on archived 1,057", rz["job"] == "saved_checkpoint__messidor_archived", "archive labels, not adjudicated grades")
    grp("Messidor: matched checkpoints on archived 1,057 (three seeds)", (rz["stage"] == "messidor_archived") & rz["job"].str.startswith("matched"), "archive labels, not adjudicated grades; Qwen saw CLAHE-on 512 px, RetiZero CLAHE-off 224 px")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Data split
def build_split():
    ds = EV / "datasets" / "aptos_512"
    rows = []
    counts = {}
    for scheme, folder in (("original split", "previous_split"), ("clean split", "duplicate_excluded")):
        for part in ("train", "val", "test"):
            p = need(ds / folder / "clahe_off" / f"{part}.csv")
            df = pd.read_csv(p)
            cnt = [int((df["label"] == g).sum()) for g in GRADES]
            counts[(scheme, part)] = cnt
            rows.append(dict(section="images per split and grade", item=f"{scheme}, {part}", split_scheme=scheme, subset=part, n_images=len(df),
                             g0=cnt[0], g1=cnt[1], g2=cnt[2], g3=cnt[3], g4=cnt[4], value="", note="", source=rel(p)))
        tot = [sum(counts[(scheme, part)][i] for part in ("train", "val", "test")) for i in range(5)]
        rows.append(dict(section="images per split and grade", item=f"{scheme}, all three subsets", split_scheme=scheme, subset="total", n_images=sum(tot),
                         g0=tot[0], g1=tot[1], g2=tot[2], g3=tot[3], g4=tot[4], value="", note="labelled APTOS 2019 training images", source=rel(ds / folder / "clahe_off" / "train.csv") + "; val.csv; test.csv"))
    # split sizes and per-grade counts against the dataset manifests written when the splits were built
    for scheme, folder in (("original split", "previous_split"), ("clean split", "duplicate_excluded")):
        man_path = need(ds / folder / "manifest.json")
        man = json.loads(man_path.read_text(encoding="utf-8"))["branches"]["off"]
        for part in ("train", "val", "test"):
            cnt = counts[(scheme, part)]
            exp = [man[part]["rows"]] + [man[part]["grades"][str(g)] for g in GRADES]
            check("Data split", f"{scheme} {part}: n and per-grade counts equal the dataset manifest", [sum(cnt)] + cnt, exp, rel(man_path), tol=0)

    # exact-duplicate analysis from the SHA-256 of every APTOS image file (analysis/image_sha256.csv)
    ana = EV / "analysis"
    ih_src = rel(ana / "image_sha256.csv")
    so_src = rel(ana / "split_overlap.csv")
    pa_src = rel(ds / "previous_split" / "pairing_audit.csv")
    ex_src = rel(ds / "excluded.csv")
    ih = pd.read_csv(need(ana / "image_sha256.csv"), dtype={"image_id": str})
    pa = ih.rename(columns={"original_split": "split", "sha256_of_image_file": "original_sha256"})[["image_id", "split", "grade", "original_sha256"]]
    audit = pd.read_csv(need(ds / "previous_split" / "pairing_audit.csv"), usecols=["image_id", "split", "grade", "original_sha256"], dtype={"image_id": str})
    mg = pa.merge(audit, on="image_id", how="outer", suffixes=("", "_audit"), indicator=True)
    same_audit = bool((mg["_merge"] == "both").all() and (mg["original_sha256"] == mg["original_sha256_audit"]).all()
                      and (mg["split"] == mg["split_audit"]).all() and (mg["grade"] == mg["grade_audit"]).all())
    check("Data split", "image file SHA-256, split and grade of every labelled image equal the original-split pairing audit", same_audit, True, ih_src + "; " + pa_src, tol=0)
    ex = pd.read_csv(need(ds / "excluded.csv"), dtype={"image_id": str})
    clean_ids = set()
    for part in ("train", "val", "test"):
        clean_ids |= set(pd.read_csv(ds / "duplicate_excluded" / "clahe_off" / f"{part}.csv", dtype={"id_code": str})["id_code"])
    removed = pa[~pa["image_id"].isin(clean_ids)]
    check("Data split", "images missing from the clean split label CSVs are exactly the images in excluded.csv", set(removed["image_id"]) == set(ex["image_id"]), True, ex_src, tol=0)
    groups = pa.groupby("original_sha256")
    gsize = groups["image_id"].transform("count")
    dup = pa[gsize > 1].copy()
    ngroups = dup["original_sha256"].nunique()
    n_dup_images = len(dup)
    conflict_groups = int((dup.groupby("original_sha256")["grade"].nunique() > 1).sum())
    test = pa[pa["split"] == "test"]
    leak_ids = []
    leak_train = leak_val = conflict_test = conflict_pairs = 0
    for _, t in test.iterrows():
        others = pa[(pa["original_sha256"] == t["original_sha256"]) & (pa["image_id"] != t["image_id"]) & (pa["split"].isin(["train", "val"]))]
        if len(others):
            leak_ids.append(t["image_id"])
            if (others["split"] == "train").any():
                leak_train += 1
            else:
                leak_val += 1
            if (others["grade"] != t["grade"]).any():
                conflict_test += 1
            conflict_pairs += int((others["grade"] != t["grade"]).sum())
    tt = test.groupby("original_sha256")["image_id"].count()
    tt_groups = int((tt > 1).sum())
    ex_test = removed[removed["split"] == "test"]
    ex_conflict = 0
    ex_tt = 0
    for _, t in ex_test.iterrows():
        g = pa[pa["original_sha256"] == t["original_sha256"]]
        if g["grade"].nunique() > 1:
            ex_conflict += 1
        else:
            ex_tt += 1
    ct = pa[pa["image_id"].isin(set(pd.read_csv(ds / "duplicate_excluded" / "clahe_off" / "test.csv", dtype={"id_code": str})["id_code"]))]
    clean_vs_train = clean_vs_val = 0
    for _, t in ct.iterrows():
        o = pa[(pa["original_sha256"] == t["original_sha256"]) & (pa["image_id"] != t["image_id"])]
        if (o["split"] == "train").any():
            clean_vs_train += 1
        elif (o["split"] == "val").any():
            clean_vs_val += 1
    cnt_o = sum(counts[("original split", "test")])
    cnt_c = sum(counts[("clean split", "test")])
    n_rm_train = int((removed["split"] == "train").sum())
    n_rm_val = int((removed["split"] == "val").sum())
    facts = [
        ("labelled images in the APTOS 2019 training set", len(pa), "labelled_images", "labelled images, each hashed from its image file"),
        ("exact-duplicate groups (identical image files) in the labelled set", ngroups, "duplicate_groups", f"{ngroups} groups"),
        ("images in those groups", n_dup_images, "images_in_duplicate_groups", f"{n_dup_images} images"),
        ("groups whose copies carry different grades", conflict_groups, "duplicate_groups_with_conflicting_grades", f"{conflict_groups} groups"),
        ("original-split test images with an exact copy in train or validation", len(leak_ids), "original_test_with_copy_in_train_or_val", f"{len(leak_ids)} of {cnt_o}"),
        ("  of which a copy is in train", leak_train, "original_test_with_copy_in_train", ""),
        ("  of which the only copies are in validation", leak_val, "original_test_with_copy_only_in_val", ""),
        ("  of which a copy has a different grade", conflict_test, "original_test_with_copy_of_different_grade",
         f"{conflict_test} images ({conflict_pairs} differing pairs, because a test image can have more than one copy)"),
        ("exact-duplicate groups inside the original test set", tt_groups, "duplicate_groups_inside_original_test", f"{tt_groups} test-test pairs"),
        ("images removed from the original test set to make the clean test set", len(ex_test), "removed_from_original_test", f"{cnt_o} - {len(ex_test)} = {cnt_c}"),
        ("  removed because a copy has a different grade", ex_conflict, "removed_from_test_conflicting_grade", ""),
        ("  removed as one half of a test-test duplicate pair", ex_tt, "removed_from_test_as_duplicate_pair", ""),
        ("images removed in total", len(removed), "removed_total", f"{n_rm_train} train, {n_rm_val} validation, {len(ex_test)} test"),
        ("  removed from train", n_rm_train, "removed_from_train", ""),
        ("  removed from validation", n_rm_val, "removed_from_val", ""),
        ("clean test images with an exact copy in the original train split", clean_vs_train, "clean_test_with_copy_in_original_train", "applies to models trained on the original split and scored on the 361"),
        ("clean test images with an exact copy in the original validation split", clean_vs_val, "clean_test_with_copy_only_in_original_val", "no copy in the original train split"),
    ]
    sof_all = pd.read_csv(need(ana / "split_overlap.csv"))
    so = dict(zip(sof_all["fact"], sof_all["value"]))
    for item, got, key, note in facts:
        rows.append(dict(section="exact-duplicate overlap (SHA-256 of image files)", item=item, split_scheme="", subset="", n_images="", g0="", g1="", g2="", g3="", g4="", value=got, note=note, source=so_src))
        check("Data split", f"{item.strip()}: recomputed from analysis/image_sha256.csv equals analysis/split_overlap.csv", got, int(so[key]), ih_src + "; " + so_src, tol=0)
    for scheme, key_scheme in (("original split", "original"), ("clean split", "clean")):
        for part in ("train", "val", "test"):
            check("Data split", f"{scheme} {part} size equals analysis/split_overlap.csv", sum(counts[(scheme, part)]), int(so[f"{key_scheme}_{part}_n"]), so_src, tol=0)
    # arithmetic: clean test = original test - conflicts - test-test removals
    check("Data split", "clean test = original test - conflicts - test-test removals", cnt_o - ex_conflict - ex_tt, cnt_c, so_src, tol=0)
    ov = PT / "split_overlap_facts.csv"
    sof = pd.read_csv(need(ov)).iloc[0]
    for key, got in (("original_test_n", cnt_o), ("original_test_identical_to_train", leak_train), ("original_test_identical_to_val", leak_val), ("clean_test_n", cnt_c),
                     ("clean_test_identical_to_previous_train", clean_vs_train), ("clean_test_identical_to_previous_val", clean_vs_val)):
        check("Data split", f"split_overlap_facts.csv {key}", got, int(sof[key]), rel(ov), tol=0)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Messidor
def build_messidor(q, rz):
    rows = []
    mf = pd.read_csv(need(PT / "final_01" / "messidor_all_models_per_run.csv"))
    mf_all = mf
    mf = mf[mf["source"] == "final_01"].reset_index(drop=True)  # the replay row is read separately below
    f_src = rel(PT / "final_01" / "messidor_all_models_per_run.csv")
    sa_path = need(EV / "datasets" / "messidor_224_archived" / "source_audit.csv")
    sa = pd.read_csv(sa_path)
    sa_src = rel(sa_path)
    per_run = []
    for _, r in mf.iterrows():
        job = r["job"]
        pred = EV / "final_runs" / "final_01" / "scores" / job / "predictions.csv"
        summ = json.loads(need(pred.parent / "summary.json").read_text(encoding="utf-8"))
        rc = recompute(read_pred(pred))
        cmp_run("Messidor", job, rc, r, rel(pred), f1_recorded=summ["metrics"]["macro_f1"])
        is_fp32 = isinstance(r["fallback_for"], str) and r["fallback_for"] != ""
        per_run.append(dict(section="per run", model="Qwen3-VL-8B + LoRA" if r["adapter"] == "yes" else "Qwen3-VL-8B zero-shot", condition=r["condition"],
                            trainer_seed=(int(r["trainer_seed"]) if not pd.isna(r["trainer_seed"]) else NA), scoring_precision=("FP32 fallback" if is_fp32 else r["scoring_precision"]),
                            n=int(r["n"]), accuracy=r["accuracy"], qwk=r["qwk"], macro_f1=rc["macro_f1"], macro_auc=r["macro_auc"],
                            pred=rc["pred_counts"], constant=bool(r["degenerate"]), note=("FP32 fallback seed" if is_fp32 else ""), source=rel(pred), cm=rc["cm"], y=read_pred(pred)["y_true"].to_numpy()))
    # Retraining 2 replay on Messidor
    mr = pd.read_csv(need(PT / "retraining2_replay" / "messidor_archived_metrics.csv")).iloc[0]
    pred = EV / "repo" / "finetuning" / "Qwen3" / "artifacts" / "paper_reproduction" / "training_batches" / "run02_reproduction_01" / "replay_run_02" / "evaluation" / "messidor_external.predictions.csv"
    summ = json.loads(need(pred.parent / "messidor_external.summary.json").read_text(encoding="utf-8"))
    rc = recompute(read_pred(pred))
    cmp_run("Messidor", "Retraining 2 replay", rc, mr, rel(pred), f1_recorded=summ["macro_f1"])
    per_run.append(dict(section="per run", model="Qwen3-VL-8B + LoRA", condition="Retraining 2 replay (archived run_02, original split)", trainer_seed=42, scoring_precision=mr["scoring_precision"],
                        n=int(mr["n"]), accuracy=mr["accuracy"], qwk=mr["qwk"], macro_f1=rc["macro_f1"], macro_auc=mr["macro_auc"], pred=rc["pred_counts"], constant=False, note="", source=rel(pred),
                        cm=rc["cm"], y=read_pred(pred)["y_true"].to_numpy()))
    # RetiZero
    rzm = rz[rz["stage"] == "messidor_archived"]
    for _, r in rzm.iterrows():
        pred = r["pred_path"]
        rcc = recompute(read_pred(pred))
        per_run.append(dict(section="per run", model="RetiZero ViT-L/16 + LoRA", condition=("saved app checkpoint (original split)" if r["job"].startswith("saved") else "matched, clean split, CLAHE off, 224 px"),
                            trainer_seed=r["seed"], scoring_precision="CUDA automatic mixed precision (FP16)", n=int(r["n_test"]), accuracy=r["accuracy"], qwk=r["qwk"], macro_f1=r["macro_f1"],
                            macro_auc=r["macro_auc"], pred=[int(x) for x in str(r["predicted_per_grade"]).split()], constant=bool(r["constant_predictor"]), note="", source=rel(pred), cm=rcc["cm"],
                            y=read_pred(pred)["y_true"].to_numpy()))
    ys = [x["y"] for x in per_run]
    for y in ys[1:]:
        if not np.array_equal(y, ys[0]):
            raise RuntimeError("Messidor label order differs between prediction files")
    counts = [int((ys[0] == g).sum()) for g in GRADES]
    n_total = len(ys[0])
    check("Messidor", "every Messidor prediction file has the same 1,057 labels in the same order", True, True, "all Messidor predictions.csv files", tol=0)
    check("Messidor", "archive size equals the number of rows in source_audit.csv", n_total, len(sa), sa_src, tol=0)
    check("Messidor", "per-grade counts of the archive (grades 0-4) equal the grade column of source_audit.csv", counts, [int((sa["grade"] == g).sum()) for g in GRADES], sa_src, tol=0)
    shares = [c / n_total for c in counts]
    adjud = [0.58, 0.15, 0.20, 0.04, 0.02]
    out = []
    for g in GRADES:
        out.append(dict(section="label distribution", model="archive (1,057 images) against adjudicated Messidor-2 (1,744 gradable images)", condition=f"grade {g} {LABELS[g]}",
                        trainer_seed="", scoring_precision="", n=counts[g], accuracy="", qwk="", macro_f1="", macro_auc="", pred=None, constant="",
                        share_archive=shares[g], share_adjudicated=adjud[g], diff_pp=(shares[g] - adjud[g]) * 100.0,
                        note="adjudicated share is quoted to whole percent in the source" if g == 0 else "", source=sa_src + "; adjudicated shares: " + KRAUSE))
    out.append(dict(section="label distribution", model="note", condition="", trainer_seed="", scoring_precision="", n="", accuracy="", qwk="", macro_f1="", macro_auc="", pred=None, constant="",
                    share_archive="", share_adjudicated="", diff_pp="",
                    note="The archive shares do not match the adjudicated shares, and the archive labels are the archive's own derived labels. They appear not to be the adjudicated grades. The adjudicated grade file has not been obtained, so this cannot be confirmed. The 1,057 archive images also have no confirmed link to the 1,744 gradable adjudicated images.",
                    source=sa_src + "; " + KRAUSE))
    # group means
    msum = pd.read_csv(need(PT / "final_01" / "messidor_all_models_summary.csv"))
    ms_src = rel(PT / "final_01" / "messidor_all_models_summary.csv")
    pr = pd.DataFrame(per_run)
    groups = []
    for cond, sub in pr.groupby("condition", sort=False):
        groups.append((cond, sub))
    fp32_seed = pr[(pr["condition"].str.contains("alpha 32")) & (pr["scoring_precision"] == "FP16")]
    groups.append(("4-bit r16 CLAHE-on lr 0.0005 alpha 32 (FP16-scored seeds only)", fp32_seed))
    for cond, sub in groups:
        if len(sub) < 2:
            continue
        am, asd = msd(sub["accuracy"])
        qm, qsd = msd(sub["qwk"])
        fm, fsd = msd(sub["macro_f1"])
        um, usd = msd(sub["macro_auc"])
        out.append(dict(section="mean over seeds", model=sub["model"].iloc[0], condition=cond, trainer_seed="", scoring_precision="; ".join(sorted(sub["scoring_precision"].unique())),
                        n=int(sub["n"].iloc[0]), accuracy=am, qwk=qm, macro_f1=fm, macro_auc=um, pred=None, constant=f"{int(sub['constant'].sum())} of {len(sub)} predict one grade",
                        accuracy_sd=asd, qwk_sd=qsd, macro_f1_sd=fsd, macro_auc_sd=usd, note=f"{len(sub)} runs; sample SD across runs",
                        source="; ".join(sorted(set(sub["source"]))[:1]) + " (and sibling seed folders)"))
        lookup = "4-bit r16 CLAHE-on lr 0.0005 alpha 32" if cond.endswith("(FP16-scored seeds only)") else cond
        ref = msum[(msum["condition"] == lookup) & (msum["n_runs"] == len(sub))] if lookup in set(msum["condition"]) else None
        if ref is not None and len(ref) >= 1:
            # FP32-fallback group has two summary rows; compare the mixed row for n_runs==3 only
            ref = ref.iloc[0]
            check("Messidor", f"{cond}: mean accuracy over {len(sub)} runs", am, ref["accuracy_mean"], ms_src)
            check("Messidor", f"{cond}: mean QWK", qm, ref["qwk_mean"], ms_src)
            check("Messidor", f"{cond}: mean macro AUC", um, ref["macro_auc_mean"], ms_src)
    rows = out + [dict(**{k: v for k, v in r.items() if k not in ("cm", "y")}) for r in per_run]
    rows = [{**r, "pred_g0": (r["pred"][0] if r.get("pred") else ""), "pred_g1": (r["pred"][1] if r.get("pred") else ""), "pred_g2": (r["pred"][2] if r.get("pred") else ""),
             "pred_g3": (r["pred"][3] if r.get("pred") else ""), "pred_g4": (r["pred"][4] if r.get("pred") else "")} for r in rows]
    df = pd.DataFrame(rows).drop(columns=["pred"])
    return df, pr, counts


MCOLS = ["section", "model", "condition", "trainer_seed", "scoring_precision", "n", "accuracy", "accuracy_sd", "qwk", "qwk_sd", "macro_f1", "macro_f1_sd", "macro_auc", "macro_auc_sd",
         "pred_g0", "pred_g1", "pred_g2", "pred_g3", "pred_g4", "constant", "share_archive", "share_adjudicated", "diff_pp", "note", "source"]


# ---------------------------------------------------------------- Efficiency
def build_efficiency(q, rz):
    rows = []

    def add(section, model, config, cohort, precision, n, metric, unit, mean, sd=None, note="", source=""):
        rows.append(dict(section=section, model=model, configuration=config, cohort=cohort, precision=precision, n_runs=n, metric=metric, unit=unit,
                         mean=mean, sd=(sd if (sd is not None and not (isinstance(sd, float) and math.isnan(sd))) else ""), note=note, source=source))

    # parameters
    em = pd.read_csv(need(PT / "efficiency" / "efficiency_models.csv"))
    em_src = rel(PT / "efficiency" / "efficiency_models.csv")
    pc_q = json.loads(need(EV / "final_runs" / "final_01" / "parameter_counts.json").read_text(encoding="utf-8"))
    pc_r = json.loads(need(EV / "retizero_runs" / "retizero_01" / "parameter_counts.json").read_text(encoding="utf-8"))
    pq_src = rel(EV / "final_runs" / "final_01" / "parameter_counts.json")
    pr_src = rel(EV / "retizero_runs" / "retizero_01" / "parameter_counts.json")
    base_params = pc_q["base_parameters"]
    check("Efficiency", "Qwen base parameter count", base_params, 8767123696, pq_src, tol=0)
    for rk, exp in (("rank_8", 25673472), ("rank_16", 51346944), ("rank_32", 102693888)):
        tp = pc_q["adapters"][rk]["trainable_parameters"]
        check("Efficiency", f"Qwen trainable parameters, {rk}", tp, exp, pq_src, tol=0)
        add("parameters", "Qwen3-VL-8B + LoRA", f"LoRA {rk.replace('_', ' ')}", NA, NA, 1, "trainable parameters", "count", tp, note=f"adapter file {pc_q['adapters'][rk]['adapter_file_bytes']:,} bytes; {pc_q['adapters'][rk]['percent_of_base']:.3f}% of the base", source=pq_src)
    add("parameters", "Qwen3-VL-8B + LoRA", "frozen base (vision and language)", NA, NA, 1, "base parameters", "count", base_params, note="product of tensor shapes in the safetensors headers", source=pq_src)
    check("Efficiency", "RetiZero model parameters", pc_r["model_total"], 304093189, pr_src, tol=0)
    check("Efficiency", "RetiZero trained parameters (all weights)", pc_r["trained_in_fine_tuning"], 304093189, pr_src, tol=0)
    add("parameters", "RetiZero ViT-L/16 + LoRA r8", "vision encoder and five-grade head, every weight trained", NA, NA, 1, "trainable parameters", "count", pc_r["trained_in_fine_tuning"],
        note="encoder 304,088,064 + LoRA branches 786,432 are inside the encoder count as logged; head 5,125; the checkpoint file also holds an unused second encoder copy and text model", source=pr_src)
    add("parameters", "RetiZero ViT-L/16 + LoRA r8", "checkpoint file", NA, NA, 1, "file size", "bytes", 1216540027, note="best_epoch_*.pth; not copied into the evidence folder", source=em_src)

    # Qwen training by configuration
    tb = pd.read_csv(need(PT / "efficiency" / "efficiency_training_by_config.csv"))
    tb_src = rel(PT / "efficiency" / "efficiency_training_by_config.csv")
    for _, r in tb.iterrows():
        if bool(r["is_pilot"]):
            continue
        cfg = r["config"]
        prec = r["base"] + (" base, 16 layers CPU-offloaded" if r["base"] == "BF16" else " base")
        n = int(r["n"])
        for metric, unit, mcol, scol in (("training wall time, all attempts", "s", "train_wall_all_attempts_s_mean", "train_wall_all_attempts_s_sd"),
                                         ("epochs run (early stopping)", "epochs", "epochs_run_mean", "epochs_run_sd"),
                                         ("training time per epoch, including validation", "s/epoch", "s_per_epoch_mean", "s_per_epoch_sd"),
                                         ("training throughput", "images/s", "images_per_s_mean", "images_per_s_sd"),
                                         ("peak GPU memory, PyTorch allocated", "GiB", "torch_peak_alloc_gib_mean", "torch_peak_alloc_gib_sd"),
                                         ("peak GPU memory, whole card", "MiB", "card_peak_vram_mib_mean", "card_peak_vram_mib_sd"),
                                         ("mean GPU utilisation", "fraction", "gpu_mean_pct_mean", "gpu_mean_pct_sd")):
            v = r[mcol]
            sdv = r[scol]
            if unit == "fraction":
                v, sdv = v / 100.0, sdv / 100.0
            add("training", "Qwen3-VL-8B + LoRA", cfg, "APTOS clean train (2,786)" if "previous split" not in cfg else "APTOS original train (2,929)", prec, n, metric, unit, float(v), float(sdv) if n > 1 else None,
                note=("whole-card figures come from nvidia-smi samples every 5 s and include other processes" if unit in ("MiB", "fraction") else ""), source=tb_src)
        add("training", "Qwen3-VL-8B + LoRA", cfg, "APTOS clean train (2,786)" if "previous split" not in cfg else "APTOS original train (2,929)", prec, n, "GPU power and energy", "W, Wh", NR,
            note="no power field in utilization.csv or system_specs.json", source=tb_src)

    # Qwen scoring by configuration
    sb = pd.read_csv(need(PT / "efficiency" / "efficiency_scoring_by_config.csv"))
    sb_src = rel(PT / "efficiency" / "efficiency_scoring_by_config.csv")
    for _, r in sb.iterrows():
        if bool(r["is_pilot"]) or str(r["model"]).startswith("RetiZero"):
            continue
        n = int(r["n"])
        prec = r["scoring_precision"]
        cfg = r["model"]
        pairs = [("model load time", "s", "model_load_s_mean", "model_load_s_sd"), ("warm median time per image, batch 1 (likelihood scorer, five forward passes)", "s/image", "warm_median_s_mean", "warm_median_s_sd"),
                 ("throughput, batch 1", "images/s", "warm_images_per_s_mean", "warm_images_per_s_sd"), ("greedy generation, warm median", "s/image", "gen_median_s_mean", "gen_median_s_sd"),
                 ("peak GPU memory, PyTorch allocated", "GiB", "torch_peak_alloc_gib_mean", "torch_peak_alloc_gib_sd"), ("peak GPU memory, whole card", "MiB", "card_peak_vram_mib_mean", "card_peak_vram_mib_sd"),
                 ("mean GPU utilisation", "fraction", "gpu_mean_pct_mean", "gpu_mean_pct_sd")]
        for metric, unit, mcol, scol in pairs:
            v = r[mcol]
            if pd.isna(v):
                continue
            sdv = r[scol]
            if unit == "fraction":
                v, sdv = v / 100.0, sdv / 100.0
            add("scoring", "Qwen3-VL-8B + LoRA" if "zero-shot" not in cfg else "Qwen3-VL-8B zero-shot", cfg, r["cohort"], prec, n, metric, unit, float(v), (float(sdv) if (n > 1 and not pd.isna(sdv)) else None),
                note=("FP32 fallback timing is not comparable with FP16" if prec == "FP32 fallback" else ""), source=sb_src)

    # RetiZero training from the retizero_01 queue
    tr = rz[rz["kind"] == "train"]
    root_src = rel(EV / "retizero_runs" / "retizero_01" / "summary.csv")
    for stage, label, split in (("historical", "historical (original split, CLAHE off)", "APTOS original train (2,929)"), ("matched", "matched (clean split, CLAHE off)", "APTOS clean train (2,786)"),
                                ("clahe", "CLAHE (clean split, CLAHE on at 224 px)", "APTOS clean train (2,786)")):
        sub = tr[tr["stage"] == stage]
        n = len(sub)
        prec = "CUDA automatic mixed precision (FP16)"
        wall = msd(sub["stage_wall_s"])
        epochs = msd(sub["epochs_run"])
        spe = msd([w / e for w, e in zip(sub["stage_wall_s"], sub["epochs_run"])])
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "training wall time", "s", wall[0], wall[1], note="stage time from summary.csv (evaluation excluded); one attempt per job", source=root_src)
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "epochs run (early stopping)", "epochs", epochs[0], epochs[1], source=root_src)
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "training time per epoch", "s/epoch", spe[0], spe[1], note="derived: stage wall time divided by epochs run; includes validation", source=root_src)
        v = msd(sub["train_images_per_s"])
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "training throughput", "images/s", v[0], v[1], note="recorded by the trainer", source=root_src)
        v = msd(sub["torch_peak_gib"])
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "peak GPU memory, PyTorch allocated", "GiB", v[0], v[1], source=root_src)
        v = msd(sub["card_peak_vram_mib"])
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "peak GPU memory, whole card", "MiB", v[0], v[1], note="nvidia-smi samples every 5 s, whole machine", source=root_src)
        v = msd(sub["mean_gpu_pct"])
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "mean GPU utilisation", "fraction", v[0], v[1], source=root_src)
        v = msd(sub["energy_wh"])
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "GPU energy per job (training plus evaluation window)", "Wh", v[0], v[1],
            note="GPU board power from nvidia-smi every 5 s, integrated over the job window; whole card, includes about 20 W idle draw; CPU, RAM and disk excluded", source=rel(EV / "retizero_runs" / "retizero_01" / "energy_per_job.csv"))
        v = msd(sub["mean_power_w"])
        add("training", "RetiZero ViT-L/16 + LoRA r8", label, split, prec, n, "mean GPU board power over the job window", "W", v[0], v[1], note="whole card", source=rel(EV / "retizero_runs" / "retizero_01" / "energy_per_job.csv"))
    # RetiZero scoring
    sc = rz[rz["kind"] == "score"]
    for label, mask, cohort in (("saved checkpoint", sc["job"] == "saved_checkpoint__aptos361_clean", "aptos361_clean"),
                                ("saved checkpoint", sc["job"] == "saved_checkpoint__aptos367_original", "aptos367_original"),
                                ("matched checkpoints (benchmark rescoring)", sc["stage"] == "benchmark", "aptos361_clean"),
                                ("saved checkpoint", sc["job"] == "saved_checkpoint__messidor_archived", "messidor_archived"),
                                ("matched checkpoints", (sc["stage"] == "messidor_archived") & sc["job"].str.startswith("matched"), "messidor_archived")):
        sub = sc[mask]
        n = len(sub)
        prec = "CUDA automatic mixed precision (FP16)"
        for metric, unit, col, note in (("warm median time per image, batch 1", "s/image", "median_ms_per_image", "converted from ms; includes image read and transform, transfer, forward pass and softmax"),
                                        ("throughput, batched (batch 64)", "images/s", "images_per_s_batched", "recorded"),
                                        ("model load time", "s", "model_load_s", ""), ("peak GPU memory, PyTorch allocated (batch 1)", "GiB", "torch_peak_gib", ""),
                                        ("peak GPU memory, whole card", "MiB", "card_peak_vram_mib", ""), ("mean GPU utilisation", "fraction", "mean_gpu_pct", "job lasts under a minute, so few samples"),
                                        ("GPU energy per scoring job", "Wh", "energy_wh", "whole job window including model load; whole-card power")):
            vals = [x for x in sub[col] if not isinstance(x, str)]
            if not vals:
                add("scoring", "RetiZero ViT-L/16 + LoRA r8", label, cohort, prec, n, metric, unit, NR, note="job ran before power logging started" if col == "energy_wh" else "", source=root_src)
                continue
            m, s_ = msd(vals)
            if col == "median_ms_per_image":
                m, s_ = m / 1000.0, s_ / 1000.0
            add("scoring", "RetiZero ViT-L/16 + LoRA r8", label, cohort, prec, n, metric, unit, m, (s_ if n > 1 else None), note=note, source=root_src if col != "energy_wh" else rel(EV / "retizero_runs" / "retizero_01" / "energy_per_job.csv"))
            if col == "energy_wh":
                per_img = [float(e) / float(k) for e, k in zip(sub["energy_wh"], sub["n_test"]) if not isinstance(e, str)]
                pm, ps = msd(per_img)
                add("scoring", "RetiZero ViT-L/16 + LoRA r8", label, cohort, prec, n, "GPU energy per image (job energy divided by images scored)", "Wh/image", pm, (ps if n > 1 else None),
                    note="derived; includes model load and idle draw of the whole card, so it overstates the marginal cost of one image", source=rel(EV / "retizero_runs" / "retizero_01" / "energy_per_job.csv"))

    # gaps, stated plainly; no file holds these quantities, so there is no source file
    NOSRC = "none: quantity not recorded"
    for model, cfg, metric, note in (
        ("Qwen3-VL-8B + LoRA", "all training and scoring runs", "GPU power and energy", "not recorded; utilization.csv holds GPU %, VRAM, CPU % and RAM only. Measuring it needs a new run."),
        ("Qwen3-VL-8B + LoRA", "7 training configurations with only an inline score stage", "warm per-image scoring time", "not recorded; only the stage wall time (model load plus five passes per image) was saved. A short rescoring job per configuration would measure it."),
        ("Qwen3-VL-8B + LoRA", "all scoring jobs", "batched or multi-image throughput", "not recorded; scoring was timed at batch size 1 only."),
        ("Qwen3-VL-8B + LoRA", "original paper adapter", "training time, epochs, trainable parameters", "not recorded; its folder is outside the experiment tree and no training record was read."),
        ("RetiZero ViT-L/16 + LoRA r8", "saved checkpoint and replay jobs", "training time and energy", "not recorded; the saved checkpoint came from the original notebook, and the two replay jobs ran before power logging began."),
        ("RetiZero ViT-L/16 + LoRA r8", "scoring jobs", "SD across repeats", "not recorded; one scoring job per cohort and checkpoint, so SD over repeats cannot be given."),
        ("both models", "all runs", "per-process host memory", "not recorded; utilization.csv is whole-machine (the first sample already shows a 1.4 GB card baseline and about 28 GB RAM in use)."),
        ("both models", "energy figures", "scope of energy", "RetiZero energy is whole-card GPU board power sampled every 5 s and includes about 20 W idle draw. Training energy covers training plus evaluation, while the wall time covers training only."),
        ("both models", "efficiency tables built from the Qwen runs", "RetiZero full training rows", "the earlier efficiency tables show RetiZero training as not run; the RetiZero rows in this sheet come from the retizero_01 queue and those tables have not been rebuilt.")):
        value = NA if metric in ("scope of energy", "RetiZero full training rows") else NR
        add("gap", model, cfg, "", "", "", metric, "", value, note=note, source=NOSRC)

    df = pd.DataFrame(rows)
    # cross-checks against the published efficiency tables
    comp = pd.read_csv(need(PT / "efficiency" / "efficiency_comparison.csv"))
    ref = comp[(comp["quantity"] == "Scoring warm median time per image (batch 1)") & (comp["model"].str.contains("r16, 4-bit"))].iloc[0]
    mine = df[(df["section"] == "scoring") & (df["configuration"] == "4-bit r16 CLAHE-on lr 0.0005 alpha 16") & (df["cohort"] == "aptos361_on") & (df["metric"].str.startswith("warm median"))].iloc[0]
    check("Efficiency", "Qwen r16 4-bit warm median time per image equals efficiency_comparison.csv", float(mine["mean"]), float(ref["mean"]), rel(PT / "efficiency" / "efficiency_comparison.csv"))
    ref = comp[(comp["quantity"] == "Training wall time, all attempts") & (comp["model"].str.contains("r16, 4-bit"))].iloc[0]
    mine = df[(df["section"] == "training") & (df["configuration"] == "4-bit r16 CLAHE-on lr 0.0005 alpha 16") & (df["metric"] == "training wall time, all attempts")].iloc[0]
    check("Efficiency", "Qwen r16 4-bit training wall time equals efficiency_comparison.csv", float(mine["mean"]), float(ref["mean"]), rel(PT / "efficiency" / "efficiency_comparison.csv"))
    return df


# ---------------------------------------------------------------- Head to head, confusion
def build_head_to_head(q, rz):
    rows = []
    conf_rows = []
    cm_sources = {}
    clean_ref = None

    def qsel(mask):
        return q[mask].sort_values("trainer_seed")

    lr2 = qsel(q["stage"].eq("learning-rate study") & (q["lr"] == 0.0002))
    lr5 = qsel((q["result_set"] == "A") & (q["rank"] == 16) & (q["clahe"] == "on"))
    rzm = rz[rz["stage"] == "matched"].sort_values("seed")
    rzc = rz[rz["stage"] == "clahe"].sort_values("seed")
    best_lr2 = lr2.sort_values("accuracy", ascending=False).iloc[0]
    sets = [
        ("Qwen3-VL-8B + LoRA", "rank 16, alpha 16, lr 2e-4, CLAHE on, 4-bit (best stable condition)", lr2, "q"),
        ("Qwen3-VL-8B + LoRA", "rank 16, alpha 16, lr 5e-4, CLAHE on, 4-bit (main setting; includes one weak seed)", lr5, "q"),
        ("RetiZero ViT-L/16 + LoRA r8", "matched: clean split, CLAHE off, 224 px", rzm, "r"),
        ("RetiZero ViT-L/16 + LoRA r8", "CLAHE on at 224 px, clean split", rzc, "r"),
    ]
    ids_ref = None
    for model, cond, sub, kind in sets:
        for _, r in sub.iterrows():
            pred_path = r["pred_path"]
            df = read_pred(pred_path)
            ids = df["image_id"].tolist() if "image_id" in df.columns else df["original_image_id"].tolist()
            yt = df["y_true"].tolist()
            if ids_ref is None:
                ids_ref = (ids, yt)
            same = (ids == ids_ref[0]) and (yt == ids_ref[1])
            check("Head to head", f"{model} / {cond} / seed {int(r['trainer_seed'] if kind == 'q' else r['seed'])}: same 361 test IDs, same order, same grades as the first file", bool(same), True, rel(pred_path), tol=0)
            seed = int(r["trainer_seed"] if kind == "q" else r["seed"])
            rows.append(dict(model=model, condition=cond, row_type="seed", seed=seed, n_test=int(r["n_test"]), accuracy=r["accuracy"], accuracy_sd="", qwk=r["qwk"], qwk_sd="", macro_f1=r["macro_f1"], macro_f1_sd="",
                             macro_auc=r["macro_auc"], macro_auc_sd="", predicts_one_grade=("yes" if (r["collapsed"] if kind == "q" else r["constant_predictor"]) else "no"),
                             note=("highest-accuracy seed of this condition; its confusion matrix is on sheet Confusion_matrices" if (kind == "q" and "2e-4" in cond and r["run_id"] == best_lr2["run_id"]) else ""),
                             source=rel(pred_path)))
        am, asd = msd(sub["accuracy"])
        qm, qsd = msd(sub["qwk"])
        fm, fsd = msd(sub["macro_f1"])
        um, usd = msd(sub["macro_auc"])
        rows.append(dict(model=model, condition=cond, row_type="mean over 3 seeds", seed="", n_test=int(sub["n_test"].iloc[0]), accuracy=am, accuracy_sd=asd, qwk=qm, qwk_sd=qsd, macro_f1=fm, macro_f1_sd=fsd,
                         macro_auc=um, macro_auc_sd=usd, predicts_one_grade="", note="sample SD across seeds", source="; ".join(sorted(set(sub["source"]))) if kind == "q" else rel(EV / "retizero_runs" / "retizero_01" / "summary.csv")))
    # differences of means
    def mean_of(model_cond):
        r = [x for x in rows if x["row_type"] == "mean over 3 seeds" and x["condition"].startswith(model_cond)][0]
        return r
    mq2 = mean_of("rank 16, alpha 16, lr 2e-4")
    mq5 = mean_of("rank 16, alpha 16, lr 5e-4")
    mr = mean_of("matched")
    for lab, mq in (("lr 2e-4", mq2), ("lr 5e-4", mq5)):
        rows.append(dict(model="difference of means: RetiZero matched minus Qwen " + lab, condition="", row_type="difference of means (accuracy in percentage points, shown as %)", seed="", n_test=361,
                         accuracy=mr["accuracy"] - mq["accuracy"], accuracy_sd="", qwk=mr["qwk"] - mq["qwk"], qwk_sd="", macro_f1=mr["macro_f1"] - mq["macro_f1"], macro_f1_sd="",
                         macro_auc=mr["macro_auc"] - mq["macro_auc"], macro_auc_sd="", predicts_one_grade="",
                         note="descriptive difference between two three-seed means on the same 361 images; seeds of the two models are not paired; no significance test has been run", source="rows above"))
    rows.append(dict(model="note", condition="", row_type="note", seed="", n_test="", accuracy="", accuracy_sd="", qwk="", qwk_sd="", macro_f1="", macro_f1_sd="", macro_auc="", macro_auc_sd="", predicts_one_grade="",
                     note="The paired comparison (paired bootstrap and McNemar test on the same 361 images) is still pending. Inputs exist for both models. Until it is run, differences above are not tested for significance.",
                     source="none: analysis not yet run"))
    rows.append(dict(model="note", condition="", row_type="note", seed="", n_test="", accuracy="", accuracy_sd="", qwk="", qwk_sd="", macro_f1="", macro_f1_sd="", macro_auc="", macro_auc_sd="", predicts_one_grade="",
                     note="Selection protocols differ: Qwen checkpoints are chosen by validation loss, RetiZero checkpoints by validation accuracy with patience 4. Inputs differ: Qwen uses 512 px, RetiZero 224 px.",
                     source="; ".join([rel(next(iter(sorted((EV / "runs").glob("full_r16_on_duplicate_excluded_t42_a3407_01/pilot_configuration.json"))))),
                                      rel(EV / "retizero_runs" / "retizero_01" / "train" / "matched_t42" / "train_result.json"),
                                      rel(EV / "retizero_runs" / "retizero_01" / "train" / "matched_t42" / "configuration.json")])))
    # confusion matrices of the headline runs
    heads = []
    heads.append(("Qwen r16 lr 2e-4 CLAHE on, best seed (seed %d)" % int(best_lr2["trainer_seed"]), best_lr2["pred_path"], best_lr2["cm"], "q", best_lr2["run_id"]))
    q5 = lr5[lr5["trainer_seed"] == 42].iloc[0]
    heads.append(("Qwen r16 lr 5e-4 CLAHE on, seed 42", q5["pred_path"], q5["cm"], "q", q5["run_id"]))
    for _, r in rzm.iterrows():
        heads.append((f"RetiZero matched, seed {int(r['seed'])}", r["pred_path"], r["cm"], "r", r["job"]))
    for label, pred_path, cm, kind, rid in heads:
        pred_counts = cm.sum(axis=0)
        for g in GRADES:
            tot = int(cm[g].sum())
            conf_rows.append(dict(run=label, run_id=rid, true_grade=g, true_label=LABELS[g], pred_0=int(cm[g, 0]), pred_1=int(cm[g, 1]), pred_2=int(cm[g, 2]), pred_3=int(cm[g, 3]), pred_4=int(cm[g, 4]),
                                  n_true=tot, recall=(cm[g, g] / tot if tot else float("nan")), n_test=int(cm.sum()), source=rel(pred_path)))
        conf_rows.append(dict(run=label, run_id=rid, true_grade="predicted total", true_label="", pred_0=int(pred_counts[0]), pred_1=int(pred_counts[1]), pred_2=int(pred_counts[2]), pred_3=int(pred_counts[3]),
                              pred_4=int(pred_counts[4]), n_true=int(cm.sum()), recall="", n_test=int(cm.sum()), source=rel(pred_path)))
        # cross-check with a saved confusion matrix
        if kind == "q":
            if rid.startswith("full_"):
                cmcsv = pd.read_csv(PT / "aptos_361_clean" / "confusion_matrices_per_run.csv")
                saved = cmcsv[cmcsv["run"] == rid][[f"pred_{k}" for k in range(5)]].to_numpy()
                sv = rel(PT / "aptos_361_clean" / "confusion_matrices_per_run.csv")
            else:
                cmcsv = pd.read_csv(PT / "final_01" / "confusion_matrices_per_job.csv")
                saved = cmcsv[cmcsv["job"] == rid][[f"pred_{k}" for k in range(5)]].to_numpy()
                sv = rel(PT / "final_01" / "confusion_matrices_per_job.csv")
        else:
            summ = json.loads((pred_path.parent / "aptos_test.summary.json").read_text(encoding="utf-8"))
            saved = np.array(summ["metrics"]["confusion_matrix"] if "metrics" in summ else summ["confusion_matrix"])
            sv = rel(pred_path.parent / "aptos_test.summary.json")
        check("Confusion matrices", f"{label}: matrix recomputed from predictions equals the saved matrix", bool(np.array_equal(saved, cm)), True, sv, tol=0)
        check("Confusion matrices", f"{label}: matrix sums to 361", int(cm.sum()), 361, rel(pred_path), tol=0)
    return pd.DataFrame(rows), pd.DataFrame(conf_rows)


HCOLS = ["model", "condition", "row_type", "seed", "n_test", "accuracy", "accuracy_sd", "qwk", "qwk_sd", "macro_f1", "macro_f1_sd", "macro_auc", "macro_auc_sd", "predicts_one_grade", "note", "source"]


# ---------------------------------------------------------------- headline checks against the per-run metric files
def raw_metrics(pred_path):
    """Metric dictionary written by the scorer next to a prediction file, and its evidence path."""
    folder = Path(pred_path).parent
    name = Path(pred_path).name
    for cand in (folder / "score_result.json", folder / "summary.json", folder / name.replace(".predictions.csv", ".summary.json")):
        if cand.is_file():
            j = json.loads(cand.read_text(encoding="utf-8"))
            return rel(cand), j.get("metrics", j)
    raise FileNotFoundError(f"no metric file next to {pred_path}")


def check_headline(q, qs, rzs, rz):
    """Mean and SD of accuracy in every Qwen and RetiZero group, taken straight from the per-run metric JSON files."""
    def pct2(x):
        return round(100 * float(x) + 1e-9, 2)

    def one(label, sheet_mean, sheet_sd, pred_paths):
        vals, srcs = [], []
        for pth in pred_paths:
            src, met = raw_metrics(pth)
            vals.append(float(met["accuracy"]))
            srcs.append(src)
        m, sd = msd(vals)
        where = srcs[0] + (" (and the sibling runs' metric files)" if len(srcs) > 1 else "")
        check("Headline numbers", f"{label}: mean accuracy (percent, two decimals) equals the mean of the per-run metric files", pct2(sheet_mean), pct2(m), where, tol=0)
        if len(vals) > 1:
            check("Headline numbers", f"{label}: SD of accuracy (percentage points, two decimals) equals the SD of the per-run metric files", pct2(sheet_sd), pct2(sd), where, tol=0)

    for (label, mask, _), (_, r) in zip(condition_groups(q), qs.iterrows()):
        assert r["condition"] == label
        one("Qwen " + label, r["accuracy_mean"], r["accuracy_sd"], list(q[mask]["pred_path"]))
    for _, r in rzs.iterrows():
        jobs = r["member_jobs"].split(" | ")
        one("RetiZero " + r["group"], r["accuracy_mean"], r["accuracy_sd"], list(rz[rz["job"].isin(jobs)]["pred_path"]))
    # every RetiZero job in the summary table is recorded as passed in the queue state
    qpath = EV / "retizero_runs" / "retizero_01" / "queue_state.json"
    queue = {x["job"]: x for x in json.loads(need(qpath).read_text(encoding="utf-8"))}
    for job in rz["job"]:
        check("Headline numbers", f"RetiZero {job}: queue outcome is passed", queue.get(job, {}).get("outcome"), "passed", rel(qpath), tol=0)


# ---------------------------------------------------------------- writing
FILL_HEADER = PatternFill("solid", fgColor="D9D9D9")
FONT = "Arial"


def write_sheet(wb, name, df, fmt=None, wrap=(), max_width=70, min_width=8, first=False):
    ws = wb.create_sheet(name) if not first else wb.active
    if first:
        ws.title = name
    fmt = fmt or {}
    cols = list(df.columns)
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=j, value=c)
        cell.font = Font(name=FONT, bold=True, size=10)
        cell.fill = FILL_HEADER
        cell.alignment = Alignment(vertical="top", wrap_text=False)
    widths = [len(str(c)) for c in cols]
    for i, rec in enumerate(df.itertuples(index=False), 2):
        for j, v in enumerate(rec, 1):
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            if isinstance(v, float) and math.isnan(v):
                v = None
            if isinstance(v, (list, tuple, np.ndarray)):
                v = str(list(v))
            cell = ws.cell(row=i, column=j, value=v)
            cell.font = Font(name=FONT, size=10)
            f = fmt.get(cols[j - 1])
            if f and isinstance(v, (int, float)) and not isinstance(v, bool):
                cell.number_format = f
            if cols[j - 1] in wrap:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            else:
                cell.alignment = Alignment(vertical="top")
            if v is not None:
                s = f"{v:.4f}" if isinstance(v, float) else str(v)
                widths[j - 1] = max(widths[j - 1], min(len(s), 400))
    for j, c in enumerate(cols, 1):
        w = widths[j - 1]
        if c in wrap:
            w = min(w, 60)
        ws.column_dimensions[get_column_letter(j)].width = max(min_width, min(max_width, w + 2))
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(df) + 1}"
    return ws


def per_row_formats(ws, df, unit_col="unit", value_cols=("mean", "sd")):
    """Efficiency sheet: number format follows the unit of each row."""
    cols = list(df.columns)
    ui = cols.index(unit_col) + 1
    for i in range(2, len(df) + 2):
        unit = ws.cell(row=i, column=ui).value
        for vc in value_cols:
            c = ws.cell(row=i, column=cols.index(vc) + 1)
            if not isinstance(c.value, (int, float)) or isinstance(c.value, bool):
                continue
            if unit == "fraction":
                c.number_format = "0.00%"
            elif unit in ("count", "bytes", "MiB"):
                c.number_format = "#,##0"
            elif unit in ("s/image",):
                c.number_format = "0.0000"
            elif unit == "Wh/image":
                c.number_format = "0.000000"
            else:
                c.number_format = "#,##0.00"


def main():
    global EV, PT
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    EV = Path(args.evidence)
    PT = EV / "repo" / "finetuning" / "paper_tables" / "output"
    out = Path(args.output) if args.output else EV.parent / "Afterimage_results.xlsx"
    if not str(out).lower().startswith("d:"):
        raise SystemExit("output must be on D:")

    q = build_qwen()
    qs = build_qwen_summary(q)
    rz = build_retizero()
    rzs = build_retizero_summary(rz)
    split = build_split()
    mess, mess_runs, mess_counts = build_messidor(q, rz)
    eff = build_efficiency(q, rz)
    h2h, conf = build_head_to_head(q, rz)
    check_headline(q, qs, rzs, rz)

    built = dt.date.today().isoformat()
    n_fail = sum(1 for c in CHECKS if c["result"] != "pass")
    n_pass = len(CHECKS) - n_fail

    pct = "0.00%"
    f4 = "0.0000"
    wb = Workbook()

    index_rows = [
        ("Index", "This sheet. Sheet list, test sets used, conventions.", "", ""),
        ("Data_split", "Images per split and grade for the original split (2,929 / 366 / 367) and the clean split (2,786 / 357 / 361). Exact-duplicate counts recomputed from SHA-256 of the raw PNG bytes.", "", "analysis/split_overlap.csv, analysis/image_sha256.csv, datasets"),
        ("Qwen_runs", "One row per Qwen3-VL-8B + LoRA run or scoring job: settings, seed, test set, accuracy, QWK, macro F1, macro AUC, bootstrap CIs, scoring precision, status, training wall time.", "", "runs, final_runs/final_01 and repo run02_reproduction_01"),
        ("Qwen_summary", "Condition means and sample SDs over seeds, built from Qwen_runs.", "", "paper_tables output"),
        ("RetiZero_runs", "One row per job of the RetiZero queue retizero_01: replay, historical, matched, CLAHE, archived Messidor and timing jobs.", "", "retizero_runs/retizero_01"),
        ("RetiZero_summary", "Group means and sample SDs for the RetiZero jobs.", "", "retizero_runs/retizero_01"),
        ("Head_to_head", "Qwen r16 lr 2e-4 and r16 lr 5e-4 against RetiZero matched on the same 361 images. The paired test is pending.", "", ""),
        ("Messidor", "Archived 1,057-image cohort results for both models, and the archive label distribution against the adjudicated shares.", "", ""),
        ("Efficiency", "Time per image, throughput, memory, training time, GPU utilisation, energy (RetiZero only), parameter counts. Gaps are listed as rows with section 'gap'.", "", ""),
        ("Confusion_matrices", "Confusion matrices of the headline runs (rows are the true grade, columns the predicted grade).", "", ""),
        ("Checks", f"{len(CHECKS)} cross-checks: {n_pass} pass, {n_fail} fail.", "", ""),
    ]
    idx = pd.DataFrame([dict(sheet=a, contents=b, note=c, source=d) for a, b, c, d in index_rows])
    extra = [
        ("", "", "", ""),
        ("test sets", f"{T_CLEAN}: APTOS 2019 test images of the duplicate-excluded split. Main result set. Models trained on the clean 2,786-image train set are clean for it.", "", rel(EV / "analysis" / "split_overlap.csv")),
        ("test sets", f"{T_ORIG}: APTOS 2019 test images of the original split. 19 of the 367 have an exact copy outside test. Historical reference only.", "", rel(EV / "analysis" / "split_overlap.csv")),
        ("test sets", f"{T_MESS}: processed archive of the Messidor PNG set; the labels are the archive's own and appear not to be the adjudicated grades. It is not an independently verified cohort.", "", rel(EV / "datasets" / "messidor_224_archived" / "source_audit.csv")),
        ("convention", "Accuracy, shares and utilisation are stored as fractions and shown as percentages with two decimals. QWK, AUC and F1 are shown to four decimals. SD is the sample SD (n-1) across runs. CIs are per-run 95% bootstrap intervals (2,000 resamples) from the saved tables.", "", rel(PT / "aptos_361_clean" / "per_run_metrics.csv")),
        ("convention", "The letters A to F in Qwen condition names are the result-set labels of the paper tables (A clean suite, B previous-split training, C BF16 control, D original-split repeats, E Retraining 2 replay, F final_01 queue).", "", rel(PT / "all_sets_per_run_metrics.csv")),
        ("convention", "'not recorded' means no source file holds the quantity. 'n/a' means the quantity does not apply to that row.", "", ""),
        ("convention", "Macro F1 is computed here from each run's prediction file (mean of the five per-grade F1 values) and compared with the saved metric file where one records it. The saved paper tables do not hold macro F1 for the 4-bit and BF16 clean-suite runs.", "", ""),
        ("convention", "The source column holds the evidence-folder path of the file the row comes from. source_table, where present, names the saved table that holds the same values.", "", ""),
        ("convention", "Benchmark rescoring jobs in final_01 that repeat the clean-suite adapters on the 361 images are not listed as runs in Qwen_runs; their timing is used in Efficiency.", "", rel(PT / "final_01" / "inference_benchmark.csv")),
        ("convention", "Values are written as numbers, not formulas. Every written number is compared with its source on the Checks sheet.", "", ""),
        ("built", f"Built {built} from the evidence folder.", "", ""),
    ]
    idx = pd.concat([idx, pd.DataFrame([dict(sheet=a, contents=b, note=c, source=d) for a, b, c, d in extra])], ignore_index=True)
    write_sheet(wb, "Index", idx, wrap=("contents",), first=True)

    write_sheet(wb, "Data_split", split, fmt={"n_images": "#,##0", "g0": "#,##0", "g1": "#,##0", "g2": "#,##0", "g3": "#,##0", "g4": "#,##0", "value": "#,##0"}, wrap=("note",))

    qd = q[QCOLS].copy()
    qfmt = {"accuracy": pct, "acc_ci_low": pct, "acc_ci_high": pct, "qwk": f4, "qwk_ci_low": f4, "qwk_ci_high": f4, "macro_f1": f4, "macro_auc": f4, "auc_ci_low": f4, "auc_ci_high": f4,
            "train_wall_s": "#,##0.0", "lr": "0.0000", "n_test": "#,##0"}
    write_sheet(wb, "Qwen_runs", qd, fmt=qfmt, wrap=())

    qsd = qs.copy()
    qsfmt = {"accuracy_mean": pct, "accuracy_sd": pct, "accuracy_min": pct, "accuracy_max": pct, "qwk_mean": f4, "qwk_sd": f4, "macro_f1_mean": f4, "macro_f1_sd": f4, "macro_auc_mean": f4, "macro_auc_sd": f4, "n_test": "#,##0"}
    write_sheet(wb, "Qwen_summary", qsd, fmt=qsfmt)

    rd = rz[RCOLS].copy()
    rfmt = {"accuracy": pct, "best_val_acc": pct, "qwk": f4, "macro_f1": f4, "macro_auc": f4, "lr": "0.00000000", "stage_wall_s": "#,##0.0", "energy_wh": "0.000", "mean_power_w": "0.0",
            "median_ms_per_image": "0.000", "images_per_s_batched": "#,##0.0", "model_load_s": "0.00", "train_images_per_s": "0.0", "torch_peak_gib": "0.00", "card_peak_vram_mib": "#,##0", "mean_gpu_pct": pct, "n_test": "#,##0"}
    write_sheet(wb, "RetiZero_runs", rd, fmt=rfmt)

    rsd = rzs.copy()
    rsfmt = {"accuracy_mean": pct, "accuracy_sd": pct, "accuracy_min": pct, "accuracy_max": pct, "qwk_mean": f4, "qwk_sd": f4, "macro_f1_mean": f4, "macro_f1_sd": f4, "macro_auc_mean": f4, "macro_auc_sd": f4,
             "stage_wall_s_mean": "#,##0.0", "stage_wall_s_sd": "#,##0.0", "energy_wh_mean": "0.000", "energy_wh_sd": "0.000", "median_ms_per_image_mean": "0.000", "median_ms_per_image_sd": "0.000", "n_test": "#,##0"}
    write_sheet(wb, "RetiZero_summary", rsd, fmt=rsfmt)

    hfmt = {"accuracy": pct, "accuracy_sd": pct, "qwk": f4, "qwk_sd": f4, "macro_f1": f4, "macro_f1_sd": f4, "macro_auc": f4, "macro_auc_sd": f4}
    write_sheet(wb, "Head_to_head", h2h[HCOLS], fmt=hfmt)

    mfmt = {"accuracy": pct, "accuracy_sd": pct, "qwk": f4, "qwk_sd": f4, "macro_f1": f4, "macro_f1_sd": f4, "macro_auc": f4, "macro_auc_sd": f4, "share_archive": pct, "share_adjudicated": "0%", "diff_pp": "0.00", "n": "#,##0"}
    for c in MCOLS:
        if c not in mess.columns:
            mess[c] = ""
    write_sheet(wb, "Messidor", mess[MCOLS], fmt=mfmt)

    ws = write_sheet(wb, "Efficiency", eff, wrap=("note",), max_width=60)
    per_row_formats(ws, eff)

    write_sheet(wb, "Confusion_matrices", conf, fmt={"recall": pct})

    chk = pd.DataFrame(CHECKS)
    chk["workbook_or_recomputed_value"] = chk["workbook_or_recomputed_value"].map(lambda v: v if isinstance(v, (int, float, str, bool)) else str(v))
    chk["source_value"] = chk["source_value"].map(lambda v: v if isinstance(v, (int, float, str, bool)) else str(v))
    write_sheet(wb, "Checks", chk, wrap=("note",), max_width=80)

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"written {out}")
    print(f"sheets: {wb.sheetnames}")
    print(f"checks: {len(CHECKS)} total, {n_pass} pass, {n_fail} fail")
    for c in CHECKS:
        if c["result"] != "pass":
            print("FAIL:", c["area"], "|", c["check"], "|", c["workbook_or_recomputed_value"], "vs", c["source_value"])
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
