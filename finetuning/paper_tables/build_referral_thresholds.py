"""Referable-DR operating points for the headline models (decision: report the low Messidor-2 sensitivity as a
limitation only, or also show other thresholds).

Read-only on the run records. Writes CSV tables to finetuning/paper_tables/output/referral_thresholds/.

Referable DR is grade 2 or higher. Each model's referral score is the probability it gives to grades 2-4
(p2 + p3 + p4, the same score as the referable AUC in the analysis tables). Rules compared, per seed:

  default            refer when the predicted grade is 2 or higher (what the paper reports)
  aptos_sens90       threshold set on APTOS-361 to reach 90% sensitivity there, then applied unchanged to Messidor-2
  aptos_youden       threshold set on APTOS-361 at the maximum of sensitivity + specificity - 1, applied unchanged
  posthoc_sens80     POST HOC: threshold set on Messidor-2 itself to reach 80% sensitivity there
  posthoc_youden     POST HOC: threshold set on Messidor-2 itself at its maximum Youden index

No validation-set predictions were saved, so the pre-specified thresholds come from the in-domain APTOS test set,
never from Messidor-2. Their APTOS figures are therefore optimistic; their Messidor-2 figures are a fair external test.
The post hoc rules show how much room the score leaves, and must be labelled post hoc wherever they are reported.

Messidor-2 is reported per image and per examination (an examination is referable when either eye is; its score is
the higher of the two eyes' scores), using the dataset's eye-pairing file.

Usage (repo root, CPU only, under a minute):
  python -I -B finetuning/paper_tables/build_referral_thresholds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_revision_analysis as bra  # noqa: E402

EXPERIMENTS = Path(r"D:\Afterimage_Experiments")
PAIRS = Path(r"D:\Afterimage_data_share\Messidor-2\03_eye_pairing\pairs.csv")
OUT = HERE / "output" / "referral_thresholds"
MODELS = {"Qwen r16 CLAHE-on lr2e-4": "Qwen", "RetiZero CLAHE-off": "RetiZero"}


def rates(y: np.ndarray, refer: np.ndarray) -> dict:
    tp = int((refer & y).sum()); fn = int((~refer & y).sum())
    tn = int((~refer & ~y).sum()); fp = int((refer & ~y).sum())
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "referred_share": (tp + fp) / len(y),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def threshold_for_sensitivity(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity (score >= threshold) reaches the target."""
    pos = np.sort(s[y])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[k])


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    best, thr = -1.0, 0.5
    for t in np.unique(s):
        r = rates(y, s >= t)
        j = r["sensitivity"] + r["specificity"] - 1
        if j > best:
            best, thr = j, float(t)
    return thr


def exam_level(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per examination: referable if either eye is; score and predicted grade are the worse eye's."""
    f = frame.set_index("image_id")
    rows = []
    for left, right in zip(pairs.left, pairs.right):
        eyes = [e for e in (left, right) if e in f.index]
        if not eyes:
            continue
        sub = f.loc[eyes]
        rows.append({"y": int(sub.y_true.max() >= 2), "score": float(sub.score.max()), "pred": int(sub.y_pred.max())})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [r for r in bra.discover(EXPERIMENTS) if r["condition"] in MODELS and r["cohort"] in ("aptos361", "messidor2")]
    bra.load_all(runs)
    assert len(runs) == 12, len(runs)
    pairs = pd.read_csv(PAIRS, sep=";")
    pairs.columns = [c.strip() for c in pairs.columns]
    for c in ("left", "right"):
        pairs[c] = pairs[c].str.strip().map(lambda s: Path(s).stem)

    rows = []
    for cond, model in MODELS.items():
        for seed in sorted({r["seed"] for r in runs if r["condition"] == cond}):
            frames = {}
            for r in runs:
                if r["condition"] == cond and r["seed"] == seed:
                    f = r["frame"].copy()
                    f["score"] = f[["p2", "p3", "p4"]].sum(axis=1)
                    frames[r["cohort"]] = f
            a, m = frames["aptos361"], frames["messidor2"]
            ya, sa = (a.y_true.to_numpy() >= 2), a.score.to_numpy()
            ym, sm = (m.y_true.to_numpy() >= 2), m.score.to_numpy()
            ex = exam_level(m, pairs)
            ye, se = ex.y.to_numpy().astype(bool), ex.score.to_numpy()
            thresholds = {
                "aptos_sens90": threshold_for_sensitivity(ya, sa, 0.90),
                "aptos_youden": youden_threshold(ya, sa),
                "posthoc_sens80": threshold_for_sensitivity(ym, sm, 0.80),
                "posthoc_youden": youden_threshold(ym, sm),
            }
            # Exam-level post hoc thresholds are set on the exam scores themselves.
            exam_thresholds = dict(thresholds, posthoc_sens80=threshold_for_sensitivity(ye, se, 0.80),
                                   posthoc_youden=youden_threshold(ye, se))
            base = {"model": model, "condition": cond, "seed": seed}
            for level, y, s, pred, thr_map, n in (
                    ("APTOS-361 image", ya, sa, a.y_pred.to_numpy(), thresholds, len(a)),
                    ("Messidor-2 image", ym, sm, m.y_pred.to_numpy(), thresholds, len(m)),
                    ("Messidor-2 exam", ye, se, ex.pred.to_numpy(), exam_thresholds, len(ex))):
                auc = roc_auc_score(y, s)
                rows.append({**base, "level": level, "rule": "default", "post_hoc": False, "threshold": np.nan,
                             "n": n, "referable_auc": auc, **rates(y, pred >= 2)})
                for rule, t in thr_map.items():
                    if rule.startswith("posthoc") and level.startswith("APTOS"):
                        continue
                    rows.append({**base, "level": level, "rule": rule, "post_hoc": rule.startswith("posthoc"),
                                 "threshold": t, "n": n, "referable_auc": auc, **rates(y, s >= t)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "referral_thresholds_per_seed.csv", index=False)
    summary = (per_seed.groupby(["model", "level", "rule", "post_hoc"], sort=False)
               .agg(seeds=("seed", "count"), threshold_mean=("threshold", "mean"),
                    sensitivity_mean=("sensitivity", "mean"), sensitivity_sd=("sensitivity", "std"),
                    specificity_mean=("specificity", "mean"), specificity_sd=("specificity", "std"),
                    referred_share_mean=("referred_share", "mean"), referable_auc_mean=("referable_auc", "mean"))
               .reset_index())
    summary.to_csv(OUT / "referral_thresholds_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
