"""Image-quality and source factors behind the headline models' errors (reviewer request: explain misclassifications).

Read-only on the run records and images. Writes CSV tables to finetuning/paper_tables/output/error_analysis/.

What it measures, per test image, on the 512 px CLAHE-off model input (the same crop, pad and resize for APTOS and
Messidor-2, so the two cohorts are comparable):
  brightness   mean luminance inside the field of view (0-255)
  contrast     standard deviation of luminance inside the field of view
  sharpness    variance of the Laplacian of the green channel inside the field of view (higher = sharper)
  red_green    mean red / mean green inside the field of view (colour cast)
  fov_fraction share of the 512 x 512 square covered by the field of view
For Messidor-2 it also records the original file type and resolution (the two acquisition sources of the release)
and the adjudicated macular-oedema flag.

What it reports:
  image_quality_features.csv   one row per image and cohort
  cohort_shift.csv             each feature's median and IQR on APTOS-361 and Messidor-2, and the standardised
                               difference between them (domain shift)
  error_by_factor.csv          for each headline model and each named error (for example true Mild graded No DR on
                               Messidor-2): the error rate within each feature tertile and within each source group,
                               with Wilson 95% intervals, and the Mann-Whitney p for the feature in error vs correct
                               images. Rates pool the three seeds (each image counts once per seed).
Nothing here establishes a cause: it shows which measurable image properties go with the errors.

Usage (repo root, CPU only, about two minutes):
  python -I -B finetuning/paper_tables/build_error_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_revision_analysis as bra  # noqa: E402

EXPERIMENTS = Path(r"D:\Afterimage_Experiments")
MESSIDOR2_RAW = Path(r"D:\Afterimage_data_share\Messidor-2\01_raw_images")
MESSIDOR2_GRADES = Path(r"D:\Afterimage_data_share\Messidor-2\02_adjudicated_grades\messidor_data.csv")
APTOS_512 = EXPERIMENTS / "datasets" / "aptos_512" / "duplicate_excluded" / "clahe_off" / "images" / "test"
MESSIDOR2_512 = EXPERIMENTS / "datasets" / "messidor2_verified" / "qwen_512" / "clahe_off" / "images"
OUT = HERE / "output" / "error_analysis"

MODELS = {"Qwen r16 CLAHE-on lr2e-4": "Qwen", "RetiZero CLAHE-off": "RetiZero"}
COHORTS = {"aptos361": "APTOS-361", "messidor2": "Messidor-2"}
FEATURES = ["brightness", "contrast", "sharpness", "red_green", "fov_fraction"]
# Named errors: (cohort, true grade, predicted grade or None for "any error", label)
ERRORS = [
    ("messidor2", 1, 0, "true Mild graded No DR"),
    ("messidor2", 2, 1, "true Moderate graded Mild"),
    ("messidor2", 0, 1, "true No DR graded Mild"),
    ("messidor2", None, None, "any error"),
    ("aptos361", None, None, "any error"),
]


def image_features(path: Path) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fov = lum > 10          # the preprocessing pads with black
    fov = ndimage.binary_erosion(fov, iterations=4)   # keep the Laplacian away from the FOV edge
    lap = ndimage.laplace(g)
    return {"brightness": lum[fov].mean(), "contrast": lum[fov].std(), "sharpness": lap[fov].var(),
            "red_green": r[fov].mean() / max(g[fov].mean(), 1e-6), "fov_fraction": fov.mean()}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [r for r in bra.discover(EXPERIMENTS) if r["condition"] in MODELS and r["cohort"] in COHORTS]
    bra.load_all(runs)
    assert len(runs) == 12, f"expected 2 models x 2 cohorts x 3 seeds, got {len(runs)}"
    preds = pd.concat([r["frame"][["image_id", "y_true", "y_pred"]].assign(
        model=MODELS[r["condition"]], cohort=r["cohort"], seed=r["seed"]) for r in runs], ignore_index=True)

    # Features on the 512 px CLAHE-off input of every test image.
    rows = []
    for cohort, ids in preds.groupby("cohort")["image_id"].unique().items():
        for image_id in ids:
            path = (APTOS_512 / f"aptos_aptos_{image_id}_{image_id}.png" if cohort == "aptos361"
                    else MESSIDOR2_512 / f"{image_id}.png")
            if not path.is_file():
                raise SystemExit(f"Missing model input: {path}")
            rows.append({"cohort": cohort, "image_id": image_id, **image_features(path)})
    feats = pd.DataFrame(rows)

    # Messidor-2 acquisition source and macular oedema flag.
    # Run records name Messidor-2 images by file stem; the raw files are .png or .JPG.
    grades = pd.read_csv(MESSIDOR2_GRADES).rename(columns={"adjudicated_dme": "dme"})
    grades["raw_file"] = grades["image_id"]
    grades["image_id"] = grades["raw_file"].map(lambda s: Path(s).stem)
    raw_name = dict(zip(grades.image_id, grades.raw_file))
    src = []
    for image_id in feats.loc[feats.cohort == "messidor2", "image_id"]:
        with Image.open(MESSIDOR2_RAW / raw_name[image_id]) as im:
            w, h = im.size
        src.append({"cohort": "messidor2", "image_id": image_id,
                    "file_type": Path(raw_name[image_id]).suffix.lower().lstrip("."), "resolution": f"{w}x{h}"})
    feats = feats.merge(pd.DataFrame(src), on=["cohort", "image_id"], how="left")
    feats = feats.merge(grades[["image_id", "dme"]].assign(cohort="messidor2"), on=["cohort", "image_id"], how="left")
    feats.to_csv(OUT / "image_quality_features.csv", index=False)

    # Domain shift between the two test cohorts.
    shift = []
    a, m = feats[feats.cohort == "aptos361"], feats[feats.cohort == "messidor2"]
    for f in FEATURES:
        pooled_sd = np.sqrt((a[f].var() + m[f].var()) / 2)
        shift.append({"feature": f,
                      "aptos361_median": a[f].median(), "aptos361_q1": a[f].quantile(.25), "aptos361_q3": a[f].quantile(.75),
                      "messidor2_median": m[f].median(), "messidor2_q1": m[f].quantile(.25), "messidor2_q3": m[f].quantile(.75),
                      "standardised_difference": (m[f].mean() - a[f].mean()) / pooled_sd,
                      "mann_whitney_p": stats.mannwhitneyu(a[f], m[f]).pvalue})
    pd.DataFrame(shift).to_csv(OUT / "cohort_shift.csv", index=False)

    # Error rate by feature tertile and by source group, for each named error.
    data = preds.merge(feats, on=["cohort", "image_id"], how="left")
    assert data[FEATURES].notna().all().all()
    out = []
    for model in MODELS.values():
        for cohort, true, pred, label in ERRORS:
            d = data[(data.model == model) & (data.cohort == cohort)]
            if true is not None:
                d = d[d.y_true == true]
                err = (d.y_pred == pred) if pred is not None else (d.y_pred != d.y_true)
            else:
                err = d.y_pred != d.y_true
            d = d.assign(err=err.to_numpy())
            base = {"model": model, "cohort": COHORTS[cohort], "error": label}
            k, n = int(d.err.sum()), len(d)
            out.append({**base, "factor": "all", "group": "all", "n": n, "errors": k, "rate": k / n,
                        "rate_ci_low": wilson(k, n)[0], "rate_ci_high": wilson(k, n)[1]})
            for f in FEATURES:
                cuts = d[f].quantile([1 / 3, 2 / 3]).to_numpy()
                tert = np.digitize(d[f], cuts)
                p = stats.mannwhitneyu(d.loc[d.err, f], d.loc[~d.err, f]).pvalue if 0 < k < n else np.nan
                for t, name in enumerate(("low third", "middle third", "high third")):
                    s = d[tert == t]
                    kk, nn = int(s.err.sum()), len(s)
                    lo, hi = wilson(kk, nn)
                    out.append({**base, "factor": f, "group": name, "n": nn, "errors": kk, "rate": kk / nn if nn else np.nan,
                                "rate_ci_low": lo, "rate_ci_high": hi, "mann_whitney_p_error_vs_correct": p,
                                "median_in_errors": d.loc[d.err, f].median(), "median_in_correct": d.loc[~d.err, f].median()})
            if cohort == "messidor2":
                for factor in ("file_type", "resolution", "dme"):
                    for g, s in d.groupby(factor):
                        kk, nn = int(s.err.sum()), len(s)
                        lo, hi = wilson(kk, nn)
                        out.append({**base, "factor": factor, "group": str(g), "n": nn, "errors": kk, "rate": kk / nn,
                                    "rate_ci_low": lo, "rate_ci_high": hi})
    table = pd.DataFrame(out)
    table.to_csv(OUT / "error_by_factor.csv", index=False)
    print(f"{len(feats)} images, {len(table)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
