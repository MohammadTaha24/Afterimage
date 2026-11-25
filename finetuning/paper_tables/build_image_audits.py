"""Image-level audits for the revision. CPU only; read-only on every input.

  colour      BGR/RGB/LAB check (Dr. Noor item 2, reviewer S3): the app preprocessing (RGB -> LAB) and the
              dataset builder (BGR -> LAB) must give identical pixels on raw APTOS photos; a deliberately
              wrong channel order is run as a negative control to show the check can fail
  near-dup    64-bit DCT perceptual hash of every clean-split image; cross-split pairs at Hamming distance
              <= 4 and the flagged test IDs for the sensitivity analysis in build_revision_analysis.py (S10)
  messidor    exact duplicates in the archived 1,057-image Messidor cohort, with patient IDs and grades;
              optional check against the raw Messidor-2 IMAGES split zip
  examples    contact sheets of representative misclassified images with true and predicted grades (S11)

CSV outputs (image IDs only) go to finetuning/paper_tables/output/revision_analysis/image_audits/.
Image outputs (contact sheets) are derived from APTOS and Messidor, so they go to --image-out, which must be
outside the repository: the dataset terms forbid redistribution.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.fft import dct

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "output" / "revision_analysis" / "image_audits"

LABELS = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- colour conversions

def colour_check(raw_dir: Path, ids: list[str]) -> pd.DataFrame:
    import cv2  # only this audit needs OpenCV, as do the preprocessing modules it compares
    sys.path[:0] = [str(REPO / "Qwen3"), str(REPO / "RetiZero"), str(REPO / "external" / "Afterimage_Experiments" / "code")]
    import prepare_aptos
    from preprocess_qwen import preprocess_qwen_array
    from preprocess_retizero import preprocess_retizero_array
    rows = []
    for image_id in ids:
        bgr = cv2.imread(str(raw_dir / f"{image_id}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(raw_dir / f"{image_id}.png")
        app_qwen = preprocess_qwen_array(bgr, clahe=True)                       # RGB -> LAB route (app)
        builder_qwen = prepare_aptos.apply_clahe(prepare_aptos.preprocess_standard(bgr, 512))  # BGR -> LAB route
        app_qwen_off = preprocess_qwen_array(bgr, clahe=False)
        builder_off = prepare_aptos.preprocess_standard(bgr, 512)
        app_reti = preprocess_retizero_array(bgr)
        builder_reti = prepare_aptos.preprocess_standard(bgr, 224)
        # Negative control: hand OpenCV's BGR array to the RGB->LAB conversion (the classic channel-order bug).
        wrong_lab = cv2.cvtColor(builder_off, cv2.COLOR_RGB2LAB)
        light, a, b = cv2.split(wrong_lab)
        light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
        wrong = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2RGB)
        rows.append({
            "image_id": image_id,
            "qwen_clahe_app_equals_builder": bool(np.array_equal(app_qwen, builder_qwen)),
            "qwen_no_clahe_app_equals_builder": bool(np.array_equal(app_qwen_off, builder_off)),
            "retizero_app_equals_builder": bool(np.array_equal(app_reti, builder_reti)),
            "wrong_channel_order_mean_abs_diff": float(np.abs(wrong.astype(int) - builder_qwen.astype(int)).mean()),
            "wrong_channel_order_identical": bool(np.array_equal(wrong, builder_qwen)),
            "clahe_mean_abs_change": float(np.abs(builder_qwen.astype(int) - builder_off.astype(int)).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- perceptual hash

def phash(path: Path) -> np.ndarray:
    """imagehash.phash equivalent: 32x32 greyscale, 2-D DCT, top-left 8x8, bits above the median."""
    grey = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    coeffs = dct(dct(grey, axis=0, norm=None), axis=1, norm=None)[:8, :8]
    return (coeffs > np.median(coeffs)).ravel()


def near_duplicates(image_dir: Path, splits: dict[str, pd.DataFrame], sha_csv: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames = []
    for split, frame in splits.items():
        f = frame[["id_code", "label"]].copy()
        f["split"] = split
        frames.append(f)
    allf = pd.concat(frames, ignore_index=True)
    hashes = np.stack([phash(image_dir / f"aptos_aptos_{i}_{i}.png") for i in allf["id_code"]])
    test = allf["split"].to_numpy() == "test"
    t_idx, o_idx = np.flatnonzero(test), np.flatnonzero(~test)
    dist = (hashes[t_idx][:, None, :] != hashes[o_idx][None, :, :]).sum(2)
    ti, oi = np.nonzero(dist <= 4)
    pairs = pd.DataFrame({
        "test_id": allf["id_code"].to_numpy()[t_idx[ti]], "test_grade": allf["label"].to_numpy()[t_idx[ti]],
        "other_id": allf["id_code"].to_numpy()[o_idx[oi]], "other_split": allf["split"].to_numpy()[o_idx[oi]],
        "other_grade": allf["label"].to_numpy()[o_idx[oi]], "hamming": dist[ti, oi]}).sort_values(["hamming", "test_id"])
    pairs["grade_differs"] = pairs["test_grade"] != pairs["other_grade"]
    flagged = pd.concat([pd.DataFrame({"image_id": sorted(set(pairs.loc[pairs["hamming"] <= t, "test_id"])), "threshold": t})
                         for t in (0, 2, 4)], ignore_index=True)
    summary = {f"pairs_hamming_le_{t}": int((pairs["hamming"] <= t).sum()) for t in (0, 2, 4)}
    summary.update({f"test_images_hamming_le_{t}": int(flagged["threshold"].eq(t).sum()) for t in (0, 2, 4)})
    summary["images_hashed"] = len(allf)
    # Method check: raw APTOS images with identical file bytes (most were removed from the clean split)
    # must hash to distance 0 when their prepared 512 px images are hashed the same way.
    if sha_csv is not None:
        sha = pd.read_csv(sha_csv)
        groups = sha.groupby("sha256_of_image_file")["image_id"].apply(list)
        known = [(a, b) for g in groups if len(g) > 1 for i, a in enumerate(g) for b in g[i + 1:]]
        d = [int((phash(image_dir / f"aptos_aptos_{a}_{a}.png") != phash(image_dir / f"aptos_aptos_{b}_{b}.png")).sum()) for a, b in known]
        summary.update({"method_check_identical_file_pairs": len(known),
                        "method_check_max_hamming": max(d) if d else None})
    return pairs, flagged, summary


# --------------------------------------------------------------------------- Messidor

class Concatenated(io.RawIOBase):
    """Read a byte-split archive (.zip.001, .002 ...) as one seekable stream."""

    def __init__(self, parts: list[Path]):
        self.parts = [open(p, "rb") for p in parts]
        self.sizes = [p.stat().st_size for p in parts]
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.total = sum(self.sizes)
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.total + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buffer):
        n, out = len(buffer), 0
        while out < n and self.pos < self.total:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            f = self.parts[i]
            f.seek(self.pos - self.starts[i])
            chunk = f.read(min(n - out, self.sizes[i] - (self.pos - self.starts[i])))
            buffer[out:out + len(chunk)] = chunk
            out += len(chunk)
            self.pos += len(chunk)
        return out


def messidor_duplicates(archive_dir: Path, grades_csv: Path | None, raw_parts: list[Path]) -> tuple[pd.DataFrame, dict]:
    files = sorted(archive_dir.glob("messidor_messidor_p_*_PP.png"))
    rows = []
    for f in files:
        m = re.match(r"messidor_messidor_p_(\d+)_(\d{8}_\d+_\d{4}_PP)\.png", f.name)
        data = f.read_bytes()
        pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        rows.append({"image": m.group(2), "patient_id": f"p{m.group(1)}", "file_sha256": sha256_bytes(data),
                     "pixel_sha256": sha256_bytes(pixels.tobytes())})
    frame = pd.DataFrame(rows)
    if grades_csv is not None:
        g = pd.read_csv(grades_csv)
        g["image"] = g["image_id"].str.replace(".png", "", regex=False)
        frame = frame.merge(g[["image", "adjudicated_dr_grade", "adjudicated_dme"]], on="image", how="left")
    dup = frame[frame.duplicated("pixel_sha256", keep=False)].sort_values("pixel_sha256").copy()
    dup["group"] = dup.groupby("pixel_sha256").ngroup()
    facts = {"archived_messidor_images": len(frame), "pixel_duplicate_groups": int(dup["group"].nunique()) if len(dup) else 0,
             "groups_spanning_two_patient_ids": int(dup.groupby("group")["patient_id"].nunique().gt(1).sum()) if len(dup) else 0}
    if raw_parts:
        with zipfile.ZipFile(io.BufferedReader(Concatenated(raw_parts), buffer_size=1 << 20)) as z:
            names = {Path(n).stem: n for n in z.namelist() if not n.endswith("/")}
            facts["raw_zip_files"] = len(names)
            facts["raw_zip_png"] = sum(n.lower().endswith(".png") for n in names.values())
            facts["raw_zip_jpg"] = sum(n.lower().endswith((".jpg", ".jpeg")) for n in names.values())
            raw_sha, raw_pix = {}, {}
            for image in dup["image"]:
                data = z.read(names[image])
                raw_sha[image] = sha256_bytes(data)
                raw_pix[image] = sha256_bytes(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).tobytes())
            dup["raw_file_sha256"] = dup["image"].map(raw_sha)
            dup["raw_pixel_sha256"] = dup["image"].map(raw_pix)
            dup["raw_files_identical_in_group"] = dup.groupby("group")["raw_file_sha256"].transform(lambda s: s.nunique() == 1)
            dup["raw_pixels_identical_in_group"] = dup.groupby("group")["raw_pixel_sha256"].transform(lambda s: s.nunique() == 1)
            facts["duplicate_groups_pixel_identical_in_raw"] = int(dup.groupby("group")["raw_pixels_identical_in_group"].first().sum())
            facts["archived_images_found_in_raw_zip"] = int(frame["image"].isin(names).sum())
    return dup, facts


# --------------------------------------------------------------------------- contact sheets

def sheet(items: list[dict], path: Path, cols: int = 4, size: int = 256):
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * (size + 44)), "white")
    draw = ImageDraw.Draw(canvas)
    for k, item in enumerate(items):
        x, y = (k % cols) * size, (k // cols) * (size + 44)
        canvas.paste(Image.open(item["path"]).convert("RGB").resize((size, size)), (x, y))
        draw.text((x + 4, y + size + 2), item["line1"], fill="black")
        draw.text((x + 4, y + size + 22), item["line2"], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def example_sheets(mis_csv: Path, aptos_dir: Path, messidor_dir: Path, image_out: Path, per_sheet: int = 12) -> pd.DataFrame:
    mis = pd.read_csv(mis_csv, dtype={"image_id": str})
    chosen = []
    for cohort, folder, pattern in (("aptos361", aptos_dir, "aptos_aptos_{0}_{0}.png"), ("messidor", messidor_dir, None)):
        q = mis[(mis["condition"] == "Qwen r16 CLAHE-on lr2e-4") & (mis["cohort"] == cohort)].set_index("image_id")
        r = mis[(mis["condition"] == "RetiZero CLAHE-off") & (mis["cohort"] == cohort)].set_index("image_id")
        both = q.index.intersection(r.index)
        both = [i for i in both if q.loc[i, "n_seeds_wrong"] == 3 and r.loc[i, "n_seeds_wrong"] == 3]
        cand = pd.DataFrame({"image_id": both, "y_true": [int(q.loc[i, "y_true"]) for i in both],
                             "qwen_modal": [int(q.loc[i, "modal_pred"]) for i in both],
                             "retizero_modal": [int(r.loc[i, "modal_pred"]) for i in both],
                             "distance": [max(abs(int(q.loc[i, "modal_pred"]) - int(q.loc[i, "y_true"])),
                                              abs(int(r.loc[i, "modal_pred"]) - int(r.loc[i, "y_true"]))) for i in both],
                             "p_true": [(q.loc[i, "mean_p_true_grade"] + r.loc[i, "mean_p_true_grade"]) / 2 for i in both]})
        # One or two per (true, predicted) cell so the sheet covers the error types, worst first.
        cand = cand.sort_values(["distance", "p_true"], ascending=[False, True])
        mild_normal = cand[(cand["y_true"] == 1) & (cand["qwen_modal"] == 0) & (cand["retizero_modal"] == 0)].head(4)
        if cohort != "messidor":
            mild_normal = mild_normal.head(0)
        rest = cand[~cand["image_id"].isin(mild_normal["image_id"])]
        picks = pd.concat([rest.groupby(["y_true", "qwen_modal"], sort=False).head(2).head(per_sheet - len(mild_normal)), mild_normal])
        for _, p in picks.iterrows():
            name = pattern.format(p["image_id"]) if pattern else next(folder.glob(f"messidor_messidor_p_*_{p['image_id']}.png")).name
            chosen.append({"cohort": cohort, **p.to_dict(), "file": name, "path": folder / name,
                           "line1": f"{p['image_id']}  true {LABELS[int(p['y_true'])]}",
                           "line2": f"Qwen {p['qwen_modal']}  RetiZero {p['retizero_modal']} (all 3 seeds wrong)"})
    for cohort in ("aptos361", "messidor"):
        items = [c for c in chosen if c["cohort"] == cohort]
        if items:
            sheet(items, image_out / f"misclassified_examples_{cohort}.png")
    return pd.DataFrame([{k: v for k, v in c.items() if k not in ("path", "line1", "line2")} for c in chosen])


def near_duplicate_sheet(pairs: pd.DataFrame, image_dir: Path, image_out: Path, max_pairs: int = 24):
    items = []
    for _, p in pairs[pairs["hamming"] <= 2].head(max_pairs).iterrows():
        for side, iid, grade, split in (("test", p["test_id"], p["test_grade"], "test"), ("other", p["other_id"], p["other_grade"], p["other_split"])):
            items.append({"path": image_dir / f"aptos_aptos_{iid}_{iid}.png", "line1": f"{iid} {split} grade {grade}",
                          "line2": f"pHash distance {p['hamming']}"})
    if items:
        sheet(items, image_out / "near_duplicate_pairs_hamming_le_2.png", cols=4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-dir", type=Path, required=True, help="folder with aptos_train.csv, aptos_val.csv, aptos_test.csv (clean split)")
    ap.add_argument("--aptos-512", type=Path, required=True, help="archived 512 px CLAHE-off folder (Aptos_messidor_external)")
    ap.add_argument("--aptos-raw", type=Path, default=None, help="APTOS train_images folder (raw photos) for the colour check")
    ap.add_argument("--image-sha", type=Path, default=None, help="evidence/analysis/image_sha256.csv (method check)")
    ap.add_argument("--messidor-grades", type=Path, default=None)
    ap.add_argument("--messidor-raw-parts", type=Path, nargs="*", default=[], help="IMAGES.zip.001 ... .004")
    ap.add_argument("--misclassified", type=Path, default=OUT.parent / "misclassified_headline.csv")
    ap.add_argument("--image-out", type=Path, default=None, help="folder OUTSIDE the repository for contact sheets")
    args = ap.parse_args(argv)
    if args.image_out is not None and str(args.image_out.resolve()).lower().startswith(str(REPO).lower()):
        raise SystemExit("--image-out must be outside the repository (dataset images must not be committed)")
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {s: pd.read_csv(args.split_dir / f"aptos_{s}.csv", dtype={"id_code": str}) for s in ("train", "val", "test")}
    facts = {}

    if args.aptos_raw is not None:
        colour = colour_check(args.aptos_raw, splits["test"]["id_code"].tolist())
        colour.to_csv(OUT / "colour_conversion_check.csv", index=False)
        facts["colour_opencv_version"] = __import__("cv2").__version__
        facts["colour_images_checked"] = len(colour)
        for c in ("qwen_clahe_app_equals_builder", "qwen_no_clahe_app_equals_builder", "retizero_app_equals_builder"):
            facts[f"colour_{c}"] = int(colour[c].sum())
        facts["colour_negative_control_identical"] = int(colour["wrong_channel_order_identical"].sum())
        facts["colour_negative_control_mean_abs_diff"] = round(float(colour["wrong_channel_order_mean_abs_diff"].mean()), 3)

    pairs, flagged, nd = near_duplicates(args.aptos_512, splits, args.image_sha)
    pairs.to_csv(OUT / "near_duplicate_pairs.csv", index=False)
    flagged.to_csv(OUT / "near_duplicate_test_ids.csv", index=False)
    facts.update({f"neardup_{k}": v for k, v in nd.items()})

    dup, mf = messidor_duplicates(args.aptos_512, args.messidor_grades, args.messidor_raw_parts)
    dup.to_csv(OUT / "messidor_duplicate_pairs.csv", index=False)
    facts.update({f"messidor_{k}": v for k, v in mf.items()})

    if args.image_out is not None:
        near_duplicate_sheet(pairs, args.aptos_512, args.image_out)
        if args.misclassified.is_file():
            ex = example_sheets(args.misclassified, args.aptos_512, args.aptos_512, args.image_out)
            ex.to_csv(OUT / "misclassified_examples_index.csv", index=False)
            facts["examples_exported"] = len(ex)

    (OUT / "image_audit_facts.json").write_text(json.dumps(facts, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(facts, indent=2, default=str))


if __name__ == "__main__":
    main()
