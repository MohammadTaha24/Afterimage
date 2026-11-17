"""Build the two 224-pixel datasets the RetiZero queue still waits for (REPRODUCTION_PLAN.md, step 9).

    python -X utf8 -B build_retizero_datasets.py aptos-clahe   [--check-inputs]
    python -X utf8 -B build_retizero_datasets.py messidor      [--check-inputs]

aptos-clahe
    D:/Afterimage_Experiments/datasets/aptos_224_clahe/duplicate_excluded
    The same 2,786 / 357 / 361 images, in the same order, as aptos_224_standard/duplicate_excluded.
    Each image is rebuilt from the original APTOS photo with the existing builder's own functions
    (D:/Afterimage_Experiments/code/prepare_aptos.py, imported, not copied): Otsu crop, black square
    padding, 224 x 224 resize. The rebuilt image must match the existing CLAHE-off 224 image byte for
    byte before CLAHE is applied. CLAHE is then applied last, on the 224-pixel image, with the
    function that made the Qwen 512 CLAHE build (LAB lightness, clip limit 2.0, 8 x 8 tiles).
    That is the Qwen order: resize first, CLAHE last (Qwen3/preprocess_qwen.py resizes to 512 and
    then calls apply_clahe). At 224 pixels an 8 x 8 grid gives 28-pixel tiles instead of 64.

messidor
    D:/Afterimage_Experiments/datasets/messidor_224_archived/clahe_off
    The 1,057 archived Messidor images in the order of the Qwen archived-Messidor cohort
    (finetuning/Qwen3/artifacts/paper_reproduction/data/frozen_run_02/messidor_external.csv).
    Derived from a PROCESSED archive, not raw Messidor-2: each 512-pixel CLAHE-off PNG in
    D:/recovered/DR_Messidor_Only_512_Full_Dataset/Res_512/Messidor_Standard (itself already
    cropped, padded and resized from raw with INTER_AREA) is resized to 224 x 224 with INTER_AREA.
    The original RetiZero Messidor route resized the raw crop directly to 224; this one goes
    raw -> 512 -> 224, so pixels differ slightly. Labels are the archive's derived labels.

Both builds refuse an existing output folder, write every image once, re-read it to confirm the
pixels, and record SHA-256 hashes of every input and output image and of this file.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXPERIMENTS = Path(r"D:\Afterimage_Experiments")
DATA = EXPERIMENTS / "datasets"
BUILDER = EXPERIMENTS / "code/prepare_aptos.py"
SPLITS = ("train", "val", "test")
SIZE = 224

STANDARD = DATA / "aptos_224_standard/duplicate_excluded"
QWEN_512 = DATA / "aptos_512/duplicate_excluded"
CLAHE_OUT = DATA / "aptos_224_clahe/duplicate_excluded"

ARCHIVE = Path(r"D:\recovered\DR_Messidor_Only_512_Full_Dataset\Res_512")
ARCHIVE_MIXED = Path(r"D:\recovered\DR_Aptos_Messidor_Only_Full_Dataset\Res_512")
QWEN_COHORT = REPO / "finetuning/Qwen3/artifacts/paper_reproduction/data/frozen_run_02/messidor_external.csv"
MESSIDOR_OUT = DATA / "messidor_224_archived"


def load_builder():
    spec = importlib.util.spec_from_file_location("prepare_aptos", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pa = load_builder()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def encode_png(pixels):
    ok, png = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError("Cannot encode PNG")
    return png.tobytes()


def provenance():
    return {"builder": str(Path(__file__).resolve()), "builder_sha256": sha(Path(__file__).resolve()),
            "imported_functions_from": str(BUILDER), "imported_builder_sha256": sha(BUILDER),
            "python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__,
            "created_at": datetime.now(timezone.utc).isoformat(), "opencv_threads": 1, "image_size": SIZE}


def run_parallel(function, items, workers=4):
    cv2.setNumThreads(1)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(function, items), 1):
            results.append(result)
            if index % 500 == 0 or index == len(items):
                print(f"  {index}/{len(items)} images", flush=True)
    return results


# ---------------------------------------------------------------- APTOS 224 CLAHE

def qwen_route_check(audit_512):
    """The CLAHE function reproduces the Qwen 512 CLAHE-on build, and agrees with Qwen3/preprocess_qwen.py."""
    sys.path.insert(0, str(REPO / "Qwen3"))
    import preprocess_qwen  # noqa: E402  (pinned app file, imported only)
    checked = []
    for row in audit_512[:25]:
        original = pa.decode(Path(row["original_path"]).read_bytes(), row["original_path"])
        standard_512 = pa.preprocess_standard(original, 512)
        ours = pa.apply_clahe(standard_512)
        app = preprocess_qwen.preprocess_qwen_array(original, clahe=True)
        checked.append({"image_id": row["image_id"],
                        "builder_matches_qwen_build": sha_bytes(ours.tobytes()) == row["on_pixels_sha256"],
                        "app_function_matches_qwen_build": sha_bytes(app.tobytes()) == row["on_pixels_sha256"]})
    return {"images": len(checked),
            "builder_clahe_reproduces_qwen_512_clahe_on": sum(c["builder_matches_qwen_build"] for c in checked),
            "qwen_app_preprocess_reproduces_qwen_512_clahe_on": sum(c["app_function_matches_qwen_build"] for c in checked)}


def build_aptos_clahe(check_only):
    manifest = json.loads((STANDARD / "manifest.json").read_text(encoding="utf-8"))
    audit = {row["image_id"]: row for row in read_csv(STANDARD / "pairing_audit.csv")}
    if sha(STANDARD / "pairing_audit.csv") != manifest["pairing_audit_sha256"]:
        raise ValueError("Standard 224 pairing audit changed")
    tables = {}
    for split in SPLITS:
        record = manifest["branches"]["off"][split]
        if sha(record["path"]) != record["sha256"]:
            raise ValueError(f"Standard 224 CSV changed: {record['path']}")
        tables[split] = read_csv(record["path"])
        qwen = read_csv(QWEN_512 / "clahe_on" / f"{split}.csv")
        if [(r["id_code"], r["label"]) for r in tables[split]] != [(r["id_code"], r["label"]) for r in qwen]:
            raise ValueError(f"Standard 224 and Qwen 512 CLAHE-on {split} order differs")
    rows = [row for split in SPLITS for row in tables[split]]
    missing = [r for r in rows if not Path(audit[r["id_code"]]["original_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} original APTOS photos missing")
    print(f"APTOS 224 CLAHE: {' / '.join(str(len(tables[s])) for s in SPLITS)} images; output {CLAHE_OUT}")
    if CLAHE_OUT.exists():
        raise FileExistsError(f"Choose a new output folder: {CLAHE_OUT}")
    if check_only:
        return
    for split in SPLITS:
        (CLAHE_OUT / "clahe_on/images" / split).mkdir(parents=True, exist_ok=False)

    def one(row):
        binding = audit[row["id_code"]]
        original = Path(binding["original_path"]).read_bytes()
        if sha_bytes(original) != binding["original_sha256"]:
            raise ValueError(f"Original photo changed: {row['id_code']}")
        standard = pa.preprocess_standard(pa.decode(original, binding["original_path"]), SIZE)
        if sha_bytes(standard.tobytes()) != binding["off_pixels_sha256"]:
            raise ValueError(f"Rebuilt CLAHE-off pixels differ from the standard build: {row['id_code']}")
        if sha_bytes(encode_png(standard)) != binding["off_sha256"] or sha(binding["off_path"]) != binding["off_sha256"]:
            raise ValueError(f"Rebuilt CLAHE-off PNG differs from the standard build: {row['id_code']}")
        enhanced = pa.apply_clahe(standard)
        path = CLAHE_OUT / "clahe_on/images" / row["split"] / row["filename"]
        digest = pa.save_png(path, enhanced)  # writes, re-reads and compares pixels
        new_row = {**row, "file_path": str(path), "image_path": str(path)}
        new_audit = {**binding, "on_path": str(path), "on_sha256": digest,
                     "on_pixels_sha256": sha_bytes(enhanced.tobytes())}
        return new_row, new_audit

    results = run_parallel(one, rows)
    on_hashes = [a["on_pixels_sha256"] for _, a in results]
    if len(set(on_hashes)) != len(on_hashes):
        raise ValueError("Two CLAHE images have identical pixels")
    by_split = {split: [r for r, _ in results if r["split"] == split] for split in SPLITS}
    records = {}
    for split in SPLITS:
        path = CLAHE_OUT / "clahe_on" / f"{split}.csv"
        write_csv(path, by_split[split])
        records[split] = {"path": str(path), "sha256": sha(path), "rows": len(by_split[split]),
                          "grades": dict(sorted(Counter(r["label"] for r in by_split[split]).items()))}
    audit_path = CLAHE_OUT / "pairing_audit.csv"
    write_csv(audit_path, [a for _, a in results])
    print("  checking the CLAHE function against the Qwen 512 build ...", flush=True)
    route = qwen_route_check(read_csv(QWEN_512 / "pairing_audit.csv"))
    keep = ("source_manifest", "source_manifest_sha256", "source_splits", "source_label_sha256", "aptos_root",
            "duplicate_policy")
    write_json(CLAHE_OUT / "manifest.json", {
        **{key: manifest[key] for key in keep}, **provenance(),
        "preprocessing": "BGR to RGB, grayscale Otsu foreground crop, black square padding, 224 x 224 resize "
                         "(INTER_AREA), then CLAHE as the last step on the 224-pixel image: one LAB lightness "
                         "pass, clip limit 2.0, grid 8 x 8 (28-pixel tiles at 224; the Qwen build has 64-pixel "
                         "tiles at 512). BGR PNG output.",
        "order_of_operations": "resize, then CLAHE (the Qwen order: Qwen3/preprocess_qwen.py resizes to 512, then applies CLAHE)",
        "paired_standard_manifest": str(STANDARD / "manifest.json"),
        "paired_standard_manifest_sha256": sha(STANDARD / "manifest.json"),
        "standard_images_reproduced_byte_for_byte": len(results),
        "qwen_route_check": route,
        "rows": len(results),
        "branches": {"on": records},
        "pairing_audit_sha256": sha(audit_path),
    })
    print(f"  done; route check {route}")


# ---------------------------------------------------------------- Messidor 224 (archived)

def route_difference_on_aptos(limit=361):
    """How far raw->512->224 is from raw->224, measured on APTOS where both exist (test split)."""
    standard = read_csv(STANDARD / "clahe_off/test.csv")[:limit]
    qwen_off = {r["id_code"]: r for r in read_csv(QWEN_512 / "clahe_off/test.csv")}
    diffs, identical, maxima = [], 0, []
    for row in standard:
        direct = cv2.imread(row["image_path"])
        via = cv2.resize(cv2.imread(qwen_off[row["id_code"]]["image_path"]), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        d = np.abs(direct.astype(np.int16) - via.astype(np.int16))
        diffs.append(float(d.mean()))
        maxima.append(int(d.max()))
        identical += int(not d.any())
    return {"images": len(standard), "cohort": "APTOS duplicate_excluded test (361)",
            "mean_absolute_channel_difference": round(float(np.mean(diffs)), 4),
            "median_of_per_image_max_difference": int(np.median(maxima)),
            "largest_channel_difference": int(max(maxima)), "pixel_identical_images": identical}


def build_messidor(check_only):
    cohort = read_csv(QWEN_COHORT)
    labels = {r["image_path"]: r for r in read_csv(ARCHIVE / "Split_80_10_10/dataset_80_10_10.csv")}
    if len(cohort) != 1057 or len({r["id_code"] for r in cohort}) != 1057 or len(labels) != 1057:
        raise ValueError("Expected 1,057 unique Messidor images")
    for row in cohort:
        archived = labels.get(row["filename"])
        if archived is None or archived["label"] != row["label"] or archived["patient_id"] != row["patient_id"]:
            raise ValueError(f"Archive label or patient differs from the Qwen cohort: {row['id_code']}")
        if not (ARCHIVE / "Messidor_Standard" / row["filename"]).is_file():
            raise FileNotFoundError(row["filename"])
    print(f"Messidor 224 archived: {len(cohort)} images; output {MESSIDOR_OUT}")
    if MESSIDOR_OUT.exists():
        raise FileExistsError(f"Choose a new output folder: {MESSIDOR_OUT}")
    if check_only:
        return
    images = MESSIDOR_OUT / "clahe_off/images"
    images.mkdir(parents=True, exist_ok=False)

    def one(row):
        source = ARCHIVE / "Messidor_Standard" / row["filename"]
        data = source.read_bytes()
        pixels = pa.decode(data, source)
        if pixels.shape != (512, 512, 3):
            raise ValueError(f"Unexpected archived shape {pixels.shape}: {source}")
        mixed = ARCHIVE_MIXED / "Aptos_messidor_external" / row["filename"]
        mixed_same = mixed.is_file() and sha(mixed) == sha_bytes(data)
        clahe_qwen = Path(row["image_path"])
        resized = cv2.resize(pixels, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        path = images / row["filename"]
        digest = pa.save_png(path, resized)
        new_row = {"id_code": row["id_code"], "filename": row["filename"], "file_path": str(path),
                   "image_path": str(path), "label": row["label"], "source_dataset": "messidor",
                   "split": "test", "patient_id": row["patient_id"],
                   "archive_split": labels[row["filename"]]["split"]}
        audit = {"image_id": row["id_code"], "grade": row["label"], "source_path": str(source),
                 "source_sha256": sha_bytes(data), "same_bytes_in_mixed_archive": mixed_same,
                 "qwen_cohort_image_path": str(clahe_qwen), "qwen_cohort_image_sha256": sha(clahe_qwen),
                 "off_path": str(path), "off_sha256": digest, "off_pixels_sha256": sha_bytes(resized.tobytes())}
        return new_row, audit

    results = run_parallel(one, cohort)
    rows = [r for r, _ in results]
    test_path = MESSIDOR_OUT / "clahe_off/test.csv"
    write_csv(test_path, rows)
    audit_path = MESSIDOR_OUT / "source_audit.csv"
    write_csv(audit_path, [a for _, a in results])
    print("  measuring the route difference on APTOS ...", flush=True)
    route = route_difference_on_aptos()
    write_json(MESSIDOR_OUT / "manifest.json", {
        **provenance(),
        "derived_from_processed_archive": True,
        "warning": "NOT raw Messidor-2. Built by resizing an archived, already preprocessed 512-pixel CLAHE-off "
                   "export to 224. Raw Messidor-2 images, the adjudicated grade file and the eye-pairing list are "
                   "not on this PC (docs/paper/MESSIDOR_SOURCES_FOUND.md). Labels are the archive's derived labels "
                   "and cannot be checked against the original adjudication. Replace this build when a verified "
                   "raw cohort exists.",
        "source_archive": str(ARCHIVE / "Messidor_Standard"),
        "source_label_csv": str(ARCHIVE / "Split_80_10_10/dataset_80_10_10.csv"),
        "source_label_csv_sha256": sha(ARCHIVE / "Split_80_10_10/dataset_80_10_10.csv"),
        "source_note": "Every source PNG is byte-identical to the same file in "
                       f"{ARCHIVE_MIXED / 'Aptos_messidor_external'} (column same_bytes_in_mixed_archive).",
        "order_source": str(QWEN_COHORT), "order_source_sha256": sha(QWEN_COHORT),
        "order": "Same image IDs, grades and order as the Qwen archived-Messidor cohort (final_runs/final_01 "
                 "*__messidor_archived), which used the CLAHE-on 512 export of the same archive.",
        "preprocessing": "Archived 512 x 512 CLAHE-off PNG (raw -> Otsu crop -> black square pad -> INTER_AREA "
                         "512, done in Colab by finetuning/preprocessing/ICT302_Messidor_Only (1).ipynb) -> "
                         "cv2.resize to 224 x 224 with INTER_AREA -> PNG. No CLAHE.",
        "difference_from_original_route": "The original RetiZero Messidor notebook used DR_Messidor_Only_224_Full, "
                         "made by resizing the raw crop directly to 224 with INTER_AREA. This build goes "
                         "raw -> 512 -> 224 (two area resamplings plus PNG quantisation at 512), so pixels differ "
                         "slightly. The size of that difference, measured on APTOS where both routes can be built "
                         "from the same originals, is in route_difference_estimate.",
        "route_difference_estimate": route,
        "rows": len(rows), "grades": dict(sorted(Counter(r["label"] for r in rows).items())),
        "archive_split_counts": dict(Counter(r["archive_split"] for r in rows)),
        "branches": {"off": {"test": {"path": str(test_path), "sha256": sha(test_path), "rows": len(rows)}}},
        "source_audit_sha256": sha(audit_path),
        "clahe_on_branch": "not built: the queue's messidor_archived stage scores the CLAHE-off branch only",
    })
    print(f"  done; route difference on APTOS {route}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("build", choices=("aptos-clahe", "messidor"))
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    (build_aptos_clahe if args.build == "aptos-clahe" else build_messidor)(args.check_inputs)


if __name__ == "__main__":
    main()
