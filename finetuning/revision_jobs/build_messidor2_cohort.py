"""Build the verified Messidor-2 cohort from the raw images (CPU only).

Cohort: every gradable Messidor-2 image (1,744 = 1,748 minus 4 ungradable) with its adjudicated ICDR grade
(Krause et al. 2018). Each eye pair from pairs.csv becomes one exam ID, which the analysis uses as the
resampling cluster. The 1,057 Messidor-Original PNGs of the archived cohort are a subset, flagged in the CSV.

Preprocessing is the APTOS builders' own code (external/Afterimage_Experiments/code/prepare_aptos.py), straight
from each raw photo, so no 512 -> 224 detour:
  qwen_512/clahe_on      Otsu crop, square pad, 512 px, CLAHE       (Qwen's training input)
  qwen_512/clahe_off     Otsu crop, square pad, 512 px              (Qwen CLAHE-off runs)
  retizero_224/clahe_off Otsu crop, square pad, 224 px              (RetiZero's training input)
  retizero_224/clahe_on  Otsu crop, square pad, 224 px, then CLAHE  (RetiZero CLAHE runs; same order as aptos_224_clahe)

Each branch gets test.csv in the column format both scorers read, and manifest.json records the hash of every
input and output file, the builder's code and the OpenCV version. Rerunning with the same inputs is a no-op;
changed inputs stop the build instead of overwriting.

  python -B finetuning/revision_jobs/build_messidor2_cohort.py --raw-dir <01_raw_images> --grades <messidor_data.csv>
         --pairs <pairs.csv> [--out D:/Afterimage_Experiments/datasets/messidor2_verified] [--archive-dir <...>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILDER_CODE = REPO / "external" / "Afterimage_Experiments" / "code"
DEFAULT_OUT = REPO.with_name("Afterimage_Experiments") / "datasets" / "messidor2_verified"
BRANCHES = {"qwen_512/clahe_on": (512, True), "qwen_512/clahe_off": (512, False),
            "retizero_224/clahe_off": (224, False), "retizero_224/clahe_on": (224, True)}
FIELDS = ["id_code", "filename", "file_path", "image_path", "label", "source_dataset", "split", "patient_id",
          "exam_id", "messidor_part", "adjudicated_dme", "in_archived_1057", "source_file", "source_sha256"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def cohort_rows(raw_dir: Path, grades_csv: Path, pairs_csv: Path) -> list[dict]:
    files = {p.name.lower(): p for p in raw_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif")}
    exam_of = {}
    lines = [l.strip() for l in pairs_csv.read_text(encoding="utf-8-sig").splitlines()[1:] if l.strip()]
    for k, line in enumerate(lines, 1):
        for name in line.split(";"):
            exam_of[name.strip().lower()] = f"exam_{k:04d}"
    rows, missing = [], []
    for g in read_csv(grades_csv):
        key = g["image_id"].strip().lower()
        if g["adjudicated_gradable"] != "1":
            continue
        source = files.get(key)
        if source is None:
            missing.append(g["image_id"])
            continue
        stem = Path(source.name).stem
        rows.append({"id_code": stem, "label": str(int(float(g["adjudicated_dr_grade"]))), "source_dataset": "messidor",
                     "split": "test", "exam_id": exam_of.get(key, f"exam_single_{stem}"),
                     "messidor_part": "original_png" if source.suffix.lower() == ".png" else "extension_jpg",
                     "adjudicated_dme": g["adjudicated_dme"], "source_file": str(source)})
    if missing:
        raise SystemExit(f"{len(missing)} graded images are missing from {raw_dir}, e.g. {missing[:3]}")
    for r in rows:
        r["patient_id"] = r["exam_id"]
    return sorted(rows, key=lambda r: r["id_code"])


def preprocess(row: dict, size: int, clahe: bool):
    import cv2
    import prepare_aptos
    bgr = cv2.imread(row["source_file"], cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV cannot read {row['source_file']}")
    out = prepare_aptos.preprocess_standard(bgr, size)
    return prepare_aptos.apply_clahe(out) if clahe else out


def build(args) -> dict:
    sys.path.insert(0, str(BUILDER_CODE))
    import cv2
    import numpy as np
    rows = cohort_rows(args.raw_dir, args.grades, args.pairs)
    if args.limit:
        by_grade = {g: [r for r in rows if r["label"] == str(g)] for g in range(5)}
        rows = sorted([r for g in range(5) for r in by_grade[g][: max(1, args.limit // 5)]], key=lambda r: r["id_code"])
    archived = set()
    if args.archived_ids:
        archived = {l.strip() for l in args.archived_ids.read_text(encoding="utf-8").splitlines() if l.strip()}
    elif args.archive_dir is not None:
        archived = {p.stem.split("_", 4)[-1] for p in args.archive_dir.glob("messidor_messidor_p_*_PP.png")}
    with ThreadPoolExecutor(args.workers) as pool:
        source_sha = dict(zip([r["id_code"] for r in rows], pool.map(lambda r: sha256_file(Path(r["source_file"])), rows)))
    for r in rows:
        r["source_sha256"] = source_sha[r["id_code"]]
        r["in_archived_1057"] = str(r["id_code"] in archived) if archived else ""

    inputs = {"grades": {"path": str(args.grades), "sha256": sha256_file(args.grades)},
              "pairs": {"path": str(args.pairs), "sha256": sha256_file(args.pairs)},
              "raw_images_sha256": hashlib.sha256("".join(f"{r['id_code']}:{r['source_sha256']}\n" for r in rows).encode()).hexdigest()}
    code = {"builder": sha256_file(Path(__file__)), "prepare_aptos": sha256_file(BUILDER_CODE / "prepare_aptos.py")}
    out_manifest = args.out / "manifest.json"
    if out_manifest.is_file():
        old = json.loads(out_manifest.read_text(encoding="utf-8"))
        same_inputs = (old["inputs"]["grades"]["sha256"], old["inputs"]["pairs"]["sha256"], old["inputs"]["raw_images_sha256"]) == \
            (inputs["grades"]["sha256"], inputs["pairs"]["sha256"], inputs["raw_images_sha256"])
        if not same_inputs or old["code"] != code or old["images"] != len(rows):
            raise SystemExit(f"{args.out} already holds a different build. Use another --out to keep both.")
        print(f"Already built: {args.out} ({len(rows)} images). Nothing to do.")
        return old

    branches = {}
    for branch, (size, clahe) in BRANCHES.items():
        folder = args.out / branch / "images"
        folder.mkdir(parents=True, exist_ok=True)

        def write(r, size=size, clahe=clahe, folder=folder):
            path = folder / f"{r['id_code']}.png"
            if not cv2.imwrite(str(path), preprocess(r, size, clahe)):
                raise OSError(f"Could not write {path}")
            return sha256_file(path)

        with ThreadPoolExecutor(args.workers) as pool:
            hashes = list(pool.map(write, rows))
        table = []
        for r, h in zip(rows, hashes):
            path = folder / f"{r['id_code']}.png"
            table.append({**r, "filename": path.name, "file_path": str(path), "image_path": str(path), "sha256": h})
        csv_path = args.out / branch / "test.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
        branches[branch] = {"size": size, "clahe": clahe, "csv": str(csv_path), "csv_sha256": sha256_file(csv_path),
                            "images_sha256": {t["id_code"]: t["sha256"] for t in table}}
        print(f"  built {branch}: {len(table)} images")

    counts = Counter(r["label"] for r in rows)
    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "images": len(rows),
        "grade_counts": [counts[str(g)] for g in range(5)],
        "parts": dict(Counter(r["messidor_part"] for r in rows)), "exams": len({r["exam_id"] for r in rows}),
        "limit": args.limit, "inputs": inputs, "code": code, "opencv": cv2.__version__, "numpy": np.__version__,
        "preprocessing": "prepare_aptos.preprocess_standard (Otsu crop, black square pad, INTER_AREA resize), "
                         "then prepare_aptos.apply_clahe (LAB lightness, clip 2.0, 8x8) for clahe_on branches",
        "branches": branches,
    }
    if args.archive_dir is not None:
        manifest["archive_check"] = archive_check(rows, args.out, args.archive_dir)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def archive_check(rows, out: Path, archive_dir: Path) -> dict:
    """Compare the rebuilt 512 CLAHE-off PNGs with the archived 512 export the earlier Messidor scores used."""
    import cv2
    import numpy as np
    diffs, identical, matched = [], 0, 0
    for r in rows:
        if r["messidor_part"] != "original_png":
            continue
        hits = list(archive_dir.glob(f"messidor_messidor_p_*_{r['id_code']}.png"))
        if not hits:
            continue
        matched += 1
        a = cv2.imread(str(hits[0]), cv2.IMREAD_COLOR).astype(int)
        b = cv2.imread(str(out / "qwen_512/clahe_off/images" / f"{r['id_code']}.png"), cv2.IMREAD_COLOR).astype(int)
        if a.shape != b.shape:
            diffs.append(float("nan"))
            continue
        d = np.abs(a - b)
        identical += int(d.max() == 0)
        diffs.append(float(d.mean()))
    finite = [d for d in diffs if d == d]
    return {"archived_images_compared": matched, "pixel_identical": identical,
            "mean_abs_diff_grey_levels": sum(finite) / len(finite) if finite else None,
            "max_image_mean_abs_diff": max(finite) if finite else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, required=True, help="folder of raw Messidor-2 images (01_raw_images)")
    ap.add_argument("--grades", type=Path, required=True, help="messidor_data.csv (adjudicated grades)")
    ap.add_argument("--pairs", type=Path, required=True, help="pairs.csv (eye pairing)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--archive-dir", type=Path, default=None,
                    help="optional: archived 512 px CLAHE-off folder to compare the rebuilt PNGs with")
    ap.add_argument("--archived-ids", type=Path, default=None, help="optional: text file of the 1,057 archived image IDs")
    ap.add_argument("--limit", type=int, default=0, help="build only about this many images (test run)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    manifest = build(args)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("branches",)}, indent=2))


if __name__ == "__main__":
    main()
