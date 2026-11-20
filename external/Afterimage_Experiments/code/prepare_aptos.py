"""Build APTOS PNG datasets with the recovered split and optional CLAHE."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "val", "test")
BRANCHES = ("off", "on")
SIZE = 512
KEEP_ORDER = {"test": 0, "val": 1, "train": 2}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def original_rows(aptos_root, manifest_path):
    label_path = aptos_root / "train.csv"
    labels = read_csv(label_path)
    grades = {row["id_code"]: row["diagnosis"] for row in labels}
    if len(grades) != len(labels):
        raise ValueError("The APTOS label file repeats an image ID")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, split_paths, seen, filenames = [], {}, set(), set()
    for split in SPLITS:
        adjacent = manifest_path.with_name(f"{split}.csv")
        path = adjacent if adjacent.is_file() else Path(manifest["csv_paths"][split])
        if sha256(path) != manifest["csv_sha256"][split]:
            raise ValueError(f"Saved split CSV hash differs: {path}")
        subset = read_csv(path)
        if len(subset) != manifest["counts"][split]:
            raise ValueError(f"Unexpected number of photos in {path}")
        for row in subset:
            ident, name = row["id_code"], row["filename"]
            if (Path(ident).name != ident or not ident
                    or Path(name).name != name or not name.lower().endswith(".png")
                    or ident in seen or name.casefold() in filenames
                    or row["split"] != split or row["source_dataset"] != "aptos"
                    or row["label"] not in {"0", "1", "2", "3", "4"}
                    or grades.get(ident) != row["label"]):
                raise ValueError(f"Incorrect, repeated, or unmatched APTOS row: {ident}")
            seen.add(ident)
            filenames.add(name.casefold())
            rows.append(row)
        split_paths[split] = {"path": str(path.resolve()), "sha256": manifest["csv_sha256"][split]}
    if seen != grades.keys():
        raise ValueError("The split lists and original APTOS labels cover different IDs")
    return rows, split_paths, label_path


def archive_records(config_path, off_dir, rows):
    if config_path is None:
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {(row["id_code"], row["split"], row["label"], row["filename"]) for row in rows}
    archived = set()
    for split in SPLITS:
        record = config["csvs"][split]
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Archived split CSV hash differs: {path}")
        archived.update((row["id_code"], row["split"], row["label"], row["filename"]) for row in read_csv(path))
    if archived != expected:
        raise ValueError("Archived Qwen split IDs and grades differ from the source manifest")
    return {
        "config": config_path,
        "on": Path(config["images"]["root"]),
        "off": off_dir,
        "on_sha256": {row["filename"]: row["sha256"] for row in config["images"]["files"]},
    }


def preprocess_standard(bgr, size=SIZE):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # BoundingRect on this binary mask is the notebook's min/max nonzero crop.
    x, y, width, height = cv2.boundingRect(mask)
    if width and height:
        rgb = rgb[y:y + height, x:x + width]
    height, width = rgb.shape[:2]
    difference = abs(height - width)
    first, last = difference // 2, difference - difference // 2
    padding = (first, last, 0, 0) if height < width else (0, 0, first, last)
    square = cv2.copyMakeBorder(rgb, *padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    interpolation = cv2.INTER_AREA if size < square.shape[0] else cv2.INTER_CUBIC
    resized = cv2.resize(square, (size, size), interpolation=interpolation)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


def apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((enhanced, a, b)), cv2.COLOR_LAB2BGR)


def decode(png, path):
    pixels = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f"Cannot decode image: {path}")
    return pixels


def save_png(path, pixels):
    ok, png = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data = png.tobytes()
    path.write_bytes(data)
    written = path.read_bytes()
    if data != written or not np.array_equal(decode(written, path), pixels):
        raise ValueError(f"Saved pixels differ: {path}")
    return hashlib.sha256(written).hexdigest()


def image_path(version, branch, row):
    return version / f"clahe_{branch}" / "images" / row["split"] / row["filename"]


def process_one(row, raw_dir, version, archive, branches, size):
    ident, filename = row["id_code"], row["filename"]
    source = raw_dir / f"{ident}.png"
    original = source.read_bytes()
    standard = preprocess_standard(decode(original, source), size)
    audit = {"image_id": ident, "split": row["split"], "grade": row["label"],
             "original_path": str(source), "original_sha256": hashlib.sha256(original).hexdigest()}
    comparison = {"image_id": ident, "split": row["split"], "grade": row["label"]}
    for branch in branches:
        pixels = standard if branch == "off" else apply_clahe(standard)
        path = image_path(version, branch, row)
        audit[f"{branch}_path"] = str(path)
        audit[f"{branch}_sha256"] = save_png(path, pixels)
        audit[f"{branch}_pixels_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        if archive:
            old_path = archive[branch] / filename
            old = old_path.read_bytes()
            digest = hashlib.sha256(old).hexdigest()
            if branch == "on" and digest != archive["on_sha256"][filename]:
                raise ValueError(f"Archived CLAHE-on hash differs: {old_path}")
            archived_pixels = decode(old, old_path)
            if archived_pixels.shape != pixels.shape:
                raise ValueError(f"Archived image has an unexpected shape: {old_path}")
            diff = np.abs(pixels.astype(np.int16) - archived_pixels.astype(np.int16))
            comparison[f"{branch}_archived_sha256"] = digest
            comparison[f"{branch}_pixel_match"] = not np.any(diff)
            comparison[f"{branch}_changed_channel_values"] = int(np.count_nonzero(diff))
            comparison[f"{branch}_max_channel_difference"] = int(diff.max())
    return row, audit, comparison


def process_rows(rows, raw_dir, version, archive, workers, branches, size):
    cv2.setNumThreads(1)
    if workers <= 1:
        for row in rows:
            yield process_one(row, raw_dir, version, archive, branches, size)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        try:
            for row in rows:
                futures.append(executor.submit(process_one, row, raw_dir, version, archive, branches, size))
            for future in futures:
                yield future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def select_deduplicated(items, branches):
    groups = defaultdict(list)
    for item in items:
        groups[item[1]["original_sha256"]].append(item)
    kept, excluded = [], []
    for group in groups.values():
        conflict = len({row["label"] for row, _, _ in group}) > 1
        representative = None if conflict else min(group, key=lambda item: (KEEP_ORDER[item[0]["split"]], item[0]["id_code"]))
        if representative is not None:
            kept.append(representative)
        for item in group:
            if item is representative:
                continue
            row, audit, _ = item
            excluded.append({"image_id": row["id_code"], "split": row["split"], "grade": row["label"],
                             "original_sha256": audit["original_sha256"], "reason": "conflicting_grade" if conflict else "duplicate",
                             "kept_id": representative[0]["id_code"] if representative is not None else ""})
    for branch in branches:
        hashes = [item[1][f"{branch}_pixels_sha256"] for item in kept]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"The deduplicated CLAHE-{branch} branch still has identical processed pixels")
    return kept, excluded


def write_version(version, items, provenance, policy, branches, copy_images=False):
    audit_rows = []
    rows = {branch: {split: [] for split in SPLITS} for branch in branches}
    for row, original_audit, _ in items:
        audit = dict(original_audit)
        for branch in branches:
            path = image_path(version, branch, row)
            if copy_images:
                shutil.copyfile(Path(original_audit[f"{branch}_path"]), path)
                if sha256(path) != audit[f"{branch}_sha256"]:
                    raise ValueError(f"Copy differs: {path}")
                audit[f"{branch}_path"] = str(path)
            rows[branch][row["split"]].append({**row, "image_path": str(path), "file_path": str(path)})
        audit_rows.append(audit)
    split_records = {}
    for branch in branches:
        split_records[branch] = {}
        for split in SPLITS:
            path = version / f"clahe_{branch}" / f"{split}.csv"
            subset = sorted(rows[branch][split], key=lambda row: row["id_code"])
            write_csv(path, subset)
            split_records[branch][split] = {
                "path": str(path), "sha256": sha256(path), "rows": len(subset),
                "grades": dict(sorted(Counter(row["label"] for row in subset).items())),
            }
    path = version / "pairing_audit.csv"
    write_csv(path, audit_rows)
    write_json(version / "manifest.json", {
        **provenance, "rows": len(items), "duplicate_policy": policy, "branches": split_records,
        "pairing_audit_sha256": sha256(path),
    })
    return split_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aptos-root", type=Path, required=True, help="Directory containing train.csv and train_images")
    parser.add_argument("--split-manifest", type=Path, required=True, help="Recovered APTOS dataset_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the selected dataset build")
    parser.add_argument("--clahe", choices=("both", "on", "off"), default="both")
    parser.add_argument("--size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--archive-config", type=Path, help="Saved Qwen configuration for split provenance")
    parser.add_argument("--archive-off-dir", type=Path, help="Archived directory of CLAHE-off 512-pixel PNGs")
    parser.add_argument("--workers", type=int, default=4, help="Parallel image workers. Use 0 or 1 for serial processing.")
    parser.add_argument("--check-inputs", action="store_true", help="Check the inputs and output path before building")
    args = parser.parse_args()
    if (args.archive_off_dir and not args.archive_config) or args.workers < 0:
        parser.error("Archive comparison requires --archive-config and workers must be nonnegative")
    if args.size == 224 and (args.clahe != "off" or args.archive_off_dir):
        parser.error("The 224-pixel RetiZero build uses --clahe off and a fresh output directory")
    branches = BRANCHES if args.clahe == "both" else (args.clahe,)

    root, split_manifest, output = args.aptos_root.resolve(strict=True), args.split_manifest.resolve(strict=True), args.output.resolve()
    if root == output or root in output.parents or split_manifest.parent == output:
        raise ValueError("Choose an output directory outside the original dataset and split directories")
    rows, split_paths, labels = original_rows(root, split_manifest)
    off_dir = args.archive_off_dir.resolve(strict=True) if args.archive_off_dir else None
    reference = archive_records(args.archive_config.resolve(strict=True), off_dir, rows) if args.archive_config else None
    archive = reference if off_dir else None
    raw_dir = root / "train_images"
    if output.exists():
        raise FileExistsError(f"Choose a new output folder: {output}")
    inputs = [raw_dir / f"{row['id_code']}.png" for row in rows]
    if archive:
        inputs.extend(archive[branch] / row["filename"] for branch in branches for row in rows)
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input files missing. First missing file: {missing[0]}")
    mode = f"{args.workers} parallel workers" if args.workers > 1 else "serial"
    if args.check_inputs:
        print(f"Ready: {len(rows)} originals. {args.size} px, CLAHE {args.clahe}, {mode}. Output: {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    previous, deduplicated = output / "previous_split", output / "duplicate_excluded"
    for version in (previous, deduplicated):
        for branch in branches:
            for split in SPLITS:
                (version / f"clahe_{branch}/images" / split).mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing {len(rows)} original APTOS photos at {args.size} px, CLAHE {args.clahe}, using {mode}.", flush=True)
    items = []
    for index, item in enumerate(process_rows(rows, raw_dir, previous, archive, args.workers, branches, args.size), start=1):
        items.append(item)
        if index % 400 == 0:
            print(f"{index}/{len(rows)} pairs written and verified.", flush=True)
    kept, excluded = select_deduplicated(items, branches)
    fields = ["image_id", "split", "grade", "original_sha256", "reason", "kept_id"]
    write_csv(output / "excluded.csv", excluded, fields)
    provenance = {
        "source_manifest": str(split_manifest), "source_manifest_sha256": sha256(split_manifest),
        "source_splits": split_paths, "source_label_sha256": sha256(labels), "aptos_root": str(root),
        "script_sha256": sha256(Path(__file__).resolve()), "python": sys.version,
        "opencv": cv2.__version__, "numpy": np.__version__, "created_at": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers, "opencv_threads": 1, "image_size": args.size,
        "preprocessing": f"BGR to RGB, grayscale Otsu foreground crop, black square padding, {args.size} x {args.size} resize, BGR output. "
                         "CLAHE on applies one LAB lightness pass (clip limit 2.0, grid 8 x 8).",
    }
    if reference:
        provenance.update({"source_config": str(reference["config"]), "source_config_sha256": sha256(reference["config"])})
    versions = {
        "previous_split": write_version(previous, items, provenance, "All original rows and split assignments retained.", branches),
        "duplicate_excluded": write_version(
            deduplicated, kept, provenance,
            "Exclude all rows from conflicting-grade groups. Retain one copy of each agreeing group, "
            "preferring test, then validation, then training. Use the lowest ID within a split.",
            branches, copy_images=True,
        ),
    }
    report = {
        **provenance, "versions": versions, "exclusions": dict(Counter(row["reason"] for row in excluded)),
        "exclusion_list_sha256": sha256(output / "excluded.csv"),
    }
    if archive:
        comparisons = [comparison for _, _, comparison in items]
        path = output / "archive_pixel_comparison.csv"
        write_csv(path, comparisons)
        report["archive_comparison"] = {
            "path": str(path), "sha256": sha256(path), "rows": len(comparisons),
            "matches": {branch: sum(row[f"{branch}_pixel_match"] for row in comparisons) for branch in branches},
        }
    write_json(output / "dataset_manifest.json", report)
    print(f"Saved {len(branches) * 2} dataset branches. Full split: {len(items)} photos per branch. Duplicate-excluded: {len(kept)}.", flush=True)
    print(f"Manifest: {output / 'dataset_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
