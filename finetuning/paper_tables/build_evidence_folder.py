"""Copy the text logs and evaluation files behind the paper tables into one evidence folder.

Reads only. Source folders are never modified. Only raw data files are copied (.json, .csv, .log,
.txt) and only when they are 20 MB or smaller. Notes files (Markdown), model weights, adapters,
checkpoints, caches, image folders and Arrow dataset caches are not copied. A text file that
points at a notes file is also left out.

Also computes, from the SHA-256 of the APTOS image files and the split label CSVs:
  analysis/image_sha256.csv     one row per labelled APTOS image: split, grade, file SHA-256
  analysis/split_overlap.csv    exact-duplicate and split-size facts derived from that table

Writes, under the evidence root:
  MANIFEST.csv          relative_path, source_path, bytes, sha256 (one row per copied or computed file)
  MANIFEST_SKIPPED.csv  one row per excluded folder or file group, with the rule that excluded it

Usage:
  python -X utf8 -B finetuning\\paper_tables\\build_evidence_folder.py
"""
import csv
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

EXP = Path(r"D:\Afterimage_Experiments")
REPO = Path(r"D:\Afterimage_Reproduction")
OUT = EXP / "deliverables_2026-11-17"
EVIDENCE = OUT / "evidence"

TEXT_EXT = {".json", ".csv", ".log", ".txt"}
NOTES_EXT = {".md"}
NOTES_RULE = "notes file, not raw evidence"
POINTS_AT_NOTES = re.compile(rb"\.md\b", re.I)
APTOS_IMAGES = Path(r"D:\aptos2019-blindness-detection\train_images")
LABEL_CSV = EXP / "datasets" / "aptos_512"
MAX_BYTES = 20 * 1024 * 1024
TOTAL_LIMIT = 3 * 1024 ** 3
SKIP_DIRS = {"cache", "best_lora", "checkpoints", "recovery_artifacts", "dataset_cache",
             "precision-check-vjx2at6q", "__pycache__", "images", "partial-kw0pnalg"}

# (source root, destination below EVIDENCE, extra skipped folder names)
TREES = [
    (EXP / "runs", Path("runs")),
    (EXP / "final_runs" / "final_01", Path("final_runs") / "final_01"),
    (EXP / "retizero_runs" / "retizero_01", Path("retizero_runs") / "retizero_01"),
    (REPO / "finetuning" / "Qwen3" / "artifacts" / "paper_reproduction" / "training_batches" / "run02_reproduction_01",
     Path("repo") / "finetuning" / "Qwen3" / "artifacts" / "paper_reproduction" / "training_batches" / "run02_reproduction_01"),
    (REPO / "finetuning" / "Qwen3" / "artifacts" / "paper_reproduction" / "reference_run_02",
     Path("repo") / "finetuning" / "Qwen3" / "artifacts" / "paper_reproduction" / "reference_run_02"),
    (REPO / "finetuning" / "paper_tables" / "output", Path("repo") / "finetuning" / "paper_tables" / "output"),
]

# single files: (source, destination below EVIDENCE)
FILES = []
for split in ("previous_split", "duplicate_excluded"):
    base = EXP / "datasets" / "aptos_512" / split
    FILES += [(base / "manifest.json", Path("datasets") / "aptos_512" / split / "manifest.json"),
              (base / "pairing_audit.csv", Path("datasets") / "aptos_512" / split / "pairing_audit.csv")]
    for part in ("train", "val", "test"):
        FILES.append((base / "clahe_off" / f"{part}.csv", Path("datasets") / "aptos_512" / split / "clahe_off" / f"{part}.csv"))
FILES += [(EXP / "datasets" / "aptos_512" / "excluded.csv", Path("datasets") / "aptos_512" / "excluded.csv"),
          (EXP / "datasets" / "aptos_512" / "archive_pixel_comparison.csv", Path("datasets") / "aptos_512" / "archive_pixel_comparison.csv"),
          (EXP / "datasets" / "aptos_512" / "dataset_manifest.json", Path("datasets") / "aptos_512" / "dataset_manifest.json"),
          (EXP / "datasets" / "messidor_224_archived" / "manifest.json", Path("datasets") / "messidor_224_archived" / "manifest.json"),
          (EXP / "datasets" / "messidor_224_archived" / "source_audit.csv", Path("datasets") / "messidor_224_archived" / "source_audit.csv")]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build_split_analysis():
    """Hash every labelled APTOS image and derive the split-overlap facts from the hashes and label CSVs."""
    def labels(folder):
        out = {}
        for part in ("train", "val", "test"):
            for r in read_csv_rows(LABEL_CSV / folder / "clahe_off" / f"{part}.csv"):
                if r["id_code"] in out:
                    raise RuntimeError(f"{r['id_code']} appears twice in {folder}")
                out[r["id_code"]] = (part, int(r["label"]))
        return out

    orig = labels("previous_split")
    clean = labels("duplicate_excluded")
    for i, (part, g) in clean.items():
        if i not in orig or orig[i][1] != g:
            raise RuntimeError(f"clean split image {i} is not an original-split image with the same grade")
    sha = {}
    for i in sorted(orig):
        sha[i] = sha256(APTOS_IMAGES / f"{i}.png")
    img_rows = [(i, orig[i][0], orig[i][1], clean[i][0] if i in clean else "", sha[i], (APTOS_IMAGES / f"{i}.png").stat().st_size) for i in sorted(orig)]
    write_csv(EVIDENCE / "analysis" / "image_sha256.csv", ["image_id", "original_split", "grade", "clean_split", "sha256_of_image_file", "file_bytes"], img_rows)

    by_hash = {}
    for i, h in sha.items():
        by_hash.setdefault(h, []).append(i)
    dup_groups = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    conflict_groups = sum(1 for ids in dup_groups.values() if len({orig[i][1] for i in ids}) > 1)
    test_ids = [i for i in orig if orig[i][0] == "test"]
    leak = train_hit = val_only = grade_conflict = 0
    for t in test_ids:
        others = [o for o in by_hash[sha[t]] if o != t and orig[o][0] in ("train", "val")]
        if others:
            leak += 1
            if any(orig[o][0] == "train" for o in others):
                train_hit += 1
            else:
                val_only += 1
            if any(orig[o][1] != orig[t][1] for o in others):
                grade_conflict += 1
    tt_groups = len({sha[t] for t in test_ids if sum(1 for o in by_hash[sha[t]] if orig[o][0] == "test") > 1})
    removed = [i for i in orig if i not in clean]
    removed_test = [i for i in removed if orig[i][0] == "test"]
    rem_conflict = sum(1 for i in removed_test if len({orig[o][1] for o in by_hash[sha[i]]}) > 1)
    rem_tt = len(removed_test) - rem_conflict
    clean_test = [i for i in clean if clean[i][0] == "test"]
    ct_train = ct_val = 0
    for t in clean_test:
        others = [o for o in by_hash[sha[t]] if o != t]
        if any(orig[o][0] == "train" for o in others):
            ct_train += 1
        elif any(orig[o][0] == "val" for o in others):
            ct_val += 1

    def n(lab, sp):
        return sum(1 for v in lab.values() if v[0] == sp)

    facts = [
        ("labelled_images", len(orig), "images in the original split label CSVs (train, val and test), each hashed from its image file"),
        ("duplicate_groups", len(dup_groups), "groups of two or more images whose image files have identical SHA-256"),
        ("images_in_duplicate_groups", sum(len(v) for v in dup_groups.values()), "images that belong to those groups"),
        ("duplicate_groups_with_conflicting_grades", conflict_groups, "groups whose members do not all carry the same grade"),
        ("original_train_n", n(orig, "train"), "rows in previous_split/clahe_off/train.csv"),
        ("original_val_n", n(orig, "val"), "rows in previous_split/clahe_off/val.csv"),
        ("original_test_n", n(orig, "test"), "rows in previous_split/clahe_off/test.csv"),
        ("clean_train_n", n(clean, "train"), "rows in duplicate_excluded/clahe_off/train.csv"),
        ("clean_val_n", n(clean, "val"), "rows in duplicate_excluded/clahe_off/val.csv"),
        ("clean_test_n", n(clean, "test"), "rows in duplicate_excluded/clahe_off/test.csv"),
        ("original_test_with_copy_in_train_or_val", leak, "original test images that have an identical image file in train or validation"),
        ("original_test_with_copy_in_train", train_hit, "of those, a copy is in train"),
        ("original_test_with_copy_only_in_val", val_only, "of those, the only copies are in validation"),
        ("original_test_with_copy_of_different_grade", grade_conflict, "of those, a copy carries a different grade"),
        ("duplicate_groups_inside_original_test", tt_groups, "groups with two or more members inside the original test set"),
        ("removed_from_original_test", len(removed_test), "original test images absent from the clean test set"),
        ("removed_from_test_conflicting_grade", rem_conflict, "removed test images whose duplicate group has more than one grade"),
        ("removed_from_test_as_duplicate_pair", rem_tt, "removed test images that are the second copy of an agreeing test-test pair"),
        ("removed_total", len(removed), "original-split images absent from the clean split"),
        ("removed_from_train", sum(1 for i in removed if orig[i][0] == "train"), "removed images that were in the original train split"),
        ("removed_from_val", sum(1 for i in removed if orig[i][0] == "val"), "removed images that were in the original validation split"),
        ("clean_test_with_copy_in_original_train", ct_train, "clean test images with an identical image file in the original train split"),
        ("clean_test_with_copy_only_in_original_val", ct_val, "clean test images with no copy in the original train split but one in the original validation split"),
    ]
    write_csv(EVIDENCE / "analysis" / "split_overlap.csv", ["fact", "value", "definition"], facts)
    out = []
    for rel_path in ("analysis/image_sha256.csv", "analysis/split_overlap.csv"):
        p = EVIDENCE / rel_path
        out.append((rel_path, "computed from the APTOS image files in D:\\aptos2019-blindness-detection\\train_images and the split label CSVs", p.stat().st_size, sha256(p)))
    return out


def main():
    if EVIDENCE.exists():
        if EVIDENCE.resolve().parent != OUT.resolve() or EVIDENCE.name != "evidence":
            raise RuntimeError("refusing to clear an unexpected folder")
        shutil.rmtree(EVIDENCE)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    copied = []      # (rel, src, bytes, sha)
    skipped = {}     # (group, rule) -> [files, bytes]

    def note_skip(group, rule, size):
        rec = skipped.setdefault((group, rule), [0, 0])
        rec[0] += 1
        rec[1] += size

    def copy_one(src, dst_rel, group):
        size = src.stat().st_size
        dst = EVIDENCE / dst_rel
        data = src.read_bytes()
        if POINTS_AT_NOTES.search(data):
            note_skip(group, "text refers to a notes file by name; left out", size)
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_hash = hashlib.sha256(data).hexdigest()
        dst.write_bytes(data)
        if sha256(dst) != src_hash:
            raise RuntimeError(f"copy hash mismatch for {src}")
        shutil.copystat(src, dst)
        copied.append((dst_rel.as_posix(), str(src), size, src_hash))

    for src_root, dst_root in TREES:
        if not src_root.exists():
            note_skip(str(src_root), "source folder not found", 0)
            continue
        for dp, dn, fn in os.walk(src_root):
            dpath = Path(dp)
            keep = []
            for d in dn:
                if d in SKIP_DIRS:
                    total = 0
                    count = 0
                    for dp2, _, fn2 in os.walk(dpath / d):
                        for f2 in fn2:
                            count += 1
                            total += (Path(dp2) / f2).stat().st_size
                    rel_dir = (dpath / d).relative_to(src_root).as_posix()
                    skipped[(f"{src_root.name}/{rel_dir}", f"folder name '{d}' is not copied (weights, caches, images)")] = [count, total]
                else:
                    keep.append(d)
            dn[:] = keep
            for f in fn:
                src = dpath / f
                size = src.stat().st_size
                ext = src.suffix.lower()
                rel = src.relative_to(src_root)
                group = f"{src_root.name}/{rel.parent.as_posix()}"
                if ext in NOTES_EXT:
                    note_skip(group, NOTES_RULE, size)
                    continue
                if ext not in TEXT_EXT:
                    note_skip(group, f"extension {ext or '(none)'} is not a text log or table", size)
                    continue
                if size > MAX_BYTES:
                    note_skip(group, "larger than 20 MB", size)
                    continue
                copy_one(src, dst_root / rel, group)

    for src, dst_rel in FILES:
        if not src.exists():
            note_skip(str(src), "source file not found", 0)
            continue
        copy_one(src, dst_rel, str(src.parent))

    copied.extend(build_split_analysis())
    total_bytes = sum(c[2] for c in copied)
    if total_bytes > TOTAL_LIMIT:
        print(f"WARNING: total {total_bytes} bytes exceeds 3 GB; largest per-step logs would need to be skipped", file=sys.stderr)

    copied.sort()
    with open(EVIDENCE / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "source_path", "bytes", "sha256"])
        for row in copied:
            w.writerow(row)

    with open(EVIDENCE / "MANIFEST_SKIPPED.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder_or_group", "rule", "files", "bytes"])
        for (group, rule), (n, b) in sorted(skipped.items()):
            w.writerow([group, rule, n, b])

    print(f"files copied: {len(copied)}")
    print(f"bytes copied: {total_bytes} ({total_bytes / 1024 ** 2:.1f} MiB)")
    print(f"skipped groups: {len(skipped)}; skipped bytes: {sum(v[1] for v in skipped.values())}")


if __name__ == "__main__":
    main()
