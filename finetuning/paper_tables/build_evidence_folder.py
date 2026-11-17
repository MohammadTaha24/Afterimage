"""Copy the text logs and evaluation files behind the paper tables into one evidence folder.

Reads only. Source folders are never modified. Only text files are copied (.json, .csv, .log,
.md, .txt) and only when they are 20 MB or smaller. Model weights, adapters, checkpoints, caches,
image folders and Arrow dataset caches are not copied.

Writes, under the evidence root:
  MANIFEST.csv          relative_path, source_path, bytes, sha256 (one row per copied file)
  MANIFEST_SKIPPED.csv  one row per excluded folder or file group, with the rule that excluded it

Usage:
  python -X utf8 -B finetuning\\paper_tables\\build_evidence_folder.py
"""
import csv
import hashlib
import os
import shutil
import sys
from pathlib import Path

EXP = Path(r"D:\Afterimage_Experiments")
REPO = Path(r"D:\Afterimage_Reproduction")
OUT = EXP / "deliverables_2026-11-17"
EVIDENCE = OUT / "evidence"

TEXT_EXT = {".json", ".csv", ".log", ".md", ".txt"}
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
FILES = [
    (REPO / "docs" / "paper" / "REMAINING_WORK.md", Path("repo") / "docs" / "paper" / "REMAINING_WORK.md"),
    (REPO / "docs" / "paper" / "HEADLINE_VERIFICATION.md", Path("repo") / "docs" / "paper" / "HEADLINE_VERIFICATION.md"),
    (REPO / "docs" / "paper" / "MESSIDOR_KAGGLE_CANDIDATES.md", Path("repo") / "docs" / "paper" / "MESSIDOR_KAGGLE_CANDIDATES.md"),
    (REPO / "finetuning" / "RetiZero" / "RESULTS_RETIZERO.md", Path("repo") / "finetuning" / "RetiZero" / "RESULTS_RETIZERO.md"),
]
for split in ("previous_split", "duplicate_excluded"):
    base = EXP / "datasets" / "aptos_512" / split
    FILES += [(base / "manifest.json", Path("datasets") / "aptos_512" / split / "manifest.json"),
              (base / "pairing_audit.csv", Path("datasets") / "aptos_512" / split / "pairing_audit.csv")]
    for part in ("train", "val", "test"):
        FILES.append((base / "clahe_off" / f"{part}.csv", Path("datasets") / "aptos_512" / split / "clahe_off" / f"{part}.csv"))
FILES += [(EXP / "datasets" / "aptos_512" / "excluded.csv", Path("datasets") / "aptos_512" / "excluded.csv"),
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


def main():
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_hash = sha256(src)
        if not (dst.exists() and dst.stat().st_size == size and sha256(dst) == src_hash):
            shutil.copy2(src, dst)
            if sha256(dst) != src_hash:
                raise RuntimeError(f"copy hash mismatch for {src}")
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
