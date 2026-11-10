"""One-shot, local, CPU-only evidence audit. Importing this file does no work.

Only main() writes files, and only phase1_report.md and phase1_run.log beside
this worker. Old Python/notebook code is read as text/AST, never imported or run.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import traceback


OUTPUT_DIR = Path(__file__).resolve().parent
REFERENCE = "Qwen3-VL-8B Retraining 2"
REPORTED = "80.65% accuracy; 0.8915 QWK; 0.9362 macro ROC-AUC"
EXPECTED_APTOS = {"train": [1444, 296, 799, 154, 236],
                  "val": [180, 37, 100, 20, 29], "test": [181, 37, 100, 19, 30]}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SKIP_DIRS = {".git", ".venv", "venv", "env", ".verification", "site-packages",
             "node_modules", "__pycache__", ".ipynb_checkpoints", ".cache"}
MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 1_000_000
ID_COLS = ("original_image_id", "source_image_id", "image_id", "id_code", "img_id",
           "image_file", "filename", "image_name", "image", "img_path", "image_path",
           "file_path", "path", "file")
LABEL_COLS = ("y_true", "true_label", "true_grade", "ground_truth", "adjudicated_dr_grade",
              "diagnosis", "dr_grade", "label", "grade", "target", "class", "y")
PRED_COLS = ("y_pred", "pred_label", "predicted_label", "pred_grade", "predicted_grade",
             "prediction", "predicted", "pred")
RAW_COLS = ("raw_output", "raw_response", "generated_text", "gen_text", "response",
            "qwen_response", "raw_text", "output_text")
RELEVANT = re.compile(r"qwen|aptos|messidor|preproc|clahe|retrain|adapter|trainer|train|"
                      r"run_info|manifest|eval|predict|score|split|label|audit\.json", re.I)
SPLIT_RE = re.compile(r"(?<![a-z])(train(?:ing)?|val(?:idation)?|valid|test(?:ing)?|external)(?![a-z])", re.I)
ANSWER_PATTERN = r"<answer>\s*([0-4])\s*-\s*.*?</answer>"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def dataset(value):
    value = norm(value)
    aptos = "aptos" in value
    messidor = "messidor" in value
    return "aptos" if aptos and not messidor else "messidor2" if messidor and not aptos else "unknown"


def split_name(value):
    value = norm(value)
    return {"training": "train", "validation": "val", "valid": "val",
            "testing": "test", "external": "external"}.get(value, value)


def filename_split(path):
    found = {split_name(m.group()) for m in SPLIT_RE.finditer(path.stem)}
    return next(iter(found)) if len(found) == 1 else ""


def grade(value):
    text = str(value).strip()
    if re.fullmatch(r"[0-4](?:\.0+)?", text):
        return int(float(text))
    return {"normal": 0, "no_dr": 0, "no": 0, "mild": 1, "mild_dr": 1,
            "moderate": 2, "moderate_dr": 2, "severe": 3, "severe_dr": 3,
            "proliferative": 4, "proliferative_dr": 4}.get(norm(text))


def image_id(value, source):
    # Preserve leading zeros and case. Never use row position, patient ID, fuzzy
    # matching, or strip Messidor patient prefixes to manufacture an ID match.
    text = str("" if value is None else value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(text).suffix.lower()
    if suffix in IMAGE_EXT:
        text = text[:-len(suffix)]
    if source == "aptos" and text.startswith("aptos_"):
        text = text[len("aptos_"):]
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def md_table(headers, rows):
    rows = list(rows)
    if not rows:
        return "_None available._\n"
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows]) + "\n"


def examples(values, limit=8):
    values = sorted(map(str, values))
    return ", ".join(values[:limit]) + (f" (+{len(values) - limit} more)" if len(values) > limit else "")


@dataclass
class Evidence:
    path: Path
    kinds: set = field(default_factory=set)
    size: int = 0
    sha: str = "not hashed"
    note: str = ""


@dataclass
class Table:
    path: Path
    location: str
    rows: list
    fields: tuple
    sha: str
    meta: dict = field(default_factory=dict)


@dataclass
class Cohort:
    source: str
    split: str
    family: str
    paths: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    duplicate_count: int = 0


class Audit:
    def __init__(self, roots, log):
        self.roots = roots
        self.log_file = log
        self.evidence = []
        self.tables = []
        self.documents = {}
        self.sources = []
        self.images = defaultdict(list)
        self.image_index = defaultdict(list)
        self.warnings = []
        self.checks = []
        self.findings = []
        self.results = []
        self.fallbacks = []
        self.adapters = []
        self.cohorts = []
        self.verified = []
        self.status = "PARTIAL — audit started"
        self.started = now()
        self.excel = []
        self.fallback_coverage = []

    def progress(self, text):
        line = f"[{now()}] {text}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def warn(self, text):
        self.warnings.append(text)
        self.progress("PARTIAL: " + text)

    def inside(self, path):
        return any(path == root or root in path.parents for root in self.roots)

    def linked(self, table, key, required_hash=True):
        value = table.meta.get(key)
        if not value:
            return None
        path = Path(str(value)).expanduser()
        path = (table.path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if not self.inside(path) or not path.is_file():
            raise ValueError(f"{key}: unavailable or outside configured roots: {path}")
        if path.stat().st_size > MAX_BYTES:
            raise ValueError(f"{key}: over 64 MiB evidence limit: {path}")
        expected = table.meta.get(key + "_sha256")
        if required_hash and (not expected or digest(path) != expected):
            raise ValueError(f"{key}: missing/mismatched SHA-256 for {path}")
        return path

    def report(self):
        target = OUTPUT_DIR / "phase1_report.md"
        parts = ["# Phase 1 diabetic retinopathy evidence audit\n",
                 f"Status: **{self.status}**  \nStarted: {self.started}; report updated: {now()}\n",
                 "## Accepted reference (reported, not recomputed)\n",
                 f"Dr. Noor accepted **{REFERENCE}**. Reported APTOS test: **{REPORTED}**. "
                 "These supplied figures are not validation targets, and are never substituted for missing results. "
                 "Run identity is not inferred from a similar metric, rank, or folder name.\n",
                 "## Scope and methods\n",
                 "Local CPU audit only. No old code or notebook cells are executed/imported; no model, image pixels, "
                 "GPU libraries, servers, installs, network access, or subprocesses are used by this worker. "
                 "Only this report and phase1_run.log are written. Roots:\n\n" +
                 "\n".join(f"- `{root}`" for root in self.roots) + "\n",
                 "Directories skipped: " + ", ".join(sorted(SKIP_DIRS)) +
                 "; the worker's own audit directory and symlink/junction directories are skipped. "
                 "Archives are listed but not extracted. Files over 64 MiB and tables over 1,000,000 rows "
                 "are not consumed; such omissions make the report partial. Historical absolute paths "
                 "are recorded, never followed outside the configured roots.\n",
                 "## Evidence inventory\n",
                 md_table(["Path", "Evidence type", "Bytes", "SHA-256", "Read status"],
                          [(e.path, ", ".join(sorted(e.kinds)), e.size, e.sha, e.note) for e in self.evidence]),
                 "## Parsed table schemas\n",
                 md_table(["Location", "Rows", "Columns"], [(t.location, len(t.rows), ", ".join(t.fields)) for t in self.tables]),
                 "## Original image availability for a future CLAHE comparison\n",
                 "Availability below means filename/path presence, not pixel validation. 'raw/original' is "
                 "a directory naming clue, not proof of an untouched source. Matched IDs may have several versions. "
                 "A future comparison needs verified original-to-processed mapping and consistent preprocessing.\n",
                 md_table(["Folder", "Images", "Path-based assessment"],
                          [(p, len(paths), self.image_kind(p)) for p, paths in sorted(self.images.items())]),
                 "## Completed integrity checks\n",
                 md_table(["Check", "Evidence", "Result"], self.checks),
                 "## Label cohorts and class counts\n",
                 "Counts are ordered grades 0, 1, 2, 3, 4. Duplicate rows are counted and block verification; "
                 "they are never silently deduplicated for metrics. Separate split definitions remain separate.\n",
                 md_table(["Dataset/split", "Paths", "Unique IDs", "Counts", "Issues"],
                          [(f"{c.source}/{c.split}", "; ".join(map(str, c.paths)), len(c.labels),
                            [Counter(c.labels.values())[i] for i in range(5)], "; ".join(c.issues) or "none") for c in self.cohorts]),
                 "APTOS counts declared in the reference reproduction helper (not independently established here): "
                 + json.dumps(EXPECTED_APTOS) + ". Matching counts alone do not establish identity.\n",
                 "## Recomputed results — verified exports only\n",
                 "Accuracy is exact agreement. QWK uses fixed grades 0–4 and quadratic weights; it is undefined "
                 "when expected weighted disagreement is zero. Macro ROC-AUC is the arithmetic mean of five "
                 "one-vs-rest, rank-based AUCs with average ranks for ties; all classes need positives and negatives. "
                 "Hard labels, one-hot predictions, and a single confidence value never supply ROC-AUC.\n",
                 md_table(["Prediction export", "Join / verification", "Recomputed metrics", "Reference linkage / AUC gate"], self.results),
                 "## Saved-code findings (static evidence, not confirmed historical cases)\n",
                 "The excerpts below identify parser/fallback, augmentation, and color-conversion paths. "
                 "They do not establish which revision produced a past artifact. Absence of a keyword is "
                 "not proof of training-only augmentation or correct color handling.\n",
                 md_table(["Topic", "Source location", "Finding"], self.findings),
                 "## Confirmed fallback cases and Messidor-2 Mild → Normal contribution\n",
                 "A confirmed saved fallback requires a unique image ID, an independently joined valid label, "
                 "an explicit saved parse_ok=false, raw-output column, and exported y_pred=0. "
                 "Without the saved flag, only an explicitly hash-linked, recognized parser/producer can "
                 "establish fallback. Current parser replay and invalid-format candidates are reported separately. "
                 "Counts from different exports may refer to the same images; do not sum across runs.\n",
                 md_table(["Export", "Coverage and confirmed cases", "Mild → Normal", "Uncertainty / replay"], self.fallbacks),
                 "## Adapter and rank-16 CLAHE control handoff\n",
                 md_table(["Adapter configuration", "Recorded properties / weights", "Decision"], self.adapters),
                 "## Missing inputs and handoff\n",
                 self.handoff(),
                 "\n## Read errors, ambiguous inputs, and limitations\n",
                 "\n".join(f"- {w}" for w in self.warnings) + "\n" if self.warnings else "_No read errors recorded._\n",
                 "\nExecution log: `" + str(OUTPUT_DIR / "phase1_run.log") + "`. "
                 "This report is refreshed between stages, so an interruption may leave a partial report. "
                 "A forced termination can stop before the next refresh. No inference or rescoring was performed.\n"]
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(parts))
            stream.flush()

    @staticmethod
    def image_kind(path):
        text = norm(str(path))
        if re.search(r"clahe|preproc|processed|res_?\d+|512|224|384", text):
            return "processed/CLAHE candidate; not verified originals"
        if re.search(r"(?:^|_)(raw|original|originals|unprocessed|train_images|test_images)(?:_|$)", text):
            return "raw/original candidate; verify provenance before CLAHE rebuild"
        return "image files available; original status unknown"

    def handoff(self):
        kinds = set().union(*(e.kinds for e in self.evidence)) if self.evidence else set()
        lines = []
        for title, kind in [("APTOS train/val/test lists", "split/label candidate"),
                            ("prediction exports", "prediction candidate"),
                            ("raw Qwen responses", "raw response candidate"),
                            ("genuine per-grade scores with class order and producer provenance", "grade-score candidate"),
                            ("training histories", "training record"), ("adapter configuration", "adapter configuration")]:
            lines.append(f"- {title}: {'candidate evidence inventoried; see verification above' if kind in kinds else 'MISSING — add local evidence under an input root'}.")
        lines.extend([
            "- Messidor-2: supply the adjudicated image-level labels and exact evaluated cohort; a patient pairing file alone is insufficient. Preserve original image IDs when filenames were renamed.",
            "- Baseline: supply provenance tying exports and adapter to Qwen3-VL-8B Retraining 2. Unidentified runs remain separate from the accepted reference.",
            "- Reuse: verified labels/splits and ID-joined hard predictions can support accuracy/QWK. Saved raw responses can be reparsed in a later, separately authorized step. Authentic grade-score vectors can support AUC only with verified order and producer; labels alone cannot recover AUC.",
            "- Rescore later: if scores are absent or synthetic, original images plus the exact adapter, processor, prompt and preprocessing revision are needed for new score generation. This audit does not do that work.",
            "- CLAHE: rank alone does not establish a reusable control. Check exact training split/labels, image lineage, preprocessing/channel order, training-only augmentation, checkpoint selection and run identity. Decisions above are deliberately conservative.",
            "- CSV is preferred. Export Excel sheets to CSV if the compatible local reader is unavailable. Preserve IDs as text, labels, split, raw_output, parse_ok, y_pred and five score columns; do not reconstruct probabilities from predictions.",
            "- Optional provenance sidecar `<export filename>.audit.json` is described in README.md. It supplies explicit links; it cannot by itself authenticate the underlying records.",
        ])
        if not self.images:
            lines.append("- MISSING: no loose original image files discovered. Archives and model weights are not original-image evidence.")
        if not self.verified:
            lines.append("- MISSING: no fully verified test export. Review each blocked join and add exact split lists, labels and provenance; no baseline metrics were invented.")
        for source, parts in (("aptos", ("train", "val", "test")), ("messidor2", ("test", "external"))):
            found = {c.split for c in self.cohorts if c.source == source and c.labels and not c.issues}
            missing = set(parts) - found if source == "aptos" else (set() if found & set(parts) else {"test/external"})
            if missing:
                lines.append(f"- MISSING/INVALID {source} independent label lists: {', '.join(sorted(missing))}. Add CSVs with image IDs, grades, dataset and split.")
        if not self.adapters:
            lines.append("- Rank-16 CLAHE control: **lacks enough evidence to decide**; no readable adapter configuration was found.")
        lines.extend("- Excel: " + note for note in self.excel)
        return "\n".join(lines) + "\n"

    def inventory(self):
        seen = set()
        def skip_folder(path):
            if path.name.lower() in SKIP_DIRS:
                return True
            try:
                # Covers NTFS junctions on Python versions before is_junction().
                attributes = getattr(path.lstat(), "st_file_attributes", 0)
                return (path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
                        or path.resolve() == OUTPUT_DIR)
            except OSError as error:
                self.warn(f"Cannot inspect directory {path}: {error}")
                return True
        for root in self.roots:
            if not root.is_dir():
                self.warn(f"Input root is missing/not a directory: {root}")
                continue
            self.progress(f"Inventorying {root}")
            def walk_error(error):
                self.warn(f"Directory not readable: {error}")
            for directory, folders, files in os.walk(root, followlinks=False, onerror=walk_error):
                parent = Path(directory)
                folders[:] = sorted(d for d in folders if not skip_folder(parent / d))
                for name in sorted(files):
                    path = parent / name
                    if path.is_symlink():
                        continue
                    try:
                        path = path.resolve()
                        if not self.inside(path) or path in seen or path.parent == OUTPUT_DIR:
                            continue
                        seen.add(path)
                        suffix = path.suffix.lower()
                        if suffix in IMAGE_EXT:
                            self.images[path.parent].append(path)
                            self.image_index[image_id(path.name, "unknown")].append(path)
                            continue
                        relevant = root != self.roots[0] or bool(RELEVANT.search(str(path)))
                        archive = suffix in {".zip", ".7z", ".tar", ".gz"} or bool(re.search(r"\.(?:zip|7z)\.\d+$", name, re.I))
                        if suffix not in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson",
                                          ".py", ".ipynb", ".log", ".txt", ".md"} and not archive:
                            continue
                        if suffix not in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".log"} and not relevant and not archive:
                            continue
                        kinds = set()
                        text = norm(str(path))
                        if re.search(r"split|label|dataset|messidor_data|train|test|val", text):
                            kinds.add("split/label candidate")
                        if "pred" in text:
                            kinds.add("prediction candidate")
                        if re.search(r"raw|response|generation", text):
                            kinds.add("raw response candidate")
                        if re.search(r"score|prob|logit", text):
                            kinds.add("grade-score candidate")
                        if suffix == ".log" or re.search(r"trainer_state|run_info|training_config|run_config|history|train.*log", name, re.I):
                            kinds.add("training record")
                        if name.lower() == "adapter_config.json":
                            kinds.add("adapter configuration")
                        if suffix in {".py", ".ipynb"}:
                            kinds.add("saved source" if suffix == ".py" else "saved notebook (never executed)")
                        if archive:
                            kinds.add("archive (not extracted)")
                        e = Evidence(path, kinds or {"local evidence candidate"}, path.stat().st_size)
                        self.evidence.append(e)
                    except OSError as error:
                        self.warn(f"Cannot inventory {path}: {error}")
                if len(seen) and len(seen) % 5000 == 0:
                    self.progress(f"Inventory progress: {len(seen)} unique local files")
        self.progress(f"Found {len(self.evidence)} evidence candidates and {sum(map(len, self.images.values()))} images")

    def add_table(self, e, rows, location, metadata=None):
        if len(rows) > MAX_ROWS:
            raise ValueError("table exceeds row limit; export a smaller evidence table")
        normalized = []
        fields = set()
        for raw in rows:
            if not isinstance(raw, dict):
                raise ValueError("expected a table of objects / headered records")
            keys = [norm(k) for k in raw]
            if len(set(keys)) != len(keys) or not all(keys):
                raise ValueError("empty/duplicate normalized column names")
            row = {norm(k): "" if v is None else v for k, v in raw.items()}
            if all(str(v).strip() == "" for v in row.values()):
                continue
            fields.update(row)
            normalized.append(row)
        fields = tuple(sorted(fields))
        t = Table(e.path, location, normalized, fields, e.sha, dict(metadata or {}))
        # A sidecar is an explicit link, never executed. A binding hash is needed
        # before its dataset/run/provenance claims influence verification.
        sidecar = Path(str(e.path) + ".audit.json")
        if sidecar.is_file() and self.inside(sidecar.resolve()):
            if sidecar.stat().st_size > MAX_BYTES:
                raise ValueError(f"sidecar over 64 MiB limit: {sidecar}")
            extra = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            if not isinstance(extra, dict) or extra.get("prediction_export_sha256") != e.sha:
                self.warn(f"Ignored unbound sidecar (missing/mismatched prediction_export_sha256): {sidecar}")
            else:
                t.meta.update(extra)
        self.tables.append(t)
        if set(PRED_COLS) & set(fields):
            e.kinds.add("prediction candidate")
        if set(RAW_COLS) & set(fields):
            e.kinds.add("raw response candidate")
        if any(re.fullmatch(r"(?:p|score_?|prob_?|grade_?)\d", key) for key in fields):
            e.kinds.add("grade-score candidate")
        if set(LABEL_COLS) & set(fields):
            e.kinds.add("split/label candidate")

    def load(self):
        # CSV first; workbook exports with the same stem in the same directory
        # are inventoried but do not override the CSV.
        for e in sorted(self.evidence, key=lambda x: (x.path.suffix.lower() not in {".csv", ".tsv"}, str(x.path))):
            path, suffix = e.path, e.path.suffix.lower()
            try:
                if "archive (not extracted)" in e.kinds:
                    e.note = "not extracted"
                    continue
                if e.size > MAX_BYTES:
                    raise ValueError("over 64 MiB limit; not read")
                e.sha = digest(path)
                if suffix in {".csv", ".tsv"}:
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        sample = stream.read(65536)
                        stream.seek(0)
                        try:
                            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                        except csv.Error:
                            dialect = csv.excel_tab if suffix == ".tsv" else csv.excel
                        reader = csv.DictReader(stream, dialect=dialect)
                        headers = reader.fieldnames or []
                        if not headers or len(set(map(norm, headers))) != len(headers):
                            raise ValueError("missing/duplicate table headers")
                        rows = []
                        for row in reader:
                            if None in row:
                                raise ValueError("row has extra fields; check delimiter/quoting")
                            rows.append(row)
                            if len(rows) > MAX_ROWS:
                                raise ValueError("over row limit; table not consumed")
                        self.add_table(e, rows, str(path))
                elif suffix in {".json", ".jsonl", ".ndjson"}:
                    if suffix != ".json":
                        rows = []
                        with path.open(encoding="utf-8-sig") as stream:
                            for line in stream:
                                if line.strip():
                                    rows.append(json.loads(line))
                                if len(rows) > MAX_ROWS:
                                    raise ValueError("over row limit")
                        self.add_table(e, rows, str(path))
                    else:
                        doc = json.loads(path.read_text(encoding="utf-8-sig"))
                        self.documents[path] = doc
                        if isinstance(doc, list) and doc and all(isinstance(r, dict) for r in doc):
                            self.add_table(e, doc, str(path))
                        elif isinstance(doc, dict):
                            for key in ("predictions", "records", "responses", "data"):
                                if isinstance(doc.get(key), list) and doc[key]:
                                    meta = {k: v for k, v in doc.items() if k not in {"predictions", "records", "responses", "data"}}
                                    self.add_table(e, doc[key], f"{path}#{key}", meta)
                elif suffix in {".xlsx", ".xls"}:
                    self.load_excel(e)
                elif suffix == ".ipynb":
                    nb = json.loads(path.read_text(encoding="utf-8-sig"))
                    for index, item in enumerate(nb.get("cells", []), 1):
                        source = item.get("source", [])
                        source = "".join(source) if isinstance(source, list) else source
                        if item.get("cell_type") == "code":
                            self.sources.append((path, f"{path}:cell {index}", source))
                        outputs = item.get("outputs", [])
                        if outputs:
                            e.kinds.add("saved notebook output")
                            for out in outputs:
                                text = out.get("text", out.get("data", {}).get("text/plain", ""))
                                text = "".join(text) if isinstance(text, list) else str(text)
                                hits = [line[:500] for line in text.splitlines()
                                        if re.search(r"accuracy|qwk|auc|parse|eval_loss|best_epoch|retraining", line, re.I)]
                                if hits:
                                    self.findings.append(("Saved output; not recomputed", f"{path}:cell {index}", " / ".join(hits[:12])))
                                    if any(re.search(r"eval_loss|best_epoch|train_loss", hit, re.I) for hit in hits):
                                        e.kinds.add("training record")
                elif suffix == ".py":
                    self.sources.append((path, str(path), path.read_text(encoding="utf-8-sig")))
                else:
                    text = path.read_text(encoding="utf-8-sig")
                    hits = [(i, line[:500]) for i, line in enumerate(text.splitlines(), 1)
                            if re.search(r"accuracy|qwk|roc.?auc|eval_loss|best_epoch|retraining|lora_r", line, re.I)]
                    for i, line in hits[:20]:
                        self.findings.append(("Recorded log/document; not recomputed", f"{path}:{i}", line))
                e.note = e.note or "read locally"
            except Exception as error:
                # A partly read file must not leave apparently usable tables.
                self.tables[:] = [t for t in self.tables if t.path != path]
                self.sources[:] = [item for item in self.sources if item[0] != path]
                self.documents.pop(path, None)
                e.note = f"NOT CONSUMED / possibly partial: {type(error).__name__}: {error}"
                self.warn(f"{path}: {e.note}")
            self.progress(f"Inspected {path}")
        for path, doc in self.documents.items():
            if isinstance(doc, dict) and re.search(r"run_info|trainer_state|manifest|config", path.name, re.I):
                keys = ("model_name", "base_model_name_or_path", "train_csv", "val_csv", "test_csv", "image_dir",
                        "best_epoch", "best_step", "best_eval_loss", "lora_r", "r", "prediction_method",
                        "helper_sha256", "adapter_dir", "reference_run", "preprocessing", "augmentation")
                self.findings.append(("Run/configuration record", str(path),
                                      json.dumps({k: doc[k] for k in keys if k in doc}, ensure_ascii=False)))
                for key in ("train_csv", "val_csv", "test_csv", "image_dir", "final_lora_dir"):
                    if not isinstance(doc.get(key), str) or not doc[key]:
                        continue
                    recorded = Path(doc[key]).expanduser()
                    recorded = (path.parent / recorded).resolve() if not recorded.is_absolute() else recorded.resolve()
                    available = self.inside(recorded) and recorded.exists()
                    self.checks.append(("Recorded input path", f"{path}: {key}",
                                        f"{recorded}: " + ("available within configured roots" if available else "not available within configured roots; add its current local location")))
        self.attach_manifest_provenance()

    def attach_manifest_provenance(self):
        """Use the existing helper's saved manifest/summary without executing it."""
        hashes = {e.path: e.sha for e in self.evidence}
        for table in self.tables:
            ending = ".predictions.csv"
            if not table.path.name.endswith(ending):
                continue
            name = table.path.name[:-len(ending)]
            manifest_path = table.path.parent / "evaluation_manifest.json"
            summary_path = table.path.parent / f"{name}.summary.json"
            manifest, summary = self.documents.get(manifest_path), self.documents.get(summary_path)
            if not isinstance(manifest, dict) or not isinstance(summary, dict):
                continue
            if manifest.get("prediction_method") != "full_answer_logprob_argmax":
                continue
            helper_sha = manifest.get("helper_sha256")
            if not helper_sha or summary.get("helper_sha256") != helper_sha or summary.get("adapter_sha256") != manifest.get("adapter_sha256"):
                self.warn(f"Manifest/summary identity mismatch for {table.path}")
                continue
            csv_hashes = manifest.get("csv_sha256", {})
            label_sha = csv_hashes.get(name) if isinstance(csv_hashes, dict) else None
            if not label_sha or summary.get("test_csv_sha256") != label_sha or summary.get("samples") != len(table.rows):
                self.warn(f"Manifest/summary cohort or row-count mismatch for {table.path}")
                continue
            producers = [p for p, _, _ in self.sources if p.suffix.lower() == ".py" and hashes.get(p) == helper_sha]
            labels = [t.path for t in self.tables if t.sha == label_sha and not set(PRED_COLS) & set(t.fields)]
            if not producers or not labels:
                self.checks.append(("Saved score provenance", table.location,
                                    "manifest present, but matching source and/or label-file hash not available locally"))
                continue
            source = dataset(name)
            inferred = {"dataset": source, "split": "test" if source == "aptos" else "external",
                        "label_file": str(labels[0]), "label_file_sha256": label_sha,
                        "score_columns": [f"p{i}" for i in range(5)], "class_order": list(range(5)),
                        "score_kind": "probability", "score_origin": "Saved full-answer likelihood softmax; same-directory helper manifest and summary",
                        "producer_file": str(producers[0]), "producer_file_sha256": helper_sha,
                        "adapter_dir": manifest.get("adapter_dir", ""),
                        "provenance_note": f"{manifest_path} + {summary_path}; manifest links source/cohort, does not hash prediction CSV"}
            for key, value in inferred.items():
                table.meta.setdefault(key, value)
            self.checks.append(("Saved score provenance", table.location, inferred["provenance_note"]))

    def load_excel(self, e):
        if any(e.path.with_suffix(ext).is_file() for ext in (".csv", ".tsv")):
            e.note = "CSV/TSV sibling preferred; workbook not read (freshness not assumed)"
            return
        if e.path.suffix.lower() == ".xlsx":
            try:
                import openpyxl  # Optional CPU-only reader; never installed here.
            except ImportError:
                note = f"Excel-only: {e.path}; openpyxl unavailable — export each evidence sheet to CSV"
                self.excel.append(note)
                self.warn(note)
                e.note = note
                return
            book = openpyxl.load_workbook(e.path, read_only=True, data_only=True, keep_links=False)
            try:
                for sheet in book.worksheets:
                    values = sheet.iter_rows(values_only=True)
                    headers = next(values, None)
                    if not headers:
                        continue
                    if any(h is None for h in headers) or len(set(map(norm, headers))) != len(headers):
                        raise ValueError(f"sheet {sheet.title}: empty/duplicate headers; export to CSV")
                    rows = []
                    for items in values:
                        rows.append(dict(zip(map(str, headers), items)))
                        if len(rows) > MAX_ROWS:
                            raise ValueError("workbook sheet over row limit")
                    self.add_table(e, rows, f"{e.path}#{sheet.title}")
                e.note = "optional openpyxl; cached formula values only, not recalculated; numeric IDs may have lost leading zeros"
                self.excel.append(f"{e.path}: read with openpyxl; verify text IDs and cached formulas")
            finally:
                book.close()
        else:
            note = f"Excel-only: {e.path}; legacy .xls not consumed by this worker — export to CSV"
            self.excel.append(note)
            self.warn(note)
            e.note = note

    def row_info(self, table, row):
        explicit = [dataset(row[k]) for k in ("source_dataset", "dataset", "source", "cohort")
                    if str(row.get(k, "")).strip()]
        explicit = [s for s in explicit if s != "unknown"]
        source = explicit[0] if explicit else dataset(table.meta.get("dataset", str(table.path)))
        error = "dataset columns disagree" if len(set(explicit)) > 1 else ""
        ids = [(k, image_id(row.get(k), source)) for k in ID_COLS if str(row.get(k, "")).strip()]
        ident = ids[0][1] if ids else ""
        # An explicit original/source ID can intentionally differ from renamed paths.
        if ids and ids[0][0] not in {"original_image_id", "source_image_id"} and len({v for _, v in ids}) > 1:
            error = error or "image ID/path aliases disagree; add original_image_id"
        labels = [grade(row[k]) for k in LABEL_COLS if str(row.get(k, "")).strip()]
        label = labels[0] if labels else None
        if any(value is None for value in labels):
            error = error or "invalid ground-truth grade"
        if len(set(labels)) > 1:
            error = error or "label columns disagree"
        raw_split = row.get("split", row.get("subset", row.get("partition", table.meta.get("split", ""))))
        part = split_name(raw_split) if str(raw_split).strip() else filename_split(table.path)
        if not part and source == "messidor2":
            part = "external"
        return source, part, ident, label, error

    def build_cohorts(self):
        groups = {}
        for table in self.tables:
            derived = bool((set(PRED_COLS) | set(RAW_COLS)) & set(table.fields))
            derived = derived or any(re.fullmatch(r"(?:p|score_?|prob_?|grade_?)\d", key) for key in table.fields)
            if derived or not (set(LABEL_COLS) & set(table.fields)):
                continue
            parts = {self.row_info(table, r)[1] for r in table.rows}
            # Single-split CSVs often repeat a split column. Group siblings by
            # their split-name pattern even when that column is present.
            combined = len(parts) > 1 or not filename_split(table.path)
            signature = table.location if combined else str(table.path.parent / SPLIT_RE.sub("SPLIT", table.path.stem))
            for row in table.rows:
                source, part, ident, label, error = self.row_info(table, row)
                key = (signature, source, part)
                cohort = groups.setdefault(key, Cohort(source, part, signature))
                if table.path not in cohort.paths:
                    cohort.paths.append(table.path)
                if source == "unknown" or part not in {"train", "val", "test", "external"}:
                    error = error or "dataset/split unresolved"
                named_split = filename_split(table.path)
                if named_split and part and named_split != part and not (named_split == "test" and part == "external"):
                    error = error or "row split disagrees with filename"
                if not ident or label is None:
                    error = error or "missing/invalid image ID or grade"
                if error and error not in cohort.issues:
                    cohort.issues.append(error)
                if ident in cohort.labels:
                    cohort.duplicate_count += 1
                    if cohort.labels[ident] != label and "duplicate ID has conflicting labels" not in cohort.issues:
                        cohort.issues.append("duplicate ID has conflicting labels")
                elif ident and label is not None:
                    cohort.labels[ident] = label
        self.cohorts = list(groups.values())
        for c in self.cohorts:
            if c.duplicate_count:
                c.issues.append(f"{c.duplicate_count} duplicate ID rows")
            self.checks.append(("Label IDs / class counts", "; ".join(map(str, c.paths)),
                                f"{c.source}/{c.split}; {len(c.labels)} IDs; duplicate rows={c.duplicate_count}; " + ("; ".join(c.issues) or "valid")))
        for i, left in enumerate(self.cohorts):
            for right in self.cohorts[i + 1:]:
                if left.source != right.source or left.source == "unknown":
                    continue
                common = set(left.labels) & set(right.labels)
                disagreements = [k for k in common if left.labels[k] != right.labels[k]]
                if disagreements:
                    self.checks.append(("Cross-file label disagreement", f"{left.paths} / {right.paths}",
                                        f"{len(disagreements)} conflicting IDs: {examples(disagreements)}"))
                if common and left.family != right.family and left.split != right.split:
                    self.checks.append(("Cross-definition split overlap (review)", f"{left.paths} / {right.paths}",
                                        f"{len(common)} shared IDs: {examples(common)}; these are separate split definitions, so this is not automatically leakage in the selected family"))
                if left.family == right.family and left.split != right.split:
                    self.checks.append(("Split overlap", f"{left.paths} / {right.paths}",
                                        f"{len(common)} shared IDs" + (": " + examples(common) if common else "")))
                    if common:
                        left.issues.append("split overlap")
                        right.issues.append("split overlap")
        for c in self.cohorts:
            if c.source == "aptos" and c.split in EXPECTED_APTOS:
                actual = [Counter(c.labels.values())[i] for i in range(5)]
                self.checks.append(("Declared reference count comparison", str(c.paths),
                                    f"{actual}; expected {EXPECTED_APTOS[c.split]}; match={actual == EXPECTED_APTOS[c.split]} (counts alone insufficient)"))
            matched, original_candidates, multiple = 0, 0, 0
            for ident in c.labels:
                paths = list(self.image_index.get(ident, []))
                if c.source == "aptos":
                    paths += self.image_index.get("aptos_" + ident, [])
                matched += bool(paths)
                multiple += len(paths) > 1
                original_candidates += any(self.image_kind(p.parent).startswith("raw/original") for p in paths)
            self.checks.append(("Image path coverage (no pixels read)", str(c.paths),
                                f"{c.source}/{c.split}: {matched}/{len(c.labels)} IDs have image candidates; "
                                f"{original_candidates} have raw/original folder candidates; {multiple} have multiple versions; provenance unverified"))

    def inspect_sources(self):
        topics = {
            "Parser / invalid-output fallback": r"ANSWER_RE|def parse_|parse_ok|pred\s*=\s*0|pred\s+is\s+None|bad_parse|fallback|return\s+0",
            "Augmentation / split routing": r"augment|RandomHorizontal|RandomVertical|RandomRotation|ColorJitter|train_transform|val_transform|set_transform|train_dataset=|eval_dataset=|WeightedRandomSampler",
            "BGR / RGB / LAB / CLAHE": r"cvtColor|COLOR_(?:BGR|RGB|LAB)|createCLAHE|apply_clahe|resized_rgb|resized_bgr|\.convert\([\"']RGB|imwrite|imread",
        }
        for path, location, source in self.sources:
            lines = source.splitlines()
            for topic, pattern in topics.items():
                hits = [(i, line.strip()) for i, line in enumerate(lines, 1) if re.search(pattern, line, re.I)]
                for i, line in hits[:45]:
                    context = " ".join(s.strip() for s in lines[max(0, i - 2):min(len(lines), i + 2)])
                    self.findings.append((topic, f"{location}:L{i}", context[:1000]))
                if len(hits) > 45:
                    self.findings.append((topic, location, f"{len(hits) - 45} further matching lines; inspect full file"))
            if re.search(r"augment|RandomHorizontal|RandomRotation|ColorJitter", source, re.I):
                self.findings.append(("Training-only augmentation verdict", location,
                                      "Requires call-path and producing-revision review: inspect whether the train branch alone receives transforms; keyword presence does not certify this."))
            if path.name == "train_dr_qwen3vl_runsave.py":
                image_aug = re.search(r"RandomHorizontal|RandomVertical|RandomRotation|ColorJitter|albumentations|ImageEnhance|ImageOps\.(?:flip|mirror)|def\s+\w*augment", source, re.I)
                maps = all(re.search(rf"{part}_raw\.map\(\s*to_conversation\s*,", source) for part in ("train", "val", "test"))
                if maps and not image_aug:
                    self.findings.append(("Training-only augmentation verdict", location,
                                          "Current source maps the same to_conversation formatting over train/val/test and has no recognized explicit image augmentation. WeightedRandomSampler affects the training dataloader only. This does not rule out library-internal transforms or pre-augmented input files; historical augmentation remains unverified."))
            original = re.search(r'''ORIGINAL_SPLIT_SHA256\s*=\s*["']([0-9a-f]{64})["']''', source)
            if original:
                matches = [str(t.path) for t in self.tables if t.sha == original.group(1)]
                self.checks.append(("Declared original split fingerprint", location,
                                    f"SHA-256={original.group(1)}; matching tables=" + ("; ".join(matches) or "MISSING — provide the recovered original split CSV")))
            for match in re.finditer(r"(?:val|test|eval)(?:_ds|_dataset|_raw)?\.(?:map|set_transform|with_transform)\(\s*(\w*(?:augment|train_transform)\w*)", source, re.I):
                line = source[:match.start()].count("\n") + 1
                self.findings.append(("Potential evaluation augmentation", f"{location}:L{line}",
                                      f"Evaluation data receives {match.group(1)}; inspect this callable before treating augmentation as training-only."))
            # Recognize the actual saved preprocessing routes, rather than
            # declaring BGR/RGB coexistence itself to be an error.
            compact = re.sub(r"\s+", "", source)
            bgr_helper = "defapply_clahe_bgr(" in compact and "cv2.COLOR_BGR2LAB" in compact and "cv2.COLOR_LAB2BGR" in compact
            rgb_base = "cv2.COLOR_BGR2RGB" in compact and "resized_rgb=cv2.resize(base_img," in compact
            if bgr_helper and rgb_base and "apply_clahe_bgr(resized_rgb)" in compact:
                self.findings.append(("Static color mismatch candidate", location,
                                      "RGB resized image is passed directly to apply_clahe_bgr, whose conversion expects BGR. This is a saved-code finding; it is not proof this revision produced the evaluated images."))
            elif bgr_helper and rgb_base and "resized_bgr=cv2.cvtColor(resized_rgb,cv2.COLOR_RGB2BGR)" in compact and "apply_clahe_bgr(resized_bgr)" in compact:
                self.findings.append(("Recognized color route", location,
                                      "Current code explicitly routes BGR read → RGB crop/pad/resize → BGR → LAB lightness CLAHE → BGR save. The conversion immediately before CLAHE is consistent with the BGR helper; no direct RGB-as-BGR handoff is present in this recognized route. Historical image lineage still needs linkage."))
            elif "COLOR_BGR2LAB" in source and "COLOR_BGR2RGB" in source:
                self.findings.append(("Color routing review", location,
                                      "Both RGB loading and BGR-to-LAB conversion occur. Trace the call argument into CLAHE; an explicit RGB-to-BGR conversion may make this correct. No pixel comparison performed."))
            if "COLOR_RGB2LAB" in source and "COLOR_LAB2RGB" in source:
                self.findings.append(("Color routing review", location,
                                      "RGB-to-LAB and LAB-to-RGB conversions are present; verify caller input order and final save conversion."))
                if "apply_clahe(resized_rgb)" in compact and "final_bgr=cv2.cvtColor(final_rgb,cv2.COLOR_RGB2BGR)" in compact:
                    self.findings.append(("Recognized color route", location,
                                          "Current source sends resized RGB to RGB→LAB→RGB CLAHE, then explicitly converts final RGB to BGR for cv2.imwrite. This recognized path is internally consistent; it does not certify the historical preprocessing revision."))
        if not any(re.search(r"augment|RandomHorizontal|RandomRotation|ColorJitter", s, re.I) for _, _, s in self.sources):
            self.findings.append(("Training-only augmentation verdict", "all inspected sources",
                                  "No explicit image augmentation operations recognized. WeightedRandomSampler is class sampling, not image augmentation. Historical training-only augmentation remains unverified."))

    def prediction_rows(self, table):
        rows, issues, duplicates = {}, Counter(), 0
        pred_columns = [key for key in PRED_COLS if key in table.fields]
        for row in table.rows:
            source, part, ident, label, error = self.row_info(table, row)
            values = [grade(row.get(k)) for k in pred_columns if str(row.get(k, "")).strip()]
            prediction = values[0] if values else None
            if not ident:
                error = error or "missing image ID"
            if source == "unknown":
                error = error or "dataset unresolved"
            if prediction is None or len(set(values)) > 1:
                error = error or "invalid/disagreeing prediction columns"
            if error:
                issues[error] += 1
            key = (source, ident)
            if key in rows:
                duplicates += 1
            else:
                rows[key] = (row, label, prediction, part)
        if duplicates:
            issues["duplicate image IDs"] += duplicates
        return rows, issues

    def choose_cohort(self, table, rows, issues):
        sources = {s for s, _ in rows}
        if len(sources) != 1:
            return None, "mixed/unknown dataset: export datasets separately"
        source = next(iter(sources))
        ids = {ident for _, ident in rows}
        choices = [c for c in self.cohorts if c.source == source and c.split in {"test", "external"}]
        try:
            selected = self.linked(table, "label_file")
        except (ValueError, OSError) as error:
            return None, str(error)
        if selected:
            choices = [c for c in choices if selected in c.paths]
        exact = [c for c in choices if set(c.labels) == ids]
        if not exact:
            closest = max(choices, key=lambda c: len(set(c.labels) & ids), default=None)
            if closest:
                missing, extra = set(closest.labels) - ids, ids - set(closest.labels)
                return None, f"not exact test coverage vs {closest.paths}: missing={len(missing)} ({examples(missing)}); extra={len(extra)} ({examples(extra)})"
            return None, "no independent test/external label cohort found"
        distinct = {tuple(sorted(c.labels.items())) for c in exact}
        if len(distinct) > 1:
            return None, "alternative test label sources disagree; hash-link the authoritative label_file"
        valid = []
        reasons = []
        for c in exact:
            blocked = list(c.issues)
            if source == "aptos":
                family = [other for other in self.cohorts if other.source == source and other.family == c.family]
                for part in ("train", "val"):
                    matches = [other for other in family if other.split == part]
                    if len(matches) != 1 or matches[0].issues:
                        blocked.append(f"missing/invalid {part} in same split definition")
            disagreements = [ident for (s, ident), (_, label, _, _) in rows.items()
                             if label is not None and label != c.labels[ident]]
            if disagreements:
                blocked.append(f"{len(disagreements)} prediction/label disagreements: {examples(disagreements)}")
            declared = {part for _, _, _, part in rows.values() if part}
            if declared - {"test", "external"}:
                blocked.append("prediction rows declare a non-test split")
            if not blocked:
                valid.append(c)
            reasons.extend(blocked)
        if issues:
            reasons.append(str(dict(issues)))
        if not valid or issues:
            return None, "; ".join(dict.fromkeys(reasons)) or "unverified"
        c = valid[0]
        return c, f"exact one-to-one ID join to {c.paths}; labels agree; {len(ids)} rows; split checks passed"

    @staticmethod
    def metrics(truth, predicted):
        n = len(truth)
        observed = sum((t - p) ** 2 for t, p in zip(truth, predicted))
        true_counts, pred_counts = Counter(truth), Counter(predicted)
        expected = sum((i - j) ** 2 * true_counts[i] * pred_counts[j] for i in range(5) for j in range(5)) / n
        qwk = None if expected == 0 else 1 - observed / expected
        return sum(t == p for t, p in zip(truth, predicted)) / n, qwk

    @staticmethod
    def binary_auc(truth, scores):
        positives = sum(truth)
        negatives = len(truth) - positives
        if not positives or not negatives:
            return None
        ordered = sorted(zip(scores, truth))
        rank_sum, i = 0.0, 0
        while i < len(ordered):
            end = i + 1
            while end < len(ordered) and ordered[end][0] == ordered[i][0]:
                end += 1
            mean_rank = (i + 1 + end) / 2
            rank_sum += mean_rank * sum(label for _, label in ordered[i:end])
            i = end
        return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    def score_auc(self, table, rows, cohort):
        # No reconstruction from y_pred and no default class order for p0..p4.
        score_table = table
        try:
            score_path = self.linked(table, "score_file")
            if score_path:
                choices = [t for t in self.tables if t.path == score_path]
                if len(choices) != 1:
                    raise ValueError("score_file must resolve to one parsed table")
                score_table = choices[0]
            meta = table.meta
            columns = meta.get("score_columns")
            order = meta.get("class_order")
            kind = meta.get("score_kind")
            origin = str(meta.get("score_origin", "")).strip()
            producer = self.linked(table, "producer_file")
            if (not isinstance(columns, list) or len(columns) != 5 or len(set(columns)) != 5
                    or not isinstance(order, list) or len(order) != 5
                    or any(type(v) is not int for v in order) or set(order) != set(range(5))):
                raise ValueError("need five score_columns and explicit integer class_order [0..4] permutation")
            columns = [norm(c) for c in columns]
            if kind not in {"probability", "logit", "log_probability", "full_answer_log_probability"} or not origin:
                raise ValueError("score_kind / score_origin not documented")
            if re.search(r"one.?hot|hard.?label|synthetic|from.?pred", origin, re.I):
                raise ValueError("score provenance describes synthetic/hard-label scores")
            if producer is None:
                raise ValueError("missing hash-linked score producer_file")
            source = next((s for p, _, s in self.sources if p == producer), "")
            # Recognize the existing helper's complete-answer likelihood path.
            # Unknown producers stay pending manual review rather than being
            # accepted because they merely mention logits or probabilities.
            required = ("def score_full_answers(", "-float(output.loss.item()) * count",
                        "scores.append(score)", "return np.exp(scores) / np.exp(scores).sum()",
                        'f"p{i}": float(scores[i])', '"y_pred": int(np.argmax(scores))')
            if not all(token in source for token in required):
                raise ValueError("unrecognized score producer; source/provenance needs manual review")
            if columns != [f"p{i}" for i in range(5)] or order != list(range(5)) or kind != "probability":
                raise ValueError("declared columns/order/kind disagree with recognized producer's p0..p4 probabilities")
            if 'CANDIDATES = [f"<answer>{i} - {name}</answer>" for i, name in enumerate(LABEL_NAMES)]' not in source:
                raise ValueError("producer class-candidate order not recognized")
            indexed, duplicates = {}, 0
            for row in score_table.rows:
                source_name, _, ident, label, error = self.row_info(score_table, row)
                if score_table is table:
                    source_name = cohort.source
                if source_name != cohort.source or error or not ident:
                    raise ValueError("score table has unresolved/mixed IDs or conflicting aliases")
                if ident in indexed:
                    duplicates += 1
                indexed[ident] = (row, label)
            if duplicates or set(indexed) != set(cohort.labels):
                raise ValueError("score IDs are duplicate or not an exact cohort match")
            vectors, truths = [], []
            for (_, ident), (_, _, pred, _) in rows.items():
                row, label = indexed[ident]
                if label is not None and label != cohort.labels[ident]:
                    raise ValueError("score table labels disagree")
                values = [float(row[c]) for c in columns]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-5:
                    raise ValueError("non-finite/non-probability vector or row sum differs from 1")
                if all(v in {0.0, 1.0} for v in values):
                    raise ValueError("one-hot row present; cannot verify genuine grade scores")
                if max(range(5), key=lambda i: values[i]) != pred:
                    raise ValueError("score argmax disagrees with exported prediction")
                vectors.append(values)
                truths.append(cohort.labels[ident])
            aucs = [self.binary_auc([t == i for t in truths], [v[i] for v in vectors]) for i in range(5)]
            if any(v is None for v in aucs):
                raise ValueError("at least one grade lacks positives/negatives; five-class macro AUC undefined")
            return sum(aucs) / 5, f"hash-linked producer {producer}; columns/order/probabilities verified statically; per-grade AUC={aucs}; " + str(meta.get("provenance_note", "export-bound sidecar provenance"))
        except (ValueError, KeyError, TypeError, OSError) as error:
            return None, f"AUC withheld: {error}"

    def reference_link(self, table):
        if table.meta.get("reference_run") == REFERENCE:
            return f"sidecar explicitly identifies {REFERENCE}; inspect linked provenance before publication"
        return "accepted reference linkage UNVERIFIED; run name/rank/metric similarity is insufficient"

    def evaluate_exports(self):
        for table in self.tables:
            if not (set(PRED_COLS) & set(table.fields)):
                continue
            self.progress(f"Checking prediction joins: {table.location}")
            rows, issues = self.prediction_rows(table)
            self.checks.append(("Prediction duplicate/invalid IDs", table.location, str(dict(issues)) if issues else f"{len(rows)} unique valid rows"))
            true_counts = Counter(label for _, label, _, _ in rows.values() if label is not None)
            pred_counts = Counter(pred for _, _, pred, _ in rows.values() if pred is not None)
            self.checks.append(("Prediction export class counts", table.location,
                                f"embedded truth={[true_counts[i] for i in range(5)]}; predicted={[pred_counts[i] for i in range(5)]}; counts include only valid grades, and do not override integrity issues"))
            cohort, reason = self.choose_cohort(table, rows, issues)
            self.checks.append(("Independent test label join", table.location, reason))
            if cohort is None:
                self.results.append((table.location, "BLOCKED: " + reason, "not recomputed", self.reference_link(table)))
            else:
                truth = [cohort.labels[ident] for _, ident in rows]
                predicted = [value[2] for value in rows.values()]
                accuracy, qwk = self.metrics(truth, predicted)
                auc, auc_note = self.score_auc(table, rows, cohort)
                metric_text = f"accuracy={accuracy:.6f} ({accuracy * 100:.4f}%); QWK=" + (f"{qwk:.6f}" if qwk is not None else "undefined")
                metric_text += "; macro OVR ROC-AUC=" + (f"{auc:.6f}" if auc is not None else "withheld")
                self.results.append((table.location, f"{cohort.source}/{cohort.split}: {reason}", metric_text, self.reference_link(table) + "; " + auc_note))
                self.verified.append((table, cohort, auc))
            self.audit_fallbacks(table, rows, issues)
            self.report()
        for table in self.tables:
            if set(RAW_COLS) & set(table.fields) and not set(PRED_COLS) & set(table.fields):
                self.fallbacks.append((table.location, "Raw-only table inventoried; no exported predictions to confirm Grade-0 assignment",
                                       "not established", "Link as raw_file with hash to the matching prediction export; preserve parse_ok. Never join unrelated runs by position."))

    def known_parser(self, table):
        try:
            path = self.linked(table, "parser_file")
            producer = self.linked(table, "producer_file")
            if not path or not producer:
                return False
            source = next((s for p, _, s in self.sources if p == path), "")
            producer_text = next((s for p, _, s in self.sources if p == producer), "")
            tree = ast.parse(source)
            expected = ast.parse('''def parse_pred_label(gen_text: str) -> Optional[int]:
    m = ANSWER_RE.search(gen_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None
''').body[0]
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_pred_label"]
            expected_re = ast.parse('ANSWER_RE = re.compile(r"<answer>\\s*([0-4])\\s*-\\s*.*?</answer>", re.IGNORECASE)').body[0]
            regex_ok = any(ast.dump(n, include_attributes=False) == ast.dump(expected_re, include_attributes=False) for n in tree.body)
            parser_ok = len(functions) == 1 and ast.dump(functions[0], include_attributes=False) == ast.dump(expected, include_attributes=False)
            normalized = "\n".join(line.strip() for line in producer_text.splitlines())
            fallback_ok = "pred = parse_pred_label(gen)\nparse_ok = pred is not None\nif not parse_ok:\nbad += 1\npred = 0" in normalized
            return regex_ok and parser_ok and fallback_ok and table.meta.get("fallback_to_grade0") is True
        except (ValueError, OSError, SyntaxError):
            return False

    def audit_fallbacks(self, table, rows, issues):
        raw_rows = None
        raw_location = table.location
        try:
            raw_path = self.linked(table, "raw_file")
            if raw_path:
                choices = [t for t in self.tables if t.path == raw_path]
                if len(choices) != 1:
                    raise ValueError("raw_file must identify one parsed table")
                raw_table = choices[0]
                raw_location = raw_table.location
                raw_rows = {}
                for row in raw_table.rows:
                    source, _, ident, label, error = self.row_info(raw_table, row)
                    key = (source, ident)
                    if key in raw_rows or error or not ident:
                        raise ValueError("duplicate/ambiguous raw-output IDs")
                    raw_rows[key] = (row, label)
                if set(raw_rows) != set(rows):
                    raise ValueError("raw_file IDs are not an exact match to predictions")
        except (ValueError, OSError) as error:
            self.fallbacks.append((table.location, "Separate raw join BLOCKED", "not established", str(error)))
            return
        recognized = self.known_parser(table)
        counts = Counter()
        confirmed_ids = []
        mild_ids = []
        for key, (pred_row, exported_label, prediction, _) in rows.items():
            source, ident = key
            raw_row, raw_label = raw_rows[key] if raw_rows is not None else (pred_row, exported_label)
            raw_keys = [k for k in RAW_COLS if k in raw_row]
            if not raw_keys:
                continue
            counts["raw"] += 1
            raw_values = {str(raw_row[k]) for k in raw_keys}
            if len(raw_values) != 1:
                counts["raw_alias_conflicts"] += 1
                continue
            text = str(raw_row[raw_keys[0]])
            match = re.search(ANSWER_PATTERN, text, re.IGNORECASE)
            parsed = int(match.group(1)) if match else None
            if parsed is None:
                counts["current_parser_invalid"] += 1
            flags = [norm(raw_row[k]) for k in ("parse_ok", "parse_success", "parsed_ok") if k in raw_row and str(raw_row[k]).strip()]
            false_flag = bool(flags) and all(v in {"false", "0", "0_0", "no"} for v in flags)
            true_flag = bool(flags) and all(v in {"true", "1", "1_0", "yes"} for v in flags)
            if flags and not (false_flag or true_flag):
                counts["flag_conflicts"] += 1
                continue
            if false_flag or true_flag:
                counts["saved_parse_flags"] += 1
            if true_flag and parsed is None or false_flag and parsed is not None:
                counts["current_parser_flag_disagreement"] += 1
            # Independent label agreement can be checked for partial coverage;
            # full-cohort verification is required only for aggregate metrics.
            labels = {c.labels[ident] for c in self.cohorts if c.source == source and ident in c.labels and not c.issues}
            if len(labels) != 1:
                counts["no_unambiguous_independent_label"] += 1
                continue
            label = next(iter(labels))
            if (exported_label is not None and exported_label != label) or (raw_label is not None and raw_label != label):
                counts["label_conflicts"] += 1
                continue
            counts["joined"] += 1
            if source == "messidor2" and label == 1 and prediction == 0:
                counts["messidor_mild_normal"] += 1
            confirmed = not issues and prediction == 0 and (false_flag or (not flags and recognized and parsed is None))
            if false_flag and prediction != 0:
                counts["false_flag_nonzero_prediction"] += 1
            if confirmed:
                counts["confirmed"] += 1
                confirmed_ids.append(f"{source}:{ident}")
                if source == "messidor2" and label == 1:
                    counts["confirmed_mild_normal"] += 1
                    mild_ids.append(ident)
        if not counts["raw"]:
            self.fallbacks.append((table.location, "No row-level raw responses joined", "not established", "Supply matching raw_output and parse_ok with IDs and labels."))
            return
        self.fallbacks.append((table.location,
                               f"raw={counts['raw']}; independent-label joins={counts['joined']}; confirmed saved fallbacks={counts['confirmed']}; IDs={examples(confirmed_ids, 20)}",
                               f"Messidor-2 Mild→Normal among joined rows={counts['messidor_mild_normal']}; confirmed fallback contribution={counts['confirmed_mild_normal']}; IDs={examples(mild_ids, 20)}",
                               f"raw source={raw_location}; hash-linked historical parser recognized={recognized}; "
                               f"current-parser invalid candidates={counts['current_parser_invalid']} (not automatically historical fallbacks); other checks={dict(counts)}"))
        self.fallback_coverage.append({"datasets": {source for source, _ in rows},
                                       "complete": not issues and counts["joined"] == len(rows)
                                       and (counts["saved_parse_flags"] == len(rows) or recognized)})

    def control_handoff(self):
        for path, doc in self.documents.items():
            if path.name.lower() != "adapter_config.json" or not isinstance(doc, dict):
                continue
            weights = [(path.parent / name, (path.parent / name).stat().st_size)
                       for name in ("adapter_model.safetensors", "adapter_model.bin")
                       if (path.parent / name).is_file()]
            usable_weights = any(size > 0 for _, size in weights)
            rank = doc.get("r")
            properties = {key: doc.get(key) for key in ("r", "lora_alpha", "lora_dropout", "base_model_name_or_path", "peft_type", "revision", "rank_pattern")}
            related = [(p, d) for p, d in self.documents.items() if isinstance(d, dict)
                       and p.name in {"run_info.json", "training_config.json", "run_config.json"}
                       and p.parent in {path.parent, path.parent.parent}]
            properties["related_records"] = [str(p) for p, _ in related]
            properties["weights_present_not_loaded"] = [(str(p), size) for p, size in weights]
            clahe_hint = "clahe" in str(path).lower() or any("clahe" in json.dumps(d).lower() for _, d in related)
            blockers = []
            decision = "lacks enough evidence to decide"
            if rank is not None and rank != 16:
                decision = "needs rebuilding for a rank-16 control"
                blockers.append(f"recorded rank is {rank}")
            elif doc.get("rank_pattern"):
                blockers.append("per-layer rank_pattern requires review; scalar r=16 is insufficient")
            if not usable_weights:
                blockers.append("non-empty local adapter weights missing; restore checkpoint or rebuild")
            if rank == 16:
                blockers.append("rank-16 confirmed in config; weight contents not loaded/validated")
            blockers.append("CLAHE lineage hinted by records" if clahe_hint else "CLAHE lineage not established")
            # Source correctness and historical augmentation cannot be certified
            # from config/path hints. Never bless a control on rank alone.
            blockers.append("exact image lineage/channel order, train-only augmentation, baseline run identity and producing revision need review")
            linked_results = [(t, c) for t, c, _ in self.verified if str(t.meta.get("adapter_dir", "")) == str(path.parent)]
            blockers.append(f"{len(linked_results)} verified exports explicitly name this adapter directory (path linkage is not a weight hash)")
            self.adapters.append((path, json.dumps(properties, ensure_ascii=False), f"**{decision}**: " + "; ".join(blockers)))

    def finalize(self):
        # A successful scan can still be a partial scientific audit. Preserve
        # nonzero status until required reference evidence is actually linked.
        baseline = [(t, c, auc) for t, c, auc in self.verified
                    if c.source == "aptos" and t.meta.get("reference_run") == REFERENCE
                    and [Counter(c.labels.values())[i] for i in range(5)] == EXPECTED_APTOS["test"]]
        messidor = any(c.source == "messidor2" for _, c, _ in self.verified)
        raw_confirmable = any(item["datasets"] == {"messidor2"} and item["complete"] for item in self.fallback_coverage)
        training = any(isinstance(doc, dict) and (doc.get("history") or doc.get("log_history"))
                       for doc in self.documents.values()) or any(e.path.suffix.lower() == ".log" or
                       ("saved notebook output" in e.kinds and "training record" in e.kinds) for e in self.evidence)
        originals = any(self.image_kind(path).startswith("raw/original") for path in self.images)
        complete = bool(baseline and any(auc is not None for _, _, auc in baseline)
                        and messidor and raw_confirmable and training and originals and self.adapters and not self.warnings)
        # Control evidence is intentionally a review handoff; no automatic claim
        # that a historical pipeline or original image lineage is certified.
        if complete:
            self.status = "FINISHED — metric/input checks completed; static provenance and control handoff require review"
            return 0
        self.status = "PARTIAL — scan finished; missing/unverified evidence is listed below"
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True, help="Read-only project reference root")
    parser.add_argument("--input-root", type=Path, action="append", default=[], help="Additional local input folder; repeatable")
    args = parser.parse_args(argv)
    roots = list(dict.fromkeys(p.expanduser().resolve() for p in [args.project_root, *args.input_root]))
    # No GPU framework is imported; these also document CPU-only intent if an
    # optional workbook reader is present in the selected Python environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    with (OUTPUT_DIR / "phase1_run.log").open("w", encoding="utf-8", newline="\n") as log:
        audit = Audit(roots, log)
        code = 1
        try:
            audit.progress("Starting Phase 1 local CPU audit; no models or old code will run")
            audit.report()
            audit.inventory()
            audit.report()
            audit.load()
            audit.report()
            audit.build_cohorts()
            audit.inspect_sources()
            audit.report()
            audit.evaluate_exports()
            audit.control_handoff()
            code = audit.finalize()
        except KeyboardInterrupt:
            audit.status = "PARTIAL — interrupted"
            audit.progress("Interrupted; saving available evidence")
            code = 130
        except Exception as error:
            audit.status = "PARTIAL — unexpected worker error"
            audit.warn(f"{type(error).__name__}: {error}")
            log.write(traceback.format_exc())
            log.flush()
            code = 1
        finally:
            audit.report()
            audit.progress(f"Exit status {code}; report: {OUTPUT_DIR / 'phase1_report.md'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
