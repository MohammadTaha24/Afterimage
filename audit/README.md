# Phase 1 diabetic retinopathy audit

Prepared for execution tomorrow morning. The files were written and reviewed by inspection only; no notebook cells, worker, tests, background jobs or GPU work were run during preparation.

Open **[phase1_orchestrator.ipynb](phase1_orchestrator.ipynb)** in Jupyter with a Python 3.10+ kernel.

1. In **Configure paths**, set `PROJECT_ROOT` to the read-only project reference, normally `C:\Users\moht2\Documents\Afterimage`. Add local folders containing split lists, labels, saved predictions, raw responses, scores, run records or original images to `EXTRA_INPUT_ROOTS`. Historical paths recorded under `C:\Users\Mohammad\...` are clues; add their current local locations explicitly. `WORKER` already points at this folder; change it only if the audit files move.
2. Run **Configure paths**. This only defines paths.
3. Run **Run audit** once. It starts exactly one Python child using the notebook kernel's interpreter, streams its output, waits for exit and prints the exit status and report path. Interrupting this cell terminates and waits up to 15 seconds, then kills and waits if necessary; its output pipe is closed.
4. Read `phase1_report.md` and `phase1_run.log` in this folder. You can **shut down the notebook kernel after the worker exits**; the audit's data and working memory belonged to the child and are released on exit.

Opening the notebook, editing paths, or executing only Configure paths never starts the audit. Avoid concurrent runs: each invocation replaces the same two output files. The Python worker imports no project modules, loads no models or image pixels, starts no processes, and performs no inference, training, installs, downloads or uploads. The optional installed `openpyxl` reader is used only for Excel input; it is never installed automatically. Legacy `.xls` files require CSV exports.

The same worker can be run directly in PowerShell (do this tomorrow, not as a preparation check):

```powershell
Set-Location -LiteralPath 'C:\Users\moht2\Documents\Afterimage'
python -u .\audit\phase1_audit.py --project-root 'C:\Users\moht2\Documents\Afterimage'
$LASTEXITCODE
```

With additional folders, repeat `--input-root`:

```powershell
python -u .\audit\phase1_audit.py --project-root 'C:\Users\moht2\Documents\Afterimage' --input-root 'D:\DR evidence\APTOS exports' --input-root 'D:\DR evidence\Messidor-2' --input-root 'D:\DR evidence\original images'
$LASTEXITCODE
```

Use the full path to your Python executable in place of `python` if needed. Exit **0** means the worker reached its completion criteria, with provenance/control review still handed off; **2** means a finished scan with missing or unverified scientific inputs; **1** means an unexpected worker error; **130** means the worker handled an interruption. A notebook interrupt may terminate the process at the OS level with a different exit code. Reports are refreshed between stages; an interrupted report can be partial, and an immediate forced stop can leave the previous run's report. Check timestamps and the log.

## Evidence and verification

Inputs remain in place. The worker scans only the configured roots, skipping environments, caches, `.git`, links/junction directories and its own audit folder. Put evidence outside this audit folder. It inventories CSV/TSV, Excel, JSON/JSONL, saved source/notebooks, training logs and archives; archives are not extracted. It does not follow historical paths outside the roots. Files over 64 MiB and tables over 1,000,000 rows are flagged as unconsumed. Unsupported encodings, malformed schemas and missing roots produce a partial report rather than silently dropping records. Prefer UTF-8 CSV exports; workbook formulas are not recalculated, numeric Excel IDs may have lost leading zeros, and same-stem CSV exports take precedence without an assumption about freshness.

Useful table columns include:

| Purpose | Recognized examples |
| --- | --- |
| Image ID | `original_image_id`, `source_image_id`, `image_id`, `id_code`, `img_id`, `image_file`, `filename`, `image_path` |
| Ground truth | `y_true`, `adjudicated_dr_grade`, `diagnosis`, `dr_grade`, `label`, `grade` |
| Prediction | `y_pred`, `predicted_label`, `predicted_grade`, `pred_label` |
| Cohort | `source_dataset` or `dataset`: `aptos`, `messidor`, `messidor2`; `split`: `train`, `val`, `test`, `external` |
| Raw result | `raw_output`, `raw_response`, `generated_text`, `response`; preserve `parse_ok` |
| Scores | Five numeric columns such as `p0` through `p4`, plus explicit class order and score-generation provenance |

Dataset and split can also be recognized from unambiguous paths/filenames, such as `aptos_existing_split_exports\train_split.csv`. Generic paths need explicit dataset/split fields or a bound sidecar. Keep separate APTOS split files together with matching names (`train_split.csv`, `val_split.csv`, `test_split.csv`); a combined list with a split column is also supported. Distinct split definitions remain distinct. The original reference list is not regenerated. Messidor labels without a split are treated as an external label cohort; the export must cover that exact cohort to support aggregate metrics. If only a subset was evaluated, supply the actual image-level cohort list rather than a larger label pool. Exports containing mixed datasets should be separated for verification.

IDs preserve case and leading zeros; image extensions and path directories are removed, and `aptos_` is removed only for APTOS IDs. Patient IDs, row positions and fuzzy matching are never used to join predictions. If a renamed image path disagrees with the actual ID, supply `original_image_id` or `source_image_id` explicitly. Duplicate IDs, conflicting labels, split overlap, missing train/val lists for APTOS, invalid grades, or incomplete test coverage block aggregate metrics. Fallback counts can use unambiguous independent label joins even when an export is partial, with coverage stated.

The accepted reference is **Qwen3-VL-8B Retraining 2**. Dr. Noor's reported **80.65% accuracy, 0.8915 QWK and 0.9362 macro ROC-AUC** are printed in a separate reported-results section. They are never filled in as recomputed values or used to identify a run. Every verified export is named separately; its relationship to the accepted run remains unverified unless explicitly documented.

Current saved-code findings and confirmed historical cases are separate. Inline `raw_output`, saved `parse_ok=false`, exported `y_pred=0` and independently joined labels can confirm saved fallback cases; the report counts the Messidor-2 Mild → Normal contribution. Replaying the recognized current regex without historical linkage only counts parser-invalid candidates. Raw-only files remain pending unless explicitly joined to their matching prediction export. Color conversion and augmentation excerpts include source/cell/line locations and require producing-revision/call-path review. Image availability is based on paths, not a guarantee that the originals are untouched.

## Optional explicit provenance

Ambiguous exports may have a local sidecar named **`<export filename>.audit.json`**, for example `aptos_test.predictions.csv.audit.json`. It is optional for accuracy/QWK when split and label evidence already verifies the cohort. The sidecar's `prediction_export_sha256` must match the export before its claims are used. Each linked file must lie within the configured roots and have its corresponding SHA-256. Paths are relative to the prediction export's folder or absolute local paths. These hashes link files; they do not prove who generated them.

Example fields (replace illustrative hashes with actual file hashes; do not invent evidence):

```json
{
  "prediction_export_sha256": "SHA256_OF_THIS_PREDICTION_CSV",
  "dataset": "aptos",
  "split": "test",
  "reference_run": "Qwen3-VL-8B Retraining 2",
  "label_file": "../splits/test.csv",
  "label_file_sha256": "SHA256_OF_TEST_CSV",
  "score_columns": ["p0", "p1", "p2", "p3", "p4"],
  "class_order": [0, 1, 2, 3, 4],
  "score_kind": "probability",
  "score_origin": "Saved full-answer likelihood softmax from the hash-linked producer",
  "producer_file": "../../finetuning/Qwen3/paper_reproduction.py",
  "producer_file_sha256": "SHA256_OF_PRODUCING_SOURCE"
}
```

Do not name an unrelated source revision as the producer. AUC requires the exact verified test IDs, all five classes, finite genuine grade scores, explicit grade order, and a hash-linked recognized producer. This version recognizes the existing `paper_reproduction.py` complete-answer likelihood exporter by static source inspection. It can also recover score provenance from that helper's same-directory `evaluation_manifest.json` and `<subset>.summary.json` when the saved source/label hashes, run identity and row counts agree. That manifest links source and cohort but does not hash the prediction CSV; this limitation is reported explicitly. Other scoring implementations are inventoried and left for manual provenance review; the worker does not run them. One-hot vectors, hard predictions and a single confidence value cannot supply AUC. Scores in a separate file can be linked with `score_file` and `score_file_sha256`; that file needs exactly matching unique IDs and dataset metadata/columns.

For separately saved raw responses, use `raw_file` and `raw_file_sha256` with exact matching IDs. To establish fallback without a saved `parse_ok` flag, additionally provide `parser_file`, `parser_file_sha256`, `producer_file`, `producer_file_sha256` and `fallback_to_grade0: true`. The worker only recognizes the inspected `parse_pred_label` AST/regex and the evaluator's saved Grade-0 fallback branch. It never imports or executes those files. A different historical parser requires manual review. Preserve the original raw output rather than rewriting it to the current format.

Rank-16 CLAHE reuse is deliberately conservative. Adapter rank and weight-file presence are recorded without loading weights. A wrong rank requires rebuilding for a rank-16 comparison. Rank 16 plus a CLAHE path hint leaves **not enough evidence to decide** until exact splits, labels, image lineage, color handling, augmentation routing and producing revision are established. The report states what is available to reuse or rescore and which files to add next.

## Preparation review

The notebook follows the reference orchestrator's `Popen` / `sys.executable` / `-u` argument-list pattern and `paper_reproduction.py::run_live` interruption cleanup. Static inspection checks: one launch site; foreground stdout/stderr streaming; normal wait; terminate → timed wait → kill → wait; pipe close in `finally`; exit status and report path displayed; empty saved outputs/execution counts; configuration and imports have no launch side effects. No runtime validation has been performed.
