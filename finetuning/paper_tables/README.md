# Paper tables

`build_paper_tables.py` rebuilds every paper table from the saved Qwen3-VL prediction exports. It runs on the CPU, needs no GPU and only reads its inputs. It writes only to `output/`.

## Run

From the repo root (`D:\Afterimage_Reproduction`):

```powershell
C:\Users\moht2\.conda\envs\afterimage_env\python.exe -I -B finetuning\paper_tables\test_metrics.py
C:\Users\moht2\.conda\envs\afterimage_env\python.exe -I -B finetuning\paper_tables\build_paper_tables.py
```

Optional arguments, with their defaults:

| Argument | Default |
|---|---|
| `--runs-root` | `D:\Afterimage_Experiments\runs` |
| `--original-split-root` | `finetuning\Qwen3\artifacts\paper_reproduction\training_batches\run02_reproduction_01` |
| `--reference-root` | `...\paper_reproduction\reference_run_02` |
| `--datasets-root` | `D:\Afterimage_Experiments\datasets` |
| `--output-dir` | `finetuning\paper_tables\output` (must stay inside this folder) |

The script needs only numpy and pandas. The legacy copy on C: is not used.

## Result sets

Sets are kept apart. Every table carries a `result_set` column.

| Set | What | Test n | Scoring |
|---|---|---|---|
| A | Duplicate-excluded split, 4-bit base, ranks 8/16/32 × CLAHE on/off × 3 seed pairs | 361 | FP16 |
| B | Previous train/val split, rank 16, CLAHE on, 4-bit base | 361 (same IDs as A) | FP16 |
| C | BF16 base, rank 16, CLAHE on, 16 layers offloaded to CPU | 361 | FP16 |
| D | Original 2929/366/367 split, rank 16, CLAHE on/off, three same-seed pairs (42/3407) | 367 | BF16 |
| E | Retraining 2 saved-adapter replay. APTOS, plus the archived 1,057-image Messidor export | 367 / 1,057 | FP16 |

The script discovers runs from the folders. It reads rank, alpha, CLAHE, split, base precision and seeds from each run's saved configuration:
- sets A, B and C: `pilot_configuration.json`
- set D: `configuration.json`, `run_info.json` and the dataset manifest's `clahe` field
- set E: `reference_run_02/paper_training_configuration.json`

For sets A, B and C the scoring precision is not recorded in `score_result.json`. It is FP16 because `qwen_pilot.py` `score()` calls `model.half()`; the `scoring_precision_source` column says this.

## Checks (the build stops on any failure)

- **Inputs:** every expected file exists, and every A/B/C run has `latest_attempt.json` status `passed`.
- **Predictions:** IDs are unique, every probability vector is finite and sums to 1, and `y_pred` equals the probability argmax.
- **Recomputation against saved values** (absolute tolerance 1e-9):
  - accuracy, QWK, macro AUC and per-grade AUC
  - n
  - the saved confusion matrix, where one is saved
- **Prediction-file SHA-256** against `score_result.json` (sets A, B, C).
- **Set E:** the replay's IDs and grades must match the archived export, and the replay's metrics must match the archived summaries.

## Metrics

- **QWK:** weights (i−j)²/16. It is undefined (NaN) when the expected weighted disagreement is zero. A constant predictor gives QWK = 0.
- **AUC:** one-vs-rest for each grade, using that grade's own probability column. Ties count as one half (Mann-Whitney). Macro AUC is the mean of the five per-grade AUCs.
- **Bootstrap CIs:** 2,000 resamples of the test set with replacement, seed 20261115, percentile 2.5/97.5.
  - A resample with a grade absent has an undefined AUC for that grade. That resample's macro AUC is treated as undefined and excluded; the saved scorers also require all five grades.
  - Resamples with undefined QWK are excluded from the QWK CI.
  - `boot_auc_undefined` and `boot_qwk_undefined` count the excluded resamples. Both are 0 in the current outputs.
- **Class-wise:**
  - sensitivity = TP/(TP+FN)
  - specificity = TN/(TN+FP)
  - precision = TP/(TP+FP), NaN when the grade is never predicted
  - F1 = 2TP/(2TP+FP+FN)
  - Per-condition means skip NaN values; the `*_n_defined` columns give how many runs contributed.
- **Run flags:** `degenerate` = the run predicts a single grade. `grades_never_predicted` lists grades with zero predictions. No run is ever dropped.

## Outputs (`output/`)

- `SUMMARY.md`: headline tables 2–5, pooled confusion matrices, Messidor, efficiency and caveats.
- `manifest.json`: input files with SHA-256, output files with rows and SHA-256, script and test SHA-256, environment versions, arguments and a timestamp. The manifest is the only file that changes between identical runs.
- `all_sets_per_run_metrics.csv`: table 1 for sets A–E, with a `result_set` column.
- `split_overlap_facts.csv`: exact-duplicate overlap counts, from the saved pairing audits.
- Files in `aptos_361_clean/` (sets A, B, C), `aptos_367_original_split/` (set D) and `retraining2_replay/` (set E):
  - `per_run_metrics.csv` (table 1)
  - `condition_summary.csv` (table 2)
  - `confusion_matrices_per_run.csv` and `confusion_matrices_pooled.csv` (table 6)
  - `classwise_per_run.csv` and `classwise_condition_mean.csv` (table 7)
- Additional files in `aptos_361_clean/`:
  - `rank_sensitivity.csv` (table 3)
  - `clahe_ablation_paired_by_seed.csv` (table 4)
  - `bf16_vs_4bit_paired_by_seed.csv` (table 5)
- Additional file in `aptos_367_original_split/`:
  - `clahe_pairs_same_seed_repeats.csv` (table 4, same-seed repeats)
- Additional files in `retraining2_replay/`:
  - `messidor_archived_*.csv` (table 8)
  - `misclassified_*.csv` (table 9; image IDs and probabilities only, no images)
- `efficiency/`: `qwen_efficiency_per_run.csv` and `qwen_efficiency_by_condition.csv` (table 10, Qwen only).

## Caveats

- **Workbook:** `paper_tables.xlsx` is not produced, because openpyxl is not installed in `afterimage_env`.
- **Efficiency:**
  - Times cover the final attempt only.
  - Seconds per image is an upper bound: it includes model load and five forward passes per image.
  - VRAM is the whole-device peak from nvidia-smi.
  - Total parameters are copied exactly from the Unsloth log line. The 4-bit runs log 8.8 billion and the BF16 runs 6.1 billion, so the totals need an independent count.
  - The set D logs have no torch peak values.
  - Set E has no training record on this machine.
- **Not measured:** RetiZero. No matched RetiZero run exists.
- **Messidor cohort:** the Messidor figures use the archived 1,057-image export, not an independently verified cohort.
