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
| F | Final queue `final_01`: zero-shot, inference benchmark, original adapter audit, archived Messidor for all models, lr and alpha studies (rank 16, CLAHE on) | 361 / 367 / 1,057 | FP16; one labelled FP32 fallback |

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
- `final_01/` (set F, `--final-root`, default `D:\Afterimage_Experimentsinal_runsinal_01`; `--skip-final` leaves it out): `SUMMARY.md` (tables F1-F6), `per_job_metrics.csv`, `zero_shot.csv`, `lr_study_*.csv`, `alpha_study_*.csv`, `inference_benchmark*.csv`, `original_adapter_audit.csv`, `messidor_all_models_*.csv`, confusion and class-wise files. Every row of the queue's `summary.csv` and every group row of its `summary.md` is checked against the recomputed values. LoRA scaling is alpha/r from `adapter_config.json`; `train_result.json`'s `effective_scaling` is stored as 16/rank and is wrong for alpha != 16 (kept as `recorded_effective_scaling` with a `scaling_note`). The alpha-32 seed-43 run's FP16 score overflowed; its FP32-fallback rows are flagged and kept out of FP16-only means.

## Revision analyses (added 22 November 2026)

Two CPU-only scripts answer Dr. Noor's revision plan and the reviewer points from the saved predictions. Neither retrains or rescores anything, and neither edits a hash-pinned file. Both need numpy, pandas and scipy; `build_image_audits.py` also needs opencv-python-headless and Pillow.

```powershell
python -I -B finetuning\paper_tables\test_revision_analysis.py
python -I -B finetuning\paper_tables\build_image_audits.py --split-dir <run folder with aptos_{train,val,test}.csv> --aptos-512 <...\Res_512\Aptos_messidor_external> --aptos-raw <APTOS train_images> --image-sha <evidence\analysis\image_sha256.csv> --messidor-grades <messidor_data.csv> --image-out <folder outside the repo>
python -I -B finetuning\paper_tables\build_revision_analysis.py --experiments-root D:\Afterimage_Experiments --messidor-grades <messidor_data.csv> --messidor-pairs <pairs.csv> --original-split D:\Afterimage_Experiments\reference\aptos_split --near-duplicates finetuning\paper_tables\output\revision_analysis\image_audits\near_duplicate_test_ids.csv
```

`--experiments-root` also accepts the `evidence\` folder built by `build_evidence_folder.py`, because it holds every prediction file.

**`build_revision_analysis.py`** writes to `output/revision_analysis/`. Open `SUMMARY.md` first.

| Output | Answers |
|---|---|
| `reproduction_check.csv`, `image_alignment_check.csv` | Every run's accuracy, QWK and macro AUC recomputed and matched to the committed tables or the saved run summaries (1e-8). Per-grade AUC re-derived independently with scipy's Mann-Whitney U. Every run on a cohort scores the same images with the same labels. The build stops on any mismatch. |
| `per_run_metrics.csv`, `condition_summary.csv` | Accuracy, QWK and macro AUC with bootstrap CIs, plus balanced accuracy, macro-F1, referable DR (grade ≥ 2) AUC, sensitivity and specificity, adjacent vs distant errors, Mild → Normal, ECE, Brier and NLL. Collapsed = at most 2 grades predicted. Collapsed runs are never dropped. |
| `classwise_per_run.csv`, `classwise_condition_mean.csv` | Per-grade sensitivity, specificity, precision, F1 and AUC for every Qwen and RetiZero run (item 4) |
| `rank_*.csv` | LoRA rank 8 / 16 / 32 at lr 5e-4, alpha 16 (item 1, S6) |
| `lr_selection_by_validation_loss.csv`, `collapse_vs_validation.csv` | The learning rate chosen from validation loss alone, and whether validation loss flags the collapsed runs (G2, S8) |
| `clahe_*.csv` | CLAHE on vs off for Qwen ranks 8/16/32, Qwen zero-shot and RetiZero (item 2, S9) |
| `training_*.csv` | Learning-rate pairs, and the LoRA adapter against zero-shot (S7) |
| `head_to_head_*.csv` | Qwen vs RetiZero on the 361 images, H1 to H4 (S12) |
| `messidor_*.csv`, `mild_vs_normal.csv` | Messidor label audit against the adjudicated grades, patient-clustered CIs, and the Mild-vs-Normal analysis (item 4) |
| `error_structure.csv` | Off-by-1, off-by-2 and larger errors (S13) |
| `misclassified_headline.csv` | Images misclassified by the headline conditions, with each seed's prediction (S11) |
| `class_counts.csv` | Images per split and grade; expected per-epoch draws under Qwen's inverse-frequency sampler; RetiZero's weighted loss and online augmentation (S9) |
| `calibration_reliability_bins.csv` | Reliability diagram data for the headline conditions |
| `efficiency_final.csv` | Qwen vs RetiZero parameters, training time, peak memory and inference time, all on the same RTX 5070 Ti. RetiZero training now comes from the full `retizero_01` runs (item 3, S14). |
| `grade0_fallback_check.csv` | Unparseable generations and their Grade-0 fallback for every generation check (item 4) |
| `near_duplicate_sensitivity.csv` | The headline metrics recomputed without the near-duplicate test images (S10) |

Statistics:
- 95% percentile bootstrap, 2,000 resamples, seed 20261122.
- Paired comparisons give a per-seed exact McNemar test, Holm-adjusted within each family, with a paired bootstrap CI for each Δ.
- The aggregate Δ uses a hierarchical bootstrap: it resamples the shared images and each arm's seeds. Seed numbers are only a label across models, so this is the interval to quote for Qwen vs RetiZero.
- Messidor intervals resample patients (both eyes together).

**`build_image_audits.py`** writes ID-level CSVs to `output/revision_analysis/image_audits/`. Its contact sheets go to `--image-out`, which must be outside the repository: they are dataset images and must not be committed.

| Output | Answers |
|---|---|
| `colour_conversion_check.csv` | On the 361 raw APTOS test photos, the app's preprocessing (RGB → LAB) gives pixels identical to the dataset builder (BGR → LAB), with CLAHE on and off and at 224 px. A deliberately wrong channel order is the negative control (item 2, S3). |
| `near_duplicate_pairs.csv`, `near_duplicate_test_ids.csv` | 64-bit DCT perceptual hash, cross-split pairs at Hamming distance ≤ 4. As a check of the method, byte-identical pairs hash to 0 (S10). |
| `messidor_duplicate_pairs.csv` | Pixel-identical images in the archived Messidor cohort, with patient IDs and adjudicated grades. With `--messidor-raw-parts`, also checked against the raw Messidor-2 zip. |
| `misclassified_examples_index.csv` | Images misclassified by all three seeds of both Qwen (lr 2e-4) and RetiZero, one or two per error type (S11) |

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
