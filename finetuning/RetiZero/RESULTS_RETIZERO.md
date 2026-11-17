# RetiZero reproduction results (`retizero_01`)

All 18 jobs of the RetiZero queue are finished (16 result jobs plus the two benchmark-only rescorings that duplicate the matched training runs' own test scores). Results folder: `D:\Afterimage_Experiments\retizero_runs\retizero_01` (`summary.md`, `summary.csv`, `energy_per_job.csv`). Metrics come from the pinned Qwen metric function (`finetuning/Qwen3/paper_reproduction.py`), so they are directly comparable with the Qwen tables.

The last 7 jobs (3 `clahe` runs, 4 `messidor_archived` scorings) ran on 2026-11-17, 07:04–07:20 UTC, after the two datasets below were built. Each finished with status `complete`. Every attempt folder of all 18 jobs holds `command.json`, `system_specs.json`, its `.log`, `utilization.csv` and `run_report.json`.

## Datasets built for the last 7 jobs

Builder: [`data/build_retizero_datasets.py`](data/build_retizero_datasets.py). It imports the existing builder's functions from `D:\Afterimage_Experiments\code\prepare_aptos.py` rather than copying them, and records hashes of its own file, the imported builder and every image.

| Build | Path | Images | Preprocessing |
|---|---|---|---|
| APTOS 224 CLAHE | `D:\Afterimage_Experiments\datasets\aptos_224_clahe\duplicate_excluded` (`manifest.json`, `pairing_audit.csv`, `clahe_on\{train,val,test}.csv`) | 2,786 / 357 / 361. The IDs, grades and order match `aptos_224_standard\duplicate_excluded` and the Qwen 512 build. | Original APTOS photo → Otsu crop → black square pad → 224 resize (INTER_AREA) → **CLAHE last** (LAB lightness, clip limit 2.0, 8×8 grid) |
| Messidor 224 (archived) | `D:\Afterimage_Experiments\datasets\messidor_224_archived` (`manifest.json`, `source_audit.csv`, `clahe_off\test.csv`) | 1,057, in the same order as the Qwen archived-Messidor cohort | Archived 512-px CLAHE-off PNG → 224 resize (INTER_AREA). No CLAHE. |

**CLAHE order.** The order is resize first, then CLAHE, which is the Qwen order: `Qwen3/preprocess_qwen.py` resizes to 512 and then calls `apply_clahe`. At 224 px the 8×8 grid gives 28-px tiles. The Qwen build has 64-px tiles at 512 px.

Checks the builder ran:
- All 3,504 rebuilt CLAHE-off images matched the existing standard 224 build byte for byte before CLAHE was applied.
- On 25 images, the CLAHE function reproduced the Qwen 512 CLAHE-on build exactly, through both `prepare_aptos.apply_clahe` and the app's `preprocess_qwen_array`.
- `checks()` passed for `clean_clahe`: the ordered IDs and grades are the same as the Qwen build.

**Messidor source.** The images come from `D:\recovered\DR_Messidor_Only_512_Full_Dataset\Res_512\Messidor_Standard`.
- All 1,057 files are byte-identical to `DR_Aptos_Messidor_Only_Full_Dataset\Res_512\Aptos_messidor_external`.
- The archived CLAHE images are exactly `CLAHE(Standard)`, checked on 50 images. This means Qwen's CLAHE-on cohort and this CLAHE-off cohort come from the same 512-px pixels.
- Labels and patient IDs agree with the Qwen cohort CSV on all 1,057 rows.
- Each row also carries `archive_split`, the archive's original 80/10/10 assignment.

## All jobs

Wall min is the training stage only for training runs (evaluation excluded) and the scoring stage for scoring jobs. Energy is integrated over the whole job window from `power_log.csv` (nvidia-smi samples every 5 s, GPU board power only). The two replay jobs ran before power logging started, so they have no energy figure.

| Stage | Job | Test images | Accuracy | QWK | Macro AUC | Best epoch / run | Wall min | Energy Wh | Mean W |
|---|---|---|---|---|---|---|---|---|---|
| replay | saved_checkpoint__aptos367_original | 367 | 82.29% | 0.9097 | 0.9451 | – | 0.3 | n/a | n/a |
| replay | saved_checkpoint__aptos361_clean | 361 | 82.55% | 0.9101 | 0.9462 | – | 0.3 | n/a | n/a |
| historical | historical_t42 | 367 | 80.38% | 0.9122 | 0.9491 | 6 / 10 | 5.3 | 18.41 | 192 |
| historical | historical_t43 | 367 | 81.47% | 0.8985 | 0.9490 | 7 / 11 | 5.6 | 19.62 | 202 |
| historical | historical_t44 | 367 | 80.11% | 0.9027 | 0.9418 | 2 / 6 | 3.0 | 11.29 | 210 |
| matched | matched_t42 | 361 | 80.89% | 0.8988 | 0.9494 | 10 / 14 | 6.7 | 24.87 | 217 |
| matched | matched_t43 | 361 | 77.56% | 0.8840 | 0.9402 | 5 / 9 | 4.2 | 15.80 | 215 |
| matched | matched_t44 | 361 | 78.67% | 0.9005 | 0.9383 | 3 / 7 | 3.4 | 12.45 | 211 |
| **clahe** | **clahe_t42** | 361 | 80.06% | 0.8996 | 0.9384 | 5 / 9 | 4.5 | 15.79 | 204 |
| **clahe** | **clahe_t43** | 361 | 80.33% | 0.8737 | 0.9384 | 5 / 9 | 4.4 | 16.39 | 214 |
| **clahe** | **clahe_t44** | 361 | 77.56% | 0.9015 | 0.9381 | 5 / 9 | 4.4 | 16.19 | 212 |
| benchmark | matched_t42__aptos361_clean | 361 | 80.89% | 0.8988 | 0.9494 | – | 0.3 | 0.34 | 75 |
| benchmark | matched_t43__aptos361_clean | 361 | 77.56% | 0.8840 | 0.9402 | – | 0.3 | 0.24 | 53 |
| benchmark | matched_t44__aptos361_clean | 361 | 78.67% | 0.9005 | 0.9383 | – | 0.3 | 0.30 | 64 |
| **messidor_archived** | **saved_checkpoint__messidor_archived** | 1057 | 50.24% | 0.5849 | 0.7633 | – | 0.6 | 0.96 | 102 |
| **messidor_archived** | **matched_t42__messidor_archived** | 1057 | 42.10% | 0.5096 | 0.7373 | – | 0.5 | 0.81 | 88 |
| **messidor_archived** | **matched_t43__messidor_archived** | 1057 | 32.45% | 0.4481 | 0.7388 | – | 0.6 | 0.94 | 101 |
| **messidor_archived** | **matched_t44__messidor_archived** | 1057 | 46.83% | 0.5310 | 0.7572 | – | 0.6 | 0.91 | 99 |

The benchmark rescorings reproduce the training workers' own test predictions; their metrics are identical.

## Comparison 1: CLAHE against no CLAHE (same split, same seeds)

| Seed | No CLAHE (matched) acc / QWK / AUC | CLAHE acc / QWK / AUC | Δ accuracy | Same grade | Correct only without / only with CLAHE |
|---|---|---|---|---|---|
| 42 | 80.89% / 0.8988 / 0.9494 | 80.06% / 0.8996 / 0.9384 | −0.83 pt | 320 / 361 | 20 / 17 |
| 43 | 77.56% / 0.8840 / 0.9402 | 80.33% / 0.8737 / 0.9384 | +2.77 pt | 301 / 361 | 20 / 30 |
| 44 | 78.67% / 0.9005 / 0.9383 | 77.56% / 0.9015 / 0.9381 | −1.11 pt | 326 / 361 | 17 / 13 |
| Mean | 79.04 ± 1.69 / 0.8944 / 0.9426 | 79.32 ± 1.53 / 0.8916 / 0.9383 | +0.28 pt | | |

**Headline:** CLAHE makes no measurable difference for RetiZero at 224 px.
- The mean accuracy change (+0.28 pt) is far smaller than the seed-to-seed spread (SD about 1.5–1.7 pt), and the sign flips between seeds.
- QWK and macro AUC are slightly lower with CLAHE (−0.003 and −0.004).
- The CLAHE runs were also more uniform: best epoch 5 of 9 for all three seeds.
- Energy per run was similar (16.1 Wh mean with CLAHE vs 17.7 Wh without), because both have similar run lengths.

## Comparison 2: archived Messidor (1,057 images; models trained on APTOS)

| Model | Images | Accuracy | QWK | Macro AUC |
|---|---|---|---|---|
| RetiZero saved app checkpoint (APTOS original split, notebook trial 3) | 1057 | 50.24% | 0.5849 | 0.7633 |
| RetiZero matched, mean of 3 seeds (APTOS clean split, CLAHE off, 224 px) | 1057 | 40.46% (42.10 / 32.45 / 46.83) | 0.4962 | 0.7444 |
| Qwen3-VL r16 CLAHE-on, 4-bit paper controls p1–p3 (final_01), mean | 1057 | 47.78% (43.52 / 48.16 / 51.66) | 0.4401 (0.18 / 0.51 / 0.63) | 0.7573 |
| Qwen3-VL bf16 r16 CLAHE-on p1–p3 (final_01), mean | 1057 | 49.83% (45.70 / 49.67 / 54.12) | 0.6021 | 0.7897 |
| Qwen3-VL zero-shot 4-bit | 1057 | 45.98% | 0.4536 | 0.7349 |

**Headlines:**
- On archived Messidor, the three freshly trained RetiZero runs (mean 40.5%) are below the Qwen paper controls (47.8%) in accuracy. Their QWK is higher (0.50 vs 0.44, because Qwen p1 collapses to 0.18), and their macro AUC is a little lower (0.744 vs 0.757).
- The saved app checkpoint does better than the retrained RetiZero runs (50.2%, QWK 0.58).
- The RetiZero seeds spread widely (32–47%). The worst seed, t43, predicts grade 1 for 635 of 1,057 images; the true count is 207.
- All RetiZero models under-call grade 2 on Messidor. The saved checkpoint predicts 181 grade-2 images against 290 true.
- The two models see different inputs here: Qwen scored the CLAHE-on 512-px export and RetiZero the CLAHE-off 224-px derivative of the same pixels. Each uses its own native preprocessing.

**Against the original RetiZero Messidor notebook** (`legacy/ICT302_Retizero_Training_Unbalanced_Messidor_2.ipynb`). That notebook printed accuracy 0.5849, QWK 0.7421 and macro AUC 0.8492 on 106 test images (macro precision 0.4324, recall 0.4457, F1 0.4144). That model was **trained on Messidor**, on the 80/10/10 split's 846 training images, and tested on its own 10% split. It is an in-domain result and is not comparable with the APTOS-trained scores above.

For reference, here are the APTOS-trained checkpoints on the same 106 test images. The archive's `test` split has the same grade counts as the notebook's printout (53/20/24/7/2), and its 512 build used the same `GroupShuffleSplit(random_state=42)` on the same records.

| Checkpoint | Accuracy | QWK | Macro AUC |
|---|---|---|---|
| Saved app checkpoint | 59.43% | 0.7131 | 0.8038 |
| matched_t42 | 51.89% | 0.6916 | 0.7927 |
| matched_t43 | 37.74% | 0.5352 | 0.7912 |
| matched_t44 | 56.60% | 0.7051 | 0.7979 |

The APTOS-trained saved checkpoint scores about the same on these 106 images as the Messidor-trained notebook model reported (59.4% vs 58.5% accuracy), but with lower QWK and AUC. With 106 images (7 grade 3, 2 grade 4) the difference is within noise.

## Caveats

1. **The archived Messidor cohort is not raw Messidor-2.** It is derived from a processed 512-px export. The raw images, the adjudicated grade file and the eye-pairing list are not on this PC (`docs/paper/MESSIDOR_SOURCES_FOUND.md`). The labels are the archive's derived labels and cannot be checked against the adjudication. Replace this build and rescore when a verified raw cohort exists. Scoring takes about 30 s per checkpoint.
2. **The Messidor route differs from the original RetiZero route.** The original Messidor notebook resized the raw crop directly to 224 (INTER_AREA). This build goes raw → 512 (INTER_AREA, PNG) → 224 (INTER_AREA). Measured on the 361 APTOS test images, where both routes can be built from the same originals:
   - mean absolute channel difference 0.18 grey levels
   - median per-image maximum difference 8, largest 19
   - no image pixel-identical

   The effect on predictions was not measured.
3. **RetiZero and Qwen do not see the same Messidor inputs.** Qwen saw CLAHE-on 512 px; RetiZero saw CLAHE-off 224 px. A CLAHE-on 224 Messidor branch was not built because the queue does not expect one. It could be scored with the CLAHE checkpoints if a matched-preprocessing Messidor comparison is wanted.
4. **CLAHE at 224 px is not the same CLAHE as at 512 px.** The function and settings are the same, but the 8×8 grid covers 28-px tiles instead of 64-px tiles. Enhancing at 512 and then shrinking was not tested.
5. **Three seeds only.** The CLAHE effect is within seed noise, and no significance test is claimed.
6. **Selection protocol.** RetiZero selects the epoch on validation accuracy with patience 4 (the original recipe). The Qwen runs use their own protocol.
7. **Saved checkpoint provenance.** The app checkpoint was trained on the original (not duplicate-excluded) split. Its clean-361 score is not leak-free, and its Messidor score comes from a model chosen as the best of a learning-rate search.
8. **Energy measurement.**
   - The figures are GPU board power from nvidia-smi only; CPU, RAM and disk are excluded.
   - Samples are taken every 5 s and integrated over each job window.
   - The two replay jobs have no energy figure.
   - Training energy covers training plus evaluation, while Wall min covers training only.
