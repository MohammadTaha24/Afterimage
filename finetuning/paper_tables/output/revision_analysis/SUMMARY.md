# Revision analyses

Built by `build_revision_analysis.py` from the saved per-image predictions only. CIs are 95% percentile bootstrap (2000 resamples). Δ rows use a hierarchical bootstrap that resamples the images (shared by both arms) and each arm's seeds. Per-seed exact McNemar tests with Holm adjustment are in the `*_pairs.csv` files. Seed numbers are only a label across models, so the hierarchical interval is the one to quote for Qwen vs RetiZero.

## Headline conditions (361 clean test images, mean ± SD over 3 seeds)

| Condition | Accuracy | QWK | Macro AUC | Balanced acc. | Referable DR sens. / spec. | ECE | Collapsed runs (≤2 grades) |
|---|---|---|---|---|---|---|---|
| Qwen r16 CLAHE-on lr2e-4 | 78.12% ± 2.22% | 0.894 ± 0.007 | 0.942 ± 0.004 | 65.63% ± 3.88% | 87.59% / 94.91% | 0.066 | 0 of 3 |
| Qwen r16 CLAHE-on lr1e-4 | 74.98% ± 2.08% | 0.880 ± 0.029 | 0.932 ± 0.007 | 61.97% ± 4.11% | 82.30% / 95.52% | 0.068 | 0 of 3 |
| Qwen r16 CLAHE-on lr5e-4 | 69.71% ± 13.46% | 0.670 ± 0.337 | 0.898 ± 0.043 | 46.20% ± 12.32% | 68.28% / 93.83% | 0.154 | 0 of 3 |
| Qwen r16 CLAHE-off lr5e-4 | 73.96% ± 3.05% | 0.859 ± 0.059 | 0.934 ± 0.011 | 61.96% ± 11.06% | 74.48% / 97.99% | 0.093 | 0 of 3 |
| RetiZero CLAHE-off | 79.04% ± 1.69% | 0.894 ± 0.009 | 0.943 ± 0.006 | 65.50% ± 2.10% | 86.21% / 96.14% | 0.066 | 0 of 3 |
| RetiZero CLAHE-on | 79.32% ± 1.53% | 0.892 ± 0.016 | 0.938 ± 0.000 | 65.49% ± 2.13% | 91.26% / 93.98% | 0.054 | 0 of 3 |
| Qwen zero-shot 4bit CLAHE-on | 46.54% ± n/a | 0.674 ± n/a | 0.773 ± n/a | 35.79% ± n/a | 100.00% / 64.81% | 0.364 | 0 of 1 |
| Qwen zero-shot 4bit CLAHE-off | 54.57% ± n/a | 0.735 ± n/a | 0.803 ± n/a | 38.86% ± n/a | 97.24% / 76.85% | 0.294 | 0 of 1 |

## Learning rate selected on validation loss only (G2, S8)

| Condition | Mean validation loss | Max validation loss | Collapsed | Test accuracy | Rank by val. loss | Rank by test acc. |
|---|---|---|---|---|---|---|
| Qwen r16 CLAHE-on lr1e-4 | 0.00791 | 0.00823 | 0 | 74.98% | 2 | 2 |
| Qwen r16 CLAHE-on lr2e-4 | 0.00768 | 0.00802 | 0 | 78.12% | 1 | 1 |
| Qwen r16 CLAHE-on lr5e-4 | 0.01051 | 0.01506 | 0 | 69.71% | 3 | 3 |

Validation loss alone ranks the learning rates in the same order as test accuracy.

Across all 63 fine-tuned Qwen runs, the 10 runs that predict at most 3 of the 5 grades have validation loss 0.01417 to 0.01989; the 53 runs that predict 4 or 5 grades have 0.00688 to 0.01066. The ranges do not overlap, so every failed run is detectable from validation data alone, without the test set. (`collapse_vs_validation.csv`)

## Qwen vs RetiZero (S12)

| Comparison | A | B | Δ accuracy [95% CI] | Δ QWK [95% CI] | Δ macro AUC [95% CI] |
|---|---|---|---|---|---|
| H1 proposed primary: Qwen (lr chosen by validation loss) vs RetiZero, native preprocessing | Qwen r16 CLAHE-on lr2e-4 | RetiZero CLAHE-off | -0.92 pt [-5.26, +3.14] | -0.001 [-0.027, +0.026] | -0.001 [-0.017, +0.013] |
| H2 Qwen original paper setting vs RetiZero, native preprocessing | Qwen r16 CLAHE-on lr5e-4 | RetiZero CLAHE-off | -9.33 pt [-23.55, +1.85] | -0.224 [-0.607, -0.010] | -0.044 [-0.091, -0.009] |
| H3 both CLAHE-off | Qwen r16 CLAHE-off lr5e-4 | RetiZero CLAHE-off | -5.08 pt [-9.88, -0.18] | -0.035 [-0.101, +0.017] | -0.009 [-0.026, +0.009] |
| H4 both CLAHE-on | Qwen r16 CLAHE-on lr2e-4 | RetiZero CLAHE-on | -1.20 pt [-5.54, +3.23] | +0.002 [-0.027, +0.033] | +0.003 [-0.010, +0.017] |

## CLAHE (item 2, S3, S9)

| Comparison | A | B | Δ accuracy [95% CI] | Δ QWK [95% CI] | Δ macro AUC [95% CI] |
|---|---|---|---|---|---|
| Qwen r8 CLAHE on vs off (lr5e-4) | Qwen r8 CLAHE-on lr5e-4 | Qwen r8 CLAHE-off lr5e-4 | +5.54 pt [-27.52, +45.24] | -0.182 [-0.783, +0.423] | +0.000 [-0.227, +0.265] |
| Qwen r16 CLAHE on vs off (lr5e-4) | Qwen r16 CLAHE-on lr5e-4 | Qwen r16 CLAHE-off lr5e-4 | -4.25 pt [-18.75, +7.20] | -0.189 [-0.560, +0.050] | -0.036 [-0.082, +0.000] |
| Qwen r32 CLAHE on vs off (lr5e-4) | Qwen r32 CLAHE-on lr5e-4 | Qwen r32 CLAHE-off lr5e-4 | -16.81 pt [-56.33, +15.06] | -0.034 [-0.617, +0.565] | -0.033 [-0.313, +0.251] |
| Qwen zero-shot 4-bit CLAHE on vs off | Qwen zero-shot 4bit CLAHE-on | Qwen zero-shot 4bit CLAHE-off | -8.03 pt [-12.47, -3.88] | -0.061 [-0.098, -0.026] | -0.031 [-0.054, -0.006] |
| RetiZero CLAHE on vs off | RetiZero CLAHE-on | RetiZero CLAHE-off | +0.28 pt [-3.14, +3.79] | -0.003 [-0.027, +0.023] | -0.004 [-0.016, +0.004] |

## LoRA rank, alpha fixed at 16 (item 1, S6)

| Comparison | A | B | Δ accuracy [95% CI] | Δ QWK [95% CI] | Δ macro AUC [95% CI] |
|---|---|---|---|---|---|
| Qwen rank 8 vs 16, CLAHE-on (lr5e-4, alpha 16) | Qwen r8 CLAHE-on lr5e-4 | Qwen r16 CLAHE-on lr5e-4 | -13.94 pt [-34.07, +6.74] | -0.260 [-0.760, +0.262] | -0.108 [-0.230, +0.033] |
| Qwen rank 32 vs 16, CLAHE-on (lr5e-4, alpha 16) | Qwen r32 CLAHE-on lr5e-4 | Qwen r16 CLAHE-on lr5e-4 | -20.22 pt [-59.02, +9.33] | -0.118 [-0.653, +0.384] | -0.115 [-0.377, +0.052] |
| Qwen rank 8 vs 16, CLAHE-off (lr5e-4, alpha 16) | Qwen r8 CLAHE-off lr5e-4 | Qwen r16 CLAHE-off lr5e-4 | -23.73 pt [-64.45, +1.57] | -0.267 [-0.842, +0.077] | -0.144 [-0.419, +0.008] |
| Qwen rank 32 vs 16, CLAHE-off (lr5e-4, alpha 16) | Qwen r32 CLAHE-off lr5e-4 | Qwen r16 CLAHE-off lr5e-4 | -7.66 pt [-22.07, +4.80] | -0.273 [-0.831, +0.067] | -0.118 [-0.334, +0.017] |

Rank means at lr 5e-4 (accuracy, mean ± SD; collapsed runs included):

- Qwen r8 CLAHE-on lr5e-4: 55.77% ± 17.19%, QWK 0.410 ± 0.461, 2 of 3 collapsed, mean of non-collapsed runs 75.07%
- Qwen r16 CLAHE-on lr5e-4: 69.71% ± 13.46%, QWK 0.670 ± 0.337, 0 of 3 collapsed, mean of non-collapsed runs 69.71%
- Qwen r32 CLAHE-on lr5e-4: 49.49% ± 35.77%, QWK 0.552 ± 0.479, 1 of 3 collapsed, mean of non-collapsed runs 70.08%
- Qwen r8 CLAHE-off lr5e-4: 50.23% ± 36.39%, QWK 0.593 ± 0.513, 1 of 3 collapsed, mean of non-collapsed runs 71.19%
- Qwen r16 CLAHE-off lr5e-4: 73.96% ± 3.05%, QWK 0.859 ± 0.059, 0 of 3 collapsed, mean of non-collapsed runs 73.96%
- Qwen r32 CLAHE-off lr5e-4: 66.30% ± 14.56%, QWK 0.586 ± 0.508, 1 of 3 collapsed, mean of non-collapsed runs 74.38%

## Learning rate and adapter (S7, S8)

| Comparison | A | B | Δ accuracy [95% CI] | Δ QWK [95% CI] | Δ macro AUC [95% CI] |
|---|---|---|---|---|---|
| Qwen r16 CLAHE-on lr2e-4 vs lr5e-4 | Qwen r16 CLAHE-on lr2e-4 | Qwen r16 CLAHE-on lr5e-4 | +8.40 pt [-3.05, +21.89] | +0.224 [+0.008, +0.581] | +0.043 [+0.009, +0.087] |
| Qwen r16 CLAHE-on lr1e-4 vs lr5e-4 | Qwen r16 CLAHE-on lr1e-4 | Qwen r16 CLAHE-on lr5e-4 | +5.26 pt [-6.28, +19.02] | +0.209 [-0.012, +0.555] | +0.033 [-0.001, +0.079] |
| Qwen r16 CLAHE-on lr2e-4 vs lr1e-4 | Qwen r16 CLAHE-on lr2e-4 | Qwen r16 CLAHE-on lr1e-4 | +3.14 pt [-1.02, +7.39] | +0.014 [-0.019, +0.054] | +0.010 [-0.002, +0.024] |
| S7 Qwen LoRA adapter (lr2e-4) vs no adapter, 4-bit CLAHE-on | Qwen r16 CLAHE-on lr2e-4 | Qwen zero-shot 4bit CLAHE-on | +31.58 pt [+25.39, +37.03] | +0.220 [+0.166, +0.275] | +0.169 [+0.135, +0.200] |
| S7 Qwen LoRA adapter (lr5e-4) vs no adapter, 4-bit CLAHE-on | Qwen r16 CLAHE-on lr5e-4 | Qwen zero-shot 4bit CLAHE-on | +23.18 pt [+8.31, +35.09] | -0.004 [-0.382, +0.232] | +0.126 [+0.073, +0.174] |
| S7 Qwen LoRA adapter (lr5e-4) vs no adapter, 4-bit CLAHE-off | Qwen r16 CLAHE-off lr5e-4 | Qwen zero-shot 4bit CLAHE-off | +19.39 pt [+13.11, +25.48] | +0.124 [+0.042, +0.198] | +0.131 [+0.101, +0.160] |

## Error structure (S13)

| Condition | Cohort | Exact | Off by 1 | Off by 2 | Off by ≥3 | Off-by-1 share of errors |
|---|---|---|---|---|---|---|
| Qwen r16 CLAHE-on lr5e-4 | aptos361 | 69.71% | 17.73% | 8.40% | 4.16% | 58.54% |
| Qwen r16 CLAHE-on lr2e-4 | aptos361 | 78.12% | 17.82% | 3.88% | 0.18% | 81.43% |
| Qwen r16 CLAHE-on lr2e-4 | messidor | 48.53% | 42.95% | 7.85% | 0.66% | 83.46% |
| Qwen r16 CLAHE-on lr5e-4 | messidor | 47.78% | 36.39% | 13.31% | 2.52% | 69.69% |
| Qwen r16 CLAHE-on lr2e-4 | messidor2 | 55.35% | 36.45% | 7.38% | 0.82% | 81.64% |
| Qwen r16 CLAHE-on lr5e-4 | messidor2 | 55.41% | 31.38% | 11.22% | 1.99% | 70.38% |
| RetiZero CLAHE-on | aptos361 | 79.32% | 15.88% | 4.43% | 0.37% | 76.79% |
| RetiZero CLAHE-off | aptos361 | 79.04% | 16.99% | 3.51% | 0.46% | 81.06% |
| RetiZero CLAHE-off | messidor | 40.46% | 48.47% | 10.60% | 0.47% | 81.41% |
| RetiZero CLAHE-on | messidor2 | 47.36% | 40.00% | 11.89% | 0.75% | 76.00% |
| RetiZero CLAHE-off | messidor2 | 40.21% | 50.31% | 8.93% | 0.55% | 84.14% |

## Messidor-2 (archived 1,057-image cohort)

- archived_images: 1057
- archived_patients: 529
- adjudicated_rows: 1748
- adjudicated_png_rows: 1058
- archived_matched_to_adjudicated: 1057
- archived_label_equals_adjudicated: 1057
- png_not_in_archive: 20060411_58550_0200_PP.png (gradable=0)
- archived_grade_counts: 468 207 290 71 21
- adjudicated_png_gradable_counts: 468 207 290 71 21
- adjudicated_all_gradable_counts: 1017 270 347 75 35
- pairs_exams_total: 874
- pairs_png_exams: 529
- pairs_both_images_archived: 528
- pairs_same_patient_id: 528
- pairs_one_image_archived: 1
- patient_clusters_after_merging_duplicates: 527

Mild → Normal (mean over seeds):

| Condition | Cohort | Mild predicted Normal | Mild recall | Normal predicted Mild | Mild-vs-Normal AUC |
|---|---|---|---|---|---|
| Qwen r16 CLAHE-on lr5e-4 | aptos361 | 18.10% | 47.62% | 5.71% | 0.974 |
| Qwen r16 CLAHE-on lr2e-4 | aptos361 | 3.81% | 73.33% | 3.87% | 0.983 |
| Qwen r16 CLAHE-on lr2e-4 | messidor | 71.66% | 26.73% | 15.46% | 0.623 |
| Qwen r16 CLAHE-on lr5e-4 | messidor | 83.57% | 15.30% | 10.40% | 0.538 |
| Qwen zero-shot 4bit CLAHE-on | aptos361 | 20.00% | 5.71% | 1.10% | 0.855 |
| Qwen zero-shot 4bit CLAHE-on | messidor | 66.18% | 0.48% | 2.35% | 0.547 |
| Qwen r16 CLAHE-on lr2e-4 | messidor2 | 71.36% | 26.05% | 16.75% | 0.566 |
| Qwen r16 CLAHE-on lr5e-4 | messidor2 | 83.21% | 15.06% | 12.72% | 0.487 |
| Qwen zero-shot 4bit CLAHE-on | messidor2 | 60.00% | 1.48% | 2.56% | 0.473 |
| RetiZero CLAHE-off | aptos361 | 5.71% | 74.29% | 2.95% | 0.987 |
| RetiZero CLAHE-off | messidor | 61.67% | 36.55% | 38.96% | 0.500 |
| RetiZero saved app checkpoint | aptos361 | 8.57% | 71.43% | 2.76% | 0.988 |
| RetiZero saved app checkpoint | aptos367 | 8.11% | 70.27% | 2.76% | 0.989 |
| RetiZero saved app checkpoint | messidor | 79.23% | 14.01% | 15.60% | 0.520 |
| RetiZero CLAHE-off | messidor2 | 54.69% | 42.59% | 47.85% | 0.464 |
| RetiZero saved app checkpoint | messidor2 | 73.33% | 21.11% | 22.62% | 0.477 |

Messidor head-to-head (patient-clustered):

| Comparison | A | B | Δ accuracy [95% CI] | Δ QWK [95% CI] | Δ macro AUC [95% CI] |
|---|---|---|---|---|---|
| Messidor: Qwen lr2e-4 vs RetiZero, native preprocessing | Qwen r16 CLAHE-on lr2e-4 | RetiZero CLAHE-off | +8.07 pt [+0.57, +16.75] | +0.107 [+0.045, +0.173] | +0.054 [+0.028, +0.083] |
| Messidor: Qwen lr5e-4 vs RetiZero, native preprocessing | Qwen r16 CLAHE-on lr5e-4 | RetiZero CLAHE-off | +7.32 pt [-0.60, +16.40] | -0.056 [-0.300, +0.139] | +0.013 [-0.018, +0.047] |

## Verified Messidor-2, 1,744 gradable images (exam-clustered)

| Comparison | A | B | Δ accuracy [95% CI] | Δ QWK [95% CI] | Δ macro AUC [95% CI] |
|---|---|---|---|---|---|
| Messidor-2: Qwen lr2e-4 vs RetiZero, native preprocessing | Qwen r16 CLAHE-on lr2e-4 | RetiZero CLAHE-off | +15.14 pt [+5.57, +27.24] | +0.134 [+0.063, +0.208] | +0.068 [+0.041, +0.096] |
| Messidor-2: Qwen lr5e-4 vs RetiZero, native preprocessing | Qwen r16 CLAHE-on lr5e-4 | RetiZero CLAHE-off | +15.19 pt [+6.07, +27.21] | -0.029 [-0.259, +0.157] | +0.027 [-0.007, +0.068] |
| Messidor-2: both CLAHE-off | Qwen r16 CLAHE-off lr5e-4 | RetiZero CLAHE-off | +16.78 pt [+6.90, +28.92] | +0.053 [-0.027, +0.139] | +0.013 [-0.018, +0.046] |
| Messidor-2: both CLAHE-off, Qwen lr2e-4 | Qwen r16 CLAHE-off lr2e-4 | RetiZero CLAHE-off | +15.44 pt [+5.95, +27.49] | +0.099 [+0.031, +0.172] | +0.051 [+0.027, +0.076] |
| Messidor-2: both CLAHE-on | Qwen r16 CLAHE-on lr2e-4 | RetiZero CLAHE-on | +7.99 pt [+4.08, +12.34] | +0.134 [+0.084, +0.186] | +0.053 [+0.028, +0.079] |
| Messidor-2: Qwen adapter (lr2e-4) vs zero-shot, CLAHE-on | Qwen r16 CLAHE-on lr2e-4 | Qwen zero-shot 4bit CLAHE-on | +11.54 pt [+7.38, +15.85] | +0.299 [+0.247, +0.349] | +0.089 [+0.065, +0.114] |
| Messidor-2: RetiZero CLAHE on vs off | RetiZero CLAHE-on | RetiZero CLAHE-off | +7.15 pt [-1.87, +19.30] | +0.001 [-0.056, +0.065] | +0.014 [-0.003, +0.033] |

## LoRA rank at the validation-selected lr 2e-4

| Comparison | A | B | Δ accuracy [95% CI] | Δ QWK [95% CI] | Δ macro AUC [95% CI] |
|---|---|---|---|---|---|
| Qwen rank 8 vs 16, CLAHE-on (lr2e-4, alpha 16) | Qwen r8 CLAHE-on lr2e-4 | Qwen r16 CLAHE-on lr2e-4 | +0.09 pt [-3.51, +3.97] | +0.002 [-0.022, +0.028] | -0.004 [-0.020, +0.011] |
| Qwen rank 32 vs 16, CLAHE-on (lr2e-4, alpha 16) | Qwen r32 CLAHE-on lr2e-4 | Qwen r16 CLAHE-on lr2e-4 | -3.60 pt [-8.03, +0.83] | -0.007 [-0.045, +0.021] | -0.002 [-0.013, +0.009] |
| Qwen rank 8 vs 16, CLAHE-off (lr2e-4, alpha 16) | Qwen r8 CLAHE-off lr2e-4 | Qwen r16 CLAHE-off lr2e-4 | +1.11 pt [-4.25, +6.93] | +0.005 [-0.030, +0.034] | +0.002 [-0.011, +0.014] |
| Qwen rank 32 vs 16, CLAHE-off (lr2e-4, alpha 16) | Qwen r32 CLAHE-off lr2e-4 | Qwen r16 CLAHE-off lr2e-4 | +0.28 pt [-4.52, +6.28] | +0.003 [-0.020, +0.031] | +0.002 [-0.012, +0.015] |
| Qwen rank 8 vs 16 with alpha = rank (alpha/r = 1), CLAHE-on (lr2e-4) | Qwen r8 CLAHE-on lr2e-4 alpha8 | Qwen r16 CLAHE-on lr2e-4 | +0.55 pt [-3.14, +4.16] | +0.007 [-0.016, +0.032] | -0.003 [-0.016, +0.010] |
| Qwen rank 32 vs 16 with alpha = rank (alpha/r = 1), CLAHE-on (lr2e-4) | Qwen r32 CLAHE-on lr2e-4 alpha32 | Qwen r16 CLAHE-on lr2e-4 | -5.36 pt [-11.45, +1.39] | -0.014 [-0.056, +0.023] | -0.017 [-0.039, +0.006] |
| CLAHE on vs off, rank 16 (lr2e-4) | Qwen r16 CLAHE-on lr2e-4 | Qwen r16 CLAHE-off lr2e-4 | +2.31 pt [-3.33, +8.40] | -0.001 [-0.028, +0.025] | +0.002 [-0.011, +0.018] |

## Archived 1,057 vs the same images rebuilt from raw

| model | condition | seed | archived_job | verified_job | shared_images | same_predicted_grade | max_abs_probability_difference | accuracy_archived | accuracy_verified_same_images | qwk_archived | qwk_verified_same_images | accuracy_verified_all_1744 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | Qwen r16 CLAHE-off lr2e-4 | 42 | r16_off_alpha16_lr0p0002_t42_a3407__messidor_archived_off | r16_off_alpha16_lr0p0002_t42_a3407__messidor2_off | 1057 | 1057 | 0.0000 | 0.5033 | 0.5033 | 0.5823 | 0.5823 | 0.5883 |
| Qwen | Qwen r16 CLAHE-off lr2e-4 | 43 | r16_off_alpha16_lr0p0002_t43_a3408__messidor_archived_off | r16_off_alpha16_lr0p0002_t43_a3408__messidor2_off | 1057 | 1057 | 0.0000 | 0.4816 | 0.4816 | 0.5782 | 0.5782 | 0.5579 |
| Qwen | Qwen r16 CLAHE-off lr2e-4 | 44 | r16_off_alpha16_lr0p0002_t44_a3409__messidor_archived_off | r16_off_alpha16_lr0p0002_t44_a3409__messidor2_off | 1057 | 1057 | 0.0000 | 0.4655 | 0.4655 | 0.5605 | 0.5605 | 0.5235 |
| Qwen | Qwen r16 CLAHE-on lr2e-4 | 42 | r16_on_lr0p0002_t42_a3407__messidor_archived | r16_on_lr0p0002_t42_a3407__messidor2_on | 1057 | 1057 | 0.0000 | 0.4957 | 0.4957 | 0.6277 | 0.6277 | 0.5866 |
| Qwen | Qwen r16 CLAHE-on lr2e-4 | 43 | r16_on_lr0p0002_t43_a3408__messidor_archived | r16_on_lr0p0002_t43_a3408__messidor2_on | 1057 | 1057 | 0.0000 | 0.4995 | 0.4995 | 0.6055 | 0.6055 | 0.5298 |
| Qwen | Qwen r16 CLAHE-on lr2e-4 | 44 | r16_on_lr0p0002_t44_a3409__messidor_archived | r16_on_lr0p0002_t44_a3409__messidor2_on | 1057 | 1057 | 0.0000 | 0.4607 | 0.4607 | 0.5754 | 0.5754 | 0.5442 |
| Qwen | Qwen r16 CLAHE-on lr5e-4 | 42 | r16_on_p1__messidor_archived | r16_on_p1__messidor2_on | 1057 | 1057 | 0.0000 | 0.4352 | 0.4352 | 0.1829 | 0.1829 | 0.5579 |
| Qwen | Qwen r16 CLAHE-on lr5e-4 | 43 | r16_on_p2__messidor_archived | r16_on_p2__messidor2_on | 1057 | 1057 | 0.0000 | 0.4816 | 0.4816 | 0.5090 | 0.5090 | 0.5459 |
| Qwen | Qwen r16 CLAHE-on lr5e-4 | 44 | r16_on_p3__messidor_archived | r16_on_p3__messidor2_on | 1057 | 1057 | 0.0000 | 0.5166 | 0.5166 | 0.6285 | 0.6285 | 0.5585 |
| Qwen | Qwen zero-shot 4bit CLAHE-on | <NA> | zeroshot_4bit__messidor_archived | zeroshot_4bit__messidor2_on | 1057 | 1057 | 0.0000 | 0.4598 | 0.4598 | 0.4536 | 0.4536 | 0.4381 |
| RetiZero | RetiZero CLAHE-off | 42 | matched_t42__messidor_archived | matched_t42__messidor2_224_off | 1057 | 1031 | 0.1676 | 0.4210 | 0.4257 | 0.5096 | 0.5161 | 0.4323 |
| RetiZero | RetiZero CLAHE-off | 43 | matched_t43__messidor_archived | matched_t43__messidor2_224_off | 1057 | 1042 | 0.0991 | 0.3245 | 0.3273 | 0.4481 | 0.4558 | 0.2810 |
| RetiZero | RetiZero CLAHE-off | 44 | matched_t44__messidor_archived | matched_t44__messidor2_224_off | 1057 | 1033 | 0.1779 | 0.4683 | 0.4664 | 0.5310 | 0.5291 | 0.4931 |
| RetiZero | RetiZero saved app checkpoint | <NA> | saved_checkpoint__messidor_archived | saved_checkpoint__messidor2_224_off | 1057 | 1033 | 0.2723 | 0.5024 | 0.5052 | 0.5849 | 0.5834 | 0.5407 |

## Near-duplicate sensitivity (S10)

All 361 test images by grade 0-4: 181 35 97 18 30. Removal is not random, so compare against the excluded grade mix.

| pHash threshold | Excluded test images (grades 0-4) | Condition | Accuracy all → kept (mean) | QWK all → kept |
|---|---|---|---|---|
| ≤0 | 6 (2 0 0 2 2) | Qwen r16 CLAHE-on lr2e-4 | 78.12% → 78.69% | 0.894 → 0.894 |
| ≤0 | 6 (2 0 0 2 2) | Qwen r16 CLAHE-on lr5e-4 | 69.71% → 70.33% | 0.670 → 0.669 |
| ≤0 | 6 (2 0 0 2 2) | RetiZero CLAHE-off | 79.04% → 79.72% | 0.894 → 0.895 |
| ≤0 | 6 (2 0 0 2 2) | RetiZero CLAHE-on | 79.32% → 79.72% | 0.892 → 0.891 |
| ≤2 | 32 (20 2 6 2 2) | Qwen r16 CLAHE-on lr2e-4 | 78.12% → 78.12% | 0.894 → 0.893 |
| ≤2 | 32 (20 2 6 2 2) | Qwen r16 CLAHE-on lr5e-4 | 69.71% → 69.00% | 0.670 → 0.662 |
| ≤2 | 32 (20 2 6 2 2) | RetiZero CLAHE-off | 79.04% → 79.33% | 0.894 → 0.894 |
| ≤2 | 32 (20 2 6 2 2) | RetiZero CLAHE-on | 79.32% → 79.23% | 0.892 → 0.890 |
| ≤4 | 120 (81 9 23 2 5) | Qwen r16 CLAHE-on lr2e-4 | 78.12% → 74.00% | 0.894 → 0.878 |
| ≤4 | 120 (81 9 23 2 5) | Qwen r16 CLAHE-on lr5e-4 | 69.71% → 63.90% | 0.670 → 0.628 |
| ≤4 | 120 (81 9 23 2 5) | RetiZero CLAHE-off | 79.04% → 75.38% | 0.894 → 0.877 |
| ≤4 | 120 (81 9 23 2 5) | RetiZero CLAHE-on | 79.32% → 76.76% | 0.892 → 0.883 |
