# Evidence for Dr. Noor's revision plan

22 November 2026; the Qwen rank section (point 1) and the S8 failure check updated 24 November 2026 with the lr 2e-4 rank and alpha grid (`final_runs/final_02`, 27 new adapters). Apart from that grid, this note answers the revision plan from results that already existed. Every number comes from `finetuning/paper_tables/output/revision_analysis/` (open `SUMMARY.md` there). Those tables are rebuilt from the saved per-image predictions by `build_revision_analysis.py` and `build_image_audits.py` (see `finetuning/paper_tables/README.md`).

All results are on the 361 clean APTOS test images and give the mean ± SD over three seeds, unless stated otherwise. CIs are 95% bootstrap intervals. "Δ" is a hierarchical bootstrap over images and seeds.

## Integrity checks run first

| Check | Result | File |
|---|---|---|
| Recompute every run's accuracy, QWK and macro AUC from its predictions | All 85 runs with a stored reference match the committed tables or the saved run summaries (tolerance 1e-8). The 27 runs of the lr 2e-4 grid have no stored reference in this check; they match the queue's own `final_runs/final_02/summary.csv` to within 1e-10 (checked separately, 24 November 2026). | `reproduction_check.csv` |
| Same images, order and labels for every run in a cohort | All runs on APTOS-361, APTOS-367 and Messidor agree | `image_alignment_check.csv` |
| ROC-AUC uses each grade's own probability column | An independent scipy Mann-Whitney computation agrees exactly (max difference 0) for every run and grade | `reproduction_check.csv` |
| Messidor labels | All 1,057 archived labels equal the Krause 2018 adjudicated ICDR grades. The 1,058th Messidor-Original PNG is the only ungradable PNG. The eye pairing agrees for all 528 complete exams. | `messidor_label_audit.csv` |

The earlier concern that "our Messidor labels do not match the published grade distribution" is resolved. Our cohort is the PNG (Messidor-Original) half of Messidor-2, and its grade mix (468/207/290/71/21) is exactly the adjudicated mix for those PNGs. The comparison had been made against all 1,744 gradable images, PNG and JPG together.

## 1. LoRA rank (reviewer comments 5 and 7; S6)

Ranks 8, 16 and 32 were trained at the validation-selected learning rate of 2e-4 (see S8), CLAHE on and off, three seeds each, with everything else fixed. Alpha 16 was kept for every rank, and rank 8 and 32 were also trained with alpha = rank, so that rank can be separated from LoRA scaling (alpha/r).

A run **fails** when it predicts at most 3 of the 5 grades. This one definition is used throughout this note.

| Rank, alpha, CLAHE | Accuracy | QWK | Macro AUC | Failed runs |
|---|---|---|---|---|
| 8, 16, on | 78.21 ± 1.05% | 0.896 ± 0.001 | 0.937 | 0 of 3 |
| 8, 8, on | 78.67 ± 1.27% | 0.901 ± 0.008 | 0.939 | 0 of 3 |
| **16, 16, on** (headline model) | **78.12 ± 2.22%** | **0.894 ± 0.007** | **0.942** | 0 of 3 |
| 32, 16, on | 74.52 ± 2.64% | 0.887 ± 0.023 | 0.939 | 0 of 3 |
| 32, 32, on | 72.76 ± 5.77% | 0.880 ± 0.033 | 0.924 | 0 of 3 |
| 8, 16, off | 76.92 ± 1.05% | 0.900 ± 0.014 | 0.941 | 0 of 3 |
| 8, 8, off | 76.18 ± 3.46% | 0.889 ± 0.011 | 0.941 | 0 of 3 |
| 16, 16, off | 75.81 ± 5.21% | 0.895 ± 0.020 | 0.939 | 0 of 3 |
| 32, 16, off | 76.08 ± 0.42% | 0.898 ± 0.005 | 0.941 | 0 of 3 |
| 32, 32, off | 78.58 ± 3.04% | 0.892 ± 0.008 | 0.943 | 0 of 3 |

| Comparison at lr 2e-4 | Δ accuracy [95% CI] | Δ QWK [95% CI] |
|---|---|---|
| Rank 8 − 16, alpha 16, CLAHE on | +0.09 pt [−3.51, +3.97] | +0.002 [−0.022, +0.028] |
| Rank 32 − 16, alpha 16, CLAHE on | −3.60 pt [−8.03, +0.83] | −0.007 [−0.045, +0.021] |
| Rank 8 − 16, alpha 16, CLAHE off | +1.11 pt [−4.25, +6.93] | +0.005 [−0.030, +0.034] |
| Rank 32 − 16, alpha 16, CLAHE off | +0.28 pt [−4.52, +6.28] | +0.003 [−0.020, +0.031] |
| Rank 8 (alpha 8) − 16, CLAHE on | +0.55 pt [−3.14, +4.16] | +0.007 [−0.016, +0.032] |
| Rank 32 (alpha 32) − 16, CLAHE on | −5.36 pt [−11.45, +1.39] | −0.014 [−0.056, +0.023] |

(`rank_lr2e-4_aggregate.csv`; per-seed McNemar tests with Holm adjustment in `rank_lr2e-4_pairs.csv`.)

**Answer.** At the validation-selected learning rate, LoRA rank does not matter. None of the 30 runs fails: every run predicts all five grades. Every rank-vs-16 interval includes zero, for accuracy and QWK, with alpha fixed at 16 and with alpha = rank. Ranks 8 and 16 are within 0.6 pt of each other. Rank 32 is slightly lower and more variable with CLAHE on, but not significantly. Rank 16 is therefore a reasonable choice, not a necessary one; rank 8 would do as well with fewer trainable parameters.

**Why the original sweep looked different.** The first sweep ran at the original paper rate of 5e-4, where training is unstable (S8). There, 6 of the 18 rank runs fail (r8: 2 with CLAHE on, 1 off; r16: 1 on, 0 off; r32: 1 on, 1 off), and the accuracy SDs reach 36 pt (`rank_aggregate.csv`, `condition_summary.csv`). These differences come from the learning rate, not the rank: at 2e-4 they disappear.

**Caveats to state.**
- Seed-to-seed spread is about ±5 pt for some settings (for example r16 CLAHE-off: 78.39%, 69.81% and 79.22%). In 6 of 21 single-seed comparisons, McNemar's test survives Holm adjustment in one direction or the other, while no three-seed comparison does. Results should be quoted as three-seed means.
- The grid uses the same 361 test images as every other result. Rank was not chosen on them: the headline rank 16, lr 2e-4 model was fixed on validation loss before this grid was trained.

## S8 / G2: learning rate, chosen on validation data only

| lr (rank 16, CLAHE on) | Mean val. loss | Test accuracy | QWK | Failed runs (≤ 3 grades) |
|---|---|---|---|---|
| 1e-4 | 0.00791 | 74.98 ± 2.08% | 0.880 | 0 of 3 |
| **2e-4** | **0.00768** | **78.12 ± 2.22%** | **0.894** | 0 of 3 |
| 5e-4 (original paper setting) | 0.01051 | 69.71 ± 13.46% | 0.670 | 1 of 3 (seed 42 predicts only grades 0, 1 and 4) |

- Validation loss alone picks 2e-4, the same rate that test accuracy ranks first. The test set was not needed to make this choice.
- Across all 63 fine-tuned Qwen runs, the 10 failed runs (≤ 3 grades predicted) have validation loss 0.0142–0.0199. The 53 working runs have 0.0069–0.0107. The ranges do not overlap, so failure is visible on validation data (`collapse_vs_validation.csv`).
- All 30 runs of the lr 2e-4 rank and alpha grid (point 1) work, with validation loss 0.0069–0.0097.

Note: `SUMMARY.md` and the `collapsed` column of the CSV tables still use the narrower flag "at most 2 grades predicted". The two definitions differ for one run only: r16 CLAHE-on lr 5e-4 seed 42, which predicts 3 grades.

## S7: with and without the LoRA adapter

LoRA r16 at lr 2e-4 against the same 4-bit model with no adapter (zero-shot), both with CLAHE on:
- accuracy +31.6 pt [+25.4, +37.0]
- QWK +0.220 [+0.166, +0.275]
- macro AUC +0.169 [+0.135, +0.200]

## 2. CLAHE (reviewer comments 3 and 8; S3, S9)

| Comparison | Δ accuracy [95% CI] | Δ QWK [95% CI] |
|---|---|---|
| Qwen r16, CLAHE on − off (lr 5e-4) | −4.25 pt [−18.75, +7.20] | −0.189 [−0.560, +0.050] |
| Qwen r8, on − off | +5.54 pt [−27.52, +45.24] | −0.182 [−0.783, +0.423] |
| Qwen r32, on − off | −16.81 pt [−56.33, +15.06] | −0.034 [−0.617, +0.565] |
| Qwen zero-shot, on − off | **−8.03 pt [−12.47, −3.88]** | **−0.061 [−0.098, −0.026]** |
| RetiZero, on − off | +0.28 pt [−3.14, +3.79] | −0.003 [−0.027, +0.023] |

**Answer.** CLAHE gives no measurable benefit to either fine-tuned model. Every fine-tuned interval includes zero. CLAHE *hurts* the untuned Qwen model. Omitting CLAHE for RetiZero was therefore not a disadvantage: with CLAHE, RetiZero scores 79.32 ± 1.53% against 79.04 ± 1.69% without.

**Colour conversions.**
- The app preprocessing converts RGB → LAB. The dataset builder converts BGR → LAB. Both are correct, because each matches the channel order of the array it is given.
- `build_image_audits.py` (`colour_conversion_check.csv`) checked this pixel by pixel on the 361 raw APTOS test photos. Both routes give **identical pixels on 361 of 361** images for Qwen with CLAHE, Qwen without CLAHE, and RetiZero 224 (OpenCV 5.1.0).
- A deliberately wrong conversion (BGR data passed as RGB) ran as a negative control. It differs on **all 361** images, mean absolute difference 5.18 grey levels, so the check would catch the bug.

## 3. Computational efficiency (reviewer comment 13; S14)

Every row was measured on the same PC: one NVIDIA RTX 5070 Ti 16 GB (`efficiency_final.csv`).

| | Qwen3-VL-8B + LoRA r16 (4-bit) | RetiZero ViT-L/16 + LoRA r8 |
|---|---|---|
| Total parameters | 8.82 B | 304.1 M |
| Trainable parameters | 51.3 M (0.59%) | 304.1 M (all weights) |
| Training time, full run | 4,545 ± 3,120 s at 5e-4; 5,001 ± 845 s at 2e-4 | 286 ± 105 s |
| Training time per epoch | 894 s | 24.5 s |
| Peak training GPU memory (PyTorch / whole card) | 8.9 GiB / 11.5 GB | 11.2 GiB / 14.5 GB |
| Inference per image (batch 1, warm median) | 0.861 s | 0.018 s |
| Inference throughput (batch 1) | 1.15 images/s | 54.5 images/s |
| Peak inference GPU memory (PyTorch) | 8.65 GiB | 1.18 GiB |
| Model load time | 9.9 s | 5.9 s |

Notes to state:
- Qwen's inference time is for the likelihood scorer, which runs five forward passes per image. Greedy generation takes 1.57 s per image.
- RetiZero trains every weight. Its training memory is therefore higher than 4-bit Qwen's even though the model is 29 times smaller.

## 4. Evaluation checks and misclassification (reviewer comments 10–12; S11, S13)

**ROC-AUC.** It is computed one-vs-rest from each grade's own softmax or likelihood column, and independently reproduced (see the integrity checks above).

**Invalid Qwen outputs and the Grade-0 fallback.**
- The reported Qwen metrics come from a likelihood scorer: it picks the most likely of the five fixed answers, so it can never produce an invalid output.
- The greedy-generation check found **0 unparseable outputs** in all 10 generation jobs, including the original paper adapter on the 367 original images. So the Grade-0 fallback never fired (`grade0_fallback_check.csv`).

**Class-wise sensitivity and specificity, APTOS-361** (`classwise_condition_mean.csv`):

| Grade | Qwen lr2e-4 sens / spec | RetiZero sens / spec |
|---|---|---|
| 0 No DR | 0.945 / 0.993 | 0.963 / 0.989 |
| 1 Mild | 0.733 / 0.923 | 0.743 / 0.922 |
| 2 Moderate | 0.625 / 0.910 | 0.639 / 0.929 |
| 3 Severe | 0.389 / 0.940 | 0.407 / 0.942 |
| 4 PDR | 0.589 / 0.975 | 0.522 / 0.971 |

**Error structure (S13).** About 80% of errors are off by one grade:
- Qwen lr2e-4: 81.4% of errors are off by one; 0.18% of all images are off by three or more.
- RetiZero: 81.1% of errors are off by one.

That is why QWK (0.894) is high while accuracy is about 78%.

**Mild → Normal on Messidor.**

| | APTOS: Mild predicted Normal | Messidor: Mild predicted Normal | Messidor: Mild-vs-Normal AUC |
|---|---|---|---|
| Qwen lr2e-4 | 3.8% | 71.7% | 0.62 |
| Qwen lr5e-4 | 18.1% | 83.6% | 0.54 |
| RetiZero | 5.7% | 61.7% | 0.50 |

- On APTOS both models separate Mild from Normal well (AUC about 0.98).
- On Messidor neither model does better than chance. RetiZero also calls 39% of Normal eyes Mild.
- This points to a domain shift in what "Mild" looks like (cameras, resolution, microaneurysm visibility), not a labelling error: the labels are the adjudicated grades.
- Representative images where all three seeds of both models are wrong, including four Mild → Normal cases, are in the contact sheets. Those are kept outside the repository (`image_audits\`) because they are dataset images. The index is `misclassified_examples_index.csv`.
- **Suspected causes must be labelled as suspected:** they have not been confirmed by a grader.

**Splits and augmentation** (`class_counts.csv`).
- Clean split: 2,786 / 357 / 361 images.
- No image appears in two splits. Six cross-split pairs are perceptual-hash identical and are reported as near-duplicates (see below).
- Augmentation is training-only:
  - RetiZero applies RandomResizedCrop, flip and rotation to training images only; evaluation only resizes.
  - Qwen applies no image augmentation. It balances classes with an inverse-frequency WeightedRandomSampler (about 557 draws per grade per epoch, against 1,436 / 267 / 729 / 140 / 214 images on disk).
  - RetiZero instead weights the loss by inverse class frequency.

## Additional robustness evidence (S10, S12)

**Qwen vs RetiZero on APTOS-361.**
- Qwen at the validation-selected rate minus RetiZero: **−0.92 pt [−5.26, +3.14]**; QWK −0.001 [−0.027, +0.026]; macro AUC −0.001 [−0.017, +0.013].
- The two models are statistically indistinguishable in-domain.
- No per-seed McNemar test survives Holm correction (`head_to_head_pairs.csv`).

**Qwen vs RetiZero on Messidor** (APTOS-trained, patient-clustered).
- Qwen lr2e-4 minus RetiZero: **+8.07 pt [+0.57, +16.75]** accuracy, QWK +0.107, macro AUC +0.054. Qwen transfers better.
- Both models are far below their in-domain scores (48.5% and 40.5%).
- Referable-DR sensitivity on Messidor is only about 0.33 for every fine-tuned model, with specificity about 0.99. **This is a clinically important limitation and must be reported.**

**Near-duplicates.**
- A perceptual hash flags 6 test images at distance 0 and 32 at distance ≤ 2. The same counts came out of an independent reimplementation, and all 134 byte-identical APTOS pairs hash to distance 0.
- Removing the 6 or the 32 moves every model's mean accuracy by less than 1 pt, in either direction, and QWK by at most 0.008. Near-duplicates did not inflate the results (`near_duplicate_sensitivity.csv`).
- Removing all 120 images at distance ≤ 4 lowers accuracy about 4 pt for both models. Those 120 are 67% grade 0, the easiest class (50% of the full test set), so the drop is a change in class mix, not leakage.

**Messidor duplicates.**
- Four photos appear twice under two patient IDs (p89/p91, p90/p92).
- They are pixel-identical in the raw Messidor-2 release too, so the duplication comes from the dataset itself.
- Each pair has the same grade. Patient-clustered CIs treat each pair as one patient (527 clusters).

**Archived images against raw.** The verified Messidor-2 cohort was rebuilt straight from the raw photos with the APTOS builders' own code (`finetuning/revision_jobs/build_messidor2_cohort.py`).
- All 1,057 archived 512 px images are **pixel-identical** to the raw rebuild.
- So the archived inputs Qwen was scored on are exactly what the raw pipeline produces.
- Only RetiZero's archived route (raw → 512 → 224) differs from the direct raw → 224 build.

## Still open

1. **Decision:** the primary comparison and endpoint. Proposed: H1, Qwen at the validation-selected lr 2e-4 against RetiZero, each with its native preprocessing, on accuracy and QWK.
2. **GPU queue, ready to run.** `finetuning/revision_jobs/` (see its README) runs in one command:
   - every model on the full verified Messidor-2 cohort: 1,744 gradable images, the standard external benchmark, about 6 h
   - timing and power for the 7 Qwen configurations without a timing record, about 2 h

   The rank sweep at lr 2e-4, including alpha = rank, is done (point 1; `final_runs/final_02`), so the runner's `--rank-sweep` option is not needed.

   The analysis picks the new results up automatically: Messidor-2 head-to-head with exam-clustered CIs, the archived-vs-verified agreement check, and timing and energy.
3. Manuscript text for G1–G3 and S1–S14. The tables above are ready to cite.
