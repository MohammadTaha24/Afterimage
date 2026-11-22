# Evidence for Dr. Noor's revision plan

22 November 2026. This note answers the four points of the revision plan from results that already exist. **No new training was needed.** Every number comes from `finetuning/paper_tables/output/revision_analysis/` (open `SUMMARY.md` there). Those tables are rebuilt from the saved per-image predictions by `build_revision_analysis.py` and `build_image_audits.py` (see `finetuning/paper_tables/README.md`).

All results are on the 361 clean APTOS test images and give the mean ± SD over three seeds, unless stated otherwise. CIs are 95% bootstrap intervals. "Δ" is a hierarchical bootstrap over images and seeds.

## Integrity checks run first

| Check | Result | File |
|---|---|---|
| Recompute every run's accuracy, QWK and macro AUC from its predictions | 73 of 73 runs match the committed tables or the saved run summaries (tolerance 1e-8) | `reproduction_check.csv` |
| Same images, order and labels for every run in a cohort | All runs on APTOS-361, APTOS-367 and Messidor agree | `image_alignment_check.csv` |
| ROC-AUC uses each grade's own probability column | An independent scipy Mann-Whitney computation agrees exactly (max difference 0) for every run and grade | `reproduction_check.csv` |
| Messidor labels | All 1,057 archived labels equal the Krause 2018 adjudicated ICDR grades. The 1,058th Messidor-Original PNG is the only ungradable PNG. The eye pairing agrees for all 528 complete exams. | `messidor_label_audit.csv` |

The earlier concern that "our Messidor labels do not match the published grade distribution" is resolved. Our cohort is the PNG (Messidor-Original) half of Messidor-2, and its grade mix (468/207/290/71/21) is exactly the adjudicated mix for those PNGs. The comparison had been made against all 1,744 gradable images, PNG and JPG together.

## 1. LoRA rank (reviewer comments 5 and 7; S6)

Ranks 8, 16 and 32 were trained with everything else fixed: alpha 16, lr 5e-4, CLAHE on and off, three seeds each.

| Rank, CLAHE | Accuracy | QWK | Collapsed runs | Mean of non-collapsed runs |
|---|---|---|---|---|
| 8, on | 55.77 ± 17.19% | 0.410 ± 0.461 | 2 of 3 | 75.07% |
| 16, on | 69.71 ± 13.46% | 0.670 ± 0.337 | 0 of 3 | 69.71% |
| 32, on | 49.49 ± 35.77% | 0.552 ± 0.479 | 1 of 3 | 70.08% |
| 8, off | 50.23 ± 36.39% | 0.593 ± 0.513 | 1 of 3 | 71.19% |
| 16, off | 73.96 ± 3.05% | 0.859 ± 0.059 | 0 of 3 | 73.96% |
| 32, off | 66.30 ± 14.56% | 0.586 ± 0.508 | 1 of 3 | 74.38% |

Collapsed = at most 2 of the 5 grades predicted.

**Answer.** Rank 16 is the only rank with no collapsed run at 5e-4, under both CLAHE settings. Among runs that did not collapse, the three ranks perform about the same (70–75%). Every rank-vs-16 Δ interval includes zero (`rank_aggregate.csv`). So rank 16 is justified as the most stable setting, not as the most accurate one.

**Caveats to state.**
- Alpha stayed at 16, so alpha/r changes with rank (2, 1, 0.5). This sweep varies rank and LoRA scaling together.
- The sweep ran at 5e-4, where training is unstable (see S8 below).
- A rank sweep at the validation-selected rate of 2e-4 would answer the question cleanly. That is the only extra GPU work this point could justify, about 20 hours.

## S8 / G2: learning rate, chosen on validation data only

| lr (rank 16, CLAHE on) | Mean val. loss | Test accuracy | QWK | Collapsed |
|---|---|---|---|---|
| 1e-4 | 0.00791 | 74.98 ± 2.08% | 0.880 | 0 of 3 |
| **2e-4** | **0.00768** | **78.12 ± 2.22%** | **0.894** | 0 of 3 |
| 5e-4 (original paper setting) | 0.01051 | 69.71 ± 13.46% | 0.670 | 0 of 3, but seed 42 predicts only grades 0, 1 and 4 |

- Validation loss alone picks 2e-4, the same rate that test accuracy ranks first. The test set was not needed to make this choice.
- Across all 36 fine-tuned Qwen runs, the 10 failed runs (≤ 3 grades predicted) have validation loss 0.0142–0.0199. The 26 working runs have 0.0070–0.0107. The ranges do not overlap, so collapse is visible on validation data (`collapse_vs_validation.csv`).

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
- `build_image_audits.py` (`colour_conversion_check.csv`) checks this pixel by pixel on the 361 raw APTOS test photos: Qwen with CLAHE, Qwen without CLAHE, and RetiZero 224.
- A deliberately wrong conversion (BGR data passed as RGB) runs as a negative control. It must come out different, which shows the check can catch the bug.

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

## Still open

1. **Decision:** the primary comparison and endpoint. Proposed: H1, Qwen at the validation-selected lr 2e-4 against RetiZero, each with its native preprocessing, on accuracy and QWK.
2. **Optional GPU work.**
   - A rank sweep at 2e-4 with alpha/r fixed (point 1): about 20 hours.
   - Qwen timing for the 7 configurations without warm per-image timing: about 2 hours.
   - Scoring on the raw-built Messidor cohort, which removes the archived raw → 512 → 224 route: under 1 hour. The raw images and grades are now in hand.
3. Manuscript text for G1–G3 and S1–S14. The tables above are ready to cite.
