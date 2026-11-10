# Three-rank results review and paper checklist

Review date: 10 November 2026. Batch: `artifacts/qwen_four_run_001`.

The three CLAHE-on conditions completed training and both-cohort evaluation. `r16_clahe_off` was intentionally excluded, not an unsuccessful training run. The saved queue state is `finished_stage` because the independent notebook stage cells were used; all three completion records are present.

This review read saved JSON/CSV/log/source records and performed CPU arithmetic on saved predictions. No notebook cells, project workers, tests, preprocessing, model loading, inference or training were launched. Existing scripts, datasets, checkpoints and generated batch results were not changed. Weight binaries were not reread or hashed in this review.

## Results

Accuracy is shown as a percentage. QWK is dimensionless and can be negative; AUC is on a 0–1 scale. Macro AUC is the mean of all five one-vs-rest grade AUCs. These are one completed realization per rank, not seed averages.

| Condition | APTOS correct / 367 | APTOS accuracy | QWK | Macro OVR AUC | Messidor correct / 1,057 | Messidor accuracy | QWK | Macro OVR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r16_clahe_on | 262 | 71.3896% | 0.895468 | 0.931862 | 569 | 53.8316% | 0.636691 | 0.806761 |
| r8_clahe_on | 256 | 69.7548% | 0.674309 | 0.893135 | 467 | 44.1816% | 0.131260 | 0.675020 |
| r32_clahe_on | 274 | 74.6594% | 0.884094 | 0.936552 | 489 | 46.2630% | 0.547092 | 0.783002 |

Primary evidence: [batch summary](artifacts/qwen_four_run_001/batch_summary.csv), and each condition's [rank-16](artifacts/qwen_four_run_001/r16_clahe_on/evaluation_complete.json), [rank-8](artifacts/qwen_four_run_001/r8_clahe_on/evaluation_complete.json), and [rank-32](artifacts/qwen_four_run_001/r32_clahe_on/evaluation_complete.json) evaluation completion record.

The six prediction CSVs have the expected unique IDs, labels and paths, no duplicate IDs, and finite five-grade vectors summing to one. Every predicted grade equals the first-max score argmax. Their file hashes match their completion records. Independent CPU calculations reproduce the saved accuracy, QWK and five-class macro AUC. Adapter-configuration and saved source-file hashes also match their records. These checks establish consistency of the saved evidence; they are not a new model evaluation or a numerical check of the weight tensors.

### Interpretation

- Rank 32 has the highest new APTOS accuracy and AUC point estimates. It gets 12 more APTOS images correct than rank 16, a 3.27 percentage-point difference.
- Rank 16 has the highest new APTOS QWK and all three highest Messidor point estimates. It gets 80 more Messidor images correct than rank 32, a 7.57 percentage-point difference.
- There is no consistent winner across all outcomes. Keep rank 16 as the predeclared shared control. Do not select a final rank, checkpoint, scorer or hyperparameter using the held-out tables.
- These results describe **rank with alpha held at 16**. Effective alpha/r is 1, 2 and 0.5 for ranks 16, 8 and 32 respectively. Capacity and scaling changed together. The comparison does not isolate a capacity-only rank effect.
- One fixed-seed realization per rank does not estimate training-seed variability. No confidence intervals, significance tests or multiple-comparison adjustments were calculated in this review.

### Findings that need investigation

| Saved-prediction finding | Rank 16 | Rank 8 | Rank 32 |
| --- | ---: | ---: | ---: |
| Messidor Mild → Normal, out of 207 Mild images | 169 (81.64%) | 206 (99.52%) | 164 (79.23%) |
| Messidor Normal predictions, out of 1,057 images | 694 | 1,012 | 658 |
| APTOS grade-4 recall | 16/30 (53.33%) | 0/30 (0%) | 8/30 (26.67%) |
| APTOS Moderate → Severe, out of 100 Moderate images | 58 | 0 | 24 |

**Rank 8 never predicts grades 3 or 4 on either cohort.** Its Messidor predicted-grade counts are `[1012, 8, 37, 0, 0]`. Its 467 correct Messidor predictions are one fewer than the descriptive all-Normal baseline of 468/1,057 = 44.2763%. Accuracy alone obscures this failure mode.

The new evaluator uses valid complete-answer score argmax, with no invalid-text Grade-0 fallback. These Mild → Normal errors therefore do not demonstrate a legacy parser fallback. The historical raw-response audit remains a separate task.

Rank 8's Messidor grade-3/4 AUCs are still 0.9232 and 0.9060 despite zero grade-3/4 hard predictions. This supports inspecting relative class scores, class bias, margins and calibration. The scorer sums full-answer log likelihood; candidate wording/token length is a hypothesis to investigate. Neither a scoring bug nor its cause is established by this pattern. Use validation data for diagnosis and any proposed adjustment, and retain the original evaluator/results as the primary frozen record.

Per-grade Messidor recall further shows the imbalance:

| Condition | Normal (468) | Mild (207) | Moderate (290) | Severe (71) | Proliferative (21) |
| --- | ---: | ---: | ---: | ---: | ---: |
| r16_clahe_on | 90.81% | 17.39% | 18.97% | 63.38% | 38.10% |
| r8_clahe_on | 99.15% | 0.48% | 0.69% | 0% | 0% |
| r32_clahe_on | 86.75% | 20.29% | 6.90% | 26.76% | 9.52% |

Additional descriptive metrics recomputed from the same saved predictions:

| Condition | APTOS balanced accuracy | APTOS macro F1 | Messidor balanced accuracy | Messidor macro F1 |
| --- | ---: | ---: | ---: | ---: |
| r16_clahe_on | 0.626108 | 0.589365 | 0.457289 | 0.470503 |
| r8_clahe_on | 0.439065 | 0.389633 | 0.200636 | 0.129712 |
| r32_clahe_on | 0.617870 | 0.580569 | 0.300446 | 0.313405 |

Paired rank-16/rank-32 accuracy outcomes: APTOS has 35 images correct only for rank 16 and 47 correct only for rank 32; Messidor has 122 correct only for rank 16 and 42 correct only for rank 32. These are matched predictions on the same images, not independent cohorts. Patient grouping must also be considered for uncertainty estimates where available.

## Previously reported reference and comparability

Dr. Noor's accepted **Qwen3-VL-8B Retraining 2** remains the separately supplied reference: **80.65% APTOS accuracy, 0.8915 QWK and 0.9362 macro ROC-AUC**. New results do not replace those figures.

A local [historical run_02 summary](../finetuning/Qwen3/artifacts/paper_reproduction/training_batches/aptos_paper_same_seed_three_runs_001/run_02/evaluation/aptos_test.summary.json) contains `0.8065395095367848 / 0.8915262317355135 / 0.936182570697056`, which round to all three supplied reference figures. This is a strong candidate artifact link; matching numbers alone do not establish that it is the exact checkpoint/export Dr. Noor accepted. Confirm that provenance explicitly.

That candidate's [saved Messidor results](../finetuning/Qwen3/artifacts/paper_reproduction/training_batches/aptos_paper_same_seed_three_runs_001/run_02/evaluation/messidor_external.summary.json) are 50.5203% accuracy, 0.552775 QWK and 0.773197 AUC. They are local saved results, not separately supplied accepted-reference figures.

The candidate historical run_02's [resolved arguments](../finetuning/Qwen3/artifacts/paper_reproduction/training_batches/aptos_paper_same_seed_three_runs_001/run_02/resolved_training_arguments.json) record validation batch **4**. Its [training record](../finetuning/Qwen3/artifacts/paper_reproduction/training_batches/aptos_paper_same_seed_three_runs_001/run_02/run_info.json) records a best validation checkpoint at epoch **2**, step **734**, loss **0.007335608825087547**. The new runs actually used validation batch **8**, as requested in the reviewed configuration; their correction records show the wrapper's 4 → 8 adjustment. Numerical validation loss and checkpoint selection may be affected. This is an observed protocol difference, not proof that it caused the performance difference.

For orientation only, the new control's APTOS accuracy is 9.26 percentage points below the supplied 80.65%; its QWK differs by +0.00397 and AUC by −0.00434. Reconcile provenance and effective protocols before interpreting these as clean reproduction gains or losses.

The separate saved adapter at `Qwen3/lr0.0005_epoch4_batchsize8/best_lora` also has a saved APTOS evaluation, but it reports different metrics (81.4714%, 0.928550, 0.948810). Do not conflate it with the supplied Retraining 2 reference.

## Training, stopping and time

| Condition | alpha/r | Best validation epoch / step | Best validation loss | Epochs completed | Training stage | Evaluation stage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r16_clahe_on | 1 | 3 / 1101 | 0.008197679184377193 | 5 | 84.49 min | 20.58 min |
| r8_clahe_on | 2 | 3 / 1101 | 0.010400880128145218 | 5 | 84.28 min | 20.72 min |
| r32_clahe_on | 0.5 | 1 / 367 | 0.008199954405426979 | 3 | 53.21 min | 20.83 min |

Rank 16's best validation loss is only 0.000002275 below rank 32's. Do not overinterpret that small numerical difference as robust model superiority. Rank 32 stopped after fewer epochs; its shorter total training time does not establish better per-epoch efficiency. All three trained fresh, with one completed training attempt and one evaluation attempt; no training checkpoint was resumed.

The saved span from preflight start to final summary is **4 h 46 m 42 s**, **08:52:41–13:39:23 Asia/Riyadh on 10 November 2026**. Per-stage times above include stage overhead. The final `queue_status.json` elapsed field describes the last individual action, not the total batch duration.

The saved records agree on Trainer seed 42, adapter initialization seed 3407, cached base revision `b5b904c3fcdc7541adf2a2bb219b0ed95288c794`, target-module regex, alpha 16, dropout 0.05, bias `none`, RS-LoRA false, 4-bit base loading, BF16 training, FP16 scoring, batch 4 × accumulation 2, LR 5e-4, AdamW-8bit, linear schedule, warmup 20, weight decay 0.001 and sequence length 2048. Checkpoint selection used APTOS validation loss; Messidor stayed external. No image augmentation was introduced; inverse-frequency sampling was confined to the training loader.

## Preprocessing handoff and remaining CLAHE experiment

Give the other researcher [DATASET_PREPROCESSING_HANDOFF.md](DATASET_PREPROCESSING_HANDOFF.md). It specifies the original-image mapping, frozen labels/splits, Otsu crop, black square padding, 512×512 resize, BGR/RGB/LAB handling, one-versus-zero CLAHE branches, QC and provenance deliverables. These are the preprocessing obligations already requested; adding image augmentation or re-splitting is not part of this handoff.

The existing [preflight](artifacts/qwen_four_run_001/preflight.json) passed shared CLAHE-on coverage/integrity checks but explicitly skipped raw-original lineage and off/on pairing. Therefore this batch supports a comparison on fixed CLAHE-on inputs. It does not yet establish an effect of CLAHE or verify a clean CLAHE ablation.

Preparing the off dataset is followed by **fresh rank-16 CLAHE-off training**, validation checkpoint selection, and evaluation on both frozen cohorts. Applying off images to an existing CLAHE-on adapter would answer a separate inference-only question.

Preserve the completed control's frozen on images. If the prepared pair uses the same decoded on-image pixels, document equivalence and keep the frozen on files untouched for hash-compatible reuse. If decoded on pixels change, build a new matched rank-16 on/off comparison and retain this completed rank study separately. Compression/metadata-only byte changes should be documented as such; they should not be confused with changed model inputs.

## Checklist

Completed in the saved batch or this read-only review:

- [x] Three fresh CLAHE-on adapters trained at ranks 16, 8 and 32; alpha/r recorded as 1, 2 and 0.5.
- [x] Minimum APTOS validation loss selected each checkpoint; Messidor excluded from training/selection.
- [x] Complete 367-image APTOS and 1,057-image Messidor evaluations saved with five grade scores per image.
- [x] Saved records, small source/configuration hashes, prediction hashes, ID/label/path joins, score validity, accuracy, QWK and AUC checked in this review.
- [x] Rank-8 hard-prediction collapse and external Mild → Normal counts identified.
- [x] CLAHE-off recorded as deliberately excluded; no failed run or CLAHE-effect claim inferred.
- [x] Separate preprocessing handoff and this results/checklist document saved; model work remains unstarted during this review.

Priority 0 — preserve evidence and resolve the reference, before making paper claims:

- [ ] Preserve/back up this batch's configs, completion markers, logs, score exports, best adapters and processor files; verify copied artifacts against the saved hashes.
- [ ] Confirm the exact adapter/export/checkpoint Dr. Noor accepted as Retraining 2. Record the link to historical run_02 if confirmed; keep the distinct presented adapter separate.
- [ ] Reconcile effective reference/new-control settings, especially actual validation batch 4 versus 8, checkpoint selection, data/image provenance, sampler and seeds, target modules, prompt and scorer. Record differences without guessing their cause.
- [ ] Complete the Phase 1 evidence audit. `audit/phase1_report.md` is currently absent from the audit folder. Do not describe the audit as completed merely because the rank runs finished.
- [ ] Collect the historical raw Qwen responses, parse flags/predictions and independent Messidor labels; establish the producing parser revision and count actual invalid-output fallbacks and their Mild → Normal contribution separately from these score-based errors.
- [ ] Record the primary outcome and model-selection rule from the agreed paper protocol. Retain every rank's held-out results and label later exploratory choices transparently.

Priority 1 — work that can proceed in parallel:

- [ ] Data researcher: complete every acceptance item in the preprocessing handoff and return the paired folders, original-image index, QC, hashes and producing code/version records.
- [ ] ML/analysis: investigate rank-8 collapse using training/validation records first. Inspect per-class score distributions, relative score margins, labels, candidate wording/token lengths and scoring behavior. Do not assume the diagnosis is parser fallback or CLAHE.
- [ ] If validation grade-score exports are needed, prepare a separately reviewed validation-only scoring step; the current batch exports test/external scores, not a saved validation score table. Do not repurpose a test export as validation.
- [ ] Reconcile the new control's exact-grade errors: APTOS Moderate → Severe and Proliferative recall, plus low Mild/Moderate recall on Messidor. Separate label/data issues from ordinal prediction and score-calibration issues.
- [ ] Produce paper-ready confusion matrices, class support/recall/precision/F1, predicted class distributions and the three requested headline metrics for every condition. Preserve the results above as the original protocol outputs.
- [ ] Add uncertainty estimates using matched predictions; use paired methods for rank differences and respect available Messidor patient/eye grouping. Distinguish test-sample uncertainty from seed variability and document multiple/exploratory comparisons.

Priority 2 — complete the CLAHE ablation after data acceptance:

- [ ] Establish whether the completed rank-16 on control is a valid paired counterpart using the handoff proof; document any missing historical original bytes.
- [ ] Review the returned preprocessing provenance and pass the original/off/on per-image pairing checks in the pinned environment.
- [ ] With unchanged frozen on inputs, source and environment, enable `INCLUDE_CLAHE_OFF=True`, set `ORIGINAL_IMAGE_INDEX`, rerun preflight, and launch only the missing condition/stages. Previously completed, verified stages can be skipped by the queue. Do not edit pinned sources to bypass compatibility failures.
- [ ] If the on inputs or experimental pipeline change, create a new matched on/off batch and a fresh shared rank-16 control; retain the existing rank-study batch.
- [ ] Train a fresh `r16_clahe_off` adapter, use APTOS validation for its checkpoint, and evaluate the frozen APTOS/Messidor cohorts with the same scorer and matching inputs.
- [ ] Report the paired rank-16 on/off comparison with uncertainty and class-level effects. A preprocessing handoff alone does not finish this experiment.

Priority 3 — interpretation and manuscript:

- [ ] Before making generalized rank claims, plan matched multi-seed confirmation in a new reviewed protocol, or restrict the claim to these single-run fixed-alpha observations.
- [ ] If the paper needs a capacity-only rank claim, plan a separate constant-scaling comparison; do not relabel this fixed-alpha study as one.
- [ ] Plan any calibration/scorer modification on validation data with its own recorded protocol and clearly separated evaluation; do not tune on these inspected held-out outcomes.
- [ ] Document cohort construction, adjudicated label source, inclusion/exclusion criteria, recovered split provenance, patient/eye grouping, imbalance and Messidor domain shift.
- [ ] Write methods covering actual effective settings, alpha/r, weighted sampling, no image augmentation, stopping/checkpoint rules, five-answer scoring and full cohort counts.
- [ ] Write results/limitations covering all ranks, single-seed uncertainty, weak external Mild/Moderate performance, reference comparability, and the unfinished/finished state of the CLAHE experiment.
- [ ] Archive a reproducible artifact/provenance index and confirm with the paper team which results enter the final tables.

Suggested next order: preserve/reference reconciliation and validation diagnostics now; delegate preprocessing in parallel; run the missing CLAHE condition only after data acceptance; then finalize uncertainty analyses and manuscript tables. The completed ranks do not need an automatic rerun just because their scores differ from expectations.

## Saved confusion matrices

Rows are true grades 0–4; columns are predicted grades 0–4. These counts were derived from the saved score-prediction CSVs, not new inference.

```text
APTOS — rank 16           APTOS — rank 8            APTOS — rank 32
175   6   0   0   0       169  10   2   0   0        175   5   1   0   0
  4  19  10   4   0        10  23   4   0   0          5  23   9   0   0
  0   4  38  58   0         2  34  64   0   0          0  21  55  24   0
  0   0   2  14   3         0   8  11   0   0          0   1   4  13   1
  0   3   0  11  16         0  13  17   0   0          0   1   9  12   8

Messidor — rank 16        Messidor — rank 8         Messidor — rank 32
425  42   1   0   0       464   3   1   0   0        406  59   3   0   0
169  36   2   0   0       206   1   0   0   0        164  42   1   0   0
 95 131  55   7   2       284   4   2   0   0         87 183  20   0   0
  3   3  16  45   4        45   0  26   0   0          1  11  40  19   0
  2   2   2   7   8        13   0   8   0   0          0   7   6   6   2
```

Evidence inventory: [preflight](artifacts/qwen_four_run_001/preflight.json), [shared configuration](artifacts/qwen_four_run_001/shared_configuration.json), [queue status](artifacts/qwen_four_run_001/queue_status.json), and per-run `resolved_configuration.json`, `adapter_constructor_arguments.json`, `effective_adapter_scaling.json`, `resolved_training_arguments.json`, `stage_status.json`, `training_complete.json`, `evaluation_complete.json`, `evaluation/*.summary.json` and `evaluation/*.predictions.csv`. The [frozen manifest](../finetuning/Qwen3/artifacts/paper_reproduction/data/4fda605471d5430c3733/dataset_manifest.json) explicitly notes that original historical preprocessed image bytes were unavailable for comparison.
