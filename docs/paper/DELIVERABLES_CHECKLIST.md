# Paper deliverables checklist

Status as of 16 November 2026. The ordered list of what is left is in [REMAINING_WORK.md](REMAINING_WORK.md). This replaces the 14 November snapshot in `PAPER_TRAINING_CHECKLIST.md`. A box is ticked only when the evidence exists on disk and was checked. The plan that orders this work is in [PLAN.md](PLAN.md); the reviewer comments are tracked one by one in [REVIEWER_CHECKLIST.md](../../REVIEWER_CHECKLIST.md).

Reviewer numbering: this file uses G1–G3 and S1–S14 from `REVIEWER_CHECKLIST.md`. Dr. Noor's email numbers are one lower from her item 5 onward (her 5 and 7 are S6 and S8, her 8 is S9, her 10–12 are S11–S13, her 13 is S14).

## A. Decisions that block other work

- [ ] **Headline result set.** Confirm with Dr. Noor that the clean 361-image suite is the paper's main result and Retraining 2 (367 images, 17 of them identical to training images) is reported only as the historical reference.
- [ ] **CLAHE position.** The results do not show a CLAHE benefit for Qwen. Decide whether the headline configuration becomes rank 16 with CLAHE off.
- [ ] **Scope of new Qwen training.** Decide on the stability and tuning runs, the fixed alpha/r runs and (if CLAHE off becomes the headline) the BF16 CLAHE-off control. See PLAN.md, workstream 3.
- [ ] **Efficiency measurement scope.** Decide between filling the measurement gaps and repeating every run. See PLAN.md, workstream 1.
- [ ] **RetiZero arrangements.** Confirm who trains, on what machine, and by when.
- [ ] **Messidor sources.** Confirm who holds the raw Messidor-2 images, the grades and the eye pairing.

## B. Qwen experiments

The unticked items in this section are queued in `finetuning/Qwen3/notebooks/03_final_runs.ipynb` (design in [FINAL_RUNS_DESIGN.md](FINAL_RUNS_DESIGN.md)), except the BF16 CLAHE-off control, which needs a new worker.

- [x] Rank 8, 16 and 32 with CLAHE on and off, three seeds each, on the clean split (18 runs).
- [x] Rank 16, CLAHE on, previous train/validation membership (3 runs).
- [x] Rank 16, CLAHE on, unquantized BF16 base (3 runs).
- [x] Original-split CLAHE on/off pairs, scored at BF16 (6 runs, all the same seed).
- [x] Replay of the Retraining 2 adapter matches the archive exactly on APTOS and archived Messidor.
- [x] **S7: Qwen with no adapter** (zero-shot), scored with the same five-answer scorer on the 361 test images: 4-bit CLAHE on 46.54%, 4-bit CLAHE off 54.57%, BF16 CLAHE on 52.63% (`final_runs\final_01`).
- [x] **G2/S8: learning-rate evidence**, rank 16, CLAHE on, three seeds: 1e-4 gives 74.98 ± 2.08%, 2e-4 gives 78.12 ± 2.22%, against 69.71 ± 13.46% at 5e-4. No collapsed run at either lower rate.
- [ ] **G2/S8: decision** on whether the main grid is repeated at a lower rate (see [REMAINING_WORK.md](REMAINING_WORK.md), phase 1).
- [ ] **S6: rank with alpha/r held fixed**, or a written limitation (decision A3). The alpha 8 and alpha 32 runs at 5e-4 are done, but four of six collapsed or could not be scored in half precision, so they do not answer S6 yet.
- [ ] BF16 base with CLAHE off, three seeds (only if decision A2 makes CLAHE off the headline).
- [x] Count of invalid outputs under the old generation-based evaluation, using the original paper adapter: 0 of 367, so the Grade 0 fallback did not affect the earlier numbers.

## C. RetiZero (runs from [02_retizero_runs.ipynb](../../finetuning/RetiZero/notebooks/02_retizero_runs.ipynb); brief for a teammate: [RETIZERO_TRAINING_BRIEF.md](RETIZERO_TRAINING_BRIEF.md))

- [x] RetiZero reproduction folder built so everything can run on this PC: helper, workers, queue, tests and notebook (plan and status: [finetuning/RetiZero/REPRODUCTION_PLAN.md](../../finetuning/RetiZero/REPRODUCTION_PLAN.md)). Pilot passed from random weights.
- [x] Saved fine-tuned checkpoint replayed on the original 367 test images. It reproduces all 27 figures the original notebook printed (82.29% / 0.9097 / 0.9451 and the per-grade report). Evidence: `D:\Afterimage_Experiments\retizero_runs\retizero_01\replay_comparison.json`.
- [x] 224-pixel dataset built on the same split and the same 361 test IDs as Qwen.
- [x] Training and evaluation workers written (`finetuning/RetiZero/paper_revision/`) and run end to end in the pilot, including stop and resume.
- [ ] Pilot repeated with the pretrained checkpoint (checks that it fits the encoder).
- [ ] Official pretrained RetiZero checkpoint obtained, with source, date and SHA-256 recorded.
- [ ] Transfer package assembled and sent (code, 224-pixel data, monitor tool, instructions). Only if a teammate trains instead of this PC.
- [ ] Seeds 42, 43 and 44 trained and scored on the 361 test images (`matched` stage).
- [ ] Original recipe repeated on the original split, three seeds (`historical` stage).
- [ ] Results returned and accepted (same IDs and order, metrics recompute, configuration identical across seeds).
- [ ] 224-pixel CLAHE dataset and 224-pixel archived Messidor build (step 9 of the RetiZero plan).
- [ ] RetiZero with CLAHE, three seeds (optional; turns S9 from an explanation into evidence).
- [ ] RetiZero scored on the verified Messidor cohort.

## D. Efficiency and utilization (S14)

- [x] Per-run 5-second log of GPU %, VRAM, CPU % and RAM, plus wall time and hardware, for all 24 clean-suite runs and the 6 original-split runs.
- [x] Qwen efficiency table built from those logs (`finetuning/paper_tables/output/efficiency/`).
- [ ] Portable resource monitor that also records GPU power and per-process VRAM.
- [ ] Telemetry summary across all attempts of a run (three runs were interrupted and only their final attempt is counted today).
- [x] Exact parameter counts for Qwen from the weight files: 8,767,123,696 in the base; 25,673,472 / 51,346,944 / 102,693,888 trainable at rank 8 / 16 / 32. The 4-bit log figure (8.82 billion) is the base plus the rank 16 adapter; the BF16 log figure (6.12 billion) omits the layers held in system RAM.
- [x] Parameter counts for RetiZero from the checkpoint: 304,093,189 (304,088,064 encoder + 5,125 head), all trained in fine-tuning; 786,432 of them are LoRA branches.
- [ ] Inference benchmark on this PC for both models: warm per-image latency, images per second, peak VRAM, model load time. Done for the saved RetiZero checkpoint (17 ms per image at batch 1, about 320 images per second at batch 64, 1.5 GiB, 6 s load); Qwen and the new RetiZero runs are queued.
- [ ] RetiZero training time, memory and utilization, with its hardware stated.
- [ ] Final efficiency comparison table for both models.

## E. External evaluation (Messidor)

- [x] Archived 1,057-image cohort scored with the Retraining 2 adapter (185 of 207 Mild images predicted Normal).
- [ ] Raw Messidor-2 images, the grade source and the eye pairing located and verified.
- [ ] The four duplicate pairs with different patient IDs in the archived cohort reconciled.
- [ ] Verified cohort frozen and built at 512 pixels (CLAHE on and off) and 224 pixels.
- [ ] Chosen Qwen models and RetiZero scored on the verified cohort.
- [ ] Mild-versus-Normal analysis on the verified cohort.

## F. Analysis and tables

- [x] Tables builder with cross-checks against every saved score file (`finetuning/paper_tables/build_paper_tables.py`).
- [x] Per-run metrics with bootstrap intervals, condition summaries, rank table, CLAHE ablation, 4-bit vs BF16.
- [x] Confusion matrices and class-wise sensitivity, specificity, precision and F1 for every run.
- [x] Misclassified-image lists for the Retraining 2 replay (APTOS and archived Messidor).
- [x] ROC-AUC confirmed to use each class's own probability score.
- [x] Train, validation and test splits audited for exact duplicates; augmentation confirmed to be training-only in the code.
- [ ] Misclassified-image lists for the clean-suite headline runs.
- [ ] Representative misclassified images exported with true and predicted labels (S11).
- [ ] Adjacent-grade versus distant-grade error table, to explain QWK next to accuracy (S13).
- [ ] Class counts per split, before and after augmentation, for both models (S9).
- [ ] Paired comparison of Qwen and RetiZero on the same 361 images (paired bootstrap or McNemar).
- [ ] Calibration analysis for the headline models.
- [ ] Near-duplicate (not only byte-identical) check across splits.
- [ ] Excel workbook of the tables for Dr. Noor.

## G. Manuscript

- [ ] G1, S2: novelty, specific research gap, contributions and challenges.
- [ ] G3, S1, S10: literature review of DR work on APTOS and Messidor-2, with a comparison table.
- [ ] S3: colour-space explanation (BGR, RGB, LAB), citing the pixel check.
- [ ] S4: why these two models and what the comparison tests.
- [ ] S5: methodology flow diagram and both architecture diagrams.
- [ ] S6, S8, G2: justification of LoRA and training settings, with the trials.
- [ ] S7: with and without QLoRA.
- [ ] S9: CLAHE and augmentation for both models, with class counts.
- [ ] S11: why samples are misclassified, with examples.
- [ ] S12: accuracy in context of similar studies.
- [ ] S13: QWK explained next to accuracy.
- [ ] S14: computational comparison.
- [ ] Limitations: seed variance and collapsed runs, duplicate overlap in the original split, alpha fixed while rank changes, mixed scoring precision, archived Messidor cohort.
- [ ] Methods text matches the code (split sizes, scorer, checkpoint selection, preprocessing route).
- [ ] Every number in the manuscript traced to a file under `finetuning/paper_tables/output/`.
- [ ] Target journal chosen; format, data-availability and code-availability statements, dataset terms of use.

## H. Reproducibility and housekeeping

- [x] Paper scripts, notebooks and tables committed locally (8 commits).
- [x] `finetuning/Qwen3` restructured and tested.
- [ ] Restructure committed.
- [ ] Backup of `D:\Afterimage_Experiments\runs` (24.7 GB) and `finetuning/Qwen3/artifacts`. Neither is in git and no second copy exists.
- [ ] Snapshot of `D:\Afterimage_Experiments\code` (pinned runner, recovery script, launcher, dataset builder) committed to the repository with hashes. None of it is under version control.
- [ ] Remove the remaining read dependency on the C: copy for new 4-bit launches.
- [ ] Pilot test folders removed (about 1.2 GB on D:).
- [ ] Stale documents refreshed (`PAPER_TRAINING_CHECKLIST.md`, `D:\Afterimage_Experiments\RUN_CHECKLIST.md`).
- [ ] Final table rebuild from a clean checkout, with the manifest saved.
- [ ] Push to the shared repository when you approve it.

## I. Communication

- [ ] Results update to Dr. Noor (she asked for results as each experiment completes; she is leaving the university in December and asked to use her Gmail address).
- [ ] RetiZero brief sent to the teammate.
- [ ] Conference presenter for IEEE ICCA confirmed to Dr. Noor (still open from the email thread).
