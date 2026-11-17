# Remaining work and phases

Updated 17 November 2026. This is the ordered view of what is done and what is left. The item-by-item list is [DELIVERABLES_CHECKLIST.md](DELIVERABLES_CHECKLIST.md) (not yet refreshed for 17 November; see phase 7).

## Where things stand

- **Qwen:** the final queue `D:\Afterimage_Experiments\final_runs\final_01` is finished (41 of 42 jobs scored with the FP16 protocol; one run scored with a labelled FP32 fallback after an FP16 overflow). Its results are in the paper tables (`finetuning/paper_tables/output/final_01/`).
- **RetiZero:** trained from the official pretrained checkpoint. All 18 jobs of `D:\Afterimage_Experiments\retizero_runs\retizero_01` are done: replay, original split and clean split (three seeds each), CLAHE (three seeds), timing, and archived Messidor. Results: [RESULTS_RETIZERO.md](../../finetuning/RetiZero/RESULTS_RETIZERO.md).
- **Efficiency:** tables built from saved records, with power and energy now recorded for the RetiZero runs (`finetuning/paper_tables/output/efficiency/`).
- **Headline set:** verified from the image files ([HEADLINE_VERIFICATION.md](HEADLINE_VERIFICATION.md)).
- **Messidor:** no raw Messidor-2 data on this PC ([MESSIDOR_SOURCES_FOUND.md](MESSIDOR_SOURCES_FOUND.md)).
- **Git:** everything up to commit `db6901a` is committed locally. Not pushed. `03_final_runs.ipynb` holds uncommitted run outputs only.

## Key numbers so far (361 clean test images)

| Model | Accuracy | Notes |
| --- | --- | --- |
| Qwen zero-shot, 4-bit, CLAHE on / off | 46.54% / 54.57% | BF16 CLAHE on: 52.63% |
| Qwen r16 CLAHE on, lr 5e-4 (3 seeds) | 69.71 ± 13.46% | collapsed runs |
| Qwen r16 CLAHE on, lr 2e-4 / 1e-4 (3 seeds) | 78.12 ± 2.22% / 74.98 ± 2.08% | no collapse |
| RetiZero, clean split (3 seeds) | 79.04 ± 1.69% | |
| RetiZero, clean split with CLAHE (3 seeds) | 79.32% (mean) | no consistent CLAHE effect |
| RetiZero, original split (3 seeds) | 80.11–81.47% | old checkpoint: 82.29% on 367 |
| Archived Messidor (1,057) | Qwen about 43–54%, RetiZero 32–50% | weak transfer for both |
| Time per image | Qwen 0.85–0.90 s, RetiZero about 16 ms | batch size 1 |

## Phase 0. Close out the Qwen queue — done

- [x] Alpha 32, seed 43: FP16 overflow at `language_model.layers.2.mlp.down_proj`; scored with the labelled FP32 fallback (`score_fp32_fallback.py`): predicts grade 0 for every image (50.14% APTOS, 44.28% Messidor). Of the six alpha runs at 5e-4, three predict one grade, one predicts only grades 0 and 1 (50.97%), two are healthy (74.52%, 72.85%).
- [x] Overflow recorded in `final_01\summary.md` and `.verification/final_runs/`.
- [x] `final_01` read into `build_paper_tables.py` (set F, tables F1–F6).
- [x] `effective_scaling` record error handled: `train_result.json` stores 16/rank, so alpha 8 and 32 runs read 1.0; the tables take alpha/r from `adapter_config.json`. Still to state in the methods audit (phase 6).
- [x] Work committed in small local commits.
- [x] Leftover folder `retizero_01__dev_01` removed.

## Phase 1. Decisions

- [x] **Learning rate:** main results stay at 5e-4. The 1e-4 and 2e-4 runs are reported as tuning and stability evidence. A lower-rate grid can still be run if Dr. Noor asks (phase 2); choose its rate on validation loss, not test accuracy.
- [x] **CLAHE:** report both on and off.
- [x] **RetiZero batch size:** 128 per optimizer step, processed 32 at a time (original notebook).
- [x] **Efficiency scope:** fill the gaps from saved records; rerun only what cannot be derived (see phase 5).
- [x] **Headline set:** clean 361-image suite proposed as main result, verified. Correct wording: 19 of the 367 original test images have an exact copy outside test (17 train, 2 validation), 3 with a different grade; 361 = 367 − those 3 − 3 test–test duplicates. Waiting for Dr. Noor's confirmation.
- [ ] **Messidor sources:** you obtain raw Messidor-2 (ADCIS registration) and the adjudicated grades (Kaggle account). See phase 4.
- [ ] **Send the results update** to Dr. Noor ([RESULTS_UPDATE_DRAFT.md](RESULTS_UPDATE_DRAFT.md)). Before sending: add the RetiZero results (79.04 ± 1.69%, CLAHE, Messidor) — the draft predates them.

## Phase 2. Qwen repeat runs (only if Dr. Noor asks for a lower learning rate)

- [ ] Main grid at the chosen rate: rank 8, 16, 32 × CLAHE on/off × three seeds (section 8 of `03_final_runs.ipynb`, about 20 hours).
- [ ] BF16 base with CLAHE off, three seeds (worker not built; only if CLAHE off becomes the headline).
- [ ] Score and time the new adapters on the 361 images and archived Messidor.
- [ ] S6 (rank with alpha/r fixed): run at the chosen rate, or write it as a limitation — the 5e-4 alpha runs do not answer it.

## Phase 3. RetiZero — done

- [x] Official pretrained `RetiZero.pth` downloaded (SHA-256 `7d06d294…601ede`, source in `RetiZero.pth.source.txt`), hash pinned in the notebook.
- [x] Readiness check ([READINESS_CHECK.md](../../finetuning/RetiZero/READINESS_CHECK.md)); non-finite FP16 loss now skips the step (counted), as the original did.
- [x] Pilot `pilot_02` passed with pretrained weights.
- [x] 224-pixel CLAHE APTOS and 224-pixel archived Messidor datasets built (`finetuning/RetiZero/data/build_retizero_datasets.py`).
- [x] All 18 jobs: original split, clean split, CLAHE, timing, archived Messidor. Zero skipped steps from non-finite loss.
- [x] Power logged every 5 s; energy per job in `retizero_01\energy_per_job.csv` (whole-card power, includes about 20 W idle).
- [ ] RetiZero results into `build_paper_tables.py` (new set; the efficiency tables already read the RetiZero folder but should be rebuilt to include the full runs).
- [ ] RetiZero on the verified Messidor cohort (after phase 4).
- [ ] Optional: export a new RetiZero checkpoint in the app's format; Grad-CAM (open questions in the plan).

## Phase 4. External evaluation on verified Messidor

- [ ] You: download raw Messidor-2 images and eye pairing (ADCIS) and the adjudicated grades (Kaggle), onto D:.
- [ ] Match them to the archived 1,057 images; document how the 1,057 were chosen.
- [ ] Reconcile the four duplicate pairs with different patient IDs.
- [ ] Freeze the verified cohort; build at 512 pixels (CLAHE on and off) and 224 pixels (direct raw → 224, removing the current raw → 512 → 224 caveat).
- [ ] Score the chosen Qwen models and RetiZero on it; Mild-versus-Normal analysis.

## Phase 5. Analysis and tables

- [ ] **Paired comparison** Qwen vs RetiZero on the same 361 images (paired bootstrap and McNemar). Inputs now exist for both models.
- [ ] **Efficiency, remaining gaps** (from `output/efficiency/GAPS.md`):
  - [ ] Rebuild the efficiency tables with the full RetiZero training runs.
  - [ ] Short scoring-only timing job for the 7 Qwen configurations without warm per-image timing (no retraining).
  - [ ] Qwen power/energy: not recorded in past runs; measure during the timing job, or state as a limitation.
  - [ ] Final Qwen-vs-RetiZero efficiency table (S14).
- [ ] Misclassified-image lists and exported examples (S11).
- [ ] Adjacent-grade versus distant-grade error table (S13).
- [ ] Class counts per split, before and after augmentation, for both models (S9).
- [ ] Calibration analysis for the headline models.
- [ ] Near-duplicate check across splits (approximate count is in HEADLINE_VERIFICATION.md section f; decide whether to report it).
- [ ] Excel workbook of the tables for Dr. Noor.

## Phase 6. Manuscript

- [ ] Reviewer points G1–G3 and S1–S14 answered in the text (deliverables checklist, section G).
- [ ] Methods: clean split and how 361 arises; scorer; checkpoint selection (Qwen by validation loss, RetiZero by validation accuracy); preprocessing (Qwen 512 with CLAHE, RetiZero 224; CLAHE tile size differs at 224).
- [ ] Limitations: seed variance and collapsed runs at 5e-4; the FP16 overflow and FP32 fallback; the `effective_scaling` record error; duplicate overlap in the original split; mixed scoring precision; archived Messidor cohort (processed archive, raw → 512 → 224 for RetiZero); energy is whole-card power.
- [ ] S9 now has evidence for both models: CLAHE shows no consistent benefit for RetiZero either.
- [ ] Every number traced to a file under `finetuning/paper_tables/output/`.
- [ ] Target journal, format, data and code availability statements.

## Phase 7. Reproducibility and housekeeping

- [ ] Refresh `DELIVERABLES_CHECKLIST.md` and `finetuning/RetiZero/README.md` status for 17 November.
- [ ] Back up `D:\Afterimage_Experiments\{runs,final_runs,retizero_runs,datasets,assets}`. No second copy exists.
- [ ] Commit a snapshot of `D:\Afterimage_Experiments\code` with hashes.
- [ ] Remove the read dependency on the C: copy for new 4-bit launches.
- [ ] Remove pilot folders when no longer needed; refresh stale documents (`PAPER_TRAINING_CHECKLIST.md`, `RUN_CHECKLIST.md`).
- [ ] Final table rebuild from a clean checkout.
- [ ] Push to the shared repository when you approve it.

## Dependencies in one view

| Phase | Status | Needs | GPU time |
| --- | --- | --- | --- |
| 0 | Done | — | — |
| 1 | Two items open | You and Dr. Noor | None |
| 2 | Optional | Dr. Noor asking for a lower rate | About 20 hours |
| 3 | Done except tables | — | — |
| 4 | Blocked | Raw Messidor-2 and grades | Scoring only |
| 5 | Can start now | Nothing for the paired comparison | Under 1 hour for the timing job |
| 6 | After 1–5 | — | None |
| 7 | Any time | — | None |
