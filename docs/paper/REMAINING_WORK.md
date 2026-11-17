# Remaining work and phases

Written 16 November 2026, after the Qwen queue `final_01` finished. This is the ordered view of what is left. The full item-by-item list, including finished items, is [DELIVERABLES_CHECKLIST.md](DELIVERABLES_CHECKLIST.md).

## Where things stand

- **Qwen queue (`D:\Afterimage_Experiments\final_runs\final_01`):** 40 of 41 jobs finished in 22.2 hours. Zero-shot, the timing benchmark, the audit of the original paper adapter, archived Messidor scoring, the learning-rate study and the alpha study are done.
- **One run could not be scored with the standard scorer:** rank 16, alpha 32, seed 43. Its arithmetic overflows in half precision at the third language layer. It is now scored with a labelled FP32 fallback and is a collapsed run (every image predicted grade 0).
- **RetiZero:** code, tests, pilot and the replay of the saved checkpoint are done. No training run exists yet; all of them wait for the official pretrained checkpoint.
- **Nothing since commit `d326be1` is committed**, and nothing is pushed.

## Phase 0. Close out the finished Qwen queue

- [x] Alpha 32, seed 43: scored with the FP32 fallback (`score_fp32_fallback.py`), labelled as such in the summary, with its archived Messidor job. It predicts grade 0 for every image: 50.14% on APTOS (QWK 0.0), 44.28% on Messidor. It is a collapsed run. Of the six alpha runs at 5e-4, three predict a single grade (alpha 8 seed 44; alpha 32 seeds 43 and 44), one predicts only grades 0 and 1 (alpha 32 seed 42, 50.97%), and two are healthy (alpha 8 seeds 42 and 43).
- [x] Overflow recorded as a finding in `final_01\summary.md` ("FP16 scoring failures"), with the diagnosis in `.verification/final_runs/`.
- [ ] Known record error to mention in the methods audit: `train_result.json` stores `effective_scaling` as 16 / rank (the pinned worker assumes alpha 16), so it reads 1.0 for the alpha 8 and alpha 32 runs. The adapters themselves were built with the right alpha (`adapter_constructor_arguments.json`, `adapter_config.json`).
- [ ] Read `final_01` into `finetuning/paper_tables/build_paper_tables.py` (zero-shot, learning-rate study, alpha study, benchmark, Messidor, original-adapter audit).
- [ ] Commit the work since `d326be1` in small groups: restructure, final-runs queue, RetiZero folder, paper documents. Local only.
- [ ] Remove the leftover folder `D:\Afterimage_Experiments\retizero_runs\retizero_01__dev_01` if it is still there (2.3 GB).

## Phase 1. Decisions and the update to Dr. Noor

Nothing in phases 2 to 4 should start before the first three are answered.

- [ ] **Learning rate for the main results.** At 5e-4 the three seeds gave 69.71 ± 13.46% with collapsed runs. At 2e-4 they gave 78.12 ± 2.22% and at 1e-4 74.98 ± 2.08%, with no collapse. Validation loss does not separate 1e-4 from 2e-4. Decide whether to repeat the main grid at a lower rate, and at which one.
- [ ] **Headline result set.** Clean 361-image suite as the main result; the original 367-image result as historical reference only.
- [ ] **CLAHE position.** Zero-shot is better with CLAHE off (54.57% against 46.54%), and the earlier fine-tuned runs showed no CLAHE benefit.
- [ ] **RetiZero batch size.** 128 images per step as in the original notebook (my recommendation) or 8 as in the earlier draft script.
- [ ] **Efficiency measurement scope** (fill the gaps, or repeat every run).
- [ ] **Messidor sources:** who holds the raw images, the grades and the eye pairing.
- [ ] Send the results update to Dr. Noor (Gmail address; she leaves in December).

## Phase 2. Qwen repeat runs (only if phase 1 adopts a lower learning rate)

- [ ] Main grid at the chosen rate: rank 8, 16 and 32, CLAHE on and off, three seeds (section 8 of `03_final_runs.ipynb`, about 20 hours).
- [ ] BF16 base with CLAHE off, three seeds (only if CLAHE off becomes the headline; its worker is not built).
- [ ] Score and time the new adapters on the 361 test images and on archived Messidor.
- [ ] Rank with alpha/r held fixed at the chosen rate, or a written limitation. The alpha runs at 5e-4 collapsed, so they do not answer S6.

## Phase 3. RetiZero

- [ ] Download the official pretrained `RetiZero.pth` to `D:\Afterimage_Experiments\assets`, record source, date, size and SHA-256, and paste the hash into Setup of `02_retizero_runs.ipynb`.
- [ ] Repeat the pilot with the pretrained checkpoint (new `pilot_tag`).
- [ ] Build the 224-pixel CLAHE dataset and the 224-pixel archived Messidor build.
- [ ] Run the full queue: 16 jobs waiting (original split, clean split, CLAHE, timing, Messidor), three seeds each.
- [ ] Record training time, memory and utilization for the efficiency table.

## Phase 4. External evaluation on verified Messidor

- [ ] Locate and verify the raw Messidor-2 images, grades and eye pairing.
- [ ] Reconcile the four duplicate pairs with different patient IDs.
- [ ] Freeze the verified cohort and build it at 512 pixels (CLAHE on and off) and 224 pixels.
- [ ] Score the chosen Qwen models and RetiZero on it; Mild-versus-Normal analysis.

## Phase 5. Analysis and tables

- [ ] Paired comparison of Qwen and RetiZero on the same 361 images.
- [ ] Final efficiency table for both models (S14), with telemetry summed over all attempts of a run.
- [ ] Misclassified-image lists and exported examples (S11).
- [ ] Adjacent-grade versus distant-grade error table (S13).
- [ ] Class counts per split, before and after augmentation (S9).
- [ ] Calibration analysis for the headline models.
- [ ] Near-duplicate check across splits.
- [ ] Excel workbook of the tables for Dr. Noor.

## Phase 6. Manuscript

- [ ] Reviewer points G1 to G3 and S1 to S14 answered in the text (list in the deliverables checklist, section G).
- [ ] Limitations: seed variance and collapsed runs, the half-precision overflow, duplicate overlap in the original split, mixed scoring precision, archived Messidor cohort.
- [ ] Methods text matches the code; every number traced to a file under `finetuning/paper_tables/output/`.
- [ ] Target journal, format, data and code availability statements.

## Phase 7. Reproducibility and housekeeping

- [ ] Back up `D:\Afterimage_Experiments\runs`, `final_runs` and `retizero_runs`. No second copy exists.
- [ ] Commit a snapshot of `D:\Afterimage_Experiments\code` with hashes.
- [ ] Remove the read dependency on the C: copy for new 4-bit launches.
- [ ] Remove pilot folders; refresh stale documents.
- [ ] Final table rebuild from a clean checkout.
- [ ] Push to the shared repository when you approve it.

## Dependencies in one view

| Phase | Needs | GPU time |
| --- | --- | --- |
| 0 | Nothing | Under 1 hour |
| 1 | You and Dr. Noor | None |
| 2 | Phase 1 learning-rate decision | About 20 hours, more with the BF16 control |
| 3 | The pretrained checkpoint | Not measured for full runs; one epoch takes about 25 s before validation |
| 4 | The Messidor sources | Scoring only |
| 5 | Phases 2 and 3 for the paired comparison; the rest can start now | None |
| 6 | Phases 1 to 5 | None |
| 7 | Can run alongside any phase | None |

Phases 2 and 3 share one GPU and cannot overlap. Phase 3 does not depend on the phase 1 decisions except the batch size, so it can start as soon as the checkpoint is on the PC.
