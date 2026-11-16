# Paper training and evaluation checklist

> Superseded on 15 November 2026 by [docs/paper/DELIVERABLES_CHECKLIST.md](docs/paper/DELIVERABLES_CHECKLIST.md). The run counts below are a 14 November snapshot; all 21 planned Qwen runs and the three BF16 controls have since completed.

Audit date: 14 November 2026. A checked training box means a saved, verified training result exists. Final comparisons also require the specified evaluation. Keep the original-split and duplicate-excluded results in separate tables.

## Original APTOS split (367 test photos)

- [x] Replay the saved Retraining 2 adapter. All 367 APTOS and 1,057 archived Messidor grades and scores matched the archive exactly. [Replay report](finetuning/Qwen3/artifacts/paper_reproduction/training_batches/run02_reproduction_01/replay_run_02/replay_comparison.json)
- [x] Pair 1 — train rank 16, CLAHE on and off.
- [x] Pair 2 — train rank 16, CLAHE on and off.
- [x] Pair 3 — train rank 16, CLAHE on and off.
- [ ] **Score Pair 1 CLAHE-on in BF16.** Rerun just **Pair 1, CLAHE on** in the reproduction notebook with the existing tags after restarting its kernel and running setup. The cell reuses the finished adapter. Then run **Compare saved scores**. All five other branches already have BF16 scores. [Reproduction notebook](finetuning/Qwen3/notebooks/02_training.ipynb)
- [ ] **Deferred at your request:** three fresh historical training repeats using the archived CLAHE-on files and original launcher. Run them if the paper will claim a fresh reproduction of historical training. The saved-adapter replay verifies evaluation and the archive already has three historical repeats.

| Pair | CLAHE on accuracy / QWK / AUC | CLAHE off accuracy / QWK / AUC |
| --- | --- | --- |
| 1 | 82.29% / 0.9156 / 0.9461 (FP16 reference) | 77.93% / 0.8979 / 0.9357 |
| 2 | 49.32% / 0.0000 / 0.4870 | 64.85% / 0.8270 / 0.9028 |
| 3 | 62.94% / 0.7775 / 0.8936 | 72.48% / 0.8731 / 0.9348 |

Pair 2 CLAHE-on predicted grade 0 for all 367 images. Keep this result in the summary. Pair 1 CLAHE-off originally failed FP16 scoring, then passed BF16 scoring with the saved adapter. Report the three-pair mean and spread once Pair 1 shares the same scoring precision. Document the original split's known exact duplicate overlap.

## Duplicate-excluded APTOS suite (361 test photos)

12 of 21 planned Qwen runs completed. Compare seed pairs 1 = 42/3407, 2 = 43/3408, and 3 = 44/3409. [Duplicate-excluded notebook](D:/Afterimage_Experiments/02_training.ipynb)

| Setting | Seed pair 1 | Seed pair 2 | Seed pair 3 |
| --- | --- | --- | --- |
| Rank 8, CLAHE on | Done | Done | Done |
| Rank 16, CLAHE on | Done | Done | Done |
| Rank 32, CLAHE on | Done | Done | Done |
| Rank 16, CLAHE off | Done | Done | Done |
| Rank 8, CLAHE off | Interrupted before checkpoint | Pending | Pending |
| Rank 32, CLAHE off | Pending | Pending | Pending |
| Rank 16, CLAHE on, previous train/val split | Pending | Pending | Pending |

- [ ] Rank 8, CLAHE off — rerun seed pair 1 with the existing tag, then train seed pairs 2 and 3. The seed-1 marker says running, but no GPU job, result, or saved epoch checkpoint was found at the audit.
- [ ] Rank 32, CLAHE off — train seed pairs 1, 2, and 3. Together with rank 8 off these six runs complete the planned three-rank × two-CLAHE study.
- [ ] Rank 16, CLAHE on, previous train/val split — train seed pairs 1, 2, and 3 if reporting the effect of duplicate exclusion. Use the same 361 evaluation photos and current runner for both split policies. These controls answer a different question from the 367-image reproduction.

## Reviewer-facing experiments

- [ ] **S7 — Qwen with versus without QLoRA.** The planned control is LoRA on a BF16 base. Train rank 16, CLAHE on, duplicate-excluded, using seed pairs 1, 2, and 3. Hold the rest of the training protocol fixed and use the same inference settings for both sides. The existing 4-bit rank-16 CLAHE-on runs are the training reference. Confirm the baseline definition with the team and obtain BF16 weights and a GPU with sufficient memory. The BF16 scoring fix for a 4-bit base does not fulfill this experiment.
- [ ] **S4/S10/S14 — matched RetiZero baseline and resource comparison.** Train the three planned RetiZero repeats (seeds 42, 43, 44) on the prepared 224-pixel duplicate-excluded data and score the same 361 test IDs as Qwen. Record time, GPU and system memory, and utilization under comparable measurement conditions. Report trainable parameter counts and discuss complexity separately.
- [ ] **G2/S6/S8 — tuning evidence.** Report the completed rank trials and that changing rank while holding alpha at 16 also changes alpha/r. Explain the basis for learning rate, dropout, bias and other fixed settings. Additional training is needed if the paper claims they were optimized or isolates the effect of rank alone without evidence from existing trials.
- [ ] **S9 — document augmentation and RetiZero preprocessing.** Report per-class counts and online transforms versus added files, and explain RetiZero's CLAHE choice. A new RetiZero CLAHE run is only needed if the team will make an empirical claim about CLAHE on RetiZero.
- [ ] **Messidor external evaluation.** Verify original photos, independent grades, and eye pairing. Then score the already-trained models needed for paper claims on the same cohort with the appropriate preprocessing. This is evaluation work. The exact historical replay alone cannot establish the modern models' external performance. A matched Messidor CLAHE comparison requires verified raw sources.

For each new run, retain the config, selected adapter, prediction export, per-grade results, and attempt logs. Six original-split training attempts took about 52–110 minutes each and recorded CPU/GPU utilization and memory. Successful scoring took about 5–6 minutes per adapter. The full manuscript response is tracked in [REVIEWER_CHECKLIST.md](REVIEWER_CHECKLIST.md).
