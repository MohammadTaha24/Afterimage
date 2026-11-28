# Handoff for ChatGPT: where the Afterimage paper revision came from and where it is now

Written 25 November 2026 for ChatGPT, which worked on this project until about 15 November. The work since then was done with Claude Code. This file brings you up to date. Read it before the other two handoffs:

- `HANDOFF_FIGURES_GUIDE.md`: what each figure shows
- `HANDOFF_REPORT_WRITER.md`: the report and plan we want from you

The open-items list is `REMAINING_WORK.md`.

Your last handoff was `.claude/handoffs/2026-11-15-182947-afterimage-paper-progress.md`. Everything in it still holds, except the items marked **changed** below.

## 1. The project in brief

- **Afterimage** is a diabetic retinopathy (DR) screening app and paper. It grades a fundus photo 0–4 on the ICDR scale.
- **The paper** compares two models fine-tuned on APTOS 2019:
  - **Qwen3-VL-8B**, a general vision-language model: 4-bit base, LoRA adapter, scored by the likelihood of five fixed answers.
  - **RetiZero**, a retina-specific ViT-L/16 foundation model: every weight trained, with a 5-class head.
- **Status:** major revision, with supervisor Dr. Noor. The reviewer comments are in `EA_REVIEW.docx` (G1–G3 and S1–S14; summary in `REVIEWER_CHECKLIST.md`).
- **Team:** the owner does the Qwen side and runs every GPU job on their PC (RTX 5070 Ti 16 GB). Colleague Sinan does the RetiZero side and the shared analysis scripts; he sends scripts, and the owner runs them.

## 2. Where we were on 15 November (your last state)

- **Qwen:** 24 runs on the clean 361-image APTOS test set: ranks 8, 16 and 32 × CLAHE on/off at lr 5e-4, a previous-split control and BF16-base controls, 3 seeds each. Many lr 5e-4 runs had collapsed to one or two grades.
- **Original split:** a 367-image CLAHE ablation and a replay of the archived adapter. The replay matched exactly.
- **Blocked:**
  - the RetiZero baseline (no official pretrained checkpoint)
  - verified Messidor (no raw images, grades or eye pairing)
- **Not started:** tables, statistics, efficiency, figures, manuscript work.

## 3. What happened from 15 to 25 November

| Date | What was done | Where |
|---|---|---|
| 15–16 Nov | Paper-tables builder. Qwen code reorganised into `legacy/`, `notebooks/` and `paper_revision/`. **Final Qwen queue `final_01`**: lr 1e-4 and 2e-4 (rank 16, CLAHE on, 3 seeds), alpha 8 and 32 at 5e-4, zero-shot 4-bit and BF16, the original paper adapter on the 367 images, and archived Messidor scoring. One FP16 overflow was scored with a labelled FP32 fallback. | `finetuning/Qwen3/paper_revision/final_runs.py`, `notebooks/03_final_runs.ipynb`, `D:\Afterimage_Experiments\final_runs\final_01` |
| 16 Nov | Paper plan, deliverables checklist, remaining-work phases | `docs/paper/PLAN.md`, `DELIVERABLES_CHECKLIST.md`, `REMAINING_WORK.md` |
| 17 Nov | **RetiZero unblocked**: the official pretrained `RetiZero.pth` was downloaded and hash-pinned. **Queue `retizero_01`**: 18 jobs (replay, original split ×3, clean split ×3, CLAHE ×3, timing, archived Messidor). RetiZero reaches 79.04 ± 1.69% on APTOS-361. | `finetuning/RetiZero/`, `D:\Afterimage_Experiments\retizero_runs\retizero_01` |
| 17 Nov | Efficiency tables, headline-set verification, the Messidor source search, a results email draft for Dr. Noor | `docs/paper/HEADLINE_VERIFICATION.md`, `finetuning/paper_tables/output/efficiency/` |
| 20 Nov | APTOS duplicates literature note. The experiment notebooks and launcher scripts that had lived outside git were copied into the repo. | `docs/paper/APTOS_DUPLICATES_LITERATURE.md`, `external/Afterimage_Experiments/code/` |
| 22 Nov | **Sinan's branch `revision-analysis-cpu`:** CPU analyses for every reviewer point. These cover hierarchical bootstrap CIs, McNemar with Holm, calibration, class-wise metrics, error structure, Mild vs Normal, near-duplicate audit, colour-conversion pixel check and Messidor label audit. He also wrote a GPU runner for the remaining jobs. **All his numbers were reproduced exactly on the owner's PC.** | `finetuning/paper_tables/build_revision_analysis.py`, `build_image_audits.py`, `finetuning/revision_jobs/` |
| 22–24 Nov | **Queue `final_02`, the lr 2e-4 grid:** 27 new adapters (rank 8/16/32 × alpha 16 or alpha = rank × CLAHE on/off, 3 seeds), each scored on APTOS-361 and archived Messidor. 81 of 81 jobs passed. Validation loss had selected lr 2e-4, so the ablation was re-run at the chosen rate. | `finetuning/Qwen3/paper_revision/orchestrator_02.py`, `notebooks/04_training_orchestrator.ipynb`, `final_runs\final_02` |
| 24 Nov | **Verified Messidor-2 obtained** from a Kaggle download in the owner's Downloads: 1,748 raw images, Krause 2018 adjudicated grades, eye pairing. Validated: 1,744 gradable, grades 1,017/270/347/75/35, 874 exams. | `D:\Afterimage_data_share\Messidor-2\` (not in git) |
| 24 Nov | Rank section of the evidence note rewritten from the lr 2e-4 grid | branch `qwen-rank-lr2e-4`, `docs/paper/REVISION_EVIDENCE.md` §1 |
| 25 Nov | Sinan's GPU runner fixed so it cannot overwrite the owner's queue files:<br>– own energy file and pilot folder<br>– `--rank-sweep` removed, since the grid already exists<br>– CLAHE-off lr 2e-4 adapters added<br>– analysis reads the CLAHE-off archived-Messidor scores and checks all 27 `final_02` runs | commit `2c833aa`, merged into `main` |
| 25 Nov | **Revision GPU queue**: 28 scoring jobs, no training. Every model on verified Messidor-2, plus warm timing for 7 Qwen configurations. All passed: timing reproduces saved predictions with 0 mismatches, and all 221,818 earlier result files are unchanged. | `finetuning/revision_jobs/revision_gpu_jobs.ipynb`, commit `6168da9` |
| 25 Nov | Tables rebuilt: 148 runs, all reproducing their stored reference. Evidence note gained §5 (Messidor-2) and the timing rows. | `finetuning/paper_tables/output/revision_analysis/`, `REVISION_EVIDENCE.md` |
| 25 Nov | **22 paper figures** (notebook; F20–F22 are training curves from the run logs), the report-writer handoff, and these handoffs | `finetuning/paper_figures/make_paper_figures.ipynb`, `docs/paper/HANDOFF_*.md` |
| 25 Nov | **Results package:** all run records (logs, training histories, predictions, summaries; no weights or images), code, tables, figures and a progress report in one folder. Its notebook rebuilds every table from the logs, checks it, and draws every figure. | `D:\Afterimage_data_share\afterimage_results_20261128\`, built by `finetuning/paper_figures/build_results_package.py` |

**Changed since your handoff:**
- **RetiZero:** no longer blocked; done.
- **Verified Messidor:** no longer blocked; done.
- **Headline Qwen setting:** now lr 2e-4, chosen on validation loss, not 5e-4.
- **Main test set:** APTOS-361, the clean split, confirmed.
- **Fresh historical repeats:** still deferred.

## 4. Where we are now: the main results

All numbers come from `finetuning/paper_tables/output/revision_analysis/`. `REVISION_EVIDENCE.md` cites the table behind each one.

1. **In-domain (APTOS-361), Qwen ≈ RetiZero.**
   - Qwen lr 2e-4: 78.12 ± 2.22%, QWK 0.894.
   - RetiZero: 79.04 ± 1.69%.
   - Difference −0.92 pt [−5.26, +3.14].
2. **External (verified Messidor-2): Qwen transfers better.**
   - Qwen 55.4% against RetiZero 40.2%, a difference of +15.1 pt [+5.6, +27.2]; QWK +0.134 [+0.063, +0.208].
   - All per-seed McNemar tests are significant after Holm adjustment.
3. **Both models drop sharply out of domain.** Referable-DR sensitivity is only about 0.3 at specificity about 0.97, and Mild vs Normal is at chance. This is a limitation to report.
4. **lr 5e-4, the original paper setting, is unstable.** 10 of 63 fine-tuned runs fail, all of them at 5e-4. At lr 2e-4 none fails, and LoRA rank and alpha make no significant difference.
5. **CLAHE gives no benefit to any fine-tuned model.** It hurts zero-shot Qwen.
6. **The adapter matters:** +31.6 pt on APTOS and +11.5 pt on Messidor-2 against zero-shot.
7. **Efficiency:**
   - Qwen: 8.82 B parameters (51 M trainable), about 0.8 s/image, 8.7 GiB at inference.
   - RetiZero: 304 M parameters, all trainable, 18 ms/image, 1.2 GiB.
8. **Integrity:**
   - every metric is recomputed from per-image predictions
   - the archived Messidor inputs are pixel-identical to a raw rebuild
   - no near-duplicate inflation was found

## 5. Decisions taken (do not reopen without the owner)

- **Headline Qwen model:** r16, alpha 16, CLAHE on, **lr 2e-4**, checkpoint chosen on validation loss. lr 5e-4 is reported as the original setting and the reason for the change.
- **Headline test set:** APTOS-361, the clean split with duplicates removed. Original-split results are kept separate.
- **External test:** verified Messidor-2, all 1,744 gradable images. CIs are clustered by exam (both eyes resampled together). Archived Messidor (1,057) is a secondary check.
- **Failed run:** one that predicts at most 3 of the 5 grades.
- **RetiZero:** trains all weights. Do not call it a LoRA model.
- **Statistics:** hierarchical bootstrap (images or exams × seeds), with per-seed exact McNemar tests and Holm adjustment.
- **Fresh historical training repeats:** stay deferred.

## 6. Where things live

| What | Where |
|---|---|
| Git repo (work here only) | `D:\Afterimage_Reproduction`, branch `main`. GitHub `MohammadTaha24/Afterimage`, up to date on branch `afterimage-re`; GitHub `main` is behind. |
| Run evidence (read-only) | `D:\Afterimage_Experiments\{runs, final_runs\final_01, final_runs\final_02, retizero_runs\retizero_01, retizero_runs\retizero_02, datasets}` |
| Messidor-2 package | `D:\Afterimage_data_share\Messidor-2\` |
| Analysis tables | `finetuning/paper_tables/output/revision_analysis/` (`SUMMARY.md` is the digest) |
| Evidence note | `docs/paper/REVISION_EVIDENCE.md` |
| Figures | `finetuning/paper_figures/output/` (made by `make_paper_figures.ipynb`) |
| Open items | `docs/paper/REMAINING_WORK.md` |
| Reviewer map | `REVIEWER_CHECKLIST.md` |

## 7. How the owner works (please respect this)

- **Everything stays on D:.** The C: drive is nearly full.
- **Run folders are evidence.** Never edit or delete them.
- **The owner runs every GPU job from a notebook themselves.** Hand over notebooks or scripts with a preflight and full logging, not instructions to type.
- **Commits and pushes happen only when the owner asks.** Commits carry no AI co-author lines.
- **Plain, short writing.** Every number must trace to a file, and anything uncertain is marked as such.
