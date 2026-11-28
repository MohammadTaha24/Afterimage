# Handoff: results report and finish-the-paper plan (for GPT 6.2 Astra)

Prepared 25 November 2026. **Read first:**

- `docs/HANDOFF_PROGRESS_FOR_CHATGPT.md`: how the project got from your last state (15 November) to now
- `docs/HANDOFF_FIGURES_GUIDE.md`: what each figure shows
- `docs/REMAINING_WORK.md`: the current open list

You get either the report bundle (`paper_report_bundle_<date>.zip`) or the full results package (`afterimage_results_<date>.zip`). Both have these three folders:

- `figures/`: 22 figures (PNG, PDF and the plotted numbers as CSV), plus `figure_index.csv` with draft captions. F20–F22 are training curves from the run logs.
- `tables/`: every analysis table
- `docs/` (report bundle only): the evidence note and the checklists

The results package also holds every run's logs, training histories and per-image predictions (`experiments/`), and the notebook that rebuilds the tables and figures from them. You do not need those to write the report.

The authors will also give you the reviewers' report (`EA_REVIEW.docx`).

## 1. The study in one paragraph

Afterimage grades diabetic retinopathy (DR) from a fundus photo on the 5-level ICDR scale: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 PDR. The manuscript compares two models fine-tuned on APTOS 2019:

- **Qwen3-VL-8B**, a general vision-language model. It is loaded in 4 bits and gets a LoRA adapter (rank 16, alpha 16). It is scored by picking the most likely of five fixed answers.
- **RetiZero**, a retina-specific ViT-L/16 foundation model with a 5-class head. Every weight is trained; despite "LoRA r8" in some table labels, it is not a LoRA fine-tune.

The paper is under major revision. Since the review, the authors have rebuilt and rerun everything:

- **A clean APTOS split:** 2,786 / 357 / 361 images, with duplicates removed.
- **Three seeds per condition.**
- **Ablations** of LoRA rank, alpha, learning rate, CLAHE and the adapter itself.
- **An external test on the verified Messidor-2 release:** 1,744 gradable images, 874 exams, adjudicated grades.
- **Efficiency measurements** on one PC.
- **A full statistical analysis:** hierarchical bootstrap CIs, and exact McNemar tests with Holm adjustment.

## 2. What we need from you

**A. Results report.** For the authors and the supervisor; it is not the manuscript itself.
- One section per reviewer point: G1–G3 and S1–S14 (see `docs/REVIEWER_CHECKLIST.md`).
- In each section:
  - what the reviewer asked
  - the evidence we now have, with numbers and figure IDs
  - whether the point is **answered**, **partly answered** or **open**
  - the sentence(s) the manuscript or response letter should say
- Use the figures by ID (F01–F22) and propose which go in the main text and which in the supplement. `HANDOFF_FIGURES_GUIDE.md` has a proposal.

**B. Finish-the-paper plan.** A prioritised list of everything still needed to resubmit. For each item give:
- the owner type (author writing, literature search, computation, decision by supervisor)
- the effort
- the dependency

Section 6 below lists what we know is open. Add anything you find missing when you check the reviewer report against the evidence.

**C. Draft response-to-reviewers table:** reviewer point → change made → where in the manuscript (section placeholder).

## 3. Rules

1. **Every number must come from a file in the bundle.** Cite the file (for example `tables/messidor2_aggregate.csv`) or the figure ID. Do not compute new statistics from summary numbers, and do not round differently from the source without saying so. If something you need is not in the bundle, list it as open; do not estimate it.
2. **Means are over three seeds** (mean ± SD) unless a row says one run. CIs are 95% hierarchical bootstrap intervals:
   - APTOS: images × seeds
   - archived Messidor: patients × seeds
   - Messidor-2: exams × seeds
   
   "Δ" means A − B, as named in the table.
3. **Suspected vs verified.** Domain shift as the cause of Mild → Normal errors is *suspected*; no grader has confirmed it. Keep that wording.
4. **One definition of a failed run:** it predicts at most 3 of the 5 grades. Note that the `collapsed` column in the CSVs uses "at most 2". The two differ for one run only: r16 CLAHE-on lr 5e-4, seed 42.
5. **Headline model:** Qwen r16, CLAHE on, **lr 2e-4**, chosen on validation loss. The original paper setting was lr 5e-4, which is unstable. Report 5e-4 as the original setting and as the reason for the change, not as the main result.
6. **Do not describe RetiZero as a LoRA model.** It trains all 304 M weights.
7. **Write plainly.** Short sentences, no hype. The audience is clinicians and ML reviewers.

## 4. What the evidence says (orientation only; the files are authoritative)

Source: `docs/REVISION_EVIDENCE.md`, which cites the table for every number.

| Finding | Key numbers | Files / figures |
|---|---|---|
| In-domain, Qwen ≈ RetiZero | Qwen lr 2e-4 78.12 ± 2.22%, QWK 0.894; RetiZero 79.04 ± 1.69%; Δ −0.92 pt [−5.26, +3.14] | `head_to_head_aggregate.csv`; F02, F03 |
| External (Messidor-2): Qwen transfers better | 55.4% vs 40.2%; Δ +15.1 pt [+5.6, +27.2]; QWK +0.134 [+0.063, +0.208]; all 3 per-seed McNemar tests significant after Holm | `messidor2_aggregate.csv`, `messidor2_pairs.csv`; F02, F03 |
| Both models lose a lot out of domain; referable-DR sensitivity is about 0.3 at specificity about 0.97 | **Clinically important limitation** | `condition_summary.csv`; F12, F14 |
| Mild vs Normal fails on Messidor | Qwen calls 71% of Mild eyes Normal; Mild-vs-Normal AUC 0.57 (Qwen) and 0.47 (RetiZero) | `mild_vs_normal.csv`; F13 |
| LoRA rank does not matter at lr 2e-4 | 30 of 30 runs work; every rank-vs-16 CI includes 0, with alpha 16 and with alpha = rank | `rank_lr2e-4_aggregate.csv`; F04 |
| lr 5e-4 is unstable | 10 of 63 fine-tuned runs fail, all at 5e-4; validation loss separates failed (0.0142–0.0199) from working (0.0069–0.0107) runs | `collapse_vs_validation.csv`; F05, F17 |
| Validation loss picks lr 2e-4 | 1e-4: 0.00791; 2e-4: 0.00768; 5e-4: 0.01051 | `lr_selection_by_validation_loss.csv`; F05 |
| The adapter matters (S7) | APTOS +31.6 pt [+25.4, +37.0]; Messidor-2 +11.5 pt [+7.4, +15.8] | `training_aggregate.csv`, `messidor2_aggregate.csv`; F06 |
| CLAHE gives no benefit to fine-tuned models | Every fine-tuned CI includes 0; it hurts zero-shot Qwen (−8.0 pt) | `clahe_aggregate.csv`; F07 |
| Most errors are adjacent grades (S13) | About 80% of errors are off by one, which is why QWK is high | `error_structure.csv`; F08, F09 |
| Near-duplicates did not inflate results (S10) | Removing pHash ≤ 2 near-duplicates moves accuracy by < 1 pt | `near_duplicate_sensitivity.csv`; F18 |
| Data integrity | 148 of 148 runs reproduce; archived Messidor inputs are pixel-identical to the raw rebuild; Qwen gives the same grade on all 1,057 shared images | `reproduction_check.csv`, `messidor_archived_vs_verified.csv`; F19 |
| Efficiency (S14) | Qwen: 8.82 B parameters (51.3 M trainable), 0.76–0.86 s/image, 8.7 GiB at inference. RetiZero: 304 M parameters, all trainable, 18 ms/image, 1.2 GiB | `efficiency_final.csv`, `timing_and_energy_final_02.csv`; F16 |

## 5. Data dictionary (the tables you will use most)

| File | What one row is |
|---|---|
| `per_run_metrics.csv` | One run on one cohort: accuracy, QWK, macro AUC, per-grade AUC, referable sensitivity, specificity and AUC, calibration, predicted-grade counts. Cohorts: `aptos361` (headline test set), `aptos367` (original split), `messidor` (archived 1,057), `messidor2` (verified 1,744). |
| `condition_summary.csv` | Mean, SD, min and max over seeds per condition and cohort |
| `*_aggregate.csv` | One paired comparison A − B with the hierarchical bootstrap CI and bootstrap p, for accuracy, QWK and macro AUC |
| `*_pairs.csv` | The same comparison per seed pair, with the exact McNemar p and its Holm adjustment |
| `classwise_*.csv` | Per-grade sensitivity, specificity, precision, F1 and AUC |
| `efficiency_final.csv` | Efficiency quantities, one per row, all measured on one RTX 5070 Ti 16 GB |
| `SUMMARY.md` | Human-readable digest of all tables |
| `figures/figure_index.csv` | Figure → reviewer point → source tables → draft caption |

Condition names read like `Qwen r16 CLAHE-on lr2e-4`. "alpha8" or "alpha32" is appended when alpha ≠ 16.

## 6. Known open items (check them and add to them)

**Writing and literature (no computation)**
1. **G1, G3, S1, S2:** novelty statement, a recent DR literature review (ML, DL, LLM and VLM studies on APTOS and Messidor-2) and a specific research gap. Nothing in the bundle covers these; they need a literature search. `docs/APTOS_DUPLICATES_LITERATURE.md` covers only the APTOS duplicate issue.
2. **S4:** justify the choice of a general VLM against a retina-specific model, and what the comparison tests.
3. **S5:** methodology and architecture diagrams for both models. These are hand-drawn schematics, not in the figure set.
4. **S3:** methods text on BGR/RGB/LAB. The evidence is in `REVISION_EVIDENCE.md` §2, colour conversions.
5. **Methods text:**
   - how the clean split arises (`docs/HEADLINE_VERIFICATION.md`)
   - the scorer
   - checkpoint selection: Qwen by validation loss, RetiZero by validation accuracy
   - preprocessing per model
   - the class balancing (F01B)
6. **Limitations:**
   - seed variance
   - lr 5e-4 instability
   - an FP16 overflow scored with a labelled FP32 fallback
   - an `effective_scaling` record error (alpha/r taken from `adapter_config.json`)
   - duplicates in the original split
   - low referable sensitivity out of domain
   - Mild/Normal domain shift
   - energy is whole-card power
   - single GPU

**Decisions (supervisor)**
7. Confirm the primary endpoint. Proposed: H1, Qwen lr 2e-4 vs RetiZero, native preprocessing, accuracy and QWK, on APTOS-361 and Messidor-2.
8. Whether to report the RetiZero "saved app checkpoint" at all. It is a single model with a different training history and transfers better than the three matched seeds (54.1% vs 40.2% on Messidor-2).
9. RetiZero's name in tables: remove "LoRA r8".

**Optional computation (CPU only, small)**
10. Referable-DR threshold recalibration on Messidor-2, if the reviewers want a clinical operating point. F12 shows there is room. Any threshold chosen on Messidor-2 itself is post hoc and must be labelled so.
11. Efficiency table: RetiZero's "training wall time, all attempts" reads the first attempt only. Fix the label or sum the attempts. Quote Qwen inference as the range 0.76–0.86 s, or name the measurement used.
12. Excel workbook of the tables for the supervisor.

**Optional, larger**
13. Grad-CAM or attention examples, and a misclassified-image figure (S11). The contact sheets exist outside the repository; images cannot be shared in the bundle.

**Housekeeping (authors)**
14. Back up `D:\Afterimage_Experiments` (no second copy exists).
15. Refresh `DELIVERABLES_CHECKLIST.md` and `REMAINING_WORK.md`, which predate 25 November.
16. Final table rebuild from a clean checkout.
17. Data and code availability statements; target journal format.

## 7. Output format

Return three Markdown documents:

1. `RESULTS_REPORT.md`, part A
2. `FINISH_PLAN.md`, part B: a table with columns item, reviewer point, owner type, effort, depends on, status
3. `RESPONSE_TABLE.md`, part C

In all three, mark any statement you could not trace to a file with **[UNVERIFIED]**.
