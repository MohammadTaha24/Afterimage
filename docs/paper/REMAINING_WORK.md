# Remaining work

Updated 25 November 2026. This is the one list of what is left before resubmission.

- Background for ChatGPT: [HANDOFF_PROGRESS_FOR_CHATGPT.md](HANDOFF_PROGRESS_FOR_CHATGPT.md)
- Figures: [HANDOFF_FIGURES_GUIDE.md](HANDOFF_FIGURES_GUIDE.md)
- Report request: [HANDOFF_REPORT_WRITER.md](HANDOFF_REPORT_WRITER.md)
- Evidence: [REVISION_EVIDENCE.md](REVISION_EVIDENCE.md)

## Where things stand

- **Experiments: finished. No GPU work is left for the paper.** Everything below is writing, decisions, small CPU tasks or housekeeping.
- **Qwen:** `final_01`, the lr 2e-4 grid `final_02` (27 adapters) and the 24 earlier runs.
- **RetiZero:** `retizero_01`, 18 jobs from the official pretrained checkpoint.
- **Messidor-2:** every model scored on verified Messidor-2 (1,744 images) in `retizero_02` and `final_02`.
- **Timing:** done for all Qwen configurations.
- **Analysis tables:** `finetuning/paper_tables/output/revision_analysis/`, 148 runs, all reproducing their reference.
- **Figures:** 22, in `finetuning/paper_figures/`. F20–F22 are training curves drawn from the run logs.
- **Results package:** `D:\Afterimage_data_share\afterimage_results_20261128\` and its zip. It holds every run's logs, training histories, predictions and summaries, the analysis code, tables, figures, documents and the notebook, without weights or images. Run inside the package, the notebook rebuilds every table from the logs, checks them against the shipped tables, then draws all figures. It is built by `finetuning/paper_figures/build_results_package.py`.
- **Git:** everything is committed and pushed to GitHub branch `afterimage-re`, including the figure notebook and its output, the package builder and the handoffs.

## Done (for reference)

- [x] Clean APTOS split 2,786 / 357 / 361 verified (`HEADLINE_VERIFICATION.md`).
- [x] Qwen at lr 5e-4: ranks 8, 16 and 32 × CLAHE on/off, BF16 controls, previous-split control; 3 seeds each.
- [x] Qwen at lr 1e-4 and 2e-4; zero-shot 4-bit and BF16; original-paper adapter audit; FP32 fallback for one FP16 overflow (`final_01`).
- [x] Qwen rank and alpha grid at lr 2e-4: 27 adapters, CLAHE on and off, all working (`final_02`).
- [x] RetiZero from the official pretrained checkpoint: replay, original split, clean split, CLAHE, archived Messidor (`retizero_01`).
- [x] Verified Messidor-2 obtained and validated (`D:\Afterimage_data_share\Messidor-2\`). Cohort built from the raw photos; pixel-identical to the archived export on the 1,057 shared images.
- [x] Every model scored on verified Messidor-2; warm timing and energy for 7 Qwen configurations (revision GPU queue, 28 jobs).
- [x] CPU analyses for every reviewer point: hierarchical bootstrap and McNemar/Holm, calibration, class-wise metrics, error distance, Mild vs Normal, near-duplicates, colour-conversion pixel check, Messidor label audit, lr chosen on validation loss, failed-run detection.
- [x] Evidence note with one section per reviewer point (`REVISION_EVIDENCE.md`).
- [x] 22 figures with draft captions, and handoffs for ChatGPT.
- [x] Results package with all run records, built and checked: the notebook inside it rebuilds every table from the logs and they match.
- [x] Figure notebook run in the repo (22 figures, no overlapping text; report bundle `paper_report_bundle_20261125.zip` made) and everything committed and pushed.

## Left to do

Owner types: **You** = the paper owner; **Noor** = supervisor decision; **Sinan** = colleague (RetiZero side); **GPT** = ChatGPT with the bundle.

### 1. Next steps (this week)

- [ ] **Look through the figures** in `finetuning/paper_figures/output/`. The notebook was run on 25 November: 25 figures (F23–F25 are the methodology and architecture diagrams), and the layout check found no overlapping text. A person still needs to look at each figure. (You)
- [ ] **Share the results package with the team.** The folder `D:\Afterimage_data_share\afterimage_results_20261128\` is ready, unzipped. It starts with `Afterimage_revision_report.pdf` (results, reviewer-comment status, open points), `Afterimage_manuscript_material.pdf` (draft text for every open reviewer comment) and `README.txt`. Editable copies of both documents are in `D:\Afterimage_data_share\` as .docx. It has no Markdown files and no internal reviewer codes. Zip it when you decide (`build_results_package.py --zip`) and share it with named people only, because it holds APTOS and Messidor-2 labels. (You)
- [ ] **Give ChatGPT the bundle.** The notebook's last cell makes `D:\Afterimage_data_share\paper_report_bundle_<date>.zip`. Add `EA_REVIEW.docx` and these handoffs. Ask for:
  - the results report
  - the finish plan
  - the response-to-reviewers table
  
  See `HANDOFF_REPORT_WRITER.md`. (You → GPT)
- [ ] **Send Sinan the RetiZero results** (Messidor-2 scores, figures) and the message about the runner fixes. The fixes are already done and merged. (You)
- [ ] **Send Dr. Noor a results update.** `RESULTS_UPDATE_DRAFT.md` predates RetiZero, the lr 2e-4 grid and Messidor-2; rewrite it from `REVISION_EVIDENCE.md`, or from GPT's report. (You)

### 2. Decisions (Dr. Noor)

- [ ] **Primary comparison and endpoint.** Proposed: H1, Qwen lr 2e-4 against RetiZero, native preprocessing, on accuracy and QWK, reported on APTOS-361 and verified Messidor-2.
- [ ] **Headline Qwen learning rate.** Proposed: 2e-4, chosen on validation loss, with 5e-4 reported as the original setting.
- [ ] **The RetiZero saved app checkpoint.** It has a different training history and transfers better than the matched seeds (54.1% against 40.2% on Messidor-2). Report it separately, or leave it out?
- [ ] **Clinical framing of the low referable sensitivity** on Messidor-2 (about 0.3). Report as a limitation only, or add a post hoc threshold analysis? (See 4.)
- [ ] **Target journal and format.**

### 3. Manuscript writing (You, with GPT drafting)

Per reviewer point; `REVIEWER_CHECKLIST.md` has the full wording.

Drafts for every point below are in `Afterimage_manuscript_material` (.pdf in the package, .docx in `D:\Afterimage_data_share\`). What is left is editing them into the manuscript.

- [ ] **G1, S2:** novelty, research gap and contributions. Draft ready (material §1–2).
- [ ] **G3, S1:** literature. A verified starting set of 13 studies with a related-work paragraph (material §1). Four details are marked [check]; widen the search if the reviewers expect a systematic review.
- [ ] **S4:** why these two models. Draft rationale (material §3); add the authors' own reasons.
- [x] **S5:** methodology and both architecture diagrams: F23, F24, F25, drawn by the figure notebook from the code and run records.
- [ ] **S3:** BGR/RGB/LAB text in §3.2 (evidence: `REVISION_EVIDENCE.md` §2; pixel check).
- [ ] **S6, S8, G2:** parameter justification. Table of every setting with value and source, tested or inherited (material §4). Optional: dropout ablation, about 8 GPU hours.
- [ ] **S7:** adapter ablation (F06).
- [ ] **S9:** CLAHE and augmentation, with class counts (F01, F07).
- [ ] **S10, S12:** fair comparison and external results (F02, F12, F18, F19). Draft comparison with published external results (material §6).
- [ ] **S11, S13:** misclassification and error structure (F08–F11, F13, F14). New image-quality analysis, `finetuning/paper_tables/build_error_analysis.py` → `output/error_analysis/`; draft paragraph (material §5). Optional: grader review of misclassified Messidor-2 images.
- [ ] **S14:** efficiency (F16), with an estimate of compute per image (material §7).
- [ ] **Methods:**
  - how the 361 clean images arise
  - the five-answer likelihood scorer
  - checkpoint selection: Qwen by validation loss, RetiZero by validation accuracy
  - how lr 2e-4 was chosen: on validation loss only, but the test scores of all three rates already existed, so do not claim the test set was unseen
  - preprocessing per model
  - class balancing
  - statistics
- [ ] **Limitations:**
  - seed variance and lr 5e-4 instability
  - the FP16 overflow scored with an FP32 fallback
  - the `effective_scaling` record error (alpha/r read from `adapter_config.json`)
  - duplicates in the original split
  - low referable sensitivity and Mild/Normal failure out of domain
  - energy is whole-card power
  - one GPU
- [ ] **Response-to-reviewers letter** (GPT drafts the table; you finalise).
- [ ] **Number audit:** every number in the text traced to a file under `finetuning/paper_tables/output/` or `finetuning/paper_figures/output/`.
- [ ] **Data and code availability statements.**

### 4. Small CPU tasks (optional; ask Claude or Sinan)

- [ ] **Internal codes in comparison names** (Sinan's `build_revision_analysis.py`): the tables name comparisons "H1 proposed primary: …", "H2 …" and "S7 …", and `SUMMARY.md` headings carry reviewer codes such as "(G2, S8)". The figures notebook strips them; the tables still have them.
- [ ] **Efficiency table fixes:**
  - quote Qwen inference as 0.76–0.86 s per image
  - remove "LoRA r8" from RetiZero's name in tables and figures (it trains all weights)
- [ ] **Referable-DR threshold analysis** on Messidor-2 (F12 shows there is room). Label any threshold chosen on Messidor-2 as post hoc.
- [ ] **Excel workbook** of the tables for Dr. Noor (`build_results_workbook.py` exists; update it for the revision tables).
- [ ] **Collapsed column:** `SUMMARY.md` and the `collapsed` CSV column use "≤ 2 grades"; the paper uses "≤ 3". Align them, or footnote the difference (it affects one run).

### 5. Optional, larger (only if the reviewers or Noor want them)

- [ ] **Grad-CAM or attention maps, and a misclassified-image figure** (S11). The contact sheets exist in `image_audits\` outside the repo; images cannot go to ChatGPT.
- [ ] **RetiZero checkpoint exported in the app's format** (Sinan).
- [ ] **Fresh historical training repeats.** Deferred; only needed if the paper claims a fresh reproduction of the original training.

### 6. Housekeeping

- [ ] **Back up `D:\Afterimage_Experiments`** (`runs`, `final_runs`, `retizero_runs`, `datasets`, `assets`) and `D:\Afterimage_data_share\Messidor-2`. No second copy exists.
- [ ] **Bring GitHub `main` up to date,** or decide that `afterimage-re` is the shared branch.
- [ ] **Remove the read dependency on the C: legacy copy** for 4-bit launches (only matters if more training is ever run).
- [ ] **Refresh or archive stale documents:**
  - `DELIVERABLES_CHECKLIST.md`
  - `PAPER_TRAINING_CHECKLIST.md`
  - `RESULTS_UPDATE_DRAFT.md`
  - `finetuning/RetiZero/README.md`
- [ ] **Delete leftover folders:**
  - pilot folders (`final_*__pilot_*`, `retizero_*__pilot_*`), once nothing needs them
  - `.verification\review*` worktree folders
- [ ] **Final table and figure rebuild** from a clean checkout before submission.

## Dependencies at a glance

| Block | Needs | GPU |
|---|---|---|
| 1. Next steps | Nothing | None |
| 2. Decisions | Results update to Noor | None |
| 3. Writing | 2 (primary endpoint); literature search for G1, G3, S1, S2 | None |
| 4. Small CPU tasks | Nothing | None |
| 5. Optional | Reviewer or Noor request | Grad-CAM: a little |
| 6. Housekeeping | Nothing | None |
