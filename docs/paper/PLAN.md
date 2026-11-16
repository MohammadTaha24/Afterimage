# Plan to finish the paper

Written 15 November 2026. Companion files: [DELIVERABLES_CHECKLIST.md](DELIVERABLES_CHECKLIST.md) (what must exist before submission) and [RETIZERO_TRAINING_BRIEF.md](RETIZERO_TRAINING_BRIEF.md) (what the RetiZero teammate does).

## 1. Where we stand

- All Qwen training asked for in Dr. Noor's plan is finished: rank 8/16/32, CLAHE on/off, the BF16 base control, 24 runs on the clean 361-image test set, plus 6 original-split runs and the Retraining 2 replay.
- The tables are built and reproducible from the saved predictions ([SUMMARY.md](../../finetuning/paper_tables/output/SUMMARY.md)).
- Missing evidence: every RetiZero number, a verified Messidor cohort, a proper inference benchmark, parameter counts, exported misclassified images, and the with/without QLoRA baseline.
- The results raise three issues the paper has to face: CLAHE does not help Qwen, 4 of 18 clean 4-bit runs collapse to a single grade, and rank 16 is not clearly better than 8 or 32.
- No manuscript text has been revised yet.

## 2. Self-grill: the questions and my answers

I ran the grill-me interview against this plan and answered each question from the files where I could. Answers marked **your call** or **Dr. Noor's call** are recommendations that need confirming.

```
Design tree
├── [✓] Framing
│   ├── [✓] Q1  New paper or revision
│   └── [✓] Q2  Headline result set                      (Dr. Noor's call)
├── [✓] Efficiency and utilization
│   ├── [✓] Q3  Redo all training?                        (your call)
│   ├── [✓] Q4  What the existing logs lack
│   ├── [✓] Q5  What to build
│   ├── [✓] Q6  How to measure without touching pinned code
│   └── [✓] Q7  Same hardware for both models
├── [✓] RetiZero handover
│   ├── [✓] Q8  Can the teammate run our script as it is
│   ├── [✓] Q9  Which checkpoint
│   ├── [✓] Q10 Protocol and fairness
│   ├── [✓] Q11 RetiZero with CLAHE                       (your call)
│   └── [✓] Q12 What comes back and how we accept it
├── [✓] Remaining Qwen evidence
│   ├── [✓] Q13 What "without QLoRA" means
│   ├── [✓] Q14 Collapsed runs and tuning evidence        (Dr. Noor's call)
│   ├── [✓] Q15 Rank and alpha confound                   (Dr. Noor's call)
│   └── [✓] Q16 CLAHE position                            (Dr. Noor's call)
├── [✓] External evaluation
│   └── [✓] Q17 Messidor cohort                           (your call: who has the raw data)
├── [✓] Analysis
│   ├── [✓] Q18 Qwen versus RetiZero statistics
│   └── [✓] Q19 Old Grade 0 fallback
└── [✓] Delivery
    ├── [✓] Q20 Safety of the evidence
    ├── [✓] Q21 Reporting to Dr. Noor and the deadline    (your call: the date)
    └── [✓] Q22 Critical path
```

**Q1. Is this a revision or a new paper?**
A new journal submission (Dr. Noor's last email). There is no response letter to write, but the 17 reviewer comments are the acceptance test, so the checklist follows them.

**Q2. Which results are the headline?**
The clean 361-image suite. It has a fixed duplicate-free split, three seeds per setting and one scorer, which is what Dr. Noor asked for. Retraining 2 stays as the historical reference with two disclosures: it is the best of three same-seed runs (77.38%, 80.65%, 77.11%), and 17 of its 367 test images are byte-identical to training images. Dr. Noor named Retraining 2 as the baseline before these facts were known, so this needs her agreement.

**Q3. Do we need to redo all the training to get GPU and computer utilization?**
I checked before answering. Every one of the 24 clean-suite runs and the 6 original-split runs already saved a log sampled every 5 seconds with GPU %, VRAM, CPU % and RAM, plus hardware specs, wall time and framework peak memory. So the utilization data largely exists.
Repeating everything would cost about 27 GPU-hours for the 24 runs (about 35 with the original-split runs), from the logged times. It would also create a second set of accuracy numbers: training is not repeatable run to run (the same seed gave 77%, 81% and 77%, and some runs collapse), so the repeats would disagree with the results already in the tables and we would have to explain which set counts.
My recommendation: do not repeat the runs. Summarize the logs we have, build the missing measurements, and rerun only where a gap cannot be filled otherwise. If you still want fresh runs, the plan supports it as an option (task 1.7) with rules that keep the accuracy story consistent. **Your call.**

**Q4. What do the existing logs not give us?**
- Per-image inference time. Today's figure is scoring time divided by 361, which includes loading the model and five forward passes per image.
- Reliable parameter counts. The logs report 8.82 billion total for 4-bit runs and 6.12 billion for BF16 runs of the same model.
- GPU power and energy. Not recorded.
- Process-level VRAM. The logged VRAM is the whole device, including about 1.3–1.5 GB used by the desktop.
- Full training time for three interrupted runs (rank 16 on / seed 42, rank 8 off / seed 42, BF16 seed 43). Only their final attempt is counted.
- Anything for RetiZero.

**Q5. So what do we build?**
A small `finetuning/efficiency/` package (workstream 1): a portable resource monitor, a telemetry summarizer that reads the existing logs, an independent parameter counter and an inference benchmark, with the results flowing into the paper tables.

**Q6. How do we measure without breaking the pinned code?**
The trainer, `paper_reproduction.py`, `evaluate_clahe_bf16.py` and `train_bf16_control.py` are pinned by hash in finished runs and must not be edited. The monitor therefore runs outside the training process and wraps any command. The benchmark is a new script that imports the pinned scorer without changing it.

**Q7. How do we get a same-hardware comparison if RetiZero is trained elsewhere?**
Dr. Noor asked for the same hardware for inference where possible. The teammate returns the three trained checkpoints, and the inference benchmark for both models runs on this PC (RTX 5070 Ti). Training time and memory are reported with each machine's hardware stated. If the teammate's GPU differs, one RetiZero seed can also be retrained here for a like-for-like training time; RetiZero is small, so this is cheap.

**Q8. Can the teammate run our RetiZero script as it is?**
Only on a Windows machine with the same `D:` paths. I read `train_retizero_control.py` (since moved to `finetuning/RetiZero/paper_revision/train_retizero.py`): it verifies every image by SHA-256 and the dataset CSVs store absolute `D:\Afterimage_Experiments\...` paths. On Linux or Colab it fails. So a transfer package is needed, and a relocatable variant if they have no `D:` drive (task 2.2). The script has also never run end to end, so the first seed is a test of the script as much as of the model.

**Q9. Which checkpoint?**
The official pretrained RetiZero from the authors' repository, with source, date, size and SHA-256 recorded. The script rejects a checkpoint that already contains a DR classifier, which protects against using the app's fine-tuned weights by mistake.

**Q10. Is the RetiZero protocol fair against Qwen?**
Same split and the same 361 test IDs, three seeds (42, 43, 44), no test-set tuning. Two differences to disclose: RetiZero picks its checkpoint by validation accuracy while Qwen uses validation loss, and RetiZero's learning rate came from the earlier hyperparameter search while Qwen's was inherited. RetiZero uses image augmentation on training images only; Qwen uses none, only class-balanced sampling.

**Q11. Should RetiZero also be run with CLAHE?**
Reviewer S9 asks why CLAHE was applied to one model only. No RetiZero CLAHE result exists. Three more seeds on a CLAHE-on 224-pixel build would answer with evidence and cost little. I list it as optional, after the main three seeds. **Your call.**

**Q12. What comes back, and how do we accept it?**
Per seed: configuration, training result, selected checkpoint, predictions, summary, console logs and monitor CSVs; plus hardware, package list and timings. Acceptance: 361 IDs in our order, probabilities sum to 1, metrics recompute, configurations identical except the seed. The tables builder will be extended to enforce this.

**Q13. What does "with and without QLoRA" (S7) mean here?**
Two things, and the paper should report both. Without quantization: LoRA on the BF16 base, which is done. Without the adapter: the base model with no fine-tuning, scored by the same five-answer scorer on the 361 images. That second one does not exist yet and is scoring only, a few minutes per base model.

**Q14. Can we publish with 4 of 18 runs collapsed?**
We must report them, and a reviewer who already asked for tuning trials (G2, S8) will ask why. The collapses come with a learning rate of 5e-4 and early stopping on validation loss with patience 2, neither of which was ever varied. A small pre-declared study would both answer the reviewer and probably explain the collapses: rank 16, CLAHE off, 4-bit, learning rates 1e-4 and 2e-4, three seeds each. That is 6 runs, about 7–8 GPU-hours. The alternative is to state that the settings were inherited and report the instability as a finding. **Dr. Noor's call.**

**Q15. Rank changes while alpha stays at 16. Does the rank study answer the reviewer?**
Only partly: alpha/r is 2, 1 and 0.5 across ranks 8, 16 and 32, so rank and scaling move together. Either say so as a limitation, or add rank 8 and rank 32 with alpha equal to rank (6 runs, about 6 GPU-hours). I recommend the limitation unless the Q14 runs are approved, in which case run both batches together. **Dr. Noor's call.**

**Q16. What do we say about CLAHE?**
On the clean suite at rank 16, CLAHE off averages 73.96% against 69.71% with CLAHE on, with far less spread; the original-split pairs point the same way. The honest position is that CLAHE gave no measurable benefit for Qwen. If the headline configuration becomes CLAHE off, the BF16 control exists only with CLAHE on, and three BF16 CLAHE-off runs (about 3 GPU-hours) would be needed to keep that comparison matched. **Dr. Noor's call.**

**Q17. What is needed for Messidor?**
No raw Messidor data is on the D: drive; I checked. The archived cohort has 1,057 images while Messidor-2 has about 1,748, and how the 1,057 were chosen is not documented. Needed: the raw images, the grade source, the eye pairing, and a decision on the four duplicate pairs that carry different patient IDs. Until then the external numbers can only be described as an archived export. **Your call: who on the team holds the raw data.**

**Q18. How do we compare Qwen and RetiZero?**
On the same 361 images, with a paired test (paired bootstrap of the metric difference, and McNemar on correct/incorrect), per seed and on the three-seed average. This goes into the tables builder once RetiZero predictions arrive.

**Q19. Did the old Grade 0 fallback affect the earlier paper's numbers?**
The old generation-based scripts do assign Grade 0 to unparseable output. Retraining 2 and every current result use the likelihood scorer and are unaffected. The earlier paper's numbers cannot be checked because no raw outputs were saved. One of the old scripts already counts bad parses and reports a strict accuracy, so running it once on the original paper adapter gives a direct answer, if that adapter is still available.

**Q20. Is the evidence safe?**
No. The 24.7 GB of runs and the artifacts folder are outside git and have no second copy. One disk failure loses the paper's results. A backup comes first in workstream 7.

**Q21. What does Dr. Noor expect, and by when?**
She asked for results as each experiment completes and none have been sent since the rank and CLAHE runs finished. She is leaving the university in December and asked to continue on her Gmail address. An update with the tables and the four decisions above should go out now. I have assumed the experimental evidence should be complete by 29 November; **your call on the real date.**

**Q22. What is the critical path?**
Pretrained checkpoint → transfer package → RetiZero runs → benchmark → comparison tables. Everything else (efficiency tooling, zero-shot baseline, Messidor sourcing, manuscript writing) runs alongside it. Extra Qwen training waits for Dr. Noor's decisions.

## 3. Workstreams

Effort figures are estimates. GPU-hours come from the logged run times.

### Workstream 0. Decisions and communication

| # | Task | Who | Done when |
| --- | --- | --- | --- |
| 0.1 | Send Dr. Noor a results update: the summary tables, the three findings (CLAHE, collapsed runs, rank), and the decisions in Q2, Q14, Q15 and Q16 | You (I can draft it) | Sent to her Gmail |
| 0.2 | Decide the efficiency scope (Q3) and RetiZero CLAHE (Q11) | You | Recorded in the checklist |
| 0.3 | Confirm the RetiZero teammate, their machine and a return date | You | Step 1 of the brief answered |
| 0.4 | Find who holds raw Messidor-2 images, grades and eye pairing | You | Files located |
| 0.5 | Confirm the ICCA presenter to Dr. Noor | You | Reply sent |

### Workstream 1. Efficiency and utilization (the part to build)

New folder `finetuning/efficiency/`. Nothing here edits pinned files.

| # | Task | Who | Done when |
| --- | --- | --- | --- |
| 1.1 | **Telemetry summarizer.** Read every attempt's `utilization.csv`, `run_report` and result JSONs for all 30 finished runs. Per run and per condition: total wall time across all attempts, epochs and time per epoch, mean and 95th-percentile GPU %, peak VRAM and peak minus the idle baseline, mean CPU %, peak RAM, adapter size on disk. | Claude | Table matches `run_report.md` for uninterrupted runs; the three interrupted runs show summed time |
| 1.2 | **Portable resource monitor.** One script that wraps any command and samples GPU %, VRAM (device and per process), GPU power and temperature, CPU % and RAM, and writes the same CSV layout plus the new columns. Works on Windows and Linux with only `nvidia-smi` and `psutil`. | Claude | Unit-tested on a short command; pilot run produces a CSV the summarizer reads |
| 1.3 | **Parameter counter.** Count total and trainable parameters directly from the loaded models: Qwen 4-bit and BF16 at ranks 8, 16 and 32, and RetiZero. Explain the 8.82 vs 6.12 billion log discrepancy. | Claude | One table, with the method stated |
| 1.4 | **Inference benchmark.** Same PC, both models, fixed sample of the 361 test images, batch 1, after warm-up: median and 95th-percentile time per image, images per second, peak VRAM, model load time. Qwen measured two ways: the paper's five-answer scorer and single-pass generation as the app does it. | Claude builds; you run it (GPU, about 30 minutes) | Results file per model; repeat run within a few percent |
| 1.5 | **Efficiency tables.** Extend `build_paper_tables.py` with the summarizer, counter and benchmark outputs, for both models, with hardware stated per row. | Claude | Table rebuilds from files only |
| 1.6 | **Targeted reruns (only if needed).** If a single uninterrupted training time is wanted for the three interrupted runs, rerun those three under the monitor (about 4–5 GPU-hours) and report them as timing runs. | You run | Decided in 0.2 |
| 1.7 | **Full rerun (only if you decide to).** All 24 clean-suite runs again under the monitor, about 27 GPU-hours, with a new tag so nothing is overwritten. Rule: the new runs are reported as additional seeds-of-record or as timing-only runs, stated in advance, never swapped in selectively. | You run | Decided in 0.2 |

### Workstream 2. RetiZero handover

| # | Task | Who | Done when |
| --- | --- | --- | --- |
| 2.1 | Teammate downloads the official checkpoint and reports its SHA-256 and their hardware | Teammate | Brief step 1 answered |
| 2.2 | **Transfer package.** Repository snapshot, 224-pixel dataset, Qwen split files, monitor script, and a one-page run sheet. Relocatable-path variant if they have no `D:` drive, without changing the protocol. Dry-run the `check` action here against the package. | Claude | `check` prints the matched 2786/357/361 counts from the package |
| 2.3 | Send the brief and the package | You | Sent |
| 2.4 | Seed 42 trained and scored; `train_result.json` reviewed before the other seeds | Teammate, then Claude | First result accepted |
| 2.5 | Seeds 43 and 44 | Teammate | Three run folders returned |
| 2.6 | **Acceptance.** Extend the tables builder to read RetiZero runs and apply the same cross-checks | Claude | RetiZero rows appear in the tables |
| 2.7 | RetiZero inference benchmark on this PC (task 1.4) | You run | Row in the efficiency table |
| 2.8 | Optional: RetiZero with CLAHE (build a 224-pixel CLAHE-on dataset, three seeds) | Claude builds data; teammate trains | Decided in 0.2 |

### Workstream 3. Remaining Qwen evidence

| # | Task | Who | GPU time | Done when |
| --- | --- | --- | --- | --- |
| 3.1 | **Zero-shot baseline (S7).** New scoring route for the base model with no adapter, same scorer and 361 images, 4-bit and BF16 | Claude builds; you run | under 30 min | Two rows in the tables |
| 3.2 | **Invalid-output count (Q19).** Run the old generation-based evaluation with parse logging on the original paper adapter | Claude prepares; you run | under 30 min | Bad-parse count and strict accuracy recorded, or "adapter unavailable" stated |
| 3.3 | **Stability and tuning study (Q14)**: rank 16, CLAHE off, learning rates 1e-4 and 2e-4, three seeds | You run | about 7–8 h | Only if Dr. Noor approves |
| 3.4 | **Fixed alpha/r (Q15)**: ranks 8 and 32 with alpha equal to rank, three seeds | You run | about 6 h | Only if Dr. Noor approves |
| 3.5 | **BF16 base with CLAHE off (Q16)**, three seeds | You run | about 3 h | Only if CLAHE off becomes the headline |

Tasks 3.3–3.5 need new launch routes in the 021 notebook, built before they run.

### Workstream 4. Messidor

| # | Task | Who | Done when |
| --- | --- | --- | --- |
| 4.1 | Copy the raw images, grades and pairing file to D: and record their source and hashes | You | Files on D: |
| 4.2 | Document how the archived 1,057 images relate to Messidor-2 and resolve the four duplicate pairs | Claude | Written note with counts |
| 4.3 | Build the verified cohort at 512 pixels (CLAHE on and off) and 224 pixels | Claude | Manifest with hashes |
| 4.4 | Score the headline Qwen runs and the RetiZero checkpoints. The BF16-base route needs the CPU offload wired into the external scorer first. | You run | Predictions for each model |
| 4.5 | Mild-versus-Normal analysis and tables on the verified cohort | Claude | Tables rebuilt |

### Workstream 5. Analysis and tables

| # | Task | Who |
| --- | --- | --- |
| 5.1 | Misclassified-image lists for the clean-suite headline runs | Claude |
| 5.2 | Export representative misclassified images with true and predicted labels into one figure folder (S11) | Claude |
| 5.3 | Adjacent-grade versus distant-grade error table (S13) | Claude |
| 5.4 | Class counts per split and the augmentation description for both models (S9) | Claude |
| 5.5 | Paired Qwen versus RetiZero comparison (Q18) | Claude, after 2.5 |
| 5.6 | Calibration analysis of the headline models | Claude |
| 5.7 | Near-duplicate check across splits | Claude |
| 5.8 | Excel workbook of all tables, built in a separate environment on D: so the pinned environment is untouched | Claude |

### Workstream 6. Manuscript

| # | Task | Who |
| --- | --- | --- |
| 6.1 | Methods rewritten from the code: data and split, preprocessing and colour spaces (S3), both training protocols, the scorer, checkpoint selection | Draft by Claude from the code; you and Dr. Noor edit |
| 6.2 | Methodology flow diagram and two architecture diagrams (S5) | Claude drafts; you finalize |
| 6.3 | Results: rank, CLAHE, quantization and no-adapter ablations, model comparison, efficiency, external evaluation | After workstreams 1–5 |
| 6.4 | Discussion: misclassification causes (S11), accuracy in context (S12), QWK (S13), limitations | You and Dr. Noor |
| 6.5 | Literature review and comparison table for APTOS and Messidor-2 (G3, S1, S10), novelty and research gap (G1, S2), model rationale (S4) | Dr. Noor said she would lead the writing; confirm the split |
| 6.6 | Number audit: every figure in the text traced to a table file | Claude |

### Workstream 7. Reproducibility and housekeeping

| # | Task | Who |
| --- | --- | --- |
| 7.1 | **Back up** `D:\Afterimage_Experiments\runs` and `finetuning/Qwen3/artifacts` to a second disk or cloud storage. Do this first. | You |
| 7.2 | Commit the restructure locally | Claude, on your word |
| 7.3 | Remove the C: read dependency for new 4-bit launches (`LEGACY` in `remaining_runs.py`); needed before any task in 3.3–3.5 | Claude |
| 7.4 | Delete the three pilot test folders | Claude, on your word |
| 7.5 | Refresh or retire the stale checklists | Claude |
| 7.6 | Push to the shared repository | When you approve |

## 4. Order of work

Weekends, since that is when you work on this. Dates assume the 29 November target in Q21.

| When | Work |
| --- | --- |
| Now | 7.1 backup. 0.1 update to Dr. Noor. 0.3 and 2.1: teammate fetches the checkpoint. 0.4: locate Messidor. |
| 21–22 Nov | Claude builds workstream 1 (1.1–1.5), the transfer package (2.2) and the zero-shot route (3.1). You run the benchmark and 3.1, and send the package. |
| During the week | Teammate trains RetiZero. Dr. Noor answers the decisions. |
| 28–29 Nov | RetiZero accepted (2.6), benchmarked here (2.7), comparison tables (5.5). Approved extra Qwen runs (3.3–3.5) start. Messidor cohort (4.2–4.3) if the raw data has arrived. |
| 5–6 Dec | Messidor scoring and analysis (4.4–4.5). Analysis tasks 5.1–5.8. Diagrams. |
| 12–13 Dec | Results and methods text, number audit, final table rebuild. |

## 5. Things you did not mention that are in this plan

- The evidence has no backup (7.1).
- The with/without QLoRA comparison still lacks its no-adapter side (3.1).
- Four of 18 clean runs collapsed and nothing explains why yet (Q14).
- The rank study changes two things at once (Q15).
- CLAHE shows no benefit, which contradicts the design the first paper presented (Q16).
- Retraining 2, the agreed baseline, is the best of three runs on a split with leaked test images (Q2).
- The teammate cannot run the RetiZero script unchanged on a different machine, and the script has never completed a run (Q8).
- Parameter counts in the logs are inconsistent (1.3).
- Inference time has never been measured properly (1.4).
- Representative misclassified images, class counts and the QWK explanation table are still to make (5.2–5.4).
- Dr. Noor is waiting for results and leaves in December (Q21).
- The ICCA presenter is still unconfirmed (0.5).
- New 4-bit runs still read from the C: copy (7.3).

## 6. Open questions

For you:
1. Efficiency: fill the gaps only (recommended), or repeat all training?
2. Who is the RetiZero teammate, what machine do they have, and when can they deliver?
3. RetiZero with CLAHE: yes or no?
4. Who has the raw Messidor-2 data?
5. What is the real deadline?

For Dr. Noor:
1. Clean 361-image suite as the headline, Retraining 2 as historical reference?
2. Run the stability and tuning study (3.3), or report the instability as it is?
3. Run fixed alpha/r (3.4), or state the limitation?
4. CLAHE off as the headline configuration?
5. Who writes which manuscript sections?
