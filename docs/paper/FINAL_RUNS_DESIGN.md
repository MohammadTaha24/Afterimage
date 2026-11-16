# Final runs: gap review, self-grill and design

Written 15 November 2026, before building `finetuning/Qwen3/notebooks/03_final_runs.ipynb`. It records a second pass through Dr. Noor's emails and the EA review, what that pass found, and the decisions behind the final-runs notebook. It extends [PLAN.md](PLAN.md); where the two differ, this file is the newer one.

## 1. Second pass through the sources

Each point was followed down to the evidence on disk, not only matched against the checklist.

### Dr. Noor's plan

**Item 1, rank sensitivity ("is rank 16 justified?").**
The rank runs exist, but the per-epoch training logs show the comparison is dominated by training failures, not by rank.
- Five runs never learn at all: their training loss sits at 0.020 from epoch 2 onward, the level of a model that ignores the image (rank 8 on seeds 42 and 43, rank 8 off seed 44, rank 32 on seed 44, rank 32 off seed 43).
- Several good runs show sudden loss spikes mid-training (0.75, 1.39, 0.59, 0.20).
- The learning rate never decays in practice. The schedule is linear over 50 epochs, but early stopping ends every run after 3–9 epochs, so training runs at about 5e-4 throughout.
- Most adapters are the epoch 1 or epoch 2 checkpoint, chosen on validation-loss differences of about 0.001.

Conclusion: until training is stable, no ablation (rank, CLAHE, quantization) can be read cleanly. A learning-rate study is the most valuable new training, and it also answers the reviewer's request for tuning trials (G2, S8). If a lower learning rate is adopted, the rank and CLAHE ablations would need repeating at that rate. That is the real case for redoing training, and it is a decision to take after the study, with Dr. Noor.

**Item 2, CLAHE.**
- I previously summarised this as "CLAHE does not help". That is too strong. The rank 16 difference (73.96% off, 69.71% on) comes from one unstable CLAHE-on seed; without it, CLAHE on averages 77.3%. The supportable statement is that no CLAHE effect is detectable within seed variance.
- "Verify the colour conversions" has a specific origin. Reviewer S3 noticed that the paper says images are converted to RGB in step 1 and then "from BGR to LAB" in step 5. If the code really called a BGR conversion on RGB data, the lightness channel would be wrong. The code is correct on both routes and reproduces the archived images pixel for pixel (3,639 of 3,662 identical, 23 within one level). What is missing is a committed test that shows this, and corrected wording.
- No RetiZero CLAHE result exists.

**Item 3, efficiency.**
- "Inference time per image" has two meanings for Qwen: the paper's scorer (five forward passes) and the app's generation (one pass). Both should be reported.
- The reviewer's phrase is "computational complexity". No FLOP counter is installed and none should be added to the pinned environment; input tokens per image and forward passes per prediction are a workable proxy.
- Exact parameter counts can be read from the weight files without loading a model.

**Item 4, evaluation.**
- Grade 0 fallback: your own email to Dr. Noor promised a follow-up on whether this affected the historical results. The original paper adapter still exists in the C: copy, so it can be scored once with generation and the invalid outputs counted. The same measurement on the new adapters shows how often the deployed decoding fails today.
- Mild as Normal on Messidor: the archived Messidor images are on D: (`D:\recovered\...`). The new adapters can be scored on that archived cohort now, which gives generalisation numbers for the new runs while the verified cohort is still being assembled. On APTOS the same models send about 9% of Mild images to Normal; on archived Messidor it is 89%, which points to a dataset shift, not a general inability.
- Splits: APTOS has no patient identifiers, so both eyes of one patient may sit in different splits. This cannot be checked; only image-level duplicates can. It belongs in the limitations.

**Other lines in the emails.**
- "Avoid unnecessary retraining": consistent with not repeating runs only for utilization.
- "Improve performance, generalisation and computational efficiency": the learning-rate study is the only planned work that could improve performance; Messidor scoring covers generalisation.
- "Share the results as each experiment is completed": nothing has been sent since the runs finished.
- She estimated one week. She leaves the university in December.
- Colab Pro was offered; it is an option for the RetiZero teammate.

### The EA review

- **G2** covers every parameter in the manuscript, including preprocessing values (CLAHE clip limit 2.0, 8×8 tiles, 512 pixels). Those need a citation or a trial.
- **S6** asks about alpha, dropout and bias as well as rank. Only rank has trials.
- **S7** most naturally means fine-tuned versus not fine-tuned. The zero-shot baseline is therefore the direct answer; the BF16 control answers a different question (quantized versus not).
- **S9** "both datasets" means APTOS and Messidor-2, not both models. It needs per-class counts for each dataset and split.
- **S10** "a fair comparison among all methods" could also be met with a conventional CNN baseline trained on the same split. This is cheap and optional; it is a question for Dr. Noor.
- **S12** (low accuracy): the duplicate audit found 96 duplicate rows and 62 rows whose duplicates carry conflicting grades. That is direct evidence of label noise in APTOS and helps explain the accuracy ceiling.
- **S5** diagrams must show the five-answer likelihood scorer, which replaced text generation as the evaluation method.

### Constraints found in the code

- The pinned launcher (`qwen_pilot.py`) takes its training settings from a saved configuration. Learning rate and early-stopping patience are read from that configuration, so they can be varied without touching pinned code.
- The pinned worker's adapter check hard-codes alpha 16 and dropout 0.05. Alpha and dropout variants need a variant-aware check, added by a wrapper script, not by editing the pinned file.
- The BF16 worker accepts only a rank 16, CLAHE-on reference run and only writes under `runs\`. BF16 variants (CLAHE off, other learning rates) need a new worker and are **not built**.
- New 4-bit launches still read their reference configuration from the C: copy.
- The scorer sums log-likelihood over answers of different lengths. For a fine-tuned model the label text after the digit is almost free; for a zero-shot model it is not, so the zero-shot baseline is also evaluated by generation.

## 2. Self-grill

```
Design tree
├── [✓] Q1  What runs are actually left
├── [✓] Q2  Which variants can the pinned code run
├── [✓] Q3  Where new results live
├── [✓] Q4  How measurement is added
├── [✓] Q5  Pilot design
├── [✓] Q6  Unattended behaviour
├── [✓] Q7  Defaults and what stays off
├── [✓] Q8  How results are summarised
└── [✓] Q9  What is deliberately not in this build
```

**Q1. What is left to run?**
Scoring only: zero-shot baseline; timing and invalid-output audit of existing adapters; the original paper adapter audit; archived Messidor for existing and new adapters. Training: learning-rate study and alpha study at rank 16; optional extra variants; optional rerun of the rank × CLAHE grid at a chosen learning rate. Nothing is repeated solely to collect utilization.

**Q2. Can the pinned code run these?**
Learning rate and patience: yes, by changing the saved configuration before launch. Alpha and dropout: yes, with the variant-aware adapter check. BF16 variants: no. So `qwen_variant.py` wraps the pinned runner, records every override and its own hash in the run configuration, and leaves pinned files untouched.

**Q3. Where do new results go?**
`D:\Afterimage_Experiments\final_runs\<tag>\`, not `runs\`. The tables builder scans `runs\` and groups by rank and CLAHE only, so a learning-rate variant placed there would be merged silently into an existing condition. Keeping new runs apart protects the 24 verified runs and the committed tables.

**Q4. How is measurement added?**
Every job goes through the existing launchers, which already log GPU %, VRAM, CPU % and RAM every 5 seconds and record wall time and hardware. The new scorer adds what the logs lacked: per-image time for the five-answer scorer and for generation (first images excluded as warm-up), model load time, prompt tokens per image, and framework peak memory. Parameter counts come from weight-file headers.

**Q5. What does the pilot do?**
One small job per kind into a separate pilot folder: a 10-step training run for a learning-rate variant and for an alpha variant (which exercises the variant-aware check), and 12-image scoring runs for zero-shot, an existing 4-bit adapter, a BF16 adapter and archived Messidor. Each pilot checks its outputs. The full queue refuses nothing if the pilot was skipped, but the notebook puts the pilot first.

**Q6. What happens when something fails overnight?**
A failed job is recorded and the queue moves on. A busy GPU lock stops the queue, because every following job would fail the same way. Rerunning the queue skips finished jobs and resumes interrupted training from its last epoch checkpoint through the pinned recovery routine.

**Q7. What is on by default?**
On: zero-shot, benchmark, original-adapter audit, archived Messidor, learning-rate study (1e-4 and 2e-4, three seeds), alpha study (8 and 32, three seeds), Messidor for new CLAHE-on adapters. Off: rerun of the full grid at a chosen learning rate, because that choice should follow the study and Dr. Noor's view. The study uses rank 16 with CLAHE on, the paper's configuration, so it compares directly with the existing rank 16, BF16 and previous-split runs.

**Q8. How are results summarised?**
One summary file per final-runs folder, computed with the pinned metric function: accuracy, QWK, macro AUC, per-class sensitivity and specificity, timing, invalid-output counts, and seed means per variant. Existing adapters rescored for timing are compared against their saved predictions, which checks that the new scorer reproduces the pinned one.

**Q9. What is not in this build?**
- BF16 variants (needs a new worker).
- RetiZero training and benchmark (teammate; no checkpoint yet).
- Verified Messidor (no raw data yet).
- Folding the new runs into `build_paper_tables.py` with learning rate and alpha as condition keys.
- The CPU-only analysis items: colour-conversion test, class-count tables, duplicate and label-noise table, adjacent-error table, misclassified-image export, calibration.
- Removing the C: read dependency.

## 3. The queue

| Order | Stage | Jobs by default | GPU time (estimate) |
| --- | --- | --- | --- |
| 1 | Parameter counts | weight-file headers | none |
| 2 | Zero-shot baseline | 4-bit base on APTOS-361 CLAHE on and off; BF16 base on CLAHE on | about 30 min |
| 3 | Benchmark and invalid-output audit | six existing adapters rescored with timing and generation | about 1 h |
| 4 | Original paper adapter audit | 367 original-split test images, generation on all | about 10 min |
| 5 | Archived Messidor | zero-shot, three rank 16 adapters, three BF16 adapters | about 2.5 h |
| 6 | Learning-rate study | rank 16, CLAHE on, 1e-4 and 2e-4, three seeds; each then scored on archived Messidor | about 9–12 h |
| 7 | Alpha study | rank 16, CLAHE on, alpha 8 and 32, three seeds; each then scored on archived Messidor | about 9–12 h |
| 8 | Extra variants | none | — |
| 9 | Grid at chosen learning rate | off | about 20 h if enabled |
| 10 | Summary | CPU | none |

Training times assume about 85 minutes per run, the mean of the existing rank 16 runs. Runs at a lower learning rate may train for more epochs and take longer.
