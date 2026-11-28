# Handoff for ChatGPT: guide to the 22 revision figures

Written 25 November 2026. The figures come from `make_paper_figures.ipynb`. In the repository they are written to `finetuning/paper_figures/output/`; in the results package (`afterimage_results_<date>`) to `figures/`. F01–F19 are drawn from the analysis tables and per-image predictions. F20–F22 are drawn from each run's training log. Each figure `Fxx` has three files:

- `Fxx_*.png`: 300 dpi
- `Fxx_*.pdf`: vector
- `Fxx_*_data.csv`: the exact numbers plotted

`figure_index.csv` there has the draft captions. Quote numbers from the `_data.csv` or the analysis tables, not from reading the image. `training_histories_all_runs.csv` holds the training history of all 72 trained runs (63 Qwen, 9 RetiZero).

Conventions in every figure:
- **Colours:** blue is the headline Qwen model (LoRA r16, lr 2e-4, CLAHE on). Pink is Qwen at the original lr 5e-4. Grey is zero-shot Qwen. Orange is RetiZero: light for CLAHE off (its native setting), dark for CLAHE on.
- **Bars and dots:** a bar is the mean over three seeds; dots are the individual seeds.
- **Cohorts:**
  - "APTOS-361": the clean in-domain test set
  - "Messidor archived": 1,057 images
  - "Messidor-2 verified": 1,744 images, the external test
- **Failed run:** one that predicts at most 3 of the 5 grades, marked ×.

Proposed placement: **main text** F01, F02, F03, F06, F08, F12, F16; everything else in the supplement.

---

### F01. Cohorts and class balance (S9, S10, S12)
- **Shows:** A, the grade mix of each cohort; B, how many training images per grade each model sees per epoch.
- **Read it as:**
  - APTOS splits are about 50% grade 0, 10% grade 1 and 27% grade 2.
  - Archived Messidor has twice the share of Mild (20%).
  - Messidor-2 is more Normal-heavy (58%) and has fewer severe and PDR cases (4% and 2%).
  - In B, Qwen's weighted sampler draws about 557 images per grade per epoch, while the training set holds 1,436 / 267 / 729 / 140 / 214. RetiZero trains on the images as they are, with a class-weighted loss.
- **Point for the paper:** the external cohort differs in grade mix as well as camera, which helps explain the drop in accuracy. It also answers S9's request for class counts before and after balancing.

### F02. Headline performance (S10, S12, G1)
- **Shows:** accuracy, QWK and macro AUC for five models, in-domain and on Messidor-2.
- **Read it as:**
  - In-domain, Qwen lr 2e-4 (78.1%) and RetiZero (79.0%) are level.
  - On Messidor-2 everything drops, RetiZero most (40.2%; one seed is at 28%).
  - Qwen lr 5e-4 has one failed seed on APTOS (54.6%) and an unstable QWK.
- **Point:** the central result. The two models are equal in-domain, and Qwen transfers better.

### F03. Forest plot of every paired comparison (S6–S10, G2)
- **Shows:** each comparison as Δ = A − B with a 95% CI, accuracy on the left and QWK on the right, grouped by question. A filled dot means the interval excludes zero.
- **Read it as, group by group:**
  - **H1 on APTOS:** no difference.
  - **Messidor-2:** Qwen better by about 15 pt.
  - **Adapter vs zero-shot:** large and significant.
  - **Learning rate:** 2e-4 vs 5e-4 is significant only for QWK, because 5e-4 varies so much.
  - **CLAHE:** no fine-tuned difference; it hurts zero-shot.
  - **Rank at lr 2e-4:** all CIs tight around zero.
  - **Rank at lr 5e-4:** huge CIs, because runs fail.
- **Point:** one picture that answers most of the "justify your parameters" comments.

### F04. LoRA rank and alpha grid (S6)
- **Shows:** accuracy and QWK per seed for rank 8, 16 and 32 with alpha 16, and with alpha = rank, CLAHE on and off. Blue is lr 2e-4 (30 runs); grey is lr 5e-4.
- **Read it as:**
  - At lr 2e-4 every configuration sits at 72–79% with QWK about 0.88–0.90, and no run fails.
  - At lr 5e-4 the same ranks scatter, with failed runs down to 8%.
- **Point:** rank does not matter at the selected rate. The earlier rank "effect" came from the learning rate. Alpha = rank (a fixed alpha/r) was tested too.

### F05. Learning rate chosen on validation loss (S8, G2)
- **Shows:** A, validation loss against test accuracy for all 63 fine-tuned Qwen runs; B, mean validation loss per learning rate at rank 16.
- **Read it as:**
  - Failed runs (×, all lr 5e-4) have validation loss 0.0142–0.0199. Working runs have 0.0069–0.0107. The grey band is the gap between them.
  - Validation loss is lowest at lr 2e-4 (0.00768). That rate also gives the best test accuracy, 78.1%, but it was chosen without the test set.
- **Point:** the hyperparameter was chosen on validation data, and failure can be detected without the test set.

### F06. With and without the LoRA adapter (S7)
- **Shows:** the same 4-bit model with no adapter (grey) and with the adapter at lr 5e-4 (pink) and 2e-4 (blue), on APTOS and Messidor-2, CLAHE on and off.
- **Read it as:** the adapter adds about 31 pt in-domain (46.5% → 78.1%, CLAHE on) and about 11 pt on Messidor-2 (43.8% → 55.4%). QWK roughly doubles on Messidor-2 (0.29 → 0.59).
- **Point:** the reviewer asked for exactly this ablation (S7). Fine-tuning is what makes Qwen work.

### F07. CLAHE on vs off (S3, S9)
- **Shows:** per seed, each model with CLAHE off (open dots) and on (filled dots); lines join the means. Four panels: accuracy and QWK, APTOS and Messidor-2.
- **Read it as:**
  - Fine-tuned Qwen and RetiZero change little and inconsistently.
  - Zero-shot Qwen loses about 8 pt with CLAHE.
  - RetiZero on Messidor-2 looks better with CLAHE (47% vs 40%), but its interval includes zero, because the CLAHE-off seeds vary so much.
- **Point:** CLAHE is not needed. Omitting it for RetiZero was not a disadvantage.

### F08. Confusion matrices (S11, S13)
- **Shows:** row-normalised matrices with counts, three seeds pooled, for Qwen lr 2e-4 and RetiZero on APTOS and Messidor-2.
- **Read it as:**
  - In-domain, errors sit next to the diagonal, mostly Severe ↔ Moderate/PDR.
  - On Messidor-2, Qwen calls 71% of Mild eyes Normal and 61% of Moderate eyes Mild.
  - RetiZero calls 48% of Normal eyes Mild.
- **Point:** shows where the models err (S11), and that errors are mostly one grade off (S13). The two models fail in different directions on Messidor-2.

### F09. Error distance (S13)
- **Shows:** the share of predictions that are exact, or off by 1, 2, or 3–4 grades, for 4 models × 3 cohorts.
- **Read it as:**
  - On APTOS, about 78% exact and about 17% off by one; off by 3 or more is close to 0 except at lr 5e-4.
  - On Messidor-2, Qwen is 55% exact, 36% off by one and 7% off by two.
- **Point:** why QWK stays high while accuracy is moderate.

### F10. Class-wise sensitivity and specificity (S11)
- **Shows:** per-grade sensitivity and specificity (mean ± SD over seeds) for the two headline models, on APTOS and Messidor-2.
- **Read it as:**
  - On APTOS, Severe is the hardest grade (sensitivity about 0.4) for both models.
  - On Messidor-2, Qwen keeps Normal (0.80) but misses Moderate (0.14).
  - RetiZero's Normal sensitivity falls to 0.48.
- **Point:** class-level detail for S11.

### F11. ROC curves per grade (reviewer comments 10–12 on ROC-AUC)
- **Shows:** one-vs-rest ROC for each grade, one line per seed, with the mean AUC printed in each panel.
- **Read it as:**
  - On APTOS every AUC is 0.89–1.00 for both models.
  - On Messidor-2, Severe and PDR stay high (0.91–0.97), and Normal and Moderate fall (Qwen 0.76 and 0.77; RetiZero 0.64 and 0.63).
  - **Mild is at chance for both models (0.49 and 0.48).**
- **Point:** shows how AUC was computed (each grade's own probability column) and locates the external failure in the Mild grade.

### F12. Referable DR: ROC and operating point (S12, limitation)
- **Shows:** ROC for referable DR (grade ≥ 2, score p2 + p3 + p4), with a dot at each model's own operating point (predicted grade ≥ 2).
- **Read it as:**
  - On APTOS, the models operate near the top-left corner.
  - On Messidor-2, the fine-tuned models sit at sensitivity about 0.26–0.41 with specificity about 0.94–0.99. Yet two of the three Qwen seed curves pass 0.8 sensitivity at about 0.2 false-positive rate (the third at about 0.4).
  - Referable AUC: Qwen 0.851, RetiZero 0.728.
  - The saved RetiZero app checkpoint reaches 0.50 sensitivity.
- **Point:** the most important clinical caveat. As trained, the models would miss most referable eyes on a new camera and population. A recalibrated threshold could help, but choosing it on Messidor-2 would be post hoc.

### F13. Mild vs Normal (S11)
- **Shows:** the share of Mild eyes called Normal, of Normal eyes called Mild, and the Mild-vs-Normal AUC, on APTOS and both Messidor cohorts.
- **Read it as:**
  - On APTOS, Mild-vs-Normal AUC is about 0.98, and under 10% of Mild eyes are called Normal (18% at lr 5e-4).
  - On Messidor, AUC is about 0.47–0.62, near the dashed chance line, and 55–83% of Mild eyes are called Normal.
- **Point:** the main failure mode. The cause is a *suspected* domain shift, not confirmed by a grader, and the paper must say so. The labels are the adjudicated grades, so it is not a labelling error.

### F14. Predicted vs true grade distribution (S11, S12)
- **Shows:** the share of images per grade, true (black) against each model's predictions.
- **Read it as:**
  - On APTOS, the predictions match the true mix.
  - On Messidor-2, every fine-tuned model predicts too few Moderate cases (about 8–10% against 20% true) and too many Mild ones (Qwen 27%, RetiZero CLAHE-off 46%, against 15% true).
  - Zero-shot Qwen over-calls Severe.
- **Point:** shows the under-grading behind the low referable sensitivity in F12.

### F15. Calibration (robustness)
- **Shows:** reliability diagrams (confidence against accuracy, 15 bins, the same bins as the ECE) and expected calibration error (ECE) per seed.
- **Read it as:**
  - On APTOS, ECE is about 0.05–0.07 (0.15 at lr 5e-4).
  - On Messidor-2 it is about 0.24–0.32: both models are over-confident, with accuracy well below confidence in the middle bins.
- **Point:** model confidence is not reliable out of domain. It supports the recalibration point.

### F16. Computational efficiency (S14)
- **Shows:** top, six side-by-side comparisons; bottom, Qwen inference time and energy per image for seven configurations.

| Quantity | Qwen | RetiZero |
|---|---|---|
| Trainable parameters | 51.3 M | 304 M |
| Total parameters | 8,818 M | 304 M |
| Training time, one full run | about 83 min | about 5 min |
| Inference per image | 861 ms | 17.7 ms |
| Peak training memory | 8.9 GiB | 11.2 GiB |
| Peak inference memory | 8.7 GiB | 1.2 GiB |

- **Read it as:**
  - The bottom panels are flat: about 760 ms and 0.09 Wh per image for every configuration.
  - The 861 ms at the top is an earlier measurement of the lr 5e-4 runs, so quote 0.76–0.86 s.
  - Energy is whole-card GPU power.
- **Point:** the cost of the general model. RetiZero is about 50 times faster and uses about 7 times less inference memory. Qwen trains far fewer weights, so it uses less training memory.

### F17. Seed-to-seed variability (S12, G2)
- **Shows:** every seed of every fine-tuned condition on APTOS-361, sorted by mean accuracy, coloured by learning rate; × marks a failed run.
- **Read it as:** every failed run is at lr 5e-4 (pink). lr 2e-4 runs (blue) and RetiZero (orange) spread over a few points, mostly 70–82%.
- **Point:** report all repeats (S12). The variability comes from the learning rate.

### F18. Near-duplicate sensitivity (S10)
- **Shows:** accuracy and QWK on all 361 test images, and after removing test images that have a near-duplicate in training (perceptual hash distance 0, ≤ 2, ≤ 4; 6, 32 and 120 images removed).
- **Read it as:**
  - Removing 6 or 32 images moves accuracy by under 1 pt.
  - Removing 120 lowers it about 4 pt for every model, because those 120 are 67% grade 0, the easiest class. This is a class-mix effect, not leakage.
- **Point:** near-duplicates did not inflate the results.

### F19. Archived vs verified Messidor (S10, data integrity)
- **Shows:** A, each run's accuracy on the archived Messidor inputs against the inputs rebuilt from the raw photos, same 1,057 images; B, archived accuracy against accuracy on all 1,744 verified images.
- **Read it as:**
  - In A, Qwen points lie exactly on the diagonal: the same grade on all 1,057 images.
  - RetiZero is within 0.5 pt (1,031–1,042 of 1,057 the same), because its archived route went raw → 512 → 224.
  - In B, most runs score higher on the full 1,744. The added JPG half differs in grade mix.
- **Point:** the earlier Messidor results were not an artefact of the old export.

### F20. Qwen training curves by learning rate (S8, G2; from the training logs)
- **Shows:** Qwen r16, CLAHE on, three seeds at each of lr 1e-4, 2e-4 and 5e-4. A, training loss (running median of the trainer log, log scale); B, validation loss after each epoch (log scale). The outlined dot is the kept checkpoint (lowest validation loss); dashed with × is a failed run.
- **Read it as:**
  - Training stops after two evaluations without improvement, so runs last 3–9 epochs.
  - Kept validation loss: lr 1e-4 0.0075–0.0082; lr 2e-4 0.0073–0.0080; working lr 5e-4 runs 0.0079–0.0086; the failed lr 5e-4 run 0.0151.
  - The two large spikes (one lr 2e-4 seed at epoch 5, one lr 5e-4 seed at epoch 8) come after those runs' kept checkpoints, so they do not affect the results.
- **Point:** shows from the raw logs how the learning rate was chosen on validation data, and that failure at 5e-4 is visible in training, not only on the test set. F05 has the summary numbers.

### F21. RetiZero training curves (S8, S12; from the training logs)
- **Shows:** RetiZero per epoch, three seeds per setting: clean split CLAHE off (native), clean split CLAHE on, and the original-split replay. Training loss, validation loss and validation accuracy; the outlined dot is the kept epoch (highest validation accuracy).
- **Read it as:**
  - Training stops after four epochs without a higher validation accuracy (at most 15), so runs last 6–14 epochs.
  - Kept epochs are 2–10, with validation accuracy 77.9–79.8%.
  - Validation loss is lowest at epoch 2–7 and then rises while training loss keeps falling, so later epochs overfit.
- **Point:** documents the RetiZero training protocol and that it selects on accuracy, while Qwen selects on loss (a protocol difference to state in Methods).

### F22. Validation loss across the LoRA grid at lr 2e-4 (S6, G2; from the training logs)
- **Shows:** validation loss after each epoch for all 30 adapters of the lr 2e-4 grid (rank 8, 16 and 32; alpha 16 or alpha = rank; CLAHE on and off; three seeds). Log scale.
- **Read it as:** every run's lowest validation loss is between 0.0069 and 0.0097, and runs stop after 3–8 epochs. No rank or alpha stands apart.
- **Point:** supports F04: at the chosen learning rate the LoRA rank and alpha do not matter, already on validation data.

---

**Not covered by the figures (still to make):**
- the S5 methodology and architecture diagrams, to draw by hand
- Grad-CAM or misclassified-image examples (S11): these need the fundus images, which stay outside the repository
