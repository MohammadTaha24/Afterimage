# Reviewer response checklist

Source: [EA_REVIEW.docx](C:/Users/moht2/Downloads/EA_REVIEW.docx). Items follow the review's order. Check an item when the manuscript or response letter includes the evidence. The notebooks provide evidence for several items below.

## General comments

- [ ] **G1. State the novelty and motivation relative to recent DR studies.** Add a specific claim and support it with recent related work.
- [ ] **G2. Justify parameters throughout model development and analysis.** List what was inherited, tested, or chosen based on prior work. Link trials and explain their effect. The historical repeats measure variability at one setting.
- [ ] **G3. Add recent DR studies to the references.** Replace or supplement generic citations with relevant primary studies.

## Specific comments

- [ ] **S1. Review ML, DL, language model, and VLM studies using APTOS and Messidor-2.** Summarize their methods, cohorts, splits, metrics, and findings.
- [ ] **S2. State a specific research gap, update contributions, and discuss challenges.** Tie each contribution to the reviewed evidence and the study's limitations.
- [ ] **S3. Explain BGR, RGB, and LAB in §3.2.** Document OpenCV BGR input, RGB for crop/pad/resize, and the conversion to LAB for CLAHE on lightness. Explain that the team script converts RGB directly to LAB. The equivalent batch implementation used BGR-to-LAB after converting resized RGB back to OpenCV's BGR. Describe the route used in each workflow and cite the pixel check. [Preprocessing source](Qwen3/preprocess_qwen.py)
- [ ] **S4. Justify Qwen3-VL-8B, RetiZero, and the comparison.** Explain what comparing a general vision-language model and a retina-specific model tests.
- [ ] **S5. Add the methodology and both architecture diagrams.** Show image inputs, model outputs, fine-tuning, and the grade selection or classification head.
- [ ] **S6. Justify LoRA rank 16, alpha 16, dropout 0.05, and bias none.** Report available rank trials, explain that fixed alpha also changes alpha/r, and document the basis or remaining evidence gap for each setting.
- [ ] **S7. Run the with/without QLoRA ablation.** Define the baseline explicitly and use matched training and evaluation. A CLAHE comparison does not vary QLoRA.
- [ ] **S8. Explain the Qwen tuning parameters and provide trials.** Report settings, selection by validation loss, and same-seed variability. Link available tuning trials or state which settings were inherited. [Training notebook](finetuning/Qwen3/notebooks/02_training.ipynb)
- [ ] **S9. Explain why RetiZero omitted CLAHE and report augmentation and class counts.** State which models and datasets use augmentation, and give train/test counts before and after. Distinguish saved images from online transforms or weighted sampling. Include the paired Qwen CLAHE results as Qwen evidence. [CLAHE notebook](finetuning/Qwen3/notebooks/01_preprocessing.ipynb)
- [ ] **S10. Compare methods fairly against existing work.** Align evaluation cohorts and metrics where possible. Record differences in preprocessing, split definitions, duplicate exposure, and endpoints. Preserve the original APTOS split for this reproduction and report its overlap.
- [ ] **S11. Explain misclassifications.** Use confusion matrices, per-grade recall, and reviewed examples. Label suspected causes separately from verified causes.
- [ ] **S12. Explain low test accuracy and compare with similar studies.** Report all repeats and their spread, and explain cohort and protocol differences.
- [ ] **S13. Explain substantial QWK alongside accuracy.** Use the confusion matrix to show how often errors are between adjacent grades versus far-apart grades.
- [ ] **S14. Compare complexity, trainable parameters, memory, and inference time.** Use the Qwen attempt reports for time and memory, add matched RetiZero measurements, and report complexity and trainable parameter counts for both.

## Evidence to collect with this workflow

- [ ] Replay the archived adapter on all 367 APTOS and 1,057 archived Messidor images. Record grade differences before retraining.
- [ ] Complete three fresh historical repeats and compare accuracy, QWK, and AUC against the three archived repeats.
- [ ] Build both APTOS CLAHE branches from the same raw originals. Verify IDs, grades, order, and source fingerprints.
- [ ] Complete all three paired CLAHE runs on the original APTOS split. Record per-grade recall and the original split's known exact duplicate overlap.
- [ ] Verify raw Messidor originals, grading, and patient pairing before a matched external CLAHE comparison. Historical Messidor replay uses the archived cohort.
- [ ] Attach system specs, utilization logs, and stage summaries to the run evidence.
