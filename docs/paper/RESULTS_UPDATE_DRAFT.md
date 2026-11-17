# Results update to Dr. Noor (draft)

**Subject:** Results update: Qwen3-VL and RetiZero paper revision

Dear Dr. Noor,

Here is a short update on the revision experiments.

**Summary.** The final Qwen runs are finished: zero-shot baselines, a learning-rate study, an alpha study, timing, and scoring on the archived Messidor cohort, all on the 361-image test set.

I propose the clean 361-image suite as the main result. In the original 367 split, 19 test images have an exact copy outside test (17 in train, 2 in validation), 3 with a different grade. The 361 is the 367 minus those 3 and 3 test-test duplicates. Please confirm. Method and script are documented and can be shared.

At the learning rate used so far (5e-4) the three-seed mean is 69.71 ± 13.46%, because some runs are unstable. At 1e-4 and 2e-4 the results are higher and stable (74.98 ± 2.08% and 78.12 ± 2.22%, no collapsed runs). RetiZero retraining on the same 361 images has not started; it needs the official pretrained checkpoint.

| Item | Accuracy | Other |
|---|---|---|
| Zero-shot, 4-bit, CLAHE on / off | 46.54% / 54.57% | 0.712 / 0.718 s per image |
| Zero-shot, BF16, CLAHE on | 52.63% | 1.344 s per image |
| Fine-tuned rank 16, 5e-4 (3 seeds) | 69.71 ± 13.46% | QWK 0.6701 |
| Fine-tuned rank 16, 2e-4 (3 seeds) | 78.12 ± 2.22% | QWK 0.8939 |
| Fine-tuned rank 16, 1e-4 (3 seeds) | 74.98 ± 2.08% | QWK 0.8795 |
| Fine-tuned, timing benchmark | n/a | 0.852 to 0.901 s per image (4-bit); 1.348 s (BF16) |
| Alpha 8, 5e-4 (3 seeds) | 51.89 ± 37.75% | one run predicts a single grade |
| Alpha 32, 5e-4 (2 seeds, FP16) | 29.64 ± 30.16% | third seed overflowed in FP16; scored in FP32 (all three: 36.47 ± 24.39%, mixed precision) |
| Original paper adapter, 367 images | 81.47% | QWK 0.9286 |
| RetiZero saved checkpoint, 367 / 361 images | 82.29% / 82.55% | not leak-free (trained on the original split) |
| Archived Messidor (1,057 images) | 43.52% to 54.12% | zero-shot 45.98% |

**Reviewer points.**
- S6: the alpha runs do not answer it. Of six, three predict one grade for every image and one more reaches only 50.97%. Rank with alpha/r fixed is still open.
- S7: the no-adapter baseline is done and sits well below the fine-tuned models; the BF16 base is done too.
- S8: the learning-rate runs give the tuning evidence.
- S14: the efficiency table is being computed from the saved run records. Parameter counts are done.

**Decisions I have made, for your correction.** Main results stay at 5e-4, with 1e-4 and 2e-4 reported as tuning and stability evidence. CLAHE on and off are both reported. Messidor-2 source verification is in progress.

**Questions.**
1. Should I repeat the full grid (ranks 8, 16, 32, CLAHE on and off, three seeds; about 20 GPU-hours) at 1e-4 or 2e-4, or keep 5e-4?
2. Do you agree that the clean 361-image result is the main result and the original 367-image result is historical reference only?
3. For S6, should I run rank with alpha equal to rank, or state it as a limitation?
4. Who holds the raw Messidor-2 images, grades and eye pairing?

**Next steps.** I will download the official RetiZero checkpoint, repeat its pilot, then train the three seeds, and finish the efficiency table. A repeat Qwen grid starts only after your answer to question 1.

Kind regards,
Mohammad Taha

---

## Sources

Reviewer numbering follows `REVIEWER_CHECKLIST.md`, one higher than your email from your item 5.

| Number | File |
|---|---|
| Zero-shot accuracy and timing; benchmark timing; original adapter 81.47% / 0.9286; Messidor range and zero-shot; alpha means; FP16 overflow and FP32 fallback | `D:\Afterimage_Experiments\final_runs\final_01\summary.md` |
| 1e-4: 74.98 ± 2.08, 2e-4: 78.12 ± 2.22, QWK means | same file, "Training variants" table |
| 5e-4: 69.71 ± 13.46, QWK 0.6701 | `finetuning/paper_tables/output/SUMMARY.md`; also `docs/paper/DELIVERABLES_CHECKLIST.md` (section B) |
| Alpha 32 seed 42 at 50.97% | `summary.md`, alpha_study row |
| Decisions, 20 GPU-hours, RetiZero waiting for checkpoint | `docs/paper/REMAINING_WORK.md` (phases 1 to 3) |
| 19 of 367 test images have an exact copy outside test (17 train, 2 validation); 3 with a different grade; 361 = 367 − 3 − 3; method and script | `docs/paper/HEADLINE_VERIFICATION.md` (Verdict; sections (a), (b); Verification method) |
| RetiZero 82.29% (367) | `finetuning/RetiZero/README.md` (Status) |
| RetiZero 82.55% (361), not leak-free | `finetuning/RetiZero/REPRODUCTION_PLAN.md` (lines 11 and 71) |
| Parameter counts, S14 status | `DELIVERABLES_CHECKLIST.md` (section D) |
