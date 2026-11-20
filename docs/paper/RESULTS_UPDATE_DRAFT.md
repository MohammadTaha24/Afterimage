Subject: Revision progress update (Qwen3-VL and RetiZero)

Dear Dr. Noor,

I hope you are well. Here is an update on the revision experiments. I have attached two spreadsheets: Afterimage_results.xlsx has all the results, each linked to the log or evaluation file it came from, and EA_review_checklist.xlsx tracks each reviewer point. I refer to the sheet names below.

1. Clean split vs original split (sheet Data_split)
While checking the data I found that 19 of the 367 test images in our original split have byte-identical copies outside the test set (17 in train, 2 in validation), and 3 of those copies have a different grade. I rebuilt the split without the overlap, which gives 361 test images (2,786 train, 357 validation). The reviewers asked us to report the split and overlap, so I would like to use the clean 361 set as the main result and keep the original 367 result as a reference. The effect is small: our saved RetiZero model gives 82.29% on the 367 images and 82.55% on the 361.

2. Results on the 361 clean test images (sheets Qwen_summary, RetiZero_summary, Head_to_head)
Each fine-tuned setup was trained with three seeds (42, 43, 44); values are mean ± SD across the three runs.
- Qwen zero-shot: 46.54% (QWK 0.674) with CLAHE, 54.57% (QWK 0.735) without
- Qwen rank 16, lr 5e-4: 69.71% ± 13.46 (QWK 0.670 ± 0.337); some seeds collapse to one or two grades
- Qwen rank 16, lr 2e-4: 78.12% ± 2.22 (QWK 0.894 ± 0.007)
- Qwen rank 16, lr 1e-4: 74.98% ± 2.08 (QWK 0.880 ± 0.029)
- RetiZero, retrained from the official pretrained checkpoint: 79.04% ± 1.69 (QWK 0.894 ± 0.009)
- RetiZero with CLAHE: 79.32% ± 1.53 (QWK 0.892 ± 0.016)
- RetiZero on the original split (367): 80.11% to 81.47% (QWK 0.905 ± 0.007)
RetiZero is also much faster: about 16 ms per image against about 0.86 s for Qwen (sheet Efficiency).

3. Remaining Qwen runs
Because the 5e-4 runs are unstable, I am running the main Qwen grid again at 2e-4 (ranks 8, 16 and 32, CLAHE on and off, three seeds). For the LoRA question (S6, your item 5) I am also running rank 8 with alpha 8 and rank 32 with alpha 32, so the alpha/rank scaling stays fixed and only the rank changes. The 5e-4 and 1e-4 runs will be reported as tuning runs.

4. Messidor (sheet Messidor)
Both models drop on the archived Messidor set of 1,057 images (Qwen about 48 to 50%, RetiZero 40.46% ± 7.33). The grade distribution of our archive does not match the published Messidor-2 grades, so I am tracing the exact Kaggle dataset we originally used to confirm the labels.

Questions
1. Do you agree with using the clean 361 set as the main result, with 367 as the reference?
2. Should Messidor stay in the paper as the external test while I confirm the source of the labels?
3. When would you like the full revised draft, so you have time to review it before December?

Kind regards,
Mohammad Taha
