Subject: Revision results update (Qwen3-VL and RetiZero)

Dear Dr. Noor,

I hope you are well. Below is an update on the revision experiments. I have attached two spreadsheets: Afterimage_results.xlsx has all the numbers, each with the log or evaluation file it came from, and EA_review_checklist.xlsx tracks each reviewer point. I mention the sheet names below so you can find the numbers.

1. Clean split vs original split (sheet Data_split)

While checking the data I found that the original split was not fully clean. 19 of the 367 test images have an exact copy outside the test set (17 in train, 2 in validation), and 3 of those copies have a different grade. I rebuilt the split so the test set has no overlap: 3 conflicting images and 3 test-test duplicates were removed, which gives 361 test images (2,786 train, 357 validation). The check compares the image files byte by byte. All new results below are on the 361 clean images, and I would like to use this as the main result, with the original 367 result kept as a historical reference. Could you confirm this?

2. Qwen3-VL (sheets Qwen_summary and Qwen_runs)

- Without fine-tuning (zero-shot): 46.54% with CLAHE and 54.57% without, using the 4-bit model.
- Rank 16, learning rate 5e-4 (the rate used so far), 3 seeds: 69.71% ± 13.46. Some seeds collapse to predicting one or two grades, which is where the large spread comes from.
- The same setup at 2e-4 gives 78.12% ± 2.22 and at 1e-4 gives 74.98% ± 2.08, with no collapsed runs.
- The alpha runs at 5e-4 were unstable too (3 of 6 predict one grade), so they do not answer the rank question (S6) yet.

3. RetiZero (sheets RetiZero_summary and RetiZero_runs)

I retrained RetiZero from the official pretrained checkpoint on the same clean split, 3 seeds:
- Clean split: 79.04% ± 1.69
- Clean split with CLAHE: 79.32% ± 1.53, so CLAHE makes no clear difference for RetiZero either
- Original split: 80.11% to 81.47%, close to our earlier 82.29%

4. Comparison (sheets Head_to_head and Efficiency)

On the same 361 images RetiZero (79.04%) is close to Qwen at 2e-4 (78.12%) and well above Qwen at 5e-4 (69.71%). I still need to run a paired statistical test. RetiZero is also much faster: about 16 ms per image against about 0.86 s for Qwen.

5. Messidor (sheet Messidor)

Both models drop on the archived Messidor set of 1,057 images (Qwen about 48 to 50%, RetiZero 40.46% ± 7.33). The label distribution of that archive does not match the published adjudicated Messidor-2 grades (Krause et al.), so I am not confident in those labels. I plan to rescore against the adjudicated grades, which are available on Kaggle.

Questions
1. Do you agree that the clean 361 result is the main result?
2. Should I repeat the full Qwen grid at a lower learning rate (about 20 GPU hours), or keep 5e-4 and report the lower rates as tuning evidence?
3. For S6, should I run rank with alpha/r fixed, or state it as a limitation?
4. Is it fine to use the adjudicated Messidor-2 grades as the reference?

Next I will do the paired comparison, finish the efficiency table, and the remaining analyses (class counts, misclassified examples, adjacent-grade errors).

Kind regards,
Mohammad Taha
