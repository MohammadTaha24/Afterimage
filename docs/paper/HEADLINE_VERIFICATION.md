# Headline result set: independent verification

Claim checked: *"The clean 361-image test suite should be the paper's main result; the original 367-image result is a historical reference only, because 17 of the original 367 test images are byte-identical to training images."*

Checked on 2026-11-16. Every number below comes from the image files, not from the existing manifests: SHA-256 of the raw Kaggle PNG bytes, SHA-256 of the decoded pixels, and the same two hashes of the prepared 512-pixel images. Split membership comes from the dataset image folders, and labels come from the Kaggle `train.csv`.

## Verdict: the claim holds, with corrections

Using the 361-image suite as the main result is justified. The "17" is right only if "training images" means the train split alone. Four other things need saying in the paper:

1. **19, not 17, of the 367 test images have an exact copy outside the test set.** 17 match a train image and 2 more match a validation image. Byte comparison and pixel comparison find the same 19, so no copy is identical in pixels but different in its file bytes.
2. **3 of the 17 train-copy pairs have different grades.** The test copy and the train copy were graded differently: `14e3f84445f7` is grade 3 in test and grade 4 in train, `4a44cc840ebe` is 2 and 3, and `51131b48f9d4` is 1 in test with two train copies graded 2. The other 16 pairs (14 with train, 2 with validation) have the same grade.
3. **The 361 set is not "the 367 minus the 17".** It is 367 − 3 − 3. Only the 3 leaked test images with a grade conflict were removed. The other 16 leaked test images stay in the test set, and their train or validation copies were removed instead. The other 3 removed test images were each one half of a test–test duplicate pair.
4. **The 361 set is clean only for models trained on the clean 2,786-image train split.** 16 of the 361 test images have exact copies in the *original* train set (14) or validation set (2). Any model trained on the original split and then scored on the 361 images still had those copies in training. That covers the three `full_r16_on_previous_split_*` Qwen runs, the saved RetiZero checkpoint scored as `saved_checkpoint__aptos361_clean`, and Retraining 2 or the original paper adapter if they are rescored on 361. Do not call those results "clean".

The rebuilt split removed exact duplicates only. A perceptual hash finds a few likely near-duplicates (section (f)), so the paper should say "exact duplicates removed", not "duplicate-free".

## Corrected wording

**Paper (Methods/Data):**

> The original random split of the 3,662 labelled APTOS 2019 images (2,929 train / 366 validation / 367 test) contained exact duplicates. 17 of the 367 test images were byte-identical to a training image, three of them with a different grade, and two more were identical to a validation image. We therefore removed exact duplicates (identical file bytes, confirmed on decoded pixels) from the whole labelled set before splitting. There were 123 duplicate groups containing 251 images. Where all copies in a group had the same grade, we kept one copy; where the grades disagreed (30 groups), we removed every copy. This removed 158 images (143 train, 9 validation, 6 test) and left 2,786 / 357 / 361 images with no identical image within or across splits. All main results are on this 361-image test set, using models trained on the 2,786-image train set. Results on the original 367-image test set are given only for comparison with earlier work. Near-duplicates (for example, the same image re-encoded or slightly altered) were not removed.

**Email to supervisor:**

> I checked the split leak again from the image files. In the original split, 17 of the 367 test images are byte-identical to a training image (3 of those pairs have different grades), and 2 more are identical to a validation image, so 19 in total. The clean split removes all exact duplicates. Where copies had the same grade it keeps one, and where the grades disagreed it removes every copy. The result is 2,786/357/361 with no identical images within or across splits. The test set lost 6 images: the 3 leaked images with a grade conflict, plus one image from each of 3 duplicate pairs that were both inside the test set. The other 16 leaked test images stay in the test set because their training copies were removed. Qwen and RetiZero use exactly the same 361 test images. I propose the 361-image results as the main result and Retraining 2 (367) as a historical reference only. One caution: models trained on the old split and scored on the 361 still had 16 of those test images in training, so only models trained on the clean split count as clean. The leak does not seem to have inflated the old numbers. On the leaked images the old models were no more accurate than on the rest, and removing them changes overall accuracy by less than 1.6 points. Could you confirm this choice?

## Numbers

### (a) Original split (`aptos_512/previous_split`; identical in the CLAHE-on, CLAHE-off and 224-pixel folders, and equal to `frozen_run_02`)

| Split | n | G0 | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|
| train | 2,929 | 1,444 | 296 | 799 | 154 | 236 |
| val | 366 | 180 | 37 | 100 | 20 | 29 |
| test | 367 | 181 | 37 | 100 | 19 | 30 |

- Test images identical to a train or validation image: **19**. 17 have a train copy and 2 have only a validation copy. This forms 20 pairs, because `51131b48f9d4` has two train copies.
- Grade agreement across the 20 pairs: 16 same grade, 4 different grade. The 4 differing pairs involve 3 test images, all with train copies.
- Raw bytes, raw pixels, prepared-512 bytes and prepared-512 pixels all give the same result.
- Whole labelled set: 123 identical groups covering 251 images (119 pairs, 3 triples, 1 group of four), of which 30 groups have conflicting grades. In the original split, 43 of these groups cross a split boundary and 80 sit within one split.

Leaked test images (test id, grade → copy id, split, grade):
`034cb07a550f` 4 → `c8d2d32f7f29` train 4; `1006345f70b7` 2 → `435d900fa7b2` train 2; **`14e3f84445f7` 3 → `f0f89314e860` train 4**; `23d7ca170bdb` 2 → `ea9e0fb6fb0b` train 2; **`4a44cc840ebe` 2 → `0cb14014117d` train 3**; **`51131b48f9d4` 1 → `42a850acd2ac` train 2 and `8cb6b0efaaac` train 2**; `6c3745a222da` 4 → `eadc57064154` train 4; `7a0cff4c24b2` 2 → `86baef833ae0` train 2; `8fc09fecd22f` 1 → `d1cad012a254` train 1; `91cbe1c775ef` 2 → `7d261f986bef` train 2; `9f4132bd6ed6` 2 → `9b7b6e4db1d5` train 2; `bb7e0a2544cd` 3 → `e76a9cbb2a8c` **val** 3; `c9f0dc2c8b43` 2 → `9c5dd3612f0c` train 2; `cac40227d3b2` 2 → `0161338f53cc` train 2; `d28bd830c171` 2 → `f9ecf1795804` train 2; `d51b3fe0fa1b` 4 → `df4913ca3712` train 4; `d51c2153d151` 0 → `7b691d9ced34` train 0; `d85ea1220a03` 4 → `bfefa7344e7d` train 4; `d994203deb64` 2 → `a505981d1cab` **val** 2.

### (b) Clean split (`duplicate_excluded`) and how 361 arises

- Byte-identical or pixel-identical images in the clean split, at raw and at 512 pixels, across splits or within a split: **0**.
- No image changed split. The transitions are: train→train 2,786, train→excluded 143; val→val 357, val→excluded 9; test→test 361, test→excluded 6. The 158 excluded images all belong to an identical group.
- Rule recovered from the data: of the 93 same-grade groups, exactly 1 member was kept in each; of the 30 conflicting-grade groups, 0 members were kept.
- The 6 test images removed:
  - Leaked, with a grade conflict, so the whole group was removed: `14e3f84445f7` (G3), `4a44cc840ebe` (G2), `51131b48f9d4` (G1).
  - One half of a test–test pair, where the partner stays in test: `77a9538b8362` (G1, partner `40e9b5630438`), `8acffaf1f4b9` (G2, partner `1411c8ab7161`), `a56230242a95` (G2, partner `1c6d119c3d70`).
- The other **16 leaked test images stayed in the test set**, and their 14 train copies and 2 validation copies were removed.

### (c) Same 361 test IDs

All 46 saved 361-row prediction files contain exactly the clean test ID set, in a single shared order, with `y_true` equal to the Kaggle grade. That covers Qwen in `runs/` and `final_runs/final_01`, and RetiZero in `retizero_runs/retizero_01/scores/saved_checkpoint__aptos361_clean`. The 224-pixel and 512-pixel `duplicate_excluded` folders hold the same IDs. All split CSVs match their folders and the Kaggle labels, with 0 mismatches.

### (d) Clean split class counts

| Split | n | G0 | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|
| train | 2,786 | 1,436 | 267 | 729 | 140 | 214 |
| val | 357 | 179 | 36 | 96 | 19 | 27 |
| test | 361 | 181 | 35 | 97 | 18 | 30 |

### (e) How much the leak could matter (saved predictions on the original 367)

| Model (file) | All 367 | 19 leaked | 348 not leaked | 16 same grade | 3 conflicting |
|---|---|---|---|---|---|
| Qwen original paper adapter (`final_01/scores/original_paper_adapter__aptos367_original`) | 299 (81.47%) | 10 (52.6%) | 289 (83.05%) | 10/16 | 0/3 |
| Qwen Retraining 2 replay (`.../replay_run_02/evaluation/aptos_test.predictions.csv`) | 296 (80.65%) | 16 (84.2%) | 280 (80.46%) | 14/16 | 2/3 |
| RetiZero saved checkpoint (`retizero_01/scores/saved_checkpoint__aptos367_original`) | 302 (82.29%) | 12 (63.2%) | 302−12 = 290 (83.33%) | 11/16 | 1/3 |

Only Retraining 2 is more accurate on the leaked images than on the rest, by 3.8 points on 19 images. The other two models are *less* accurate on them, mainly because the 3 conflicting-grade images are scored against their test grade. Removing the 19 leaked images moves overall accuracy by −0.19 to +1.58 points. **No inflation from the leak is detectable at this sample size.** The leak is still a protocol flaw, and that is the reason to replace the 367 set.

Models trained on the original split and scored on the 361 images: accuracy on the 16 images whose copies were in their training data, compared with the other 345:
- `full_r16_on_previous_split` t42: 10/16 vs 274/345 (79.42%), overall 78.67%
- t43: 11/16 vs 247/345 (71.59%), overall 71.47%
- t44: 12/16 vs 284/345 (82.32%), overall 82.00%
- RetiZero `saved_checkpoint__aptos361_clean`: 11/16 vs 287/345 (83.19%), overall 82.55%

Again there is no sign of inflation. Clean-trained Qwen runs score a similar 7–11 out of 16 on the same images.

Not part of the claim, but seen during the check: several saved 361-row files score 30/361 or 152–197/361. Examples are `full_r32_on_duplicate_excluded_t44`, `full_r8_off_duplicate_excluded_t44`, `final_01/train/r16_on_alpha8_t44` and `r16_on_alpha32_t44`. These are probably the FP16-overflow cases that commit `d063e85` addresses. Make sure the tables use the scored or FP32 files, not these.

### (f) Near-duplicates across clean splits (approximate)

These counts use a 64-bit DCT perceptual hash on the prepared 512-pixel CLAHE-off images. As a check of the method, all 20 known identical pairs from the original split come out at Hamming distance 0. The counts are **approximate and not checked by eye**. Fundus images look alike after cropping, so larger distances will include many false matches.

| Hamming distance | Cross-split pairs | Pairs involving test | Distinct test images |
|---|---|---|---|
| 0 | 6 | 6 | 6 |
| ≤2 | 100 | 57 | 32 |
| ≤4 | 894 | 469 | 120 |

The 6 pairs at distance 0, with no byte or pixel identity, are the most likely real near-duplicates (for example re-encoded or rescaled copies). They are worth checking by eye before submission (DELIVERABLES_CHECKLIST item "Near-duplicate check").

## Verification method

Anyone can repeat this check without trusting the existing manifests.

1. Inputs, all read-only: the raw Kaggle images `D:\aptos2019-blindness-detection\train_images\*.png` with `train.csv`; the dataset folders `D:\Afterimage_Experiments\datasets\aptos_512\{previous_split,duplicate_excluded}\{clahe_off,clahe_on}\images\{train,val,test}` and `aptos_224_standard\...\clahe_off`; and the saved prediction CSVs named in section (e).
2. Script: `D:\Afterimage_Reproduction\.verification\headline\verify_headline.py`. `.verification/` is git-ignored, so copy the script into `scripts/` before sharing the repository. It uses only NumPy and Pillow.
3. From the repo root, with the project environment, run:
   `C:\Users\moht2\.conda\envs\afterimage_env\python.exe -X utf8 -B .verification\headline\verify_headline.py`
   It takes about 30 seconds on CPU. It writes `.verification\headline\headline_verification.json`, and its console output is saved in `run.log`.
4. What the script does:
   - Takes each image's split from the folder it sits in, and checks that the 512 CLAHE-on, 512 CLAHE-off and 224-pixel folders agree.
   - Checks the CSVs and Kaggle labels against the folders.
   - Hashes the raw bytes and the decoded pixels (mode, shape and array) of every image, raw and prepared.
   - Groups identical images and lists every test image with a train or validation copy.
   - Maps each image from its original split to its clean split.
   - Checks the IDs in every 361-row prediction file.
   - Computes accuracy on the leaked images compared with the rest.
   - Runs the perceptual-hash sweep.
5. Expected console output (key lines):
   - `csv_vs_folder_or_kaggle_label_mismatches: 0`
   - `original_raw_bytes: {"test_images_identical_to_train_or_val": 19, "of_which_partner_in_train": 17, "of_which_partner_in_val_only": 2, "pairs": 20, "pairs_same_grade": 16, "pairs_different_grade": 4, ...}`, and the same for `raw_pixels` and both `prep512` lines
   - `dataset_duplicate_groups_bytes: {"groups": 123, "images": 251, ..., "conflicting_grade_groups": 30}`
   - `clean_raw_bytes: {"test_identical_to_train_or_val": 0, "cross_split_duplicate_groups": 0, "within_split_duplicate_groups": 0}`, and the same for the other three clean lines
   - `transitions_original_to_clean: {"test->excluded": 6, "test->test": 361, "train->excluded": 143, "train->train": 2786, "val->excluded": 9, "val->val": 357}`
   - `pixel_dup_groups_by_grade_agreement_and_members_kept: {"conflicting|kept=0": 30, "same_grade|kept=1": 93}`
   - `files_361_all_equal_clean_test: true`, `files_361_distinct_orders: 1`
   - `phash_sanity_leaked_pair_distances`: twenty zeros
