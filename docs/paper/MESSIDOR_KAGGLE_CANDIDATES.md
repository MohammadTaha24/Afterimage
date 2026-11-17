# Messidor-2 candidates on Kaggle and public mirrors

Read-only research. Nothing was downloaded or signed into. The Kaggle page itself returned only its title to the fetch tool, so the column names below come from a third-party mirror and need checking on Kaggle.

## What we have locally

- `D:\recovered\DR_Messidor_Only_512_Full_Dataset\Res_512\Messidor_CLAHE` and `Messidor_Standard`: 1,057 PNG each.
- File names: `messidor_messidor_p_<N>_<YYYYMMDD>_<seq>_0400_PP.png`, e.g. `messidor_messidor_p_0_20051109_57451_0400_PP.png`. The core `20051109_57451_0400_PP` is the original Messidor PNG naming.
- `Split_80_10_10/dataset_80_10_10.csv` header: `image_path,label,patient_id,source_dataset,vlm_prompt,split`. Labels 0 to 4 (ICDR-style scheme).
- Per-grade counts: 0 = 468, 1 = 207, 2 = 290, 3 = 71, 4 = 21 (total 1,057). 529 distinct `patient_id` values.
- Shares: 44%, 20%, 27%, 7%, 2%.
- `docs/paper/HEADLINE_VERIFICATION.md` has no Messidor mentions.

## Candidates

| # | Source | Content | Grades | Grade scheme | Notes |
|---|---|---|---|---|---|
| 1 | Kaggle `google-brain/messidor2-dr-grades` (https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) | Grades for the public Messidor-2 image set | **Adjudicated** (Krause et al. 2018; three retina specialists) | 0 to 4 DR, plus DME and gradability | 1,748 rows. Mirror lists columns `image_id`, `adjudicated_dr_grade`, `adjudicated_dme`, `adjudicated_gradable`. Owner, licence and update date not confirmed on Kaggle. |
| 2 | Baselight mirror of #1 (baselight.ai/u/kaggle/dataset/google_brain_messidor2_dr_grades) | Same table, third-party copy | Adjudicated (per mirror) | 0 to 4 | Not an original source. Use only to read the schema. |
| 3 | Hugging Face `sngsfydy/Messidor2` | 1,744 images, one `train` split, about 2.0 GB | Not stated | 0 to 4 | Grading source, adjudication and file naming not documented. Gradable count matches the 1,744 adjudicated subset, so it may derive from #1. Not verified. |
| 4 | Kaggle `eyepacs-aptos-messidor-diabetic-retinopathy` | Unified, resized, augmented mix of EyePACS, APTOS and Messidor | Not known | Not known | Seen only in search snippets. Could be the origin of our processed build, but nothing confirmed. |
| 5 | Official Messidor site (adcis.net, Messidor-Original and Messidor-Extension) | Raw images. Messidor-Original = 529 exams, 1,058 PNG. Messidor-Extension = 345 exams, 690 JPG. | None (grades come from third parties) | None | Raw source. Requires a request and an account. Licence is research and educational use only, with no redistribution. |

## Best match

**Our 1,057-image archive is most likely the PNG subset of Messidor-2 (Messidor-Original, 529 exams, 1,058 images), less one image.**

Reasons:
- The 529 `patient_id` values match the 529 exams in Messidor-Original exactly.
- The PNG naming matches the originals.
- 1,057 is one short of 1,058. One image was probably dropped in preprocessing. The cause is not recorded.

Grades: candidate **#1 (`google-brain/messidor2-dr-grades`)** is the only adjudicated source found. It holds grades for all 1,748 Messidor-2 images, so the PNG subset can be selected from it by file name.

Problem: the local archive's per-grade shares (44/20/27/7/2%) do not match the adjudicated overall shares for the 1,744 gradable images (58/15/20/4/2%). The gap may come from the subset or from the derived labels. The archive's labels are documented as derived, so they are unlikely to be adjudicated. This cannot be settled without the Kaggle file.

## What the user needs to do

1. Open the Kaggle page for `google-brain/messidor2-dr-grades` in a browser. Confirm the owner, file list, column names, row count, licence and last-update date.
2. Download the grades CSV yourself (this is a Kaggle download, so approve it first). Do not re-host the file: Messidor's licence forbids redistribution.
3. Filter the grades CSV to the PNG-named rows (the 1,058 Messidor-Original images). Compare that list to our 529 patient IDs and 1,057 files, and identify the one missing image.
4. Join our labels to `adjudicated_dr_grade` for the same images. Count agreements and disagreements per grade. Large disagreement confirms that our labels are derived rather than adjudicated.
5. If the Kaggle grades do not reproduce the 1,057 subset, check candidate #4 and the original Messidor PNG download from the official site.
