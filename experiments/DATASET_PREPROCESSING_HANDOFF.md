# Dataset preprocessing handoff for the CLAHE comparison

Prepared 10 November 2026. This is a specification and acceptance checklist for the researcher preparing the data. It does not authorize or perform training, alter existing experiment artifacts, or establish a completed CLAHE comparison.

The rank-16, rank-8 and rank-32 CLAHE-on runs are complete under `experiments/artifacts/qwen_four_run_001`. Their preflight verified the shared on-branch coverage and integrity. It explicitly skipped original-image lineage and off/on pairing because CLAHE-off was excluded. The completed rank study should be preserved.

## Requested work

Besides the CLAHE toggle, the original request included original-image availability/mapping, consistent fundus cropping, square padding, 512×512 resizing, channel-order tracing, label/split integrity, and ensuring any augmentation is confined to training. The current study deliberately introduces no image augmentation. These are the common preprocessing and verification tasks needed for a valid matched comparison, not a request to invent additional preprocessing experiments.

Preparing data is one part of the fourth condition. After acceptance, a fresh rank-16 CLAHE-off adapter still needs training, APTOS-validation checkpoint selection and evaluation on both frozen test cohorts. Feeding off images to an adapter trained on on images would be a different inference-only experiment.

## Frozen data contract

Project root: `C:\Users\moht2\Documents\Afterimage`.

Authoritative manifest:

`finetuning/Qwen3/artifacts/paper_reproduction/data/4fda605471d5430c3733/dataset_manifest.json`

Use its `train.csv`, `val.csv`, `test.csv` and `messidor_external.csv` exports to retain exact image IDs, labels and membership. Grade order is 0 Normal/No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR.

| Cohort | Images | Counts for grades 0, 1, 2, 3, 4 |
| --- | ---: | --- |
| APTOS train | 2,929 | 1444, 296, 799, 154, 236 |
| APTOS validation | 366 | 180, 37, 100, 20, 29 |
| APTOS test | 367 | 181, 37, 100, 19, 30 |
| Messidor external | 1,057 | 468, 207, 290, 71, 21 |
| Total | 4,719 | Preserve every cohort above |

The older combined mapping file under `Res_512/Split_80_10_10/dataset_80_10_10.csv` has different APTOS split assignments. It can provide ID/filename/label bindings; its split column must not replace the recovered manifest's membership. Do not regenerate train/validation/test splits. Messidor stays external to training and checkpoint selection.

Existing processed folders:

```text
C:\Users\moht2\Downloads\DR_Aptos_Messidor_Only_Full_Dataset\Res_512\Aptos_messidor_external_CLAHE
C:\Users\moht2\Downloads\DR_Aptos_Messidor_Only_Full_Dataset\Res_512\Aptos_messidor_external
```

Inspect and preserve these files. If repair or regeneration is needed, produce a separate candidate folder first. Do not overwrite the frozen CLAHE-on images or the label/split exports used by the completed ranks.

## Original-image index

Return a UTF-8 CSV named, for example, `original_image_index.csv`:

```csv
source_dataset,image_id,original_path,is_original
aptos,000c1434d8d7,D:\DR originals\APTOS\000c1434d8d7.png,true
messidor,messidor_messidor_p_0_20051109_57451_0400_PP,D:\DR originals\Messidor\20051109_57451_0400_PP.jpg,true
```

The rows above are schema examples only. Include all 4,719 frozen IDs exactly once. Use `aptos` / `messidor` and the authoritative exports' `id_code` values for `image_id`. Preserve case and leading zeros. The Messidor IDs in this workflow already include prefixes; map them explicitly to the original files. Do not infer identity by row position or strip patient prefixes heuristically.

Paths may be absolute or relative to the index file. Each original path must be unique and independent of the two processed branches. `is_original=true` asserts independent, unenhanced source-image provenance; processed images must not be represented as originals. Record original SHA-256 values and any unavailable originals explicitly.

## Common deterministic processing

Use the same implementation and recorded package versions for both branches:

1. Read the original with OpenCV in **BGR**.
2. Convert BGR → **RGB**. Convert RGB → grayscale and obtain an **Otsu** binary foreground mask.
3. Crop to the mask's bounding box. If the mask is empty, retain the full image.
4. Add black padding to make a square. Split any odd padding difference as floor/remaining pixels, matching the existing worker.
5. Resize to **512×512**. Use `INTER_AREA` when shrinking and `INTER_CUBIC` otherwise.
6. Convert RGB → **BGR** for OpenCV output.
7. For **CLAHE off**, save this common processed image with **zero** CLAHE applications.
8. For **CLAHE on**, convert the common BGR image to **LAB**, apply CLAHE to **L only**, then convert LAB → BGR and save. Use `clipLimit=2.0`, `tileGridSize=(8,8)`, and exactly **one** CLAHE application.

Save lossless **RGB PNG** files as interpreted by PIL, with identical expected filenames in both branches. Raw originals may use their native supported image formats. Keep crop, padding, resolution, interpolation, image identity, labels and cohort membership identical between branches.

The current pinned recipe is implemented by `standard_from_original`, `clahe_once` and `pairing` in [qwen_experiment_worker.py](qwen_experiment_worker.py). `Qwen3/preprocess_qwen.py` supplies relevant RGB/LAB processing context. Both explicit RGB-based and BGR-based LAB routes can be correct when caller conversions agree; the presence of both names is not itself evidence of a color bug.

The completed batch recorded OpenCV **5.0.0** and Pillow **13.0.0**. Record exact producing versions and code hashes. Do not upgrade the pinned experiment environment or modify the experiment worker as a way to regenerate data. Different versions/recipes may produce different pixels and need explicit comparison.

## Augmentation and sampling

- Introduce **no image augmentation** in the matched preprocessing handoff.
- Keep validation, test and Messidor external images deterministic and unaugmented.
- The existing inverse-frequency weighted sampler operates only in the training loader; it is sampling, not image augmentation or a new image dataset.
- Any future augmentation study must be designed separately, restricted to training, and kept out of this CLAHE-only comparison.
- Do not run old preprocessing notebooks wholesale: they contain 224-resolution variants, split-generation cells, and Messidor internal-split variants that do not match the current frozen external-cohort protocol. Extract/review the relevant functions instead.

## Acceptance checklist

- [ ] Independent originals are available for all 4,719 IDs, with a complete unique-path index. Any unavailable originals are itemized as blockers in a partial handoff.
- [ ] Both branches have exactly the same 4,719 expected filenames and ID bindings, with no added augmentation variants or silent row removal.
- [ ] Labels, all four cohort counts/class counts and recovered APTOS membership remain unchanged.
- [ ] No duplicate image IDs/paths, train/validation/test image overlap or patient-group leakage; Messidor remains external.
- [ ] Every output fully decodes and is a lossless RGB PNG at 512×512.
- [ ] For every image, the common crop/pad/resize processing of the independent original equals the off pixels exactly.
- [ ] For every image, one specified LAB-L CLAHE application to the off pixels equals the on pixels exactly.
- [ ] BGR/RGB/LAB caller/return/save order is documented and checked; training/evaluation loaders do not apply CLAHE a second time.
- [ ] Original and output hashes, exact source/code version, package versions, and per-image checks are saved.
- [ ] Mismatches, unavailable provenance and corrupt/missing files are explicitly reported; labels or membership are not changed to make QC pass.
- [ ] Existing on-branch model-input pixels are compared with the proposed pair, and control-reuse versus a new matched control is decided explicitly.

## Deliverables to return

1. Verified off/on folders or a clear report that the frozen on folder already matches the accepted pair. Supply full local paths.
2. Complete `original_image_index.csv` with the schema above.
3. Per-image QC such as `pairing_checks.csv`, with source dataset, recovered image ID, filename, original path/hash, output hashes, original-to-off equality and once-CLAHE off-to-on equality.
4. A concise QC report covering counts/class counts, duplicates, overlap, missing/corrupt files, channel order, CLAHE count and all mismatches.
5. The producing preprocessing code/notebook cell references and SHA-256/version records.
6. References to the unchanged authoritative split/label files and a statement describing which current artifacts can be reused.

## Control reuse and experiment handback

If decoded on-image pixels match the completed rank-16 control's inputs, retain the existing frozen on files and their hashes and document the correspondence. PNG compression/metadata differences can alter file hashes without altering model-input pixels; document that distinction rather than overwriting the frozen files.

If decoded on-image pixels change, preserve the three completed rank-study results and prepare a **new paired rank-16 CLAHE-on/CLAHE-off study** with a newly matched on control. Do not attribute differences between different preprocessing pipelines solely to CLAHE.

The recovered manifest states that original historical preprocessed bytes were unavailable for comparison. Do not claim byte-identical reconstruction of historical Retraining 2 without additional evidence.

After accepted data is returned, the experiment owner can set `INCLUDE_CLAHE_OFF=True`, supply the original-index path and rerun preflight in the pinned setup. The remaining work is fresh rank-16 off training, validation-based checkpoint selection and both-cohort evaluation. No such work was started as part of this handoff.
