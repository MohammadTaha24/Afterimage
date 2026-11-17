# Messidor sources found on this PC

Read-only survey of `D:\` and `C:\Users\moht2` (excluding AppData). Nothing was changed, moved or extracted; zip archives were only read through their central directory.

## Summary

| Item | Present? | Where |
|---|---|---|
| Raw Messidor / Messidor-2 fundus images | No | No `20051*_PP.png`, `IM00*.jpg`, or `Base11`..`Base34` folders anywhere on the PC |
| Adjudicated grade file (`adjudicated_dr_grade`, `adjudicated_dme`, `adjudicated_gradable`) | No | Only derived label CSVs exist (see below) |
| Eye-pairing list (`messidor-2.xls`, IMAGES pairs) | No | Not found |
| Processed Messidor images (1,057 images, 0 to 4 grades) | Yes | `D:\recovered\DR_Messidor_Only_512_Full_Dataset\` and copies in archives |

The only Messidor material is the **processed 512 px build** (1,057 images), its label CSV, and archives of the same build. The Messidor set is therefore not available in raw form on this PC.

## Locations found

**1. Processed Messidor 512 build (folder)**
- `D:\recovered\DR_Messidor_Only_512_Full_Dataset\Res_512\`
- `Messidor_CLAHE\`: 1,057 PNG files, about 397 MB
- `Messidor_Standard\`: 1,057 PNG files, about 311 MB
- `Split_80_10_10\dataset_80_10_10.csv`
- Whole folder: 2,116 files, about 707 MB

**2. Same build as archive**
- `D:\DR_Messidor_Only_512_Full_Dataset.zip` (724,689,806 bytes)
- `D:\recovered\trans\uni_projects\302\DR_Messidor_Only_512_Full_Dataset.zip` (same size, same content)
- `C:\Users\moht2\Downloads\DR_Messidor_Only_512_Full_Dataset.zip` (same size)
- Each has 2,119 entries (2,116 Messidor files plus folders and 1 CSV), 736 MB uncompressed.

**3. Messidor inside the APTOS + Messidor external build (folder and archives)**
- `D:\recovered\DR_Aptos_Messidor_Only_Full_Dataset\Res_512\`
- Identical copy: `C:\Users\moht2\Downloads\DR_Aptos_Messidor_Only_Full_Dataset\` (9,440 files, 2.8 GB)
- `Aptos_messidor_external\` and `Aptos_messidor_external_CLAHE\`: 4,719 PNG each. Of these, 1,057 are Messidor (`messidor_*`) and 3,662 are APTOS (`aptos_*`). The folder name is misleading: it is a mixed external set, not Messidor-only.
- `Split_80_10_10\dataset_80_10_10.csv` and `Split_95_5\dataset_95_5.csv`: 4,719 rows (3,662 aptos, 1,057 messidor).
- Archives: `C:\Users\moht2\Downloads\DR_Aptos_Messidor_Only_Full_Dataset.zip` (2.94 GB, 9,445 entries), `...(2).zip` (2.94 GB, 9,446 entries). `D:\recovered\` has no copy of this zip, only the folder.

**4. Other folders**
- `D:\recovered\images-20261030T130809Z-1-001\images\`: 3,662 PNG with hash names (`000c1434d8d7.png`). These are APTOS source images, not Messidor.
- `D:\recovered\aptos2019-blindness-detection\` and `.zip`: APTOS, not Messidor.
- `D:\recovered\Inference\`, `D:\recovered\TD01 - ICT302 Project\`: no Messidor-named files.

**5. Result files (not data)**
- `C:\Users\moht2\Downloads\Messidor_External_Results.xlsx` (4,750 bytes), and the same file under `C:\Users\moht2\OneDrive\Documents\ChatGPT\...\inputs\`. These hold evaluation outputs only.

## CSV headers

- `Split_80_10_10/dataset_80_10_10.csv` (Messidor 512 build; 1,057 data rows):
  `image_path,label,patient_id,source_dataset,vlm_prompt,split`
  - Labels: 0 = 468, 1 = 207, 2 = 290, 3 = 71, 4 = 21
  - Split: train 846, val 105, test 106
  - 529 unique `patient_id` values (`messidor_p_N`), so the build pairs two images per patient ID. The basis of this pairing is not recorded.
- `Split_95_5/dataset_95_5.csv` (APTOS+Messidor build): `image_path,label,patient_id,source_dataset,vlm_prompt,split`, 4,719 rows.
- `messidor_external.csv` copies under `C:\Users\moht2\Documents\Afterimage\` (`.verification\...\data\` and `experiments\artifacts\...\data\`): `id_code,filename,file_path,image_path,label,source_dataset,split,patient_id`. These point to the processed `Aptos_messidor_external_CLAHE` folder under `C:\Users\moht2\Downloads\...`.

No adjudicated Messidor-2 grade CSV, no `adjudicated_dme` or `adjudicated_gradable` columns, and no eye-pairing spreadsheet were found. The labels in the build are derived in the processed set and cannot be checked against the original adjudication.

## Original raw path claimed by the processed builds

- The Messidor 512 build has **no manifest.json**. Its CSV contains no raw source path.
- The APTOS-side manifests (`D:\Afterimage_Experiments\datasets\aptos_512\...\manifest.json`, `aptos_224_standard\...`) record only `D:\Afterimage_Experiments\reference\aptos_split\` as their source. They do not mention Messidor. `reference\` contains only `aptos_split`.
- The code references `D:\Afterimage_Experiments\datasets\messidor_verified\`, `messidor_224_archived\` and `messidor_512`-style folders (e.g. `finetuning\Qwen3\paper_revision\evaluate_verified_messidor.py`). **None of these exist on disk**. `D:\Afterimage_Experiments\datasets\` contains only `aptos_224_standard` and `aptos_512`.
- The only Messidor path that appears in the metadata is the processed build location under `C:\Users\moht2\Downloads\DR_Aptos_Messidor_Only_Full_Dataset\Res_512\...`. This is a processed output, not a raw source.

## Not found

- Raw Messidor / Messidor-2 images, in any naming scheme
- `Base11`..`Base34` folders
- Messidor-2 grade spreadsheet or CSV with adjudicated columns
- Eye-pairing list (`messidor-2.xls` or similar)
- Messidor archives containing raw images (all archives found contain only processed PNGs)

## Action items

- To reproduce the raw-based Messidor evaluation, the original Messidor-2 download (images plus adjudicated grade file) must be obtained from its source; it is not on this PC.
- The processed 512 build (`D:\recovered\DR_Messidor_Only_512_Full_Dataset\`) is the only usable Messidor material locally, and its labels are derived rather than adjudicated.
