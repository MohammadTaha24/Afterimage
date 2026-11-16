# RetiZero training brief

For the teammate who will train the RetiZero baseline for the journal paper. Written 15 November 2026, updated the same day for the new RetiZero folder.

> If the training is done on the project PC instead, only step 1 of this brief is needed. Everything else runs from `finetuning/RetiZero/notebooks/02_retizero_runs.ipynb`.

## What we need and why

The paper compares Qwen3-VL-8B + LoRA with RetiZero on diabetic retinopathy grading (grades 0–4). All Qwen runs are finished. We now need RetiZero trained and scored on **exactly the same images**, so the two models can be compared image by image, and we need its **training time, memory use and utilization** for the efficiency comparison the reviewers asked for.

Please do not reuse the earlier RetiZero result (82.29% on 367 test images). That split had 17 test images that were byte-identical to training images, and its settings came from a hyperparameter search we are not repeating.

## What you need to do

1. **Now:** get the official pretrained RetiZero checkpoint and tell us its SHA-256 and your hardware (step 1 below).
2. Set up the files and environment (steps 2–3).
3. Train and score three seeds: 42, 43 and 44 (steps 4–5).
4. Send back the results folder and the measurements (step 6).

Three training runs and three scoring runs. Nothing is tuned and the test set is used once per seed.

## Step 1. The pretrained checkpoint (please do this first)

- Download the **official pretrained** `RetiZero.pth` using the instructions in the authors' repository: https://github.com/LooKing9218/RetiZero
- This must be the original pretrained model, not our DR fine-tuned checkpoint that the app uses. The script checks this and refuses a checkpoint that already contains a trained classifier.
- Record where you downloaded it from, the date, the file size and its SHA-256:

  ```bash
  certutil -hashfile RetiZero.pth SHA256
  ```

- Send us the SHA-256 and a one-line description of your machine: GPU model and VRAM, CPU, RAM, operating system. We need the hash to pin the checkpoint, and the hardware to decide how the timing comparison is reported.

## Step 2. Files

You will receive a package with:

- the repository (`Afterimage_Reproduction`), which contains the RetiZero model code under `RetiZero/RetiZero/` and the training code under `finetuning/RetiZero/` (`retizero_reproduction.py` and `paper_revision/train_retizero.py`);
- the prepared dataset folder `datasets/aptos_224_standard/duplicate_excluded/` (about 0.4 GB) and the matching Qwen split files, which the script uses to confirm both models see the same images;
- a resource monitor script for recording GPU and CPU use.

**Paths matter.** The dataset files store full paths and the script verifies every image by hash. On a Windows machine with a `D:` drive, unpack so that these exist and the script works unchanged:

- `D:\Afterimage_Reproduction\`
- `D:\Afterimage_Experiments\datasets\aptos_224_standard\duplicate_excluded\manifest.json`
- `D:\Afterimage_Experiments\datasets\aptos_512\duplicate_excluded\manifest.json`
- `D:\Afterimage_Experiments\assets\RetiZero.pth`

If you have no `D:` drive, or you are on Linux or Colab, tell us before starting. We will send a version with relocatable paths. Please do not edit the CSV files or the script to make it run: any change breaks the hash checks and makes the result unusable for the paper.

## Step 3. Environment

- Python 3.11 and an NVIDIA GPU.
- From the repository root, in a clean environment: `python -m pip install -r requirements.txt`. It pins torch 2.14.1+cu130 and torchvision 0.29.1+cu130.
- Do not install the old `requirements.txt` inside `RetiZero/RetiZero/`.
- Send us the output of `python -m pip freeze` with your results.

## Step 4. Check, then train

Run the check first. It verifies the checkpoint hash, the dataset, that the split matches Qwen's, and that the checkpoint fits the model. It should print `Verified APTOS duplicate_excluded (CLAHE off): 2786 train, 357 val, 361 test.` and `The pretrained checkpoint fits the encoder`.

```bash
python finetuning\RetiZero\paper_revision\train_retizero.py check --dataset D:\Afterimage_Experiments\datasets\aptos_224_standard\duplicate_excluded\manifest.json --matched-dataset D:\Afterimage_Experiments\datasets\aptos_512\duplicate_excluded\manifest.json --weights D:\Afterimage_Experiments\assets\RetiZero.pth --weights-sha256 <SHA-256 from step 1> --run-dir D:\Afterimage_Experiments\retizero_runs\teammate\matched_t42 --seed 42
```

Then train each seed. Replace `check` with `train`, and use a separate run folder per seed (`matched_t42`, `matched_t43`, `matched_t44`) with `--seed 42`, `43`, `44`.

Training needs about 12 GB of GPU memory at the default of 32 images per forward pass. With less memory, add `--physical-batch 16` (or 8) to **every** command for that seed, including `check` and `evaluate`, and tell us. Each optimizer step still uses 128 images, so the result is equivalent.

- Run one seed at a time, with nothing else using the GPU.
- Start the resource monitor before each run and stop it after (its instructions are in the package).
- If a run is interrupted, run the same command again. It resumes from the last finished epoch. Tell us which runs were interrupted.
- After the first seed finishes, send us its `train_result.json` before running the other two, so we can catch a problem early.

## Step 5. Score

For each seed, run the same command with `evaluate` instead of `train`. It scores the 361 test images with the selected checkpoint and writes `evaluation\aptos_test.predictions.csv` and `evaluation\aptos_test.summary.json`. Run the monitor for this step as well.

## The protocol (fixed; please do not change)

| Setting | Value |
| --- | --- |
| Data | APTOS 2019, duplicates removed: 2,786 train, 357 validation, 361 test |
| Test grades | 181 No DR, 35 Mild, 97 Moderate, 18 Severe, 30 Proliferative |
| Images | 224 × 224, cropped and padded, **no CLAHE** |
| Model | Pretrained RetiZero vision encoder (LoRA rank 8 branches, every weight trained) with a new 5-grade linear head |
| Seeds | 42, 43, 44 |
| Optimizer | AdamW, learning rate 7.68094483149829e-5, weight decay 1e-4 |
| Schedule | Cosine annealing over 15 epochs, early stop after 4 epochs without improvement |
| Batch size | 128 images per optimizer step, as in the original notebook, processed 32 at a time |
| Loss | Cross-entropy weighted by inverse class frequency in the training set |
| Precision | CUDA mixed precision (FP16) |
| Training augmentation | Random resized crop (scale 0.8–1.0), horizontal flip, rotation up to 180° |
| Validation and test | Resize and normalize only, no augmentation |
| Checkpoint choice | Highest validation accuracy; ties keep the earlier epoch |

These are all set inside the script. The only things you pass are the seed, the paths and the checkpoint hash. Keep the default of 128 images per step.

## Step 6. What to send back

For each of the three seeds, the whole run folder:

- `configuration.json` and `train_result.json`
- the selected `best_epoch_XX.pth`
- `evaluation\aptos_test.predictions.csv`, `evaluation\aptos_test.summary.json` and `evaluation\evaluation_manifest.json`
- the console output of the train and evaluate commands, saved as text
- the resource monitor's CSV files for training and for scoring

And once, for the whole job:

- hardware: GPU model and VRAM, driver version, CPU, RAM, operating system
- `pip freeze` output
- the checkpoint's source, download date, size and SHA-256
- wall-clock time for each train and each evaluate command
- peak GPU memory during training and during scoring
- anything that differed from this brief, however small

The checkpoints are needed because we will measure inference time for both models on one machine, so the comparison is on the same hardware.

## How we will check the results

- 361 predictions per seed, with the same image IDs in the same order as our Qwen runs.
- Each row's five probabilities sum to 1 and the predicted grade is the highest one.
- Accuracy, quadratic weighted kappa and macro ROC-AUC recompute from the predictions file to the values in the summary.
- The three configurations are identical except for the seed.

## Possible follow-ups

- **RetiZero with CLAHE**, three more seeds, to answer a reviewer who asked why CLAHE was applied to Qwen only. We would send a second dataset.
- **External test on Messidor-2**, scoring only, once the verified cohort is ready.

## Questions

Ask before changing anything. A run that follows this brief exactly, even with a disappointing score, is usable. A run with a better score and an undocumented change is not.
