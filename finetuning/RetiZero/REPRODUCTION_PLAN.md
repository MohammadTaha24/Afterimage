# RetiZero fine-tuning reproduction: plan

Written 15 November 2026. **Steps 1–6 of section 8 are built and tested; see "Build status" below.** The plan gives RetiZero the same working structure the Qwen side now has under `finetuning/Qwen3/`: pinned helper, worker scripts, a queue script, and a notebook with a pilot and a full queue, so training, evaluation and measurement can all be run from this PC.

## Build status (15 November 2026)

Built: the folder layout, the shared helper, both workers, the queue, 25 CPU tests and `notebooks/02_retizero_runs.ipynb`. The old entry points (section 7 of `021_training.ipynb` and `remaining_runs.retizero`) now point here.

Run and checked on this PC:

- **Formal replay.** `best_retizero_dr.pth` on the 367 original test images reproduces all 27 figures the original notebook printed (accuracy 0.8229, QWK 0.9097, macro AUC 0.9451, macro precision, recall and F1, and per-grade support, precision, recall and F1). Evidence: `D:\Afterimage_Experiments\retizero_runs\retizero_01\replay_comparison.json`, with predictions and hashes beside it. On the clean 361 images it scores 82.55% / 0.9101 / 0.9462 (not leak-free).
- **Pilot passed** (`retizero_01__pilot_01`), training from random weights: the interrupted run matched the uninterrupted one (same image order in every epoch, losses within 3e-4), and the scoring worker reproduced the training worker's predictions exactly.
- **Measured:** 17 ms per image at batch size 1, about 320 images per second at batch 64, 1.5 GiB peak for scoring, 6 s model load. Training at 32 images per pass peaks at 11.8 GiB (13.7 GiB on the card including other programs) and runs at about 110 images per second, so a full epoch takes about 25 s before validation and saving.

Differences from the plan as written:

- Step 2: the checkpoint was copied from the C: folder and its SHA-256 compared, instead of downloaded again. Both copies are identical (`b254d11d…6871f0`).
- The pilot uses 32 images per grade, not 16, so that one epoch has a full group of 128 and a partial one.
- The pilot does not build the full original class (that needs the text model). It checks instead that the checkpoint's two stored encoder copies are identical and that the checkpoint loads with no missing or unexpected weights.
- Memory was measured at 32 images per pass only; that fits, so no search for the largest size was run.
- The original learning-rate search is extracted to `legacy/original_learning_rate_search.csv` (25 trials, 6 finished).
- A non-finite fp16 loss skips the whole 128-image step with the same `GradScaler` backoff as the original, instead of stopping the run; skips are counted per epoch and in `train_result.json`, and more than 10% in one epoch stops the run (READINESS_CHECK M1). `train_loss` averages groups with a finite loss, where the original's would have been NaN.

Still to do: steps 7–11 (pretrained checkpoint, pilot with it, the CLAHE and Messidor builds, the full queue, the paper tables).

## 1. What exists today

Read from the files, not assumed.

**The original fine-tuning** ([ICT302_Retizero_Training_Unbalanced_Aptos.ipynb](legacy/ICT302_Retizero_Training_Unbalanced_Aptos.ipynb)):

| Item | Original notebook |
| --- | --- |
| Where it ran | Google Colab, A100 GPU |
| Data | APTOS at 224 pixels, original split 2,929 / 366 / 367 |
| Model | `CLIPRModel(vision_type="lora", R=8)` from the upstream repository; only its vision encoder is used, with every encoder weight trainable, plus a new `Linear(1024, 5)` head |
| Optimizer | AdamW, weight decay 1e-4, cosine schedule over 15 epochs, mixed precision |
| Loss | Cross-entropy weighted by inverse class frequency (a focal loss is defined but not used) |
| Batch size | 128 |
| Augmentation | Training only: random resized crop (0.8–1.0), horizontal flip, rotation up to 180° |
| Tuning | Optuna, 25 trials over learning rate 1e-5 to 2e-4; 6 finished, 19 pruned |
| Selection | Highest validation accuracy, early stop after 4 epochs without improvement |
| Chosen trial | Trial 3, learning rate 7.68094483149829e-5, validation accuracy 80.05% |
| Reported test result | 82.29% accuracy, 0.9097 QWK, 0.9451 macro AUC on the 367 test images |
| Random seed | None set anywhere |

The second notebook, `ICT302_Retizero_Training_Unbalanced_Messidor_2.ipynb`, trains on Messidor with the same code and adds Grad-CAM.

**Other pieces already in the repository:**

- `RetiZero/RetiZero/`: the upstream RetiZero code, vendored. Not to be edited.
- `RetiZero/inference_server.py`: the app's server. It loads the fine-tuned checkpoint with `strict=True` into the same class layout as the notebook.
- `RetiZero/preprocess_retizero.py`: crop, pad and resize to 224, no CLAHE.
- `train_retizero_control.py` (formerly in `finetuning/Qwen3/paper_revision/`): a check / train / evaluate script for the clean split that had never run. It is now `paper_revision/train_retizero.py`, generalised.

**Files on disk:**

| Needed | Status |
| --- | --- |
| Fine-tuned checkpoint `best_retizero_dr.pth` (1.65 GB) | Present on D: at `RetiZero/best_retizero_dr.pth`, identical to the C: copy. |
| Official pretrained `RetiZero.pth` | **Not on this PC.** The download link is in `RetiZero/RetiZero/README.md` (line 21). Needed for any new training. |
| 224-pixel APTOS, original split | Present: `D:\Afterimage_Experiments\datasets\aptos_224_standard\previous_split\clahe_off` (2,929 / 366 / 367) |
| 224-pixel APTOS, clean split | Present: `...\aptos_224_standard\duplicate_excluded\clahe_off` (2,786 / 357 / 361), same test IDs as Qwen |
| 224-pixel APTOS with CLAHE | Not built |
| Messidor at 224 pixels | Not built. Archived 512-pixel Messidor exports without CLAHE exist in `D:\recovered\DR_Aptos_Messidor_Only_Full_Dataset\Res_512\Aptos_messidor_external`; raw Messidor has not been found. |

**Checked on this PC during the self-grill (section 11) with a throwaway probe, and since confirmed by the formal replay:**

- The vendored encoder imports and runs in `afterimage_env`. It has 304,088,064 parameters and returns a 1,024-value feature per image.
- The saved checkpoint loads into the vision-only build with no missing and no unexpected keys. Its second copy of the encoder (under `retizero.`) is identical.
- **The saved checkpoint reproduces the reported result exactly** on the 367 original test images built on D:: 82.29% accuracy, 0.9097 QWK, 0.9451 macro AUC, macro F1 0.6724, and the same per-grade recall as the notebook's report (0.97, 0.70, 0.76, 0.37, 0.60).
- On the 361 clean test images the same checkpoint scores 82.55% / 0.9101 / 0.9462. That is not a leak-free number: the checkpoint was trained on the original split, which holds identical copies of 14 of those images.
- Scoring all 367 images took 4.3 seconds at batch size 64 with 1.54 GiB peak memory.

## 2. What "reproduction" means for RetiZero

The same three questions the Qwen side answered, in the same order.

1. **Does the saved checkpoint give the reported numbers?** Replay `best_retizero_dr.pth` on the original 367 test images and compare with 82.29% / 0.9097 / 0.9451. This needs no pretrained weights and can run first. The probe says yes; the formal replay saves the predictions and hashes as evidence.
2. **Does retraining give the reported numbers?** Repeat the original recipe on the original split with fixed seeds. The original had no seed and picked the best of a search, so the honest expectation is a spread around the reported value, as happened with Qwen.
3. **How does RetiZero do under the paper's protocol?** Train on the clean split with three seeds and score the same 361 images as Qwen. This is the comparison the paper reports.

Two known problems carry over from the Qwen findings and must be stated wherever the original result is quoted: 17 of the 367 original test images are byte-identical to training images, and the reported model is the best of 25 search trials.

## 3. Target layout

Mirrors `finetuning/Qwen3/`.

```
finetuning/RetiZero/
  README.md                       what lives where, which notebook to open
  REPRODUCTION_PLAN.md            this file
  retizero_reproduction.py        shared helper: model build, data, transforms, training loop,
                                  evaluation, checkpoint and identity handling. Plays the role of
                                  Qwen3/paper_reproduction.py and becomes hash-pinned once runs exist.
  paper_revision/
    train_retizero.py             worker: check / train / evaluate one run (grown from
                                  train_retizero_control.py, moved here)
    score_retizero.py             worker: score any checkpoint on any cohort, with per-image timing
    retizero_runs.py              the queue: builds jobs, shows status, runs pilot or full queue,
                                  writes the summary. Same role as Qwen3/paper_revision/final_runs.py
    test_retizero_runs.py         CPU-only tests
  notebooks/
    01_retizero_data.ipynb        verify the 224-pixel datasets; build the CLAHE and Messidor variants
    02_retizero_runs.ipynb        Setup, What is left, Pilot, Full queue, single stages, Results
  legacy/
    ICT302_Retizero_Training_Unbalanced_Aptos.ipynb
    ICT302_Retizero_Training_Unbalanced_Messidor_2.ipynb
```

Results go to `D:\Afterimage_Experiments\retizero_runs\<tag>\`, never to C: and never into `runs\`, so the Qwen evidence and the committed tables are untouched. Every GPU job takes the shared lock at `D:\Afterimage_Experiments\runs\.gpu.lock`, so a RetiZero job and a Qwen job cannot overlap.

## 4. Components

**`retizero_reproduction.py` (shared helper).**
- Builds the vision encoder with the upstream `lora(pretrained=False, R=8)` and a `Linear(1024, 5)` head, as `train_retizero_control.py` does. This avoids constructing the text encoder, so Bio_ClinicalBERT is not needed.
- Loads either kind of checkpoint: the official pretrained one (keys under `vision_model.model.`) for training, and the fine-tuned app checkpoint (keys under `img_encoder.` and `classifier.`) for replay. Always with `weights_only=True`.
- One training loop and one evaluation function for every stage, so the stages differ only in data and settings.
- Metrics come from `finetuning/Qwen3/paper_reproduction.py` (`calculate_metrics`), the same function that scores Qwen, so accuracy, QWK and AUC are computed identically for both models.
- Saves, per run: configuration with hashes of the checkpoint, datasets, source files and packages; per-epoch history; selected checkpoint; predictions with five probabilities per image; summary.

**`train_retizero.py` (worker).** `check`, `train`, `evaluate`. Resumes after an interruption from the last finished epoch. Settings that vary between stages become arguments recorded in the run configuration: split, CLAHE branch, seed, learning rate, batch size and accumulation, epochs, pilot subset.

**`score_retizero.py` (worker).** Scores a checkpoint on a cohort and measures: time per image at batch size 1 after warm-up, throughput at the training batch size, model load time, framework peak memory, and total and trainable parameter counts. When given saved predictions for the same checkpoint and cohort, it reports any disagreement.

**`retizero_runs.py` (queue).** Same behaviour as the Qwen queue: skip finished jobs, resume interrupted ones, record a failure and continue, stop if the GPU lock is held, write `summary.csv` and `summary.md`. Every job runs through the existing launcher, so each gets a log, hardware specs and a 5-second record of GPU %, VRAM, CPU % and RAM.

**`02_retizero_runs.ipynb` (notebook).** Same cells as `Qwen3/notebooks/03_final_runs.ipynb`: Setup with `TRAINING_MODE = False` and stage switches, What is left, Pilot, Full queue, one cell per stage, Results.

## 5. The queue

| Order | Stage | What it answers | Needs | Runs |
| --- | --- | --- | --- | --- |
| 1 | Checks | Are checkpoints, datasets and the environment as expected | — | CPU |
| 2 | Replay | Question 1: the saved checkpoint on the original 367 test images, and on the clean 361 | Fine-tuned checkpoint on D: | 2 scoring jobs |
| 3 | Historical repeats | Question 2: original recipe, original split, seeds 42 / 43 / 44 | Pretrained checkpoint | 3 training runs |
| 4 | Matched control | Question 3 and reviewer S4, S10: clean split, seeds 42 / 43 / 44, same 361 test IDs as Qwen | Pretrained checkpoint | 3 training runs |
| 5 | CLAHE ablation | Reviewer S9: clean split with CLAHE, seeds 42 / 43 / 44 | Pretrained checkpoint, CLAHE dataset | 3 training runs |
| 6 | Benchmark | Reviewer S14: time per image, memory, parameters, on the same PC as Qwen | A trained checkpoint | Scoring jobs |
| 7 | Archived Messidor | Generalisation, on the same archived cohort used for Qwen | 224-pixel Messidor build | Scoring jobs |
| 8 | Summary | One table, plus the files the paper tables builder will read | — | CPU |

Stages 3–5 are nine training runs. The original search finished a trial in about two to three minutes on an A100; expect longer on this card. The pilot will give the real figure before the full queue is started.

The original learning-rate search is itself tuning evidence for reviewer G2. Its trial table (learning rate and best validation accuracy for the 6 finished trials) should be extracted from the notebook output into a small CSV and cited, with the note that it was run on the original split.

## 6. Protocol decisions

Each needs a decision before the build. My recommendation is first.

1. **Batch size.** The original used 128 on a 40 GB GPU; the existing control script uses 8 with the same learning rate. A sixteen-fold smaller batch at the same learning rate is a different training recipe. Recommendation: keep the effective batch at 128 by accumulating gradients over the largest batch that fits in 16 GB (found by the pilot), for both the historical repeats and the matched control. The model has no batch normalisation, so accumulation is a close match.
2. **Learning rate.** Fix it at the trial 3 value. Do not rerun the search: it would tune on a split with known leakage and add a second selection step.
3. **Seeds.** 42, 43 and 44, matching the Qwen trainer seeds, applied to weight initialisation of the head, data order and augmentation. Record that the original run had none.
4. **Checkpoint selection.** Highest validation accuracy, ties to the earlier epoch, as in the original. The paper must state that Qwen selects on validation loss.
5. **Model construction.** Vision encoder only. The probe confirmed it is the original model: the saved checkpoint loads strictly and reproduces the reported metrics.
6. **Precision.** Mixed precision for training and scoring, as in the original and the app.
7. **CLAHE for RetiZero.** Run it (stage 5). It is three short runs and turns reviewer S9 from an explanation into a result. Apply CLAHE as the last preprocessing step on the 224-pixel image, the same position it has in the Qwen pipeline. State in the paper that the 8×8 tile grid then covers 28-pixel tiles instead of 64-pixel ones.
8. **Messidor at 224 pixels.** Raw Messidor is not available, so the only option today is to resize the archived 512-pixel non-CLAHE export to 224. That differs slightly from preprocessing raw images directly to 224. Recommendation: build it, label it as derived from the archived export, and replace it when the verified cohort exists.

## 7. Pilot

Run from the notebook before the full queue, into a separate pilot folder.

- **Replay pilot:** the fine-tuned checkpoint on 12 test images; outputs identical through both model constructions (decision 5).
- **Memory pilot:** find the largest batch that trains without running out of memory, and set the accumulation steps from it.
- **Training pilot:** one epoch on a balanced subset (16 images per grade), validation on 2 per grade, checkpoint saved, then scored on 12 test images.
- **Resume pilot:** interrupt after the first epoch of a two-epoch run and confirm it resumes and finishes with the same result as an uninterrupted run.
- **Scoring pilot:** timing and parameter counts produced; predictions pass the pinned metric function.

The notebook prints **Pilot passed** only when every check holds.

## 8. Build order

| # | Step | Done when |
| --- | --- | --- |
| 1 | Move the two original notebooks to `legacy/` with `git mv`; add `README.md` | Nothing else referenced them; repository tests still pass |
| 2 | Bring the fine-tuned checkpoint to D: with `download_model_assets.py` and record its SHA-256 | File at `RetiZero/best_retizero_dr.pth`, hash recorded |
| 3 | Write `retizero_reproduction.py` and its CPU tests (key mapping, transforms, split checks, metrics hand-off) | Tests pass |
| 4 | Write `score_retizero.py`; run the replay | Replay reproduces 82.29% / 0.9097 / 0.9451 on the 367 images, or the difference is explained |
| 5 | Move `train_retizero_control.py` into `paper_revision/train_retizero.py` and generalise it; update section 7 of `021_training.ipynb` and `remaining_runs.retizero` to point at the new notebook | Old entry points removed or redirected; `check` passes |
| 6 | Write `retizero_runs.py`, its tests, and `02_retizero_runs.ipynb` | Setup, What is left and the pilot plan run with `TRAINING_MODE = False` |
| 7 | Obtain the official pretrained checkpoint, place it at `D:\Afterimage_Experiments\assets\RetiZero.pth`, pin its SHA-256 in Setup | `check` verifies it |
| 8 | Run the pilot on the GPU; fix what it finds | Pilot passed |
| 9 | Write `01_retizero_data.ipynb` and a small dataset builder; build the 224-pixel CLAHE dataset and the Messidor build. The existing builder (`D:\Afterimage_Experiments\code\prepare_aptos.py`) refuses CLAHE at 224 pixels, so the new builder imports its functions and must first reproduce the existing non-CLAHE images byte for byte. | Manifests with hashes; same IDs and order as the non-CLAHE build; byte-identical check passed |
| 10 | Run the full queue | Summary written; nine training runs and all scoring jobs done |
| 11 | Extend `finetuning/paper_tables/build_paper_tables.py` to read RetiZero runs and add the paired comparison with Qwen on the 361 images | RetiZero rows in the tables with the same cross-checks |

Steps 1–6 need no pretrained checkpoint and can be built and tested now. Step 7 is the only external dependency.

## 9. Risks

- **The pretrained checkpoint.** Until it is on disk, only the replay can run. The script already refuses a checkpoint that contains a trained classifier, which protects against using the fine-tuned one by mistake.
- **Checkpoint provenance.** No file records that the app's checkpoint is the trial 3 model. The exact match of every reported metric is strong evidence that it is. The formal replay should record its SHA-256 and compare the C: copy with the copy `download_model_assets.py` fetches.
- **Launcher code is outside git.** `D:\Afterimage_Experiments\code` (the pinned runner, the recovery script, the notebook launcher and the dataset builder) is not under version control. A snapshot with hashes should be committed to the repository. This affects the Qwen runs as much as RetiZero.
- **Retraining will not land on 82.29%.** Different GPU, no original seed, and the reported model was the best of a search. Report all seeds.
- **Pinned files.** Once runs exist, `retizero_reproduction.py` and the workers are part of their identity and must not be edited, as on the Qwen side. Get them right in the pilot.
- **`.pth` files are gitignored and large.** Checkpoints live under `D:\Afterimage_Experiments\retizero_runs\` and need the same backup as the Qwen runs.
- **Vendored code.** `RetiZero/RetiZero/` is imported as it is; its own old `requirements.txt` is never installed.

## 10. Open questions

1. Effective batch 128 through accumulation (recommended), or keep batch 8 as the current control script has it?
2. Run the historical repeats on the original split (stage 3), or only the replay and the matched control?
3. RetiZero with CLAHE (stage 5): yes or no?
4. Who downloads the pretrained checkpoint, and is the teammate still involved? If you run everything here, [docs/paper/RETIZERO_TRAINING_BRIEF.md](../../docs/paper/RETIZERO_TRAINING_BRIEF.md) reduces to its first step.
5. Should a new RetiZero checkpoint be exportable in the app's format, so the app can serve it?
6. Does the paper need Grad-CAM for RetiZero (the Messidor notebook has it)? Qwen has no equivalent, so it would be a RetiZero-only figure.

## 11. Self-grill

Run with the grill-me protocol against this plan. Questions that the code or a quick run could answer were answered that way; the rest carry a recommendation.

```
Design tree
├── [✓] Is the plan's model the original model
│   ├── [✓] Q1  Vision-only build versus the original class
│   ├── [✓] Q2  Does the environment run the vendored code
│   └── [✓] Q3  What is actually trained
├── [✓] Question 1: the saved checkpoint
│   ├── [✓] Q4  Does it reproduce the reported result
│   ├── [✓] Q5  Which checkpoint is it
│   └── [✓] Q6  What exactly the replay must match
├── [✓] Questions 2 and 3: retraining
│   ├── [✓] Q7  Batch size and memory
│   ├── [✓] Q8  Seeds, TF32 and data loading
│   ├── [✓] Q9  Was the learning rate search meaningful
│   └── [✓] Q10 Are the historical repeats worth running
├── [✓] Data
│   ├── [✓] Q11 Can the CLAHE dataset be built with the existing builder
│   ├── [✓] Q12 Is CLAHE at 224 pixels the same operation
│   └── [✓] Q13 Messidor for RetiZero
├── [✓] Comparison with Qwen
│   ├── [✓] Q14 Parameters and what "LoRA" means in the paper
│   ├── [✓] Q15 Is the protocol fair to both models
│   └── [✓] Q16 Inference benchmark
└── [✓] Delivery
    ├── [✓] Q17 What moves and what could break
    ├── [✓] Q18 What is still outside version control
    └── [✓] Q19 What still blocks training
```

**Q1. Is the vision-only build the same model as the original notebook's class?**
Yes, verified. The original wraps `CLIPRModel` and uses only `vision_model.model`, which the upstream code builds with the same `lora(...)` function the plan uses. The saved checkpoint's 390 encoder tensors load with no missing and no unexpected keys.

**Q2. Does the vendored code even run in `afterimage_env`?**
Yes, verified. It had never been imported in this environment; the earlier test of the control script stopped at the missing checkpoint before reaching it. The encoder builds and a forward pass returns a plain 1,024-value tensor per image, which the `Sequential(encoder, head)` layout needs.

**Q3. What is actually trained?**
All of it. The upstream wrapper freezes the base and leaves only 786,432 LoRA weights trainable, but the notebook then sets every encoder weight trainable. So this is full fine-tuning of 304,088,064 encoder weights plus a 5,125-weight head, with LoRA branches present. Describing it as "LoRA rank 8" alone would be wrong.

**Q4. Does the saved checkpoint reproduce the reported result?**
Yes, exactly, on the 367 test images rebuilt on D: (figures in section 1). This also shows the D: 224-pixel preprocessing is equivalent to what the Colab run used.

**Q5. Which checkpoint is it?**
The app's copy from the C: folder. Nothing records its origin, but it matches trial 3's reported test metrics to four decimals, including per-grade recall. Record its SHA-256 in the replay.

**Q6. What should the formal replay match?**
More than three numbers. The notebook also prints macro precision 0.6661, macro recall 0.6796, macro F1 0.6724 and a per-grade report with supports 181 / 37 / 100 / 19 / 30. The replay should check all of them. The original confusion matrix exists only as a picture.

**Q7. Batch size: is 128 reachable on 16 GB?**
Unknown for training; the pilot measures it. Inference at batch 64 used 1.54 GiB, so training at a physical batch of 32 or more is plausible. Recommendation unchanged: effective batch 128 through accumulation.

**Q8. Seeds, TF32 and data loading?**
The original set no seed, enabled TF32 because it detected an A100, and used four loader workers. Here: fixed seeds, TF32 set explicitly and recorded, loader workers set to 0 as on the Qwen side (Windows), a seeded generator for shuffling. Bit-for-bit agreement with the Colab run is not expected and should not be promised.

**Q9. Was the original learning-rate search meaningful?**
Only weakly. The six finished trials span learning rates from 2.8e-5 to 1.26e-4 and reach validation accuracy between 78.4% and 80.1%. With 366 validation images, one image is 0.27%, so trial 3 won by one image over two others. The search shows the model is insensitive to learning rate in that range, which is useful evidence for reviewer G2, and it shows the chosen value is not special.

**Q10. Are the historical repeats worth running?**
Yes. They are short, and they answer how typical 82.29% is. Without them the paper has one unseeded best-of-search number for RetiZero next to three-seed means for Qwen.

**Q11. Can the CLAHE dataset be built with the existing builder?**
No. `prepare_aptos.py` rejects CLAHE at 224 pixels, and the control script rejects any dataset that is not CLAHE-off. Both need the generalisation described in build steps 5 and 9. The builder is also outside git (Q18).

**Q12. Is CLAHE at 224 pixels the same operation as for Qwen?**
Same code and settings, different scale: an 8×8 grid gives 28-pixel tiles at 224 pixels and 64-pixel tiles at 512. Applying it last on the model's own input is the natural reading of the reviewer's question. The alternative, enhancing at 512 and then shrinking, would double the runs; not recommended unless a reviewer asks.

**Q13. What about Messidor for RetiZero?**
The Messidor notebook trained on Messidor and tested on its own 10% split, so its 58.49% is not an external result. For the paper both models must be trained on APTOS and scored on the same Messidor cohort. The saved checkpoint can be scored on a 224-pixel build of the archived cohort in seconds once that build exists.

**Q14. How do the parameter counts compare with Qwen?**
RetiZero: about 304.1 million parameters, all trainable. Qwen: 8,767 million in the base, 51.3 million trainable at rank 16. So RetiZero is about 29 times smaller but trains about 6 times more weights. The efficiency table needs both columns, and the text must not call Qwen the model with more trainable parameters.

**Q15. Is the protocol fair to both models?**
There are three asymmetries to state: RetiZero's learning rate was searched and Qwen's was inherited (the Qwen learning-rate study reduces this); RetiZero selects on validation accuracy and Qwen on validation loss; RetiZero uses image augmentation and a weighted loss, Qwen uses a weighted sampler and no augmentation. None needs changing, all need a sentence.

**Q16. How is inference compared?**
Batch size 1 after warm-up for both models on this PC, with preprocessing time reported separately. RetiZero's batch throughput is also reported, since batching is how it would be used.

**Q17. What moves, and what could break?**
`train_retizero_control.py` moves out of the Qwen folder; section 7 of `021_training.ipynb`, `remaining_runs.retizero` and `show_checklist` refer to it and must be redirected and retested. No saved run pins its hash, so moving it is safe. The two original notebooks move to `legacy/`; nothing imports them.

**Q18. What is still outside version control?**
`D:\Afterimage_Experiments\code`: four scripts including the pinned Qwen runner and the dataset builder. Added to the risks.

**Q19. What still blocks training?**
Only the official pretrained checkpoint. Everything up to and including the replay, the pilot plan and the dataset builds can be done without it.
