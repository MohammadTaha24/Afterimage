# RetiZero training readiness check (17 November 2026)

Read-and-test review of `notebooks/02_retizero_runs.ipynb` and the code behind it, done the day before the first training runs. No code was edited, nothing was committed, and no GPU job was run. The scripts and logs are in `.verification/retizero_readiness/` (`verify.py`, `plan.py`, `*.log`).

## Verdict

**Ready, once `RetiZero.pth` is on disk.** The recipe matches the original notebook. The 224-pixel datasets match the Qwen test IDs. All 25 CPU tests pass. The pretrained key mapping loads every encoder tensor in a simulation. Disk space and resume are fine. No code problem blocks tomorrow's runs.

The only blocker is outside the code: `D:\Afterimage_Experiments\assets\RetiZero.pth` is missing and its SHA-256 is not pinned. Six training runs (historical and matched) and the three benchmark scorings become runnable once it is in place. The three CLAHE runs and the four Messidor scorings still wait for datasets that have not been built (plan step 9).

The one medium code issue, M1, is resolved (17 November 2026), before any full run; L5 was fixed with it. The pilot and replay folders record the earlier `retizero_reproduction.py` hash, as expected.

## What was checked

### 1. Recipe against `legacy/ICT302_Retizero_Training_Unbalanced_Aptos.ipynb` (cell 8)

| Item | Original | This code | Match |
|---|---|---|---|
| Optimizer | `AdamW(model.parameters(), lr, weight_decay=1e-4)` | `retizero_reproduction.py:439-440` | yes. The original's extra text-model parameters get no gradient, so AdamW skips them. |
| Learning rate | Optuna trial 3: 7.68094483149829e-05, best validation accuracy 0.8005 | `RECIPE`, passed as `repr()` | yes. Confirmed in the notebook output and `original_learning_rate_search.csv`. |
| Schedule | `CosineAnnealingLR(T_max=15)`, stepped once per epoch | `:441`, `:491` | yes |
| AMP | `autocast('cuda')` + `GradScaler('cuda')` | `:410`, `:442`, `:483-484` | yes. One scaler step per group of 128, and `unscale` happens inside `step`. |
| Loss | `CrossEntropyLoss(weight=N/(5·count))` | `class_weights_for` `:387-393` | yes. Clean weights are [0.388, 2.087, 0.764, 3.980, 2.604]. |
| Augmentation | RandomResizedCrop(224, 0.8–1.0), horizontal flip, rotation 180°, ImageNet normalisation. Evaluation uses Resize(224) only. | `:227-240` | yes, identical |
| Trainable weights | Whole vision encoder plus a new `Linear(1024, 5)` | `build_model` `:267-290`. 304,093,189 parameters, all trainable. | yes |
| Selection / early stop | Strictly higher validation accuracy, patience 4, check after the update | `:509-515`, check at the top of the next epoch (`:468`) | yes. The same number of epochs runs, and ties keep the earlier epoch. |
| TF32 | On (A100) | On (`configure_backend`) | yes |
| Batch | 128, `shuffle=True`, `drop_last=False` | Groups of 128 in four passes of 32. The partial last group is kept (22 steps per epoch on the clean split, last group 98; 23 on the original split, last group 113). | yes, see below |

**Accumulation equals a true 128-image weighted CE, including the last partial group.** Each piece adds `Σ w_y·CE / Σ_group w_y`, where the denominator comes from the group's own grades (`:406`). PyTorch's weighted `mean` reduction is exactly `Σ w_y·CE / Σ w_y`, so the pieces sum to the full-group loss and gradient. A shorter last group uses its own denominator, exactly as the original's short last batch did. ViT-L has only LayerNorm and no batch statistics. Numerical check (`verify.py` §1): a full clean epoch of 22 groups (last group 98) in float32 differs from one-shot `cross_entropy(weight=…)` by at most 2.7e-7 in loss and 7.5e-8 in gradient. The unit test `test_accumulated_gradient_equals_one_weighted_batch` (11 rows, pieces of 4) also passes.

### 2. Datasets against the Qwen 361 test IDs

- `checks()` verifies each split's CSV hash, the audit hash, and the per-row path and grade bindings. For both `previous_split` (2929/366/367) and `duplicate_excluded` (2786/357/361), the ordered IDs and grades of all three splits are identical to the Qwen 512-pixel builds.
- The RetiZero 224 test CSV has the same ordered (ID, grade) list as the Qwen 512 CLAHE-off and CLAHE-on test CSVs, and as all 42 Qwen 361-row prediction files under `D:\Afterimage_Experiments\runs\full_*duplicate_excluded*`. This includes the `full_bf16_r16_on_…_paper_controls_01` runs.
- The clean test IDs are a subset of the original test split, with 0 in the original train or validation split. The learning rate was chosen on the original validation split, so it never saw these test images by ID.

### 3. CPU tests

`python -X utf8 -B -m unittest test_retizero_runs -v` in `paper_revision/`: **25 tests, all OK** (2.7 s).

### 4. Read-only checks and plan (`plan.py`, the notebook's Section 2)

```
[ok] Fine-tuned checkpoint matches its pinned SHA-256
[--] Pretrained RetiZero.pth missing
[ok] Dataset 'original' 2929/366/367, 'clean' 2786/357/361: same ordered IDs and grades as Qwen
[--] Dataset 'clean_clahe' and cohort 'aptos361_clean_clahe' not built
[--] Cohort 'messidor_archived' not built
[ok] GPU lock is free
2 of 18 jobs done, 16 waiting. About 2.0 h estimated.
```

| Waiting job(s) | Waits for |
|---|---|
| historical_t42/43/44, matched_t42/43/44 | `RetiZero.pth` and a pinned SHA-256 |
| clahe_t42/43/44 | `RetiZero.pth` **and** `datasets\aptos_224_clahe\duplicate_excluded\manifest.json` (not built) |
| matched_t4x__aptos361_clean (benchmark) | its matched training run |
| saved_checkpoint__messidor_archived | `datasets\messidor_224_archived\clahe_off\test.csv` (not built) |
| matched_t4x__messidor_archived | its matched run **and** the Messidor build |

Done: both replays (367 original: every reported figure reproduced; 361 clean). Pilot `pilot_01`: all 12 pilot checks passed, from random weights. Its recorded code identity is still identical to the current files, so the evidence is not stale. With `pilot_tag = "pilot_02"` and the SHA set, the pilot's training jobs switch to the pretrained weights. `random_init` is True today only because the file is missing.

### 5. Once `RetiZero.pth` is supplied

- **Key matching.** Upstream `Finetuning.py` loads `RetiZero.pth` with `strict=True` into `iden_modules.CLIPRModel`, the same class the original notebook used. Its vision keys are therefore `vision_model.model.*`. A simulated CLIPRModel state dict was built from the `retizero.*` copy inside the app checkpoint: 592 tensors, of which 390 are `vision_model.model.*`. `pretrained_state` plus `load_state_dict` gave **0 missing, 0 unexpected**, every encoder tensor copied exactly, and only the head (`1.weight`, `1.bias`) left fresh. The model has no buffers, so the text-side `position_ids` buffer the original had to delete is irrelevant here.
- **Early refusals (good).** A `module.`-prefixed (DataParallel) file and a `{'model': …}` wrapper both fail with "No pretrained vision encoder tensors". The fine-tuned app checkpoint fails with "This is a fine-tuned DR checkpoint". A missing file, an unset or malformed SHA, or a wrong SHA stops the worker before CUDA is touched (`train_retizero.py:27-39`).
- **SHA handling.** `checks()` prints the file's SHA-256 for you to paste when the file exists and the setting is empty (`retizero_runs.py:242-243`). Both places compare `.strip().lower()`. A file that is present but has no SHA leaves the full runs `waiting` (blocked), not failing. There is no upstream-published hash, so the pinned value only proves the file has not changed since you downloaded it. It does not prove the download was correct (L4).
- **What could fail late rather than early:**
  - Each run re-hashes the file and reloads it at the start. A key mismatch would fail at the start of every run (minutes, not hours). The queue does not stop after the first failure, though; it logs a failure for each of the six runs (L2). The pilot and the `check` command below catch this first.
  - `torch.load(..., weights_only=True, mmap=True)` (`retizero_reproduction.py:294`) refuses a legacy non-zip pickle, and a truncated Drive download or HTML error page. Both fail at load time, which is early.
  - A non-finite fp16 loss mid-training now skips that step and is counted (M1, resolved); the run stops only if more than 10% of an epoch's steps are skipped.

### 6. Resume, disk and time

- **Resume.** `checkpoint.pth` is written atomically after every epoch, with the model, optimizer, scheduler, scaler, CPU, GPU and order RNG, history, best epoch and best SHA. On resume the configuration must be equal, including code hashes, package versions and the GPU, and the best-epoch file's hash is re-checked. The pilot proved a stop after epoch 1 and a resume: identical per-epoch image order, losses within 2.7e-4 (not bitwise, see L1). A crash during `evaluate` is recovered by rerunning; `train` sees `train_result.json`, verifies it and moves on. Rerunning the full-queue cell is the resume step. Do not edit any file under `finetuning/RetiZero`, `finetuning/Qwen3/paper_reproduction.py` or the four vendored sources, and do not change packages, while a run is unfinished. Any of these makes resume refuse.
- **Disk.** D: is a Samsung 990 PRO NVMe with **415.6 GiB free**. A best-epoch file is 1.13 GiB. The resume checkpoint (model plus two Adam moments) is about 3.4 GiB and is deleted at the end of each run. Peak per run is about 9 GiB: old best, new best, `checkpoint.pth` and its `.partial`. Kept per run is about 1.2 GiB, so the nine runs keep about 11 GiB. This is not a constraint.
- **Time.** The pilot measured 60–111 images/s on the RTX 5070 Ti (160-image subset, warm-up heavy) and a peak of 11.8 GiB reserved of 15.9 GiB. On the full clean split that is about 25–46 s of training per epoch, plus a few seconds of validation and NVMe saves. That is roughly 10–15 min per 15-epoch run, fewer with early stopping, which agrees with the queue's 12-minute guess (`retizero_runs.py:40`). Expect for **tomorrow's runnable set** (pilot_02 plus 6 training runs plus 3 benchmarks) **about 1.5–2 h**, and for the **full 18-job queue** about 2–2.5 h once the CLAHE and Messidor builds exist. Re-read the real `train_images_per_second` after the first run.

### 7. Comparability with Qwen

Same as Qwen: ordered train, validation and test IDs and grades (361 test), seeds 42/43/44, and the metric function (`finetuning/Qwen3/paper_reproduction.py`, hash pinned in every run). The differences below must be stated in the paper. None of them prevents running.

- **Preprocessing.** The Qwen paper runs (`full_bf16_r16_on_duplicate_excluded_*_paper_controls_01`) use **CLAHE on, 512 px**. The RetiZero `matched` stage is **CLAHE off, 224 px**. Each model uses its own native preprocessing, as in the app. The like-for-like CLAHE comparison (`clahe` stage, reviewer S9) waits for the 224-pixel CLAHE build (M2).
- **Model selection.** Qwen keeps the epoch with the lowest validation loss (`metric_for_best_model="eval_loss"`, `paper_reproduction.py:50`, `:549`). RetiZero keeps the highest validation accuracy, with patience 4, as its original did. Each is faithful to its own original. They are different protocols.
- **Precision and tuning.** Qwen is bf16 or 4-bit. RetiZero uses fp16 AMP. The RetiZero learning rate comes from a 25-trial search on the original split's validation set. It is not re-searched on the clean split.
- **Run-to-run determinism.** Seeds fix the order and augmentation, but GPU kernels are not bitwise deterministic (L1). Seed spread therefore includes a little kernel noise. The same is true for Qwen.

## Problems by severity

No fixes were applied.

### Blocking for tomorrow (input, not code)

**B1. `RetiZero.pth` is missing and its SHA-256 is unset.** `D:\Afterimage_Experiments\assets\RetiZero.pth`, notebook Setup `SETTINGS["pretrained_sha256"] = ""`. Fix: download it from the link in `RetiZero/RetiZero/README.md` to D: (not C:), then follow the steps below.

### Medium

**M1 (resolved 17 November 2026). A non-finite fp16 training loss aborted the run, where the original would have skipped the step.** Fix: `training_step` in `retizero_reproduction.py` skips the whole 128-image step when any piece's loss is non-finite (no `optimizer.step`, so no partial update), backs off the `GradScaler` scale and resets its growth tracker exactly as the original's `scaler.step`/`scaler.update` did on non-finite gradients, and leaves the per-epoch scheduler untouched. Skips are counted per epoch in `history` (`skipped_steps_nonfinite_loss`), which `checkpoint.pth` carries through resume, and totalled in `train_result.json`. More than 10% skipped in one epoch stops the run with a clear error. CPU tests in `paper_revision/test_retizero_runs.py`. Original finding: `retizero_reproduction.py:481-482` raises `ValueError` when the accumulated loss is inf or NaN. The original computed the same AMP loss and let `GradScaler` skip the step. This code would therefore stop on a forward overflow that the original survived. Resume restores every RNG state and so replays the same batch, which likely makes the run fail the same way again. It is the only late, non-recoverable failure found. It is unlikely with pretrained weights (none in the pilot), but `retizero_reproduction.py` is frozen once the first full run exists. Proposed fix, to decide **before** the first full run: when `value` is not finite, call `optimizer.zero_grad()` and `scaler.update()` (backing off the scale), count the step as skipped in `history`, and raise only after several consecutive skips (for example 5). Alternatively accept the risk and write that decision into `REPRODUCTION_PLAN.md`.

**M2. The headline comparison is CLAHE-off RetiZero against CLAHE-on Qwen, and the CLAHE control cannot run yet.** `retizero_runs.py:36-38`, `:57-66`. `aptos_224_clahe` is not built (plan step 9, `01_retizero_data.ipynb` does not exist yet). The code is not wrong. Fix: build the CLAHE dataset (with the byte-identical check against the non-CLAHE build) before the paper claims a preprocessing-matched result, and state the 224/512 and CLAHE difference and the selection-protocol difference in the paper.

### Low

- **L1. Resume and reruns are close but not bitwise.** Pilot loss difference 2.7e-4 (`pilot_checks.json`), although `cudnn.deterministic` is on (`retizero_reproduction.py:373-379`). Attention backward kernels are not covered by that flag. Fix: none needed; state "seeded, not bitwise deterministic". Optionally set `torch.use_deterministic_algorithms(True, warn_only=True)` in a new file if bitwise is wanted.
- **L2. No queue-level check that the pretrained file actually loads.** `checks()` (`retizero_runs.py:239-246`) checks only existence and hash. A wrong or corrupt file makes each of the six runs fail separately, though early in each. Fix: run pilot_02 first (the notebook already says so), or the `check` command in step 5 below. The code could call `train_retizero.py check` from `checks()` when the file is present.
- **L3. The training-time estimate is a guess** (`retizero_runs.py:40`, `MINUTES["train"] = 12`), taken from a 160-image pilot. Fix: replace it with the measured wall minutes after the first run (that file is not the frozen helper).
- **L4. The SHA is self-pinned.** Upstream publishes no checksum. Fix: record in `REPRODUCTION_PLAN.md` the download URL, date, byte size and SHA, and that `check` loaded it with 0 missing and 0 unexpected keys.
- **L5 (resolved with M1). GradScaler-skipped steps were not recorded.** Now counted per epoch as `skipped_steps_scaler_overflow` (steps where the scale decreased) and totalled in `train_result.json`; they do not count toward the 10% limit. `retizero_reproduction.py:483-485` and the history at `:496-505`. Fix (only together with M1, before the first full run): log `scaler.get_scale()` per epoch, or count the steps where it decreased.
- **L6. `history.train_loss` is the image-count-weighted mean of the per-group weighted means.** `:485`, `:498`. This is the same as the original's logging, and only logging. It is not the dataset-level weighted CE. Fix: none; do not report it as the training loss value of the objective.

## Exact steps for tomorrow

1. **Get the checkpoint.** Download `RetiZero.pth` from the Google Drive link in `D:\Afterimage_Reproduction\RetiZero\RetiZero\README.md`. Save it to `D:\Afterimage_Experiments\assets\RetiZero.pth`. Check that it is a real `.pth` of a few GB and not an HTML page.
2. **M1 is resolved** (skip and count). After step 7, `retizero_reproduction.py` is frozen.
3. **Close other GPU work.** No Qwen queue may run; the two queues share the GPU lock.
4. **Get the SHA.** Open `notebooks\02_retizero_runs.ipynb` with the `afterimage_env` (Python 3.11) kernel. Run **1. Setup**, then **2. What is left**. The checks line prints `Set SETTINGS['pretrained_sha256'] = '<sha>'`. Paste that value into Setup, set `SETTINGS["pilot_tag"] = "pilot_02"`, and keep `TRAINING_MODE = False` for now. Rerun Setup and Section 2. Expect the pretrained line to read "matches its pinned SHA-256", and `historical_*` and `matched_*` to read `pending`, while the clahe and Messidor jobs stay `waiting`.
5. **Optional CPU fit check** (about 1–2 min, no GPU, writes nothing; also re-hashes every image), from a terminal with the env active:
   ```
   cd D:\Afterimage_Reproduction\finetuning\RetiZero\paper_revision
   python -X utf8 -B train_retizero.py check --dataset D:\Afterimage_Experiments\datasets\aptos_224_standard\duplicate_excluded\manifest.json --matched-dataset D:\Afterimage_Experiments\datasets\aptos_512\duplicate_excluded\manifest.json --weights D:\Afterimage_Experiments\assets\RetiZero.pth --weights-sha256 <sha> --run-dir D:\Afterimage_Experiments\retizero_runs\_unused --seed 42
   ```
   Expect `The pretrained checkpoint fits the encoder (390 tensors)` and `Model: 304,093,189 parameters, 304,093,189 trained`.
6. **Pilot with pretrained weights.** Set `TRAINING_MODE = True`, rerun Setup, and run **4. Pilot** (show_plan, then the launch cell). Takes about 3–5 min. It must end with **"Pilot passed. The full queue can be started."** and must *not* print the "trained from random weights" note. Read the reported training speed and peak memory.
7. **Full queue.** Run **5. Full queue** and leave it. It runs historical_t42–44, matched_t42–44 and the three benchmark scorings, about 1.5–2 h. CLAHE and Messidor jobs are skipped as `waiting`. If interrupted, rerun the same cell; it resumes from the last finished epoch. Do not edit code or packages until it finishes.
8. **Check results.** Run **7. Results**. Confirm 6 training rows with no `constant_predictor` and best epochs that are not all 1, the three `matched` rows on 361 images, and `benchmark` rows with 0 saved-prediction mismatches. Record the measured wall minutes per run (L3) and the checkpoint provenance (L4) in `REPRODUCTION_PLAN.md`.
9. **Later.** Build the 224-pixel CLAHE and Messidor datasets (plan step 9), then rerun step 7. Only the `clahe` and `messidor_archived` jobs will run. Then extend `build_paper_tables.py` (plan step 11).
