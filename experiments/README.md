# Four-condition Qwen3-VL-8B queue

After the three CLAHE-on runs: see [POST_RUN_REVIEW.md](POST_RUN_REVIEW.md) for the saved-results comparison and prioritized checklist, and [DATASET_PREPROCESSING_HANDOFF.md](DATASET_PREPROCESSING_HANDOFF.md) for the data researcher's preprocessing/QC specification.

Prepared by reading source/configuration files, writing these three files, and reviewing them statically. No notebook cells, tests, preprocessing, model loading or training were executed during preparation. Existing reproduction and Phase 1 audit files are unchanged.

Open **[qwen_four_run_orchestrator.ipynb](qwen_four_run_orchestrator.ipynb)** and select **Python 3.11 — Python (afterimage_env)**. The recorded interpreter is `C:\Users\moht2\.conda\envs\afterimage_env\python.exe`. Use the existing environment and installed packages; this queue never installs packages or downloads assets.

1. Edit and run **Configure paths**, then **Shared setup**. These define paths and load the standard-library controller; they start no workers or model work.
2. Leave `UNATTENDED_QUEUE = False` for the default layout. Run **Shared preflight**, then use each run's independent **Training**, **Verify saved adapter**, and **Evaluation** cells. Each stage streams into its own cell output. Running one stage never starts another condition.
3. Each of the four run sections has separate cells for its saved **configuration**, **status/results**, **training log**, **verification log**, and **evaluation log**. These views only read saved files. Increase `lines=40` in a log cell to inspect more history. After restarting the kernel, configuration/setup plus these view cells are enough to inspect earlier work; preflight/training are unnecessary for inspection.
4. For unattended operation, either use Jupyter **Run All** with the default mode (the independent stage cells execute in order), or set `UNATTENDED_QUEUE = True`, rerun configuration/setup, and run **Optional unattended queue** once. That short launcher queues all eligible conditions; individual stage cells skip in this mode, while the per-run views still work.
5. Keep the kernel, Jupyter, and computer awake while a worker is active. Interrupting an action terminates/waits, then kills/waits if needed, for the active child and ends that action/queue. After interruption or a shared failure, rerun **Shared preflight** before another stage. Shut down the kernel after the active action returns if desired.

For an evaluation-only failure, return to that run's **Evaluation** cell. It rechecks the retained adapter in a CPU child, then launches evaluation in a fresh process; it does not queue training. Use the run's verification/status/log cells to pinpoint a problem before retrying.

To run only the three CLAHE-on rank experiments, use `INCLUDE_CLAHE_OFF = False` (the current default). Set `UNATTENDED_QUEUE = True`, run configuration/setup, then **Optional unattended queue** once to run ranks 16 → 8 → 32 sequentially, including both-cohort evaluation for each. `ORIGINAL_IMAGE_INDEX` may stay `None`: the off-branch/original-image pairing checks are skipped. The fourth condition is recorded as `excluded_by_configuration`, not as an input failure. To include the ablation later, use `INCLUDE_CLAHE_OFF = True` with the required original-image index and rerun preflight.

| Order | Run ID | Rank | CLAHE | alpha/r |
| --- | --- | --- | --- | --- |
| 1 | `r16_clahe_on` | 16 | On | 1 |
| 2 | `r8_clahe_on` | 8 | On | 2 |
| 3 | `r32_clahe_on` | 32 | On | 0.5 |
| 4 | `r16_clahe_off` | 16 | Off | 1 |

The first run is a fresh control shared by the rank and CLAHE comparisons. **Retraining 2** remains the previously reported reference: **80.65% APTOS accuracy, 0.8915 QWK, 0.9362 macro ROC-AUC**. Those figures are printed separately from new results. This queue does not reuse its weights or pick winners using test metrics.

## Paths

Defaults point at the inspected local files:

- `PROJECT_ROOT`: `C:\Users\moht2\Documents\Afterimage`.
- `REVIEWED_CONFIGURATION`: `finetuning/Qwen3/artifacts/paper_reproduction/training_batches/aptos_paper_same_seed_three_runs_001/paper_training_configuration.json` below the project.
- `DATASET_MANIFEST`: `finetuning/Qwen3/artifacts/paper_reproduction/data/4fda605471d5430c3733/dataset_manifest.json` below the project.
- `TARGET_ADAPTER_CONFIGURATION`: `run_01/best_lora/adapter_config.json` next to the reviewed configuration. Only configuration metadata is used to pin the target-module regex; its trained weights are never a starting point.
- `CLAHE_ON_IMAGES` and `CLAHE_OFF_IMAGES`: the two requested folders below `C:\Users\moht2\Downloads\DR_Aptos_Messidor_Only_Full_Dataset\Res_512`.
- `BASE_SNAPSHOT`: `None` uses the cached revision from the reviewed configuration. Set a local path only if the cache moved, preserving revision `b5b904c3fcdc7541adf2a2bb219b0ed95288c794`.
- `BATCH_DIR`: defaults to `experiments/artifacts/qwen_four_run_001`. Keep it for restart; use a new directory name for a fresh experiment. Project-local outputs must remain below `experiments/artifacts`; reproduction directories are protected.
- `INCLUDE_CLAHE_OFF`: `False` selects the three CLAHE-on ranks only. `True` also requests the ablation and enables its pairing checks.
- `ORIGINAL_IMAGE_INDEX`: **fill this in when requesting verified CLAHE ablation**. It may stay `None` with `INCLUDE_CLAHE_OFF = False`. If the ablation is requested but originals cannot be established, that condition is blocked while the rank comparison may proceed.

`GPU_INDEX="0"` exposes one GPU. Preflight requires CUDA/BF16, at least 10 GiB free GPU memory, exact reviewed package versions, and 80 GiB free output disk by default (`MINIMUM_DISK_GIB`). Each later stage requires 10 GiB disk free. There is no automatic batch-size, precision, optimizer, seed or training-length reduction.

## Original-image index and CLAHE verification

Use a UTF-8 CSV outside the batch output with columns:

```csv
source_dataset,image_id,original_path,is_original
aptos,000c1434d8d7,D:\DR originals\APTOS\000c1434d8d7.png,true
messidor,messidor_messidor_p_0_20051109_57451_0400_PP,D:\DR originals\Messidor\20051109_57451_0400_PP.jpg,true
```

These are illustrative rows, not a complete index. Include **all 4,719 frozen IDs**, exactly once, using `aptos` / `messidor` and the recovered CSVs' `id_code` as `image_id`. Original paths may be absolute or relative to the index. `is_original=true` is your provenance assertion that these are independent, unenhanced originals; do not point at either processed branch. The worker hashes every original and records the source-to-output evidence.

After manual launch, preflight checks both folders for exact matching filename coverage, RGB PNG format, 512×512 dimensions and image integrity. It uses the recovered CSV's explicit ID→renamed-filename binding; it never guesses a Messidor patient prefix or joins by row position. Both branches retain the same frozen labels/splits. It then checks every pair in memory:

1. Read the independent original in OpenCV BGR; convert to RGB; use Otsu foreground cropping, black square padding and resize to 512 (AREA when shrinking, CUBIC otherwise); convert back to BGR. This must equal the off PNG exactly.
2. Convert the off pixels BGR→LAB, apply CLAHE to L **once**, with clip limit 2.0 and tile grid 8×8, then LAB→BGR. This must equal the on PNG exactly.

No processed images are written. `pairing_checks.csv` records checks, IDs and original hashes. RGB/BGR/LAB order is explicit in the worker; CLAHE is never applied in the training or evaluation paths. Missing originals, differing recipes/channel order, incomplete pairing, lossy files, or mismatched pixels block `r16_clahe_off`. The existing folder names and historical preprocessing claims alone are insufficient to establish a clean ablation. Exact pixel comparison is intentionally strict.

The shared CLAHE branch separately requires the reviewed manifest/mapping/original split hashes, exact 2,929/366/367 APTOS and 1,057 Messidor counts/class counts, unique IDs and paths, no image or patient split overlap, valid labels, existing images and frozen bytes. If it fails, the entire queue stops before training. If only off-branch pairing fails, the three on-branch ranks may proceed. Their comparison holds the identical existing CLAHE bytes fixed; historical raw-to-CLAHE lineage remains an ablation blocker until verified.

The historical image mapping has different split assignments from the recovered original APTOS split. The worker checks its ID/label/path bindings but deliberately ignores its split column. It preserves the recovered manifest membership and verifies it against the original split; it never regenerates splits or trains on Messidor.

## Fixed training and evaluation

The reviewed JSON supplies epochs 50; batch 4 × accumulation 2 = effective batch 8 on one GPU; Trainer seed 42; LoRA initialization 3407; learning rate 5e-4; AdamW-8bit; linear scheduler; warmup 20; weight decay 0.001; image 512; max sequence length 2048; 4-bit loading; BF16 training; epoch validation/checkpoints; retain two checkpoints; patience 2 and threshold 0.0; best checkpoint by minimum APTOS validation loss. Alpha 16, dropout 0.05 and bias `none` are verified against the reviewed configuration and target adapter record. RS-LoRA and rank/alpha patterns are excluded, so scaling is alpha/r.

The worker uses the existing `make_sft_config`, `BalancedSFTTrainer`, prompt/candidate definitions and full-answer likelihood scorer, with source hashes checked against the reviewed JSON. It does **not** invoke the old fixed-rank `train_one` or fixed-configuration validator. New datasets contain deterministic formatting only; no image augmentation is introduced. Inverse-frequency sampling with replacement applies only to the training dataloader. APTOS validation alone selects checkpoints; test metadata may be checked for integrity, but no test or Messidor examples are passed to the training/validation datasets.

The reviewed JSON explicitly requests validation batch 8. If the installed Trainer wrapper changes that default-valued field to training batch 4, the worker restores 8 before validation and records `library_argument_corrections.json`. Other effective-setting mismatches stop the shared queue. This preserves the supplied reviewed setting rather than adopting the historical run's coerced argument.

Every fresh condition loads the same fingerprinted cached base and creates a new adapter. Constructor arguments explicitly include the selected rank, pinned target-module regex and seed. In-memory adapter configuration/scaling and saved `adapter_config.json` are checked before completion is marked. No existing adapter is used as a fresh initializer. A compatible interrupted checkpoint continues only its own condition.

All four use the same FP16 full-answer likelihood evaluation implementation in fresh processes, with their matching on/off input branch. It scores all five valid grade answers, saves normalized scores `p0`…`p4`, and selects the highest-scoring grade (first grade on exact ties). Invalid/nonfinite scores fail evaluation; there is no invalid-output Grade-0 fallback. Accuracy, fixed-five-grade QWK and five-class macro one-vs-rest ROC-AUC reuse compatible Phase 1 metric calculations. Every completed export has an exact ID/label/path join to its frozen test cohort. Legacy fallback investigation remains separate under `audit`.

New data/prediction exports include an explicit `original_image_id` to preserve the recovered ID when image filenames have patient/dataset prefixes. Completed predictions also get a hash-bound `.audit.json` sidecar with score order and producer/label links, compatible with the Phase 1 audit. These new conditions are not labeled as Retraining 2.

## Restart and failure behavior

The shared `NotebookOrchestrator` class in `qwen_experiment_worker.py` owns an OS-backed batch lock and one `Popen` launch site using `sys.executable`, `-u` and a list of arguments. It is imported into the notebook without starting work. CLI stage workers never launch child processes. Each explicitly requested notebook action holds the lock while active, streams/logs each child, waits for it, and closes its pipe before the next child starts. The lock is released between independent cells; read-only inspection needs no lock. Models and image datasets stay in workers; the kernel holds launch settings and saved diagnostic views. Abruptly killing the kernel or powering off is different from interrupting a cell; keep it alive and use the interrupt control to stop the active worker cleanly.

On restart with the same `BATCH_DIR`, rerun preflight before stage launches. The optional queue does this automatically. Source, environment, base, input/configuration and image hashes must still match. Verified completed training and evaluations are skipped without loading a model. Evaluation failures preserve the adapter and can resume their own verified prediction prefix independently. Changed/corrupt completion artifacts are flagged and retained. Inspection cells show saved records without claiming to revalidate them; the Verify action performs the adapter checks.

Each newly saved checkpoint gets `checkpoint_identity.json` only after the Trainer save callback. Resume requires the exact condition/configuration fingerprint, verified adapter rank/targets, matching hashes, Trainer state, optimizer, scheduler, RNG, training arguments and early-stopping state. The newest compatible complete checkpoint is chosen; incomplete candidates are logged. A previous training attempt with no compatible complete checkpoint is not silently restarted from scratch: that condition fails and needs a new batch or restored complete checkpoint. Old reproduction checkpoints are never imported.

An isolated condition failure (including rank-specific out-of-memory) is recorded and the next eligible condition starts. A shared setup/data/source/environment/base mutation, disk shortage or CUDA driver/device failure stops the remaining queue. A notebook interrupt stops the active child and ends the queue; no new training/evaluation process is launched afterward. Each stage records attempts, start/end times, durations and status. A shared preflight failure writes exact blockers and requested settings for every run without starting training.

Roughly **12 hours** is a planning window, not a hard kill deadline. Actual time depends on validation stopping and scoring both cohorts; no test-based stopping or deadline-based epoch trimming is introduced. The queue exits when all eligible work is handled, or earlier on shared failure/interruption.

```text
<BATCH_DIR>/
  launch_request.json, preflight.json, shared_configuration.json
  pairing_checks.csv, queue_status.json, stage_status.json
  batch_summary.csv, batch_report.md, preflight.log, summarize.log
  data/clahe_on/                       frozen CSVs; clahe_off/ only if verified
  r16_clahe_on/                        same layout for each other condition
    resolved_configuration.json       rank/scaling, data/source/base identity
    stage_status.json                 all attempts and elapsed times
    train.log, verify.log, evaluate.log
    adapter_constructor_arguments.json, effective_adapter_scaling.json
    resolved_training_arguments.json, checkpoints/, best_lora/
    training_complete.json, run_info.json
    evaluation/                       five scores/image, summaries, progress
    evaluation_complete.json
```

Worker exit codes: **0** stage verified/completed; **1** isolated failure; **3** shared failure; **130** handled interruption. A preflight that blocks only the fourth condition returns 0 with explicit eligibility flags. Review `batch_summary.csv` and `batch_report.md`; `queue_status.json` distinguishes normal completion of eligible work from an interrupted/shared stop. Large generated artifacts, images, checkpoints and weights are runtime outputs, not files to commit with these three source files.

## Static preparation review

Reviewed against `train_aptos_paper_three_runs.ipynb`, `paper_reproduction.py`, the requested configuration/manifest, saved rank-16 adapter metadata, dataset CSV headers/rows, relevant installed Unsloth source, and Phase 1 audit code. The notebook now separates configuration, shared setup, preflight, optional queue control, all four runs' individual stages/diagnostic views, and the saved batch comparison. Inspected the controller import boundary, stage dispatch, lock ownership, terminate/wait/kill/wait/pipe-close paths, explicit rank propagation, per-condition isolation, checkpoint identity gates, held-out separation and reported/reference separation. All code cells have empty outputs and null execution counts. **No notebook cells, workers, preprocessing, model loading, training, runtime validation or tests were executed during this revision.**
