# finetuning/Qwen3

Qwen3-VL-8B + LoRA training and evaluation for the paper revision.

| Path | What it is |
|---|---|
| `paper_reproduction.py` | Shared training/evaluation helper and the five-answer likelihood scorer. Its path and SHA-256 are pinned in saved run configurations. Do not move or edit it. |
| `train_dr_qwen3vl_runsave.py` | Shared trainer definitions (`BalancedSFTTrainer`). Its path, SHA-256 and line endings are pinned. Do not move or edit it. |
| `evaluate_clahe_bf16.py` | BF16 scorer for the original-split CLAHE pairs. Its SHA-256 is recorded in each `evaluation_bf16/evaluation_manifest.json`, and it imports `paper_reproduction` from this folder, so it stays here. |
| `artifacts/` | Saved data, runs and evaluations (gitignored). Read-only evidence; configurations inside store absolute paths. |
| `paper_revision/` | Live paper helpers imported by the notebooks: `reproduce_run_02.py`, `remaining_runs.py`, `train_bf16_control.py`, `probe_bf16_training.py`, `evaluate_verified_messidor.py`. `train_bf16_control.py` and `probe_bf16_training.py` are hash-pinned by saved runs and checks: do not edit them. |
| `notebooks/` | The notebooks to open (kernel: Python (afterimage_env)). |
| `legacy/` | Earlier generation-based evaluation scripts and notebooks, kept for reference. Not used by the paper workflow. |

## Which notebook to open

- `notebooks/03_final_runs.ipynb`: the queue for everything still to run for the paper (zero-shot baseline, timing and invalid-output checks, archived Messidor, learning-rate and alpha variants). Run its pilot first, then the full queue. It uses `paper_revision/final_runs.py` (the queue), `qwen_variant.py` (training variants through the pinned runner), `score_cohort.py` (scoring with per-image timing) and `test_final_runs.py` (CPU-only tests). Results go to `D:\Afterimage_Experiments\final_runs\<tag>`.
- `notebooks/021_training.ipynb`: control panel for the remaining paper runs. Covers the BF16 base control, verified Messidor, the rank 8/32 CLAHE-off runs and the previous-split controls. Start with Setup and "Check what is left".
- `notebooks/02_training.ipynb`: Retraining 2 replay and the original-split CLAHE pairs.
- `notebooks/01_preprocessing.ipynb`: builds the paired CLAHE-on/off APTOS datasets for the original split.
- `../paper_tables/build_paper_tables.py`: rebuilds the paper tables from saved predictions (CPU only).

The duplicate-excluded 361-image runs live in `D:\Afterimage_Experiments\runs`. They are launched by `D:\Afterimage_Experiments\code\qwen_pilot.py`, which reads the pinned helper and trainer copies recorded in each run's `pilot_configuration.json`.

RetiZero has its own folder: `finetuning/RetiZero/` (start with its README).
