# RetiZero fine-tuning

Training, evaluation and measurement of the RetiZero baseline for the paper. Not imported by the app. The layout mirrors `finetuning/Qwen3/`.

**Open [`notebooks/02_retizero_runs.ipynb`](notebooks/02_retizero_runs.ipynb).** It checks the inputs, runs a pilot, runs the queue and shows the results.

## What lives where

| Path | What it is |
| --- | --- |
| `retizero_reproduction.py` | Shared helper: model construction, checkpoint handling, dataset checks, transforms, the training loop and scoring. Its hash is recorded in every result. **Do not edit it once full runs exist**; add a new file instead. |
| `paper_revision/train_retizero.py` | Worker for one training run: `check`, `train`, `evaluate`. |
| `paper_revision/score_retizero.py` | Worker that scores any fine-tuned checkpoint on any cohort and measures time per image, memory and parameter counts. |
| `paper_revision/retizero_runs.py` | The queue used by the notebook: checks, plan, pilot, full queue, summary. |
| `paper_revision/test_retizero_runs.py` | CPU-only tests. |
| `notebooks/02_retizero_runs.ipynb` | The control panel. |
| `legacy/` | The two original Colab notebooks (APTOS and Messidor-2), unchanged, and `original_learning_rate_search.csv`, the 25-trial search copied from the APTOS notebook's saved output. |
| `REPRODUCTION_PLAN.md` | The plan, the decisions and what is still open. |

## What the code reproduces

The original APTOS notebook (`legacy/ICT302_Retizero_Training_Unbalanced_Aptos.ipynb`): the RetiZero vision encoder with every weight trainable and a new five-grade linear head, AdamW (weight decay 1e-4), cosine schedule over 15 epochs, mixed precision, cross-entropy weighted by inverse class frequency, training-only augmentation, selection on validation accuracy with early stopping after 4 epochs, and the learning rate of its chosen trial (7.68094483149829e-5).

Differences from the original, all recorded in each run's `configuration.json`:

- **Seeds are fixed** (42, 43, 44). The original set none.
- **Batches of 128 are processed 32 images at a time.** The original used 128 on a 40 GB GPU. Gradients are accumulated over the group so each optimizer step uses the same 128-image weighted loss. The model has no batch statistics, so this is equivalent up to rounding.
- **The model is built from the vision encoder directly**, without the text encoder the classifier never uses. The saved checkpoint loads into it with no missing or unexpected weights.
- **One loader process** instead of four, and deterministic cuDNN.
- **The learning rate is not searched again.**

Metrics come from `finetuning/Qwen3/paper_reproduction.py`, the same function that scores Qwen.

## Where things are

| Item | Location |
| --- | --- |
| Results | `D:\Afterimage_Experiments\retizero_runs\<tag>\` (pilot: `<tag>__<pilot_tag>\`) |
| Fine-tuned checkpoint used by the app | `RetiZero/best_retizero_dr.pth` (gitignored), SHA-256 `b254d11d…6871f0` |
| Official pretrained checkpoint | Expected at `D:\Afterimage_Experiments\assets\RetiZero.pth`. **Not on this PC yet.** Download link: `RetiZero/RetiZero/README.md`. |
| 224-pixel APTOS builds | `D:\Afterimage_Experiments\datasets\aptos_224_standard\{previous_split,duplicate_excluded}` |
| Launcher, monitor and GPU lock | Shared with Qwen: `D:\Afterimage_Experiments\code`, lock in `D:\Afterimage_Experiments\runs` |

## Status (15 November 2026)

- The saved checkpoint is replayed. It reproduces every figure the original notebook printed for the 367 test images: 82.29% accuracy, 0.9097 QWK, 0.9451 macro AUC, and all per-grade figures. Evidence: `retizero_runs\retizero_01\replay_comparison.json`.
- The pilot passed, training from random weights because the pretrained checkpoint is missing.
- No training run for the paper exists yet. All nine wait for the pretrained checkpoint; the CLAHE runs and the Messidor scoring also wait for datasets that are not built.

## Tests

```bash
cd finetuning/RetiZero/paper_revision
python -B -m unittest test_retizero_runs -v
```
