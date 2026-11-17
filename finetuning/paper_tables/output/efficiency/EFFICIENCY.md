# S14 efficiency: Qwen3-VL-8B + LoRA vs RetiZero

Computed by `build_efficiency_tables.py` from saved run records only (no new GPU work). Mean ± SD is over seeds (sample SD); n is shown when above 1. Missing quantities are listed in [GAPS.md](GAPS.md).

## Definitions

- **Training wall time (all attempts)**: sum over every attempt of the training stage. Recorded stage times are used where they exist (`run_report.md`); an interrupted attempt has none, so the span of its `utilization.csv` is used (about 5 s short).
- **s/epoch and images/s** use the Trainer's own `train_runtime` (training loop, per-epoch validation and checkpointing, without model loading) divided by the epochs run (maximum `epoch` in `train_result.json`). Epochs vary (3 to 9) because of early stopping, so totals are not comparable across runs, s/epoch is. The Trainer's logged `train_samples_per_second` is not used: it divides by the 50 planned epochs, not the epochs run.
- **Whole-card memory / GPU % / CPU % / RAM** come from `utilization.csv` (5 s samples, whole machine, including the desktop; baseline = first sample). **PyTorch peak** is process-level.
- **Warm per-image time** skips the first 3 images. Qwen: log-likelihood of all five grades (five forward passes), the path that produces the reported metrics; greedy generation (24 new tokens) is listed separately. RetiZero: one forward pass at 224 px, batch 1, transfer + forward + softmax + image read/transform.
- Runs with extra attempts (interrupted or failed first attempts) are counted in full in wall time; GPU, memory and utilisation figures use the attempt that completed training (`train_wall_completing_attempt_s` and `train_wall_other_attempts_s` in the per-run CSV separate the two). Pilot runs are flagged and excluded from the comparison.

## Comparison (APTOS 361-image test set)

| Quantity | Unit | Qwen3-VL-8B + LoRA r16, 4-bit, CLAHE-on | Qwen3-VL-8B + LoRA r16, BF16, CLAHE-on | RetiZero ViT-L/16 + LoRA r8 |
|---|---|---|---|---|
| Total parameters | count | 8,818,470,640 | 8,818,470,640 | 304,093,189 |
| Trainable parameters | count | 51,346,944 | 51,346,944 | 304,093,189 |
| Trainable share of base | % | 0.586 | 0.586 | 100.000 |
| Adapter / checkpoint file | bytes | 205,492,968 | 205,492,968 | 1,216,540,027 |
| Training epochs run (early stopping) | epochs | 5.0 ± 3.5 (n=3) | 3.7 ± 1.2 (n=3) | not run yet |
| Training wall time, all attempts | s | 4,545 ± 3,120 (n=3) | 3,378 ± 774 (n=3) | not run yet |
| Training time per epoch (incl. validation) | s/epoch | 893.6 ± 7.4 (n=3) | 925.2 ± 69.8 (n=3) | not run yet |
| Training throughput | images/s | 3.12 ± 0.03 (n=3) | 3.02 ± 0.24 (n=3) | not run yet |
| Training peak GPU memory, PyTorch allocated | GiB | 8.93 ± 0.00 (n=3) | 13.38 ± 0.00 (n=3) | not run yet |
| Training peak GPU memory, whole card | MiB | 11,463 ± 43 (n=3) | 15,526 ± 129 (n=3) | not run yet |
| Training mean / peak GPU utilisation | % | 88.4 / 99 (n=3) | 90.1 / 100 (n=3) | not run yet |
| Scoring model load time | s | 9.9 ± 0.5 (n=3) | 25.8 | 5.9 |
| Scoring warm median time per image (batch 1) | s/image | 0.8608 ± 0.0121 (n=3) | 1.3484 | 0.0177 |
| Scoring throughput (batch 1) | images/s | 1.15 ± 0.02 (n=3) | 0.74 | 54.47 |
| Qwen greedy generation, warm median | s/image | 1.570 ± 0.121 (n=3) | 2.453 | n/a |
| Scoring throughput (batched, 64) | images/s | n/a | n/a | 323.7 |
| Scoring peak GPU memory, PyTorch allocated (batch 1) | GiB | 8.65 ± 0.00 (n=3) | 12.05 | 1.18 |
| Scoring peak GPU memory, whole card | MiB | 12,057 ± 0 (n=3) | 14,178 | 3,326 |
| Scoring mean GPU utilisation | % | 43.6 ± 2.9 (n=3) | 91.9 | 27.8 |

RetiZero training: **not run yet** (only a 2-epoch pilot on 160 random-initialised images exists; see GAPS.md). 
Pilot indication, not a result: 61.9 images/s, PyTorch peak 11.2 GiB, whole-card peak 13592 MiB (`pilot_resumed_t42`).

Qwen scoring figures are for the same seeds' adapters re-scored in final_01 (`timing.csv`); the BF16 column has one scoring job (n=1). RetiZero scoring is one deterministic job (n=1).

## Hardware and software

| Kind | Item | Value |
|---|---|---|
| hardware/OS | os | Windows-10-10.0.26300-SP0 (80 attempt files) |
| hardware/OS | python | 3.11.18 / packaged by Anaconda, Inc. / (main, Oct 26 2026, 18:13:29) [MSC v.1942 64 bit (AMD64)] (80 attempt files) |
| hardware/OS | cpu | AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD (80 attempt files) |
| hardware/OS | logical_cpus | 16 (80 attempt files) |
| hardware/OS | physical_cores | 8 (80 attempt files) |
| hardware/OS | ram_bytes | 67767488512 (80 attempt files) |
| hardware/OS | gpu_name | NVIDIA GeForce RTX 5070 Ti (80 attempt files) |
| hardware/OS | gpu_total_vram_mib | 16303 (80 attempt files) |
| hardware/OS | gpu_driver | 621.24 (66 attempt files); 621.14 (14 attempt files) |
| software (Qwen scoring) | bitsandbytes | 0.50.2 (30 attempt files) |
| software (Qwen scoring) | datasets | 4.8.5 (30 attempt files) |
| software (Qwen scoring) | numpy | 2.4.6 (30 attempt files) |
| software (Qwen scoring) | pandas | 3.0.6 (30 attempt files) |
| software (Qwen scoring) | peft | 0.21.2 (30 attempt files) |
| software (Qwen scoring) | torch | 2.14.1+cu130 (30 attempt files) |
| software (Qwen scoring) | torchvision | 0.29.1+cu130 (30 attempt files) |
| software (Qwen scoring) | transformers | 5.17.0 (30 attempt files) |
| software (Qwen scoring) | trl | 1.15.0 (30 attempt files) |
| software (Qwen scoring) | unsloth | 2026.10.10 (30 attempt files) |
| software (Qwen scoring) | unsloth_zoo | 2026.10.10 (30 attempt files) |
| software (Qwen training) | bitsandbytes | 0.50.2 (37 attempt files) |
| software (Qwen training) | datasets | 4.8.5 (37 attempt files) |
| software (Qwen training) | numpy | 2.4.6 (37 attempt files) |
| software (Qwen training) | pandas | 3.0.6 (37 attempt files) |
| software (Qwen training) | peft | 0.21.2 (37 attempt files) |
| software (Qwen training) | torch | 2.14.1+cu130 (37 attempt files) |
| software (Qwen training) | torchvision | 0.29.1+cu130 (37 attempt files) |
| software (Qwen training) | transformers | 5.17.0 (37 attempt files) |
| software (Qwen training) | trl | 1.15.0 (37 attempt files) |
| software (Qwen training) | unsloth | 2026.10.10 (37 attempt files) |
| software (Qwen training) | unsloth_zoo | 2026.10.10 (37 attempt files) |
| software (RetiZero scoring) | Pillow | 13.0.0 (4 attempt files) |
| software (RetiZero scoring) | numpy | 2.4.6 (4 attempt files) |
| software (RetiZero scoring) | torch | 2.14.1+cu130 (4 attempt files) |
| software (RetiZero scoring) | torchvision | 0.29.1+cu130 (4 attempt files) |
| software (RetiZero scoring) | transformers | 5.17.0 (4 attempt files) |

## Parameters and checkpoint size

| Model | Frozen base | Trainable | Total | Trainable % of base | Adapter/checkpoint bytes | Base weight files bytes |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct + LoRA r8 (4-bit base) | 8,767,123,696 | 25,673,472 | 8,792,797,168 | 0.293 | 102,797,504 | 7,699,477,714 |
| Qwen3-VL-8B-Instruct + LoRA r16 (4-bit base) | 8,767,123,696 | 51,346,944 | 8,818,470,640 | 0.586 | 205,492,968 | 7,699,477,714 |
| Qwen3-VL-8B-Instruct + LoRA r16 (BF16 base) | 8,767,123,696 | 51,346,944 | 8,818,470,640 | 0.586 | 205,492,968 | 17,534,339,512 |
| Qwen3-VL-8B-Instruct + LoRA r32 (4-bit base) | 8,767,123,696 | 102,693,888 | 8,869,817,584 | 1.171 | 410,881,056 | 7,699,477,714 |
| RetiZero ViT-L/16 + LoRA r8, all weights trained | 0 | 304,093,189 | 304,093,189 | 100.000 | 1,216,540,027 | n/a |

Notes per row are in `efficiency_models.csv`.

## Training by configuration

| Configuration | n | Epochs | Wall s (all attempts) | s/epoch | Images/s | PyTorch peak GiB | Card peak MiB | GPU mean / peak % | CPU mean % | RAM peak GiB | Runs with >1 attempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4-bit r8 CLAHE-on lr 0.0005 alpha 16 | 3 | 4.0 ± 1.0 | 3,549 ± 846 | 882 ± 10 | 3.16 ± 0.04 | 8.68 ± 0.00 | 11,205 ± 22 | 88 / 99 | 14.4 ± 0.3 | 24.3 ± 0.2 | 0 |
| 4-bit r8 CLAHE-off lr 0.0005 alpha 16 | 3 | 3.7 ± 0.6 | 3,530 ± 794 | 899 ± 14 | 3.10 ± 0.05 | 8.68 ± 0.00 | 11,216 ± 41 | 88 / 100 | 16.1 ± 1.8 | 29.5 ± 0.8 | 1 |
| 4-bit r16 CLAHE-on lr 0.0001 alpha 16 | 3 | 4.0 ± 0.0 | 3,568 ± 12 | 884 ± 3 | 3.15 ± 0.01 | 8.93 ± 0.00 | 11,476 ± 27 | 88 / 100 | 16.3 ± 0.2 | 28.1 ± 0.1 | 0 |
| 4-bit r16 CLAHE-on lr 0.0002 alpha 16 | 3 | 5.3 ± 1.2 | 5,003 ± 846 | 940 ± 70 | 2.97 ± 0.22 | 8.93 ± 0.00 | 11,490 ± 94 | 88 / 99 | 22.5 ± 5.5 | 29.3 ± 0.8 | 0 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 8 | 3 | 3.7 ± 0.6 | 3,668 ± 531 | 993 ± 36 | 2.81 ± 0.10 | 8.93 ± 0.00 | 11,424 ± 53 | 87 / 99 | 24.0 ± 3.9 | 30.0 ± 0.2 | 0 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 16 | 3 | 5.0 ± 3.5 | 4,545 ± 3,120 | 894 ± 7 | 3.12 ± 0.03 | 8.93 ± 0.00 | 11,463 ± 43 | 88 / 99 | 16.3 ± 3.1 | 24.5 ± 0.3 | 1 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 32 | 3 | 4.0 ± 1.0 | 3,804 ± 887 | 945 ± 16 | 2.95 ± 0.05 | 8.93 ± 0.00 | 11,409 ± 63 | 88 / 99 | 19.0 ± 0.3 | 29.7 ± 0.1 | 0 |
| 4-bit r16 CLAHE-off lr 0.0005 alpha 16 | 3 | 5.0 ± 1.0 | 4,521 ± 1,028 | 895 ± 27 | 3.11 ± 0.09 | 8.93 ± 0.00 | 11,493 ± 44 | 89 / 99 | 17.1 ± 3.7 | 25.1 ± 0.8 | 0 |
| 4-bit r32 CLAHE-on lr 0.0005 alpha 16 | 3 | 3.3 ± 0.6 | 3,042 ± 463 | 906 ± 25 | 3.08 ± 0.08 | 9.44 ± 0.00 | 12,051 ± 49 | 87 / 99 | 15.9 ± 2.2 | 25.0 ± 0.2 | 0 |
| 4-bit r32 CLAHE-off lr 0.0005 alpha 16 | 3 | 4.0 ± 1.0 | 3,620 ± 794 | 902 ± 30 | 3.09 ± 0.10 | 9.44 ± 0.00 | 12,057 ± 31 | 87 / 100 | 15.8 ± 1.1 | 29.7 ± 0.2 | 0 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 16 previous split | 3 | 4.3 ± 0.6 | 4,041 ± 505 | 926 ± 8 | 3.16 ± 0.03 | 8.93 ± 0.00 | 11,443 ± 99 | 87 / 100 | 15.3 ± 0.3 | 29.2 ± 0.2 | 0 |
| BF16 (+16 layers CPU-offloaded) r16 CLAHE-on lr 0.0005 alpha 16 | 3 | 3.7 ± 1.2 | 3,378 ± 774 | 925 ± 70 | 3.02 ± 0.24 | 13.38 ± 0.00 | 15,526 ± 129 | 90 / 100 | 21.8 ± 5.6 | 32.6 ± 0.3 | 1 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 16 (pilot) | 1 | 1.0 | 223 | 196 | 0.41 | 8.93 | 11,295 | 18 / 99 | 18.4 | 31.0 | 0 |
| RetiZero ViT-L/16 + LoRA r8, random init, 2-epoch 160-image pilot (pilot) | 2 | 2.0 ± 0.0 | 26 ± 8 | 2 ± 0 | 74.85 ± 18.35 | 11.17 ± 0.01 | 13,657 ± 92 | 13 / 71 | 20.3 ± 1.2 | 31.1 ± 0.5 | 1 |

## Scoring by model and cohort

| Model | Cohort | Precision | n jobs | Load s | Warm median s/image | Images/s | Generation median s | Batched images/s | PyTorch peak GiB | Card peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| BF16 (+16 layers CPU-offloaded) r16 CLAHE-on lr 0.0005 alpha 16 | aptos361_on | FP16 | 1 | 25.8 | 1.3484 | 0.74 | 2.453 | n/a | 12.05 | 14,178 |
| BF16 (+16 layers CPU-offloaded) r16 CLAHE-on lr 0.0005 alpha 16 | messidor_archived | FP16 | 3 | 17.7 ± 1.4 | 1.3339 ± 0.0007 | 0.75 ± 0.00 | n/a | n/a | 12.05 ± 0.00 | 14,240 ± 10 |
| original paper adapter (legacy best_lora, outside the experiment folders) | aptos367_original | FP16 | 1 | 9.1 | 0.8449 | 1.18 | 1.717 | n/a | 8.65 | 12,078 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 32 | messidor_archived | FP16 | 2 | 9.0 ± 0.1 | 0.8647 ± 0.0362 | 1.15 ± 0.06 | n/a | n/a | 8.63 ± 0.00 | 12,096 ± 61 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 32 | aptos361_on | FP32 fallback | 1 | 7.5 | 0.9965 | 1.00 | n/a | n/a | 12.69 | 15,287 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 32 | messidor_archived | FP32 fallback | 1 | 6.5 | 1.0166 | 0.98 | n/a | n/a | 12.69 | 15,693 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 8 | messidor_archived | FP16 | 3 | 9.6 ± 0.5 | 0.8705 ± 0.0368 | 1.15 ± 0.05 | n/a | n/a | 8.63 ± 0.00 | 12,057 ± 19 |
| 4-bit r16 CLAHE-on lr 0.0001 alpha 16 | messidor_archived | FP16 | 3 | 9.1 ± 0.8 | 0.8188 ± 0.0014 | 1.22 ± 0.00 | n/a | n/a | 8.62 ± 0.00 | 12,101 ± 0 |
| 4-bit r16 CLAHE-on lr 0.0002 alpha 16 | messidor_archived | FP16 | 3 | 9.5 ± 0.9 | 0.8837 ± 0.0588 | 1.13 ± 0.08 | n/a | n/a | 8.63 ± 0.00 | 12,113 ± 59 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 16 | aptos361_on | FP16 | 3 | 9.9 ± 0.5 | 0.8608 ± 0.0121 | 1.15 ± 0.02 | 1.570 ± 0.121 | n/a | 8.65 ± 0.00 | 12,057 ± 0 |
| 4-bit r16 CLAHE-on lr 0.0005 alpha 16 | messidor_archived | FP16 | 3 | 8.3 ± 0.2 | 0.7826 ± 0.0021 | 1.28 ± 0.00 | n/a | n/a | 8.63 ± 0.00 | 12,093 ± 12 |
| 4-bit r32 CLAHE-on lr 0.0005 alpha 16 | aptos361_on | FP16 | 1 | 11.9 | 0.9005 | 1.10 | 1.543 | n/a | 8.81 | 12,496 |
| 4-bit r8 CLAHE-on lr 0.0005 alpha 16 | aptos361_on | FP16 | 1 | 10.9 | 0.8961 | 1.11 | 1.599 | n/a | 8.61 | 11,976 |
| zero-shot 4-bit | aptos361_off | FP16 | 1 | 8.1 | 0.7177 | 1.39 | 1.041 | n/a | 8.56 | 11,780 |
| zero-shot 4-bit | aptos361_on | FP16 | 1 | 8.7 | 0.7125 | 1.39 | 1.038 | n/a | 8.56 | 11,872 |
| zero-shot 4-bit | messidor_archived | FP16 | 1 | 7.4 | 0.6674 | 1.43 | n/a | n/a | 8.45 | 11,917 |
| zero-shot BF16 | aptos361_on | FP16 | 1 | 21.1 | 1.3441 | 0.74 | 2.552 | n/a | 11.91 | 14,053 |
| RetiZero ViT-L/16 + LoRA r8 (saved fine-tuned checkpoint) | aptos361_clean | CUDA automatic mixed precision (FP16) | 1 | 5.9 | 0.0177 | 54.47 | n/a | 323.7 | 1.18 | 3,326 |
| RetiZero ViT-L/16 + LoRA r8 (saved fine-tuned checkpoint) | aptos367_original | CUDA automatic mixed precision (FP16) | 1 | 6.2 | 0.0182 | 52.48 | n/a | 317.2 | 1.18 | 3,326 |
| RetiZero ViT-L/16 + LoRA r8 (saved fine-tuned checkpoint) (pilot) | pilot_test | CUDA automatic mixed precision (FP16) | 1 | 6.1 | 0.0190 | 46.80 | n/a | 159.0 | 1.18 | 2,960 |
| RetiZero ViT-L/16 + LoRA r8 (saved fine-tuned checkpoint) (pilot) | aptos367_original | CUDA automatic mixed precision (FP16) | 1 | 6.8 | 0.0216 | 45.77 | n/a | 131.5 | 1.18 | 2,920 |

FP32 rows are the fallback jobs for runs whose FP16 scoring overflowed; their times are not comparable to FP16. Per-run and per-job values: `efficiency_training_runs.csv`, `efficiency_scoring_jobs.csv`.
