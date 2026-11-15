# Handoff: Afterimage paper experiment progress

## Session Metadata

- Created on 15 November 2026. Results are a point-in-time audit.
- Active project and Git root are `D:\Afterimage_Reproduction`, branch `main`, cloned from local commit `c28aa9e`.
- `D:\Afterimage_Experiments` holds the current Qwen runs, datasets, model snapshots and shared launcher. The Codex workspace on C: holds staging copies of the helpers.
- This is the first handoff for the D: clone. The paper checklist dated 14 November and the older run checklist both predate several completed runs.
- Recent clone commits include `c28aa9e` (codebase handoff), `10066c5` (Phase 1 audit), `d0dc1ef` (Qwen four-run worker), and `085b68c` (paper reproduction).

## Current State Summary

All 21 planned 4-bit Qwen runs and all three matched BF16-base LoRA controls have passed training and evaluation on the 361-image APTOS test cohort. Each of the six rank-16 original-split CLAHE adapters also has a complete BF16-inference score on the separate 367-image cohort. The saved Retraining 2 adapter replay fully matches the archive. The remaining empirical work is the matched RetiZero baseline and verified external Messidor evaluation. The fresh historical training repeats were deferred by the user.

## Architecture Overview

The D: reproduction clone contains the archived training and evaluation code plus notebooks for original-split replay and CLAHE. The 361-image Qwen suite uses the pinned worker under `D:\Afterimage_Experiments\code`. The reviewer-control BF16 runner reuses the pinned splits, trainer, checkpoint format and five-answer scorer. The curated 224-pixel RetiZero dataset uses the same 361 test IDs as the cleaned 512-pixel Qwen dataset.

## Critical Files

| File | Purpose |
| --- | --- |
| finetuning/Qwen3/021_training.ipynb | Sections 3–9 contain evaluation, training, prerequisites and deferred repeats. Section 6 is the BF16 base control. |
| finetuning/Qwen3/remaining_runs.py | Notebook routing to 4-bit cases, BF16 control, RetiZero and verified Messidor. |
| finetuning/Qwen3/train_bf16_control.py | BF16 training, FP16 scoring, resume and hardware checks with saved offload count. |
| finetuning/Qwen3/reproduce_run_02.py | Historical replay and paired original-split CLAHE workflow. |
| finetuning/Qwen3/train_retizero_control.py | RetiZero match, train, resume and evaluate support. |
| finetuning/Qwen3/evaluate_verified_messidor.py | Evaluate saved Qwen adapters once the independently verified external cohort is prepared. |
| PAPER_TRAINING_CHECKLIST.md | 14 November planning snapshot. Use this handoff and saved manifests for completion status. |
| REVIEWER_CHECKLIST.md | Mapping from reviewer comments to methods, experiments and analysis. |

Additional roots are [current Qwen runs](D:/Afterimage_Experiments/runs), [original-split runs and replay](D:/Afterimage_Reproduction/finetuning/Qwen3/artifacts/paper_reproduction/training_batches/run02_reproduction_01), and [shared notebook launcher](D:/Afterimage_Experiments/code/notebook_runner.py). The reviewer source is [EA_REVIEW.docx](C:/Users/moht2/Downloads/EA_REVIEW.docx).

## Work Completed

- Verified all 24 current Qwen runs have passed completion markers, train and score identity agreement, and intact prediction checksums. Independent cross-checks confirmed 361 unique IDs per export and train/score adapter agreement.
- Verified the BF16-base and 4-bit rank-16 CLAHE-on controls match ordered input image and split hashes. Their saved training settings differ in the quantization flag. The base snapshots and 16-layer CPU offload are part of the reported setup.
- Verified the original-split BF16 inference summary for Pair 1 CLAHE on now exists. It scored 82.56% / 0.9166 / 0.9463 across 367 images. Earlier FP16 outputs remain archived.
- Verified the Retraining 2 FP16 replay agrees exactly with all original APTOS and archived Messidor grades and probability scores. APTOS was 80.65% / 0.8915 / 0.9362 and archived Messidor was 50.52% / 0.5528 / 0.7732. This verifies replay of the saved adapter.

### 361-image APTOS results

Each row reports three repeats. Accuracy variability is sample SD in percentage points. QWK is quadratic weighted kappa. AUC is macro one-vs-rest ROC-AUC.

| Qwen condition | Mean accuracy ± SD | Mean QWK | Mean AUC |
| --- | ---: | ---: | ---: |
| 4-bit, rank 8, CLAHE on | 55.77% ± 17.19 | 0.4101 | 0.7904 |
| 4-bit, rank 8, CLAHE off | 50.23% ± 36.39 | 0.5925 | 0.7903 |
| 4-bit, rank 16, CLAHE on | 69.71% ± 13.46 | 0.6701 | 0.8984 |
| 4-bit, rank 16, CLAHE off | 73.96% ± 3.05 | 0.8591 | 0.9341 |
| 4-bit, rank 32, CLAHE on | 49.49% ± 35.77 | 0.5520 | 0.7836 |
| 4-bit, rank 32, CLAHE off | 66.30% ± 14.56 | 0.5865 | 0.8162 |
| 4-bit, rank 16, CLAHE on, previous train/validation membership | 77.38% ± 5.38 | 0.8818 | 0.9388 |
| BF16 base + LoRA, rank 16, CLAHE on | 76.55% ± 1.89 | 0.8856 | 0.9366 |

- BF16 seeds 42/43/44 reached 75.90%, 75.07%, 78.67%. Accuracy differences vs the corresponding 4-bit seeds were +21.33, −5.26 and +4.43 percentage points. Mean difference was +6.83 points and was sensitive to seed 42. BF16 QWK and AUC were higher in all three matched seeds. Report descriptive results and preserve all seeds.
- The four constant 4-bit predictors were rank 8 on / seed 43 (all grade 0), rank 8 off / seed 44 (all grade 4), rank 32 on / seed 44 (all grade 4), and rank 32 off / seed 43 (all grade 0). They finished technically and belong in the results.
- Rank-16 previous-split controls use old train/validation membership and the same 361 holdout IDs. Known identical-image overlap in the previous split limits interpretation of its higher mean.
- BF16 training peaked at approximately 15,385–15,638 MiB of dedicated VRAM on the RTX 5070 Ti. Each run saved train/score logs, system specs, utilization, validation-selected checkpoint and metrics. Full scoring used FP16 arithmetic for comparability with the matched 4-bit references.

### 367-image original-split CLAHE results

| Pair | CLAHE on accuracy / QWK / AUC | CLAHE off accuracy / QWK / AUC |
| --- | ---: | ---: |
| 1 | 82.56% / 0.9166 / 0.9463 | 77.93% / 0.8979 / 0.9357 |
| 2 | 49.32% / 0.0000 / 0.4870 | 64.85% / 0.8270 / 0.9028 |
| 3 | 62.94% / 0.7775 / 0.8936 | 72.48% / 0.8731 / 0.9348 |
| Mean | 64.94% / 0.5647 / 0.7756 | 71.75% / 0.8660 / 0.9244 |

All six values use BF16 evaluation. CLAHE off led in two pairs. CLAHE on predicted grade 0 for every image in pair 2. These are the three original-protocol repeats in each CLAHE branch, all using trainer/LoRA seeds 42/3407. They use the original 2,929/366/367 split, including its known exact duplicate overlap. Keep them separate from the clean 361-image suite and the archived adapter replay.

## Files Modified

- The D: clone contains the archive-matching trainer overlay and the three task notebooks, including 021, plus dedicated historical, BF16 control, RetiZero and Messidor helpers. Use the installed files as the runtime source.
- Section 6 passes 16 CPU-offloaded layers for all three formal runs. Setup reloads project helper modules to fix stale Jupyter imports.
- The shared launcher now forces UTF-8 in Python child output and pipe decoding. Seed 2 previously stopped on a Unicode logging error before an epoch checkpoint. Its subsequent same-folder run passed.
- Training and evaluation worker hashes are part of saved run identity. Keep the installed pinned code intact when resuming or retrying scoring.

## Decisions Made

- Keep the original-split replay, original-split CLAHE ablation and duplicate-excluded suite as separate results. Each answers a different comparison and has its own IDs and protocol.
- Use the teammate's preprocessing recipe with a CLAHE switch and worker count. The optional contrast step is the only pixel operation varied by the branch. The archived replay consumes the archived PNGs.
- Use rank 16, alpha 16 and the same seed pairs for the BF16-base and 4-bit-base comparison. The change is in base weights, with matching FP16 scoring. Keep offload and resource differences visible when reporting efficiency.
- Retain constant predictors and report seed variability. Fixed alpha 16 means rank changes also vary alpha/r.

## Pending Work

### Deferred Items

- Three fresh historical training repeats from archived CLAHE-on inputs are deferred at the user's direction. The replay and three archived retrains already exist. Only add fresh repeats if the manuscript claims fresh reproduction of historical training.

### Blockers and Open Questions

- Matched RetiZero repeats still need the genuine pretrained RetiZero checkpoint from the upstream authors, confirmed provenance and a pinned SHA-256. The team's existing DR fine-tuned checkpoint has a different role. The 224-pixel matched split is already prepared. Seed pairs correspond to trainer seeds 42, 43 and 44. The RetiZero runner's full GPU path still needs validation with genuine pretrained weights.
- Fresh verified Messidor evaluation requires the original image archive, independent grades and the eye-pairing CSV. The archived replay only verifies the old export. The archived derivatives contain four exact duplicate pairs assigned to different patient IDs. Reconcile those with the verified sources before freezing the final external cohort and building its on/off and 224-pixel variants.
- Manuscript work remains on reviewer-response mapping, methods diagrams, tuning justification, confusion and calibration analysis, model comparison, resource and parameter reporting, and literature.

## Immediate Next Steps

1. Refresh the submission checklist from the completed artifacts and assemble the APTOS tables from the saved predictions. Preserve all repeats, per-grade recall, confusion matrices and same-cohort uncertainty. Treat the 14 November checklist as a historical snapshot.
2. Obtain the official RetiZero checkpoint. Locate the raw Messidor archive parts, independent grades and eye pairing in the team data archive in parallel. Confirm cohort provenance before scoring and keep APTOS the training cohort.
3. Once pretrained RetiZero weights are available, open 021, run Setup and Section 7 check, then launch the matched seeds one at a time. Capture resource logs and check the first end-to-end score before the remaining seeds.
4. Once the verified Messidor cohort is frozen, run Section 8 matched external evaluation for the models included in the paper. A BF16-base external scorer on this PC must load with the saved CPU layer offload before attaching the adapter. The current external BF16-base route still needs this integration and a short validation, or a larger GPU.

## Important Context

The user normally starts jobs manually from notebook cells. For help with a new screenshot, check the exact run folder and latest saved attempt first. For a completed seed, rerunning its cell should verify and reuse it. Escalated D: writes may be needed because the Codex writable workspace is on C:. The archive and experiment runs are the evidence and should be treated as read-only during audits.

## Assumptions Made

- Completion means a verified score export and matching training identity, separately from model quality. Failed and interrupted attempts are preserved in each run's history.
- This handoff captures status at audit time. Refresh artifact timestamps and the GPU lock before any new launch.

## Potential Gotchas

- Section 6 preflight only checks readiness. The optional pilot performs two updates on a small train/validation subset. Full named seed cells are below the pilot. Enable `TRAINING_MODE` and rerun Setup before launching.
- Reload Setup after helper changes. Restart the kernel after shared launcher updates to avoid keeping an old import in memory. The 16-layer setting was near the card's memory ceiling and should stay pinned for completed controls.
- GPU lock is shared across notebooks at `D:\Afterimage_Experiments\runs\.gpu.lock`. A saved status can be stale after hard termination, so verify the process/log state before deciding to resume.
- Preserve FP16 archives and report BF16 original-split results with all six adapters evaluated at the same precision. Do not infer the 367-image score from the 361-image cohort.

## Environment State

- Use `C:\Users\moht2\.conda\envs\afterimage_env\python.exe` and the Python (afterimage_env) Jupyter kernel. Training jobs, caches, data, weights and logs live on D:.
- RTX 5070 Ti (16,303 MiB as reported by nvidia-smi), approximately 63 GiB system RAM. Model snapshots are under [D: assets](D:/Afterimage_Experiments/assets). The official BF16 snapshot uses Qwen/Qwen3-VL-8B-Instruct revision 1dd1e02d981403da25ed73d43430e4ef598cb94b.
- All formal Qwen runs had terminal completion artifacts and the compute-process query showed no separate GPU job at the time of audit. No new training or scoring was launched for this handoff.

## Related Resources

- [RetiZero authors' repository and pretrained checkpoint instructions](https://github.com/LooKing9218/RetiZero)
- [Current 4-bit and BF16 run folders](D:/Afterimage_Experiments/runs)
- [Retraining 2 replay report](D:/Afterimage_Reproduction/finetuning/Qwen3/artifacts/paper_reproduction/training_batches/run02_reproduction_01/replay_run_02/replay_comparison.json)
