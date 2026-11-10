# Pending changes and Qwen3 notebook review

Reviewed on 4 November 2026 against `main` at `0d2b89e` and the current working tree.
No files were staged and no Git commit was created during this review.

## What the commits establish

The repository contains 14 commits, all recorded under `MohammadTaha24`.
Git does not identify which changes were assisted by Claude; the setup verification
document describes earlier Claude-assisted work that is still uncommitted.

| Commit(s) | Project state established |
| --- | --- |
| `fb93f46` | Initial Streamlit frontend, FastAPI backend, model servers, and vendored RetiZero research code. |
| `690498f` through `02d21fd` | Initial setup documentation and model naming. |
| `acef8ba` | Results rendering and conditional flag display changes. |
| `c66d058`, `9fda94b` | Expanded README and Word user manual. |
| `ce5cc00` | Qwen3 training/evaluation scripts and separate APTOS/Messidor notebooks; RetiZero training notebooks. |
| `8cfc4a1` | Preprocessing notebooks and frontend logo assets. |
| `054e055`, `bf2cfeb`, `c50bad1` | Setup/requirements documentation and manual revisions. |
| `0d2b89e` | Backend asset presence checks and application launch commands. |

The application has three tiers: Streamlit sends images to the FastAPI backend,
which preprocesses them and starts one RetiZero or Qwen3 model server at a time.
The Qwen3 training/evaluation material is separate from application inference.

## Existing uncommitted work

These changes were present before the new notebook was written:

| Group | Files and behavior |
| --- | --- |
| Dependency/setup fixes | Root, backend, frontend and Qwen3 requirements pin a Windows/Python 3.11/CUDA combination; the root Qwen3 include is corrected. `check_environment.py` checks imports, installed versions, compiled CUDA torchvision operations and optional model assets. |
| Shared model assets | New `model_assets.py` centralizes portable paths, environment overrides, JSON/LoRA validation and complete model-shard checks. `download_model_assets.py` downloads all four startup assets, skips valid local assets, forwards force options and uses atomic file replacement. |
| Backend lifecycle | `backend/main.py` uses UTF-8 subprocess I/O, retains bounded recent output, reports unexpected exits, ties readers/health monitors to one process generation, stops timed-out processes, supports retries and validates shared asset definitions. |
| Model-server/UI integration | Qwen3 gains the shared base-model setting, an explicit CUDA check and early pandas import. RetiZero uses the shared text-model setting. The frontend allows 60 seconds for model preparation while the prior process shuts down. |
| Regression/smoke tools | New `tests/test_runtime_setup.py`, `tests/test_frontend.py` and `scripts/smoke_test.py` cover asset handling, process lifecycle, frontend ready/error behavior, and optional real HTTP model checks. |
| Documentation | README, launch commands, Word manual, new PDF manual, `scripts/build_manual.py`, `docs/SETUP_VERIFICATION.md` and `CLAUDE.md` describe the revised setup and reproducible manual generation. |
| Historical handoff | `CODEBASE_HANDOFF.md` is an optional historical analysis snapshot. Its claims about unavailable Python 3.11 and tests not running are not the current environment state. Keep it as a dated snapshot if committing it; do not treat it as current setup instructions. |
| Generated inventories | `all_files_list.txt` (~13 MB), `file_structure.txt` (~6.7 MB), and empty `list_all_files.txt` are local file listings. They include generated/cache paths and should remain outside Git. |

The existing DOCX/PDF were not edited or visually re-rendered in this pass.
The earlier visual verification is recorded in `docs/SETUP_VERIFICATION.md`.

## New notebook and required script corrections

`finetuning/Qwen3/train_aptos&messidor.ipynb` targets the **existing Anaconda
`afterimage_env`**. It uses `sys.executable` for all subprocesses, not a hardcoded
Python installation or Colab environment.

The supplied CSV has 4,719 images. The chosen protocol preserves its split:

| Split | APTOS | Messidor | Total |
| --- | ---: | ---: | ---: |
| Train | 2,929 | 0 | 2,929 |
| Validation | 366 | 0 | 366 |
| Test | 367 | 1,057 | 1,424 |

Messidor remains external to training and validation. The source CSV and images
are read-only inputs. The notebook rejects missing/corrupt files, invalid labels,
duplicate image paths, and patient IDs shared between splits. It exports only
paths, labels, patient IDs, source names and split membership; label-bearing
`vlm_prompt` captions never enter the training prompt.

Training uses the existing weighted class sampler, validation-based early stopping,
isolated attempt directories and the saved best adapter. All test images are
evaluated once, then combined/APTOS/Messidor reports are written from those same
predictions. Independent source evaluation passes are optional. Commands, versions,
script/data hashes, predictions, confusion matrices, per-class metrics and logs
are retained under ignored `finetuning/Qwen3/artifacts/`.

Targeted corrections to the three existing Python scripts:

- Support the installed TRL processor argument while retaining the older tokenizer argument when present.
- Let Trainer/Accelerate construct the dataloader while preserving inverse-frequency weighted sampling.
- Apply the configured image size to the training collator and evaluation pixels.
- Allow `--max_length 0` to disable image-token truncation and keep evaluation batch size at 1 by default.
- Include CSV content hashes in dataset-cache identity and refuse silently dropped rows.
- Export processor files alongside the selected adapter; reject a partially specified explicit split.
- Remove duplicate adapter attachment in the standalone evaluator and support this CSV's path/label schema there.
- Decode only newly generated tokens; temperature 0 selects greedy generation.
- Record parse status and strict accuracy, evaluation settings and seed. QWK uses the fixed five-grade scale.

For compatibility, unparseable outputs still fall back to grade 0 for QWK and
legacy accuracy. The notebook highlights parse counts and uses **strict accuracy**
to count every failed parse as wrong. This policy is recorded in each report.

## Git ignore decisions

The updated `.gitignore` retains existing rules and adds:

- Notebook-generated artifacts, trainer checkpoints, adapter folders, model weight
  extensions and known model-weight `.bin` filenames.
- Jupyter checkpoint directories and common Python/test/tool caches.
- Local virtual environments, `.env` secrets, and `.env.*` files, with `.env.example` allowed.
- TensorBoard/W&B output and the three generated repository inventories.

Keep the notebook, source scripts, regression tests, environment checker, asset
module, smoke/manual builders, setup verification and user manuals in version control.
Do not broadly ignore all CSVs, JSON files, notebooks, PDFs or DOCX files: some are
useful project sources/documentation. Generated run CSV/JSON files are covered by
their artifact directory. Existing tracked `run_info.json` metadata remains tracked.

Review `CODEBASE_HANDOFF.md` separately before including it. Its historical status
does not require a blanket ignore rule for handoff or contributor documentation.

## Validation performed in this review

- All 4,719 referenced files exist and passed PNG integrity and 512-by-512 dimension checks.
- No duplicate image paths or source/patient IDs crossing splits were found.
- Notebook JSON/schema and every code cell compile successfully.
- Data-validation/export cells ran in the actual `afterimage_env` Python 3.11.18 interpreter.
- Explicit split arguments, existing-run selection, unlimited greedy evaluation commands,
  UTF-8 subprocess logs, metadata joins, source reports and incomplete-evaluation rejection were exercised.
- Report calculations were checked with synthetic predictions, including an invalid
  grade-0 answer that must reduce strict accuracy. These are validation fixtures,
  not trained-model results.
- All three scripts imported with the installed Unsloth/TRL/Transformers stack.
  The actual `SFTConfig` was constructed with epoch validation/checkpoint selection,
  evaluation batch size 1 and no token truncation. This caught and corrected
  Unsloth silently ignoring the obsolete `evaluation_strategy` argument.
- The real training dataset builder formatted and cached 2,929/366/1,424 rows;
  the evaluation loader retained all 1,424 test rows. The weighted sampler and
  generated-token-only greedy prediction path were checked without loading weights.
- CUDA detected the NVIDIA GeForce RTX 5070 Ti. The installed incompatible
  xformers extension fell back to PyTorch attention, consistent with the existing
  setup verification record; no package reinstall was performed.
- The existing offline regression suite passed: **15 tests**.
- `git diff --check` passed, and representative generated artifacts/secrets/inventories
  were confirmed ignored with `git check-ignore`.

Full model training and held-out inference were not launched. The notebook's
Anaconda setup cell includes installing/registering `ipykernel`, which is currently
missing from `afterimage_env`; no packages were installed or upgraded by this review.

## Suggested commit message

This message covers the existing pending runtime/setup work and the new notebook.
It assumes the actual project sources, tests and documentation listed above are
included, while generated inventories and artifacts are excluded.

```text
Harden Windows runtime setup and add reproducible Qwen3 training

- Pin the verified CUDA/Python stack and add environment preflight checks
- Centralize model assets, validate complete downloads, and copy weights atomically
- Fix Windows model subprocess logging, failure reporting, retries, and cleanup
- Add offline runtime/UI regressions and HTTP smoke-test tooling
- Refresh setup instructions, user manuals, and reproducible manual generation
- Add an Anaconda Qwen3 notebook preserving APTOS splits and external Messidor testing
- Update training/evaluation APIs, image sizing, cache identity, and parse reporting
- Save combined and source-specific predictions and metrics with experiment provenance
- Ignore local environments, secrets, model outputs, and generated file inventories
```

For smaller commits, separate the pre-existing runtime/setup changes from the
Qwen3 notebook and training-script changes. Keep the `.gitignore` rules with the
changes that introduce the corresponding generated artifacts.
