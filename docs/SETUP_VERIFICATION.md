# Setup review and verification

Reviewed on 28-29 October 2026 against the working tree and the earlier Claude conversation. This document records what was verified, what was corrected, and what remains outside the evidence.

## Findings from the earlier changes

| Earlier change or claim | Review result and correction |
| --- | --- |
| Root requirements included `- r Qwen3/requirements.txt` | The include was corrected to `-r`. The included Qwen3 file still contained an incompatible NumPy/OpenCV combination and outdated or floating pins. All runtime requirement files now specify the tested versions. |
| Install Unsloth separately, then reinstall the latest PyTorch | This could replace the CUDA build or select a mismatched torchvision. The root install now includes the exact CUDA wheels and pinned Unsloth stack in one dependency resolution. A clean pip-only install was tested. |
| xformers was pinned to torch 2.14.1 | The installed package metadata actually requires `torch>=2.10`; its compiled Windows extension reports it was built for torch 2.10.0/CUDA 12.8. The tested Unsloth version falls back to PyTorch attention. The README no longer treats this as proof of a matched extension. |
| The base-model cache was complete if config plus any safetensors file existed | This accepted an interrupted multi-shard download. Validation now reads the index and checks every referenced shard, tokenizer and processor configuration. |
| `--force` re-downloaded everything | It was forwarded only for the Qwen3 base. It now reaches the checkpoint and LoRA Hub calls too. Regression tests assert the flag reaches each API. |
| The download script fetched all startup models | Bio_ClinicalBERT was missing. It is now downloaded and checked alongside the other three assets. |
| The model path needed a Windows user-specific cache path | Runtime paths were already based on `__file__` or Hub model IDs. Shared settings preserve that portability and support consistent cache/model overrides. |
| UTF-8 model logs crashed the backend's cp1252 reader | The fix was present and correct. It remains in place and is covered with actual subprocess bytes including an emoji, a progress-bar glyph, and an invalid UTF-8 byte. |
| A ready model could exit without losing ready state | Output readers now report an unexpected exit even after readiness. Reader and health threads are bound to one process and cancellation event so an unloaded generation cannot overwrite a later runtime. |
| The pandas circular import had been fixed by an install | The logs do not establish its original cause. Plain pip reported pandas already installed. Conda later replaced pandas and NumPy-related packages. The updated setup uses pinned pip wheels and was verified in a clean environment. No specific pandas root cause is claimed. |
| The Word manual was updated and validated | Earlier work checked XML and ZIP integrity but did not render the pages. The text is now generated from README.md; visual review is handled separately so content parity is reproducible. |

## Changes made during this review

- `model_assets.py`: shared portable asset locations, model-ID environment overrides, JSON/LoRA validation, and complete shard checks.
- `download_model_assets.py`: all four assets, consistent skip/force behavior, and atomic file copies so an interrupted copy cannot appear to be a completed checkpoint. Existing non-empty arbitrary weights are not cryptographically verified.
- `requirements.txt` and the backend/frontend/Qwen3 requirement files: one tested dependency combination, including exact CUDA local-version pins, pandas and NumPy 2-compatible OpenCV. The upstream research requirements remain separate.
- `backend/main.py`: UTF-8 subprocess I/O, bounded recent output, actionable process-failure details, generation-specific monitor cancellation, cleanup on timeout, retry of failed live processes, and shared asset validation accepting either supported LoRA weight format.
- `Qwen3/qwen3vl_server.py`: shared base-model override, a clear CUDA failure, and explicit pandas import before Unsloth to expose its first import exception.
- `RetiZero/inference_server.py`: shared text-model override passed to the existing model architecture.
- `frontend/app.py`: allow up to 60 seconds for the prepare request while the previous model shuts down. Model loading itself remains asynchronous from the UI's perspective.
- `check_environment.py`: standalone import/version checks, compiled CUDA torchvision test, and optional local asset checks without loading weights.
- `scripts/smoke_test.py`: actual model preparation, preprocessing, predictions, switching and unload checks through HTTP, with reports and explicit process ownership.
- `tests/`: offline regressions for incomplete downloads, force forwarding, interrupted local copies, process output decoding, crash reporting, cancellation and the UI's ready/error behavior.
- `README.md`, `Afterimage_User_Manual.docx`, `scripts/build_manual.py`, `commmands.txt`, and `CLAUDE.md`: matching setup instructions, troubleshooting, reproducible manual generation and current contributor guidance.
- `.gitignore`: verification environments, logs and interrupted copy staging files stay out of Git.

## Verified environment

Windows x64, Python 3.11.18, NVIDIA GeForce RTX 5070 Ti with approximately 16 GB VRAM. Core package versions:

| Component | Version |
| --- | --- |
| torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| Unsloth / Unsloth Zoo | 2026.10.10 / 2026.10.10 |
| transformers / PEFT | 5.17.0 / 0.21.2 |
| NumPy / pandas / OpenCV | 2.4.6 / 3.0.6 / 5.0.0.93 |
| datasets / bitsandbytes | 4.8.5 / 0.50.2 |
| FastAPI / Streamlit | 0.162.0 / 1.66.1 |

The original `afterimage_env` contained pip and conda packages. A separate `.verification/clean-env` virtual environment was created from the Python 3.11 interpreter, then installed using `python -m pip install -r requirements.txt`. Its packages came from pip; the existing environment was not reinstalled. Both environments were used for real prediction tests.

## Verification results

1. **Fresh dependency resolution and installation:** passed, including the exact CUDA wheels. The dry run used `--ignore-installed --only-binary=:all:`; a subsequent actual install into the clean environment also succeeded.
2. **Dependency consistency:** `python -m pip check` passed in the clean environment.
3. **Environment preflight:** all checks passed in the clean environment, including actual CUDA torchvision NMS, NumPy/pandas/OpenCV operations, FastAPI/Streamlit imports, Unsloth/PEFT/bitsandbytes/datasets imports, vendored RetiZero imports, and all four assets.
4. **Repeat downloader with offline Hub mode:** all four existing valid assets were skipped successfully. No large model re-download was needed.
5. **Regression tests:** 15 tests passed following the atomic-copy correction and UI checks. They do not load model weights or require a network connection.
6. **Streamlit behavior:** the initial upload screen executed through Streamlit AppTest with zero exceptions and the expected model selector and confirmation button. Mocked ready/error responses confirmed that the uploader unlocks only after readiness and stays locked after a backend connection failure.
7. **Real HTTP predictions:** both models passed through a temporary backend using the existing environment, and again using the clean pip environment. Both tests used `HF_HUB_OFFLINE=1` and the bundled `RetiZero/test_image.png`.
8. **Model switching and unloading:** the smoke test confirmed only the selected model was running and both model states ended stopped with no remaining model process.

Clean-environment measurements from one run:

| Model | Load time | Prediction time | HTTP response |
| --- | --- | --- | --- |
| RetiZero | 10.55 s | 0.44 s | 200, grade 2, label Moderate |
| Qwen3-VL | 23.14 s | 28.70 s | 200, grade 2, label Moderate |

These timings are observations from one machine with existing weights and caches. They are not performance guarantees or medical accuracy measurements. The models' output need not agree on other images.

Machine-readable test results and logs are retained locally under ignored `.verification/` (`clean-smoke.json`, `clean-smoke.log`, `clean-preflight.log`, `clean-install.log` and `install-plan.json`). The README's test commands reproduce the checks without requiring these local files to be committed.

## Limits and remaining checks

- The large original model downloads were already present. Skip behavior and offline loading were exercised with real assets; missing/partial and force paths were exercised with isolated files and mocked Hub APIs. A real multi-GB forced re-download was not performed.
- The user exported the generated manual through Word. All six pages were rendered to PNG and reviewed. After the final title-border/caption-color cleanup, the user re-exported the PDF; pages 1 and 6 were inspected again, and page-image hashes confirmed pages 2-5 were unchanged. The DOCX and PDF now match the reviewed text and layout, with two retained application screenshots.
- Streamlit's initial Python render and the backend prediction path were tested. Browser upload widgets and every results-page interaction were not exhaustively automated.
- The locked-down agent sandbox initially stalled Unsloth's writeability probe in the read-only user Hub cache. A diagnostic stack trace located the stall in `unsloth_zoo_downloads.hf_cache._is_writable`. Tests passed with normal cache access. This was separate from the user's earlier pandas traceback.
- Package pins constrain the documented runtime, but do not freeze all transitive dependencies or model revisions. Upgrading packages, changing GPUs, moving platforms, or changing the base model requires another compatibility and prediction check.
- `Qwen3/lr0.0005_epoch4_batchsize8/run_info.json` retains historical training-machine paths. The app does not use those dataset/output paths to locate inference weights. The training record was not rewritten.

## Reference behavior

The implementation was checked against Hugging Face's [download and cache documentation](https://huggingface.co/docs/huggingface_hub/guides/download), the installed Hub function signatures, the installed Unsloth package metadata and fallback message, and pip's [repeatable installation guidance](https://pip.pypa.io/en/stable/topics/repeatable-installs/). Installed versions and executable tests, rather than claims about the latest available release, define the supported setup here.
