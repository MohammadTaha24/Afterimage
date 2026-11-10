# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Afterimage is a diabetic retinopathy (DR) screening app: a Streamlit frontend, a FastAPI orchestration backend, and two interchangeable model servers (RetiZero and Qwen3-VL + LoRA) that each grade a fundus image 0–4. Research/educational use only. The verified runtime is Windows x64, Python 3.11 and an NVIDIA GPU with at least 16 GB VRAM. Regression tests use unittest; there is no lint/build configuration.

## Running it

Two terminals, backend first (from the repo root, with the conda env active; Python 3.11):

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9000
python -m streamlit run frontend/app.py
```

Setup (full steps in [README.md](README.md)): use `python -m pip install -r requirements.txt` once in a clean environment. The root file pins torch 2.14.1+cu130 and torchvision 0.29.1+cu130 and declares the CUDA index; included files pin the tested Unsloth and array stack. Do not separately upgrade Unsloth or install upstream RetiZero research requirements. Check imports with `python check_environment.py --skip-assets`, run `hf auth login`, download with `python download_model_assets.py`, then check assets with `python check_environment.py`.

The downloader places the checkpoint and LoRA relative to the repo and caches both the Qwen3 base and `emilyalsentzer/Bio_ClinicalBERT`. `model_assets.py` owns paths and validation. All indexed shards must exist before a cache is accepted. `--force` reaches every remote download API; `--clear-qwen3-target` refreshes the local LoRA from cache; `--skip-qwen3-base` and `--skip-retizero-text` defer downloads. Copies use atomic replacement. Set `HF_HOME`, `QWEN3_BASE_MODEL` and `RETIZERO_TEXT_MODEL` consistently before downloading and starting the backend. Use `HF_HUB_OFFLINE=1` only with complete assets.

Verification: `python -m unittest discover -s tests -v` runs fast regressions. `python scripts/smoke_test.py` grades the bundled sample with each real model through HTTP and checks switching/unload. Stop existing model servers first; `--backend-url` intentionally switches/unloads an existing backend's models. Logs are ignored in `.verification/`. [docs/SETUP_VERIFICATION.md](docs/SETUP_VERIFICATION.md) records clean-install evidence and corrections to earlier troubleshooting claims. README is the manual's text source; `scripts/build_manual.py` requires python-docx in a documentation environment, followed by visual PDF review.

## Architecture

Three tiers, all localhost, with hardcoded URLs/ports (9000 backend, 8000 RetiZero, 8001 Qwen3-VL; the frontend hardcodes the 9000 URLs at the top of [frontend/app.py](frontend/app.py)):

```
Streamlit (frontend/app.py) → FastAPI (backend/main.py :9000) → subprocess model server (:8000 or :8001)
```

**Backend is a process supervisor, not just a proxy.** [backend/main.py](backend/main.py) launches each model server as a child process on demand (`/model/prepare`), never keeps both alive (starting one stops the other, since GPU memory is the constraint), and tracks lifecycle in a `ModelRuntime` per model. Loading progress reaches the UI through a stdout protocol: model servers print `PROGRESS:<pct>:<message>` lines (`emit_progress`), a reader thread regex-parses them into `runtime.progress`, and a health-monitor thread polls the child's `/health` (503 until loaded) to flip the state to `ready`. The frontend polls `/model/status` to render the progress bar. When changing a model server's startup, keep emitting `PROGRESS:` lines and a `/health` endpoint, or the backend will report `error`/timeout (900 s).

**`/predict` request flow:** validate upload (JPG/PNG) → save raw to `unprocessed_image_cache/` → preprocess in-process (the backend imports `preprocess_qwen_array` / `preprocess_retizero_array` by prepending `Qwen3/` and `RetiZero/` to `sys.path`) → save to `processed_image_cache/` → POST the processed PNG to the model server → map `grade` to a label. Raw file is deleted after preprocessing succeeds; processed file is deleted once the inference server was reached. The model servers' only contract with the backend is `POST /predict` returning `{"grade": int}`.

**Preprocessing is model-specific and must match training** (the same logic lives in the notebooks under `finetuning/preprocessing/`): both crop to the Otsu-thresholded ROI and square-pad with black; RetiZero then resizes to 224 (no CLAHE); Qwen3 resizes to 512 and applies CLAHE on the LAB L channel. The Qwen3 server's `--img_size` default (512) and RetiZero's `Resize((224, 224))` assume this. The two preprocess files are near-duplicates, so edit both when changing the shared crop/pad logic.

**Model servers:**
- [RetiZero/inference_server.py](RetiZero/inference_server.py) builds `CLIPRModel(vision_type="lora", R=8)` from the vendored upstream repo, takes the vision encoder plus a new `Linear(1024, 5)` head, and loads the fine-tuned state dict with `strict=True`. It imports `iden_modules` from `RetiZero/RetiZero/` (the vendored upstream RetiZero research code, with its own old-pinned `requirements.txt` that is *not* used by the app — avoid editing it or running `timm_models/` code as if it were project code).
- [Qwen3/qwen3vl_server.py](Qwen3/qwen3vl_server.py) loads the 4-bit base via Unsloth `FastVisionModel` and attaches the LoRA with `PeftModel` at startup (inside the FastAPI lifespan, which is why `/health` stays 503 for a while). It must write the upload to a temp file because the processor takes a path; output is parsed from `<answer>X - LABEL</answer>` and unparseable output returns 422.

**Frontend** is one Streamlit script driven by `st.session_state` (`DEFAULTS` dict at the top): a `page` switch between upload and results, a model-load state machine (`idle → requested → loading → ready | error`) that auto-reruns every 0.5 s while loading, and a two-step inference trigger (button sets `pending_inference` and reruns so the overlay renders before the blocking request). HTML/CSS lives in [frontend/templates/](frontend/templates/) (`upload.html`, `results.html`) and is filled with `string.Template.safe_substitute`, so literal `$` in templates must be escaped. The UI shows only the grade/label; the severity explanation text is the `SEVERITY` dict in `app.py`.

## Repo layout notes

- `finetuning/` holds training/eval scripts (Qwen3 via Unsloth + TRL `SFTTrainer`) and Colab-style notebooks (APTOS/Messidor-2) for both models; these are not imported by the app.
- `*/prev/` directories (`backend/prev`, `frontend/prev`, `Qwen3/prev`, `RetiZero/prev`) are older copies of files kept for reference; the live code is the non-`prev` file. Large commented-out blocks inside `backend/main.py`, `frontend/app.py` and `Qwen3/qwen3vl_server.py` are likewise superseded versions.
- `Qwen3/lr0.0005_epoch4_batchsize8/run_info.json` is the only tracked file from the training run (it contains the author's local Windows paths); the adapter itself is gitignored.
- [commmands.txt](commmands.txt) (sic) holds the historical two run commands; prefer the module-based commands above to ensure the active Python environment is used.
- The cache directories and `unsloth_compiled_cache/` are gitignored runtime output.
