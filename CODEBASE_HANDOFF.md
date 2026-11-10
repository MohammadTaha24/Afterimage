# CODEBASE_HANDOFF.md — Afterimage

Written for: an AI engineer with no access to this repository and no memory of earlier sessions.
Analysis date: 2026-10-30. Repository root: `c:\Users\moht2\Documents\Afterimage` (Windows, Git Bash / PowerShell).

Numbering: sections 1–30 follow the required order. Sections 32–38 (AI contribution record) follow section 30 as the spec requests. Section 31 (Source Coverage) is placed last, because the spec asks the document to finish with the coverage report.

---

## 1. Executive Summary

Afterimage is a localhost **diabetic retinopathy (DR) screening research app**. A user uploads a fundus photograph (JPG/PNG) to a Streamlit page. The page chooses one of two model servers, and the app grades the image on a 0–4 scale: No DR, Mild, Moderate, Severe, Proliferative DR. The app is explicitly for research and education (`CLAUDE.md`).

Three runtime tiers, all bound to `127.0.0.1` with hard-coded ports:

```
Streamlit frontend/app.py  ->  FastAPI backend/main.py :9000  ->  model server (child process)
                                                                    RetiZero/inference_server.py :8000
                                                                    Qwen3/qwen3vl_server.py      :8001
```

The backend is a **process supervisor**. It starts exactly one model server at a time as a child process, watches its stdout for `PROGRESS:` lines, polls its `/health`, and stops the other model before starting a new one. Image preprocessing runs in the backend process, using functions imported from `Qwen3/preprocess_qwen.py` and `RetiZero/preprocess_retizero.py`. Model weights are fetched by a separate downloader (`download_model_assets.py`) and validated by `model_assets.py`.

**Key state facts for the next engineer:**

- **The committed `HEAD` (`0d2b89e`, branch `main`) is not the runnable state.** Thirteen tracked files are modified in the working tree, and eight project files are untracked (`CLAUDE.md`, `check_environment.py`, `model_assets.py`, `docs/`, `scripts/`, `tests/`, `Afterimage_User_Manual.pdf`, and an empty `list_all_files.txt`). The working tree contains the refactored backend, the asset layer and the tests. HEAD's `backend/main.py` is an older design (no UTF-8 subprocess handling, a different reader signature). Always analyse and edit the **working tree**, and say so explicitly when comparing against HEAD.
- Git history has **14 commits**, all by the author `MohammadTaha24`. No commit message or trailer attributes work to an AI.
- Large model artifacts exist locally but are gitignored: `RetiZero/best_retizero_dr.pth` (about 1.65 GB) and `Qwen3/lr0.0005_epoch4_batchsize8/best_lora/adapter_model.safetensors` (about 205 MB). Do not read or commit them.
- **No automated tests were executed in this analysis.** The active `python` is 3.12.7 with no project dependencies, and no Python 3.11 interpreter is present. The unittest discovery run failed at import (`No module named 'requests'`, then `'cv2'`). Static checks passed: `py_compile` succeeded for all 14 project Python files inspected.
- The main verified runtime risks are summarised in section 27. The most important are: blocking GPU inference inside an `async def` handler, raw medical images retained when preprocessing fails, no authentication on any endpoint (including the model servers' `/shutdown`), and substantial duplicated logic and dead code.

### AI Development Involvement

- **What I can confirm:** In this session I made no edits to any project file. My actions were read-only inspection, `py_compile` checks, a failed unittest run, `git` read queries, and deleting 14 `*.cpython-312.pyc` bytecode files that my own compile check had generated (gitignored). I also wrote this file.
- **What I cannot confirm:** I have no accessible record of any earlier session, OpenCode or otherwise. No transcript, summary or compaction record is available to me. The task text says I worked on this repository before, but I cannot verify that from anything I can see. Section 32 explains what can and cannot be attributed.
- **Evidence of earlier AI-assisted work in the repo:** `docs/SETUP_VERIFICATION.md` (untracked) is a review record that refers to "the earlier Claude conversation". Several working-tree changes match its claims (section 32). Git cannot show who made those uncommitted changes.

---

## 2. Repository Identity and Analysis Metadata

| Item | Value | Evidence |
|---|---|---|
| Project name | Afterimage — DR Screening | `backend/main.py:127` (`title="Afterimage DR Screening API"`), `frontend/app.py:23` (`page_title`) |
| Purpose | Research/educational DR grading from fundus photos | `CLAUDE.md` |
| Git branch | `main` (remote tracking `origin/HEAD` present) | `git rev-parse --abbrev-ref HEAD` |
| HEAD commit | `0d2b89e` "Update backend main and add commands file" (2026-04-11) | `git log` |
| Commits | 14 | `git rev-list --count HEAD` |
| Working tree | 13 tracked files modified, 8 untracked project items | `git status --short` |
| License file | None found in repo root | `ls -la` |
| CI/CD config | None found (no `.github/`, no Dockerfile) | `ls -la`, `git ls-files` |
| Analysis OS / shell | Windows 11, Git Bash for most commands | session environment |
| Interpreter used for checks | Python 3.12.7 (documented runtime is Python 3.11) | `python --version` |
| Section numbering | Per the spec; section 31 is last (see note above) | — |

Monorepo boundaries: none in the build-tool sense. The repository is a single Python project with subdirectories for each service. `RetiZero/RetiZero/` is a vendored upstream research repository (see section 4).

---

## 3. Technology Stack

Verified from manifests and imports. Versions are pins from `requirements.txt` and included files.

| Layer | Technology | Evidence |
|---|---|---|
| Language | Python 3.11 (documented), code compiles on 3.12 syntax check | `check_environment.py:84`, `py_compile` run |
| Backend | FastAPI 0.162.0, uvicorn[standard] 0.54.0, python-multipart 0.0.32, httpx 0.28.1 | `backend/requirements.txt` |
| Frontend | Streamlit 1.66.1, requests 2.34.2, Pillow 13.0.0 | `frontend/requirements.txt`, `Qwen3/requirements.txt` |
| Image processing | OpenCV (`opencv-python` 5.0.0.93), NumPy 2.4.6 | `backend/requirements.txt` |
| Deep learning (Qwen3) | torch 2.14.1+cu130, torchvision 0.29.1+cu130, Unsloth 2026.10.10, transformers 5.17.0, PEFT 0.21.2, TRL 1.15.0, bitsandbytes 0.50.2, xformers 0.0.35, accelerate 1.15.0, datasets 4.8.5 | `requirements.txt`, `Qwen3/requirements.txt` |
| Deep learning (RetiZero) | Vendored upstream CLIP-style code `iden_modules/` (`CLIPRModel`), timm-style layers, torchvision transforms | `RetiZero/inference_server.py:47`, `RetiZero/RetiZero/` |
| Model hub | huggingface_hub 1.33.0 | `requirements.txt` |
| Test framework | `unittest` (stdlib) + Streamlit `AppTest` | `tests/*.py` |
| HTTP client | `httpx` (backend, smoke test), `requests` (frontend) | `backend/main.py:15`, `frontend/app.py:8` |
| Document generation | python-docx (documentation env only) | `scripts/build_manual.py` docstring |
| Platform | NVIDIA GPU with at least 16 GB VRAM (Qwen3 requires CUDA; RetiZero falls back to CPU) | `Qwen3/qwen3vl_server.py:271`, `RetiZero/inference_server.py:103` |

Not present: a database, a container runtime, a JavaScript build chain, Node/npm, and any cloud SDK.

---

## 4. Complete Repository Structure and File Inventory

Excluded from the inventory: `.git/`, `__pycache__/`, `.verification/` (ignored logs), `processed_image_cache/` and `unprocessed_image_cache/` (runtime output, currently empty), `unsloth_compiled_cache/` (runtime output), `.venv*/`.

### 4.1 Project-owned files

| Path | Lines | Role | Status |
|---|---|---|---|
| `backend/main.py` | 541 | FastAPI process supervisor and `/predict` orchestrator | Modified (working tree) |
| `backend/requirements.txt` | 7 | Backend pins | Modified |
| `backend/prev/main.py` | 144 | Older backend copy, kept for reference | Tracked, superseded |
| `frontend/app.py` | 1090 | Streamlit single-page app (upload, load, results) | Modified (one timeout change) |
| `frontend/requirements.txt` | 3 | Frontend pins | Modified |
| `frontend/templates/upload.html` | 382 | Upload page HTML/CSS template (`$logo_tag` placeholder) | Tracked |
| `frontend/templates/results.html` | 511 | Results page HTML/CSS template | Tracked |
| `frontend/prev/app.py`, `frontend/prev/results.html` | 256, 325 | Superseded copies | Tracked |
| `frontend/logo.png`, `frontend/logo/logo_Notext.png`, `frontend/logo/logo_text.png`, `frontend/logo.zip` | binary | Logo assets (`logo.png` is embedded as base64 by `logo_tag()`) | Tracked |
| `Qwen3/qwen3vl_server.py` | 414 | Qwen3-VL + LoRA inference server (port 8001) | Modified |
| `Qwen3/preprocess_qwen.py` | 85 | Qwen preprocessing (crop, pad, 512, CLAHE) | Tracked |
| `Qwen3/requirements.txt` | 24 | Qwen-environment pins (Unsloth stack) | Modified |
| `Qwen3/prev/qwen3vl_server.py`, `Qwen3/prev/preprocess_qwen.py` | 373, 70 | Superseded copies | Tracked |
| `Qwen3/lr0.0005_epoch4_batchsize8/run_info.json` | 53 | Training run record (contains local Windows paths; not reproduced) | Tracked |
| `RetiZero/inference_server.py` | 209 | RetiZero inference server (port 8000) | Modified |
| `RetiZero/preprocess_retizero.py` | 72 | RetiZero preprocessing (crop, pad, 224, no CLAHE) | Tracked |
| `RetiZero/inference_retizero.py` | 101 | Standalone CLI predictor, **not used by the app** and diverged from the server | Tracked |
| `RetiZero/prev/inference_server.py`, `RetiZero/prev/preprocess_retizero.py` | 251, 56 | Superseded copies | Tracked |
| `RetiZero/test_image.png` | binary | Bundled sample image used by the smoke test | Tracked |
| `model_assets.py` | 89 | Shared asset paths and validation | Untracked (new) |
| `download_model_assets.py` | 377 | Asset downloader (Hugging Face) | Modified |
| `check_environment.py` | 125 | Import, CUDA and asset preflight | Untracked (new) |
| `requirements.txt` (root) | 13 | Root install; pins CUDA torch and includes the three sub-requirements | Modified |
| `scripts/smoke_test.py` | 139 | End-to-end HTTP smoke test through the backend | Untracked (new) |
| `scripts/build_manual.py` | 195 | Builds the Word manual from `README.md` | Untracked (new) |
| `tests/test_runtime_setup.py` | 171 | 13 offline regression tests (downloads, process I/O) | Untracked (new) |
| `tests/test_frontend.py` | 39 | 2 Streamlit `AppTest` UI tests | Untracked (new) |
| `docs/SETUP_VERIFICATION.md` | 84 | Verification record and corrections (see section 32) | Untracked (new) |
| `README.md` | 206 | User/setup manual source text (not read line-by-line in this analysis) | Modified |
| `CLAUDE.md` | 50 | Contributor guidance for AI tools | Untracked (new) |
| `commmands.txt` (sic) | 3 | Historical run commands | Modified |
| `.gitignore` | 16 | Ignore rules (caches, weights, `.verification/`) | Modified |
| `Afterimage_User_Manual.docx` / `.pdf` | binary | Generated manual | Docx modified; PDF untracked |
| `list_all_files.txt` | 0 | Empty stray file | Untracked, unexplained |

### 4.2 Finetuning (not imported by the app)

| Path | Role |
|---|---|
| `finetuning/Qwen3/train_dr_qwen3vl_runsave.py` (713 lines) | Unsloth + TRL training script; imports `unsloth`, `FastVisionModel`, `UnslothVisionDataCollator`. Header inspected only. |
| `finetuning/Qwen3/eval_infer_dr_qwen3vl.py` (354) | Evaluation/inference script. Header inspected only. |
| `finetuning/Qwen3/eval_dataset_dr_qwen3vl_selectrun.py` (263) | Dataset evaluation across runs. Header inspected only. |
| `finetuning/Qwen3/*.ipynb` (2 notebooks) | APTOS and Messidor-2 orchestrators. **Not opened.** |
| `finetuning/RetiZero/*.ipynb` (2 notebooks) | RetiZero training. **Not opened.** |
| `finetuning/preprocessing/*.ipynb` (2 notebooks) | Preprocessing experiments. **Not opened.** |

### 4.3 Vendored RetiZero research code

`RetiZero/RetiZero/` (tracked, about 1,096 files) is an upstream research repository. It holds `iden_modules/` and `clip_modules/` (both tracked, each about 269 files), `retrieval/`, `zeroshot/`, `utils/`, `Finetuning.py`, `ImageRetrieval.py`, `Zeroshot.py`, `main_pretrain.py`, `config.py` and its own `requirements.txt`. The app imports only `iden_modules.CLIPRModel` (`RetiZero/inference_server.py:47`). The `clip_modules/` copy is not imported by the app. **Only the directory structure was inventoried. No vendored file was reviewed individually,** as CLAUDE.md instructs.

### 4.4 Generated and ignored

`RetiZero/best_retizero_dr.pth` (1.65 GB), `Qwen3/.../best_lora/` (adapter 205 MB, tokenizer 11 MB), `unsloth_compiled_cache/`, `.verification/` (logs, JSON reports, a 1.6 MB `install-plan.json`), `processed_image_cache/` and `unprocessed_image_cache/` (empty at analysis time), and `__pycache__/` folders.

---

## 5. High-Level System Architecture

### 5.1 Architectural style

Three independent processes that communicate over HTTP on localhost:

- **Frontend:** a single Streamlit script, `frontend/app.py`. Session state drives the UI (no router library).
- **Backend:** a FastAPI app, `backend/main.py`, with module-level singletons (`runtimes`, `active_model`, `global_lock`). It has no database and no dependency-injection container.
- **Model servers:** two standalone FastAPI apps, each with a `lifespan` that loads the model and exposes `/health`, `/predict` and `/shutdown`.

There is no layered architecture in the usual sense. The backend mixes HTTP handlers, process management, image preprocessing and response mapping in one file.

```mermaid
flowchart LR
    U[User browser] -->|Streamlit websocket| FE[frontend/app.py<br/>Streamlit :8501]
    FE -->|POST /model/prepare<br/>GET /model/status<br/>POST /predict| BE[backend/main.py<br/>FastAPI :9000]
    BE -->|subprocess.Popen + stdout PROGRESS lines| RZ[RetiZero/inference_server.py<br/>:8000]
    BE -->|subprocess.Popen + stdout PROGRESS lines| QW[Qwen3/qwen3vl_server.py<br/>:8001]
    BE -->|GET /health polling| RZ
    BE -->|GET /health polling| QW
    BE -->|POST /predict (processed PNG)| RZ
    BE -->|POST /predict (processed PNG)| QW
    BE --- PRE[preprocess_retizero_array<br/>preprocess_qwen_array<br/>imported via sys.path]
    BE --- CACHE[(unprocessed_image_cache/<br/>processed_image_cache/)]
    RZ --- W1[(RetiZero/best_retizero_dr.pth)]
    RZ --- T1[(Bio_ClinicalBERT via HF cache)]
    QW --- W2[(Qwen3/.../best_lora/)]
    QW --- B2[(Qwen3-VL 4-bit base via HF cache)]
```

### 5.2 Communication

| From | To | Mechanism | Where |
|---|---|---|---|
| Browser | Streamlit | Streamlit websocket (framework-managed) | — |
| Streamlit | Backend | Synchronous `requests` calls with timeouts (5 s to 180 s) | `frontend/app.py:12-14`, `325`, `562`, `600` |
| Backend | Model server | `httpx.AsyncClient` POST, timeout 180 s | `backend/main.py:433` |
| Backend | Model server (health) | Sync `httpx.get`, 5 s timeout, polled every 0.5 s | `backend/main.py:219`, `227` |
| Model server | Backend (progress) | **Child stdout**, lines `PROGRESS:<pct>:<msg>` | `emit_progress` in both servers; regex `PROGRESS_RE` at `backend/main.py:33` |
| Backend | Preprocessing | In-process Python function call | `backend/main.py:26-27`, `260-277` |

### 5.3 Dependency direction

- `frontend/app.py` depends on the backend over HTTP only. It imports nothing from the backend or model code.
- `backend/main.py` imports `preprocess_qwen`, `preprocess_retizero` (from `Qwen3/` and `RetiZero/` via `sys.path.insert`, lines 23-27) and `model_assets` (line 28). It does **not** import the model servers.
- `Qwen3/qwen3vl_server.py` and `RetiZero/inference_server.py` import `model_assets` by inserting the project root into `sys.path` (`qwen3vl_server.py:39-40`, `inference_server.py:37-38`).
- `download_model_assets.py` and `check_environment.py` import `model_assets`.
- Circularity: none found among project modules.

---

## 6. Application Entry Points and Execution Lifecycle

### 6.1 Entry points

| Process | Command (from `CLAUDE.md` and `commmands.txt`) | Code entry |
|---|---|---|
| Backend | `python -m uvicorn backend.main:app --host 127.0.0.1 --port 9000` | `backend/main.py:127` `app = FastAPI(...)` |
| Frontend | `python -m streamlit run frontend/app.py` | `frontend/app.py:1087-1090` (module-level router) |
| RetiZero server | Spawned by backend: `python -u RetiZero/inference_server.py --weights <pth> --host 127.0.0.1 --port 8000` | `RetiZero/inference_server.py:206-209` `__main__` |
| Qwen3 server | Spawned by backend: `python -u Qwen3/qwen3vl_server.py --run_path <run dir> --host 127.0.0.1 --port 8001` | `Qwen3/qwen3vl_server.py:405-414` `__main__` |
| Smoke test | `python scripts/smoke_test.py` (starts its own backend on 9000) | `scripts/smoke_test.py:138-139` |
| Preflight | `python check_environment.py [--skip-assets]` | `check_environment.py:124-125` |
| Asset downloader | `python download_model_assets.py [--force] [--clear-qwen3-target] [--skip-qwen3-base] [--skip-retizero-text]` | `download_model_assets.py:376-377` |
| Tests | `python -m unittest discover -s tests -v` | `tests/` |

### 6.2 Backend startup

1. Module import: `sys.path` insertion (lines 23-24), preprocessing and asset imports (26-28), logging (30), constants (33-86), `ModelRuntime` objects, singletons (`runtimes` at 129, `active_model` at 133, `global_lock` at 134).
2. `@app.on_event("startup")` `startup()` (`backend/main.py:137-141`) creates the two cache directories. `on_event` is deprecated in current FastAPI. Verified by reading only.
3. `@app.on_event("shutdown")` `shutdown()` (`144-147`) calls `_stop_runtime` on both runtimes.

### 6.3 Model load lifecycle (per model)

1. `POST /model/prepare` → `model_prepare` (`backend/main.py:368`) → `_start_runtime` (`292`).
2. `_start_runtime` takes `global_lock`, validates assets (`_assert_model_assets_present`, `280`), stops the other active model, returns early if this model is already `loading` or `ready`, then calls `_stop_runtime` on the target.
3. `subprocess.Popen` with stdout piped, UTF-8 forced via `encoding="utf-8"` and `PYTHONIOENCODING=utf-8` (`backend/main.py:315-328`).
4. Two daemon threads: `_read_process_output` (`172`) parses `PROGRESS:` lines and sets `loading` progress; `_background_health_monitor` (`207`) polls `/health` until 200, then sets `ready`. Timeout is `MODEL_READY_TIMEOUT_SECONDS = 900` (`84`).
5. The model server prints progress from its lifespan (`emit_progress` in `RetiZero/inference_server.py:95-127` and `Qwen3/qwen3vl_server.py:267-296`). Its `/health` returns 503 until `state["ready"]` is true.
6. If the process exits after `ready`, the reader thread sets `error` (`backend/main.py:198-204`, covered by `tests/test_runtime_setup.py` `test_exit_after_ready_is_an_error`).

### 6.4 Shutdown

- `_stop_runtime` (`238-256`): sets the stop event, `terminate()`, waits 20 s, then `kill()`. Joins threads with a 6 s timeout each. Sets state `stopped`.
- The model servers also expose `POST /shutdown`, which sends `SIGTERM` to their own process (`RetiZero/inference_server.py:199-203`, `Qwen3/qwen3vl_server.py:394-399`).

---

## 7. Backend Architecture

### 7.1 Module map (`backend/main.py`, working-tree version)

| Symbol | Line | Responsibility |
|---|---|---|
| `PROJECT_ROOT` | 21 | Repo root, derived from `__file__` |
| `GRADE_LABELS` | 35-41 | Grade-to-label map (0–4) |
| `MODEL_NAMES`, `MODEL_PORTS`, `MODEL_COMMANDS` | 43-76 | Display names, ports 8000/8001, child process argv |
| `ALLOWED_EXTENSIONS`, `ALLOWED_CONTENT_TYPES` | 78-79 | Upload allow-list (JPG/PNG only) |
| `CACHE_RAW`, `CACHE_PROCESSED` | 81-82 | Cache directories under the repo root |
| `MODEL_READY_TIMEOUT_SECONDS`, `PROCESS_TERMINATE_TIMEOUT_SECONDS`, `HEALTHCHECK_INTERVAL_SECONDS` | 84-86 | 900 s, 20 s, 0.5 s |
| `class ModelRuntime` | 89-124 | Per-model state: `process`, `state`, `progress`, `message`, `last_error`, `stdout_lines` (deque of 80), threads, lock, stop event; `to_dict()` |
| `runtimes`, `active_model`, `global_lock` | 129-134 | Module singletons |
| `_err(status, msg)` | 150 | Returns `JSONResponse({"error": msg})` |
| `_validate_image` | 155-169 | Extension, content-type, non-empty, `PIL.Image.verify()` |
| `_read_process_output` | 172-204 | Stdout reader; updates progress; sets error on exit |
| `_background_health_monitor` | 207-235 | Health polling thread |
| `_stop_runtime` | 238-256 | Terminate, kill, join, reset state |
| `_preprocess_bytes` | 260-277 | OpenCV decode, model-specific preprocessing, PNG encode |
| `_assert_model_assets_present` | 280-289 | Calls `require_file` / `validate_lora` |
| `_start_runtime` | 292-348 | Start the child process under `global_lock` |
| `_safe_delete` | 391-394 | `unlink(missing_ok=True)` with a logged warning |
| Commented duplicate `/predict` | 464-518 | Superseded copy (dead code) |

### 7.2 Controllers and services

The backend has no controller/service split. The HTTP handlers hold all logic. The main functions are:

- `model_prepare` → `_start_runtime` → `_assert_model_assets_present` → `require_file`/`validate_lora` (`model_assets.py`).
- `predict` → `_validate_image` → `_preprocess_bytes` → `preprocess_*_array` → `httpx.AsyncClient.post` to the model server → `GRADE_LABELS` mapping.
- `model_unload` → `_stop_runtime`.
- `model_status` → `ModelRuntime.to_dict()` for each runtime.

### 7.3 Lifecycle and state transitions

```
stopped --prepare--> loading --health 200--> ready
loading/ready --process exit or health timeout--> error
any --unload / switch model / shutdown--> stopped
```

`ModelRuntime.state` values: `stopped`, `loading`, `ready`, `error`. The frontend also has `idle` and `requested`, which are frontend-only states (see section 18).

### 7.4 Business-logic placement

Business rules live in four places:

1. `backend/main.py` (process policy: one model at a time, upload validation, cache lifecycle, grade range check at line 455).
2. `Qwen3/qwen3vl_server.py` (`parse_pred_label` at line 100, `ANSWER_RE` at 78: a 422 response for unparseable output).
3. `frontend/app.py` (`SEVERITY` dict at 464-544: user-facing explanations, and the "flag" section shown only for grade != 0 at line 1028).
4. Preprocessing modules (training-matched image transforms).

### 7.5 Backend error handling

- Errors return `JSONResponse` with `{"error": "..."}` and status 400, 409, 500 or 503 (`_err`, line 150).
- The model servers return FastAPI `HTTPException`, whose body is `{"detail": "..."}`. The backend wraps those as `500` with text `Inference server error (<code>): <raw body>` (`backend/main.py:446-447`). The frontend extracts `error` or falls back to raw text (`frontend/app.py:354-359`). **Inconsistent error envelope** (see section 27).
- Exceptions inside `_start_runtime` become `500 {"error": "Failed to prepare ..."}` and set the runtime to `error` (`backend/main.py:375-380`).

---

## 8. Complete API Endpoint Inventory

### 8.1 Backend (port 9000, `backend/main.py`)

| Method | Route | Handler (line) | Request | Response | Calls | Side effects | Auth |
|---|---|---|---|---|---|---|---|
| GET | `/health` | `health` (352) | — | `{"status":"ok","active_model":str\|null}` | — | none | none |
| GET | `/model/status` | `model_status` (357) | — | `{"active_model", "models": {retizero, qwen3vl: {state, progress, message, running, last_error}}}` | `ModelRuntime.to_dict` | none | none |
| POST | `/model/prepare` | `model_prepare` (368, sync `def`) | form `model` ∈ {`retizero`,`qwen3vl`} | `{ok, model, display_name, state, progress, message}` or 400/500 | `_start_runtime` | Spawns child process; may stop the other model | none |
| POST | `/predict` | `predict` (397, `async def`) | multipart `file`, form `model` | `{label, grade, model}` or 400/409/500/503 | `_validate_image`, `_preprocess_bytes`, `httpx` POST to model server | Writes raw and processed files to cache; deletes raw after preprocessing; deletes processed after the model server was reached | none |
| POST | `/model/unload` | `model_unload` (521, sync `def`) | form `model` | `{ok, model, display_name, state, progress, message}` or 400 | `_stop_runtime` | Stops the child process | none |

Startup and shutdown hooks (`startup` 137, `shutdown` 144) are lifecycle events, not routes.

**Dead code:** a commented-out `/predict` implementation at `backend/main.py:464-518`. It lacks the `running` check and the `try/finally` cache cleanup.

### 8.2 RetiZero model server (port 8000, `RetiZero/inference_server.py`)

| Method | Route | Handler (line) | Request | Response | Notes |
|---|---|---|---|---|---|
| GET | `/health` | `health` (145) | — | 200 `{status, device, cuda_device}` or 503 | Ready flag from lifespan |
| POST | `/predict` | `predict` (157, `async def`) | multipart `file` (image/*) | `{grade, label, confidence}` | Only `grade` is used by the backend. Compute is synchronous inside `async def` (line 180-188). |
| POST | `/shutdown` | `shutdown` (199) | — | `{detail}` | Sends `SIGTERM` to own PID. No auth. |

### 8.3 Qwen3-VL model server (port 8001, `Qwen3/qwen3vl_server.py`)

| Method | Route | Handler (line) | Request | Response | Notes |
|---|---|---|---|---|---|
| GET | `/health` | `health` (322) | — | 200 `{status:"ready", device, cuda_device}` or 503 | — |
| POST | `/predict` | `predict` (334, `async def`) | multipart `file` | `{grade}` or 400/422/503 | Writes upload to a temp file (`tempfile.NamedTemporaryFile`, line 364). Generation runs synchronously in an `async def` (line 370). |
| POST | `/shutdown` | `shutdown` (394) | — | `{detail}` | `SIGTERM` to own PID. No auth. |

### 8.4 Route uncertainty

All routes are static string decorators. None is generated dynamically. The only dynamic element is the `model` form field, validated against the `runtimes` dict.

---

## 9. Controllers and Service Dependencies

**Not applicable as an MVC/controller layer.** The backend's module dependencies are:

```
backend/main.py
 ├─ cv2, numpy, PIL, httpx, fastapi            (third-party)
 ├─ preprocess_qwen  (Qwen3/preprocess_qwen.py)  via sys.path
 ├─ preprocess_retizero (RetiZero/preprocess_retizero.py) via sys.path
 └─ model_assets  (model_assets.py)
        └─ no third-party imports; reads JSON and file metadata only

frontend/app.py
 ├─ streamlit, requests, PIL, base64, string.Template
 └─ frontend/templates/*.html  (string.Template.safe_substitute)

Qwen3/qwen3vl_server.py
 ├─ torch, fastapi, uvicorn
 ├─ model_assets (DEFAULT_QWEN3_BASE_MODEL)
 └─ unsloth.FastVisionModel, peft.PeftModel, pandas (imported inside lifespan)

RetiZero/inference_server.py
 ├─ torch, torchvision, fastapi, uvicorn, PIL
 ├─ model_assets (DEFAULT_RETIZERO_TEXT_MODEL)
 └─ RetiZero/RetiZero/iden_modules (CLIPRModel)

download_model_assets.py ─ model_assets; huggingface_hub
check_environment.py      ─ model_assets; download_model_assets.find_cached_model (lazy import)
scripts/smoke_test.py     ─ httpx; spawns backend via subprocess
```

Service-style units that exist: `ModelRuntime` (process supervisor), the two preprocessing functions, the asset functions in `model_assets.py`, and the downloader functions.

---

## 10. Business Logic and Domain Rules

### 10.1 Grading scale

Integer grades 0–4 with the labels below. The same labels are defined in four places (see section 12):

| Grade | Label | Backend `GRADE_LABELS` | Qwen `LABEL_NAMES` | RetiZero `DR_LABELS` | Frontend `SEVERITY` key |
|---|---|---|---|---|---|
| 0 | No DR | yes | yes | yes | `"No DR"` |
| 1 | Mild | yes | yes | yes | `"Mild"` |
| 2 | Moderate | yes | yes | yes | `"Moderate"` |
| 3 | Severe | yes | yes | yes | `"Severe"` |
| 4 | Proliferative DR | yes | yes | yes | `"Proliferative DR"` |

The standalone `RetiZero/inference_retizero.py` uses different labels (`"Normal (No DR)"`, `"Mild NPDR"`, etc.). It is not used by the app.

### 10.2 Model-server output contract

- The backend trusts only `payload["grade"]` (`backend/main.py:451`) and checks the range 0–4 (`455-456`).
- Qwen3 must emit `<answer>X - LABEL</answer>`. Its regex is `ANSWER_RE` (`Qwen3/qwen3vl_server.py:78-81`). Output that does not match gives HTTP 422.
- Grade is the argmax of a 5-way softmax head. RetiZero adds a `Linear(1024, 5)` classifier (`RetiZero/inference_server.py:69`). The server also returns `confidence` (`194`), which the app never displays.

### 10.3 Upload rules

- JPG/JPEG/PNG only, by filename extension and `Content-Type` (`backend/main.py:78-79`, `155-161`).
- Non-empty and decodable by Pillow `verify()` (`163-169`).
- The frontend's uploader enforces the same extensions (`frontend/app.py:767-771`).
- No maximum file size is enforced by the app (only by Streamlit's default upload limit, not verified).

### 10.4 Model-loading rules

- **One model at a time:** `_start_runtime` stops the other model first (`backend/main.py:303-304`). The frontend disables a second load while one is in progress (`frontend/app.py:730-736`, `746`).
- **Upload gate:** the uploader and "Run Analysis" are only enabled when `loaded_model_key == selected model` (`frontend/app.py:760`, `862`).
- **Timeouts:** frontend load timeout 900 s (`LOAD_TIMEOUT_SECONDS`, `frontend/app.py:19`, `593`); backend readiness timeout 900 s (`backend/main.py:84`); prepare request 60 s (`frontend/app.py:565`).

### 10.5 Preprocessing rules (must match training)

- Both preprocessors: OpenCV BGR to RGB, Otsu threshold on grayscale, bounding-box crop, black square padding (`Qwen3/preprocess_qwen.py:18-47`, `RetiZero/preprocess_retizero.py:7-46`).
- RetiZero: resize to 224×224, **no CLAHE** (`preprocess_retizero.py:48-50`).
- Qwen3: resize to 512×512, then CLAHE on the LAB L channel with `clipLimit=2.0`, `tileGridSize=(8,8)` (`preprocess_qwen.py:59-63`, `7-15`).
- The model servers apply their own normalisation. RetiZero applies ImageNet mean/std (`RetiZero/inference_server.py:116-118`). Qwen3 does not normalise in the app; the processor handles it.
- CLAUDE.md states the preprocessing must match training notebooks under `finetuning/preprocessing/`. **That match was not verified in this analysis** (the notebooks were not opened).

### 10.6 Result-presentation rules

- The "What This Means" text comes from `SEVERITY[label]["meaning"]` (`frontend/app.py:997-1024`).
- The "Why this may have been flagged" block appears only when `grade != 0` and `flag` is non-empty (`frontend/app.py:1028`).
- Every results page includes a clinical disclaimer, and the results template (`results.html`) carries disclaimer text (not read line-by-line).

---

## 11. Database Schema and Relationships

**Not applicable.** The application has no database, ORM, migrations, or SQL. Persistence is limited to:

- Filesystem caches: `unprocessed_image_cache/` and `processed_image_cache/`, both created at runtime and gitignored (`backend/main.py:81-82`, `139-140`).
- Model files and the Hugging Face cache (see section 19).
- `.verification/` logs written by `scripts/smoke_test.py`.

Session state in Streamlit is in memory only.

---

## 12. Data Models, DTOs, and Validation

There are no Pydantic models or typed DTOs. Shapes are implicit in dictionaries and form fields.

| Shape | Defined in | Fields | Validation |
|---|---|---|---|
| `ModelRuntime` (class) | `backend/main.py:89-124` | `model_key`, `process`, `state`, `progress`, `message`, `last_error`, `stdout_lines`, `_reader_thread`, `_health_thread`, `_lock`, `_stop_event` | `set_status` clamps progress to 0–100 |
| `/model/status` response | `backend/main.py:357-365` | `active_model`, `models` | — |
| `/predict` response (backend) | `backend/main.py:458-462` | `label`, `grade`, `model` | grade range check |
| `/predict` response (RetiZero) | `RetiZero/inference_server.py:190-196` | `grade`, `label`, `confidence` | — |
| `/predict` response (Qwen3) | `Qwen3/qwen3vl_server.py:391` | `grade` | regex parse of model text |
| Frontend session state | `frontend/app.py:28-55` (`DEFAULTS`) | 26 keys (see section 18) | — |
| `SEVERITY` entries | `frontend/app.py:464-544` | `sev_class`, `short`, `full`, `guidance`, `meaning`, `flag` | — |
| Asset manifest (HF index) | read by `model_assets._validate_weights` | `weight_map` | every shard must exist, not empty; no absolute paths or `..` |
| LoRA config | read by `model_assets.validate_lora` | `peft_type`, `base_model_name_or_path` | required keys |

Mapping logic: the backend maps an integer grade to a label through `GRADE_LABELS` (`backend/main.py:459`). The frontend maps the label back to presentation text through `SEVERITY` (`frontend/app.py:985`).

---

## 13. Frontend Architecture

### 13.1 Application structure

- **Entry:** `frontend/app.py`, run with `streamlit run`. Module-level code runs on every rerun. The router is at lines 1087-1090: `page_results()` if `page == "results"` and `label` is set, otherwise `page_upload()`.
- **Page config:** `st.set_page_config` (`frontend/app.py:22-26`), `layout="centered"`.
- **Templates:** HTML/CSS in `frontend/templates/`, rendered by `render()` (`184-187`) with `string.Template.safe_substitute`. Literal `$` in templates must be escaped. The upload template takes `$logo_tag`. The results template takes `$logo_tag`, `$sev_class`, `$short`, `$full`, `$guidance`, `$grade`, `$ts`, `$model_name`.
- **Inline HTML/CSS:** much of the UI is also written as f-strings inside `app.py` (overlay CSS at 236-300, explanation modal at 61-175).
- **Global state:** `st.session_state` only. No cache, no context provider, no hooks. Constants `BACKEND_*_URL` are at lines 12-14.
- **Auth handling:** none.
- **UI libraries:** Streamlit built-ins (`selectbox`, `button`, `file_uploader`, `progress`, `image`). No component library.
- **Styling conventions:** CSS custom properties (`var(--navy)`, `var(--text-muted)`, `var(--shadow-sm)`) referenced in inline styles. The variables are presumably defined in the templates (not confirmed in this analysis). Google Fonts are imported via `@import` in both templates (`upload.html:2`, `results.html:2`), which makes an external network request at render time.

### 13.2 Session-state machine

`frontend/app.py:41` documents `load_state`: `idle | requested | loading | ready | error`. `requested` is set by `request_model_load` (`574`). `loading` and `ready`/`error` come from backend polling (`refresh_model_status`, `587-630`). Auto-refresh while loading uses `time.sleep(0.5)` then `st.rerun()` (`maybe_autorefresh_loading`, `633-636`).

### 13.3 Two-step inference trigger

1. "Run Analysis" button (`frontend/app.py:864-876`) reads the upload bytes into `pending_file_*` session keys, sets `is_inferencing=True` and `pending_inference=True`, then calls `st.rerun()`.
2. On the next run, `page_upload` (`678-683`) shows the overlay and calls `run_pending_inference`, which makes the blocking `requests.post` (`325`).
3. On success it stores the result and calls `go("results")` (`381`), which sets `page` and calls `st.rerun()`.

This is documented in the code as a deliberate pattern, so the overlay renders before the blocking request.

---

## 14. Frontend Pages, Routes, and Components

Streamlit has no URL routes. "Pages" are session-state values.

| Screen | Trigger | Primary function | Key widgets | Notes |
|---|---|---|---|---|
| Upload / Select model | Default (`page == "upload"`) | `page_upload` (`frontend/app.py:672-878`) | `st.selectbox` (`model_select`, 700-706), "Confirm and Load Model" button (738), `st.file_uploader` (767-771), preview with `st.image` (791), "Run Analysis" (864) | Shows load progress via `show_loading_widgets` (639-669). Upload is locked until ready. |
| Results | `page == "results"` and `label` set | `page_results` (`frontend/app.py:981-1085`) | Template `results.html`, explanation text, flag box, analysed image (`st.image`, 1054), "<- Analyse Another Image" button (1068) | Resets session to `DEFAULTS` but keeps model-load keys (1069-1084). |

### 14.1 Components and helpers

| Name | Line | Purpose |
|---|---|---|
| `show_explanation_modal` | 61 | Renders an overlay explanation card. **No call site in `app.py`** (verified by grep). Dead code. |
| `logo_tag` | 177 | Base64-embeds `logo.png` |
| `render` | 184 | Template renderer |
| `error_box`, `success_box`, `info_box`, `disabled_upload_box` | 190-229 | Styled notices |
| `go` | 232 | Sets page and reruns |
| `show_inference_overlay` | 236 | Full-screen spinner card |
| `run_pending_inference` | 302 | Blocking `POST /predict` and result storage |
| `reset_loading_state`, `request_model_load`, `refresh_model_status`, `maybe_autorefresh_loading`, `show_loading_widgets` | 551-669 | Load flow |
| `page_upload`, `page_results` | 672, 981 | Page renderers |

### 14.2 Commented-out code (dead)

- `frontend/app.py:383-462`: previous `SEVERITY` dict (superseded by the live one at 464).
- `frontend/app.py:807-860`: previous synchronous "Run Analysis" implementation.
- `frontend/app.py:881-979`: previous `page_results` version.

### 14.3 Naming conventions in the frontend

Functions are `snake_case` with `page_` prefixes for screens and `show_` for renderers. Session keys are `snake_case`. Template files use `snake_case.html`. CSS classes use the `af-` prefix (`af-modal-card`, `af-overlay`, `af-preview-card`). The `sev-*` classes are used in `SEVERITY` (`sev-none`, `sev-mild`, `sev-mod`, `sev-sev`, `sev-prolif`).

---

## 15. Frontend-to-Backend API Mapping

| Frontend source | Function | Method and path | Backend handler | Backend service | Request | Response consumed | Triggering user action |
|---|---|---|---|---|---|---|---|
| `frontend/app.py:560-565` | `request_model_load` | POST `/model/prepare` | `model_prepare` (`backend/main.py:368`) | `_start_runtime` (292) | form `model` | `ok`, `state`, `progress`, `message` (only `raise_for_status` checked) | "Confirm and Load Model" button (738) |
| `frontend/app.py:600` | `refresh_model_status` | GET `/model/status` | `model_status` (357) | `ModelRuntime.to_dict` | — | `models[key].state/progress/message/last_error` | Automatic polling while loading (`674`, `764`, `774`, `878`) |
| `frontend/app.py:325-330` | `run_pending_inference` | POST `/predict` | `predict` (397) | `_preprocess_bytes`, model server | multipart `file`, form `model` | `label`, `grade`, `model` | "Run Analysis" button (864) |
| — | — | POST `/model/unload` | `model_unload` (521) | `_stop_runtime` | — | — | **No frontend call.** Unused by the UI. |
| — | — | GET `/health` (backend) | `health` (352) | — | — | — | **No frontend call.** Used by `scripts/smoke_test.py` only. |

### 15.1 Mismatches and gaps

- Unused backend endpoints from the UI: `/model/unload` and `/health` (the smoke test uses both).
- The frontend ignores the `ok`/`state` fields from `/model/prepare`. It relies on polling `/model/status`.
- `run_pending_inference` treats any non-200 response as an error and reads `error` from JSON. It does not distinguish 409 (not ready) from 503 (model server unreachable). Both show as error boxes.
- The frontend does **not** call the model servers directly, which is correct.

---

## 16. Authentication, Authorization, and Security Architecture

- **Authentication:** none, on any service. Every endpoint is unauthenticated.
- **Authorization:** none.
- **Network exposure:** the backend's bind address comes from the uvicorn flags in the documented command (`CLAUDE.md`), not from code. The model servers default to `127.0.0.1` (`Qwen3/qwen3vl_server.py:223`, `RetiZero/inference_server.py:80`) and accept `--host`, which can be set to `0.0.0.0`. **Nothing in the code prevents that.**
- **CORS:** no CORS middleware on any service.
- **Process control:** `POST /shutdown` on each model server sends `SIGTERM` to its own PID (`RetiZero/inference_server.py:199-203`, `Qwen3/qwen3vl_server.py:394-399`). Any local client can call it.
- **Secrets:** no secret values are present in tracked source. A pattern scan of tracked project files (`sk-`, `ghp_`, `hf_`, JWT, Slack, AWS key, PEM markers) found no matches. Hugging Face auth uses `hf auth login` or `HF_TOKEN`, handled by `huggingface_hub` (`download_model_assets.py:102-108`). The token is not stored by the app.
- **Hard-coded identifiers:** Hugging Face repo IDs are in source (`download_model_assets.py:22-26`: `moht24/retizero-dr-model`, `moht24/qwen3-vl-dr-lora`). These are identifiers, not credentials.
- **Local paths in tracked files:** `Qwen3/lr0.0005_epoch4_batchsize8/run_info.json` contains a Windows user-profile path. Not reproduced here.
- **Upload handling:** the backend validates type and decodability before writing (`backend/main.py:406-417`). Filenames are not used as paths, except for `Path(file.filename).suffix` (line 415), which is validated first. The Qwen3 server uses the upload's suffix for its temp file without validation (`qwen3vl_server.py:363`).
- **Medical data:** uploads are fundus images, which are medical data. They are written to `unprocessed_image_cache/` and `processed_image_cache/` and are not deleted in every failure path (section 27).
- **Clinical disclaimer:** present in the UI (`frontend/app.py:157-170`, `results.html`). The app is explicitly not a diagnostic tool.

---

## 17. End-to-End Feature Workflows

### 17.1 Load a model

1. **User intent:** select RetiZero or Qwen3-VL and load it for analysis.
2. **Entry:** `page_upload` (`frontend/app.py:672`), model selectbox (`700`), "Confirm and Load Model" button (`738`).
3. **Frontend:** `reset_loading_state()` then `request_model_load(model_key)` (`560`) posts to `/model/prepare` with a 60 s timeout. On success, state becomes `requested`, then `refresh_model_status()` polls.
4. **Backend:** `model_prepare` (`backend/main.py:368`) → `_start_runtime` (`292`): asset check (`280`), stop other model (`303-304`), spawn child (`315-328`), start reader and health threads (`331-347`).
5. **Model server:** lifespan emits `PROGRESS` lines; loads weights (RetiZero: `torch.load` + `load_state_dict(strict=True)` at `RetiZero/inference_server.py:110-111`; Qwen3: `FastVisionModel.from_pretrained` at `Qwen3/qwen3vl_server.py:280`, `PeftModel.from_pretrained` at `286`).
6. **Readiness:** health monitor sees 200 on `/health` → `ready`. The reader thread updates progress from `PROGRESS` lines while state is `loading`.
7. **Frontend:** `refresh_model_status` (`587`) copies state into `load_state`. When `ready`, it sets `loaded_model_key` (`627-628`) and `show_loading_widgets` shows a success box (`664-665`). The uploader appears (`767`).
8. **Errors:** a backend error sets `error` in `/model/status` and `last_error`. The frontend shows `error_box` (`667`). A 900 s timeout produces an error in both places.
9. **Concurrency:** a second load is blocked in the UI while one is in progress (`730-736`). The backend serialises load requests through `global_lock`.

### 17.2 Analyse an image

1. **User intent:** grade one fundus image with the loaded model.
2. **Entry:** file uploader (`767`), preview (`777-800`), "Run Analysis" (`864`).
3. **Frontend two-step:** store bytes in `pending_file_*`, set `is_inferencing` and `pending_inference`, rerun (`864-876`). Next run shows the overlay (`678-680`) and calls `run_pending_inference(selected_model_key)` (`682`).
4. **HTTP:** `POST /predict` (`325`) with a 180 s timeout.
5. **Backend `predict`** (`backend/main.py:397-462`):
   - state check (`403`): must be `ready` and `is_running()`, else 409.
   - read bytes, validate (`406-411`).
   - save raw to `unprocessed_image_cache/<uuid>.<ext>` (`413-417`).
   - preprocess via `_preprocess_bytes` (`420-423`). On failure: **return 500, raw file kept**.
   - save processed PNG to `processed_image_cache/<uuid>.png` (`426-427`), delete raw (`428`).
   - POST processed PNG to the model server with `httpx.AsyncClient` (`432-437`, 180 s).
   - on `httpx.HTTPError`: return 503, **processed file kept**. On any response, the processed file is deleted in `finally` (`442-444`).
   - non-200 → 500 with server text (`446-447`). Parse `grade` (`449-453`). Range check (`455-456`).
6. **Model server:** RetiZero: decode, transform, `torch.amp.autocast`, softmax, argmax (`RetiZero/inference_server.py:170-196`). Qwen3: write temp file, `_run_inference`, parse `<answer>` (`Qwen3/qwen3vl_server.py:362-391`), delete temp file in `finally` (`380`).
7. **Frontend:** `st.session_state` gets `label`, `grade`, `model_name`, `ts`, `img_bytes`, etc. (`366-373`). Pending keys are cleared (`375-379`). `go("results")` (`381`).
8. **Results page:** `page_results` (`981`) renders the template and the `SEVERITY` text, then the analysed image.
9. **Errors shown:** 400/409/500/503 from the backend show as `error_box` (`354-361`). Connection errors, timeouts and other exceptions show specific messages (`332-349`).
10. **Retry:** the user can press "Run Analysis" again. The uploader keeps its file. There is no automatic retry.

### 17.3 Return to upload ("Analyse Another Image")

`page_results` (`1068-1084`) resets all `DEFAULTS` keys, then restores the model-load keys (`loaded_model_key`, `load_state`, etc.) so the loaded model stays ready. Calls `go("upload")`.

### 17.4 Switch models

Selecting a different model in the selectbox and pressing "Confirm and Load Model" calls `/model/prepare` for the new model. The backend stops the old model before starting the new one (`backend/main.py:303-304`). The frontend shows "A different model is loaded" until the new one is ready (`755`).

### 17.5 Preflight and assets (outside the app)

1. `python download_model_assets.py` → `download_retizero` (`260`), `download_qwen3_lora` (`302`), `download_qwen3_base` (`231`), `download_cached_model("retizero-text")` (`235`).
2. `python check_environment.py` → version checks from manifests (`55-76`), import checks in subprocesses (`CHECKS`, `16-52`), asset checks (`107-119`).

---

## 18. State Management and Data Flow

### 18.1 Frontend session state

Defined in `DEFAULTS` (`frontend/app.py:28-55`). Groups:

| Group | Keys | Written by | Read by |
|---|---|---|---|
| Navigation | `page` | `go` (232) | Router (1087) |
| Result | `label`, `grade`, `model_name`, `ts`, `img_bytes`, `img_name`, `img_size`, `img_kb` | `run_pending_inference` (366-373) | `page_results` |
| Model selection | `selected_model_key`, `selected_model_display` | `page_upload` (689-710) | `page_upload` |
| Load state | `loaded_model_key`, `load_state`, `load_progress`, `load_message`, `load_started_at`, `loading_model_key`, `load_error` | `request_model_load`, `refresh_model_status`, `reset_loading_state` | `show_loading_widgets`, `page_upload` |
| Inference | `is_inferencing`, `pending_inference`, `pending_file_bytes`, `pending_file_name`, `pending_file_type`, `pending_img_size`, `pending_img_kb` | `page_upload` (868-875), `run_pending_inference` | `run_pending_inference` |
| Unused | `show_explanation_modal` | — | — (defined, never set or read) |

Session state is per browser session. Refreshing the page loses results but the backend model stays loaded, so the frontend re-derives state from `/model/status` only when a load is in progress. After a refresh, `loaded_model_key` is `None` even if the backend is ready, so the uploader stays locked until the user presses "Confirm and Load Model" again. **This is a behavioural gap:** the frontend never queries `/model/status` on startup.

### 18.2 Backend in-memory state

- `runtimes` dict (`backend/main.py:129`): one `ModelRuntime` per model, holding a `subprocess.Popen`.
- `active_model` (`133`): the currently running model key, or `None`.
- `global_lock` (`134`): guards `_start_runtime` and `model_unload`. `predict` reads `runtime.state` without the lock (`403`).

### 18.3 Data at rest

- Raw upload → `unprocessed_image_cache/<uuid><ext>` → deleted after preprocessing succeeds.
- Processed PNG → `processed_image_cache/<uuid>.png` → deleted after the model server responds (any status).
- Results are not persisted. The frontend keeps the upload bytes in session memory for display.

---

## 19. External Integrations

| Integration | Where | Purpose | Auth | Failure handling |
|---|---|---|---|---|
| Hugging Face Hub (model weights and tokenizers) | `download_model_assets.py` (`hf_hub_download`, `snapshot_download`) | Download RetiZero checkpoint, Qwen3 LoRA, Qwen3-VL base, Bio_ClinicalBERT | `--token`, `HF_TOKEN`, or `hf auth login` | Exceptions caught in `main` (`371-373`), return 1. Offline validation via `local_files_only=True` (`model_assets` + `find_cached_model`, `209-224`). |
| Qwen3-VL base (`unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`) | `Qwen3/qwen3vl_server.py:280`, default from `model_assets.py:11-13` | 4-bit base model | HF cache | Fetched on first load if not pre-downloaded. Documented, not verified here. |
| Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) | `RetiZero/inference_server.py:64-67` (passed as `bert_type`) | Text encoder required by `CLIPRModel` | HF cache | If absent, may need internet at load time (documented in `download_model_assets.py:89`). |
| Google Fonts | `frontend/templates/upload.html:2`, `results.html:2` | Web fonts loaded by the browser | none | Fallback fonts if blocked (browser-side). |
| CUDA / GPU | `Qwen3/qwen3vl_server.py:271`; `RetiZero/inference_server.py:103` | Inference acceleration | — | Qwen3 raises `RuntimeError` at startup without CUDA. RetiZero falls back to CPU. |
| Model-server subprocesses | `backend/main.py:315` | Process isolation of GPU memory | — | Exit detected by reader thread (`198-204`). |

No email, payment, analytics, or cloud-storage integrations are present.

---

## 20. Configuration and Environment Requirements

### 20.1 Environment variables (names, purposes, defaults)

| Name | Read by | Purpose | Default (non-secret) |
|---|---|---|---|
| `QWEN3_BASE_MODEL` | `model_assets.py:11-13` | Qwen3-VL base model ID or local path | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` |
| `RETIZERO_TEXT_MODEL` | `model_assets.py:14-16` | Text encoder ID or local path | `emilyalsentzer/Bio_ClinicalBERT` |
| `HF_TOKEN` | Hugging Face libraries (documented in `download_model_assets.py:105-107`) | Hub authentication | not set in repo |
| `HF_HOME` | Hugging Face libraries (documented in `CLAUDE.md`) | Cache location | system default |
| `HF_HUB_OFFLINE` | Hugging Face libraries (used by `.verification/` runs) | Forces offline mode | — |
| `PYTHONIOENCODING` | Set by backend for child processes (`backend/main.py:327`) | UTF-8 child output | forced to `utf-8` |
| `TOKENIZERS_PARALLELISM` | `Qwen3/qwen3vl_server.py:42`, training scripts | Silences tokenizer fork warning | `false` (setdefault) |

No `.env` file is read by the code. No `python-dotenv` dependency.

### 20.2 Hard-coded configuration

| Setting | Value | Location |
|---|---|---|
| Backend URL (frontend) | `http://127.0.0.1:9000/...` | `frontend/app.py:12-14` |
| Backend port | 9000 | `backend/main.py` run command (`CLAUDE.md`), `scripts/smoke_test.py:38` |
| RetiZero port | 8000 | `backend/main.py:49`, `63` |
| Qwen3 port | 8001 | `backend/main.py:50`, `74` |
| RetiZero weights path | `RetiZero/best_retizero_dr.pth` | `backend/main.py:59`, `model_assets.py:8` |
| Qwen3 run path | `Qwen3/lr0.0005_epoch4_batchsize8` | `backend/main.py:70`, `model_assets.py:9` |
| Qwen3 image size | 512 | `Qwen3/qwen3vl_server.py:217`, `Qwen3/preprocess_qwen.py:59` |
| RetiZero input size | 224 | `RetiZero/preprocess_retizero.py:48`, `RetiZero/inference_server.py:116` |
| Qwen3 generation | `max_new_tokens=48`, `temperature=0.2`, `min_p=0.1` | `Qwen3/qwen3vl_server.py:220-222` |
| Timeouts | 900 s load, 180 s predict, 60 s prepare, 5 s health | see section 10.4 and 17 |

### 20.3 Required external infrastructure

- Windows x64, Python 3.11, NVIDIA GPU with at least 16 GB VRAM (per `CLAUDE.md`). Verified in `docs/SETUP_VERIFICATION.md` (RTX 5070 Ti, approximately 16 GB).
- CUDA 13.0 torch wheels (`requirements.txt:2-5`).
- Network access to Hugging Face for the first download (or pre-populated caches).

---

## 21. Build, Run, Test, and Deployment Processes

### 21.1 Install

```
python -m pip install -r requirements.txt     # from repo root, in a clean env
python check_environment.py --skip-assets     # imports and CUDA only
hf auth login                                 # if the repos need auth
python download_model_assets.py
python check_environment.py                   # full preflight incl. assets
```

Source: `CLAUDE.md`, `README.md` (not read in this analysis). Install is pip-only, pinned. Do not install Unsloth separately (`requirements.txt` comment at line 1-3 and CLAUDE.md).

### 21.2 Run

```
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9000
python -m streamlit run frontend/app.py
```

Start the backend first. The backend spawns model servers on demand.

### 21.3 Test commands

| Command | What it runs | Requires | Executed in this analysis? |
|---|---|---|---|
| `python -m unittest discover -s tests -v` | 15 offline tests | Full dependency set (cv2, httpx, huggingface_hub, streamlit, requests) | **Attempted; failed at import** (`No module named 'requests'`, then `'cv2'`) |
| `python scripts/smoke_test.py` | Real model loads and predictions through HTTP | GPU, models, Python 3.11 env, ports 8000/8001/9000 free | **Not executed.** Results from earlier runs are recorded in `.verification/smoke-test.json` and `clean-smoke.json` (not re-run). |
| `python check_environment.py` | Imports, CUDA, asset checks | Full env | Not executed |
| `python -m py_compile <files>` | Syntax check | Python only | **Executed**: all 14 project files compiled |

### 21.4 Build

No build step for the app. The Word manual is built by `scripts/build_manual.py` (python-docx, documentation environment only) from `README.md`.

### 21.5 Deployment

**Not applicable.** No deployment configuration, Dockerfile, CI workflow, or service definition exists. The app is a local research tool run by hand.

---

## 22. Naming Conventions and Established Code Patterns

### 22.1 Established conventions (followed consistently)

- Python modules and functions in `snake_case`. Constants in `UPPER_SNAKE_CASE`. Classes in `PascalCase` (`ModelRuntime`, `RetiZeroInference`).
- Private helpers prefixed with `_` (`_start_runtime`, `_stop_runtime`, `_read_process_output`, `_validate_image`).
- Endpoint handlers named after the operation (`health`, `model_status`, `model_prepare`, `predict`, `model_unload`, `shutdown`).
- Shared progress protocol: `PROGRESS:<int>:<message>` (`emit_progress` in both servers and `PROGRESS_RE` in the backend). CLAUDE.md requires this to be kept.
- Each model server exposes exactly `GET /health`, `POST /predict`, `POST /shutdown`.
- Model keys are lowercase `retizero` and `qwen3vl` everywhere.
- Frontend template placeholders are `$snake_case`. Templates use `string.Template`.
- Streamlit session keys are `snake_case`, grouped in `DEFAULTS`.
- Preprocessing functions are named `preprocess_<model>_array` (in-memory) and `preprocess_for_<model>` (file path).

### 22.2 Inconsistencies

- **Error envelopes:** the backend returns `{"error": ...}` and the model servers return `{"detail": ...}` (FastAPI default). See section 7.5.
- **Label spelling:** "Proliferative DR" in code, "Prolif. DR" as the `short` label (`frontend/app.py:530`).
- **File naming:** `commmands.txt` (sic), `Afterimage_User_Manual.docx`, the `prev/` folders, and the `lr0.0005_epoch4_batchsize8` directory name.
- **Duplicate constants:** `LOAD_TIMEOUT_SECONDS` (900) in frontend and `MODEL_READY_TIMEOUT_SECONDS` (900) in backend. `GRADE_LABELS` defined in the backend, Qwen3 (`LABEL_NAMES`), RetiZero (`DR_LABELS`), and frontend (`SEVERITY` keys).
- **Doc-string references to missing files:** `Qwen3/qwen3vl_server.py:55`, `84`, `125` say "mirrored from `infer_one_dr_qwen3vl.py`". That file is **not in the repository** (the nearest is `finetuning/Qwen3/eval_infer_dr_qwen3vl.py`).
- **Pillow pinned twice** in the same environment (`frontend/requirements.txt:3`, `backend/requirements.txt:6`, and root includes both). Harmless but redundant.
- **Mixed sync/async:** `model_prepare` and `model_unload` are sync `def`; `predict` in all three services is `async def` but does blocking work (section 27).

### 22.3 Frontend component naming

Components are Streamlit calls inline in page functions, not reusable components. There are no `components/` or `hooks/` directories. Overlay and modal helpers use the `af-` CSS prefix.

---

## 23. Git History and Architectural Evolution

### 23.1 Commit history (oldest to newest, `git log --reverse` order from the earlier listing)

| Date | Hash | Subject |
|---|---|---|
| 2026-03-30 | `fb93f46` | Initial commit |
| 2026-03-31 | `690498f` | Add initial README for Afterimage project |
| 2026-03-31 | `483d232` | Update model names in README.md |
| 2026-03-31 | `533f605` | Update important notes in README |
| 2026-03-31 | `02d21fd` | Update README with prerequisites and setup instructions |
| 2026-03-31 | `acef8ba` | Fix results page rendering and conditional flag display |
| 2026-04-01 | `c66d058` | Revise README for clarity and updated instructions |
| 2026-04-01 | `9fda94b` | add user manual |
| 2026-04-02 | `ce5cc00` | Add finetuning files |
| 2026-04-09 | `8cfc4a1` | Add preprocessing files and frontend logo assets |
| 2026-04-11 | `054e055` | Update README and requirements |
| 2026-04-11 | `bf2cfeb` | Simplify application usage instructions |
| 2026-04-11 | `c50bad1` | Update README and updated User manual |
| 2026-04-11 | `0d2b89e` | Update backend main and add commands file |

All commits are by `MohammadTaha24`, under two e-mail identities. No commit trailer names an AI tool. Author e-mail addresses are not reproduced here.

### 23.2 What the history shows

- March–April 2026: the app was built and documented. Changes went from initial code, to README and manual work, to fine-tuning and preprocessing files.
- The last commit (`0d2b89e`) added 72 lines to `backend/main.py` and changed `commmands.txt`. The HEAD backend is an older design. The working-tree backend differs substantially (`git diff --stat`: 159 lines changed).
- **Working-tree changes (after HEAD), dated 2026-10-28 to 2026-10-30:** asset validation, the downloader rewrite, the CUDA and pandas preflight, UTF-8 subprocess handling, cancellation-aware health monitoring, the test suite, the smoke test, the manual generator, and the new `docs/SETUP_VERIFICATION.md`. None of this is committed. Its authorship is not recoverable from git (see section 32).

### 23.3 Modules under active change

Highest churn in the working tree (`git diff --stat`): `README.md` (251 lines), `download_model_assets.py` (203), `backend/main.py` (159), `requirements.txt` files, `Afterimage_User_Manual.docx` (binary), `Qwen3/qwen3vl_server.py` (10), `RetiZero/inference_server.py` (9), `frontend/app.py` (4).

---

## 24. Error Handling, Logging, and Observability

### 24.1 Logging

- Backend: `logging.basicConfig(level=INFO)` (`backend/main.py:30`) to stdout. Logger `afterimage`. Child stdout is re-logged with the prefix `[retizero]` or `[qwen3vl]` (`184`).
- RetiZero: logger `retizero_server` (`inference_server.py:25`). Qwen3: logger `qwen3vl` (`qwen3vl_server.py:52`), with timestamps.
- Frontend: uses `print(..., flush=True)` for debugging in `run_pending_inference` (lines 303-364). **Console prints are the frontend's only logging.**
- `PROGRESS:` lines are printed to stdout and logged (`emit_progress`).
- No structured logging, no log files, no rotation, no request IDs.

### 24.2 Observability

- Health: `/health` on every service. The backend's `/model/status` is the only aggregate view.
- Metrics: none. Timing is recorded only in `scripts/smoke_test.py` (load and prediction seconds).
- Errors surfaced to the user: `error_box` messages from backend `error` fields and connection failures.
- Process failures: `_read_process_output` builds `last_error` from the last stdout line that contains `Error:` or `Exception:` (`backend/main.py:201-203`).

### 24.3 Error propagation

- Model-load failures are returned as `last_error` in status, not raised to the client.
- Predict failures are converted to `_err` responses. Exceptions inside `predict` other than the explicit try blocks propagate as FastAPI 500 responses.
- Frontend exceptions around `requests` are caught broadly (`except Exception`, `frontend/app.py:344-349`).

---

## 25. Performance Considerations

### 25.1 Measured (from `.verification/` logs; not re-measured here)

| Model | Load time | Prediction time | Source |
|---|---|---|---|
| RetiZero | 9.05 s / 10.55 s | 0.39 s / 0.44 s | `.verification/smoke-test.json`, `clean-smoke.json` |
| Qwen3-VL | 20.2 s / 23.14 s | 26.27 s / 28.70 s | same |

These are from one machine with warm caches. They are not guarantees.

### 25.2 Structural performance issues (verified by reading)

- **Blocking inference on the event loop:** both model servers run their model call synchronously inside `async def predict` (`Qwen3/qwen3vl_server.py:335-370`, `RetiZero/inference_server.py:158-188`). While one request runs, the server cannot answer `/health` or other requests. With Qwen3 taking about 28 s, this is visible.
- **Frontend polling:** `maybe_autorefresh_loading` sleeps 0.5 s and calls `st.rerun()` on every load-state change (`frontend/app.py:633-636`). This triggers a full script rerun per poll, plus `/model/status` GET calls.
- **Image decoding repeated:** the backend decodes the upload with PIL (`_validate_image`, `backend/main.py:167`), then decodes again with OpenCV (`_preprocess_bytes`, `262`). The frontend also decodes it with PIL for preview (`frontend/app.py:777`) and again for results (`1054`).
- **Single-model policy:** only one model is resident, so switching models pays the full load cost (about 10–23 s each, per the logs). This is intentional (GPU memory), per CLAUDE.md.
- **Global lock held during stop:** `_start_runtime` and `model_unload` call `_stop_runtime` while holding `global_lock`. That can block other load and unload requests for up to 20 s plus 6 s per join (`backend/main.py:244-253`).
- **Large files in memory:** the upload is read fully into memory (`await file.read()`, `backend/main.py:406`). No size limit is enforced.

---

## 26. Testing Strategy and Existing Tests

### 26.1 Test inventory (static, by reading)

**`tests/test_runtime_setup.py`** (171 lines, 13 tests):

| Class | Test | What it asserts |
|---|---|---|
| `AssetTests` | `test_all_shards_and_processor_are_required` | Every index shard and processor config must exist and be non-empty |
| | `test_corrupt_index_does_not_skip_download` | A broken index JSON makes the cache lookup return `None` |
| | `test_lora_bin_and_invalid_config` | LoRA validation accepts `.bin`, rejects empty config |
| | `test_existing_asset_performs_no_remote_download` | No `snapshot_download` call when LoRA is present |
| | `test_force_reaches_retizero_download_api` | `force_download=True` reaches `hf_hub_download` |
| | `test_force_reaches_lora_download_api` | `force_download=True` reaches `snapshot_download` for LoRA |
| | `test_force_reaches_base_download_api` | `force_download=True` reaches `snapshot_download` for base model |
| | `test_clinicalbert_accepts_bin_and_requires_vocab` | Text encoder needs `vocab.txt` |
| | `test_interrupted_copy_preserves_existing_checkpoint` | A failed copy leaves the old file intact and no `.partial` files |
| `ProcessTests` | `test_utf8_and_malformed_output_survive_and_keep_failure_detail` | UTF-8 and malformed bytes in child output; error detail kept |
| | `test_exit_after_ready_is_an_error` | Process exit after `ready` sets `error` |
| | `test_cancelled_health_monitor_cannot_restore_ready` | A cancelled monitor makes no HTTP calls |
| | `test_intentional_stop_keeps_stopped_state` | `_stop_runtime` leaves `stopped` with no error |

**`tests/test_frontend.py`** (39 lines, 2 tests): `AppTest` of the real `frontend/app.py`. Checks the uploader is locked until a mocked `ready` response, and stays locked after a mocked connection error. Both mock `requests`.

### 26.2 Coverage gaps (verified by absence)

- No test for `/predict` (backend), `/model/unload`, `/model/prepare` HTTP behaviour, or upload validation.
- No test for preprocessing functions (`preprocess_qwen_array`, `preprocess_retizero_array`).
- No test for either model server (both require GPU and weights).
- No test for the results page or severity mapping.
- No test for `check_environment.py` or `scripts/build_manual.py`.
- `scripts/smoke_test.py` is an end-to-end script, not a unittest, and needs the GPU and weights.

No coverage percentage can be stated; no coverage tool is configured. Do not invent one.

### 26.3 Mocking strategies

- `unittest.mock.patch.object` on module attributes (`downloads.snapshot_download`, `backend.httpx.get`, `requests.post`).
- Real subprocesses with `sys.executable -c "..."` for process-output tests (no mock).
- Streamlit `AppTest` for the UI (no browser).

### 26.4 Test infrastructure and CI

- No pytest configuration, no `conftest.py`, no tox/nox, no CI workflow.
- `discover -s tests` needs the repo root on `sys.path` (satisfied when run as `python -m unittest` from the root).

### 26.5 Reported results (not verified here)

`docs/SETUP_VERIFICATION.md` (line 57) reports "15 tests passed" after the atomic-copy correction. This analysis did not reproduce that result. The tests were **not executed** here, because the dependencies are not installed in the active interpreter.

---

## 27. Technical Debt, Risks, and Inconsistencies

Severity key: **Critical** (would cause data loss or a security breach in the intended use), **High** (correctness or privacy risk, or a serious maintainability problem), **Medium**, **Low**.

### 27.1 Verified problems

| # | Severity | Component | File:line | Finding | Evidence | Impact | Suggested improvement |
|---|---|---|---|---|---|---|---|
| V1 | High | Repo state | git (HEAD vs working tree) | The committed `HEAD` is not the working system. 13 tracked files and 8 untracked project items differ. | `git status`, `git diff --stat` | Any clone or checkout of `HEAD` runs an older backend (no UTF-8 subprocess handling). Reviewers may test the wrong code. | Commit the working tree in logical changes before further work (not done here). |
| V2 | High | Qwen3 server | `Qwen3/qwen3vl_server.py:335-370` | `predict` is `async def` but calls the blocking `_run_inference` (GPU generation, about 28 s measured). | Source | `/health`, `/shutdown`, and any other request stall during each prediction. The backend's 0.5 s health poll can also stall. | Use `def predict` (FastAPI runs it in a threadpool) or `await run_in_threadpool(...)`. |
| V3 | High | RetiZero server | `RetiZero/inference_server.py:157-196` | Same blocking pattern in `async def predict`. | Source | Same stall, smaller (about 0.4 s). | Same fix. |
| V4 | High | Backend cache | `backend/main.py:419-423` | When preprocessing fails, the raw upload stays in `unprocessed_image_cache/` (the function returns before `_safe_delete(raw_path)` at line 428). | Source | Medical images accumulate on disk with no cleanup job. Privacy risk. | Delete raw in a `finally`, or on preprocessing failure too. |
| V5 | Medium | Backend cache | `backend/main.py:441-444` | On `httpx` failure, the processed PNG is kept on purpose ("we don't know if inference ran"). | Source comment | Same accumulation risk, documented as intentional. | Add a retention limit or cleanup at startup. |
| V6 | Medium | Backend concurrency | `backend/main.py:403` | `predict` reads `runtime.state` and `is_running()` without `global_lock`. A concurrent `/model/unload` can stop the model between the check and the HTTP call. | Source | A predict can fail with 503 or an unclear error. Not verified by a test. | Take a short lock or snapshot the process reference. |
| V7 | Medium | Backend locking | `backend/main.py:298-308`, `529-530` | `_stop_runtime` (which can wait about 26 s per process) runs under `global_lock`. | Source | Other load and unload requests block. | Stop outside the lock, or document as a known wait. |
| V8 | Medium | Model servers | `RetiZero/inference_server.py:199-203`, `Qwen3/qwen3vl_server.py:394-399` | Unauthenticated `POST /shutdown` sends `SIGTERM` to the process. | Source | Any local client can stop a model server. If `--host 0.0.0.0` is used, any network client can. | Remove, or require a token, or bind only to localhost. |
| V9 | Medium | Network exposure | `Qwen3/qwen3vl_server.py:223`, `RetiZero/inference_server.py:80` | `--host` can be set to any address. No auth on any service. | Source | Patient-image endpoints can be exposed if the host is changed. | Document or enforce localhost binding. |
| V10 | Medium | Error envelope | `backend/main.py:150-152` vs `RetiZero/inference_server.py:148`, `Qwen3/qwen3vl_server.py:326` | Backend returns `{"error":...}`, model servers return `{"detail":...}`. | Source | Frontend error text is inconsistent. The backend's 500 for an upstream error includes raw server text (`backend/main.py:447`). | Standardise the envelope. |
| V11 | Medium | Preprocessing | `Qwen3/preprocess_qwen.py:18-47`, `RetiZero/preprocess_retizero.py:7-46` | Crop and pad logic is duplicated (CLAUDE.md acknowledges this). | Source | A change to one file can silently diverge from the other, and from training. | Extract a shared `crop_and_pad` helper. |
| V12 | Medium | Standalone script | `RetiZero/inference_retizero.py` | Diverged from the server: no `bert_type` (so the text encoder is not configured), non-strict `load_state_dict`, `torch.load` without `weights_only`, different labels, unused `numpy` import, and a hard-coded relative `./RetiZero` path. | Source | Results from the script would not match the app. | Delete, or make it call the shared model code. |
| V13 | Medium | Stale reference | `Qwen3/qwen3vl_server.py:55`, `84`, `125` | Comments cite `infer_one_dr_qwen3vl.py`, which is not in the repo. | `git ls-files` | Misleading for maintainers. | Update the comment. |
| V14 | Medium | Dead code | `backend/main.py:464-518`; `frontend/app.py:383-462`, `807-860`, `881-979`; `frontend/app.py:61-175` (unused function) | Large commented-out blocks and one uncalled function. | Source | Hinders navigation; the commented `/predict` lacks cache cleanup. | Delete; rely on git history. |
| V15 | Medium | Frontend startup | `frontend/app.py:28-55`, `page_upload` | After a browser refresh, `loaded_model_key` is `None` even when the backend is `ready`. The frontend never calls `/model/status` at startup. | Source | The user must reload the model after a refresh, even though it is still loaded. | Query `/model/status` on the first run. |
| V16 | Medium | Frontend image handling | `frontend/app.py:777` | `Image.open(uploaded)` has no `try`. A corrupt file raises an unhandled Streamlit exception. | Source | Ugly error page; the backend's `_validate_image` is not reached. | Wrap in `try`, show `error_box`. |
| V17 | Medium | Qwen3 validation | `Qwen3/qwen3vl_server.py:352` | `file.content_type.startswith(...)` raises `AttributeError` if `content_type` is `None`. | Source | 500 instead of 400. Low likelihood through the backend (it sets the type). | Guard for `None`. |
| V18 | Medium | Qwen3 generation | `Qwen3/qwen3vl_server.py:178-184` | `temperature=0.2` and `min_p=0.1` are passed to `generate` without `do_sample=True`. | Source | Whether these settings have any effect depends on transformers defaults. Not verified. | Verify against the transformers version used, and set `do_sample` explicitly if intended. |
| V19 | Low | Asset validation | `download_model_assets.py:286-290` | Redundant check. `RETIZERO_TARGET.name` is a constant, so the check can never fail. | Source | No risk, but misleading. | Delete. |
| V20 | Low | Downloader | `download_model_assets.py:209-224` | `find_cached_model` catches only `LocalEntryNotFoundError`. Other `OSError`s from `snapshot_download` propagate. | Source | An unusual cache error stops `check_environment.py` with a traceback instead of a clean FAIL. | Catch `OSError` as well. |
| V21 | Low | Duplicate constants | `frontend/app.py:19`, `backend/main.py:84` | 900 s timeouts duplicated across processes. | Source | Changing one does not change the other. | Document the coupling, or expose the value through `/model/status`. |
| V22 | Low | Duplicate labels | `backend/main.py:35-41`, `Qwen3/qwen3vl_server.py:70-76`, `RetiZero/inference_server.py:27-33`, `frontend/app.py:464-544` | Grade labels defined four times. | Source | Divergence risk. | Single source (for example `model_assets.py` or a small `labels.py`). |
| V23 | Low | Requirements | `frontend/requirements.txt:3`, `backend/requirements.txt:6` | `Pillow==13.0.0` pinned in two included files. | Source | Redundant, no conflict (same pin). | Keep one. |
| V24 | Low | Deprecated API | `backend/main.py:137`, `144` | `@app.on_event` is deprecated in FastAPI. | Source (general knowledge of FastAPI, not verified against the pinned version) | Future removal risk. | Migrate to `lifespan`. |
| V25 | Low | Frontend network | `frontend/templates/upload.html:2`, `results.html:2` | Google Fonts are loaded from the internet at render time. | Source | Privacy and offline-use note. Fonts fall back if blocked. | Self-host the fonts. |
| V26 | Low | Stray file | `list_all_files.txt` | Empty untracked file at the repo root. | `ls -la` | Clutter. Origin unknown. | Delete after confirming. |
| V27 | Low | Personal data | `Qwen3/lr0.0005_epoch4_batchsize8/run_info.json` | Contains local Windows user-profile paths. | Source | Minor personal data in a tracked file. Acknowledged in `docs/SETUP_VERIFICATION.md:80`. | Leave, or scrub before sharing. |
| V28 | Low | Unused backend routes | `backend/main.py:521` (`/model/unload`), `352` (`/health`) | No frontend caller. | Source | Used only by the smoke test. | Keep; document. |
| V29 | Low | Shell naming | `commmands.txt` | File name typo (sic). | `git ls-files` | Cosmetic. | Rename if the README references are also updated. |

### 27.2 Potential risks requiring further testing

| # | Severity (estimated) | Area | Risk | Why unverified |
|---|---|---|---|---|
| R1 | High | Backend readiness race | If the model process exits between the `/model/status` poll and a `/predict` call, `predict` could send a request to a closed port (`httpx` error, 503). | Needs a running backend and a forced kill. |
| R2 | Medium | Backend health monitor | `_background_health_monitor` calls `_stop_runtime` (`backend/main.py:233`) without `global_lock`. A concurrent `_start_runtime` could interleave. | Needs concurrency testing. |
| R3 | Medium | Windows process termination | `process.terminate()` on Windows maps to `TerminateProcess`. A Qwen3 server with CUDA state may not clean up GPU memory on `terminate`. | Needs GPU runtime testing. The app depends on the OS releasing VRAM before the next load. |
| R4 | Medium | Qwen3 memory growth | `FastVisionModel` and `PeftModel` are loaded once and never reloaded. `torch.cuda.empty_cache()` runs on shutdown only. | Needs GPU runtime testing across many requests. |
| R5 | Medium | Parsing robustness | `ANSWER_RE` requires `<answer>X - LABEL</answer>`. Other output formats return 422. The frequency of this is unknown. | Needs a labelled evaluation run. |
| R6 | Low | Model equivalence | RetiZero's preprocessing relies on `cv2.resize` on an image already at 224 (the server resizes again with PIL). This is a no-op, but the double resize is not verified against training. | Needs comparison with training code in `finetuning/`. |
| R7 | Low | Streamlit rerun cost | The 0.5 s polling loop reruns the entire script, including reading the template and base64 logo, on each poll. | Needs profiling. |

---

## 28. Important Files and Function Reference

### 28.1 Navigation table

| Feature / Responsibility | File Path | Important Symbols | Related Components |
|---|---|---|---|
| Process supervision and model lifecycle | `backend/main.py` | `ModelRuntime`, `_start_runtime`, `_stop_runtime`, `_read_process_output`, `_background_health_monitor`, `global_lock`, `runtimes`, `active_model` | `RetiZero/inference_server.py`, `Qwen3/qwen3vl_server.py`, `model_assets.py` |
| Upload validation and inference orchestration | `backend/main.py` | `predict`, `_validate_image`, `_preprocess_bytes`, `_safe_delete`, `CACHE_RAW`, `CACHE_PROCESSED`, `GRADE_LABELS` | `Qwen3/preprocess_qwen.py`, `RetiZero/preprocess_retizero.py` |
| Backend HTTP surface | `backend/main.py` | `health`, `model_status`, `model_prepare`, `predict`, `model_unload` | Frontend (`frontend/app.py`) |
| Streamlit UI and routing | `frontend/app.py` | `page_upload`, `page_results`, router at 1087, `DEFAULTS` | `frontend/templates/*.html` |
| Model-load UI flow | `frontend/app.py` | `request_model_load`, `refresh_model_status`, `show_loading_widgets`, `maybe_autorefresh_loading` | Backend `/model/prepare`, `/model/status` |
| Inference UI flow | `frontend/app.py` | `run_pending_inference`, `show_inference_overlay`, `pending_*` keys | Backend `/predict` |
| Severity text and flags | `frontend/app.py` | `SEVERITY`, `MODEL_OPTIONS` | Results page |
| Results and upload templates | `frontend/templates/results.html`, `frontend/templates/upload.html` | `$sev_class`, `$short`, `$full`, `$guidance`, `$grade`, `$ts`, `$model_name`, `$logo_tag` | `frontend/app.py` `render()` |
| RetiZero inference | `RetiZero/inference_server.py` | `RetiZeroInference`, `lifespan`, `predict`, `health`, `DR_LABELS` | `RetiZero/RetiZero/iden_modules/` (CLIPRModel), `model_assets.py` |
| RetiZero preprocessing | `RetiZero/preprocess_retizero.py` | `preprocess_retizero_array` | `backend/main.py` |
| Qwen3-VL inference | `Qwen3/qwen3vl_server.py` | `INSTRUCTION`, `ANSWER_RE`, `parse_pred_label`, `resolve_lora_dir`, `_run_inference`, `lifespan`, `predict` | `model_assets.py`, Unsloth, PEFT |
| Qwen3 preprocessing | `Qwen3/preprocess_qwen.py` | `preprocess_qwen_array`, `apply_clahe` | `backend/main.py` |
| Shared asset paths and validation | `model_assets.py` | `RETIZERO_TARGET`, `QWEN3_LORA_TARGET`, `DEFAULT_QWEN3_BASE_MODEL`, `DEFAULT_RETIZERO_TEXT_MODEL`, `validate_lora`, `validate_snapshot`, `require_file` | Servers, downloader, preflight |
| Asset download | `download_model_assets.py` | `download_retizero`, `download_qwen3_lora`, `download_qwen3_base`, `download_cached_model`, `find_cached_model`, `safe_copy_file`, `safe_copy_tree`, `main` | Hugging Face Hub, `model_assets.py` |
| Environment preflight | `check_environment.py` | `CHECKS`, `check_versions`, `main` | `download_model_assets.find_cached_model` |
| End-to-end smoke test | `scripts/smoke_test.py` | `main`, `wait_for_backend` | Backend HTTP API |
| Manual generation | `scripts/build_manual.py` | `build`, `inline`, `cell_border` | `README.md`, python-docx |
| Backend regression tests | `tests/test_runtime_setup.py` | `AssetTests`, `ProcessTests` | `download_model_assets`, `backend.main`, `model_assets` |
| Frontend UI test | `tests/test_frontend.py` | `ModelLoadUITests` | `frontend/app.py` |
| Dependency pins | `requirements.txt`, `backend/requirements.txt`, `frontend/requirements.txt`, `Qwen3/requirements.txt` | pins | All services |
| Setup verification record | `docs/SETUP_VERIFICATION.md` | findings table, verification results | — |
| Contributor guidance | `CLAUDE.md` | run commands, architecture, preprocessing rules | — |
| Training (not in app) | `finetuning/Qwen3/train_dr_qwen3vl_runsave.py`, `finetuning/Qwen3/eval_*.py` | — | Unsloth, TRL |

### 28.2 Most important functions (quick lookup)

- `backend/main.py:predict` (397): end-to-end `/predict`.
- `backend/main.py:_start_runtime` (292): spawn and watch a model server.
- `backend/main.py:_preprocess_bytes` (260): dispatch to the model-specific preprocessor.
- `frontend/app.py:run_pending_inference` (302): blocking inference call and result storage.
- `frontend/app.py:refresh_model_status` (587): poll and copy backend state.
- `Qwen3/qwen3vl_server.py:lifespan` (236): load base model and LoRA.
- `Qwen3/qwen3vl_server.py:parse_pred_label` (100): `<answer>` parsing.
- `RetiZero/inference_server.py:lifespan` (94): load checkpoint with `strict=True`.
- `model_assets.py:validate_snapshot` (68): completeness check for downloaded models.
- `download_model_assets.py:main` (349): downloader entry.

---

## 29. Guide for Future AI-Assisted Development

### 29.1 Where to look first

1. `CLAUDE.md`: run commands, architecture summary and preprocessing rules.
2. `backend/main.py`: all process and request logic.
3. `frontend/app.py`: the entire UI, lines 28-55 (state), 302-381 (inference), 560-669 (load flow), 672-878 (upload page), 981-1085 (results page).
4. `RetiZero/inference_server.py` and `Qwen3/qwen3vl_server.py`: the model contract (`/health`, `/predict`, `PROGRESS:` lines).
5. `model_assets.py` and `download_model_assets.py`: where model files must be.
6. `git status` first. The working tree is the truth, not HEAD.

### 29.2 Implementing a new feature

- **New model:** add a server file exposing `/health`, `/predict` (returning `{"grade": int}`) and `/shutdown`. Emit `PROGRESS:` lines. Add a `ModelRuntime`, a `MODEL_PORTS` and `MODEL_COMMANDS` entry, a `MODEL_NAMES` entry, and a preprocessing branch in `_preprocess_bytes` (`backend/main.py:260`). Add asset checks in `_assert_model_assets_present` (`280`) and `model_assets.py`. Add an option to `MODEL_OPTIONS` and `descriptions` in `frontend/app.py`. Add a test. Update `CLAUDE.md` and README.
- **New UI page:** add a `page_<name>` function and a branch in the router (`frontend/app.py:1087`). Add any new keys to `DEFAULTS`. Add templates to `frontend/templates/` using `$placeholders` and escape literal `$`.
- **New backend endpoint:** add it in `backend/main.py` using the same `_err` envelope. Note the existing envelope inconsistency (section 27, V10).
- **New preprocessing step:** edit both `Qwen3/preprocess_qwen.py` and `RetiZero/preprocess_retizero.py` if it is shared. Check against the notebooks in `finetuning/preprocessing/`.

### 29.3 Modifying existing behaviour (trace across layers)

- Change a grade label: edit `GRADE_LABELS` (backend), `LABEL_NAMES` (Qwen3), `DR_LABELS` (RetiZero), and `SEVERITY` keys (frontend). The strings must match exactly.
- Change a route: update the backend handler and every frontend call listed in section 15. There is no shared client module.
- Change a timeout: check both sides (section 27, V21).
- Change preprocessing: check training (`finetuning/`), both preprocess files, and the model server's `--img_size` / `Resize`.

### 29.4 Files that usually change together

- `backend/main.py` ↔ `frontend/app.py` (any API change).
- `RetiZero/preprocess_retizero.py` ↔ `Qwen3/preprocess_qwen.py` (shared crop logic).
- `model_assets.py` ↔ `download_model_assets.py` ↔ `check_environment.py` (asset paths and validation).
- `requirements*.txt` ↔ `check_environment.py` (`check_versions` reads them).
- `README.md` ↔ `docs/SETUP_VERIFICATION.md` ↔ `scripts/build_manual.py` (docs and manual).

### 29.5 Do not do

- Do not edit `RetiZero/RetiZero/` (vendored upstream code), and do not run `timm_models/` code as project code (CLAUDE.md).
- Do not install upstream RetiZero requirements (CLAUDE.md).
- Do not separately upgrade Unsloth or torch (`requirements.txt` comments).
- Do not edit `*/prev/` files. They are reference copies.
- Do not commit the `.pth`, `.safetensors`, or cache folders (gitignored, large).
- Do not read or print secrets. None are in the repo, but `HF_TOKEN` may be set in the environment.

---

## 30. Unknowns, Unverified Assumptions, and Missing Information

| # | Unknown | Why it matters | What would settle it |
|---|---|---|---|
| U1 | Whether the unit tests pass in the documented environment | The "15 tests passed" claim is only reported in `docs/SETUP_VERIFICATION.md` | Run `python -m unittest discover -s tests -v` in a Python 3.11 env with dependencies |
| U2 | Whether the smoke test passes now | The last recorded run is in `.verification/` (October 28–29) | Run `python scripts/smoke_test.py` with GPU and models |
| U3 | Whether Qwen3 `temperature`/`min_p` are effective (section 27, V18) | Affects output determinism | Inspect the installed transformers `generate` and test with `do_sample` |
| U4 | Whether preprocessing matches training | CLAUDE.md requires it; the notebooks were not opened | Compare `finetuning/preprocessing/*.ipynb` and `finetuning/Qwen3/*.py` with the two preprocess files |
| U5 | Model accuracy | Not measured anywhere in the repo as an evaluation (only one sample image) | Run the evaluation scripts in `finetuning/` on a held-out set |
| U6 | Who authored the uncommitted working-tree changes | Git cannot attribute uncommitted work | The user's own records; no git metadata exists for them |
| U7 | Whether the Hugging Face repos `moht24/retizero-dr-model` and `moht24/qwen3-vl-dr-lora` are accessible and match the local files | The downloader depends on them | Run the downloader with network access; compare file names and sizes |
| U8 | Whether `README.md` is accurate | Not read in this analysis | Read it and compare with the commands in `CLAUDE.md` |
| U9 | Whether the CSS variables used in the templates (`--navy`, etc.) are defined | Affects rendering | Read the `:root` block in both templates |
| U10 | Whether `on_event` deprecation affects the pinned FastAPI 0.162.0 | Deprecation warnings only, not verified | Run the backend and check logs |
| U11 | Whether Streamlit's default upload limit applies | No app-level size limit is set | Upload a large file |
| U12 | GPU memory release after `terminate()` on Windows (section 27, R3) | Affects switching models | Run load/switch cycles with `nvidia-smi` observation |
| U13 | Contents of `list_all_files.txt` and its origin | Empty file, purpose unknown | Ask the user |
| U14 | Contents of the `finetuning/` notebooks | Not opened | Open them and summarise |
| U15 | Whether the vendored `clip_modules/` and `iden_modules/` differ | Both are tracked and similar in size | `diff -r` the two trees (not done) |
| U16 | Whether `frontend/templates/results.html` contains any script | Not fully read | Read the full file |
| U17 | `scripts/build_manual.py` behaviour | Only signatures read | Read the file |
| U18 | Whether the frontend should query `/model/status` on startup (section 27, V15) | Stated as a gap based on reading | Reproduce: load a model, refresh the browser |

---

## 31. Source Coverage and Verification Report

This section is last, per the spec. Section numbering above is as specified.

### 31.1 Totals

- **Tracked files:** 1,136 (`git ls-files`). Of these, 1,096 are under `RetiZero/RetiZero/` (vendored).
- **Untracked project files:** 10 (`CLAUDE.md`, `check_environment.py`, `model_assets.py`, `docs/SETUP_VERIFICATION.md`, `scripts/smoke_test.py`, `scripts/build_manual.py`, `tests/test_runtime_setup.py`, `tests/test_frontend.py`, `Afterimage_User_Manual.pdf`, `list_all_files.txt`).
- **Relevant project source files (excluding vendored, binary, generated, and superseded):** 27 Python/HTML/config files.

### 31.2 Files read directly (full or substantially full)

| File | Read |
|---|---|
| `backend/main.py` | Full (541 lines) |
| `frontend/app.py` | Full (1090 lines) |
| `Qwen3/qwen3vl_server.py` | Full (414) |
| `RetiZero/inference_server.py` | Full (209) |
| `Qwen3/preprocess_qwen.py` | Full (85) |
| `RetiZero/preprocess_retizero.py` | Full (72) |
| `RetiZero/inference_retizero.py` | Full (101) |
| `model_assets.py` | Full (89) |
| `download_model_assets.py` | Full (377) |
| `check_environment.py` | Full (125) |
| `scripts/smoke_test.py` | Full (139) |
| `tests/test_runtime_setup.py` | Full (171) |
| `tests/test_frontend.py` | Full (39) |
| `docs/SETUP_VERIFICATION.md` | Full (84) |
| `requirements.txt`, `backend/requirements.txt`, `frontend/requirements.txt`, `Qwen3/requirements.txt` | Full |
| `.gitignore`, `commmands.txt` | Full |
| `CLAUDE.md` | Provided in the session context (not re-read from disk) |
| `frontend/templates/upload.html`, `results.html` | **Searched only** (placeholders, `@import`, headings). Not read in full. |
| `Qwen3/lr0.0005_epoch4_batchsize8/run_info.json` | Key names and metadata only. Paths not reproduced. |

### 31.3 Files reviewed through metadata, grep, or headers only

- `scripts/build_manual.py`: function list and docstring.
- `finetuning/Qwen3/*.py`: first lines only.
- `finetuning/**/*.ipynb`: **filenames only. Not opened.**
- `RetiZero/RetiZero/**` (vendored): directory listing and counts only.
- `README.md`: existence and size only.
- `backend/prev/main.py`, `frontend/prev/*`, `Qwen3/prev/*`, `RetiZero/prev/*`: existence and diff size only (where diffed).
- `RetiZero/best_retizero_dr.pth`, LoRA weights and tokenizer: sizes only. Not read (by design).
- `frontend/logo*.png`, `RetiZero/test_image.png`, `Afterimage_User_Manual.docx/.pdf`: binary, not read.

### 31.4 Excluded directories and reasons

| Path | Reason |
|---|---|
| `.git/` | Metadata; queried with `git` commands only |
| `__pycache__/`, `*.pyc` | Generated bytecode. I deleted the 14 `cpython-312` files my compile check created. |
| `.verification/` | Ignored run logs and reports. Only the two smoke-test JSON reports were read. |
| `processed_image_cache/`, `unprocessed_image_cache/` | Runtime output, empty at analysis time |
| `unsloth_compiled_cache/` | Runtime output (ignored) |
| `RetiZero/RetiZero/**` | Vendored research code (CLAUDE.md says not to treat it as project code) |

### 31.5 Verification performed

- **Executed:** `python -m py_compile` on 14 project Python files (all succeeded). `git` read-only queries (`status`, `log`, `diff`, `ls-files`, `rev-list`, `grep`). A secret-pattern scan over tracked project files (no matches; file names only were reported). A `grep` for call sites of `show_explanation_modal`. `find` to delete my generated bytecode.
- **Attempted and failed:** `python -m unittest discover -s tests -v` under Python 3.12.7 (missing `requests`, then `cv2`). Not installed, because installing packages into the global interpreter was out of scope.
- **Not executed:** the smoke test, the preflight, any server, the downloader, the Streamlit app.

### 31.6 Major features traced end to end

- Model load (frontend → backend → model server → status → UI): **traced**. Verified from source.
- Image inference (upload → preprocessing → model server → result → results page): **traced**. Verified from source.
- Asset validation and download: **traced**.
- Model switching and unloading: **traced** in source. Verified by smoke-test logs only (not re-run).

### 31.7 Features with incomplete traceability

- Qwen3 generation parameters (U3) and whether generation settings take effect.
- Preprocessing equivalence with training (U4): the notebooks were not opened.
- Frontend CSS variable definitions (U9).
- `scripts/build_manual.py` and the Word/PDF manual pipeline (U17).
- Finetuning code (section 4.2): headers only.

### 31.8 Verified versus inferred

- **Verified by reading source or running a command:** all architecture, routes, line references, data flow, and the HEAD-versus-working-tree differences.
- **Verified by a command:** py_compile success; git history; the absence of CI files, secrets matches, and test runs (which failed to start).
- **Reported by earlier documents, not reproduced:** the 15-test result, smoke-test timings, and the statements in `docs/SETUP_VERIFICATION.md`.
- **Inferred:** the purpose of `list_all_files.txt`; CSS variable definitions; the notebook contents (from filenames only); that `CLAUDE.md`'s architecture matches the code (confirmed for every point I checked, but not every point).

---

## 32. AI Development History and Contributions

### 32.1 What I can confirm about myself

- **In this session:** I made no edits to any project file. Actions taken, in order: read files; listed the git state; compiled 14 Python files (which wrote 14 `cpython-312` bytecode files, which I then deleted); ran the unittest discovery (failed at import); ran read-only grep/git queries; wrote `CODEBASE_HANDOFF.md` at the repo root.
- **Before this session:** I have no accessible record. I have no transcript, summary, compaction record, or OpenCode session data. I cannot describe any earlier change as mine from memory. The task text says I worked on this repo before. I cannot verify or refute that.
- **Tool mismatch:** the task text mentions OpenCode. My environment identifies as Claude Code. This does not change the analysis, but it is one more reason not to assume continuity.

### 32.2 Evidence in the repo of earlier AI-assisted work

The strongest evidence is `docs/SETUP_VERIFICATION.md` (untracked, 84 lines). It says it was "Reviewed on 28-29 October 2026 against the working tree and the earlier Claude conversation". It lists earlier changes and corrections, and it lists changes made during the review. Each claim below was checked against the working-tree code by reading or grep.

| Claim in `docs/SETUP_VERIFICATION.md` | Working-tree evidence | Status |
|---|---|---|
| Root requirements' `-r Qwen3/requirements.txt` include was corrected | `requirements.txt:10-12` uses `-r` | Present |
| Exact CUDA pins and a single dependency resolution | `requirements.txt:2-5` | Present |
| Model cache must include all shards | `model_assets.py:48-65` | Present |
| `--force` reaches every remote download API | `download_model_assets.py:260-275`, `302-323`, `231-233`; tests `test_force_reaches_*` | Present |
| Bio_ClinicalBERT downloaded and checked | `download_model_assets.py:364-365`, `235-257` | Present |
| UTF-8 subprocess I/O | `backend/main.py:315-328` (`encoding="utf-8"`, `PYTHONIOENCODING`) | Present (HEAD does not have it) |
| Output-reader and health-thread cancellation tied to a process | `backend/main.py:172-204`, `207-235`, `_stop_event` | Present |
| Qwen3: shared base model, CUDA check, explicit pandas import | `Qwen3/qwen3vl_server.py:40`, `271-276` | Present |
| RetiZero: shared text model passed to the architecture | `RetiZero/inference_server.py:64-67` | Present |
| Frontend: allow 60 s for the prepare request | `frontend/app.py:565` (`timeout=60`; HEAD has `5`) | Present (uncommitted) |
| `check_environment.py` preflight | `check_environment.py` | Present (untracked) |
| `scripts/smoke_test.py` real-model test | `scripts/smoke_test.py` | Present (untracked) |
| 15 regression tests | `tests/`: 13 + 2 = 15 | Present; **pass status not reproduced** |
| `.gitignore` covers `.verification/` and `*.partial` | `.gitignore:2-4` | Present |
| Manual generated from README by `scripts/build_manual.py` | `scripts/build_manual.py` | Present; output not verified |

What the evidence does **not** show: who made these changes, and whether the review record is accurate in its claims about tests (I could not run them).

### 32.3 Contribution report

Because I cannot attribute any change to myself with evidence, I have not created per-change records as if they were mine. The table below lists the changes that are **present in the working tree and described in `docs/SETUP_VERIFICATION.md`**. Their authorship is unknown to me.

#### Change group A — Asset layer (`model_assets.py`, `download_model_assets.py`)

- **Identification:** untracked `model_assets.py`; modified `download_model_assets.py` (203 lines changed).
- **Files affected:** `model_assets.py` (new), `download_model_assets.py`, `check_environment.py` (new), `tests/test_runtime_setup.py` (new), `.gitignore`.
- **Functions:** `require_file`, `read_json`, `validate_lora`, `_validate_weights`, `validate_snapshot`, `find_cached_model`, `download_cached_model`, `download_retizero`, `download_qwen3_lora`, `safe_copy_file`, `safe_copy_tree`.
- **Original implementation (HEAD):** `download_model_assets.py` at HEAD was not analysed in this session. Its HEAD version was not read. Its prior behaviour is therefore **unknown to me**. The doc says the earlier version fetched only some assets, did not forward `--force` everywhere, and accepted interrupted multi-shard caches.
- **What it does now (verified):** validates every shard before accepting a cache (`_validate_weights`, `model_assets.py:48-65`); copies files atomically via a `.partial` staging file (`safe_copy_file`, `download_model_assets.py:159-170`); forwards `force_download` to all three download APIs (`260-275`, `302-323`, `231-233`); downloads four assets.
- **Verification:** 9 asset tests in `tests/test_runtime_setup.py` (not executed here).
- **Current status:** present in the working tree; not committed.

#### Change group B — Backend process supervision (`backend/main.py`)

- **Identification:** `backend/main.py` modified (159 lines changed vs HEAD).
- **Files affected:** `backend/main.py`, `tests/test_runtime_setup.py` (`ProcessTests`).
- **Functions:** `_read_process_output`, `_background_health_monitor`, `_stop_runtime`, `_start_runtime`, `ModelRuntime`.
- **Original (HEAD) implementation:** `_read_process_output(runtime)` took one argument; `_background_health_monitor(model)` took one argument and used `time.time()`; `model_prepare` and `model_unload` were `async def`; no UTF-8 subprocess encoding; no cancellation event. This is verifiable from `git show HEAD:backend/main.py` (line numbers 191, 220, 371, 524 in that version).
- **What it does now:** UTF-8 child output; generation-scoped stop event; output reader reports an unexpected exit even after `ready`; health monitor cannot restore `ready` after cancellation; `_stop_runtime` cleans up on timeout; bounded 80-line output buffer.
- **Verification:** 4 process tests (not executed here).
- **Current status:** present, uncommitted.

#### Change group C — Model servers (`Qwen3/qwen3vl_server.py`, `RetiZero/inference_server.py`)

- **Qwen3:** shared base-model default (line 40, 205); CUDA precondition check (271-272); explicit `import pandas` before Unsloth (273-276). Verified present.
- **RetiZero:** shared text-model default and `bert_type` (`inference_server.py:37-38`, `64-67`). Verified present.
- **Verification:** not executed here.
- **Current status:** present, uncommitted.

#### Change group D — Frontend timeout (`frontend/app.py`)

- **Change:** `request_model_load` timeout changed from 5 s to 60 s (`frontend/app.py:565`).
- **Why (per the doc):** the previous model must shut down before the new prepare response, which can take longer than 5 s.
- **Verification:** the UI test file mocks `requests`; it does not test the timeout value.
- **Current status:** present, uncommitted. The HEAD value (`timeout=5`, line 565) is verifiable.

#### Change group E — Verification, documentation and tooling

- `check_environment.py`, `scripts/smoke_test.py`, `scripts/build_manual.py`, `tests/test_frontend.py`, `CLAUDE.md`, `docs/SETUP_VERIFICATION.md`, `README.md` (251 lines changed), `Afterimage_User_Manual.docx` (binary), `Afterimage_User_Manual.pdf` (new), `commmands.txt`.
- **Verification:** `.verification/` logs (October 28–29) record a smoke test and a clean install. Their content is consistent with the doc's claims, but I did not re-run them.
- **Current status:** present, uncommitted. `README.md` was not read in this analysis.

---

## 33. AI Change Inventory

| Change | Purpose | Files Affected | Previous Behavior (HEAD) | New Behavior (working tree) | Commit / Evidence | Current Status |
|---|---|---|---|---|---|---|
| Asset layer and validation | Consistent asset paths; complete-cache check; atomic copies | `model_assets.py` (new), `download_model_assets.py`, `check_environment.py` | Partial cache check; some `--force` paths did not reach APIs | Shard-complete checks; `--force` reaches all APIs; atomic copies | Working tree; `docs/SETUP_VERIFICATION.md` | Present, uncommitted; authorship unknown |
| Backend subprocess I/O and lifecycle | Fix Windows cp1252 crash; surface failures; prevent stale ready state | `backend/main.py`, `tests/test_runtime_setup.py` | Default locale encoding; no cancellation; `async` prepare/unload | UTF-8 I/O; generation-scoped threads; sync prepare/unload | `git show HEAD:backend/main.py` vs working tree | Present, uncommitted |
| Qwen3 server preconditions | Clear CUDA error; explicit pandas import; shared model default | `Qwen3/qwen3vl_server.py` | No CUDA check; hard-coded base model | CUDA check; pandas import; `DEFAULT_QWEN3_BASE_MODEL` | Working tree (10 lines) | Present, uncommitted |
| RetiZero text-model config | Pass text encoder to architecture | `RetiZero/inference_server.py` | No `bert_type` | `bert_type=DEFAULT_RETIZERO_TEXT_MODEL` | Working tree (9 lines) | Present, uncommitted |
| Frontend prepare timeout | Allow the old model to shut down first | `frontend/app.py` (1 line) | `timeout=5` | `timeout=60` | Working tree; HEAD line 565 | Present, uncommitted |
| Requirements pinning | One tested dependency set; CUDA pins | `requirements.txt`, `backend/requirements.txt`, `frontend/requirements.txt`, `Qwen3/requirements.txt` | Floating or conflicting pins; `-r` typo | Exact pins; `-r` include | Working tree | Present, uncommitted |
| Verification tooling | Preflight, end-to-end smoke test, manual builder | `check_environment.py`, `scripts/smoke_test.py`, `scripts/build_manual.py` | None | Present | Untracked | Present, untracked |
| Regression tests | Offline tests for downloads, process I/O, UI load flow | `tests/` | None | 15 tests | Untracked | Present; **not executed here** |
| Documentation and manual | Setup guidance; contributor guide; Word/PDF manual | `README.md`, `CLAUDE.md`, `docs/SETUP_VERIFICATION.md`, `Afterimage_User_Manual.*`, `commmands.txt` | Earlier README and manual (HEAD) | Rewritten; manual regenerated | Working tree; `.verification/` logs | Present, uncommitted |
| `.gitignore` hygiene | Keep logs and staging files out of Git | `.gitignore` | Shorter ignore list | Adds `.verification/`, `.venv*/`, `*.partial` | Working tree | Present, uncommitted |

---

## 34. Architectural Decisions Introduced by the AI

**I cannot confirm that I introduced any of these decisions.** The list below describes decisions that appear in the working tree and that `docs/SETUP_VERIFICATION.md` attributes to "the earlier Claude conversation". Each entry records what can and cannot be shown.

1. **Shared asset module.** Previous pattern (HEAD): paths and checks scattered in backend, servers and downloader. New pattern: `model_assets.py` owns paths and validation, imported by all. Requested or proposed: unknown. Follows convention: partially. The `sys.path` insertion is the same mechanism the backend already uses for preprocessing. Pros: one source of truth. Cons: every server now depends on a root-level module through `sys.path` manipulation. Status: present.
2. **Atomic copy staging.** Previous pattern: direct `shutil.copy2` to the target. New pattern: copy to a `.partial` file, then `replace()`. Pros: an interrupted copy cannot leave a valid-looking file. Cons: adds a temp file in the target directory. Status: present (`download_model_assets.py:159-170`).
3. **Generation-scoped process threads.** Previous pattern: reader and health threads not tied to one process. New pattern: each thread receives its own `process` and `stop_event` (`backend/main.py:331-347`). Pros: stale threads cannot overwrite a newer runtime. Cons: more state to reason about. Status: present.
4. **Explicit CUDA preflight in the Qwen3 server** (`qwen3vl_server.py:271`). Pros: clear error. Cons: none significant. Status: present.
5. **Exact CUDA pins in the root requirements.** Pros: reproducible installs. Cons: the install is tied to CUDA 13.0 wheels and Windows. Status: present.

---

## 35. AI Modifications Versus Established Project Conventions

I did not compare against HEAD's full tree in this session. The following compares the working-tree changes against conventions visible in the **current** code and in CLAUDE.md. I cannot say which deviations were introduced by an AI and which by the user.

| Convention (observed) | Working-tree change | Difference and possible impact |
|---|---|---|
| Backend errors use `_err` (`{"error": ...}`) | Model servers still use FastAPI `HTTPException` (`{"detail": ...}`) | Pre-existing inconsistency. The new downloader and tests do not touch it. |
| Model servers emit `PROGRESS:` lines and `/health` (CLAUDE.md requirement) | Preserved | No deviation. |
| Python modules import sibling code through `sys.path` (backend, servers) | `model_assets` is imported this way in both servers | Consistent with the existing pattern, though fragile. |
| Preprocessing duplicated across two files (CLAUDE.md acknowledges) | Unchanged | Not changed. |
| `frontend/app.py` uses inline HTML and Streamlit calls | Unchanged except one line | No deviation. |
| Tests use `unittest` (CLAUDE.md) | Tests use `unittest` and Streamlit `AppTest` | Consistent. |
| `requirements.txt` included by `-r` | Corrected to `-r` | Restores the intended include. |
| Shared constants live in one module | `model_assets.py` added, but grade labels and timeouts remain duplicated | Partially applied. Duplicates remain (section 27, V21, V22). |

---

## 36. Incomplete AI Work and Outstanding Issues

I cannot attribute these items to any earlier AI session. They are listed because they are open in the working tree.

- **Tests not executed.** The 15 tests and the smoke test were not re-run in this analysis. The documented environment (Python 3.11, dependencies) was not available.
- **Uncommitted work.** 13 tracked files and 8 untracked items are not committed. The most important backend and downloader changes exist only in the working tree.
- **Blocking inference in `async def` handlers** (section 27, V2, V3). Not addressed.
- **Raw-image retention on preprocessing failure** (section 27, V4). Not addressed. The processed-file retention on proxy failure is documented as intentional (V5).
- **Dead code left in place** (section 27, V14): a commented `/predict` duplicate and unused functions.
- **Duplicated preprocessing** (V11) and duplicated labels (V22) remain.
- **Standalone `RetiZero/inference_retizero.py`** still diverges from the server (V12).
- **Stale comment reference** to `infer_one_dr_qwen3vl.py` (V13).
- **Frontend does not restore the loaded-model state after refresh** (V15).
- **`list_all_files.txt`** is an empty stray file with unknown purpose (V26, U13).
- **Qwen3 generation settings** may be ineffective (V18, U3).
- **Manual rendering:** `docs/SETUP_VERIFICATION.md` says the manual was reviewed by rendering pages. The render step itself was not run here.

---

## 37. Chronological Development Timeline

Dates are from git commit timestamps and file modification times. I have no record of who performed each step.

| Sequence | Date | Event | Source | Verification | Outcome |
|---|---|---|---|---|---|
| 1 | 2026-03-12 | Qwen3 training run recorded (`run_info.json` `created_at`) | `Qwen3/lr0.0005_epoch4_batchsize8/run_info.json` | — | Best epoch 4, adapter saved |
| 2 | 2026-03-30 | Initial commit | git `fb93f46` | — | Repo created |
| 3 | 2026-03-31 to 2026-04-01 | README, fixes, user manual | git | — | Documentation and results-page fix |
| 4 | 2026-04-02 | Finetuning files added | git `ce5cc00` | — | Training code in repo |
| 5 | 2026-04-09 | Preprocessing files and logo | git `8cfc4a1` | — | Preprocess modules tracked |
| 6 | 2026-04-11 | README, requirements, commands, backend update | git `054e055` to `0d2b89e` | — | Last commit (HEAD) |
| 7 | 2026-10-28 | Working-tree backend, asset layer and verification work begins; `.verification/` install plan and logs | file times; `.verification/install-plan.log` (23:40) | Clean install log (23:46), preflight (23:59) | Per the doc, a clean environment was created and tested |
| 8 | 2026-10-28 23:49 | Smoke-test log and JSON written | `.verification/smoke-test.json` | Both models returned grade 2 (Moderate) | Recorded success (not re-run) |
| 9 | 2026-10-28 to 10-29 | Tests, UI test, manual build and page renders | `.verification/manual-pages`, `manual-page-hashes.json` | Per the doc, pages were rendered and reviewed | Manual regenerated |
| 10 | 2026-10-29 00:00 | Clean smoke test JSON | `.verification/clean-smoke.json` | Both models grade 2 | Recorded success (not re-run) |
| 11 | 2026-10-30 17:46 | `list_all_files.txt` created (empty) | file time | — | Unknown purpose |
| 12 | 2026-10-30 | This analysis (read-only) | this session | Compile check passed; unittest import failed | This document |

The entries for steps 7–10 come from file timestamps and logs. Their content is partly from `docs/SETUP_VERIFICATION.md`, which I cannot independently verify. **Verified history:** git commits 1–6, file timestamps, and log contents. **Reconstructed history:** the narrative in `docs/SETUP_VERIFICATION.md`.

---

## 38. Warnings and Recommendations for the Next AI

### 38.1 Start from the working tree

- Run `git status` and treat the working tree as the current system. HEAD lacks the UTF-8 subprocess handling and the asset layer.
- Do not commit on the user's behalf. The working tree contains large uncommitted changes whose authorship is unknown.

### 38.2 Most important to understand

1. `backend/main.py` process lifecycle: `_start_runtime`, `_stop_runtime`, the reader and health threads, and `global_lock`.
2. `backend/main.py:predict` cache handling and its failure paths (V4, V5).
3. `frontend/app.py` two-step inference trigger and load state machine.
4. `model_assets.py` and its use by the downloader and both servers.
5. The shared `PROGRESS:` and `/health` contract.

### 38.3 Fragile areas

- **Process supervision on Windows.** `terminate()` and VRAM release are untested here (R3). Model switching depends on them.
- **Preprocessing parity** with training (section 10.5). Any change to crop or padding must be made in both preprocessing files and checked against the notebooks.
- **The Qwen3 `<answer>` parser.** Non-conforming model output becomes a 422 that the UI shows as an error (R5).
- **Frontend session state** and the 0.5 s polling loop (R7).
- **Label and timeout duplication** (V21, V22).

### 38.4 Decisions to reconsider

- `async def` handlers that run blocking GPU work (V2, V3). Change to `def`, or offload.
- The four-way duplication of grade labels.
- Keeping processed images after a proxy failure (V5) without a cleanup policy.
- The unauthenticated `/shutdown` endpoints (V8).

### 38.5 Under-tested parts

- Everything under `backend/main.py:predict` (no HTTP test).
- Preprocessing functions (no test).
- Both model servers (need GPU).
- The results page (no test).
- The UI's model-switch path (no test).
- The 15-test result itself: reported, not reproduced here.

### 38.6 Mistakes not to repeat

- Do not treat `docs/SETUP_VERIFICATION.md` as proof. It is a record. Check its claims against code and against a test run.
- Do not assume `HEAD` runs. Use the working tree and say which one you mean.
- Do not commit or publish `.pth`, `.safetensors`, or the caches (gitignored, large).
- Do not leave new `prev/` copies. Use git history.
- Do not run `timm_models/` or the upstream RetiZero requirements as project code.

### 38.7 Conventions that must be preserved

- `PROGRESS:<int>:<message>` lines from both model servers (`CLAUDE.md`).
- `/health` returns 503 until ready (the backend's health monitor depends on it).
- `POST /predict` returns `{"grade": int}` (the backend depends on it).
- Preprocessing matches training. The two preprocess files change together.
- Streamlit `$placeholder` templates with `string.Template`, and literal `$` escaped.
- Module-based run commands (`python -m uvicorn ...`, `python -m streamlit ...`) to use the active environment (`CLAUDE.md`).

### 38.8 Recommended first steps

1. Create a Python 3.11 environment and run `python -m pip install -r requirements.txt`. Then `python -m unittest discover -s tests -v`. Record the real result.
2. Run `python check_environment.py --skip-assets`, then the full check after `download_model_assets.py`.
3. Only then run `python scripts/smoke_test.py` (GPU required). Stop existing servers first.
4. Decide on a commit strategy with the user (group by change group A–E in section 32.3).
5. Address V4 (raw-image retention) and V2/V3 (blocking inference) before any broader change.
