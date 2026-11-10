# Afterimage User Manual

Afterimage grades retinal fundus images with RetiZero or Qwen3-VL with a LoRA adapter. A Streamlit interface sends images to a FastAPI backend, which preprocesses them and starts one model server at a time.

These instructions target **Windows x64, Anaconda Prompt, Python 3.11 and an NVIDIA GPU with at least 16 GB VRAM**. The pinned CUDA 13.0 wheels require a compatible NVIDIA driver. Other platforms and GPU configurations need separate validation. See [setup verification](docs/SETUP_VERIFICATION.md) for the reviewed changes, test results and limits.

## 1. Clone and create the environment

Install Anaconda or Miniconda first. In Anaconda Prompt:

```bat
git clone https://github.com/MohammadTaha24/Afterimage.git
cd Afterimage
conda create -n afterimage_env python=3.11 -y
conda activate afterimage_env
python -m pip install --upgrade pip
```

Run subsequent commands from the repository root. If `afterimage_env` already contains mixed pip/conda packages or failed installations, create a fresh environment with another name, for example `afterimage_clean`, and activate that name in both terminals.

## 2. Install the pinned dependencies

```bat
python -m pip install -r requirements.txt
python -m pip check
```

The root requirements include Unsloth, the backend, the frontend and the exact CUDA pair `torch==2.14.1+cu130` and `torchvision==0.29.1+cu130`. The CUDA wheel index is declared in the file. A separate `pip install unsloth` or reinstall of the latest PyTorch is unnecessary and can change the tested combination.

NumPy, pandas and OpenCV are pinned together. Do not install the old requirements in `RetiZero/RetiZero/requirements.txt` into this environment; they belong to the upstream research project. Use separate environments for training notebooks.

## 3. Check the environment

```bat
python check_environment.py --skip-assets
```

The check verifies the interpreter and pinned versions, array libraries, CUDA, actual compiled torchvision operators, backend/frontend packages, and both models' imports. It does not load the model weights or run a prediction. A successful command exits with code 0.

For a quick CUDA check:

```bat
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Expected: `2.14.1+cu130 0.29.1+cu130 True`. CUDA being available alone does not prove torchvision or model imports work; run the full environment check too.

## 4. Authenticate with Hugging Face

Create a Hugging Face account and a token with read access to the model repositories. In the same environment:

```bat
hf auth login
```

Paste the token when prompted. Downloading models does not require write access. Keep the token out of source files and Git. Internet access is needed for the initial package and model downloads.

## 5. Download and check all model assets

```bat
python download_model_assets.py
python check_environment.py
```

The downloader fetches four assets:

| Asset | Source or destination |
| --- | --- |
| RetiZero fine-tuned checkpoint | `moht24/retizero-dr-model` to `RetiZero/best_retizero_dr.pth` |
| Qwen3 LoRA adapter | `moht24/qwen3-vl-dr-lora` to `Qwen3/lr0.0005_epoch4_batchsize8/best_lora/` |
| Qwen3 4-bit base model | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` in the Hugging Face cache |
| RetiZero text encoder | `emilyalsentzer/Bio_ClinicalBERT` in the Hugging Face cache |

RetiZero's current architecture constructs the text encoder during startup even though image grading uses its vision encoder. Downloading just the fine-tuned checkpoint is therefore insufficient for a fresh offline setup.

Existing valid assets are skipped. The cached base-model check requires the config, tokenizer, processor configuration and every shard listed in the weight index. These checks detect absent/empty files and malformed configuration; they do not checksum every tensor. A suspicious non-empty checkpoint may need `--force`.

Useful options:

- `--force`: bypass skip checks and force Hugging Face to download all requested remote assets again.
- `--clear-qwen3-target`: replace the local LoRA directory from the Hub cache; combine with `--force` to force a remote download too.
- `--skip-qwen3-base` or `--skip-retizero-text`: skip that cache download; the corresponding model can still need internet at first load.
- `--verbose`: show more download diagnostics.

Run `python download_model_assets.py --help` for repository overrides. `--force` does not replace a model directory explicitly supplied as a local path.

## 6. Model paths and cache location

Project asset paths are calculated from the code's location. No Windows username or Hugging Face snapshot hash needs to be edited. Cloning into another directory works with the same relative layout.

Hugging Face resolves model IDs in the current user's cache. To choose another cache drive, set this **before downloading**, and set it in the backend terminal too:

```bat
set HF_HOME=D:\afterimage-model-cache
```

In PowerShell, the equivalent is `$env:HF_HOME = 'D:\afterimage-model-cache'`. The frontend does not load the weights.

Advanced overrides `QWEN3_BASE_MODEL` and `RETIZERO_TEXT_MODEL` accept a compatible model ID or complete local model directory. Set them before running both the downloader and backend. Download flags alone do not reconfigure the backend. The default LoRA was trained against the named Qwen3 base; arbitrary replacement models are not supported by that adapter.

For offline operation after all four assets are cached, set this in the backend terminal before startup:

```bat
set HF_HUB_OFFLINE=1
```

This avoids online Hub metadata checks as well as downloads. A missing cached asset then fails locally; run the downloader online to repair it.

## 7. Run the backend

Open an Anaconda Prompt, navigate to your cloned `Afterimage` directory, and activate the environment:

```bat
conda activate afterimage_env
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9000
```

Wait for `Application startup complete`. Backend health: `http://127.0.0.1:9000/health`. Model state: `http://127.0.0.1:9000/model/status`. Backend health does not mean a model is loaded yet.

## 8. Run the frontend

Open another Anaconda Prompt, navigate to the same repository and activate the same environment:

```bat
conda activate afterimage_env
python -m streamlit run frontend/app.py
```

Open `http://localhost:8501`. Keep both terminals open. The backend manages model ports 8000 (RetiZero) and 8001 (Qwen3); do not start those servers separately while using the app. Run only one backend worker because it owns the model processes and GPU state.

## 9. Use and stop the application

1. Select RetiZero or Qwen3-VL and click **Confirm and Load Model**.
2. Wait for the model to become ready. Imports and the first load can take several minutes.
3. Upload a JPG or PNG fundus image and run analysis.
4. Review the returned grade and label. Switching models unloads the previous model to free GPU memory.
5. Press Ctrl+C in the frontend terminal, then in the backend terminal, to stop the app and its managed model process.

The backend crops the image to the Otsu ROI and square-pads it. RetiZero receives a 224 x 224 image without CLAHE; Qwen3 receives a 512 x 512 image with CLAHE. These pipelines must stay consistent with training.

## 10. Verify a real prediction

Stop the app's backend/model servers first so their ports and GPU memory are free, then run:

```bat
python scripts/smoke_test.py
```

This starts a temporary backend, loads RetiZero, grades `RetiZero/test_image.png`, switches to Qwen3, grades it again, checks each response is a grade 0-4, and unloads both. It records logs/results under ignored `.verification/`. The test checks the HTTP pipeline, not diagnostic accuracy. The same image can receive different grades from the two models.

Use `--port 19000` if only backend port 9000 is occupied. Ports 8000/8001 and GPU memory must still be free. `--backend-url http://127.0.0.1:9000` uses an existing backend and switches/unloads its models. The first Qwen3 prediction may compile kernels; inspect the backend log if it times out.

Fast regression tests, without downloading or loading model weights:

```bat
python -m unittest discover -s tests -v
```

## Troubleshooting

### Missing Unsloth or the wrong environment

Use `python -m pip --version` to check which interpreter owns pip. Activate the intended environment and run `python -m pip install -r requirements.txt`, followed by `python check_environment.py --skip-assets`. Using `python -m uvicorn` and `python -m streamlit` avoids accidentally running another environment's entry point.

### CPU PyTorch or torchvision nms errors

`operator torchvision::nms does not exist` means torchvision's compiled operators failed to load; a torch/torchvision version or CUDA-build mismatch is a common cause. Stop the backend and model servers, then repair the tested pair:

```bat
python -m pip install --force-reinstall --no-deps torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip check
python check_environment.py --skip-assets
```

If CUDA still reports `False`, check the NVIDIA GPU and driver. Repeating the same pip command will not repair a driver issue. Do not bypass Unsloth's torchvision check as a fix for a broken binary.

### Pandas datetime CAPI or partially initialized module errors

The conversation's pandas error was intermittent; its original cause was not established. `pip install pandas` reported "already satisfied" and did not repair anything. `conda install pandas` later replaced pandas 3.1.0 with 3.0.6 and installed a NumPy/MKL stack, so that was a material environment change, not proof of a particular root cause.

Run `python check_environment.py --skip-assets` to capture the first failing import separately. Prefer a fresh environment using the pinned pip requirements if binary errors persist. Avoid adding conda pandas/NumPy over pip-installed copies as a routine setup step. The Qwen3 server now imports pandas explicitly before Unsloth so an initial pandas failure is visible.

### Windows charmap decode errors during loading

The backend now launches model processes with `PYTHONIOENCODING=utf-8` and reads their logs with UTF-8 plus replacement for malformed bytes. This fixes the cp1252 failure on Unsloth banners and progress bars. Restart the backend after updating this code; an already-running process still uses the old version.

### Warnings and slow startup

- The installed xformers wheel can report that it was built for another torch/CUDA version. This pinned Unsloth version falls back to PyTorch attention; that message alone is not a load failure. The wheel is a declared Unsloth dependency, so removing it is not part of the tested setup.
- Hub 404 probes for optional files are normal if a supported file is found afterward. An eventual traceback or missing required weight file still needs investigation.
- The Windows symlink-cache warning affects disk usage. Developer Mode is optional; administrator privileges are not required to run the app.
- `torch.jit` and enum-registration deprecation warnings do not by themselves indicate a failed prediction.
- Repeated `GET /model/status 200` lines mean the UI is polling. Check the JSON `state`, `progress` and `last_error`; HTTP 200 alone does not mean the model loaded.
- If a process exits, the backend reports an error even if it was previously ready. Intentional unloads stay stopped, and failed processes can be retried.

## Repository and generated files

`backend/` and `frontend/` contain the live app; `Qwen3/` and `RetiZero/` contain model servers and preprocessing. `RetiZero/RetiZero/` is the bundled upstream research code. `finetuning/` and `*/prev/` are training material and older versions, not the running application.

Weights, `processed_image_cache/`, `unprocessed_image_cache/`, `unsloth_compiled_cache/` and `.verification/` are ignored by Git. Raw uploads are deleted after successful preprocessing; processed images are deleted after an inference response is received. Failed preprocessing or a failed connection can leave files in the cache directories for inspection.

The Word manual is generated from this README, with the retained application screenshots. See `scripts/build_manual.py` for the optional documentation build (requires `python-docx` in a separate documentation environment).

## Disclaimer

This project is intended for research and educational purposes only. Results must be reviewed by a qualified healthcare professional before any clinical decision is made.
