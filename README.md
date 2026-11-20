# Afterimage User Manual

Afterimage grades retinal fundus images with RetiZero or Qwen3-VL with a LoRA adapter. A Streamlit interface sends images to a FastAPI backend, which preprocesses them and starts one model server at a time.

This branch (`afterimage-re`) also contains the experiments, evaluation and evidence for the paper revision. Sections 1 to 10 cover running the app. The [paper revision](#paper-revision-work) section covers the research work.

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

## Paper revision work

This work responds to the reviewers' comments on the Afterimage paper. It retrains and re-evaluates both models under a controlled protocol. None of it is imported by the running app.

### What changed

| Date (2026) | Work |
| --- | --- |
| 10 Nov | App hardening: pinned CUDA and dependency stack, asset downloader and validation, environment check, real-model smoke test, rebuilt manual. Qwen3 APTOS reproduction training and evaluation, the four-run Qwen experiment queue, and the phase 1 dataset audit. |
| 15 Nov | Qwen preprocessing with a CLAHE on/off switch. Retraining 2 replay, original-split CLAHE pairs, BF16 base control, remaining-run routing. Helpers for a matched RetiZero setup and verified Messidor. Paper tables builder and the first generated tables. |
| 16 Nov | `finetuning/Qwen3` reorganized into `legacy/`, `notebooks/` and `paper_revision/`. Final Qwen runs queue (zero-shot, timing, archived Messidor, learning-rate and alpha studies), plus labelled FP32 fallback scoring for FP16 overflow. RetiZero reproduction folder. Paper plan, deliverables checklist and remaining-work phases. |
| 17 Nov | RetiZero retrained from the official pretrained checkpoint (non-finite FP16 steps are skipped instead of stopping the run). Final-queue results added to the tables. Efficiency tables built from saved run records. Independent check of the headline split, Messidor source search, the evidence-folder and results-workbook builders, and the results email draft. |
| 17 to 20 Nov | The evidence builder computes the split overlap directly from image SHA-256 hashes, and the workbook checks every number against raw run files instead of notes. Literature review on APTOS duplicates. |

### Key finding on the data split

19 of the 367 images in the original APTOS test split have a byte-identical copy in train (17) or validation (2), and 3 of those copies carry a different grade. A clean split removes the overlap, leaving 2,786 train, 357 validation and 361 test images. New results are reported on the clean 361-image test set, with the original 367 kept as a reference. See [HEADLINE_VERIFICATION.md](docs/paper/HEADLINE_VERIFICATION.md) and [APTOS_DUPLICATES_LITERATURE.md](docs/paper/APTOS_DUPLICATES_LITERATURE.md).

### Where to start

| Path | What it is |
| --- | --- |
| [docs/paper/REMAINING_WORK.md](docs/paper/REMAINING_WORK.md) | Current status, results so far and the remaining phases. Read this first. |
| [docs/paper/](docs/paper/) | Plan, final-runs design, deliverables checklist, headline verification, Messidor source notes, RetiZero training brief and the results update draft. |
| [REVIEWER_CHECKLIST.md](REVIEWER_CHECKLIST.md), [PAPER_TRAINING_CHECKLIST.md](PAPER_TRAINING_CHECKLIST.md) | Each reviewer point, and the training and evaluation checklist. |
| [finetuning/Qwen3/](finetuning/Qwen3/README.md) | Qwen3-VL-8B + LoRA training and scoring. `notebooks/03_final_runs.ipynb` is the final queue; `paper_revision/` holds the helpers. Several files are hash-pinned by saved runs; the folder README lists which ones must not be edited. |
| [finetuning/RetiZero/](finetuning/RetiZero/README.md) | RetiZero reproduction: helper, workers, queue notebook, plan and [results](finetuning/RetiZero/RESULTS_RETIZERO.md). |
| [finetuning/paper_tables/](finetuning/paper_tables/README.md) | CPU-only builders for the paper tables, efficiency tables, evidence folder and results workbook. Generated tables are in `output/`. |
| [experiments/](experiments/README.md) | The earlier four-condition Qwen queue and its post-run review. |
| [audit/](audit/README.md) | Phase 1 dataset audit. |

### Reproducing the experiments: prerequisites

The app (sections 1 to 10) runs from this repository alone. The paper experiments do not. They need the data, models and helper scripts below, most of which are not in Git. Check this list before starting.

**1. Hardware**

| Item | What was used | Notes |
| --- | --- | --- |
| GPU | NVIDIA RTX 5070 Ti, 16 GB VRAM | 16 GB is the minimum for Qwen training. One Qwen run takes about 2 to 3 hours. |
| System RAM | About 64 GB | Needed by the BF16 base control, which offloads 16 layers to the CPU. |
| Free disk | About 60 GB, plus room for runs | Qwen models about 42 GB, RetiZero checkpoint 1.6 GB, datasets about 5 GB. Each Qwen run folder holds checkpoints and adapters. |

**2. Python environment**

Use the same `afterimage_env` as the app (sections 1 to 3). The research code also needs these packages, which `requirements.txt` does not install:

```bat
python -m pip install openpyxl psutil jupyterlab ipykernel
python -m ipykernel install --user --name afterimage_env --display-name "Python (afterimage_env)"
```

`openpyxl` is used by the results workbook builder, `psutil` by the Qwen run worker, and Jupyter by every notebook. The notebooks expect the kernel named `Python (afterimage_env)`.

**3. Models**

| Model | Source | Expected location | Used by |
| --- | --- | --- | --- |
| Qwen3-VL-8B 4-bit base | Hugging Face `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`, revision `b5b904c3fcdc7541adf2a2bb219b0ed95288c794` | `D:\Afterimage_Experiments\assets\qwen3-vl-8b-4bit\` | All Qwen training and scoring |
| Qwen3-VL-8B BF16 base | Hugging Face `Qwen/Qwen3-VL-8B-Instruct`, revision `1dd1e02d981403da25ed73d43430e4ef598cb94b` | `D:\Afterimage_Experiments\assets\qwen3-vl-8b-bf16\` | BF16 base control and BF16 zero-shot only |
| RetiZero official pretrained checkpoint | Google Drive link in [RetiZero/RetiZero/README.md](RetiZero/RetiZero/README.md) (file id `14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM`), SHA-256 `7d06d294...1601ede` | `D:\Afterimage_Experiments\assets\RetiZero.pth` | RetiZero retraining |
| `emilyalsentzer/Bio_ClinicalBERT` | Downloaded by `python download_model_assets.py` | Hugging Face cache | RetiZero code imports |
| Fine-tuned app weights (RetiZero checkpoint, Qwen LoRA) | Downloaded by `python download_model_assets.py` | See section 5 | Original adapter audit and the reference results |

Download a Hugging Face model to a folder with, for example, `hf download Qwen/Qwen3-VL-8B-Instruct --revision 1dd1e02d981403da25ed73d43430e4ef598cb94b --local-dir D:\Afterimage_Experiments\assets\qwen3-vl-8b-bf16`. Saved run configurations record the revision and file hashes, so a different revision will fail the checks.

**4. Data**

| Data | Source | Expected location |
| --- | --- | --- |
| APTOS 2019 training images and labels (3,662 images) | Kaggle competition `aptos2019-blindness-detection`. Join the competition to download it. | `D:\aptos2019-blindness-detection\` with `train.csv` and `train_images\` |
| The recovered original 80/10/10 split | Not public. Copy `D:\Afterimage_Experiments\reference\aptos_split\` (train, val and test CSVs plus `dataset_manifest.json`) from the machine that ran the experiments. **Without it the original 367-image split and the clean 361-image split cannot be rebuilt.** | `D:\Afterimage_Experiments\reference\aptos_split\` |
| Archived preprocessed APTOS and Messidor exports (512 px) | The original project's preprocessed archive, `DR_Aptos_Messidor_Only_Full_Dataset` and `DR_Messidor_Only_512_Full_Dataset`. Not public. | `D:\recovered\` |
| Messidor-2 (raw images and adjudicated grades) | Not obtained yet. Raw images come from ADCIS (registration required); adjudicated grades from Kaggle `google-brain/messidor2-dr-grades`. See [MESSIDOR_SOURCES_FOUND.md](docs/paper/MESSIDOR_SOURCES_FOUND.md). | Not used yet |

**Getting the private data.** The recovered split and the archived exports are not public and must not be committed or redistributed: they derive from APTOS 2019 competition data and Messidor. Ask Mohammad Taha for access to the private share. It holds three zips, each with a SHA-256 in `SHA256SUMS.txt`:

| Zip | Contents | Size | Extract into |
| --- | --- | --- | --- |
| `afterimage_aptos_split.zip` | `aptos_split\` (train, val and test CSVs, `dataset_manifest.json`) | 1.4 MB | `D:\Afterimage_Experiments\reference\` |
| `afterimage_recovered_aptos_messidor_512.zip` | `DR_Aptos_Messidor_Only_Full_Dataset\` (9,440 files) | 3.0 GB | `D:\recovered\` |
| `afterimage_recovered_messidor_512.zip` | `DR_Messidor_Only_512_Full_Dataset\` (2,116 files) | 0.7 GB | `D:\recovered\` |

Check each download with `Get-FileHash <zip> -Algorithm SHA256` in PowerShell before extracting. Download APTOS 2019 itself from Kaggle; it is not in the share.

The archived Messidor set (1,057 images) used so far is derived from that preprocessed archive, not from raw Messidor-2, and its grades do not match the published adjudicated grades. Treat Messidor results as provisional.

**5. Code that runs from the experiments folder**

These files were run from `D:\Afterimage_Experiments\`, outside the repository. Byte-identical copies are in [external/Afterimage_Experiments/](external/Afterimage_Experiments/README.md). Copy them to `D:\Afterimage_Experiments\`, keeping the `code\` subfolder, because the notebooks and scripts refer to them by that path. Saved runs record their SHA-256, so do not edit them.

| File | What it does |
| --- | --- |
| `01_preprocessing.ipynb` | Builds the 512-pixel APTOS datasets for both splits (original and clean), CLAHE on and off |
| `02_training.ipynb` | Ran the main Qwen grid on the clean split (ranks 8/16/32, CLAHE on/off, three seeds) |
| `code\qwen_pilot.py` | Qwen training and scoring worker used by both training notebooks and `finetuning/Qwen3/paper_revision/` |
| `code\prepare_aptos.py` | Dataset builder behind `01_preprocessing.ipynb` and `finetuning/RetiZero/data/build_retizero_datasets.py` |
| `code\notebook_runner.py` | Launches workers from notebooks and keeps their console output |
| `code\recover_qwen.py` | Resumes or rescores a stopped Qwen run |

**6. Folder layout and paths**

The research code uses absolute Windows paths, mostly under `D:\Afterimage_Experiments\` and `D:\aptos2019-blindness-detection\`. The simplest setup is to recreate the same folders on a `D:` drive:

```text
D:\aptos2019-blindness-detection\     Kaggle APTOS download
D:\recovered\                         archived preprocessed exports
D:\Afterimage_Experiments\
    01_preprocessing.ipynb, 02_training.ipynb   copied from external/Afterimage_Experiments
    assets\                           the three model folders and RetiZero.pth
    code\                             the four helper scripts, copied from external/Afterimage_Experiments/code
    reference\aptos_split\            the recovered split
    datasets\                         built by the preprocessing notebook
    runs\  final_runs\  retizero_runs\    created by the runs
```

Otherwise, change the path constants at the top of each notebook and script. Some notes and saved configurations also record paths from the original machine (`C:\Users\...`). Those are a record of where files were at the time, not settings to edit.

**7. Order of work**

1. Build the datasets with `D:\Afterimage_Experiments\01_preprocessing.ipynb` (Qwen, 512 px, both splits), then `finetuning/RetiZero/data/build_retizero_datasets.py` (RetiZero, 224 px). `finetuning/Qwen3/notebooks/01_preprocessing.ipynb` only builds the earlier original-split CLAHE pairs.
2. Run the main Qwen grid with `D:\Afterimage_Experiments\02_training.ipynb`, then the remaining Qwen queue in `finetuning/Qwen3/notebooks/03_final_runs.ipynb`. Run each pilot first.
3. Run the RetiZero queue in `finetuning/RetiZero/notebooks/02_retizero_runs.ipynb`. Run its pilot first.
4. Rebuild the tables on the CPU with the scripts in `finetuning/paper_tables/` (see its README).

Run one GPU job at a time. The Qwen and RetiZero runners share a GPU lock in `D:\Afterimage_Experiments\runs`.

### What is in Git and what is not

Trained adapters, checkpoints, datasets, run logs and per-run predictions are gitignored. They stay in the experiments folder on the machine that ran them. The tables and summaries built from them are committed under `finetuning/paper_tables/output/`, so the reported results can be read without the run folders.

## Repository and generated files

`backend/` and `frontend/` contain the live app; `Qwen3/` and `RetiZero/` contain model servers and preprocessing. `RetiZero/RetiZero/` is the bundled upstream research code. `finetuning/`, `experiments/`, `audit/` and `docs/paper/` hold the paper revision work described above. `*/prev/` directories hold older versions. None of these are part of the running application.

Weights, `processed_image_cache/`, `unprocessed_image_cache/`, `unsloth_compiled_cache/` and `.verification/` are ignored by Git. Raw uploads are deleted after successful preprocessing; processed images are deleted after an inference response is received. Failed preprocessing or a failed connection can leave files in the cache directories for inspection.

The Word manual is generated from this README, with the retained application screenshots. See `scripts/build_manual.py` for the optional documentation build (requires `python-docx` in a separate documentation environment).

## Disclaimer

This project is intended for research and educational purposes only. Results must be reviewed by a qualified healthcare professional before any clinical decision is made.
