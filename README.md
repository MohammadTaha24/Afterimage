# Afterimage User Manual

## Prerequisites

Ensure Anaconda is installed and Python 3.11 is available.

- NVIDIA CUDA capable GPU with at least 16GB of VRAM
- Stable internet connection
- Hugging Face account and access token

## Create Environment

```bash
conda create -n afterimage_env python=3.11 -y
conda activate afterimage_env
```

## 1. Clone Repository

```bash
git clone https://github.com/MohammadTaha24/Afterimage.git
cd Afterimage
```

## 2. Install PyTorch

Install the latest PyTorch version based on your system:

https://pytorch.org/get-started/locally/

As of April 2026 for NVIDIA GPUs:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install Unsloth

```bash
pip install unsloth unsloth_zoo
```

## 5. Install PyTorch Again

This is not a mistake.

After installing the other packages, PyTorch may default to CPU as the device. Re-install the latest PyTorch version based on your system:

https://pytorch.org/get-started/locally/

As of April 2026 for NVIDIA GPUs:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## 6. Hugging Face Authentication

Create or log in to a Hugging Face account, then create an access token.

Run the following command:

```bash
hf auth login
```

When prompted, enter the token from the Hugging Face website.

Basic flow:

- Settings
- Access Tokens
- Create new token
- Copy token
- Paste token into the terminal after running `hf auth login`

## 7. Download Model Assets

```bash
python download_model_assets.py --clear-qwen3-target
```

This downloads the required model files, including:

- `RetiZero/best_retizero_dr.pth`
- `Qwen3/lr0.0005_epoch4_batchsize8/best_lora/`

## 8. Run Backend

Open a new Anaconda Prompt:

```bash
cd Afterimage
conda activate afterimage_env
uvicorn backend.main:app --host 127.0.0.1 --port 9000
```

## 9. Run Frontend

Open another Anaconda Prompt:

```bash
cd Afterimage
conda activate afterimage_env
streamlit run frontend\app.py
```

The Afterimage system requires two concurrent terminal sessions. Ensure the backend server is fully initialized before interacting with the Streamlit frontend.

## 10. Use the Application

1. Select the target model from the dropdown menu.
2. Click `Confirm and Load Model`.
3. Wait for the selected model to finish loading.
4. Upload a retinal image.
5. Run analysis.


Selecting a target model and clicking `Confirm and Load Model` unlocks the image upload interface.

## System Overview

Afterimage is a diabetic retinopathy screening system built with:

- FastAPI backend
- Streamlit frontend
- RetiZero
- Qwen3-VL with LoRA adapters

## Main Repository Structure

```text
Afterimage/
|
|-- backend/
|-- frontend/
|-- Qwen3/
|-- RetiZero/
|-- finetuning/
|-- processed_image_cache/
|-- unprocessed_image_cache/
|-- unsloth_compiled_cache/
|-- download_model_assets.py
|-- requirements.txt
|-- README.md
|-- newREADME.md
|-- structure.txt
```

Useful subfolders:

- `Qwen3/lr0.0005_epoch4_batchsize8/` contains the local Qwen3 run folder and `best_lora/`
- `RetiZero/RetiZero/` contains the bundled upstream RetiZero codebase
- `backend/prev`, `frontend/prev`, `Qwen3/prev`, and `RetiZero/prev` contain earlier versions of key files

## Models Used

### RetiZero

- Checkpoint path: `RetiZero/best_retizero_dr.pth`
- Inference server: `RetiZero/inference_server.py`
- Preprocessing: crop ROI, square pad, resize to `224x224`, no CLAHE

### Qwen3-VL

- Base model: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`
- Adapter path: `Qwen3/lr0.0005_epoch4_batchsize8/best_lora/`
- Preprocessing: crop ROI, square pad, resize to `512x512`, apply CLAHE

## Additional Project Material

The repository also includes training and research material:

- Qwen3 training and evaluation scripts under `finetuning/Qwen3/`
- RetiZero training notebooks under `finetuning/RetiZero/`
- Preprocessing notebooks under `finetuning/preprocessing/`
- The bundled `RetiZero/RetiZero/` research codebase

## Access

- Backend: `http://127.0.0.1:9000`
- Frontend: usually `http://localhost:8501`

## Notes

- The base model `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` is auto-downloaded on first run
- NVIDIA CUDA capable GPU is required for proper performance
- Stable internet connection is required
- The backend starts the selected model server on demand
- Raw uploads are written to `unprocessed_image_cache/`
- Preprocessed outputs are written to `processed_image_cache/`

## Disclaimer

This project is intended for research and educational purposes only.

Results must be reviewed by a qualified healthcare professional before any clinical decision is made.
