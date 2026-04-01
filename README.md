# Afterimage – Diabetic Retinopathy AI System

## Prerequisites

Before starting, ensure you have the following:

- Anaconda installed
- Python 3.11 available
- An NVIDIA CUDA-capable GPU
- Minimum 12 GB VRAM recommended for running the system properly
- Stable internet connection

## Create Environment

It is recommended to create a fresh Anaconda environment before setup.

```bash
conda create -n afterimage_env python=3.11 -y
conda activate afterimage_env
```

## Overview

Afterimage is an AI-powered diabetic retinopathy classification system built for research, experimentation, and prototyping in medical image analysis.

It uses:

- RetiZero contrastive vision-language model
- Qwen3-VL generative vision-language model with LoRA fine-tuning
- FastAPI backend
- Streamlit frontend

## Repository Structure

```text
Afterimage/
│
├── backend/                  # FastAPI backend
├── frontend/                 # Streamlit frontend
├── RetiZero/                 # RetiZero model weights (downloaded)
├── Qwen3/                    # Qwen3 LoRA adapter (downloaded)
├── download_model_assets.py  # Model download script
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MohammadTaha24/Afterimage.git
cd Afterimage
```

### 2. Install PyTorch

Install the latest PyTorch version that matches your system from the official PyTorch website:

https://pytorch.org/get-started/locally/

For NVIDIA GPUs, the manual currently uses:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Unsloth

```bash
pip install unsloth unsloth_zoo
```

### 5. Reinstall PyTorch

After installing the other packages, PyTorch may default back to CPU in some cases. Reinstall the latest GPU-enabled PyTorch version again.

Use the official PyTorch website if needed:

https://pytorch.org/get-started/locally/

For NVIDIA GPUs, the manual currently uses:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 6. Authenticate with Hugging Face

Create or log in to your Hugging Face account, generate an access token, and then run:

```bash
hf auth login
```

When prompted, paste your Hugging Face token.

### 7. Download Model Assets

```bash
python download_model_assets.py --clear-qwen3-target
```

### 8. Run the Backend

Open a **new Anaconda Prompt**, then run:

```bash
cd Afterimage
conda activate afterimage_env
uvicorn backend.main:app --host 127.0.0.1 --port 9000
```

### 9. Run the Frontend

Open **another Anaconda Prompt**, then run:

```bash
cd Afterimage
conda activate afterimage_env
streamlit run frontend\app.py
```

## Access the Application

- Backend: http://127.0.0.1:9000
- Frontend: opens automatically through Streamlit

## Models Used

### Base Model
`unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`

## Important Notes

- The base Qwen3-VL model is automatically downloaded on first run.
- A CUDA-capable NVIDIA GPU is required for proper performance.
- A minimum of 12 GB VRAM is recommended.
- The first run may take longer because required assets are downloaded.
- A stable internet connection is required during setup and first-time downloads.

## Disclaimer

This project is intended for research and educational purposes only.
