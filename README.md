# Afterimage – Diabetic Retinopathy AI System

## Overview

Afterimage is an AI-powered system for diabetic retinopathy classification using:

- RetiZero Contrastive Vision-Language model
- Qwen3-VL GenAI Vision-Language Model with LoRA fine-tuning
- FastAPI backend
- Streamlit frontend

This project is designed for research, experimentation, and prototyping in medical image analysis.

---

## Repository Structure

```
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

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/MohammadTaha24/Afterimage.git
cd Afterimage
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Authenticate with Hugging Face

```bash
hf auth login
```

---

### 4. Download model assets

```bash
python download_model_assets.py --clear-qwen3-target
```

---

### 5. Run the backend

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 9000
```

---

### 6. Run the frontend

```bash
streamlit run frontend\app.py
```

---

### 7. Access the application

- Backend: http://127.0.0.1:9000  
- Frontend: opens automatically via Streamlit

---

## Models Used

### Base Model
unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit

---

## Important Notes

- Do NOT commit tokens or model weights
- First run may take time due to downloads
- GPU recommended

---

## Disclaimer

This project is for research and educational purposes only.
