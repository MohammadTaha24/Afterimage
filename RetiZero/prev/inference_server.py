"""
RetiZero Diabetic Retinopathy Inference Server
-----------------------------------------------
Loads the model ONCE on startup, keeps it hot in GPU memory,
and serves predictions over HTTP.

Usage:
    python inference_server.py --weights best_retizero_dr.pth
    python inference_server.py --weights best_retizero_dr.pth --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict      - Send a retinal image, get back grade 0-4
    GET  /health       - Check if the server and model are ready
    POST /shutdown     - Gracefully shut down the server and unload the model

Shutdown alternatives (all work cleanly):
    - POST /shutdown endpoint
    - Ctrl+C in terminal
    - SIGTERM from OS (kill <pid>)
"""

import argparse
import io
import os
import signal
import sys
import logging
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retizero")

# ---------------------------------------------------------------------------
# Model definition (mirrors inference_retizero.py exactly)
# ---------------------------------------------------------------------------
RETI_ZERO_PATH = "./RetiZero"
if RETI_ZERO_PATH not in sys.path:
    sys.path.append(RETI_ZERO_PATH)

try:
    from iden_modules import CLIPRModel
except ImportError:
    log.error(
        "Could not import CLIPRModel. "
        f"Make sure the RetiZero repo is at {RETI_ZERO_PATH}"
    )
    sys.exit(1)


class RetiZeroInference(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.retizero = CLIPRModel(vision_type="lora", from_checkpoint=False, R=8)
        self.img_encoder = self.retizero.vision_model.model
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.img_encoder(x)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Preprocessing pipeline (must match training — no CLAHE)
# ---------------------------------------------------------------------------
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Global server state
# ---------------------------------------------------------------------------
# Stored in a plain dict so the lifespan context and endpoint functions
# share the same references without module-level mutation.
state: dict = {}

# ---------------------------------------------------------------------------
# Argument parsing (done at import time so uvicorn can re-use the values)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RetiZero DR Inference Server")
    parser.add_argument(
        "--weights",
        type=str,
        default="best_retizero_dr.pth",
        help="Path to the fine-tuned state dict (.pth)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# FastAPI lifespan — model loads here, unloads on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    args = state["args"]

    if not os.path.exists(args.weights):
        log.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device.type.upper()}")

    log.info("Loading model architecture...")
    model = RetiZeroInference(num_classes=5).to(device)

    log.info(f"Loading weights from: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    state["model"] = model
    state["device"] = device
    state["ready"] = True
    log.info("Model ready. Server is accepting requests.")

    yield  # <-- server runs here

    # ---- SHUTDOWN ----
    log.info("Shutting down — unloading model from memory...")
    state["ready"] = False
    del state["model"]

    if device.type == "cuda":
        torch.cuda.empty_cache()
        log.info("GPU cache cleared.")

    log.info("Model unloaded. Goodbye.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RetiZero DR Inference Server",
    description="5-class diabetic retinopathy grading (grades 0–4)",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Returns 200 when the model is loaded and ready."""
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready yet")
    device = state["device"]
    return {
        "status": "ready",
        "device": device.type.upper(),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a retinal image and return a DR grade.

    Response:
        { "grade": 0 }   — integer in range [0, 4]

    Grades:
        0  Normal (No DR)
        1  Mild NPDR
        2  Moderate NPDR
        3  Severe NPDR
        4  Proliferative DR
    """
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready")

    # --- Validate and read image ---
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got: {file.content_type}",
        )

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # --- Preprocess ---
    device = state["device"]
    tensor = PREPROCESS(image).unsqueeze(0).to(device)

    # --- Inference ---
    model = state["model"]
    with torch.no_grad():
        with torch.amp.autocast(device.type):
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().float().cpu().numpy()

    grade = int(probs.argmax())
    log.info(f"Prediction: grade={grade}  confidence={probs[grade]*100:.1f}%  file={file.filename}")

    return JSONResponse(content={"grade": grade})


@app.post("/shutdown")
def shutdown():
    """
    Gracefully shut down the server.
    The lifespan cleanup will unload the model and free GPU memory.
    """
    log.info("Shutdown requested via /shutdown endpoint.")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"detail": "Server is shutting down"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    state["args"] = args  # share with lifespan before app starts

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
