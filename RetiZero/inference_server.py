import argparse
import io
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retizero_server")

DR_LABELS: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR / "RetiZero"

if not REPO_DIR.exists():
    raise RuntimeError(f"Expected nested RetiZero repo at: {REPO_DIR}")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

try:
    from iden_modules import CLIPRModel
except ImportError as exc:
    raise RuntimeError(
        f"Could not import CLIPRModel from {REPO_DIR}. "
        f"Check that the nested repo folder contains iden_modules.py or the package."
    ) from exc


def emit_progress(percent: int, message: str) -> None:
    line = f"PROGRESS:{percent}:{message}"
    print(line, flush=True)
    log.info(line)


class RetiZeroInference(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.retizero = CLIPRModel(vision_type="lora", from_checkpoint=False, R=8)
        self.img_encoder = self.retizero.vision_model.model
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.img_encoder(x)
        logits = self.classifier(features)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RetiZero DR Inference Server")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


state: Dict[str, object] = {
    "ready": False,
    "model": None,
    "device": None,
    "transform": None,
    "args": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = state["args"]
    weights_path = Path(args.weights).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    emit_progress(5, "Resolving weights path")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["device"] = device

    emit_progress(20, f"Initialising model on {device.type.upper()}")
    model = RetiZeroInference(num_classes=5).to(device)

    emit_progress(55, "Loading fine-tuned weights")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)

    emit_progress(80, "Preparing transforms")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model.eval()
    state["model"] = model
    state["transform"] = transform
    state["ready"] = True

    emit_progress(100, "Model ready")
    yield

    log.info("Shutting down RetiZero server")
    state["ready"] = False
    state["model"] = None
    state["transform"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="RetiZero DR Inference Server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready yet")
    device = state["device"]
    return {
        "status": "ready",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected image file, got: {file.content_type}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    transform = state["transform"]
    device = state["device"]
    model = state["model"]

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)

        probs = F.softmax(logits, dim=1).squeeze(0).float().cpu()
        grade = int(torch.argmax(probs).item())

    return JSONResponse(
        content={
            "grade": grade,
            "label": DR_LABELS[grade],
            "confidence": float(probs[grade].item()),
        }
    )


@app.post("/shutdown")
def shutdown():
    log.info("Shutdown requested")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"detail": "Server is shutting down"}


if __name__ == "__main__":
    args = parse_args()
    state["args"] = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")