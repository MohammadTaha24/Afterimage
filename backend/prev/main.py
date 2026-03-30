"""
Afterimage — DR Screening Backend
Validates uploads and proxies requests to the correct inference server.

RetiZero server  → http://127.0.0.1:8000   (run: python RetiZero/inference_server.py)
Qwen3-VL server  → http://127.0.0.1:8001   (run: python Qwen3/qwen3vl_server.py)

Usage:
    uvicorn backend.main:app --host 127.0.0.1 --port 9000 --reload
"""

import io
import logging
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("afterimage")

# ── Constants ─────────────────────────────────────────────────────────────────
INFERENCE_URLS = {
    "retizero": "http://127.0.0.1:8000/predict",
    "qwen3vl":  "http://127.0.0.1:8001/predict",
}

MODEL_NAMES = {
    "retizero": "RetiZero",
    "qwen3vl":  "Qwen3-VL",
}

GRADE_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

ALLOWED_EXTENSIONS   = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
CACHE_RAW       = PROJECT_ROOT / "unprocessed_image_cache"
CACHE_PROCESSED = PROJECT_ROOT / "processed_image_cache"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Afterimage DR Screening API", version="2.0.0")


@app.on_event("startup")
def _startup():
    CACHE_RAW.mkdir(parents=True, exist_ok=True)
    CACHE_PROCESSED.mkdir(parents=True, exist_ok=True)
    log.info("Cache directories ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _err(status: int, msg: str) -> JSONResponse:
    log.warning(f"[{status}] {msg}")
    return JSONResponse(status_code=status, content={"error": msg})


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file:  UploadFile = File(...),
    model: str        = Form("retizero"),
):
    # ── Validate model choice ────────────────────────────────────────────────
    if model not in INFERENCE_URLS:
        return _err(400, f"Unknown model '{model}'. Choose 'retizero' or 'qwen3vl'.")

    # ── Validate file type ───────────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return _err(400, f"Unsupported file type '{suffix}'. Use JPG or PNG.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return _err(400, f"Unexpected content type '{file.content_type}'.")

    raw = await file.read()
    if not raw:
        return _err(400, "Uploaded file is empty.")

    # ── Validate image is decodable ──────────────────────────────────────────
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception as exc:
        return _err(400, f"Image could not be decoded: {exc}")

    # ── Save to raw cache ────────────────────────────────────────────────────
    uid      = uuid.uuid4().hex
    raw_path = CACHE_RAW / f"{uid}{suffix}"
    raw_path.write_bytes(raw)
    log.info(f"Saved raw image: {raw_path.name}")

    # ── Forward to inference server ──────────────────────────────────────────
    url = INFERENCE_URLS[model]
    log.info(f"Forwarding to {MODEL_NAMES[model]} at {url}")

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                url,
                files={"file": (file.filename, raw, file.content_type)},
            )
    except httpx.ConnectError:
        return _err(
            503,
            f"{MODEL_NAMES[model]} inference server is not running. "
            f"Start it first and try again.",
        )
    except httpx.TimeoutException:
        return _err(504, "Inference timed out. The model is taking too long.")
    except Exception as exc:
        return _err(500, f"Unexpected error calling inference server: {exc}")

    # ── Parse response ───────────────────────────────────────────────────────
    if resp.status_code != 200:
        return _err(500, f"Inference server error ({resp.status_code}): {resp.text}")

    try:
        grade = int(resp.json()["grade"])
    except (KeyError, ValueError, TypeError):
        return _err(500, f"Inference server returned unexpected payload: {resp.text}")

    if grade not in GRADE_LABELS:
        return _err(500, f"Grade {grade} is out of range [0–4].")

    label = GRADE_LABELS[grade]
    log.info(f"Result → model={model}  grade={grade}  label={label}  file={file.filename}")

    return {"label": label, "grade": grade, "model": MODEL_NAMES[model]}
