import io
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "Qwen3"))
sys.path.insert(0, str(PROJECT_ROOT / "RetiZero"))

from preprocess_qwen import preprocess_qwen_array
from preprocess_retizero import preprocess_retizero_array
from model_assets import QWEN3_LORA_TARGET, RETIZERO_TARGET, require_file, validate_lora

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("afterimage")

PROGRESS_RE = re.compile(r"PROGRESS:(\d{1,3}):(.*)")

GRADE_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

MODEL_NAMES = {
    "retizero": "RetiZero",
    "qwen3vl": "Qwen3-VL",
}

MODEL_PORTS = {
    "retizero": 8000,
    "qwen3vl": 8001,
}

MODEL_COMMANDS = {
    "retizero": [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "RetiZero" / "inference_server.py"),
        "--weights",
        str(PROJECT_ROOT / "RetiZero" / "best_retizero_dr.pth"),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ],
    "qwen3vl": [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "Qwen3" / "qwen3vl_server.py"),
        "--run_path",
        str(PROJECT_ROOT / "Qwen3" / "lr0.0005_epoch4_batchsize8"),
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
    ],
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

CACHE_RAW = PROJECT_ROOT / "unprocessed_image_cache"
CACHE_PROCESSED = PROJECT_ROOT / "processed_image_cache"

MODEL_READY_TIMEOUT_SECONDS = 900
PROCESS_TERMINATE_TIMEOUT_SECONDS = 20
HEALTHCHECK_INTERVAL_SECONDS = 0.5


class ModelRuntime:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.process: Optional[subprocess.Popen] = None
        self.state = "stopped"
        self.progress = 0
        self.message = "Not loaded"
        self.last_error: Optional[str] = None
        self.stdout_lines: "deque[str]" = deque(maxlen=80)
        self._reader_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{MODEL_PORTS[self.model_key]}"

    def set_status(self, state: str, progress: int, message: str) -> None:
        with self._lock:
            self.state = state
            self.progress = max(0, min(100, int(progress)))
            self.message = message

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "progress": self.progress,
                "message": self.message,
                "running": self.is_running(),
                "last_error": self.last_error,
            }


app = FastAPI(title="Afterimage DR Screening API", version="3.1.0")

runtimes: Dict[str, ModelRuntime] = {
    "retizero": ModelRuntime("retizero"),
    "qwen3vl": ModelRuntime("qwen3vl"),
}
active_model: Optional[str] = None
global_lock = threading.Lock()


@app.on_event("startup")
def startup() -> None:
    CACHE_RAW.mkdir(parents=True, exist_ok=True)
    CACHE_PROCESSED.mkdir(parents=True, exist_ok=True)
    log.info("Cache directories ready.")


@app.on_event("shutdown")
def shutdown() -> None:
    for runtime in runtimes.values():
        _stop_runtime(runtime)


def _err(status: int, msg: str) -> JSONResponse:
    log.warning("[%s] %s", status, msg)
    return JSONResponse(status_code=status, content={"error": msg})


def _validate_image(file_name: str, content_type: str, raw: bytes) -> None:
    suffix = Path(file_name or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Use JPG or PNG.")

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(f"Unexpected content type '{content_type}'.")

    if not raw:
        raise ValueError("Uploaded file is empty.")

    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception as exc:
        raise ValueError(f"Image could not be decoded: {exc}") from exc


def _read_process_output(runtime: ModelRuntime, process: subprocess.Popen, stop_event: threading.Event) -> None:
    assert process.stdout is not None

    try:
        for raw_line in iter(process.stdout.readline, ""):
            if stop_event.is_set() or runtime.process is not process:
                return
            line = raw_line.strip()
            if not line:
                continue

            runtime.stdout_lines.append(line)
            log.info("[%s] %s", runtime.model_key, line)

            match = PROGRESS_RE.search(line)
            if match and runtime.state == "loading":
                pct = int(match.group(1))
                msg = match.group(2).strip()
                runtime.set_status("loading", pct, msg)
    except Exception as exc:
        if stop_event.is_set() or runtime.process is not process:
            return
        runtime.last_error = f"Failed while reading process output: {exc}"
        runtime.set_status("error", runtime.progress, runtime.last_error)
        return

    if stop_event.is_set() or runtime.process is not process:
        return
    rc = process.wait()
    detail = next((line for line in reversed(runtime.stdout_lines)
                   if "Error:" in line or "Exception:" in line), "See backend log for details.")
    runtime.last_error = f"Process exited with code {rc}. {detail}"
    runtime.set_status("error", runtime.progress, runtime.last_error)


def _background_health_monitor(model: str, process: subprocess.Popen, stop_event: threading.Event) -> None:
    runtime = runtimes[model]
    health_url = f"{runtime.url}/health"
    deadline = time.monotonic() + MODEL_READY_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        if stop_event.is_set() or runtime.process is not process:
            return
        if process.poll() is not None or runtime.state == "error":
            return

        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200 and not stop_event.is_set() and runtime.process is process:
                runtime.set_status("ready", 100, "Model ready")
                runtime.last_error = None
                return
        except Exception:
            pass

        if stop_event.wait(HEALTHCHECK_INTERVAL_SECONDS):
            return

    if stop_event.is_set() or runtime.process is not process:
        return
    # Do not leave a timed-out process holding GPU memory or writing into a full pipe.
    _stop_runtime(runtime)
    runtime.last_error = "Timed out waiting for model health check"
    runtime.set_status("error", runtime.progress, runtime.last_error)


def _stop_runtime(runtime: ModelRuntime) -> None:
    runtime._stop_event.set()
    process = runtime.process
    try:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
    finally:
        runtime.process = None
        for thread in (runtime._reader_thread, runtime._health_thread):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=6)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        runtime.set_status("stopped", 0, "Not loaded")
        runtime.last_error = None


def _preprocess_bytes(model: str, raw: bytes) -> bytes:
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("OpenCV failed to decode the uploaded image.")

    if model == "retizero":
        processed = preprocess_retizero_array(img)
    elif model == "qwen3vl":
        processed = preprocess_qwen_array(img)
    else:
        raise ValueError(f"Unknown model: {model}")

    ok, encoded = cv2.imencode(".png", processed)
    if not ok:
        raise RuntimeError("Failed to encode preprocessed image.")
    return encoded.tobytes()


def _assert_model_assets_present(model: str) -> None:
    try:
        if model == "retizero":
            require_file(RETIZERO_TARGET)
        else:
            validate_lora(QWEN3_LORA_TARGET)
    except ValueError as exc:
        raise FileNotFoundError(
            f"Invalid assets for {MODEL_NAMES[model]}: {exc}. Run `python download_model_assets.py`."
        ) from exc


def _start_runtime(model: str) -> None:
    global active_model

    if model not in runtimes:
        raise ValueError(f"Unknown model '{model}'")

    with global_lock:
        target = runtimes[model]

        _assert_model_assets_present(model)

        if active_model and active_model != model:
            _stop_runtime(runtimes[active_model])

        if target.is_running() and target.state in {"loading", "ready"}:
            return
        _stop_runtime(target)

        target.set_status("loading", 1, "Starting model process")
        target.last_error = None
        target.stdout_lines.clear()
        target._stop_event = threading.Event()

        process = subprocess.Popen(
            MODEL_COMMANDS[model],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            # Model servers print UTF-8 (emoji banners, tqdm bar glyphs). Without this the
            # reader decodes with the Windows locale (cp1252) and dies on bytes like 0x8d.
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        target.process = process

        reader = threading.Thread(
            target=_read_process_output,
            args=(target, process, target._stop_event),
            daemon=True,
            name=f"{model}-stdout-reader",
        )
        reader.start()
        target._reader_thread = reader

        health_thread = threading.Thread(
            target=_background_health_monitor,
            args=(model, process, target._stop_event),
            daemon=True,
            name=f"{model}-health-monitor",
        )
        health_thread.start()
        target._health_thread = health_thread

        active_model = model


@app.get("/health")
def health():
    return {"status": "ok", "active_model": active_model}


@app.get("/model/status")
def model_status():
    return {
        "active_model": active_model,
        "models": {
            key: runtime.to_dict()
            for key, runtime in runtimes.items()
        },
    }


@app.post("/model/prepare")
def model_prepare(model: str = Form(...)):
    if model not in runtimes:
        return _err(400, f"Unknown model '{model}'. Choose 'retizero' or 'qwen3vl'.")

    runtime = runtimes[model]

    try:
        _start_runtime(model)
    except Exception as exc:
        runtime.last_error = str(exc)
        runtime.set_status("error", runtime.progress, str(exc))
        return _err(500, f"Failed to prepare {MODEL_NAMES[model]}: {exc}")

    return {
        "ok": True,
        "model": model,
        "display_name": MODEL_NAMES[model],
        "state": runtime.state,
        "progress": runtime.progress,
        "message": runtime.message,
    }

def _safe_delete(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        log.warning("Could not delete %s: %s", path, exc)

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    if model not in runtimes:
        return _err(400, f"Unknown model '{model}'. Choose 'retizero' or 'qwen3vl'.")

    runtime = runtimes[model]
    if runtime.state != "ready" or not runtime.is_running():
        return _err(409, f"{MODEL_NAMES[model]} is not ready. Load it first.")

    raw = await file.read()

    try:
        _validate_image(file.filename or "", file.content_type or "", raw)
    except ValueError as exc:
        return _err(400, str(exc))

    # --- Save raw ---
    uid = uuid.uuid4().hex
    raw_suffix = Path(file.filename or "upload.png").suffix.lower() or ".png"
    raw_path = CACHE_RAW / f"{uid}{raw_suffix}"
    raw_path.write_bytes(raw)

    # --- Preprocess (keep raw on failure) ---
    try:
        processed_bytes = _preprocess_bytes(model, raw)
    except Exception as exc:
        return _err(500, f"Preprocessing failed: {exc}")

    # --- Save processed, then delete raw ---
    processed_path = CACHE_PROCESSED / f"{uid}.png"
    processed_path.write_bytes(processed_bytes)
    _safe_delete(raw_path)

    # --- Run inference, then delete processed ---
    inference_reached = False
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{runtime.url}/predict",
                files={"file": ("processed.png", processed_bytes, "image/png")},
            )
        inference_reached = True
    except httpx.HTTPError as exc:
        # Keep processed file: we don't know if inference ran.
        return _err(503, f"Failed to contact {MODEL_NAMES[model]}: {exc}")
    finally:
        if inference_reached:
            _safe_delete(processed_path)

    if resp.status_code != 200:
        return _err(500, f"Inference server error ({resp.status_code}): {resp.text}")

    try:
        payload = resp.json()
        grade = int(payload["grade"])
    except Exception as exc:
        return _err(500, f"Invalid inference payload: {exc}")

    if grade not in GRADE_LABELS:
        return _err(500, f"Grade {grade} is out of range [0-4].")

    return {
        "label": GRADE_LABELS[grade],
        "grade": grade,
        "model": MODEL_NAMES[model],
    }

# @app.post("/predict")
# async def predict(file: UploadFile = File(...), model: str = Form(...)):
#     if model not in runtimes:
#         return _err(400, f"Unknown model '{model}'. Choose 'retizero' or 'qwen3vl'.")

#     runtime = runtimes[model]
#     if runtime.state != "ready":
#         return _err(409, f"{MODEL_NAMES[model]} is not ready. Load it first.")

#     raw = await file.read()

#     try:
#         _validate_image(file.filename or "", file.content_type or "", raw)
#     except ValueError as exc:
#         return _err(400, str(exc))

#     uid = uuid.uuid4().hex
#     raw_suffix = Path(file.filename or "upload.png").suffix.lower() or ".png"
#     raw_path = CACHE_RAW / f"{uid}{raw_suffix}"
#     raw_path.write_bytes(raw)

#     try:
#         processed_bytes = _preprocess_bytes(model, raw)
#     except Exception as exc:
#         return _err(500, f"Preprocessing failed: {exc}")

#     processed_path = CACHE_PROCESSED / f"{uid}.png"
#     processed_path.write_bytes(processed_bytes)

#     try:
#         async with httpx.AsyncClient(timeout=180.0) as client:
#             resp = await client.post(
#                 f"{runtime.url}/predict",
#                 files={"file": ("processed.png", processed_bytes, "image/png")},
#             )
#     except httpx.HTTPError as exc:
#         return _err(503, f"Failed to contact {MODEL_NAMES[model]}: {exc}")

#     if resp.status_code != 200:
#         return _err(500, f"Inference server error ({resp.status_code}): {resp.text}")

#     try:
#         payload = resp.json()
#         grade = int(payload["grade"])
#     except Exception as exc:
#         return _err(500, f"Invalid inference payload: {exc}")

#     if grade not in GRADE_LABELS:
#         return _err(500, f"Grade {grade} is out of range [0-4].")

#     return {
#         "label": GRADE_LABELS[grade],
#         "grade": grade,
#         "model": MODEL_NAMES[model],
#     }


@app.post("/model/unload")
def model_unload(model: str = Form(...)):
    global active_model

    if model not in runtimes:
        return _err(400, f"Unknown model '{model}'.")

    runtime = runtimes[model]
    with global_lock:
        _stop_runtime(runtime)
        if active_model == model:
            active_model = None

    return {
        "ok": True,
        "model": model,
        "display_name": MODEL_NAMES[model],
        "state": runtime.state,
        "progress": runtime.progress,
        "message": runtime.message,
    }
