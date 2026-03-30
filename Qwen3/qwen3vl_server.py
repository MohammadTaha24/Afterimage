"""
Qwen3-VL Diabetic Retinopathy Inference Server
------------------------------------------------
Loads the base model + LoRA adapters ONCE on startup, keeps them hot in GPU
memory, and serves predictions over HTTP.

Usage:
    python qwen3vl_server.py --run_path "path/to/best_lora"
    python qwen3vl_server.py --run_path "path/to/run_folder"  # auto-resolves best_lora
    python qwen3vl_server.py --run_path "..." --host 0.0.0.0 --port 8001

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
import re
import signal
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qwen3vl")

# ---------------------------------------------------------------------------
# Constants (mirrored exactly from infer_one_dr_qwen3vl.py)
# ---------------------------------------------------------------------------
INSTRUCTION = (
    "You are a medical imaging assistant.\n"
    "Grade diabetic retinopathy severity from this retinal fundus image.\n\n"
    "Scale:\n"
    "0 - No DR\n"
    "1 - Mild\n"
    "2 - Moderate\n"
    "3 - Severe\n"
    "4 - Proliferative DR\n\n"
    "Return exactly this format:\n"
    "<answer>X - LABEL</answer>"
)

LABEL_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

ANSWER_RE = re.compile(
    r"<answer>\s*([0-4])\s*-\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------------------------
# Helpers (mirrored from infer_one_dr_qwen3vl.py)
# ---------------------------------------------------------------------------
def emit_progress(percent: int, message: str) -> None:
    line = f"PROGRESS:{percent}:{message}"
    print(line, flush=True)
    log.info(line)

def _parse_img_size(img_size: str) -> Tuple[int, int]:
    s = str(img_size).lower().replace(" ", "")
    if "x" in s:
        w, h = s.split("x", 1)
        return int(w), int(h)
    v = int(s)
    return v, v


def parse_pred_label(gen_text: str) -> Optional[int]:
    m = ANSWER_RE.search(gen_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def resolve_lora_dir(run_or_lora_path: str) -> Path:
    p = Path(run_or_lora_path)
    if p.is_dir():
        if (p / "adapter_config.json").exists():
            return p
        best_lora = p / "best_lora"
        if (best_lora / "adapter_config.json").exists():
            return best_lora
    raise FileNotFoundError(
        f"Could not find a LoRA adapter folder in: {p}\n"
        f"Pass either the actual best_lora folder or the parent run folder."
    )


# ---------------------------------------------------------------------------
# Core inference (same logic as infer_one_dr_qwen3vl.py — adapted for
# in-memory PIL images instead of a file path)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_inference(
    model,
    tokenizer,
    img_path: str,          # temp file path written for the processor
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    img_size: str,
) -> str:
    w, h = _parse_img_size(img_size)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    tokenizer_kwargs = {
        "add_special_tokens": False,
        "return_tensors": "pt",
    }

    processor_attempts = (
        {"image_size": (h, w)},
        {"size": {"height": h, "width": w}},
        {"image_sizes": [(h, w)]},
        {},
    )

    inputs = None
    last_error = None
    for extra in processor_attempts:
        try:
            inputs = tokenizer(img_path, input_text, **tokenizer_kwargs, **extra)
            break
        except TypeError as exc:
            last_error = exc
            continue

    if inputs is None:
        raise RuntimeError(f"Failed to prepare model inputs. Last error: {last_error}")

    inputs = inputs.to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        min_p=min_p,
        use_cache=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = output[0][prompt_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL DR Inference Server")
    parser.add_argument(
        "--run_path",
        type=str,
        required=True,
        help="Path to the best_lora folder or its parent run folder",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        help="HuggingFace model ID or local path for the base model",
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load base model in 4-bit quantisation (default: True)",
    )
    parser.add_argument(
        "--img_size",
        type=str,
        default="512",
        help="Image size passed to the processor, e.g. '512' or '512x512'",
    )
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min_p", type=float, default=0.1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Global server state
# ---------------------------------------------------------------------------
state: dict = {}

# ---------------------------------------------------------------------------
# FastAPI lifespan — model loads here, unloads on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    args = state["args"]

    # emit_progress(5, "Resolving LoRA adapter path")
    # lora_dir = resolve_lora_dir(args.run_path)
    # log.info(f"Resolved LoRA adapter directory: {lora_dir}")

    # emit_progress(20, "Importing Qwen dependencies")
    # from unsloth import FastVisionModel
    # from peft import PeftModel

    # emit_progress(40, f"Loading base model: {args.base_model}")
    # model, tokenizer = FastVisionModel.from_pretrained(
    #     model_name=args.base_model,
    #     load_in_4bit=args.load_in_4bit,
    # )

    # emit_progress(75, f"Attaching LoRA adapters from: {lora_dir}")
    # model = PeftModel.from_pretrained(model, str(lora_dir))

    # emit_progress(90, "Preparing model for inference")
    # FastVisionModel.for_inference(model)
    # model.eval()

    # state["model"] = model
    # state["tokenizer"] = tokenizer
    # state["ready"] = True

    # emit_progress(100, "Model ready")

    emit_progress(5, "Resolving LoRA adapter path")
    lora_dir = resolve_lora_dir(args.run_path)

    emit_progress(20, "Importing Qwen dependencies")
    from unsloth import FastVisionModel
    from peft import PeftModel

    emit_progress(40, "Loading base model")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )

    emit_progress(75, "Attaching LoRA adapters")
    model = PeftModel.from_pretrained(model, str(lora_dir))

    emit_progress(90, "Preparing model for inference")
    FastVisionModel.for_inference(model)
    model.eval()

    state["model"] = model
    state["tokenizer"] = tokenizer
    state["ready"] = True

    emit_progress(100, "Model ready")

    yield

    log.info("Shutting down - unloading model from memory...")
    state["ready"] = False
    del state["model"]
    del state["tokenizer"]
    torch.cuda.empty_cache()
    log.info("GPU cache cleared.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Qwen3-VL DR Inference Server",
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
    return {
        "status": "ready",
        "device": "CUDA",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a retinal image and return a DR grade.

    Response:
        { "grade": 0 }   — integer in range [0, 4]

    Grades:
        0  No DR
        1  Mild
        2  Moderate
        3  Severe
        4  Proliferative DR
    """
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got: {file.content_type}",
        )

    raw_bytes = await file.read()

    # The Qwen3-VL processor needs a file path (not raw bytes), so we write
    # the upload to a small temp file, run inference, then delete it.
    import tempfile
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        args = state["args"]
        gen_text = _run_inference(
            model=state["model"],
            tokenizer=state["tokenizer"],
            img_path=tmp_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            img_size=args.img_size,
        )
    finally:
        os.unlink(tmp_path)  # always clean up, even on error

    grade = parse_pred_label(gen_text)
    if grade is None:
        log.warning(f"Could not parse grade from model output: {repr(gen_text)}")
        raise HTTPException(
            status_code=422,
            detail=f"Model produced unparseable output: {repr(gen_text)}",
        )

    log.info(f"Prediction: grade={grade} ({LABEL_NAMES[grade]})  raw={repr(gen_text)}  file={file.filename}")
    return JSONResponse(content={"grade": grade})


@app.post("/shutdown")
def shutdown():
    """Gracefully shut down the server and unload the model."""
    log.info("Shutdown requested via /shutdown endpoint.")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"detail": "Server is shutting down"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    state["args"] = args

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
