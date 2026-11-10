"""Check runtime imports and local model assets without loading model weights."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from model_assets import (
    DEFAULT_QWEN3_BASE_MODEL, DEFAULT_RETIZERO_TEXT_MODEL, PROJECT_ROOT,
    QWEN3_LORA_TARGET, RETIZERO_TARGET, require_file, validate_lora,
)


CHECKS = {
    "NumPy, pandas and OpenCV": """
import numpy as np, pandas as pd, cv2
frame = pd.DataFrame({'grade': np.arange(5)})
assert frame.grade.sum() == 10
assert cv2.resize(np.zeros((12, 20, 3), dtype=np.uint8), (8, 8)).shape == (8, 8, 3)
print('numpy', np.__version__, 'pandas', pd.__version__, 'opencv', cv2.__version__)
""",
    "CUDA and compiled torchvision operators": """
import torch, torchvision
from torchvision.ops import nms
assert torch.cuda.is_available(), 'CUDA is unavailable; check the torch wheel and NVIDIA driver'
boxes = torch.tensor([[0., 0., 4., 4.], [1., 1., 3., 3.]], device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
assert nms(boxes, scores, 0.5).numel() > 0
print('torch', torch.__version__, 'torchvision', torchvision.__version__)
print('GPU', torch.cuda.get_device_name(0))
""",
    "Backend and frontend dependencies": """
import fastapi, uvicorn, multipart, httpx, streamlit, requests
from PIL import Image
print('FastAPI', fastapi.__version__, 'Streamlit', streamlit.__version__)
""",
    "Qwen3 imports": """
import pandas
from unsloth import FastVisionModel
from peft import PeftModel
import datasets, bitsandbytes
print('Unsloth, PEFT, datasets and bitsandbytes imported')
""",
    "RetiZero imports": """
import sys
sys.path.insert(0, 'RetiZero/RetiZero')
from iden_modules import CLIPRModel
print('Vendored RetiZero model imported')
""",
}


def check_versions() -> bool:
    from packaging.requirements import Requirement

    ok = True
    for filename in ("requirements.txt", "backend/requirements.txt", "frontend/requirements.txt", "Qwen3/requirements.txt"):
        for line in (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            req = Requirement(line)
            if req.marker and not req.marker.evaluate():
                continue
            try:
                version = importlib.metadata.version(req.name)
            except importlib.metadata.PackageNotFoundError:
                print(f"FAIL: {req.name} is missing", flush=True)
                ok = False
                continue
            if not req.specifier.contains(version):
                print(f"FAIL: {req.name} {version} installed; {req.specifier} required", flush=True)
                ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-assets", action="store_true", help="Check packages before downloading the models")
    args = parser.parse_args()
    print(f"Python: {sys.executable} ({sys.version.split()[0]})", flush=True)
    if sys.version_info[:2] != (3, 11):
        print("FAIL: use the documented Python 3.11 environment", flush=True)
        return 1
    ok = check_versions()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for label, code in CHECKS.items():
        print(f"Checking {label}...", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-u", "-c", code], cwd=PROJECT_ROOT, env=env,
                capture_output=True, encoding="utf-8", errors="replace", timeout=180,
            )
            if result.returncode:
                print(f"FAIL: {label}\n{result.stdout}\n{result.stderr}", flush=True)
                ok = False
            else:
                # ascii() also works when a Windows terminal cannot display Unsloth's banner.
                print(f"PASS: {label}", flush=True)
                print(result.stdout.encode("ascii", errors="backslashreplace").decode("ascii").strip(), flush=True)
        except subprocess.TimeoutExpired:
            print(f"FAIL: {label} import exceeded 180 seconds", flush=True)
            print("Check that the Hugging Face and Triton cache directories are writable by this user.", flush=True)
            ok = False
    if not args.skip_assets:
        try:
            from download_model_assets import find_cached_model

            require_file(RETIZERO_TARGET)
            validate_lora(QWEN3_LORA_TARGET)
            for repo, kind in ((DEFAULT_QWEN3_BASE_MODEL, "qwen3"), (DEFAULT_RETIZERO_TEXT_MODEL, "retizero-text")):
                if find_cached_model(repo, kind) is None:
                    raise ValueError(f"Incomplete or missing cache for {repo}")
            print("PASS: all four model assets are available locally", flush=True)
        except (ImportError, ValueError, OSError) as exc:
            print(f"FAIL: assets: {exc}\nRun python download_model_assets.py", flush=True)
            ok = False
    print("Runtime checks passed." if ok else "Runtime checks failed; see the first failure above.", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
