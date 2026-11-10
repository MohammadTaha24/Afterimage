"""Load both models through the backend and grade the bundled sample image.

Runs a private backend on port 9000 by default. Use --backend-url to exercise
an already-running backend instead (this switches/unloads that backend's models).
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def wait_for_backend(client, url, process):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Backend exited during startup; check the smoke-test log")
        try:
            if client.get(f"{url}/health").status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError("Backend startup exceeded 60 seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", help="Use an existing backend instead of starting one")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--models", nargs="+", choices=["retizero", "qwen3vl"], default=["retizero", "qwen3vl"])
    parser.add_argument("--image", type=Path, default=ROOT / "RetiZero" / "test_image.png")
    parser.add_argument("--timeout", type=float, default=900, help="Seconds allowed per model load")
    parser.add_argument("--prediction-timeout", type=float, default=180)
    parser.add_argument("--log", type=Path, default=ROOT / ".verification" / "smoke-test.log")
    parser.add_argument("--report", type=Path, default=ROOT / ".verification" / "smoke-test.json")
    args = parser.parse_args()
    raw = args.image.read_bytes()
    backend = None
    log_file = None
    url = (args.backend_url or f"http://127.0.0.1:{args.port}").rstrip("/")
    results = []
    attempted = []
    try:
        if not args.backend_url:
            # Refuse to terminate or silently take over a user's existing server.
            for port in (args.port, 8000, 8001):
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        raise RuntimeError(f"Port {port} is already in use. Stop its server first or use --backend-url.")
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log_file = args.log.open("w", encoding="utf-8")
            backend = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(args.port)],
                cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        with httpx.Client(timeout=60) as client:
            if backend:
                wait_for_backend(client, url, backend)
            for model in args.models:
                attempted.append(model)
                start = time.monotonic()
                response = client.post(f"{url}/model/prepare", data={"model": model})
                response.raise_for_status()
                deadline = start + args.timeout
                previous = None
                while time.monotonic() < deadline:
                    status_response = client.get(f"{url}/model/status")
                    status_response.raise_for_status()
                    status = status_response.json()
                    state = status["models"][model]
                    marker = (state["state"], state["progress"], state["message"])
                    if marker != previous:
                        print(f"{model}: {state['state']} {state['progress']}% {state['message']}", flush=True)
                        previous = marker
                    if state["state"] == "error":
                        raise RuntimeError(f"{model}: {state['last_error']}")
                    if state["state"] == "ready":
                        assert state["running"], f"{model} reported ready with no live process"
                        assert all(not s["running"] for name, s in status["models"].items() if name != model), "Both models are running"
                        break
                    time.sleep(0.5)
                else:
                    raise TimeoutError(f"{model} load exceeded {args.timeout}s")
                load_seconds = time.monotonic() - start
                start = time.monotonic()
                response = client.post(
                    f"{url}/predict", data={"model": model},
                    files={"file": (args.image.name, raw, "image/png")},
                    timeout=args.prediction_timeout,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"{model} prediction HTTP {response.status_code}: {response.text}")
                payload = response.json()
                assert type(payload["grade"]) is int and 0 <= payload["grade"] <= 4, payload
                results.append({"model": model, "load_seconds": round(load_seconds, 2),
                                "prediction_seconds": round(time.monotonic() - start, 2), "response": payload})
                print(f"PASS: {model} prediction {json.dumps(payload)}", flush=True)
            # Verify a stopped model cannot leave the upload flow in a ready state.
            for model in dict.fromkeys(attempted):
                client.post(f"{url}/model/unload", data={"model": model}).raise_for_status()
            status = client.get(f"{url}/model/status").json()
            assert all(s["state"] == "stopped" and not s["running"] for s in status["models"].values()), status
            print("PASS: switching and unloading release the model processes", flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"image": str(args.image.relative_to(ROOT)) if args.image.is_relative_to(ROOT) else str(args.image),
                                           "results": results}, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}\nBackend log: {args.log}", file=sys.stderr, flush=True)
        return 1
    finally:
        for model in dict.fromkeys(attempted):
            try:
                httpx.post(f"{url}/model/unload", data={"model": model}, timeout=60)
            except httpx.HTTPError:
                pass
        if backend:
            backend.terminate()
            try:
                backend.wait(timeout=20)
            except subprocess.TimeoutExpired:
                backend.kill()
                backend.wait(timeout=20)
        if log_file:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
