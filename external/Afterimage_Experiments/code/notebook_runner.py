"""Launch a notebook worker and keep a copy of its console output."""

import os
import queue
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def stay_awake(enabled):
    if not enabled or os.name != "nt":
        yield
        return
    import ctypes

    continuous, system_required = 0x80000000, 0x00000001
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint32]
    set_state.restype = ctypes.c_uint32
    previous = set_state(continuous | system_required)
    if not previous:
        print("Windows did not accept the idle sleep request.", flush=True)
    try:
        yield
    finally:
        if previous:
            set_state(continuous)


def stage_log_path(line, output):
    match = re.fullmatch(r"Starting (train|score)\. Log: (.+)", line.rstrip("\r\n"))
    if not match:
        return None
    stage, name = match.groups()
    path = Path(name).resolve()
    if path.name == f"{stage}.log" and path.parent.parent == output / "attempts":
        return path
    return None


def stream_qwen(child, output, echo):
    messages = queue.Queue()

    def read_launcher():
        try:
            for line in child.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    threading.Thread(target=read_launcher, daemon=True).start()
    stage_path = None
    stage_file = None

    def drain_stage():
        nonlocal stage_file
        if stage_path is None:
            return
        if stage_file is None:
            try:
                stage_file = stage_path.open("r", encoding="utf-8", errors="replace")
            except FileNotFoundError:
                return
        while chunk := stage_file.read(65536):
            echo(chunk)

    try:
        while True:
            try:
                line = messages.get(timeout=0.25)
            except queue.Empty:
                drain_stage()
                continue
            drain_stage()
            if line is None:
                break
            echo(line)
            new_stage = stage_log_path(line, output)
            if new_stage is not None and new_stage != stage_path:
                if stage_file is not None:
                    stage_file.close()
                    stage_file = None
                stage_path = new_stage
    finally:
        if stage_file is not None:
            stage_file.close()


def run_command(script, arguments, log_path, *, keep_pc_awake=False):
    command = [sys.executable, "-B", "-u", str(script), *map(str, arguments)]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with stay_awake(keep_pc_awake), log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {Path(script).name}\n")
        log.flush()
        print(f"Log: {log_path}", flush=True)
        def echo(text):
            print(text, end="", flush=True)
            log.write(text)
            log.flush()

        with subprocess.Popen(
            command, cwd=Path(script).parent, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        ) as child:
            try:
                if Path(script).name in {"qwen_pilot.py", "recover_qwen.py"} and "--output" in command:
                    output = Path(command[command.index("--output") + 1]).resolve()
                    stream_qwen(child, output, echo)
                else:
                    for line in child.stdout:
                        echo(line)
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    # The Qwen runner owns separate train and score subprocesses.
                    if os.name == "nt":
                        try:
                            subprocess.run(
                                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10,
                            )
                        except (OSError, subprocess.TimeoutExpired) as error:
                            log.write(f"Process tree cleanup reported {error}\n")
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                raise
    if code:
        raise RuntimeError(f"Worker exited with code {code}. See {log_path}.")
