"""Regression tests for interrupted downloads and Windows model subprocesses."""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import download_model_assets as downloads
from backend import main as backend
from model_assets import AssetValidationError, validate_lora, validate_snapshot


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def snapshot(self):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
            (self.root / name).write_text("{}", encoding="utf-8")
        index = {"weight_map": {"layer1": "model-00001-of-00002.safetensors", "layer2": "model-00002-of-00002.safetensors"}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        for name in set(index["weight_map"].values()):
            (self.root / name).write_bytes(b"test weights")
        return self.root

    def lora(self):
        (self.root / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "base_model_name_or_path": "test/base"}), encoding="utf-8")
        (self.root / "adapter_model.bin").write_bytes(b"test weights")
        return self.root

    def test_all_shards_and_processor_are_required(self):
        self.snapshot()
        validate_snapshot(self.root, "qwen3")
        shard = self.root / "model-00002-of-00002.safetensors"
        shard.unlink()
        with self.assertRaises(AssetValidationError):
            validate_snapshot(self.root, "qwen3")
        shard.write_bytes(b"")
        with self.assertRaises(AssetValidationError):
            validate_snapshot(self.root, "qwen3")
        shard.write_bytes(b"recovered")
        (self.root / "preprocessor_config.json").unlink()
        with self.assertRaises(AssetValidationError):
            validate_snapshot(self.root, "qwen3")

    def test_corrupt_index_does_not_skip_download(self):
        self.snapshot()
        (self.root / "model.safetensors.index.json").write_text("broken", encoding="utf-8")
        with patch.object(downloads, "snapshot_download", return_value=str(self.root)) as fetch:
            self.assertIsNone(downloads.find_cached_qwen3_base("test/base"))
            fetch.assert_called_once_with(repo_id="test/base", local_files_only=True)

    def test_lora_bin_and_invalid_config(self):
        self.lora()
        validate_lora(self.root)
        (self.root / "adapter_config.json").write_text("", encoding="utf-8")
        with self.assertRaises(AssetValidationError):
            validate_lora(self.root)

    def test_existing_asset_performs_no_remote_download(self):
        self.lora()
        with patch.object(downloads, "QWEN3_LORA_TARGET", self.root), patch.object(downloads, "snapshot_download") as fetch:
            downloads.download_qwen3_lora("test/lora", "", None, False, False)
            fetch.assert_not_called()

    def test_force_reaches_retizero_download_api(self):
        checkpoint = self.root / "best_retizero_dr.pth"
        checkpoint.write_bytes(b"test weights")
        with patch.object(downloads, "RETIZERO_TARGET", self.root / "target.pth"), patch.object(downloads, "safe_copy_file"), patch.object(downloads, "hf_hub_download", return_value=str(checkpoint)) as fetch:
            # The actual destination name is also part of the project's contract.
            with patch.object(downloads, "RETIZERO_TARGET", self.root / "target" / "best_retizero_dr.pth"):
                downloads.download_retizero("test/retizero", "best_retizero_dr.pth", None, True)
            self.assertTrue(fetch.call_args.kwargs["force_download"])

    def test_force_reaches_lora_download_api(self):
        self.lora()
        with patch.object(downloads, "snapshot_download", return_value=str(self.root)) as fetch, patch.object(downloads, "safe_copy_tree"), patch.object(downloads, "QWEN3_LORA_TARGET", self.root):
            downloads.download_qwen3_lora("test/lora", "", None, False, True)
            self.assertTrue(fetch.call_args.kwargs["force_download"])

    def test_force_reaches_base_download_api(self):
        self.snapshot()
        with patch.object(downloads, "snapshot_download", return_value=str(self.root)) as fetch:
            downloads.download_qwen3_base("test/base", None, True)
            fetch.assert_called_once_with(repo_id="test/base", token=None, force_download=True)

    def test_clinicalbert_accepts_bin_and_requires_vocab(self):
        (self.root / "config.json").write_text("{}", encoding="utf-8")
        (self.root / "pytorch_model.bin").write_bytes(b"test weights")
        with self.assertRaises(AssetValidationError):
            validate_snapshot(self.root, "retizero-text")
        (self.root / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
        validate_snapshot(self.root, "retizero-text")

    def test_interrupted_copy_preserves_existing_checkpoint(self):
        source = self.root / "source.pth"
        destination = self.root / "destination.pth"
        source.write_bytes(b"new complete weights")
        destination.write_bytes(b"old complete weights")

        def interrupted_copy(src, dst):
            Path(dst).write_bytes(b"partial")
            raise OSError("simulated interruption")

        with patch.object(downloads.shutil, "copy2", side_effect=interrupted_copy):
            with self.assertRaises(OSError):
                downloads.safe_copy_file(source, destination)
        self.assertEqual(destination.read_bytes(), b"old complete weights")
        self.assertEqual(list(self.root.glob("*.partial")), [])


class ProcessTests(unittest.TestCase):
    def test_utf8_and_malformed_output_survive_and_keep_failure_detail(self):
        runtime = backend.ModelRuntime("qwen3vl")
        runtime.set_status("loading", 1, "Starting")
        code = "import sys; sys.stdout.buffer.write(b'PROGRESS:40:Loading \\xe2\\x96\\x8d \\xf0\\x9f\\xa6\\xa5\\n\\xff\\nRuntimeError: useful failure detail\\n'); sys.stdout.flush()"
        process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        runtime.process = process
        try:
            backend._read_process_output(runtime, process, runtime._stop_event)
            self.assertEqual(runtime.progress, 40)
            self.assertEqual(runtime.state, "error")
            self.assertIn("useful failure detail", runtime.last_error)
            self.assertTrue(any("\U0001f9a5" in line for line in runtime.stdout_lines))
        finally:
            process.wait(timeout=10)
            process.stdout.close()

    def test_exit_after_ready_is_an_error(self):
        runtime = backend.ModelRuntime("retizero")
        process = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE, encoding="utf-8")
        runtime.process = process
        runtime.set_status("ready", 100, "Model ready")
        try:
            backend._read_process_output(runtime, process, runtime._stop_event)
            self.assertEqual(runtime.state, "error")
        finally:
            process.wait(timeout=10)
            process.stdout.close()

    def test_cancelled_health_monitor_cannot_restore_ready(self):
        runtime = backend.runtimes["qwen3vl"]
        stop = threading.Event()
        stop.set()
        with patch.object(backend.httpx, "get") as get:
            backend._background_health_monitor("qwen3vl", None, stop)
            get.assert_not_called()

    def test_intentional_stop_keeps_stopped_state(self):
        runtime = backend.ModelRuntime("retizero")
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdout=subprocess.PIPE, encoding="utf-8")
        runtime.process = process
        runtime._reader_thread = threading.Thread(target=backend._read_process_output, args=(runtime, process, runtime._stop_event))
        runtime._reader_thread.start()
        backend._stop_runtime(runtime)
        self.assertEqual(runtime.state, "stopped")
        self.assertFalse(runtime.is_running())
        self.assertIsNone(runtime.last_error)
        self.assertFalse(runtime._reader_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
