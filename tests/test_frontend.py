"""Exercise the model-load UI without running a GPU model or contacting a backend."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "frontend" / "app.py"


class ModelLoadUITests(unittest.TestCase):
    def test_ready_response_unlocks_uploader(self):
        app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("file_uploader")), 0)
        ready = Mock()
        ready.json.return_value = {"models": {"qwen3vl": {
            "state": "ready", "progress": 100, "message": "Model ready", "last_error": None,
        }}}
        with patch.object(requests, "post", return_value=Mock()), patch.object(requests, "get", return_value=ready):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.loaded_model_key, "qwen3vl")
        self.assertEqual(len(app.get("file_uploader")), 1)

    def test_backend_connection_error_keeps_uploader_locked(self):
        app = AppTest.from_file(str(APP)).run()
        with patch.object(requests, "post", side_effect=requests.ConnectionError("backend unavailable")):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.load_state, "error")
        self.assertIn("backend unavailable", app.session_state.load_error)
        self.assertEqual(len(app.get("file_uploader")), 0)


if __name__ == "__main__":
    unittest.main()
