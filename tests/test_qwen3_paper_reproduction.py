"""CPU/offline checks for split integrity, scoring and the unattended repeat queue."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

QWEN_DIR = Path(__file__).resolve().parents[1] / "finetuning/Qwen3"
sys.path.insert(0, str(QWEN_DIR))
import paper_reproduction as reproduction


class SplitAndMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self):
        (self.root / "mapped_a.png").write_bytes(b"fixture")
        (self.root / "mapped_b.png").write_bytes(b"fixture")
        originals = [
            {"image_file": "a.png", "label": "0", "split": "train"},
            {"image_file": "b.png", "label": "2", "split": "test"},
        ]
        mapping = [
            {"patient_id": "aptos_a", "source_dataset": "aptos", "image_path": "mapped_a.png", "label": "0", "split": "test"},
            {"patient_id": "aptos_b", "source_dataset": "aptos", "image_path": "mapped_b.png", "label": "2", "split": "train"},
        ]
        return originals, mapping

    def test_original_split_wins_over_different_mapping_split(self):
        originals, mapping = self.fixture()
        rows = reproduction.match_original_split(originals, mapping, self.root)
        self.assertEqual([row["split"] for row in rows], ["train", "test"])
        self.assertEqual([row["id_code"] for row in rows], ["a", "b"])
        self.assertNotIn("vlm_prompt", rows[0])
        self.assertTrue(Path(rows[0]["file_path"]).is_file())

    def test_label_changes_and_duplicate_ids_are_rejected(self):
        originals, mapping = self.fixture()
        mapping[1]["label"] = "3"
        with self.assertRaisesRegex(ValueError, "Label mismatch"):
            reproduction.match_original_split(originals, mapping, self.root)
        mapping[1]["label"] = "2"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reproduction.match_original_split(originals + [originals[0]], mapping, self.root)

    def test_missing_images_and_different_id_sets_are_rejected(self):
        originals, mapping = self.fixture()
        with self.assertRaisesRegex(ValueError, "different APTOS"):
            reproduction.match_original_split(originals, mapping[:1], self.root)
        (self.root / "mapped_b.png").unlink()
        with self.assertRaisesRegex(ValueError, "missing image"):
            reproduction.match_original_split(originals, mapping, self.root)

    def predictions(self, constant=False):
        result = []
        for truth in range(5):
            pred = 0 if constant else truth
            result.append({"image_id": str(truth), "image_path": f"image_{truth}.png", "y_true": truth,
                           "y_pred": pred, **{f"p{i}": float(i == pred) for i in range(5)}})
        return result

    def test_metrics_perfect_and_constant_predictor(self):
        perfect = reproduction.calculate_metrics(self.predictions())
        for key in ("accuracy", "qwk", "macro_ovr_auc", "macro_f1", "balanced_accuracy"):
            self.assertEqual(perfect[key], 1.0)
        constant = reproduction.calculate_metrics(self.predictions(constant=True))
        self.assertEqual(constant["accuracy"], 0.2)
        self.assertAlmostEqual(constant["qwk"], 0)
        self.assertEqual(constant["macro_ovr_auc"], 0.5)

    def test_invalid_scores_and_misaligned_resume_are_rejected(self):
        rows = self.predictions()
        expected = [{key: row[key] for key in ("image_id", "image_path", "y_true")} for row in rows]
        reproduction.validate_prediction_prefix(rows[:2], expected)
        with self.assertRaisesRegex(ValueError, "prefix"):
            reproduction.validate_prediction_prefix(rows[1:2], expected)
        rows[0]["p0"] = float("nan")
        with self.assertRaisesRegex(ValueError, "scores"):
            reproduction.calculate_metrics(rows)

    def test_training_settings_match_paper_and_original_defaults(self):
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)
        args = reproduction.make_sft_config({"training": reproduction.PAPER_TRAINING}, self.root, capture)
        self.assertEqual((args.per_device_train_batch_size, args.gradient_accumulation_steps), (4, 2))
        self.assertEqual((args.learning_rate, args.max_length, args.num_train_epochs), (5e-4, 2048, 50))
        self.assertEqual((args.warmup_steps, args.weight_decay, args.optim), (20, 0.001, "adamw_8bit"))
        self.assertEqual((args.eval_strategy, args.save_strategy, args.metric_for_best_model), ("epoch", "epoch", "eval_loss"))
        self.assertTrue(args.load_best_model_at_end)
        self.assertEqual(args.seed, 42)
        self.assertTrue(args.restore_callback_states_from_checkpoint)
        self.assertEqual(reproduction.PAPER_TRAINING["early_stopping_patience"], 2)
        self.assertEqual(reproduction.PAPER_TRAINING["expected_lora_default_seed"], 3407)

    def test_completed_adapter_must_match_its_record(self):
        config = {"fixed": True}
        adapter = self.root / "best_lora"
        adapter.mkdir()
        reproduction.write_json(adapter / "adapter_config.json", {"peft_type": "LORA", "base_model_name_or_path": "test/base"})
        weights = adapter / "adapter_model.safetensors"
        weights.write_bytes(b"test weights")
        reproduction.write_json(self.root / "training_complete.json", {
            "configuration": config, "adapter_sha256": reproduction.sha256_file(weights),
        })
        self.assertTrue(reproduction.completed_training(self.root, config))
        weights.write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "weights differ"):
            reproduction.completed_training(self.root, config)

    def test_evaluation_resumes_a_header_only_file_without_duplicate_header(self):
        import numpy as np
        _, mapping = self.fixture()
        cohort = self.root / "cohort.csv"
        reproduction.write_csv(cohort, [{
            "id_code": "b", "image_path": str(self.root / "mapped_b.png"), "label": 2,
            "split": "test", "source_dataset": "aptos",
        }])
        predictions = self.root / "aptos_test.predictions.csv"
        predictions.write_text("image_id,image_path,y_true,y_pred,p0,p1,p2,p3,p4\n", encoding="utf-8")
        with patch.object(reproduction, "score_full_answers", return_value=np.array([0., 0., 1., 0., 0.])):
            metrics = reproduction._evaluate_subset(None, None, cohort, self.root, "aptos_test", {})
        self.assertEqual(metrics["samples"], 1)
        self.assertEqual(len(reproduction.read_csv(predictions)), 1)
        self.assertEqual(len(predictions.read_text().splitlines()), 2)


class QueueTests(unittest.TestCase):
    def test_three_independent_runs_and_failure_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            reproduction.write_json(config, {"base_snapshot": "test/base"})
            manifest = {"csv_paths": {"test": "aptos.csv", "messidor_external": "messidor.csv"}}
            commands = []
            def fake_run(command, log_path):
                command = [str(part) for part in command]
                commands.append(command)
                mode = command[3]
                if mode == "train":
                    run_dir = Path(command[command.index("--run-dir") + 1])
                    if run_dir.name == "run_01":
                        raise subprocess.CalledProcessError(1, command)
                    reproduction.write_json(run_dir / "run_info.json", {
                        "best_epoch": 4, "epochs_completed": 6, "best_eval_loss": 0.01, "adapter_sha256": run_dir.name,
                    })
                else:
                    output = Path(command[command.index("--output-dir") + 1])
                    metrics = {"accuracy": 0.8, "qwk": 0.9, "macro_ovr_auc": 0.95, "macro_f1": 0.7, "balanced_accuracy": 0.7}
                    reproduction.write_json(output / "aptos_test.summary.json", metrics)
                    reproduction.write_json(output / "messidor_external.summary.json", metrics)
            with patch.object(reproduction, "_validate_training_config", return_value=manifest), patch.object(reproduction, "run_live", side_effect=fake_run):
                results = reproduction.run_three(config, root, include_external=True)
            self.assertEqual([row["status"] for row in results], ["failed", "complete", "complete"])
            train_commands = [command for command in commands if command[3] == "train"]
            self.assertEqual(len(train_commands), 3)
            directories = [command[command.index("--run-dir") + 1] for command in train_commands]
            self.assertEqual(len(set(directories)), 3)
            self.assertTrue(all("--adapter" not in command for command in train_commands))
            self.assertTrue((root / "run_01/failure.json").is_file())
            self.assertEqual(len(reproduction.read_csv(root / "batch_results.csv")), 3)


class ScoringTests(unittest.TestCase):
    def test_original_answer_masking_fp16_and_total_likelihood(self):
        import numpy as np
        import torch
        from PIL import Image
        class Batch(dict):
            def to(self, device):
                self.device = device
                return self
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.assert_prompt(messages[0])
                return "prompt" if len(messages) == 1 else messages[1]["content"][0]["text"]
            def assert_prompt(self, message):
                assert message["content"][0]["type"] == "text"
                assert message["content"][1]["type"] == "image"
            def __call__(self, images, text, return_tensors):
                ids = [1, 2] if text == "prompt" else [1, 2, reproduction.CANDIDATES.index(text) + 10]
                return Batch(input_ids=torch.tensor([ids]), pixel_values=torch.ones(1, dtype=torch.float32))
        class Model:
            device = "cpu"
            def __call__(self, input_ids, pixel_values, labels):
                assert labels[0, :2].tolist() == [-100, -100]
                assert pixel_values.dtype == torch.float16
                grade = int(labels[0, 2]) - 10
                return SimpleNamespace(loss=torch.tensor([2.0, 1.0, 0.1, 1.5, 2.5][grade]))
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.png"
            Image.new("RGB", (512, 512)).save(image)
            scores = reproduction.score_full_answers(Model(), Processor(), image)
        self.assertEqual(int(scores.argmax()), 2)
        self.assertAlmostEqual(float(np.sum(scores)), 1)


if __name__ == "__main__":
    unittest.main()
