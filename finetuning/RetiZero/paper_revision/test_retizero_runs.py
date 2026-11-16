"""CPU-only tests for the RetiZero helper and queue.

Run from this folder:  python -B -m unittest test_retizero_runs -v
"""

import tempfile
import unittest
from pathlib import Path

import torch

import retizero_runs as queue

helper = queue.helper


def settings_in(folder, *, pretrained=False):
    settings = queue.default_settings()
    weights = Path(folder) / "RetiZero.pth"
    if pretrained:
        weights.write_bytes(b"stand-in")
        settings["pretrained_sha256"] = helper.sha(weights)
    settings["pretrained"] = weights
    return settings


class CheckpointLayouts(unittest.TestCase):
    def app_state(self):
        encoder = {"lora_vit.blocks.0.attn.qkv.weight": torch.arange(6.0).reshape(2, 3),
                   "lora_vit.cls_token": torch.ones(1, 1, 4)}
        state = {f"img_encoder.{key}": value for key, value in encoder.items()}
        state.update({f"retizero.vision_model.model.{key}": value.clone() for key, value in encoder.items()})
        state.update({"retizero.text_model.weight": torch.zeros(2),
                      "classifier.weight": torch.zeros(5, 4), "classifier.bias": torch.zeros(5)})
        return state

    def test_app_checkpoint_maps_to_encoder_then_head(self):
        converted, info = helper.fine_tuned_state(self.app_state(), torch)
        self.assertEqual(set(converted), {"0.lora_vit.blocks.0.attn.qkv.weight", "0.lora_vit.cls_token",
                                          "1.weight", "1.bias"})
        self.assertIs(info["encoder_copy_identical"], True)

    def test_a_differing_encoder_copy_is_reported(self):
        state = self.app_state()
        state["retizero.vision_model.model.lora_vit.cls_token"] = torch.zeros(1, 1, 4)
        self.assertIs(helper.fine_tuned_state(state, torch)[1]["encoder_copy_identical"], False)

    def test_run_checkpoint_passes_through(self):
        state = {"0.lora_vit.cls_token": torch.ones(1), "1.weight": torch.ones(1), "1.bias": torch.ones(1)}
        converted, info = helper.fine_tuned_state(state, torch)
        self.assertEqual(set(converted), set(state))
        self.assertIsNone(info["encoder_copy_identical"])

    def test_unknown_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            helper.fine_tuned_state({"backbone.weight": torch.ones(1)}, torch)

    def test_pretrained_keeps_only_the_vision_encoder(self):
        state = {"vision_model.model.lora_vit.cls_token": torch.ones(1), "text_model.model.weight": torch.ones(1),
                 "logit_scale": torch.ones(1)}
        self.assertEqual(set(helper.pretrained_state(state)), {"0.lora_vit.cls_token"})

    def test_a_fine_tuned_checkpoint_cannot_start_training(self):
        with self.assertRaises(ValueError):
            helper.pretrained_state(self.app_state())
        with self.assertRaises(ValueError):
            helper.pretrained_state({"text_model.weight": torch.ones(1)})


class Training(unittest.TestCase):
    def test_accumulated_gradient_equals_one_weighted_batch(self):
        torch.manual_seed(0)
        images, grades = torch.randn(11, 6), torch.tensor([0, 1, 2, 3, 4, 0, 0, 2, 2, 4, 1])
        weights = torch.tensor([0.4, 2.1, 0.8, 4.0, 2.6])
        reference = torch.nn.Linear(6, 5)
        pieces = torch.nn.Linear(6, 5)
        pieces.load_state_dict(reference.state_dict())
        expected = torch.nn.functional.cross_entropy(reference(images), grades, weight=weights)
        expected.backward()
        value = helper.accumulated_backward(pieces, images, grades, weights, 4, torch)
        self.assertAlmostEqual(value, expected.item(), places=5)
        self.assertTrue(torch.allclose(pieces.weight.grad, reference.weight.grad, atol=1e-6))
        self.assertTrue(torch.allclose(pieces.bias.grad, reference.bias.grad, atol=1e-6))

    def test_class_weights_follow_the_original_formula(self):
        rows = [{"label": str(grade)} for grade, count in enumerate((6, 2, 4, 1, 2)) for _ in range(count)]
        weights = helper.class_weights_for(rows, torch, torch.device("cpu"))
        self.assertTrue(torch.allclose(weights, torch.tensor([15 / 30, 15 / 10, 15 / 20, 15 / 5, 15 / 10])))
        with self.assertRaises(ValueError):
            helper.class_weights_for(rows[:6], torch, torch.device("cpu"))

    def test_balanced_subset_keeps_order_and_needs_every_grade(self):
        rows = [{"id_code": str(i), "label": str(i % 5)} for i in range(40)]
        subset = helper.balanced_subset(rows, 3)
        self.assertEqual([row["id_code"] for row in subset], [str(i) for i in range(15)])
        with self.assertRaises(ValueError):
            helper.balanced_subset(rows, 9)

    def test_eval_transform_normalises_a_224_image(self):
        from PIL import Image
        pixels = helper.eval_transform()(Image.new("RGB", (224, 224), (255, 0, 0)))
        self.assertEqual(tuple(pixels.shape), (3, 224, 224))
        self.assertAlmostEqual(pixels[0, 0, 0].item(), (1 - helper.MEAN[0]) / helper.STD[0], places=4)
        self.assertAlmostEqual(pixels[1, 0, 0].item(), -helper.MEAN[1] / helper.STD[1], places=4)


class Metrics(unittest.TestCase):
    def test_metrics_come_from_the_pinned_function_with_macro_averages(self):
        predictions = []
        for index, (truth, guess) in enumerate([(0, 0), (0, 0), (1, 1), (1, 0), (2, 2), (3, 3), (4, 4), (4, 3)]):
            scores = [0.05] * 5
            scores[guess] = 0.8
            predictions.append({"image_id": str(index), "y_true": truth, "y_pred": guess,
                                **{f"p{i}": value for i, value in enumerate(scores)}})
        metrics = helper.calculate_metrics(predictions)
        self.assertEqual(metrics["samples"], 8)
        self.assertAlmostEqual(metrics["accuracy"], 6 / 8)
        self.assertAlmostEqual(metrics["macro_recall"], (1 + 0.5 + 1 + 1 + 0.5) / 5)
        self.assertAlmostEqual(metrics["macro_precision"], (2 / 3 + 1 + 1 + 0.5 + 1) / 5)

    def reported_metrics(self):
        reported = helper.REPORTED
        return {**{key: reported[key] for key in ("samples", "accuracy", "qwk", "macro_ovr_auc", "macro_precision",
                                                  "macro_recall", "macro_f1")},
                "per_class": [{"support": reported["support"][g], "precision": reported["precision"][g],
                               "recall": reported["recall"][g], "f1": reported["f1"][g]} for g in range(5)]}

    def test_replay_comparison_accepts_the_reported_figures(self):
        metrics = self.reported_metrics()
        metrics["accuracy"] = 0.82288828  # rounds to the reported 0.8229
        self.assertTrue(helper.compare_with_reported(metrics)["all_match"])

    def test_replay_comparison_flags_a_difference(self):
        metrics = self.reported_metrics()
        metrics["qwk"] = 0.9080
        result = helper.compare_with_reported(metrics)
        self.assertFalse(result["all_match"])
        self.assertEqual([check["item"] for check in result["checks"] if not check["match"]], ["qwk"])

    def test_specificity_from_a_confusion_matrix(self):
        metrics = {"confusion_matrix": [[8, 2, 0, 0, 0], [1, 4, 0, 0, 0], [0, 0, 5, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                   "per_class": [{"support": 10, "recall": 0.8}, {"support": 5, "recall": 0.8},
                                 {"support": 5, "recall": 1.0}, {"support": 0, "recall": 0.0},
                                 {"support": 0, "recall": 0.0}]}
        cells = queue.per_class(metrics)
        self.assertAlmostEqual(cells["specificity_0"], 9 / 10)
        self.assertAlmostEqual(cells["specificity_1"], 13 / 15)


@unittest.skipUnless(queue.default_settings()["datasets"]["clean"].is_file(), "224-pixel datasets are not on this PC")
class Datasets(unittest.TestCase):
    def test_both_splits_verify_against_their_qwen_builds(self):
        settings = queue.default_settings()
        for key, sizes in (("clean", (2786, 357, 361)), ("original", (2929, 366, 367))):
            tables, identity = helper.dataset_tables(settings["datasets"][key], "off",
                                                     matched_manifest=settings["qwen_datasets"][key],
                                                     verify_images=False)
            self.assertEqual(tuple(len(tables[split]) for split in helper.SPLITS), sizes)
            self.assertEqual(identity["split_name"], {"clean": "duplicate_excluded", "original": "previous_split"}[key])

    def test_a_missing_clahe_branch_is_refused(self):
        with self.assertRaises(ValueError):
            helper.dataset_tables(queue.default_settings()["datasets"]["clean"], "on", verify_images=False)

    def test_clean_test_ids_are_a_subset_of_the_original_test_ids(self):
        cohorts = queue.default_settings()["cohorts"]
        clean = {row["id_code"] for row in helper.cohort_rows(cohorts["aptos361_clean"])}
        original = {row["id_code"] for row in helper.cohort_rows(cohorts["aptos367_original"])}
        self.assertEqual((len(clean), len(original), len(clean - original)), (361, 367, 0))


class Queue(unittest.TestCase):
    def test_full_queue_has_every_stage_and_unique_names(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs = queue.build_queue(settings_in(folder))
        names = [job.name for job in jobs]
        self.assertEqual(len(names), len(set(names)))
        stages = {stage: sum(job.stage == stage for job in jobs) for stage in {job.stage for job in jobs}}
        self.assertEqual(stages, {"replay": 2, "historical": 3, "matched": 3, "clahe": 3, "benchmark": 3,
                                  "messidor_archived": 4})
        for job in jobs:
            self.assertEqual(job.output.parents[1], queue.RESULTS / "retizero_01")

    def test_training_waits_for_the_pretrained_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs = queue.build_queue(settings_in(folder))
            training = [job for job in jobs if job.kind == "train"]
            self.assertEqual({job.status() for job in training}, {"waiting"})
            self.assertTrue(all("--random-init" not in job.arguments for job in training))

    def test_a_present_checkpoint_without_its_hash_still_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = settings_in(folder, pretrained=True)
            settings["pretrained_sha256"] = ""
            job = next(job for job in queue.build_queue(settings) if job.name == "matched_t42")
            self.assertEqual(job.status(), "waiting")
            self.assertIn("pretrained_sha256", job.missing()[0])

    def test_stage_switches_remove_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = settings_in(folder)
            settings["stages"].update(historical=False, clahe=False, messidor_archived=False)
            jobs = queue.build_queue(settings)
        self.assertEqual({job.stage for job in jobs}, {"replay", "matched", "benchmark"})

    def test_training_arguments_carry_the_recipe(self):
        with tempfile.TemporaryDirectory() as folder:
            job = next(job for job in queue.build_queue(settings_in(folder)) if job.name == "historical_t43")
        arguments = dict(zip(job.arguments[::2], job.arguments[1::2]))
        self.assertEqual(float(arguments["--learning-rate"]), 7.68094483149829e-5)
        self.assertEqual((arguments["--effective-batch"], arguments["--seed"], arguments["--epochs"],
                          arguments["--patience"], arguments["--branch"]), (128, 43, 15, 4, "off"))
        self.assertTrue(str(arguments["--dataset"]).endswith(r"previous_split\manifest.json"))

    def test_pilot_uses_random_weights_only_without_the_pretrained_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            without = [job for job in queue.build_queue(settings_in(folder), pilot=True) if job.kind == "train"]
            ready = [job for job in queue.build_queue(settings_in(folder, pretrained=True), pilot=True)
                     if job.kind == "train"]
        self.assertTrue(all("--random-init" in job.arguments and "--pilot" in job.arguments for job in without))
        self.assertTrue(all("--random-init" not in job.arguments and "--weights" in job.arguments for job in ready))
        self.assertEqual([job.extra["stop_after"] for job in without], [None, 1])

    def test_pilot_results_stay_apart_from_full_results(self):
        settings = queue.default_settings()
        self.assertEqual(queue.root(settings, pilot=True).name, "retizero_01__pilot_01")
        self.assertNotEqual(queue.root(settings), queue.root(settings, pilot=True))
        settings["tag"] = r"..\runs"
        with self.assertRaises(ValueError):
            queue.root(settings)

    def test_job_status_follows_the_saved_files(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            job = queue.Job("matched", "matched_t42", "train", run, [], 1, needs=[Path(folder) / "weights"])
            self.assertEqual(job.status(), "waiting")
            (Path(folder) / "weights").write_bytes(b"x")
            self.assertEqual(job.status(), "pending")
            run.mkdir()
            (run / "configuration.json").write_text("{}", encoding="utf-8")
            self.assertEqual(job.status(), "interrupted")
            (run / "evaluation").mkdir()
            (run / "evaluation/aptos_test.summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(job.status(), "done")


if __name__ == "__main__":
    unittest.main()
