"""CPU-only checks for the final-runs queue and its workers.

Run from the repository root:
  python -B finetuning/Qwen3/paper_revision/test_final_runs.py -v
"""

import json
import struct
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in (HERE.parent, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import final_runs  # noqa: E402
import qwen_variant  # noqa: E402
import score_cohort  # noqa: E402
import score_fp32_fallback  # noqa: E402


class Queue(unittest.TestCase):
    def setUp(self):
        self.settings = final_runs.default_settings()

    def test_variant_names_are_unique_and_readable(self):
        self.assertEqual(final_runs.variant_name(16, "on", 1, {"lr": 1e-4}), "r16_on_lr0p0001_t42_a3407")
        self.assertEqual(final_runs.variant_name(16, "on", 3, {"lora_alpha": 32}), "r16_on_alpha32_t44_a3409")
        names = [final_runs.variant_name(*item[1:]) for item in final_runs.variants(self.settings)]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 12)  # two learning rates and two alphas, three seeds each

    def test_default_queue_trains_then_scores_each_new_adapter(self):
        jobs = final_runs.build_queue(self.settings)
        trained = [job for job in jobs if job.kind == "train"]
        self.assertEqual(len(trained), 12)
        for job in trained:
            follow = jobs[jobs.index(job) + 1]
            self.assertIn("--full", job.arguments)
            if job.status() == "fp16_overflow":  # real results: scored by the labelled fallback instead
                self.assertEqual(follow.name, f"{job.name}__aptos_test_fp32")
                continue
            self.assertEqual(follow.name, f"{job.name}__messidor_archived")
            self.assertEqual(follow.needs, job.output / "score_result.json")
        self.assertEqual(len({job.output for job in jobs}), len(jobs))
        self.assertTrue(all(final_runs.FINAL in job.output.parents for job in jobs))

    def test_new_results_stay_out_of_the_verified_runs_folder(self):
        for pilot in (False, True):
            for job in final_runs.build_queue(self.settings, pilot):
                self.assertNotIn(final_runs.RUNS, job.output.parents)

    def test_pilot_is_small_and_covers_each_kind(self):
        jobs = final_runs.build_queue(self.settings, pilot=True)
        trained = [job for job in jobs if job.kind == "train"]
        self.assertEqual(sorted(tuple(job.extra["overrides"]) for job in trained), [("lora_alpha",), ("lr",)])
        self.assertTrue(all("--full" not in job.arguments for job in trained))
        scored = [job for job in jobs if job.kind == "score"]
        self.assertTrue(all("--limit" in job.arguments for job in scored))
        self.assertTrue(any("bf16" in job.name for job in scored))
        self.assertLessEqual(len(jobs), 7)

    def test_grid_needs_a_chosen_learning_rate_and_skips_repeats(self):
        self.settings["stages"]["main_lr_grid"] = True
        with self.assertRaises(ValueError):
            final_runs.variants(self.settings)
        self.settings["main_lr"] = 1e-4
        names = [final_runs.variant_name(*item[1:]) for item in final_runs.variants(self.settings)]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 12 + 18 - 3)  # rank 16 CLAHE-on at 1e-4 is already in the study

    def test_parameter_count_from_header(self):
        header = json.dumps({"__metadata__": {"format": "pt"},
                             "a": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]},
                             "b": {"dtype": "F32", "shape": [5], "data_offsets": [48, 68]}}).encode()
        with tempfile.TemporaryDirectory(dir=HERE.parents[2] / ".verification") as folder:
            path = Path(folder) / "weights.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(68))
            self.assertEqual(final_runs.safetensors_parameters(path), 17)


def prediction_rows(grades):
    """Rows the pinned metrics accept: probabilities peak at the predicted grade."""
    rows = []
    for index, (truth, predicted) in enumerate(grades):
        scores = [0.1] * 5
        scores[predicted] = 0.6
        rows.append({"image_id": f"img{index}", "image_path": "", "y_true": truth, "y_pred": predicted,
                     **{f"p{grade}": value for grade, value in enumerate(scores)}})
    return rows


class FallbackQueue(unittest.TestCase):
    """FP16 overflow status, FP32 fallback jobs and the summary flag, on fake runs in a temp folder."""
    LAYER = "language_model.layers.2.mlp.down_proj"

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=HERE.parents[2] / ".verification")
        self.addCleanup(self.folder.cleanup)
        patcher = unittest.mock.patch.object(final_runs, "FINAL", Path(self.folder.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = final_runs.default_settings()
        self.settings["stages"] = {key: False for key in self.settings["stages"]}
        self.settings["stages"].update(alpha_study=True, messidor_for_new_runs=True)
        self.settings["alpha_values"] = [32]
        self.overflow = final_runs.variant_name(16, "on", 2, {"lora_alpha": 32})
        self.settings["fp16_overflow_diagnoses"] = {self.overflow: {"layer": self.LAYER, "detail": "test",
                                                                    "evidence": []}}

    def run_folder(self, pair):
        return final_runs.root(self.settings) / "train" / final_runs.variant_name(16, "on", pair, {"lora_alpha": 32})

    def fake_run(self, pair, *, status, log="", scored=None):
        run = self.run_folder(pair)
        attempt = run / "attempts" / "a1"
        attempt.mkdir(parents=True)
        (attempt / "score.log").write_text(log, encoding="utf-8")
        write = final_runs.reproduction.original.write_json
        write(run / "latest_attempt.json", {"path": str(attempt), "status": status})
        write(run / "pilot_configuration.json", {"rank": 16, "training": {
            "lr": 5e-4, "lora_alpha": 32, "lora_dropout": 0.05, "early_stopping_patience": 2}})
        write(run / "train_result.json", {"validation_loss": 0.1, "history": [{"epoch": 2}]})
        if scored is not None:
            final_runs.reproduction.original.write_csv(run / "aptos_test.predictions.csv", prediction_rows(scored))
            write(run / "score_result.json", {"metrics": {}})
        return run

    def fake_fallback_score(self, job, grades):
        rows = prediction_rows(grades)
        metrics = final_runs.reproduction.original.calculate_metrics(rows)
        job.output.mkdir(parents=True)
        final_runs.reproduction.original.write_json(job.output / "summary.json", {
            "identity": {"cohort": job.arguments[job.arguments.index("--cohort-name") + 1], "base_precision": "4bit",
                         "adapter": {"weights_sha256": "x"}, "scoring_precision": "FP32 fallback"},
            "metrics": metrics, "full_answer_timing": {"median_seconds": 1.0, "p95_seconds": 1.1},
            "model_load_seconds": 5.0, "generation": None, "torch_peak_allocated_bytes": 2**30,
            "fallback_reason": {"summary": f"FP16 overflow in {self.LAYER}."},
            "fp16_probe": {"outcome": "non-finite answer likelihood (FP16 failure reproduced)"}})

    def jobs_by_name(self):
        return {job.name: job for job in final_runs.build_queue(self.settings)}

    def test_fallback_names_make_the_precision_obvious(self):
        self.assertEqual(final_runs.fallback_name("r16_on_alpha32_t43_a3408", "aptos361_on"),
                         "r16_on_alpha32_t43_a3408__aptos_test_fp32")
        self.assertEqual(final_runs.fallback_name("r16_on_alpha32_t43_a3408", "messidor_archived"),
                         "r16_on_alpha32_t43_a3408__messidor_archived_fp32")

    def test_diagnosed_overflow_queues_aptos_then_messidor_fallback(self):
        run = self.fake_run(2, status="failed", log="ValueError: Non-finite answer likelihood for x.png")
        jobs = final_runs.build_queue(self.settings)
        names = [job.name for job in jobs]
        train = jobs[names.index(self.overflow)]
        self.assertEqual(train.status(), "fp16_overflow")
        self.assertEqual(train.label(), "FP16 scoring failed: overflow")
        aptos, messidor = jobs[names.index(self.overflow) + 1: names.index(self.overflow) + 3]
        self.assertEqual(aptos.name, f"{self.overflow}__aptos_test_fp32")
        self.assertEqual(messidor.name, f"{self.overflow}__messidor_archived_fp32")
        self.assertNotIn(f"{self.overflow}__messidor_archived", names)
        self.assertEqual(messidor.needs, aptos.output / "summary.json")
        self.assertEqual(messidor.status(), "waiting")
        for job in (aptos, messidor):
            self.assertEqual(job.extra["script"], final_runs.FALLBACK_SCRIPT)
            self.assertEqual(job.extra["fallback_for"], self.overflow)
            self.assertEqual(job.arguments[job.arguments.index("--overflow-layer") + 1], self.LAYER)
            self.assertEqual(job.arguments[job.arguments.index("--train-run") + 1], run)
            self.assertNotIn(run, job.output.parents)  # never writes into the training run
        self.assertEqual(aptos.arguments[aptos.arguments.index("--cohort-csv") + 1], run / "aptos_test.csv")
        self.assertIn("aptos_test_fp32", train.note)

    def test_undiagnosed_or_other_failures_queue_no_fallback(self):
        self.settings["fp16_overflow_diagnoses"] = {}
        self.fake_run(2, status="failed", log="ValueError: Non-finite answer likelihood for x.png")
        self.fake_run(3, status="failed", log="CUDA out of memory")
        jobs = self.jobs_by_name()
        self.assertEqual(jobs[self.overflow].status(), "fp16_nonfinite")
        self.assertEqual(jobs[self.run_folder(3).name].status(), "failed")
        self.assertFalse(any("fallback_for" in job.extra for job in jobs.values()))
        self.assertIn(f"{self.overflow}__messidor_archived", jobs)

    def test_queue_skips_the_overflow_run_and_finished_jobs(self):
        self.fake_run(2, status="failed", log="Non-finite answer likelihood")
        aptos = self.jobs_by_name()[f"{self.overflow}__aptos_test_fp32"]
        self.fake_fallback_score(aptos, [(grade, grade) for grade in range(5)])
        ran = []
        with unittest.mock.patch.object(final_runs, "run_job", side_effect=lambda job: ran.append(job.name)), \
                unittest.mock.patch.object(final_runs, "lock_is_free", return_value=True), \
                unittest.mock.patch.object(final_runs, "summarize"):
            outcomes = final_runs.run_queue(self.settings, only={"alpha_study"})
        self.assertNotIn(self.overflow, ran)
        self.assertNotIn(aptos.name, ran)
        self.assertIn(f"{self.overflow}__messidor_archived_fp32", ran)
        self.assertTrue(outcomes[self.overflow].startswith("FP16 scoring failed: overflow"))
        self.assertEqual(outcomes[aptos.name], "done earlier")

    def test_summary_flags_fallback_and_keeps_protocol_mean_separate(self):
        good = [(grade, grade) for grade in range(5)] * 2
        self.fake_run(1, status="passed", scored=good)
        self.fake_run(3, status="passed", scored=good[:-2] + [(3, 0), (4, 0)])
        self.fake_run(2, status="failed", log="Non-finite answer likelihood")
        aptos = self.jobs_by_name()[f"{self.overflow}__aptos_test_fp32"]
        self.fake_fallback_score(aptos, [(grade, 0) for grade in range(5)])
        rows = {row["job"]: row for row in final_runs.summarize(self.settings)}
        self.assertEqual(rows[aptos.name]["scoring_precision"], "FP32 fallback")
        self.assertEqual(rows[aptos.name]["fallback_for"], self.overflow)
        self.assertEqual(rows[self.overflow]["scoring_status"], "FP16 scoring failed: overflow")
        self.assertNotIn("accuracy", rows[self.overflow])
        self.assertEqual(rows[self.run_folder(1).name]["scoring_precision"], "FP16")
        where = final_runs.root(self.settings)
        with (where / "summary.csv").open(encoding="utf-8") as stream:
            saved = {row["job"]: row for row in __import__("csv").DictReader(stream)}
        self.assertEqual(saved[aptos.name]["scoring_precision"], "FP32 fallback")
        text = (where / "summary.md").read_text(encoding="utf-8")
        self.assertIn(f"| {aptos.name} | aptos361_on | 5 | **FP32 fallback** |", text)
        protocol = next(line for line in text.splitlines() if "| FP16 (protocol) |" in line)
        mixed = next(line for line in text.splitlines() if "| FP16 + FP32 fallback (mixed) |" in line)
        self.assertIn("| 2 (1 excluded: FP16 scoring failed) | 90.00 ± 14.14 |", protocol)
        self.assertIn("| 3 (incl. 1 FP32 fallback) |", mixed)
        self.assertIn("## FP16 scoring failures", text)
        self.assertIn(self.LAYER, text)


class FallbackScorer(unittest.TestCase):
    def test_needs_the_failed_fp16_log(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[2] / ".verification") as folder:
            log = Path(folder) / "score.log"
            log.write_text("Traceback\nValueError: Non-finite answer likelihood for a.png\n", encoding="utf-8")
            evidence = score_fp32_fallback.failure_evidence(log)
            self.assertIn("Non-finite answer likelihood", evidence["error"])
            self.assertEqual(len(evidence["log_sha256"]), 64)
            log.write_text("CUDA out of memory\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                score_fp32_fallback.failure_evidence(log)

    def test_refuses_a_run_that_already_has_a_protocol_score(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[2] / ".verification") as folder:
            run = Path(folder)
            for name in ("pilot_configuration.json", "train_result.json", "score_result.json"):
                (run / name).write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                score_fp32_fallback.check_train_run(run, {}, run / "aptos_test.csv")


class VariantCheck(unittest.TestCase):
    class Failure(Exception):
        pass

    def pinned(self, actual, config):
        for key, value in {"r": config["rank"], "lora_alpha": 16, "lora_dropout": 0.05}.items():
            if actual.get(key) != value:
                raise self.Failure(key)

    def test_declared_alpha_passes_and_undeclared_fails(self):
        check = qwen_variant.variant_adapter_check(self.pinned, self.Failure)
        config = {"rank": 16, "training": {"lora_alpha": 32, "lora_dropout": 0.05}}
        check({"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, config)
        with self.assertRaises(self.Failure):
            check({"r": 16, "lora_alpha": 16, "lora_dropout": 0.05}, config)
        with self.assertRaises(self.Failure):  # every other pinned check still applies
            check({"r": 8, "lora_alpha": 32, "lora_dropout": 0.05}, config)

    def test_unchanged_settings_behave_like_the_pinned_check(self):
        check = qwen_variant.variant_adapter_check(self.pinned, self.Failure)
        config = {"rank": 16, "training": {"lora_alpha": 16, "lora_dropout": 0.05}}
        check({"r": 16, "lora_alpha": 16, "lora_dropout": 0.05}, config)

    def test_overrides_are_recorded_with_their_pinned_values(self):
        class Runner:
            @staticmethod
            def identity(config):
                return "id-" + str(config["training"]["lr"])

        config = {"purpose": "Fresh APTOS comparison", "training": {"lr": 5e-4, "lora_alpha": 16}}
        result = qwen_variant.apply_overrides(config, {"lr": 1e-4}, "r16_on_lr0p0001_t42_a3407", Runner)
        self.assertEqual(result["training"]["lr"], 1e-4)
        self.assertEqual(result["variant"]["pinned_values"], {"lr": 5e-4})
        self.assertEqual(result["identity"], "id-0.0001")
        with self.assertRaises(ValueError):
            qwen_variant.apply_overrides(config, {"epochs": 3}, "bad", Runner)


class Scorer(unittest.TestCase):
    def test_parse_grade(self):
        self.assertEqual(score_cohort.parse_grade("<answer>2 - Moderate</answer>"), 2)
        self.assertEqual(score_cohort.parse_grade("text <answer> 4 - Proliferative DR"), 4)
        self.assertIsNone(score_cohort.parse_grade("Grade 2"))
        self.assertIsNone(score_cohort.parse_grade("<answer>7 - Unknown</answer>"))

    def test_specificity(self):
        confusion = [[8, 2], [1, 9]]
        self.assertEqual(score_cohort.specificity(confusion), [0.9, 0.8])

    def test_timing_excludes_warmup(self):
        summary = score_cohort.timing_summary([9.0, 9.0, 9.0, 1.0, 1.0, 2.0])
        self.assertEqual(summary["warmup_images_excluded"], 3)
        self.assertEqual(summary["median_seconds"], 1.0)
        self.assertAlmostEqual(summary["images_per_second"], 0.75)


if __name__ == "__main__":
    unittest.main()
