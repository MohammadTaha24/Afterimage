"""CPU-only checks for the 2e-4 training queue (orchestrator_02).

Run from the repository root:
  python -B finetuning/Qwen3/paper_revision/test_orchestrator_02.py -v
"""

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
for folder in (HERE.parent, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import final_runs  # noqa: E402
import orchestrator_02 as orch  # noqa: E402

SCRATCH = HERE.parents[2] / ".verification"


def scratch():
    SCRATCH.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=SCRATCH)


class Queue(unittest.TestCase):
    def setUp(self):
        self.settings = orch.default_settings()
        self.estimates = {}

    def jobs(self, pilot=False):
        return orch.build_queue(self.settings, pilot, estimates=self.estimates)

    def test_phase_counts_and_order(self):
        trains = [job for job in self.jobs() if job.kind == "train"]
        self.assertEqual(sum(job.stage == "phase_A" for job in trains), 12)
        self.assertEqual(sum(job.stage == "phase_B" for job in trains), 15)
        self.assertEqual([job.name for job in trains[:4]],
                         ["r8_on_alpha16_lr0p0002_t42_a3407", "r8_on_alpha16_lr0p0002_t43_a3408",
                          "r8_on_alpha16_lr0p0002_t44_a3409", "r32_on_alpha16_lr0p0002_t42_a3407"])
        self.assertTrue(all(job.extra["clahe"] == "on" for job in trains[:12]))
        self.assertTrue(all(job.extra["clahe"] == "off" for job in trains[12:]))
        self.assertNotIn("r16_on_alpha16_lr0p0002_t42_a3407", [job.name for job in trains])
        self.assertEqual(len(self.jobs()), 27 * 3)

    def test_training_arguments_match_the_clean_split_runs(self):
        for job in (job for job in self.jobs() if job.kind == "train"):
            arguments = [str(value) for value in job.arguments]
            self.assertIn("--full", arguments)
            self.assertEqual(arguments[arguments.index("--lr") + 1], "0.0002")
            self.assertEqual(arguments[arguments.index("--lora-alpha") + 1], str(job.extra["alpha"]))
            self.assertEqual(arguments[arguments.index("--lock-dir") + 1], str(final_runs.RUNS))
            self.assertEqual(arguments[arguments.index("--candidate") + 1],
                             str(final_runs.DATA / "aptos_512/duplicate_excluded/manifest.json"))
            trainer, adapter = final_runs.SEEDS[job.extra["seed_pair"]]
            self.assertEqual(arguments[arguments.index("--trainer-seed") + 1], str(trainer))
            self.assertEqual(arguments[arguments.index("--adapter-seed") + 1], str(adapter))

    def test_each_run_is_followed_by_messidor_then_timing(self):
        jobs = self.jobs()
        for index, job in enumerate(jobs):
            if job.kind != "train":
                continue
            messidor, timing = jobs[index + 1], jobs[index + 2]
            self.assertEqual(messidor.extra["role"], "messidor")
            self.assertEqual(timing.extra["role"], "timing")
            self.assertEqual(messidor.needs, job.output / "score_result.json")
            arguments = [str(value) for value in timing.arguments]
            self.assertEqual(arguments[arguments.index("--generation-samples") + 1], "-1")
            self.assertEqual(arguments[arguments.index("--reference-predictions") + 1],
                             str(job.output / "aptos_test.predictions.csv"))
            cohort = [str(value) for value in messidor.arguments]
            cohort = cohort[cohort.index("--cohort-csv") + 1]
            if job.extra["clahe"] == "on":
                self.assertEqual(cohort, str(final_runs.COHORTS["messidor_archived"]))
            else:
                self.assertEqual(cohort, str(orch.root(self.settings) / "cohorts/messidor_archived_off.csv"))

    def test_outputs_stay_in_the_new_folder(self):
        for pilot in (False, True):
            where = orch.root(self.settings, pilot)
            for job in self.jobs(pilot):
                self.assertIn(where, job.output.parents)
                self.assertNotIn(final_runs.RUNS, job.output.parents)
                self.assertNotIn(final_runs.FINAL / "final_01", job.output.parents)
        self.assertNotIn(orch.root(self.settings), orch.root(self.settings, pilot=True).parents)
        self.settings["tag"] = "final_01"
        with self.assertRaises(ValueError):
            orch.root(self.settings)

    def test_switches(self):
        self.settings["phases"].update(A=False, B=True, C=False)
        self.assertEqual({job.stage for job in self.jobs()}, {"phase_B"})
        self.settings["phases"].update(A=True, B=False, C=True)
        jobs = self.jobs()
        self.assertEqual(sum(job.stage == "phase_C" for job in jobs), 7)
        self.assertTrue(all(job.kind == "score" for job in jobs if job.stage == "phase_C"))
        self.settings["messidor_off"] = None
        self.settings["phases"].update(A=False, B=True, C=False)
        self.assertEqual(len(self.jobs()), 15 * 2)

    def test_phase_c_covers_the_configurations_without_timing(self):
        labels = [item["label"] for item in self.settings["phase_c"]]
        self.assertEqual(len(labels), 7)
        self.assertIn("r16_on_lr0p0005_previous_split_t42_a3407", labels)
        self.settings["phases"]["C"] = True
        previous = next(job for job in self.jobs() if "previous_split" in job.name)
        arguments = [str(value) for value in previous.arguments]
        self.assertTrue(arguments[arguments.index("--cohort-csv") + 1].endswith("aptos_test.csv"))

    def test_pilot_is_small(self):
        self.settings["phases"]["C"] = True
        jobs = self.jobs(pilot=True)
        trains = [job for job in jobs if job.kind == "train"]
        self.assertEqual([job.extra["clahe"] for job in trains], ["on", "off"])
        self.assertTrue(all("--steps" in job.arguments and "--full" not in job.arguments for job in trains))
        scores = [job for job in jobs if job.kind == "score"]
        self.assertTrue(all(job.arguments[job.arguments.index("--limit") + 1] == 12 for job in scores))
        self.assertEqual(len(jobs), 7)

    def test_linked_final_01_runs(self):
        names = [run.name for run in orch.linked_runs(self.settings)]
        self.assertEqual(names, ["r16_on_lr0p0002_t42_a3407", "r16_on_lr0p0002_t43_a3408", "r16_on_lr0p0002_t44_a3409"])


class Fallback(unittest.TestCase):
    def test_non_finite_fp16_score_queues_labelled_fp32_jobs(self):
        settings = orch.default_settings()
        with scratch() as folder, mock.patch.object(orch, "FINAL", Path(folder)):
            first = orch.build_queue(settings, estimates={})[0]
            attempt = first.output / "attempts" / "a1"
            attempt.mkdir(parents=True)
            (first.output / "train_result.json").write_text("{}", encoding="utf-8")
            (attempt / "score.log").write_text("ValueError: Non-finite answer likelihood\n", encoding="utf-8")
            (first.output / "latest_attempt.json").write_text(
                json.dumps({"path": str(attempt), "status": "failed"}), encoding="utf-8")
            jobs = orch.build_queue(settings, estimates={})
            self.assertEqual(jobs[0].status(), "fp16_overflow")
            self.assertEqual([job.name for job in jobs[1:3]],
                             [f"{first.name}__aptos_test_fp32", f"{first.name}__messidor_archived_fp32"])
            self.assertTrue(all(job.extra["script"] == final_runs.FALLBACK_SCRIPT for job in jobs[1:3]))
            aptos = [str(value) for value in jobs[1].arguments]
            self.assertEqual(aptos[aptos.index("--cohort-csv") + 1], str(first.output / "aptos_test.csv"))
            self.assertEqual(jobs[2].needs, jobs[1].output / "summary.json")
            self.assertNotIn(f"{first.name}__aptos361_on", [job.name for job in jobs])
            settings["auto_fp32_fallback"] = False
            self.assertEqual(orch.build_queue(settings, estimates={})[0].status(), "fp16_nonfinite")


class Logging(unittest.TestCase):
    def test_energy_integration_and_idle_subtraction(self):
        with scratch() as folder:
            where = Path(folder)
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            with (where / "power_log.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, orch.POWER_FIELDS)
                writer.writeheader()
                for second in range(0, 125, 5):
                    writer.writerow({"utc": (start + timedelta(seconds=second)).isoformat(), "power_draw_w": 100.0})
            with (where / "power_events.csv").open("w", encoding="utf-8", newline="") as stream:
                stream.write("utc,job,event,attempt\n")
                stream.write(f"{(start + timedelta(seconds=12)).isoformat()},job1,start,a\n")
                stream.write(f"{(start + timedelta(seconds=84)).isoformat()},job1,end,a\n")
            (where / "idle_baseline.json").write_text(json.dumps([{"end_utc": start.isoformat(), "mean_w": 40.0}]),
                                                      encoding="utf-8")
            rows = orch.energy_per_job(where)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["energy_wh"], 100 * 72 / 3600, places=3)
            self.assertAlmostEqual(rows[0]["energy_above_idle_wh"], 60 * 72 / 3600, places=3)
            self.assertTrue(rows[0]["fully_covered"])
            self.assertTrue((where / "energy_per_job.csv").is_file())

    def test_power_logger_writes_rows(self):
        with scratch() as folder:
            path = Path(folder) / "power_log.csv"
            reading = [{"utc": "2026-01-01T00:00:00+00:00", "gpu_index": "0", "power_draw_w": 50.0}]
            with orch.PowerLogger(path, interval=0.05, reader=lambda: reading):
                import time
                time.sleep(0.3)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split(",")[:3], ["utc", "gpu_index", "power_draw_w"])
            self.assertGreaterEqual(len(lines), 3)

    def test_power_logger_reads_once_more_after_stop(self):
        with scratch() as folder:
            calls = []
            logger = orch.PowerLogger(Path(folder) / "p.csv", interval=60,
                                      reader=lambda: calls.append(1) or [{"utc": "t", "power_draw_w": 1.0}])
            with logger:
                pass
            self.assertEqual(len(calls), 2)  # first reading, then the closing one

    def test_stop_file(self):
        settings = orch.default_settings()
        with scratch() as folder, mock.patch.object(orch, "FINAL", Path(folder)):
            self.assertFalse(orch.stop_file(settings).is_file())
            orch.request_stop(settings)
            self.assertTrue(orch.stop_file(settings).is_file())

    def test_disk_warning(self):
        usage = mock.Mock(free=20e9)
        with mock.patch.object(orch.shutil, "disk_usage", return_value=usage):
            self.assertIn("20 GB", orch.disk_warning(Path("D:/x"), 30))
        usage.free = 40e9
        with mock.patch.object(orch.shutil, "disk_usage", return_value=usage):
            self.assertIsNone(orch.disk_warning(Path("D:/x"), 30))

    def test_attempt_check_names_missing_files(self):
        with scratch() as folder:
            attempt = Path(folder)
            self.assertIn("stage log", orch.check_attempt(attempt))
            for name in orch.ATTEMPT_FILES:
                (attempt / name).write_text("{}", encoding="utf-8")
            (attempt / "utilization.csv").write_text("utc\nx\n", encoding="utf-8")
            (attempt / "score.log").write_text("ok", encoding="utf-8")
            (attempt / "system_specs.json").write_text(json.dumps({
                "gpu": ["0, GPU"], "torch_cuda_runtime": "13.0",
                "packages": {"torch": "1", "transformers": "1", "unsloth": "1", "peft": "1"}}), encoding="utf-8")
            self.assertEqual(orch.check_attempt(attempt), [])

    def test_training_speed_uses_images_seen_not_the_inflated_trainer_figure(self):
        with scratch() as folder:
            run = Path(folder)
            (run / "attempts" / "a").mkdir(parents=True)
            (run / "attempts" / "a" / "train.log").write_text("", encoding="utf-8")
            (run / "attempts" / "a" / "run_report.md").write_text(
                "- train: exit 0, 1100.0 s\n- score: exit 0, 300.0 s\n", encoding="utf-8")
            (run / "resolved_training_arguments.json").write_text(json.dumps(
                {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 2}), encoding="utf-8")
            (run / "train_result.json").write_text(json.dumps({
                "global_step": 1000, "checkpoint": "c", "validation_loss": 0.1, "torch_peak_allocated_bytes": 5,
                "history": [{"epoch": 3.0, "step": 1000},
                            {"train_runtime": 1000.0, "train_samples_per_second": 139.3, "epoch": 3.0}]}),
                encoding="utf-8")
            numbers = orch.training_numbers(run)
            self.assertEqual(numbers["images_processed"], 8000)
            self.assertAlmostEqual(numbers["train_images_per_second"], 8.0)
            self.assertEqual(numbers["train_stage_seconds"], 1100.0)
            self.assertEqual(numbers["epochs_run"], 3.0)

    def test_messidor_off_cohort_keeps_rows_and_changes_only_paths(self):
        with scratch() as folder:
            folder = Path(folder)
            images = folder / "off"
            images.mkdir()
            source = folder / "on.csv"
            rows = [{"id_code": f"m{i}", "filename": f"m{i}.png", "file_path": f"on/m{i}.png",
                     "image_path": f"on/m{i}.png", "label": str(i)} for i in range(3)]
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for row in rows[:2]:
                (images / row["filename"]).write_bytes(b"png")
            with self.assertRaises(FileNotFoundError):
                orch.build_messidor_off(folder / "out.csv", source, images)
            (images / "m2.png").write_bytes(b"png")
            built = orch.reproduction.original.read_csv(orch.build_messidor_off(folder / "out.csv", source, images))
            self.assertEqual([(r["id_code"], r["label"]) for r in built], [(r["id_code"], r["label"]) for r in rows])
            self.assertTrue(all(Path(r["image_path"]).parent == images for r in built))
            self.assertEqual(len(json.loads((folder / "out.json").read_text(encoding="utf-8"))["image_sha256"]), 3)
            orch.build_messidor_off(folder / "out.csv", source, images)  # rebuilding is a no-op

    def test_grid_table_has_one_line_per_configuration(self):
        rows = [{"phase": "A", "source": "final_02", "clahe": "on", "rank": 8, "alpha": 16, "aptos_accuracy": 0.7 + i / 100,
                 "aptos_qwk": 0.8, "aptos_macro_f1": 0.5, "aptos_macro_auc": 0.9, "aptos_collapsed": i == 0}
                for i in range(3)]
        rows.append({"phase": "linked", "source": "final_01", "clahe": "on", "rank": 16, "alpha": 16,
                     "aptos_accuracy": 0.75, "aptos_qwk": 0.88, "aptos_macro_f1": 0.6, "aptos_macro_auc": 0.94})
        lines = orch.grid_lines(rows)
        self.assertEqual(len(lines), 4)
        self.assertIn("71.00 ± 1.00", lines[2])
        self.assertIn("final_01", lines[3])


if __name__ == "__main__":
    unittest.main()
