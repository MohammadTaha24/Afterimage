"""Unit tests for the parsing and aggregation helpers in build_efficiency_tables.py (synthetic records, no experiment tree).

Run from the repo root:
  C:\\Users\\moht2\\.conda\\envs\\afterimage_env\\python.exe -X utf8 -B -m unittest finetuning.paper_tables.test_efficiency_tables
or: ...python.exe -X utf8 -B finetuning\\paper_tables\\test_efficiency_tables.py
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_efficiency_tables as bet  # noqa: E402

HEADER = "utc,stage,gpu_index,gpu_percent,vram_mib,cpu_percent,ram_bytes\n"


def util_rows(stage, start_s, n, gpu=50.0, vram=2000.0):
    rows = []
    for i in range(n):
        t = start_s + 5 * i
        rows.append({"utc": f"2026-11-16T12:{t // 60:02d}:{t % 60:02d}+00:00", "stage": stage, "gpu_index": "0",
                     "gpu_percent": str(gpu + i), "vram_mib": str(vram + 10 * i), "cpu_percent": "10.0", "ram_bytes": str(2 ** 30 * 4)})
    return rows


def write_util(folder: Path, rows):
    folder.mkdir(parents=True, exist_ok=True)
    text = HEADER + "".join(",".join(r[k] for k in HEADER.strip().split(",")) + "\n" for r in rows)
    (folder / "utilization.csv").write_text(text, encoding="utf-8")


class Statistics(unittest.TestCase):
    def test_mean_sd_uses_sample_sd_and_handles_one_value(self):
        mean, sd = bet.mean_sd([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(sd, 1.0)
        self.assertEqual(bet.mean_sd([5.0]), (5.0, None))
        self.assertEqual(bet.mean_sd([None, float("nan")]), (None, None))

    def test_timing_stats_skips_warmup(self):
        stats = bet.timing_stats([100.0, 100.0, 100.0, 1.0, 2.0, 3.0], warmup=3)
        self.assertEqual(stats["images_timed"], 3)
        self.assertAlmostEqual(stats["median_s"], 2.0)
        self.assertAlmostEqual(stats["images_per_s"], 0.5)

    def test_aggregate_groups_and_ignores_missing(self):
        rows = [{"c": "a", "x": 1.0}, {"c": "a", "x": 3.0}, {"c": "b", "x": None}]
        out = {r["c"]: r for r in bet.aggregate(rows, ["c"], ["x"])}
        self.assertEqual(out["a"]["n"], 2)
        self.assertAlmostEqual(out["a"]["x_mean"], 2.0)
        self.assertAlmostEqual(out["a"]["x_sd"], math.sqrt(2))
        self.assertIsNone(out["b"]["x_mean"])


class Parsing(unittest.TestCase):
    def test_run_report_md(self):
        text = ("Status: **failed**.\n- train: exit 0, 4696.77 s\n"
                "- train: 931 samples, mean GPU 87.1%, mean CPU 14.6%, peak device VRAM 11448 MiB, peak system RAM 24.20 GiB\n")
        parsed = bet.parse_run_report_md(text)
        self.assertEqual(parsed["status"], "failed")
        self.assertEqual(parsed["wall"]["train"], (4696.77, "0"))
        self.assertEqual(parsed["util"]["train"], (931.0, 87.1, 14.6, 11448.0, 24.2))

    def test_train_log(self):
        text = ("Trainable parameters = 51,346,944 of 8,818,470,640\n"
                "{'train_runtime': '2891', 'train_samples_per_second': '48.19', 'epoch': '3'}\n")
        parsed = bet.parse_train_log(text)
        self.assertEqual((parsed["trainable"], parsed["total"], parsed["trainer_runtime_s"]), (51346944, 8818470640, 2891.0))

    def test_utilization_summary(self):
        summary = bet.utilization_summary(util_rows("train", 0, 4) + util_rows("score", 100, 2), "train")
        self.assertEqual(summary["samples"], 4)
        self.assertAlmostEqual(summary["span_s"], 15.0)
        self.assertAlmostEqual(summary["gpu_mean_pct"], 51.5)
        self.assertEqual(summary["gpu_peak_pct"], 53.0)
        self.assertEqual(summary["card_peak_vram_mib"], 2030.0)
        self.assertEqual(summary["card_baseline_vram_mib"], 2000.0)
        self.assertAlmostEqual(summary["ram_peak_gib"], 4.0)
        self.assertIsNone(bet.utilization_summary([], "train"))


class AttemptWall(unittest.TestCase):
    def test_recorded_time_preferred_then_span_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            first, second = run / "attempts" / "a1", run / "attempts" / "a2"
            write_util(first, util_rows("train", 0, 21))               # interrupted: no report, 100 s span
            write_util(second, util_rows("train", 0, 5))
            (second / "run_report.md").write_text("Status: **passed**.\n- train: exit 0, 2693.24 s\n", encoding="utf-8")
            a = bet.attempt_stage_wall(first, "train")
            b = bet.attempt_stage_wall(second, "train")
            self.assertEqual((a["seconds"], a["exit"]), (100.0, None))
            self.assertTrue(a["source"].startswith("utilization"))
            self.assertEqual((b["seconds"], b["source"]), (2693.24, "run_report.md"))
            self.assertIsNone(bet.attempt_stage_wall(first, "score"))

    def test_run_report_json_for_scoring_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            (attempt / "run_report.json").write_text(json.dumps({"stage": "score", "elapsed_seconds": 952.5}), encoding="utf-8")
            self.assertEqual(bet.attempt_stage_wall(attempt, "score")["seconds"], 952.5)
            self.assertIsNone(bet.attempt_stage_wall(attempt, "train"))

    def test_training_wall_summed_over_all_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run_x"
            for name, rows, report in (("a1", util_rows("train", 0, 21), None),
                                       ("a2", util_rows("train", 0, 5) + util_rows("score", 30, 3),
                                        "Status: **passed**.\n- train: exit 0, 1000.0 s\n- score: exit 0, 60.0 s\n")):
                write_util(folder / "attempts" / name, rows)
                if report:
                    (folder / "attempts" / name / "run_report.md").write_text(report, encoding="utf-8")
                (folder / "attempts" / name / "train.log").write_text(
                    "Trainable parameters = 8 of 80\n{'train_runtime': '900'}\n", encoding="utf-8")
            cfg = {"rank": 16, "clahe": True, "adapter_seed": 3407, "rows": {"train": 100, "test": 20},
                   "training": {"load_in_4bit": True, "lr": 0.0005, "lora_alpha": 16, "trainer_seed": 42}}
            (folder / "pilot_configuration.json").write_text(json.dumps(cfg), encoding="utf-8")
            (folder / "train_result.json").write_text(json.dumps(
                {"global_step": 20, "history": [{"epoch": 1.0}, {"epoch": 2.0}], "torch_peak_allocated_bytes": 2 ** 30,
                 "torch_peak_reserved_bytes": 2 * 2 ** 30}), encoding="utf-8")
            record = bet.qwen_training_record(folder, "test")
            self.assertEqual(record["train_wall_all_attempts_s"], 1100.0)
            self.assertEqual(record["train_wall_completing_attempt_s"], 1000.0)
            self.assertEqual(record["train_wall_other_attempts_s"], 100.0)
            self.assertEqual(record["attempts_timed_from_utilization_span"], 1)
            self.assertEqual(record["epochs_run"], 2.0)
            self.assertAlmostEqual(record["s_per_epoch"], 450.0)
            self.assertAlmostEqual(record["images_per_s"], 100 * 2 / 900)
            self.assertAlmostEqual(record["torch_peak_alloc_gib"], 1.0)
            self.assertEqual(record["config"], "4-bit r16 CLAHE-on lr 0.0005 alpha 16")
            self.assertEqual(record["inline_score_wall_s"], 60.0)


class Labels(unittest.TestCase):
    def test_config_label_bf16_offload_and_previous_split(self):
        cfg = {"rank": 16, "clahe": False, "offload_layers": 16, "historical_split": True,
               "training": {"load_in_4bit": False, "lr": 0.0002, "lora_alpha": 32}}
        self.assertEqual(bet.config_label(cfg), "BF16 (+16 layers CPU-offloaded) r16 CLAHE-off lr 0.0002 alpha 32 previous split")

    def test_base_bytes_sums_only_safetensors(self):
        cfg = {"base_files": {"a.safetensors": {"size": 10}, "b.safetensors": {"size": 5}, "c.json": {"size": 99}}}
        self.assertEqual(bet.base_bytes(cfg), 15)
        self.assertIsNone(bet.base_bytes({}))


if __name__ == "__main__":
    unittest.main()
