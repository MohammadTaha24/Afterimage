"""CPU-only tests for the revision GPU driver and the Messidor-2 cohort builder.

    python -B finetuning/revision_jobs/test_revision_gpu_jobs.py -v

The queue is built against a temporary experiments folder, so no model, GPU or real data is needed.
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
sys.path.insert(0, str(HERE))
import revision_gpu_jobs as gpu  # noqa: E402
import build_messidor2_cohort as builder  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class FakeExperiments:
    """Temporarily point the driver and both queues at an empty experiments folder."""

    def __init__(self, root: Path):
        self.root = root
        data = root / "datasets" / "messidor2_verified"
        self.patches = [
            mock.patch.object(gpu.final, "FINAL", root / "final_runs"),
            mock.patch.object(gpu.final, "RUNS", root / "runs"),
            mock.patch.object(gpu.rz, "RESULTS", root / "retizero_runs"),
            mock.patch.object(gpu.final.reproduction, "BASE", root / "assets/qwen3-vl-8b-4bit"),
            mock.patch.object(gpu, "MESSIDOR2", data),
            mock.patch.dict(gpu.COHORT_CSV, {
                "messidor2_on": data / "qwen_512/clahe_on/test.csv", "messidor2_off": data / "qwen_512/clahe_off/test.csv",
                "messidor2_224_off": data / "retizero_224/clahe_off/test.csv",
                "messidor2_224_on": data / "retizero_224/clahe_on/test.csv"}),
        ]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


class Queue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = FakeExperiments(Path(self.tmp.name)).__enter__()

    def tearDown(self):
        self.fake.__exit__()
        self.tmp.cleanup()

    def test_full_queue_contents(self):
        jobs = gpu.build_jobs()
        stages = [(q, j.stage) for q, j in jobs]
        self.assertEqual(stages.count(("retizero", "messidor2")), 7)   # saved + 3 matched + 3 CLAHE
        self.assertEqual(stages.count(("qwen", "messidor2")), 14)      # 3 on lr2e-4 + 3 on + 3 off lr5e-4 + 3 off lr2e-4 + 2 zero-shot
        self.assertEqual(stages.count(("qwen", "timing")), 7)          # the 7 configurations in GAPS.md
        self.assertEqual(len(jobs), 28)
        self.assertFalse(any(j.kind == "train" for _, j in jobs))     # scoring only: nothing is trained
        names = [j.name for _, j in jobs]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("r16_on_lr0p0002_t43_a3408__messidor2_on", names)
        self.assertIn("r16_off_p2__messidor2_off", names)
        self.assertIn("timing_r16_on_previous_split_p1__aptos361_on", names)
        self.assertIn("r16_off_alpha16_lr0p0002_t44_a3409__messidor2_off", names)

    def test_clahe_off_lr2e4_scores_use_the_final_02_adapters(self):
        jobs = {j.name: j for _, j in gpu.build_jobs()}
        for seeds in ("t42_a3407", "t43_a3408", "t44_a3409"):
            run = f"r16_off_alpha16_lr0p0002_{seeds}"
            args = [str(a) for a in jobs[f"{run}__messidor2_off"].arguments]
            self.assertIn(str(gpu.final.FINAL / "final_02" / "train" / run / "best_lora"), args)
            self.assertIn(str(gpu.COHORT_CSV["messidor2_off"]), args)

    def test_no_rank_sweep_option(self):
        with self.assertRaises(SystemExit):
            gpu.main(["run", "--rank-sweep"])

    def test_every_job_writes_inside_the_new_campaigns(self):
        for queue, job in gpu.build_jobs():
            root = gpu.qwen_root() if queue == "qwen" else gpu.retizero_root()
            self.assertTrue(str(job.output).startswith(str(root)), job.output)

    def test_qwen_scores_use_the_matching_cohort_and_adapter(self):
        jobs = {j.name: j for _, j in gpu.build_jobs()}
        args = [str(a) for a in jobs["r16_off_p1__messidor2_off"].arguments]
        self.assertIn(str(gpu.COHORT_CSV["messidor2_off"]), args)
        self.assertIn(str(gpu.final.RUNS / "full_r16_off_duplicate_excluded_t42_a3407_01" / "best_lora"), args)
        zero = [str(a) for a in jobs["zeroshot_4bit__messidor2_on"].arguments]
        self.assertNotIn("--adapter", zero)
        timing = [str(a) for a in jobs["timing_r8_off_p1__aptos361_off"].arguments]
        self.assertIn("--reference-predictions", timing)
        self.assertEqual(timing[timing.index("--generation-samples") + 1], "-1")

    def test_retizero_clahe_runs_score_the_clahe_branch(self):
        jobs = {j.name: j for _, j in gpu.build_jobs()}
        self.assertIn(str(gpu.COHORT_CSV["messidor2_224_on"]), [str(a) for a in jobs["clahe_t44__messidor2_224_on"].arguments])
        self.assertIn(str(gpu.COHORT_CSV["messidor2_224_off"]), [str(a) for a in jobs["matched_t42__messidor2_224_off"].arguments])

    def test_pilot_is_small_and_separate(self):
        jobs = gpu.build_jobs(pilot=True)
        self.assertEqual(len(jobs), 4)   # 1 RetiZero, 2 Qwen Messidor-2, 1 timing
        for queue, job in jobs:
            # Its own folder: final_02__pilot_01 belongs to the lr 2e-4 training queue.
            self.assertIn("__revision_pilot_01", str(job.output))
            self.assertEqual(job.kind, "score")
            self.assertIn("--limit", [str(a) for a in job.arguments])

    def test_missing_inputs_wait_instead_of_failing(self):
        jobs = gpu.build_jobs()
        self.assertEqual({gpu.state_of(j) for _, j in jobs}, {"waiting"})   # nothing exists in the fake folder
        zero = next(j for _, j in jobs if j.name == "zeroshot_4bit__messidor2_on")
        self.assertIn(str(gpu.COHORT_CSV["messidor2_on"]), gpu.missing_inputs(zero))
        # Once the base and the cohort exist, the zero-shot job is ready to run.
        gpu.final.reproduction.BASE.mkdir(parents=True)
        gpu.COHORT_CSV["messidor2_on"].parent.mkdir(parents=True)
        gpu.COHORT_CSV["messidor2_on"].write_text("id_code\n", encoding="utf-8")
        self.assertEqual(gpu.state_of(zero), "pending")

    def test_finished_jobs_are_skipped(self):
        queue, job = next((q, j) for q, j in gpu.build_jobs() if j.name == "zeroshot_4bit__messidor2_off")
        job.output.mkdir(parents=True)
        (job.output / "summary.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(gpu.final, "run_job") as run, mock.patch.object(gpu.rz, "run_job") as rrun, \
                mock.patch.object(gpu.final, "lock_is_free", return_value=True), mock.patch.object(gpu, "PowerLog"):
            outcomes = gpu.run_jobs([(queue, job)], Path(self.tmp.name) / "state.json")
        run.assert_not_called()
        rrun.assert_not_called()
        self.assertEqual(outcomes[job.name], "done earlier")


class Energy(unittest.TestCase):
    def test_energy_integrates_the_job_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            start = datetime(2026, 11, 23, 1, 0, 0, tzinfo=timezone.utc)
            offset = datetime.now().astimezone().utcoffset()
            lines = ["timestamp, power.draw [W], power.limit [W], utilization.gpu [%], memory.used [MiB], temperature.gpu, clocks.current.sm [MHz]"]
            for k in range(0, 125, 5):  # 100 W for two minutes
                t = (start + timedelta(seconds=k)).replace(tzinfo=None) + offset
                lines.append(f"{t:%Y/%m/%d %H:%M:%S}.000, 100.00 W, 300.00 W, 90 %, 9000 MiB, 60, 2000 MHz")
            (tmp / "power_log_x.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
            state = tmp / "revision_queue_state.json"
            state.write_text(json.dumps([{"queue": "qwen", "job": "j", "stage": "s", "outcome": "passed",
                                          "started": start.isoformat(), "ended": (start + timedelta(seconds=120)).isoformat()}]))
            (tmp / "energy_per_job.csv").write_text("the training queue's own file\n", encoding="utf-8")
            out = gpu.energy_per_job(state)
            self.assertEqual(out.name, "revision_energy_per_job.csv")
            self.assertEqual((tmp / "energy_per_job.csv").read_text(encoding="utf-8"), "the training queue's own file\n")
            with open(out, encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["energy_wh"]), 100 * 120 / 3600, places=6)
            self.assertEqual(int(row["samples"]), 25)


class CohortBuilder(unittest.TestCase):
    def test_build_on_synthetic_images(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            raw.mkdir()
            names, grades = [], ["image_id,adjudicated_dr_grade,adjudicated_dme,adjudicated_gradable"]
            for k in range(6):
                name = f"2005{k:04d}_1_0400_PP.png" if k < 3 else f"IM00000{k}.JPG"
                image = np.zeros((300, 400, 3), np.uint8)
                cv2.circle(image, (200, 150), 120, (40 + 20 * k, 80, 160), -1)
                cv2.imwrite(str(raw / name), image)
                names.append(name)
                grades.append(f"{name.lower() if k == 4 else name},{k % 5},0,1" if k != 5 else f"{name},,,0")
            (tmp / "grades.csv").write_text("\n".join(grades) + "\n", encoding="utf-8")
            (tmp / "pairs.csv").write_text("left;right\n" + f"{names[0]};{names[1]}\n{names[3]};{names[4]}\n", encoding="utf-8")
            args = builder.argparse.Namespace(raw_dir=raw, grades=tmp / "grades.csv", pairs=tmp / "pairs.csv",
                                              out=tmp / "out", archive_dir=None, archived_ids=None, limit=0, workers=2)
            manifest = builder.build(args)
            self.assertEqual(manifest["images"], 5)                    # the ungradable image is left out
            self.assertEqual(manifest["grade_counts"], [1, 1, 1, 1, 1])
            rows = read(tmp / "out/qwen_512/clahe_on/test.csv")
            self.assertEqual({r["split"] for r in rows}, {"test"})
            exam = {r["id_code"]: r["exam_id"] for r in rows}
            self.assertEqual(exam[Path(names[0]).stem], exam[Path(names[1]).stem])   # one eye pair, one cluster
            self.assertEqual(exam[Path(names[3]).stem], exam[Path(names[4]).stem])   # case-insensitive JPG match
            for branch, (size, _) in builder.BRANCHES.items():
                image = cv2.imread(read(tmp / "out" / branch / "test.csv")[0]["image_path"])
                self.assertEqual(image.shape[:2], (size, size))
            self.assertEqual(builder.build(args)["images"], 5)          # rerun is a no-op


if __name__ == "__main__":
    unittest.main()
