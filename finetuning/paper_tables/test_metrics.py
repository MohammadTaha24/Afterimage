"""Unit tests for the metric functions in build_paper_tables.py, on hand-computed cases.

Run from the repo root:
  C:\\Users\\moht2\\.conda\\envs\\afterimage_env\\python.exe -I finetuning\\paper_tables\\test_metrics.py
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_paper_tables as bpt  # noqa: E402


class AccuracyAndKappa(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(bpt.accuracy([0, 1, 2, 3], [0, 1, 0, 3]), 0.75)

    def test_qwk_hand_computed_three_classes(self):
        # O has (0,0),(1,2),(2,2); sum(W*O) = 0.25; sum(W*E) = 1.25; kappa = 1 - 0.25/1.25 = 0.8
        self.assertAlmostEqual(bpt.qwk([0, 1, 2], [0, 2, 2], n_classes=3), 0.8)

    def test_qwk_perfect_and_constant(self):
        self.assertAlmostEqual(bpt.qwk([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(bpt.qwk([0, 1, 2, 3, 4], [0, 0, 0, 0, 0]), 0.0)

    def test_qwk_undefined_is_nan(self):
        self.assertTrue(math.isnan(bpt.qwk([2, 2, 2], [2, 2, 2])))

    def test_qwk_batched_matches_scalar(self):
        y, p = [0, 1, 2, 4, 3, 1], [0, 2, 2, 3, 3, 0]
        single = bpt.qwk(y, p)
        batch = bpt.qwk_from_confusion(np.stack([bpt.confusion(y, p)] * 3))
        np.testing.assert_allclose(batch, single)


class RocAuc(unittest.TestCase):
    def test_no_ties(self):
        # Positive scores {0.9, 0.4}, negative {0.1, 0.6}: 3 of 4 pairs ordered correctly.
        self.assertAlmostEqual(bpt.auc_ovr([1, 0, 1, 0], [0.9, 0.1, 0.4, 0.6], 1), 0.75)

    def test_ties_count_half(self):
        self.assertAlmostEqual(bpt.auc_ovr([1, 0], [0.5, 0.5], 1), 0.5)
        # pairs: (0.5 vs 0.5) = 0.5, (0.7 vs 0.5) = 1 -> 0.75
        self.assertAlmostEqual(bpt.auc_ovr([1, 1, 0], [0.5, 0.7, 0.5], 1), 0.75)

    def test_absent_class_is_nan(self):
        self.assertTrue(math.isnan(bpt.auc_ovr([1, 1, 1], [0.2, 0.3, 0.4], 1)))
        self.assertTrue(math.isnan(bpt.auc_ovr([0, 0, 0], [0.2, 0.3, 0.4], 1)))

    def test_weights_equal_explicit_repetition(self):
        truth = np.array([1, 0, 1, 0, 0])
        scores = np.array([0.3, 0.3, 0.8, 0.5, 0.1])
        weights = np.array([2, 1, 0, 3, 1])
        repeated = np.repeat(np.arange(5), weights)
        expected = bpt.auc_ovr(truth[repeated], scores[repeated], 1)
        self.assertAlmostEqual(float(bpt.weighted_auc(truth == 1, scores, weights)[0]), expected)

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        truth = rng.integers(0, 2, 60)
        scores = rng.integers(0, 10, 60) / 10  # many ties
        pos, neg = scores[truth == 1], scores[truth == 0]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        self.assertAlmostEqual(bpt.auc_ovr(truth, scores, 1), brute)


class ClassWise(unittest.TestCase):
    def test_sensitivity_specificity(self):
        matrix = np.zeros((5, 5), int)
        matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[2, 2] = 8, 2, 3, 7, 5
        rows = bpt.class_rows(matrix, [0.5] * 5)
        # Grade 0: TP 8, FN 2, FP 3, TN 12 -> sensitivity 0.8, specificity 0.8, precision 8/11, F1 16/21
        self.assertAlmostEqual(rows[0]["sensitivity"], 0.8)
        self.assertAlmostEqual(rows[0]["specificity"], 12 / 15)
        self.assertAlmostEqual(rows[0]["precision"], 8 / 11)
        self.assertAlmostEqual(rows[0]["f1"], 16 / 21)
        # Grade 3 has no support and no predictions.
        self.assertTrue(math.isnan(rows[3]["sensitivity"]))
        self.assertTrue(math.isnan(rows[3]["precision"]))
        self.assertAlmostEqual(rows[3]["specificity"], 1.0)


class Bootstrap(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(1)
        y = np.repeat(np.arange(5), 20)
        probs = rng.dirichlet(np.ones(5), size=len(y))
        probs[np.arange(len(y)), y] += 1
        probs /= probs.sum(1, keepdims=True)
        frame = pd.DataFrame(probs, columns=bpt.PROB_COLS)
        frame.insert(0, "y_pred", probs.argmax(1))
        frame.insert(0, "y_true", y)
        return frame

    def test_deterministic_and_brackets_point_estimate(self):
        frame = self.frame()
        first, second = bpt.bootstrap(frame, n_boot=300), bpt.bootstrap(frame, n_boot=300)
        self.assertEqual(first, second)
        point = bpt.point_metrics(frame)
        for key, ci in (("accuracy", "accuracy_ci"), ("qwk", "qwk_ci"), ("macro_auc", "macro_auc_ci")):
            self.assertLessEqual(first[ci][0], point[key])
            self.assertGreaterEqual(first[ci][1], point[key])

    def test_absent_class_resamples_are_counted(self):
        frame = self.frame().iloc[:21].copy()  # grade 1 appears once
        result = bpt.bootstrap(frame, n_boot=500)
        self.assertGreater(result["boot_auc_undefined"], 0)


class FinalQueueHelpers(unittest.TestCase):
    def test_scaling_uses_alpha_over_rank(self):
        self.assertEqual(bpt.lora_scaling(16, 16, 1.0), (1.0, ""))
        scaling, note = bpt.lora_scaling(32, 16, 1.0)  # recorded as 16/rank
        self.assertEqual(scaling, 2.0)
        self.assertIn("adapter_config.json", note)
        with self.assertRaises(bpt.BuildError):
            bpt.lora_scaling(32, 16, 3.0)

    def test_timing_drops_warmup_images(self):
        stats = bpt.timing_stats([9, 9, 9, 1, 2, 3])
        self.assertEqual(stats["images_timed"], 3)
        self.assertEqual(stats["median_seconds"], 2)

    def test_fallback_runs_stay_out_of_fp16_mean(self):
        frame = pd.DataFrame({"g": ["a"] * 3, "in_fp16_mean": [True, True, False], "accuracy": [0.2, 0.4, 0.9],
                              "qwk": [0.0] * 3, "macro_auc": [0.5] * 3, "best_val_loss": [0.1] * 3,
                              "degenerate": [False] * 3, "n_grades_predicted": [5] * 3, "n": [361] * 3})
        summary = bpt.study_summary(frame, ["g"]).set_index("scoring_group")
        self.assertAlmostEqual(summary.loc[bpt.GROUP_FP16_ONLY, "accuracy_mean"], 0.3)
        self.assertAlmostEqual(summary.loc[bpt.GROUP_MIXED, "accuracy_mean"], 0.5)
        self.assertEqual(summary.loc[bpt.GROUP_FP16_ONLY, "n_fallback_excluded"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
