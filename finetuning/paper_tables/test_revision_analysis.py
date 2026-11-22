"""Unit tests for build_revision_analysis.py on hand-computed cases.

Run from the repo root:
  python -I -B finetuning/paper_tables/test_revision_analysis.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_paper_tables as bpt  # noqa: E402
import build_revision_analysis as bra  # noqa: E402


def frame(y_true, y_pred, probs=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if probs is None:
        probs = np.full((len(y_true), 5), 0.05)
        probs[np.arange(len(y_true)), y_pred] = 0.8
    f = pd.DataFrame({"image_id": [f"i{k}" for k in range(len(y_true))], "y_true": y_true, "y_pred": y_pred})
    for g in range(5):
        f[f"p{g}"] = probs[:, g]
    return f


class Calibration(unittest.TestCase):
    def test_ece_perfectly_calibrated_is_zero(self):
        # Ten images at confidence 0.8, eight correct: accuracy equals confidence in the one occupied bin.
        y = np.array([0] * 8 + [1, 1])
        probs = np.tile([0.8, 0.05, 0.05, 0.05, 0.05], (10, 1))
        self.assertAlmostEqual(bra.ece(probs, y), 0.0)

    def test_ece_overconfident(self):
        y = np.array([1] * 10)
        probs = np.tile([0.9, 0.025, 0.025, 0.025, 0.025], (10, 1))
        self.assertAlmostEqual(bra.ece(probs, y), 0.9)


class PairedTests(unittest.TestCase):
    def test_mcnemar_exact(self):
        a = np.array([True] * 10 + [False] * 2 + [True] * 5)
        b = np.array([False] * 10 + [True] * 2 + [True] * 5)
        only_a, only_b, p = bra.mcnemar_exact(a, b)
        self.assertEqual((only_a, only_b), (10, 2))
        # Two-sided exact binomial, 2 of 12 at 0.5: 2 * (1 + 12 + 66) / 4096
        self.assertAlmostEqual(p, 2 * 79 / 4096)

    def test_mcnemar_no_discordant(self):
        same = np.array([True, False, True])
        self.assertEqual(bra.mcnemar_exact(same, same), (0, 0, 1.0))

    def test_holm(self):
        self.assertEqual(bra.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_boot_p(self):
        self.assertAlmostEqual(bra.boot_p(np.array([1.0, 2.0, 3.0, -1.0])), 0.5)


class Resampling(unittest.TestCase):
    def test_unit_counts_reproduce_point_metrics(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.arange(5).repeat(6)]
        p = np.clip(y + rng.integers(-1, 2, len(y)), 0, 4)
        probs = rng.dirichlet(np.ones(5), len(y))
        probs[np.arange(len(y)), p] += 2
        probs /= probs.sum(1, keepdims=True)
        f = frame(y, probs.argmax(1), probs)
        boots = bra.metrics_from_counts(f, np.ones((1, len(y))))
        point = bpt.point_metrics(f)
        for k in ("accuracy", "qwk", "macro_auc"):
            self.assertAlmostEqual(float(boots[k][0]), point[k], places=12)

    def test_cluster_resampling_keeps_patients_together(self):
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        counts = bra.resample_counts(6, groups, 200, 1)
        self.assertTrue(np.array_equal(counts[:, 0], counts[:, 1]))
        self.assertTrue(np.array_equal(counts[:, 4], counts[:, 5]))
        self.assertTrue(np.all(counts.sum(1) == 6))


class RunMetrics(unittest.TestCase):
    def test_error_distances_and_referable(self):
        #        true: 0 0 1 2 3 4
        #        pred: 0 1 0 2 1 4   -> errors off by 1, 1 and 2
        f = frame([0, 0, 1, 2, 3, 4], [0, 1, 0, 2, 1, 4])
        m = bra.full_metrics(f)
        self.assertAlmostEqual(m["exact_rate"], 3 / 6)
        self.assertAlmostEqual(m["adjacent_error_rate"], 2 / 6)
        self.assertAlmostEqual(m["distant_error_rate"], 1 / 6)
        self.assertAlmostEqual(m["adjacent_share_of_errors"], 2 / 3)
        self.assertEqual((m["mild_n"], m["mild_predicted_normal"]), (1, 1))
        # Referable (grade >= 2): true 2,3,4 ; predicted referable 2,4 -> sensitivity 2/3, specificity 3/3
        self.assertAlmostEqual(m["referable_sensitivity"], 2 / 3)
        self.assertAlmostEqual(m["referable_specificity"], 1.0)
        self.assertEqual(m["grades_never_predicted"], "3")
        self.assertFalse(m["collapsed"])

    def test_collapsed_flag(self):
        m = bra.full_metrics(frame([0, 1, 2, 3, 4], [0, 0, 0, 1, 1]))
        self.assertTrue(m["collapsed"])
        self.assertEqual(m["n_grades_predicted"], 2)

    def test_balanced_accuracy_and_macro_f1(self):
        m = bra.full_metrics(frame([0, 0, 1, 1, 2, 3, 4], [0, 0, 1, 0, 2, 3, 4]))
        self.assertAlmostEqual(m["balanced_accuracy"], (1 + 0.5 + 1 + 1 + 1) / 5)
        f1_0 = 2 * 2 / (2 * 2 + 1 + 0)
        f1_1 = 2 * 1 / (2 * 1 + 0 + 1)
        self.assertAlmostEqual(m["macro_f1"], (f1_0 + f1_1 + 1 + 1 + 1) / 5)


class VariantNames(unittest.TestCase):
    def test_variant_names_from_final_runs(self):
        self.assertEqual(bra.variant("r16_on_lr0p0002_t42_a3407")["condition"], "Qwen r16 CLAHE-on lr2e-4")
        self.assertEqual(bra.variant("r16_on_alpha8_t43_a3408")["condition"], "Qwen r16 CLAHE-on lr5e-4 alpha8")
        v = bra.variant("r32_on_alpha32_lr0p0002_t44_a3409")
        self.assertEqual((v["condition"], v["seed"], v["rank"], v["alpha"]), ("Qwen r32 CLAHE-on lr2e-4 alpha32", 44, 32, 32))
        self.assertEqual(bra.variant("r8_off_lr0p0002_t42_a3407")["condition"], "Qwen r8 CLAHE-off lr2e-4")
        self.assertIsNone(bra.variant("full_r16_on_duplicate_excluded_t42_a3407_01"))

    def test_condition_names_match_the_headline_constants(self):
        self.assertEqual(bra.variant("r16_on_lr0p0001_t42_a3407")["condition"], bra.Q_ON_1E4)
        self.assertEqual(bra.variant("r16_on_lr0p0002_t42_a3407")["condition"], bra.Q_ON_2E4)


class IdNormalisation(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(bra.normalise_id("messidor_messidor_p_0_20051109_57451_0400_PP"), "20051109_57451_0400_PP")
        self.assertEqual(bra.normalise_id("aptos_aptos_0104b032c141_0104b032c141"), "0104b032c141")
        self.assertEqual(bra.normalise_id("0104b032c141"), "0104b032c141")


if __name__ == "__main__":
    unittest.main(verbosity=2)
