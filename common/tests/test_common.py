"""P05-T02a: deterministic scientific tests using SYNTHETIC DATA ONLY."""

from __future__ import annotations

import hashlib
import os
import platform
import time
import unittest
from pathlib import Path

# CPU-only and maximum eight threads (read before NumPy/scipy imports).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from common.bootstrap import cluster_bootstrap
from common.conformal import crc_fnr_threshold, ltt_cp_threshold
from common.metrics import (
    auroc, auprc, challenge_score, clopper_pearson, ece,
    fa_suppression, fnr,
)
from common.report import mask_small, print_report


class TestCommon(unittest.TestCase):
    """Synthetic-only validation, fixed seed=42."""

    started = time.perf_counter()
    mc_mean_new_fnr = None
    bootstrap_coverage = None
    _unittest_result = None

    def run(self, result=None):
        if result is not None:
            self.__class__._unittest_result = result
        return super().run(result)

    @classmethod
    def tearDownClass(cls):
        script_path = Path(__file__).resolve()
        sha8 = hashlib.sha256(script_path.read_bytes()).hexdigest()[:8]
        test_result = cls._unittest_result
        passed = (
            test_result is not None
            and not test_result.failures
            and not test_result.errors
            and cls.mc_mean_new_fnr is not None
            and cls.bootstrap_coverage is not None
        )
        print_report("P05-T02a", {
            "status": "PASS" if passed else "FAIL",
            "pc": "reference-workstation" if os.name == "nt" else "LOCAL VALIDATION",
            "gpu": "not used",
            "python": platform.python_version(),
            "cpu": "n_jobs<=8",
            "dataset": "synthetic only; data_raw/data_proc NOT accessed",
            "seed": 42,
            "unittest_count": test_result.testsRun if test_result else 0,
            "crc_mc_repetitions": 2000,
            "crc_mc_n_records": 300,
            "crc_mc_alpha": 0.05,
            "crc_mc_mean_new_record_fnr": (
                f"{cls.mc_mean_new_fnr:.6f}"
                if cls.mc_mean_new_fnr is not None else "NOT COMPLETED"
            ),
            "crc_mc_limit": 0.055,
            "bootstrap_coverage_trials": 200,
            "bootstrap_coverage": (
                f"{cls.bootstrap_coverage:.4f}"
                if cls.bootstrap_coverage is not None else "NOT COMPLETED"
            ),
            "bootstrap_coverage_limit": 0.90,
            "test_script_sha256_8": sha8,
            "runtime_s": f"{time.perf_counter() - cls.started:.2f}",
            "outputs": "unittest stdout/stderr log",
            "G0": "CLOSED by Director; alpha primary=0.02",
            "T02b": "NOT STARTED; wait for Director T02a acceptance",
        })

    def test_auc_ece_and_challenge(self):
        p = np.array([0.02, 0.40, 0.70, 0.90, 0.20, 0.75])
        y = np.array([0, 0, 1, 1, 1, 0])
        self.assertAlmostEqual(auroc(p, y), roc_auc_score(y, p), places=12)
        self.assertAlmostEqual(auprc(p, y), average_precision_score(y, p), places=12)
        y_bin = np.array([0, 1, 1, 0, 0, 1], dtype=int)
        self.assertAlmostEqual(ece(y_bin.astype(float), y_bin), 0.0, places=15)
        self.assertAlmostEqual(challenge_score(2, 3, 1, 1), 100 * 5 / 11)
        self.assertAlmostEqual(fa_suppression(p, y, 0.50), 2 / 3)
        self.assertAlmostEqual(fnr(p, y, 0.50), 1 / 3)

    def test_auc_and_count_rejections(self):
        with self.assertRaises(ValueError):
            auroc([0.1, 0.2], [1, 1])
        with self.assertRaises(ValueError):
            ece([0.2, 1.2], [0, 1])
        with self.assertRaises(ValueError):
            challenge_score(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            clopper_pearson(2, 1)

    def test_clopper_pearson_exact_two_sided(self):
        low, high = clopper_pearson(0, 100, conf=0.95)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 1 - 0.025 ** (1 / 100), places=12)
        self.assertAlmostEqual(high, 0.0362, delta=0.0002)
        low_n, high_n = clopper_pearson(100, 100, conf=0.95)
        self.assertAlmostEqual(low_n, 1 - high, places=12)
        self.assertEqual(high_n, 1.0)

    def test_crc_hand_case_unweighted(self):
        # Two records, both positive. At lambda=.2: loss=0, bound=1/3.
        # At lambda=.8: record 0 lost, bound=2/3. Strict inequality matters.
        scores = [0.2, 0.8]
        y = [1, 1]
        groups = ["R0", "R1"]
        self.assertEqual(crc_fnr_threshold(scores, y, groups, alpha=0.34), 0.2)
        self.assertEqual(crc_fnr_threshold(scores, y, groups, alpha=0.70), 0.8)

    def test_crc_hand_case_weighted(self):
        # Record 0 has positive event weights 1 and 3; record 1 has weight 1.
        # At lambda=.8: L0=1/4, L1=0 => CRC bound=(1.25)/3.
        scores = [0.1, 0.8, 0.9]
        y = [1, 1, 1]
        groups = [0, 0, 1]
        weights = [1.0, 3.0, 1.0]
        self.assertEqual(
            crc_fnr_threshold(scores, y, groups, alpha=0.42, weights=weights),
            0.8,
        )
        self.assertEqual(
            crc_fnr_threshold([-1.0, 0.0], [1, 1], [0, 1], 0.34),
            0.0,
        )

    def test_crc_insufficient_records(self):
        with self.assertRaises(ValueError):
            crc_fnr_threshold([0.1, 0.2], [1, 1], [0, 1], alpha=0.10)
        with self.assertRaises(ValueError):
            crc_fnr_threshold([0.1, 0.2], [0, 0], [0, 1], alpha=0.50)

    @staticmethod
    def _synthetic_record_batch(rng, n_records):
        sizes = rng.integers(1, 4, size=n_records)
        groups = np.repeat(np.arange(n_records), sizes)
        scores = rng.uniform(0.0, 1.0, size=groups.size)
        y = (rng.random(groups.size) < 0.45).astype(np.int8)
        # Each record has at least one positive event by construction.
        starts = np.cumsum(np.concatenate(([0], sizes[:-1])))
        y[starts] = 1
        return scores, y, groups

    def test_crc_monte_carlo_generalization(self):
        rng = np.random.default_rng(42)
        n_repetitions = 2000
        n_records = 300
        alpha = 0.05
        new_record_fnr = np.empty(n_repetitions, dtype=np.float64)
        for repetition in range(n_repetitions):
            cal_scores, cal_y, cal_groups = self._synthetic_record_batch(
                rng, n_records
            )
            lam = crc_fnr_threshold(
                cal_scores, cal_y, cal_groups, alpha=alpha
            )
            test_scores, test_y, test_groups = self._synthetic_record_batch(
                rng, n_records
            )
            positives_per_record = np.bincount(
                test_groups, weights=test_y, minlength=n_records
            )
            missed_per_record = np.bincount(
                test_groups, weights=test_y * (test_scores < lam),
                minlength=n_records,
            )
            new_record_fnr[repetition] = float(
                np.mean(missed_per_record / positives_per_record)
            )
        self.__class__.mc_mean_new_fnr = float(new_record_fnr.mean())
        print(
            "SCIENTIFIC_CHECK_CRC_MC: "
            f"repetitions={n_repetitions}; n={n_records}; alpha={alpha:.2f}; "
            f"mean_new_record_FNR={self.mc_mean_new_fnr:.6f}; "
            "limit=0.055000",
            flush=True,
        )
        self.assertLessEqual(self.mc_mean_new_fnr, 0.055)

    def test_bootstrap_paired_shared_record_indices(self):
        record = np.repeat(np.arange(15), 2)
        base = np.linspace(0.1, 0.9, record.size)
        df = pd.DataFrame({
            "record": record,
            "score_a": base + 0.20,
            "score_b": base,
        })

        def paired_stat(sample):
            return (sample["score_a"] - sample["score_b"]).mean()

        serial = cluster_bootstrap(
            paired_stat, df, "record", B=50, seed=42, n_jobs=1
        )
        threaded = cluster_bootstrap(
            paired_stat, df, "record", B=50, seed=42, n_jobs=2
        )
        np.testing.assert_allclose(serial, threaded, atol=1e-15, rtol=0)
        np.testing.assert_allclose(serial, [0.2, 0.2, 0.2], atol=1e-15)

    def test_bootstrap_duplicate_record_draws_remain_distinct(self):
        df = pd.DataFrame({
            "record": ["A", "B", "B", "C", "C", "C"],
            "value": [0.0, 1.0, 1.0, 10.0, 10.0, 10.0],
        })

        def mean_over_records(sample):
            return float(sample.groupby("record")["value"].mean().mean())

        B = 100
        observed = cluster_bootstrap(
            mean_over_records, df, "record", B=B, seed=42, n_jobs=1
        )
        draw = np.random.default_rng(42).integers(0, 3, size=(B, 3))
        expected_stats = np.array([np.array([0.0, 1.0, 10.0])[r].mean()
                                   for r in draw])
        expected = (
            expected_stats.mean(),
            *np.percentile(expected_stats, [2.5, 97.5]),
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12)

    def test_bootstrap_ci_coverage(self):
        rng = np.random.default_rng(42)
        trials = 200
        covered = 0
        for i in range(trials):
            # The data-generating mean is exactly zero.
            values = rng.standard_normal(80)
            df = pd.DataFrame({"record": np.arange(80), "value": values})
            _, low, high = cluster_bootstrap(
                lambda sample: float(sample["value"].mean()),
                df, "record", B=250, seed=42 + i, n_jobs=1
            )
            covered += (low <= 0.0 <= high)
        self.__class__.bootstrap_coverage = covered / trials
        print(
            "SCIENTIFIC_CHECK_BOOTSTRAP: "
            f"trials={trials}; coverage={self.bootstrap_coverage:.4f}; "
            "limit=0.9000",
            flush=True,
        )
        self.assertGreaterEqual(self.bootstrap_coverage, 0.90)

    def test_ltt_one_sided_bonferroni_hand_case(self):
        # tau=.2: n=100, k=0, one-sided CP upper is
        # 1 - (delta / 2)**(1/100) ~= 0.0362.
        # tau=.95: n=105, k=5, CP upper > 0.05.
        scores = np.concatenate((np.full(100, 0.1), np.full(5, 0.9)))
        y = np.concatenate((np.zeros(100), np.ones(5)))
        delta = 0.05
        upper_at_first = 1 - (delta / 2) ** (1 / 100)
        self.assertLessEqual(upper_at_first, 0.05)
        self.assertEqual(
            ltt_cp_threshold(scores, y, alpha=0.05, delta=delta,
                             grid=[0.2, 0.95]),
            0.2,
        )
        self.assertIsNone(
            ltt_cp_threshold([0.1, 0.2], [1, 1], alpha=0.05,
                             grid=[0.1, 0.2])
        )

    def test_report_mask(self):
        self.assertEqual(mask_small(0), "0")
        self.assertEqual(mask_small(1), "<10")
        self.assertEqual(mask_small(9), "<10")
        self.assertEqual(mask_small(10), "10")
        self.assertEqual(mask_small(np.int64(2)), "<10")


if __name__ == "__main__":
    unittest.main()
