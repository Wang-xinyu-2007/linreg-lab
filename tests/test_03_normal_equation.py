"""第 3 关的验收测试：能不能一次算对最优的 w 和 b。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load  # noqa: E402

stage = load("03_normal_equation")


class TestNormalEquation(unittest.TestCase):
    def test_exact_recovery(self):
        """没有噪声的数据，应该精确恢复出 w=3, b=2。"""
        x = np.arange(10, dtype=float)
        y = 3 * x + 2
        w, b = stage.fit_normal_equation(x, y)
        self.assertAlmostEqual(w, 3.0, places=8)
        self.assertAlmostEqual(b, 2.0, places=8)

    def test_matches_polyfit_with_noise(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 200)
        y = 2.5 * x + 1.3 + rng.normal(0, 0.8, 200)

        w, b = stage.fit_normal_equation(x, y)
        expected_w, expected_b = np.polyfit(x, y, 1)

        self.assertAlmostEqual(w, expected_w, places=6)
        self.assertAlmostEqual(b, expected_b, places=6)

    def test_returns_floats(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 3.0, 5.0])
        w, b = stage.fit_normal_equation(x, y)
        self.assertIsInstance(w, float)
        self.assertIsInstance(b, float)


if __name__ == "__main__":
    unittest.main()
