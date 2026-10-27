"""第 5 关的验收测试：标准化、多特征收敛、R²。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load  # noqa: E402

stage = load("05_multi_feature")


class TestStandardize(unittest.TestCase):
    def test_mean_and_std(self):
        rng = np.random.default_rng(3)
        X = rng.normal(10, 5, size=(300, 3))
        Z, mean, std = stage.standardize(X)

        np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-10)
        np.testing.assert_allclose(mean, X.mean(axis=0))

    def test_constant_column_does_not_blow_up(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        Z, _, std = stage.standardize(X)
        self.assertFalse(np.isnan(Z).any())
        self.assertEqual(std[1], 1.0)

    def test_shape(self):
        Z, _, _ = stage.standardize(np.arange(12).reshape(4, 3).astype(float))
        self.assertEqual(Z.shape, (4, 3))


class TestFitMulti(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 400
        # 三个特征量纲差很大：面积、房龄、到地铁的公里数
        self.X = np.column_stack([
            rng.uniform(40, 140, n),      # 面积
            rng.uniform(1, 30, n),        # 房龄
            rng.uniform(0.1, 5.0, n),     # 距地铁
        ])
        self.true_w = np.array([0.08, -0.15, -0.9])
        self.true_b = 12.0
        self.y = self.X @ self.true_w + self.true_b

    def test_recovers_coefficients(self):
        w, b = stage.fit_multi(self.X, self.y, learning_rate=0.1, epochs=4000)
        np.testing.assert_allclose(w, self.true_w, atol=0.02)
        self.assertAlmostEqual(b, self.true_b, delta=0.2)

    def test_shapes(self):
        w, b = stage.fit_multi(self.X, self.y, epochs=10)
        self.assertEqual(w.shape, (3,))
        self.assertIsInstance(b, float)


class TestR2(unittest.TestCase):
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(stage.r2_score(y, y), 1.0)

    def test_mean_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 6.0])
        self.assertAlmostEqual(stage.r2_score(y, np.full(3, y.mean())), 0.0)

    def test_worse_than_mean_is_negative(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertLess(stage.r2_score(y, np.array([3.0, 2.0, 1.0])), 0)

    def test_constant_target(self):
        y = np.full(4, 5.0)
        self.assertEqual(stage.r2_score(y, y), 1.0)


if __name__ == "__main__":
    unittest.main()
