"""第 2 关的验收测试。

除了数值，还会用「数值梯度」验证你的解析梯度——
这是机器学习里非常重要的一招：写反向传播之前，先用它验一遍。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load  # noqa: E402

stage = load("02_loss")


class TestMse(unittest.TestCase):
    def test_known_value(self):
        # 残差是 [0, 0, -1]，平方平均 = 1/3
        self.assertAlmostEqual(stage.mse([1, 2, 3], [1, 2, 4]), 1 / 3, places=9)

    def test_perfect_prediction_is_zero(self):
        self.assertAlmostEqual(stage.mse([1, 2, 3], [1, 2, 3]), 0.0)

    def test_works_on_arrays(self):
        self.assertAlmostEqual(stage.mse(np.array([0.0]), np.array([2.0])), 4.0)


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0.2, 0.4, 0.9, 1.3])
        self.y = np.array([1.0, 1.4, 2.9, 3.2])
        self.w, self.b = 0.7, 0.3

    def test_returns_three_values(self):
        result = stage.loss_and_gradients(self.x, self.y, self.w, self.b)
        self.assertEqual(len(result), 3)

    def test_loss_matches_mse(self):
        loss, _, _ = stage.loss_and_gradients(self.x, self.y, self.w, self.b)
        self.assertAlmostEqual(loss, stage.mse(self.y, self.w * self.x + self.b), places=9)

    def test_gradient_matches_finite_difference(self):
        """解析梯度必须和数值梯度对得上（误差 < 1e-6）。"""
        _, dw, db = stage.loss_and_gradients(self.x, self.y, self.w, self.b)

        def loss_at(w, b):
            return stage.loss_and_gradients(self.x, self.y, w, b)[0]

        h = 1e-6
        numeric_dw = (loss_at(self.w + h, self.b) - loss_at(self.w - h, self.b)) / (2 * h)
        numeric_db = (loss_at(self.w, self.b + h) - loss_at(self.w, self.b - h)) / (2 * h)

        self.assertAlmostEqual(dw, numeric_dw, places=6)
        self.assertAlmostEqual(db, numeric_db, places=6)


if __name__ == "__main__":
    unittest.main()
