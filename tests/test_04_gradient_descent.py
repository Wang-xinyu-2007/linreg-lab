"""第 4 关的验收测试：梯度下降要能收敛，而且 loss 得一路往下走。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load  # noqa: E402

stage = load("04_gradient_descent")


def make_data(n=200, seed=1):
    """x 落在 [0, 1] 里，这样默认学习率不会炸。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.05, n)
    return x, y


class TestGradientDescent(unittest.TestCase):
    def test_converges(self):
        x, y = make_data()
        w, b, losses = stage.fit_gradient_descent(x, y, learning_rate=0.5, epochs=2000)
        self.assertAlmostEqual(w, 2.0, delta=0.05)
        self.assertAlmostEqual(b, 1.0, delta=0.05)

    def test_losses_length(self):
        x, y = make_data(50)
        _, _, losses = stage.fit_gradient_descent(x, y, epochs=100)
        self.assertEqual(len(losses), 100)

    def test_loss_goes_down(self):
        x, y = make_data()
        _, _, losses = stage.fit_gradient_descent(x, y, learning_rate=0.5, epochs=500)
        self.assertLess(losses[-1], losses[0] / 10)
        # 前 10 轮应该是单调下降的
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(losses[:10], losses[1:10])))

    def test_returns_three_values(self):
        x, y = make_data(20)
        result = stage.fit_gradient_descent(x, y, epochs=5)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
