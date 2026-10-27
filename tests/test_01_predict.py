"""第 1 关的验收测试。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load  # noqa: E402

stage = load("01_predict")


class TestPredict(unittest.TestCase):
    def test_values(self):
        np.testing.assert_allclose(stage.predict([0, 1, 2], 2, 1), [1, 3, 5])

    def test_accepts_plain_list(self):
        result = stage.predict([1, 2, 3], 0.5, 0)
        self.assertIsInstance(result, np.ndarray)

    def test_shape(self):
        x = np.linspace(0, 1, 17)
        self.assertEqual(stage.predict(x, 1.0, 2.0).shape, (17,))

    def test_negative_numbers(self):
        np.testing.assert_allclose(stage.predict([-1, 0, 1], 3, -2), [-5, -2, 1])


if __name__ == "__main__":
    unittest.main()
