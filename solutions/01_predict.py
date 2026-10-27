"""第 1 关参考答案。"""

import numpy as np


def predict(x, w, b):
    x = np.asarray(x, dtype=float)
    return w * x + b
