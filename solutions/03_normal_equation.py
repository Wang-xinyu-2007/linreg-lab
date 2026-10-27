"""第 3 关参考答案。"""

import numpy as np


def fit_normal_equation(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    design = np.hstack([x, np.ones_like(x)])      # (n, 2)
    theta = np.linalg.solve(design.T @ design, design.T @ y)
    return float(theta[0]), float(theta[1])
