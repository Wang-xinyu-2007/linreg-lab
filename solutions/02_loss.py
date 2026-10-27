"""第 2 关参考答案。"""

import numpy as np


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def loss_and_gradients(x, y, w, b):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    residual = (w * x + b) - y
    loss = float(np.mean(residual ** 2))
    dw = float(np.mean(2.0 * residual * x))
    db = float(np.mean(2.0 * residual))
    return loss, dw, db
