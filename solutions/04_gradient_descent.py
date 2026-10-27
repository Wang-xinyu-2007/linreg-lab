"""第 4 关参考答案。"""

import numpy as np


def fit_gradient_descent(x, y, learning_rate=0.1, epochs=500):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    w, b = 0.0, 0.0
    losses = []

    for _ in range(epochs):
        residual = (w * x + b) - y
        dw = 2.0 * np.mean(residual * x)
        db = 2.0 * np.mean(residual)

        w -= learning_rate * dw
        b -= learning_rate * db

        losses.append(float(np.mean(((w * x + b) - y) ** 2)))

    return w, b, losses
