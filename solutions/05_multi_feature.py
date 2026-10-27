"""第 5 关参考答案。"""

import numpy as np


def standardize(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)      # 常数列不能除 0
    return (X - mean) / std, mean, std


def fit_multi(X, y, learning_rate=0.1, epochs=2000):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    Z, mean, std = standardize(X)
    n, d = Z.shape

    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        residual = Z @ w + b - y
        w -= learning_rate * (2.0 / n) * (Z.T @ residual)
        b -= learning_rate * (2.0 / n) * residual.sum()

    # 换回原始尺度：标准化的 z = (x - mean) / std
    w_original = w / std
    b_original = b - float(np.sum(w * mean / std))
    return w_original, b_original


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot
