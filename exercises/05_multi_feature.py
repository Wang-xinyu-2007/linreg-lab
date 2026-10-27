"""第 5 关：多特征 + 标准化 + 评估。

前面都是一元回归（一个 x）。现实中你有一堆特征：面积、房龄、楼层、到地铁距离……
模型变成：

    ŷ = w₁x₁ + w₂x₂ + ... + w_d x_d + b = X @ w + b

**标准化**是这一关的重点。想想看：面积是 90 左右，到地铁距离是 0.5 左右，
房龄是 15 左右。梯度下降对每个特征用同一个学习率，尺度差 100 倍，
大的那个方向会震荡、小的那个方向几乎不动。所以先把每一列变成「均值 0、标准差 1」：

    z = (x - mean) / std

标准化之后训练出来的 w 是**标准化空间**里的，要换回原始尺度才好解释：

    w_原始 = w_标准化 / std
    b_原始 = b - Σ(w_标准化 * mean / std)

最后用一个数字评价模型好不好——**R²（决定系数）**：

    R² = 1 - SS_res / SS_tot

等于 1 表示完美预测；等于 0 表示「和直接猜平均值一样好」；可以是负数（比猜平均值还差）。
"""

import numpy as np


def standardize(X):
    """按列标准化。

    参数
    ----
    X : 形状 (n, d) 的数组

    返回
    ----
    (Z, mean, std)
        Z    : 标准化后的数组，形状 (n, d)
        mean : 每列的均值，形状 (d,)
        std  : 每列的标准差，形状 (d,)

    提示
    ----
    用 axis=0 沿「行」的方向求统计量，得到长度 d 的结果。
    某一列如果全是一样的值，std 会是 0，除它会得到 inf/nan——记得把它改成 1。
    """
    # TODO
    raise NotImplementedError("请实现 standardize()")


def fit_multi(X, y, learning_rate=0.1, epochs=2000):
    """多特征线性回归的梯度下降。

    返回 (w, b)：
        w : 形状 (d,) 的数组，**原始尺度**下的权重
        b : float

    提示
    ----
    1. Z, mean, std = standardize(X)
    2. 在标准化空间里做梯度下降：
           dw = (2/n) * Z.T @ residual
           db = (2/n) * residual.sum()
    3. 训练完把 w 换回原始尺度（公式见文件开头）
    """
    # TODO
    raise NotImplementedError("请实现 fit_multi()")


def r2_score(y_true, y_pred):
    """决定系数 R²。

    提示：SS_res = Σ(y - ŷ)²，SS_tot = Σ(y - ȳ)²。
    如果 SS_tot 是 0（所有真实值都一样），返回 1.0（当预测也完全正确时）。
    """
    # TODO
    raise NotImplementedError("请实现 r2_score()")
