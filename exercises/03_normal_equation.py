"""第 3 关：正规方程（解析解）。

不用一步步迭代，直接一步算出最优的 w 和 b。把模型写成矩阵形式：

    X = [[x_1, 1],
         [x_2, 1],
         ...
         [x_n, 1]]          # 最后一列全是 1，用来吸收偏置 b
    ŷ = X @ θ                 # θ = [w, b]

最小化 ‖Xθ - y‖²，令导数为 0，得到：

    θ = (XᵀX)⁻¹ Xᵀy

**别真的去求逆**。求逆又慢又不稳定，用 np.linalg.solve 解线性方程组：
solve(XᵀX, Xᵀy)，结果一样，数值上稳得多。
（更稳的是 np.linalg.lstsq，它连奇异矩阵都能处理。）

这一关你会用到线性代数课上的东西：矩阵乘法、转置、解方程组。
"""

import numpy as np


def fit_normal_equation(x, y):
    """用正规方程一次求出最优的 (w, b)。

    参数
    ----
    x : 形状 (n,) 的特征
    y : 形状 (n,) 的目标值

    返回
    ----
    (w, b)，两个 float

    提示
    ----
    1. 把 x 变成 (n, 1) 的形状：x = np.asarray(x).reshape(-1, 1)
    2. 用 np.hstack 拼上全是 1 的那一列，得到设计矩阵 X（形状 (n, 2)）
    3. theta = np.linalg.solve(X.T @ X, X.T @ y)
    4. theta[0] 是 w，theta[1] 是 b
    """
    # TODO
    raise NotImplementedError("请实现 fit_normal_equation()")
