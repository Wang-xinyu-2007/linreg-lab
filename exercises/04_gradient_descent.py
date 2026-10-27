"""第 4 关：梯度下降（迭代解）。

正规方程一步到位，但有两个毛病：
    1. 要算 XᵀX 的逆/解方程，特征一多就慢（O(d³)）；
    2. 换个模型（比如神经网络）就没有解析解了。

梯度下降不挑模型，套路永远是：

    θ ← θ - 学习率 × 梯度

线性回归的梯度就是第 2 关推出来的那两个式子。这一关把「算梯度」和「走一步」
拼成循环，看看它能不能收敛到和正规方程一样的答案。

两个必须亲眼见一次的现象：
    · 学习率太小 → 收敛得慢，几百轮还没到；
    · 学习率太大 → loss 越走越大，直接发散（甚至变成 inf）。
"""

import numpy as np


def fit_gradient_descent(x, y, learning_rate=0.1, epochs=500):
    """从 w=0, b=0 出发，迭代 epochs 轮。

    返回
    ----
    (w, b, losses)
        w, b    : 最终的参数（float）
        losses  : 长度 epochs 的列表，第 i 个元素是第 i 轮更新后的损失
                  （demo.py 会用这个列表画 ASCII 曲线）

    提示
    ----
    每一轮：
        1. 算残差 residual = (w*x + b) - y
        2. dw = mean(2 * residual * x)，db = mean(2 * residual)
        3. w -= learning_rate * dw，b -= learning_rate * db
        4. 记录这一轮之后的 loss = mean(residual²)（用更新后的参数重算）
    """
    # TODO
    raise NotImplementedError("请实现 fit_gradient_descent()")
