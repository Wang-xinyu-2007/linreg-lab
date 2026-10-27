"""第 2 关：损失函数和梯度。

训练模型就是「找一个 w 和 b，让预测值和真实值差得最少」。
"差得最少"要用一个数字衡量，这就是损失函数。线性回归用均方误差 MSE：

    L(w, b) = (1/n) * Σ (ŷ_i - y_i)²

要让 L 变小，就沿负梯度方向走。对 w 和 b 分别求偏导：

    ∂L/∂w = (2/n) * Σ (ŷ_i - y_i) * x_i
    ∂L/∂b = (2/n) * Σ (ŷ_i - y_i)

这一关把公式变成代码。写完之后测试会用「数值梯度」来验证你的解析解——
就是那种用 (L(w+h) - L(w-h)) / 2h 近似导数的笨办法，慢但几乎不会错。
"""

import numpy as np


def mse(y_true, y_pred):
    """均方误差：(1/n) * Σ(ŷ - y)²。返回一个 float。"""
    # TODO: 用 numpy 算，不要写 for 循环
    raise NotImplementedError("请实现 mse()")


def loss_and_gradients(x, y, w, b):
    """一次算出损失和两个梯度。

    返回 (loss, dw, db)，三个都是 float。

    提示
    ----
    1. 先算残差 residual = 预测值 - 真实值（也就是 ŷ - y）
    2. loss 是 residual 的平方的平均
    3. dw、db 按上面的公式，注意别忘了系数 2 和除以 n
    """
    # TODO
    raise NotImplementedError("请实现 loss_and_gradients()")
