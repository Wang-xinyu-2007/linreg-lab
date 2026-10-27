"""第 1 关：预测函数。

模型长这样：

    ŷ = w * x + b

其中 w 是权重（斜率），b 是偏置（截距）。听起来简单到不值得写代码，
但这一关要练的是**向量化**：对 100 万个样本做预测，不能写 for 循环。
"""

import numpy as np


def predict(x, w, b):
    """返回预测值 ŷ = w * x + b。

    参数
    ----
    x : 形状 (n,) 的数组（也可能是 Python 列表）
    w : 一个数字
    b : 一个数字

    返回
    ----
    形状 (n,) 的 numpy 数组

    提示
    ----
    numpy 的数组支持「整体运算」：`w * x + b` 会一次算完所有元素。
    另外记得先用 np.asarray(x, dtype=float) 把列表变成数组，
    这样传进来的是 list 也不会报错。
    """
    # TODO: 一行搞定
    raise NotImplementedError("请实现 predict()")
