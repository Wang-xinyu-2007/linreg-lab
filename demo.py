#!/usr/bin/env python3
"""跑一遍看效果：用同一份数据，比较「正规方程」和「梯度下降」的结果。

    python demo.py              # 用 exercises/ 里你写的代码
    python demo.py --solutions  # 用参考答案（还没做完时可以用这个看效果）

依赖 numpy，不需要 matplotlib——loss 曲线是用字符画的。
"""

from __future__ import annotations

import os
import sys

import numpy as np

import linreg_lab

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 60) -> str:
    """把一长串数字压成一行字符，用来看趋势（比一堆数字直观）。"""
    if not values:
        return ""
    step = max(1, len(values) // width)
    sampled = values[::step][:width]

    low, high = min(sampled), max(sampled)
    if high - low < 1e-12:
        return BLOCKS[0] * len(sampled)
    return "".join(
        BLOCKS[int((value - low) / (high - low) * (len(BLOCKS) - 1))]
        for value in sampled
    )


def main() -> int:
    if "--solutions" in sys.argv:
        os.environ["LINREG_SOLUTIONS"] = "1"

    try:
        normal = linreg_lab.load("03_normal_equation")
        descent = linreg_lab.load("04_gradient_descent")
        multi = linreg_lab.load("05_multi_feature")
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    # ---------------- 一元回归 ----------------
    rng = np.random.default_rng(2026)
    x = rng.uniform(0, 1, 200)
    true_w, true_b = 2.5, 1.3
    y = true_w * x + true_b + rng.normal(0, 0.08, 200)

    w1, b1 = normal.fit_normal_equation(x, y)
    w2, b2, losses = descent.fit_gradient_descent(x, y, learning_rate=0.1, epochs=1500)

    loss1 = float(np.mean((w1 * x + b1 - y) ** 2))
    loss2 = float(np.mean((w2 * x + b2 - y) ** 2))

    print("=" * 56)
    print("一元线性回归：200 个样本，真实参数 w=2.5, b=1.3")
    print("=" * 56)
    print(f"{'':12}{'w':>8}{'b':>9}{'loss':>10}")
    print(f"{'正规方程':12}{w1:>8.4f}{b1:>9.4f}{loss1:>10.5f}")
    print(f"{'梯度下降':12}{w2:>8.4f}{b2:>9.4f}{loss2:>10.5f}")
    print(f"{'两种方法差':12}{abs(w1 - w2):>8.6f}{abs(b1 - b2):>9.6f}")
    print()
    print("梯度下降的 loss 曲线（前 60 轮）：")
    print(sparkline(losses[:60]))
    print("（线性回归很简单，10 轮左右就基本到位了——换成神经网络就不会这么干脆）")
    print()

    # ---------------- 多特征 ----------------
    n = 400
    area = rng.uniform(40, 140, n)
    age = rng.uniform(1, 30, n)
    metro = rng.uniform(0.1, 5.0, n)
    X = np.column_stack([area, age, metro])
    true_weights = np.array([0.08, -0.15, -0.90])
    y2 = X @ true_weights + 12.0 + rng.normal(0, 0.3, n)

    weights, bias = multi.fit_multi(X, y2, learning_rate=0.1, epochs=4000)
    predictions = X @ weights + bias

    print("=" * 56)
    print("多特征：面积 / 房龄 / 距地铁，真实权重 [0.08, -0.15, -0.90]")
    print("=" * 56)
    print("学到的权重：", np.round(weights, 4))
    print(f"学到的偏置：{bias:.4f}（真实 12.0）")
    print(f"R² = {multi.r2_score(y2, predictions):.4f}")
    print()
    print("解读：面积每多 1 平方米，总价多约 "
          f"{weights[0]:.2f} 万；房龄每老 1 年，少约 {abs(weights[1]):.2f} 万。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotImplementedError as exc:
        # 关卡还没写完：别甩一堆 traceback 吓人
        print(f"\n还有关卡没做完：{exc}", file=sys.stderr)
        print("先跑 python check.py 看看进度，"
              "或者用 python demo.py --solutions 直接看参考答案的效果。", file=sys.stderr)
        raise SystemExit(1)
