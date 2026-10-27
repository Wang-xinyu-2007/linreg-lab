"""linreg-lab 的小工具：按顺序加载 exercises/ 或 solutions/ 里的关卡。

为什么要有这个文件？
因为测试代码要同时能测「你写的」和「参考答案」。设一个环境变量就能切换：

    python -m unittest discover -s tests              # 测 exercises/（会失败，正常）
    LINREG_SOLUTIONS=1 python -m unittest ...         # 测 solutions/（CI 用这个）
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("01_predict", "预测函数"),
    ("02_loss", "损失与梯度"),
    ("03_normal_equation", "正规方程"),
    ("04_gradient_descent", "梯度下降"),
    ("05_multi_feature", "多特征与评估"),
]


def load(stage: str):
    """加载某一关的模块。

    没用普通的 import，是因为文件名以数字开头（01_predict.py），
    不能写成 import 语句——这类文件只能按路径加载。
    """
    package = "solutions" if os.environ.get("LINREG_SOLUTIONS") else "exercises"
    path = ROOT / package / f"{stage}.py"
    if not path.exists():
        raise FileNotFoundError(f"找不到关卡文件：{path}")

    spec = importlib.util.spec_from_file_location(f"linreg_{package}_{stage}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
