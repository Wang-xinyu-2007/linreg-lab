#!/usr/bin/env python3
"""一条命令检查你做到第几关了。

    python check.py                # 检查 exercises/ 里你自己写的
    python check.py --solutions    # 检查参考答案（CI 用）

输出大概长这样：

    [1] 预测函数 ................ ✅
    [2] 损失与梯度 .............. ✅
    [3] 正规方程 ................ ❌
    [4] 梯度下降 ................ ⬜ 还没开始
    ...
    完成了 2/5 关
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

STAGES = [
    ("01_predict", "预测函数"),
    ("02_loss", "损失与梯度"),
    ("03_normal_equation", "正规方程"),
    ("04_gradient_descent", "梯度下降"),
    ("05_multi_feature", "多特征与评估"),
]


def load_tests(stage: str):
    path = TESTS / f"test_{stage}.py"
    spec = importlib.util.spec_from_file_location(f"check_{stage}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return unittest.TestLoader().loadTestsFromModule(module)


def looks_unimplemented(stage: str) -> bool:
    """靠源码里有没有 TODO 判断"还没开始"，比报错堆栈友好。"""
    package = "solutions" if os.environ.get("LINREG_SOLUTIONS") else "exercises"
    source = (ROOT / package / f"{stage}.py").read_text(encoding="utf-8")
    return "TODO" in source


def main() -> int:
    if "--solutions" in sys.argv:
        os.environ["LINREG_SOLUTIONS"] = "1"
        print("检查模式：参考答案\n")
    else:
        print("检查模式：exercises/（你自己写的）\n")

    completed = 0
    for index, (stage, title) in enumerate(STAGES, start=1):
        stream = io.StringIO()
        try:
            suite = load_tests(stage)
            result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
            ok = result.wasSuccessful()
            if not ok and looks_unimplemented(stage):
                mark = "⬜ 还没开始"
            else:
                mark = "✅" if ok else "❌"
        except Exception:  # noqa: BLE001  import 阶段就炸了
            ok = False
            mark = "⬜ 还没开始" if looks_unimplemented(stage) else "❌"

        if ok:
            completed += 1

        dots = "." * max(2, 28 - len(title) * 2)
        print(f"[{index}] {title} {dots} {mark}")

        if not ok and "❌" in mark:
            # 只打印第一段失败信息，别刷屏
            lines = [line for line in stream.getvalue().splitlines() if line.strip()]
            for line in lines[-8:]:
                print("      " + line)

    print(f"\n完成了 {completed}/{len(STAGES)} 关")
    if completed == len(STAGES):
        print("全部通过。下一步：python demo.py 看看训练效果。")
    else:
        print("提示：卡住了就看 solutions/ 里对应的文件，但先自己写 20 分钟。")
    return 0 if completed == len(STAGES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
