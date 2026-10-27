"""测试用的小公共模块：把项目根目录加进 sys.path，然后复用 linreg_lab.load。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linreg_lab import STAGES, load  # noqa: E402,F401
