# linreg-lab

**从零手写线性回归**：五关，每关一个函数，写完用测试验收。

不是为了"再学一遍线性回归"，而是为了**亲手把公式变成代码**——
以后你用 `sklearn.linear_model.LinearRegression().fit(X, y)` 的时候，知道它那一行里发生了什么。

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/deps-numpy%20only-brightgreen)
![Tests](https://github.com/Wang-xinyu-2007/linreg-lab/actions/workflows/tests.yml/badge.svg)

---

## 怎么玩

```bash
git clone https://github.com/Wang-xinyu-2007/linreg-lab.git
cd linreg-lab
python -m pip install numpy

python check.py          # 看看现在做到哪一关
```

然后打开 `exercises/01_predict.py`，把 `TODO` 换成你的代码，再跑一次 `check.py`：

```text
[1] 预测函数 .................... ✅
[2] 损失与梯度 .................. ⬜ 还没开始
[3] 正规方程 .................... ⬜ 还没开始
[4] 梯度下降 .................... ⬜ 还没开始
[5] 多特征与评估 ................ ⬜ 还没开始

完成了 1/5 关
```

全部通过之后：

```bash
python demo.py           # 用你写的代码训练，看效果
```

卡住了？`solutions/` 里有参考答案。**建议先自己写 20 分钟再看**，看完再合上文件重写一遍。

---

## 五关分别练什么

| 关 | 文件 | 你要实现 | 顺手学到的 |
| --- | --- | --- | --- |
| 1 | `01_predict.py` | `predict(x, w, b)` | numpy 向量化（别写 for 循环） |
| 2 | `02_loss.py` | `mse`、`loss_and_gradients` | 损失函数、偏导、**数值梯度校验** |
| 3 | `03_normal_equation.py` | `fit_normal_equation` | 矩阵形式、`np.linalg.solve`、为什么别求逆 |
| 4 | `04_gradient_descent.py` | `fit_gradient_descent` | 迭代优化、学习率、收敛 |
| 5 | `05_multi_feature.py` | `standardize`、`fit_multi`、`r2_score` | 多特征、标准化、系数还原、R² |

**每关 30~60 分钟，总共 4~6 小时。**

---

## 数学就这几行

**模型**（一元）：

$$\hat{y} = wx + b$$

**损失**（均方误差）：

$$L(w,b) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

**梯度**（对 $w$ 和 $b$ 分别求偏导，第 2 关要用）：

$$\frac{\partial L}{\partial w} = \frac{2}{n}\sum_i (\hat{y}_i - y_i)x_i \qquad
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_i (\hat{y}_i - y_i)$$

**正规方程**（令梯度为 0，直接解出最优解，第 3 关要用）：

$$\theta = (X^\top X)^{-1} X^\top y$$

其中 $X$ 是每行 `[x_i, 1]` 的设计矩阵，$\theta = [w, b]$。
**别真的求逆**——用 `np.linalg.solve(X.T @ X, X.T @ y)`，结果一样，数值上稳得多。

**标准化**（第 5 关要用）：

$$z = \frac{x - \mu}{\sigma}$$

**R²**：

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 几个常见的卡点

| 现象 | 原因 |
| --- | --- |
| `TypeError: unsupported operand type(s) for -: 'list' and 'float'` | 忘了 `np.asarray(x, dtype=float)` |
| `ValueError: operands could not be broadcast together` | 形状不对，`(n,)` 和 `(n,1)` 是两回事 |
| loss 越训练越大，最后 `nan`/`inf` | 学习率太大，步子迈过头了 |
| 多特征训练不动，或者只更新了一个权重 | 没做标准化，量纲差 100 倍 |
| 梯度测试差一点点过不了 | 忘了把 `2/n` 的系数写进去 |

---

## 项目结构

```text
linreg-lab/
├── check.py               # 一键检查：告诉你做到第几关
├── demo.py                # 全部做完后：训练 + 对比两种解法 + 画 ASCII loss 曲线
├── linreg_lab.py          # 关卡加载器（让测试能切换 exercises / solutions）
├── exercises/             # 你写的地方（5 个文件，每个一个 TODO）
├── solutions/             # 参考答案
└── tests/                 # 验收测试（26 个用例）
```

---

## 跑测试

```bash
python -m unittest discover -s tests -v          # 测 exercises/（没写完就会失败，正常）
python check.py --solutions                      # 测参考答案
```

CI 跑的是参考答案——因为 `exercises/` 本来就该是空的。这不算作弊：
练习题仓库的 CI 验的是「题目本身没写错」，不是「你做对了没」。

---

## 学完之后

1. 把 `changchun-housing` 升级一下：那里面只做了描述统计，现在你可以加一个**预测房价**的模块（特征：面积、房龄、楼层），用这五关学到的东西；
2. 换成 `sklearn` 再写一遍，看看它比你多做了什么（正则化、多输出、`fit_intercept`）；
3. 下一步去看**逻辑回归**——把损失换成交叉熵，结构几乎一样。

---

## 依赖

只要 `numpy`。`demo.py` 的曲线是用字符画的，不需要 matplotlib。

## 许可

[MIT](LICENSE)
