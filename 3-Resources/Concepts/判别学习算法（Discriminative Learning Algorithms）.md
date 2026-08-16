---
cssclasses:
  - 机器学习
date: 2026-05-28
---
# 线性回归

假设：输出与输入成线性关系

$$h(x) = \sum_{i=1}^n \theta_i \cdot x_i \quad  where \quad x_0 = 1$$

goal: find the best parameters $\mathbf{\theta}$ through limited numbers of sample data
$$J(\theta) = \frac{1}{2}\sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$$
minimize $J(\theta)$

## 批量梯度下降法

扫描整个数据集，将每个样本计算出来的梯度求和取平均，得到总的梯度
$$ \theta_j \leftarrow \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) = \theta_j - \alpha \big(y^{(i)}-h_\theta(x^{(i)})\big)x_j^{(i)}$$

$\alpha$：learning rate

整个数据集可以重复使用多次，直到 $\theta$ 收敛

优点：确保梯度的方向足够正确
缺点：当数据量很大时，梯度下降的每一步都很慢，因为每一次都要对整个数据集进行扫描

## 随机梯度下降

随机选取数据集里面的一个数据，用它的梯度来更新 $\theta$

优点：效率足够高，整体朝着全局最优值收敛
缺点：参数更新全程上下震荡，在最优值附近来回波动

如何判断什么时候停止下降：
绘制 $J(\theta)$ 的图像，当 $J$ 不再下降时，停止训练

## Normal Equation

直接通过计算来求 $\theta$ 的值，避免多次迭代

$$ \nabla_\theta J(\theta) = \begin{bmatrix} \dfrac{\partial J}{\partial \theta_0}\\[6pt] \dfrac{\partial J}{\partial \theta_1}\\[6pt] \vdots\\[4pt] \dfrac{\partial J}{\partial \theta_n} \end{bmatrix} = \vec{0}$$
求解方程即可

---
**basic knowledge of 线性代数**

1. if $f(A) = \mathrm{tr}(AB)$，then $\nabla_A f(A) = B^\mathrm{T}$
2. $\mathrm{tr}(AB) = \mathrm{tr}(BA)$
   $\mathrm{tr}(ABC) = \mathrm{tr}(CAB) = \mathrm{tr}(BCA)$
3. $\nabla_A \mathrm{tr}\big(AA^\mathrm{T}C\big) = CA + C^\mathrm{T}A$

记 $\vec{X} = \begin{bmatrix} \big(x^{(1)}\big)^\mathrm{T}\\[6pt] \big(x^{(2)}\big)^\mathrm{T}\\[6pt] \vdots\\[4pt] \big(x^{(n)}\big)^\mathrm{T} \end{bmatrix}$，则 $\vec{h_\theta(\vec{x})} = \vec{X} \cdot \vec{\theta}$

记 $\vec{y} = \begin{bmatrix} (y^{(1)}\\[6pt] y^{(2)}\\[6pt] \vdots\\[4pt] (y^{(n)} \end{bmatrix}$，则 $J(\theta) = \dfrac12 \big(\mathbf{X}\theta - \mathbf{y}\big)^\mathrm{T}\big(\mathbf{X}\theta - \mathbf{y}\big)$

---

经过理论推导可得：
$$ \nabla_\theta J(\theta) = \mathbf{X}^\mathrm{T}\mathbf{X}\theta - \mathbf{X}^\mathrm{T}\mathbf{y} $$
 即：$$ \hat{\theta} = \big(\mathbf{X}^\mathrm{T}\mathbf{X}\big)^{-1}\mathbf{X}^\mathrm{T}\mathbf{y} $$
## 局部加权回归

每个预测点 x 单独用附近样本加权拟合局部直线

如何确定局部直线的参数：方法同 [[回归#线性回归]]
$$J(\theta) = w^{(i)}\sum_{i=1}^n (\theta^T x^{(i)} - y^{(i)})^2$$
其中权重 w 描述误差的重要程度
当 $x^{(i)}$ 距离目标 x 越近时，$w^{(i)}$ 越大，一个很好的函数为：$$w^{(i)}=\exp\left(-\frac{\left\|x^{(i)}-x\right\|^2}{2\tau^2}\right)$$
其中 $\tau$ 用于调整宽度
# 二分类问题 ( $y \in \{0,1\}$ )

**模型前半段全是特征加权线性求和，默认决策 / 拟合关系为线性**
## 逻辑回归

用于解决 $y \in \{0,1\}$ 的情况 

---
逻辑函数：
$$ g(z)=\frac{1}{1+\mathrm{e}^{-z}} $$
![[逻辑函数.png]]

作用：将数据映射成 0~1 的**概率**，即：
$$ h_\theta(X)=g(\theta^\mathrm{T}x)=\frac{1}{1+\mathrm{e}^{-\theta^\mathrm{T}x}} $$
---
$$ \begin{cases} P(y=1\mid x;\theta)=h_\theta(x)\\ P(y=0\mid x;\theta)=1-h_\theta(x) \end{cases},\quad y\in\{0,1\} $$ 经过合并化简可得：
$$ P(y\mid x;\theta)=h_\theta(x)^y \big(1-h_\theta(x)\big)^{1-y} $$

定义似然函数：$l(\theta)=P(\vec y\mid \vec x;\theta)$，表示给定全部样本特征$\boldsymbol X$，在参数$\boldsymbol \theta$下，整套真实标签 $\boldsymbol y$ 同时发生的联合概率

目标：找到最优的 $\theta$，使似然函数最大

$$\begin{align} L(\theta)=P(\vec y\mid \vec x;\theta) &=\prod_{i=1}^m P\big(y^{(i)}\mid x^{(i)};\theta\big) \\&=\prod_{i=1}^m h_\theta(x^{(i)})^{y^{(i)}}\big(1-h_\theta(x^{(i)})\big)^{1-y^{(i)}} \end{align}$$
取对数可得：
$$ l(\theta)=\ln L(\theta)=\sum_{i=1}^m \Big[y^{(i)}\ln h_\theta(x^{(i)})+\big(1-y^{(i)}\big)\ln\big(1-h_\theta(x^{(i)})\big)\Big] $$

### 梯度上升法
$$ \theta_j \leftarrow \theta_j + \alpha \frac{\partial}{\partial \theta_j}l(\theta) $$
经过理论推导可得：
$$ \theta_j \leftarrow \theta_j + \alpha \sum_{i=1}^m \big(y^{(i)}-h_\theta(x^{(i)})\big)x_j^{(i)} $$
### 牛顿法
![[牛顿法.png]]

$$ \theta^{(t+1)} \leftarrow \theta^{(t)} - \frac{f(\theta^{(t)})}{f'(\theta^{(t)})} $$
令 $f(\theta) = l'(\theta)$，找到使 $l'(\theta) = 0$ 的 $\theta$，即为最优解

当 $\theta$ 为向量时：
$$ \theta^{(t+1)} \leftarrow \theta^{(t)} + H^{-1}\nabla_\theta l $$
其中 $H_{n+1 \times n+1}$ 为 Hesse 矩阵

Hesse 矩阵求逆需要消耗较多的算力，所以只有当 $\theta$ 的维度较小时，才使用牛顿法

## 感知回归


$$ g(z)= \begin{cases} 1 & z\ge0\\ 0 & z<0 \end{cases} $$
$$h_\theta(X)=g(\theta^\mathrm{T}x)$$
误差 $y^{(i)}-h_\theta(x^{(i)}$ 有3种情况：
- 0：算法正确
- 1：wrong, $y^{(i)} = 1$
- -1：wrong, $y^{(i)} = 0$
# 多分类问题
## softmax 回归

目标：对数据进行多分类（逻辑回归 plus）

- $\boldsymbol{K}$：总共有$K$个分类类别（class）
- $\boldsymbol{x^{(i)} \in \mathbb{R}^n}$：第 $i$ 个样本，$n$ 维实数特征向量
- $\boldsymbol{y \in \{0,1\}^K}$：$K$ 维0-1向量，若 y 对应第k类，则向量的第 k 个为1，其余都为0

![[softmax.png|204]]
 
每一个 class 都有一个独立的参数向量 $\mathbf{\theta} \in \mathbb{R}^n$
$$ \hat p(y) = h_{\theta_i}(x) =\frac{\mathrm{e}^{\theta_i^\mathrm{T}x}}{\sum_{i=1}^K \mathrm{e}^{\theta_i^\mathrm{T}x}} $$
从而计算得到所有类型的概率分布

如果每个训练样本都只存在一个类别的物体，则可以将该分类问题转换成 K 个二分类问题：
$$ \mathcal{L} = -\sum_{k=1}^{K} \left[ y_k \log \hat{y}_k + (1 - y_k) \log(1 - \hat{y}_k) \right] $$
否则，使用交叉熵：
定义第 i 个样本在输入 $x^{(i)}$ 条件下，标签 $y^{(i)}$ 发生的概率为：
$$ P(y^{(i)} \mid x^{(i)})=\prod_{k=1}^K (p_{i,k})^{y_k^{(i)}} = p_{i,k_i}$$
则全局似然概率为： 
$$ L(\theta)=\prod_{i=1}^N P(y^{(i)}\mid x^{(i)})=\prod_{i=1}^N p_{i,k_i} $$
取对数并乘-1，可以得到交叉熵损失
$$ \mathcal{L}(\theta) = -\ln L(\theta) = -\sum_{i=1}^N \ln p_{i,k_i} $$
单样本损失为：
$$ \mathcal{L} = -\sum_{k=1}^{K} y_k \ln \hat{y}_k $$
# 广义线性模型（GLM）

基本假设：
1. 在输入特征 x 和模型参数 $\theta$ 已知时，样本标签y的条件概率分布属于 [[概率分布#指数家族]] 即： $y\mid x;\theta \sim \mathrm{Exponential\ Family}(\eta)$
2. $\eta = \theta^T x$ 自然参数 $\eta$ 是特征线性组合（**人为线性假设**）
3. $h_\theta(x)=\mathbb{E}[y\mid x;\theta]$ 即最终预测值 = y的条件期望

![[GLM.jpg]]

仍然使用梯度下降进行更新：
$$ \theta_j \leftarrow \theta_j + \alpha \big(y^{(i)}-h_\theta(x^{(i)})\big)x_j^{(i)} $$
对于任何特定类型的 GLM，都采用这一 update rule
只需代入不同的 $h_\theta(x)$ 即可得到不同的学习规则

---
$\eta = \theta^T x$
$\mu = \mathbb{E}[y\mid \eta] = g(\eta)$，we call $g$ Canonical Response Function
$\eta = g^{-1}(\mu)$，we call $g^{-1}$ Canonical Link Function

---
对 GLM 应用不同的分布，可以得到不同的回归模型

| 分布类型 | 标签类型    | 链接函数                          | 模型       | 任务     |
| ---- | ------- | ----------------------------- | -------- | ------ |
| 伯努利  | 0/1 二分类 | $\eta = \ln\frac{\mu}{1-\mu}$ | **逻辑回归** | 二分类    |
| 高斯   | 连续实数    | $\eta = \mu$                  | **线性回归** | 数值预测   |
| 泊松   | 非负整数    | $\eta = \ln\mu$               | **泊松回归** | 计数预测   |
| 伽马   | 正实数     | $\eta = 1/\mu$                | **伽马回归** | 正值连续预测 |
