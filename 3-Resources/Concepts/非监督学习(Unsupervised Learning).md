---
cssclasses:
  - 机器学习
date: 2026-07-05
---
在无监督学习中，训练样本的标记是未知的，需要通过对无标记的训练样本的学习来揭示数据之间的内在性质及其规律
# 聚类（Clustering）
## K-means 聚类
```pseudo
\begin{algorithm} \caption{K-Means聚类算法} \begin{algorithmic} \INPUT{样本集 $D = \{x_1, x_2, \dots, x_m\}$；聚类簇数 $k$} \OUTPUT{簇划分结果 $\mathcal{C} = \{C_1,C_2,\dots,C_k\}$} \STATE 从 $D$ 中随机选取 $k$ 个样本作为初始均值向量 $\{\mu_1, \mu_2, \dots, \mu_k\}$ \REPEAT \STATE 令 $C_i = \varnothing \quad (1 \le i \le k)$ \FOR{$j = 1,2,\dots,m$} \STATE 计算样本 $x_j$ 与各均值向量 $\mu_i \ (1 \le i \le k)$ 的距离：$d_{ji} = \|x_j - \mu_i\|_2$ \STATE 根据距离最近的均值向量确定 $x_j$ 的簇标记：$\lambda_j = \arg\min_{i \in \{1,2,\dots,k\}} d_{ji}$ \STATE 将样本 $x_j$ 划入对应簇：$C_{\lambda_j} = C_{\lambda_j} \cup \{x_j\}$ \ENDFOR \FOR{$i = 1,2,\dots,k$} \STATE 计算新均值向量：$\mu_i' = \frac{1}{|C_i|}\sum_{x \in C_i} x$ \IF{$\mu_i' \neq \mu_i$} \STATE 将均值向量 $\mu_i$ 更新为 $\mu_i'$ \ELSE \STATE 保持均值向量不变 \ENDIF \ENDFOR \UNTIL{所有均值向量均无更新} \end{algorithmic} \end{algorithm}
```
K 的值通常手动选取
## 密度聚类

**核心假设：聚类结构可以通过样本之间的紧密程度来确定**

若 $x_i$ 与 $x_j$ 的距离小于 $\epsilon$，则称 $x_i$ 与 $x_j$ 密度相连
密度相连具有传递性

簇：由密度可达关系导出的最大的密度相连样本集合
```pseudo
\begin{algorithm} \caption{DBSCAN聚类算法} \begin{algorithmic} \INPUT{样本集 $D = \{x_1,x_2,\dots,x_m\}$；邻域参数 $(\epsilon, \text{MinPts})$} \OUTPUT{簇划分 $\mathcal{C} = \{C_1,C_2,\dots,C_k\}$} \STATE 初始化核心对象集合: $\Omega = \varnothing$ \FOR{$j = 1,2,\dots,m$} \STATE 确定样本 $x_j$ 的 $\epsilon$-邻域 $N_\epsilon(x_j)$ \IF{$|N_\epsilon(x_j)| \ge \text{MinPts}$} \STATE 将样本 $x_j$ 加入核心对象集合: $\Omega = \Omega \cup \{x_j\}$ \ENDIF \ENDFOR \STATE 初始化聚类簇数: $k = 0$ \STATE 初始化未访问样本集合: $\Gamma = D$ \WHILE{$\Omega \ne \varnothing$} \STATE 记录当前未访问样本集合: $\Gamma_{\text{old}} = \Gamma$ \STATE 随机选取一个核心对象 $o \in \Omega$，初始化队列 $Q = <o>$ \STATE $\Gamma = \Gamma \setminus \{o\}$ \WHILE{$Q \ne \varnothing$} \STATE 取出队列 $Q$ 中的首个样本 $q$ \IF{$|N_\epsilon(q)| \ge \text{MinPts}$} \STATE 令 $\Delta = N_\epsilon(q) \cap \Gamma$ \STATE 将 $\Delta$ 中的样本加入队列 $Q$ \STATE $\Gamma = \Gamma \setminus \Delta$ \ENDIF \ENDWHILE \STATE $k = k + 1$，生成聚类簇 $C_k = \Gamma_{\text{old}} \setminus \Gamma$ \STATE $\Omega = \Omega \setminus C_k$ \ENDWHILE \end{algorithmic} \end{algorithm}
```
```pseudo
```
## 层次聚类

自下而上，生成树形聚类结构
```pseudo
\begin{algorithm} \caption{层次聚类（AGNES）算法} \begin{algorithmic} \INPUT{样本集 $D = \{x_1,x_2,\dots,x_m\}$；聚类簇距离度量函数 $d$；聚类簇数 $k$} \OUTPUT{簇划分 $\mathcal{C} = \{C_1,C_2,\dots,C_k\}$} \FOR{$j = 1,2,\dots,m$} \STATE $C_j = \{x_j\}$ \ENDFOR \FOR{$i = 1,2,\dots,m$} \FOR{$j = i+1,\dots,m$} \STATE $M(i,j) = d(C_i,C_j)$ \STATE $M(j,i) = M(i,j)$ \ENDFOR \ENDFOR \STATE 设置当前聚类簇个数: $q = m$ \WHILE{$q > k$} \STATE 找出距离最近的两个聚类簇 $C_{i^*}$ 和 $C_{j^*}$ \STATE 合并 $C_{i^*}$ 和 $C_{j^*}$: $C_{i^*} = C_{i^*} \cup C_{j^*}$ \FOR{$j = j^*+1,j^*+2,\dots,q$} \STATE 将聚类簇 $C_j$ 重编号为 $C_{j-1}$ \ENDFOR \STATE 删除距离矩阵 $M$ 的第 $j^*$ 行与第 $j^*$ 列 \FOR{$j = 1,2,\dots,q-1$} \STATE $M(i^*,j) = d(C_{i^*},C_j)$ \STATE $M(j,i^*) = M(i^*,j)$ \ENDFOR \STATE $q = q - 1$ \ENDWHILE \end{algorithmic} \end{algorithm}
```
## 高斯混合聚类

通过概率模型表示聚类

假设观测样本 $x^{(i)}$ 由一个不可观测的离散隐变量 $z^{(i)}$ 控制，$z^{(i)}$ 代表第 i 个样本属于第 $z^{(i)}$ 个高斯簇
$$ P\left(x^{(i)}, z^{(i)}\right) = P\left(x^{(i)} \mid z^{(i)}\right) P\left(z^{(i)}\right) $$
若所有样本的簇归属隐变量 $z^{(i)}$ 已知，可直接闭式求解全部 GMM 参数
对数联合似然目标 ：$$ \ell(\phi,\mu,\Sigma) = \sum_{i=1}^m \log p\big(x^{(i)},z^{(i)};\phi,\mu,\Sigma\big) $$ $m$：总样本
混合权重 $\phi_j = \dfrac{\sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}{m}$
高斯均值 $\mu_j = \frac{\displaystyle \sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\} x^{(i)}}{\displaystyle \sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}$
高斯协方差 $\Sigma_j = \frac{\displaystyle \sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\displaystyle \sum_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}$

但实际上，我们并不知道隐变量 $z^{(i)}$
$$ p_{\mathcal{M}}(x) = \sum_{i=1}^{k} \phi_i \cdot p\left(x \mid \mu_i, \Sigma_i\right), $$
其中 $\phi_i$ 为混合系数，即随机抽一个样本，属于第 i 簇的概率
$$ \begin{aligned} p_{\mathcal{M}}(z_j = i \mid x_j) &= \frac{P(z_j = i) \cdot p_{\mathcal{M}}(x_j \mid z_j = i)}{p_{\mathcal{M}}(x_j)} \\ &= \frac{\phi_i \cdot p\left(x_j \mid \mu_i, \Sigma_i\right)}{\displaystyle\sum_{l=1}^{k} \phi_l \cdot p\left(x_j \mid \mu_l, \Sigma_l\right)}. \end{aligned} $$
则：
$$ \phi_i = \frac{1}{m}\sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j) $$
$$ \mu_i = \frac{\displaystyle \sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j) \cdot x_j}{\displaystyle \sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j)} $$
$$ \Sigma_i = \frac{\displaystyle \sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j) \cdot (x_j-\mu_i)(x_j-\mu_i)^T}{\displaystyle \sum_{j=1}^m p_{\mathcal{M}}(z_j = i \mid x_j)} $$
不断更新这三个值，最终回收敛，即聚类成功
```pseudo
\begin{algorithm} \caption{高斯混合聚类算法} \begin{algorithmic} \INPUT{样本集 $D = \{x_1,x_2,\dots,x_m\}$；高斯混合成分个数 $k$} \OUTPUT{簇划分 $\mathcal{C} = \{C_1,C_2,\dots,C_k\}$} \STATE 初始化高斯混合分布的模型参数 $\{(\phi_i, \mu_i, \Sigma_i) \mid 1 \le i \le k\}$ \REPEAT \FOR{$j = 1,2,\dots,m$} \STATE 计算 $x_j$ 由各混合成分生成的后验概率：$p_{\mathcal{M}}(z_j = i \mid x_j) \ (1 \le i \le k)$ \ENDFOR \FOR{$i = 1,2,\dots,k$} \STATE 计算新均值向量：$\mu_i'$ \STATE 计算新协方差矩阵：$\Sigma_i'$ \STATE 计算新混合系数：$\phi_i'$ \ENDFOR \STATE 将模型参数 $\{(\phi_i, \mu_i, \Sigma_i) \mid 1 \le i \le k\}$ 更新为 $\{(\phi_i', \mu_i', \Sigma_i') \mid 1 \le i \le k\}$ \UNTIL{满足停止条件} \STATE $C_i = \varnothing \quad (1 \le i \le k)$ \FOR{$j = 1,2,\dots,m$} \STATE 确定 $x_j$ 的簇标记 $\lambda_j$ \STATE 将 $x_j$ 划入相应的簇：$C_{\lambda_j} = C_{\lambda_j} \cup \{x_j\}$ \ENDFOR \end{algorithmic} \end{algorithm}
```
---
EM 算法

1. E步（求隐变量后验分布） 
   固定当前模型参数 $\theta$，用贝叶斯公式计算隐变量 $z^{(i)}$ 的后验分布 $Q_i(z^{(i)})$ $$ Q_i(z^{(i)}) := P\big(z^{(i)} \mid x^{(i)};\theta\big) $$
2. M步（最大化下界更新参数）
   更新全局参数 $\theta$$$ \theta := \arg\max_{\theta} \sum_{i}\sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})} $$
记 $J(\theta,Q) = \sum_{i}\sum_{z^{(i)}} Q_i(z^{(i)}) \log \dfrac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}$

- E-step：固定 $\theta$，优化 $Q$ 最大化 $J$
- M-step：固定 E 步求出的最优Q，只优化参数 $\theta$，最大化 $J$
---
# 因子分析

- 当样本数量远大于维度数量时，采用高斯混合可以得到很好的聚类效果
- 当样本数量约等于甚至小于维度数量时，则应当采用因子分析

隐因子服从 d 维标准正态分布 $z \sim \mathcal{N}(0,I),\quad z\in\mathbb R^d,(d < n)$
$x = \mu + \Lambda z + \epsilon,\quad \epsilon \sim \mathcal{N}(0,\Psi)$，即现实变量为隐变量的线性组合加上高斯噪声

---
设联合正态向量拆分
$$ \begin{bmatrix}x_1\\x_2\end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix}\mu_1\\\mu_2\end{bmatrix}, \begin{bmatrix}\Sigma_{11} & \Sigma_{12}\\ \Sigma_{21} & \Sigma_{22}\end{bmatrix} \right) $$
则：
- $x_1 \mid x_2 \sim \mathcal{N}\big(\mu_{1\mid 2},\ \Sigma_{1\mid 2}\big)$
- $\mu_{1\mid 2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2)$
- $\Sigma_{1\mid 2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$
---
$$ \begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} 0 \\ \mu \end{bmatrix},\; \begin{bmatrix} I & \Lambda^\top \\ \Lambda & \Lambda\Lambda^\top + \Psi \end{bmatrix} \right) $$
E-step：
1. $\Sigma_x = \Lambda \Lambda^\top + \Psi$
2. $\Sigma_{z|x} = I_d - \Lambda^\top \Sigma_x^{-1} \Lambda$
3. $\mu_{z|x^{(i)}} = \Lambda^\top \Sigma_x^{-1} \big(x^{(i)} - \mu\big)$
4. $Q_i\big(z^{(i)}\big) \triangleq \mathcal{N}\big(z;\mu_{z|x^{(i)}},\Sigma_{z|x}\big)$

M-step：
1. $Q_i\big(z^{(i)}\big) = \frac{1}{(2\pi)^{d/2} \left|\Sigma_{z|x}\right|^{\frac{1}{2}}} \exp\left( -\frac{1}{2} \big(z^{(i)} - \mu_{z|x^{(i)}}\big)^\top \Sigma_{z|x}^{-1} \big(z^{(i)} - \mu_{z|x^{(i)}}\big) \right)$
2. $\mathbb{E}_{z^{(i)} \sim Q_i}\left[z^{(i)}\right] = \int_{z^{(i)}} z^{(i)} \cdot Q_i\left(z^{(i)}\right) dz^{(i)} = \mu_{z|x^{(i)}}$
3.  $$\begin{align} \theta &:= \arg\max_{\theta} \sum_{i} \int_{z^{(i)}} Q_i\big(z^{(i)}\big) \log \frac{p\big(x^{(i)},z^{(i)}\big)}{Q_i\big(z^{(i)}\big)} dz^{(i)} \\ &= \sum_{i} \mathbb{E}_{z^{(i)} \sim Q_i}\left[\log \frac{p\left(x^{(i)}, z^{(i)}\right)}{Q_i\left(z^{(i)}\right)}\right] \end{align}$$
4. 分别对 $\mu,\Lambda,\Psi$ 求导，令导数 = 0，即可更新 $\mu,\Lambda,\Psi$
# 主成分分析（PCA）

先对数据进行Z-score 标准化（均值为0，方差为1）：
$$ \tilde{x}_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{\sqrt{\displaystyle\frac{1}{m}\sum_{i=1}^{m} \big(x_j^{(i)} - \mu_j\big)^2}}, \quad \mu_j = \frac{1}{m}\sum_{i=1}^{m} x_j^{(i)}$$
**核心：对高维数据进行降维**

理想超平面的性质：
- 样本点到超平面的距离足够近
- 样本点在超平面上的投影尽可能分开
```pseudo
\begin{algorithm} \caption{ PCA算法} \begin{algorithmic} \INPUT{样本集 $D = \{\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_m\}$；低维空间维数 $d'$}  \STATE 对所有样本进行中心化: $\boldsymbol{x}_i \leftarrow \boldsymbol{x}_i - \dfrac{1}{m}\sum_{i=1}^m \boldsymbol{x}_i$ \STATE 计算样本的协方差矩阵 $\boldsymbol{X}\boldsymbol{X}^\top$ \STATE 对协方差矩阵 $\boldsymbol{X}\boldsymbol{X}^\top$ 做特征值分解 \STATE 取最大的 $d'$ 个特征值所对应的特征向量 $\boldsymbol{w}_1,\boldsymbol{w}_2,\dots,\boldsymbol{w}_{d'}$ \OUTPUT{投影矩阵 $\boldsymbol{W}^* = (\boldsymbol{w}_1,\boldsymbol{w}_2,\dots,\boldsymbol{w}_{d'})$} \end{algorithmic} \end{algorithm}
```
 Try to use the raw data before applying PCA

| 数据结构假设           | Probabilistic（概率模型） | Non-probabilistic（非概率模型，无概率密度） |
| ---------------- | ------------------- | ------------------------------ |
| Subspaces（连续子空间） | Factor Analysis     | PCA                            |
| Clusters（离散簇）    | Mixture of Gaussian | K-means                        |
# 独立成分分析（ICA）

- $\boldsymbol{x} \in \mathbb{R}^n$：观测混合信号
- $\boldsymbol{s} \in \mathbb{R}^d$：独立源
则：$\boldsymbol{x} = \boldsymbol{A}\boldsymbol{s} ,\quad \boldsymbol{s} = \boldsymbol{w}\boldsymbol{x}$

ICA 只有在数据是非高斯分布的时候才可以使用

- PDF：Probability Density Function，描述连续随机变量的概率密度（单位区间内的概率变化速率）
- CDF：Cumulative Distribution Function
$$ p_X(\boldsymbol{x}) = p_S(\boldsymbol{W}\boldsymbol{x}) \cdot \left|\det \boldsymbol{W}\right| $$
 由于 $s_i$ 相互独立，有：$$ p(\boldsymbol{s}) = \prod_{i=1}^{n} p(s_i) $$
 整理可得：$$ p_X(\boldsymbol{x}) = \left(\prod_{i=1}^{n} p_S\big(\boldsymbol{w}_i^\top \boldsymbol{x}\big)\right) \cdot \left|\det \boldsymbol{W}\right| $$
goal：figure out the matrix $\boldsymbol{W}$

S 的分布是未知的，但优化本质是最大化信号**非高斯性**，只要选用非高斯分布，就不影响最终分离结果
常采用Logistic 分布：$p_S(z) = \dfrac{e^{-z}}{\left(1 + e^{-z}\right)^2}$（通过对 sigmoid 函数求导得到）

采集 m 个独立样本，则这些样本同时出现的概率即为 $P = \prod_{k=1}^{m} {p}_X\big(\boldsymbol{x}^{(k)}\big)$，只需最大化概率 P 即可

定义似然函数：
$$\begin{align} \ell(W) &= \sum_{i=1}^m \log\left( \left( \prod_{j=1}^n p_s\left(\boldsymbol{w}_j^\top \boldsymbol{x}^{(i)}\right) \right) \cdot |\det W| \right) \\ &= \sum_{i=1}^m \left[ \sum_{j=1}^n \log p_s\left(\boldsymbol{w}_j^\top \boldsymbol{x}^{(i)}\right) + \log|\det W| \right] \end{align}$$
通过梯度上升法求最优的 W$$ W_{\text{new}} = W_{\text{old}} + \eta \cdot \nabla_W \ell(W) $$
