---
cssclasses:
  - 机器学习
  - 强化学习
date: 2026-05-10
---
In large scale, it's not practical to build a table to record each value in every state-action.
- known MDP: $V(s)$
- model-free: $Q(s,a)$
Or it is too slow to learn the value of each state individually

Solution: 通过已知的 state 预测未知 state
$$ \begin{aligned} \hat{v}(s, \mathbf{w}) &\approx v_\pi(s) \\[3pt] \text{or } \hat{q}(s, a, \mathbf{w}) &\approx q_\pi(s, a) \end{aligned} $$
$\textbf{w}$ 为参数
Update parameter $\textbf{w}$ using MC or TD learning

**Only talk about Differentiable Function Approximate**
# Gradient Descent
$$ \nabla_{\mathbf{w}} J(\mathbf{w}) = \begin{pmatrix} \frac{\partial J(\mathbf{w})}{\partial w_1} \\ \vdots \\ \frac{\partial J(\mathbf{w})}{\partial w_n} \end{pmatrix} $$
$$ J(\mathbf{w}) = \mathbb{E}_\pi \left[ \left( v_\pi(S) - \hat{v}(S, \mathbf{w}) \right)^2 \right] $$
Goal: find parameter vector $\mathbf{w}$ minimising mean-squared error between approximate value fn $\hat{v}(S, \mathbf{w})$ and true value fn $v_\pi(S)$ —— **use Gradient Descent**
$$ \begin{aligned} 
\Delta \mathbf{w} &= -\frac{1}{2}\alpha \nabla_{\mathbf{w}} J(\mathbf{w}) \\[5pt]
&= \alpha \left( v_\pi(S) - \hat{v}(S, \mathbf{w}) \right) \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w}) \end{aligned} $$
# Linear Value Function Approximation
$$ \mathbf{x}(S) = \begin{pmatrix} x_1(S) \\ \vdots \\ x_n(S) \end{pmatrix} $$
其中 $x_i(S)$ 为根据已有的 state $S_i$ 得到的信息
$$ \begin{aligned} 
\hat{v}(S, \mathbf{w}) = \mathbf{x}(S)^\top \mathbf{w} \\[5pt]
So \quad \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w}) = \mathbf{x}(S) 
\end{aligned} $$
Therefore:
$$ \Delta \mathbf{w} = \alpha \left( v_\pi(S) - \hat{v}(S, \mathbf{w}) \right) \mathbf{x}(S) $$
But in reality, we don't know the true value $v_\pi(S)$. Thus we should use MC or TD learning to estimate it. 

以 TD($\lambda$) 为例：
1. Forward view
   $$ \begin{aligned} \Delta \mathbf{w} &= \alpha \left( G_t^\lambda - \hat{v}(S_t, \mathbf{w}) \right) \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w}) \\ &= \alpha \left( G_t^\lambda - \hat{v}(S_t, \mathbf{w}) \right) \mathbf{x}(S_t) \end{aligned} $$
2. Backward view
   $$ \begin{aligned} \delta_t &= R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w}) \\ E_t &= \gamma \lambda E_{t-1} + \mathbf{x}(S_t) \\ \Delta \mathbf{w} &= \alpha \delta_t E_t \end{aligned} $$
Both of these two view are equivalent.

使用测量好的 $<S_i,G_i>$ 进行 supervised learning：
通过任意选取一个 state $S_i$ ，测量 $S_i$ 与 $S_j$ 的关联度，赋予每个关联度一个权重 $w_{ij}$ ，然后计算 $v(S_i)$ ，用测好的 $G_i$ 对 $w_{ij}$ 进行修正，一共可以修正n次

同样的方式可以应用于 $q(S,A)$  
## Linear Least Squares Prediction

对于线性模型，可以这样求出 $\mathbf{w}$：
$$ \begin{aligned} \mathbb{E}_{\mathcal{D}} \left[ \Delta \mathbf{w} \right] &= 0 \\ \alpha \sum_{t=1}^{T} \mathbf{x}(s_t) \left( v_t^\pi - \mathbf{x}(s_t)^\top \mathbf{w} \right) &= 0 \\ \sum_{t=1}^{T} \mathbf{x}(s_t) v_t^\pi &= \sum_{t=1}^{T} \mathbf{x}(s_t) \mathbf{x}(s_t)^\top \mathbf{w} \\ \mathbf{w} &= \left( \sum_{t=1}^{T} \mathbf{x}(s_t) \mathbf{x}(s_t)^\top \right)^{-1} \sum_{t=1}^{T} \mathbf{x}(s_t) v_t^\pi \end{aligned} $$

---
**Converge or not**

- **评估算法：**

| On/Off-Policy | Algorithm   | Table Lookup | Linear | Non-Linear |
| ------------- | ----------- | ------------ | ------ | ---------- |
| On-Policy     | MC          | y            | y      | y          |
|               | TD          | y            | y      | n          |
|               | Gradient TD | y            | y      | y          |
| Off-Policy    | MC          | y            | y      | y          |
|               | TD          | y            | n      | n          |
|               | Gradient TD | y            | y      | y          |

- **控制算法：**

| Algorithm           | Table Lookup | Linear | Non-Linear |
| ------------------- | ------------ | ------ | ---------- |
| Monte-Carlo Control | y            | y      | n          |
| Sarsa               | y            | y      | n          |
| Q-learning          | y            | n      | n          |
| Gradient Q-learning | y            | y      | n          |

---
# Batch methods
## Least Squares Prediction（最小二乘法）

Data Set:
$$ \mathcal{D} = \left\{ \langle s_1, v_1^\pi \rangle, \langle s_2, v_2^\pi \rangle, \dots, \langle s_T, v_T^\pi \rangle \right\} $$

Goal: to minimize $\sum_{t=1}^{T} \left( v_t^\pi - \hat{v}(S_t, \mathbf{w}) \right)^2$
## Deep Q-Network

传统Q-Learning 在非线性函数近似（如神经网络）下会不稳定，因为 TD目标本身也是用当前网络估计的，随着参数更新而变化，导致训练目标 “移动”，模型难以收敛
**解决方案：使用两个网络**
- 预测网络：不断更新，输出预测值
- 目标网络：较久更新一次，用来算标准答案

初始创建两个神经网络时，二者的参数完全相同

1. Take action $a_t$ according to $ε-greedy$ policy
2. 将每一次的 $(s_t, a_t, r_{t+1}, s_{t+1})$ 存进 D
3. 从 D 中随机取样，消除时序相关性，使训练更稳定
4. 用目标网络计算 TD 目标 $Q(s,a)$
5. 用在线网络计算当前 $Q(s',a')$
6. 梯度下降更新在线网络权重 $\omega$，即两个网络相互学习。
   让 $Q$ 尽量接近 $r + \gamma Q'$ 
7. 经过一定时间后，用在线网络更新目标网络，then repeat

**DQN 根本不是让在线网络 “学目标网络的正确答案”！！！
而是让在线网络、目标网络之间，始终满足贝尔曼方程的关联结构。最后全体自动收敛到真实 Q 值**


