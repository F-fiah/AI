---
cssclasses:
  - 机器学习
  - 强化学习
date: 2026-05-13
---
- Value-Based（之前优化 policy 的策略）：先拟合价值函数，再用价值函数见解生成策略（选取使 value 最大化的 action）
- Policy-Based Reinforcement Learning：直接将策略本身参数化
- Actor-Critic：combine Value-Based RL and Policy-Based RL

$\pi_\theta(s,a)$：在状态 s 下，选取动作 $A_i$ 的概率为 $\theta_i$

Goal: find the best $\theta$
# 衡量策略的好坏

1. 回合制环境——优化**起始状态的价值**（从起点出发，按策略 $\pi_\theta$ 执行，能获得的期望累积奖励）
   $$ J_1(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta} [v_1] $$
2. 连续环境——优化**平均状态价值**
   $$ J_{avV}(\theta) = \sum_{s} d^{\pi_\theta}(s) V^{\pi_\theta}(s) $$
   $d^{\pi_\theta}(s)$：策略 $\pi_\theta$ 下智能体长期处于各个状态的频率

# Gradient Ascending
$$ \Delta \theta = \alpha \nabla_\theta J(\theta) = \alpha\begin{pmatrix} \frac{\partial J(\theta)}{\partial \theta_1} \\ \vdots \\ \frac{\partial J(\theta)}{\partial \theta_n} \end{pmatrix} $$
理论梯度无法计算：
- 状态、动作、轨迹无穷多
- 环境为黑箱，不知道状态转移概率、平稳分布$d^{\pi_\theta}(s)$

所以只能通过和环境的交换，采样出 $(s,a,r)$ ，从而近似梯度

## Score Function
$$ \nabla_\theta \log \pi_\theta(s, a) = \frac{\nabla_\theta \pi_\theta(s, a)}{\pi_\theta(s, a)} \Longrightarrow \nabla_\theta \pi_\theta(s, a) = \pi_\theta(s, a) \cdot \nabla_\theta \log \pi_\theta(s, a) $$
define the score function is $\nabla_\theta \log \pi_\theta(s, a)$
衡量参数 $\theta$ 变化时，动作 a 在状态 s 下的对数概率变化有多敏感

**Score Function 的作用：算出怎么调整参数 $\boldsymbol{\theta}$**

如何计算 Score Function：对动作概率取对数，再对参数 $\boldsymbol{\theta}$ 求（近似）梯度
## Softmax Policy（离散）

将（状态s+动作a）这一组信息转换成特征 $\phi(s,a)$ 
权重 $\theta$ ：对特征的偏好
$$score = \phi(s,a)^T\theta$$
分数越高，动作越好；分数越低，动作越差

得到动作概率：
$$ \pi_\theta(s,a) = \frac{\exp\left(\boldsymbol{\phi}(s,a)^\top \boldsymbol{\theta}\right)}{\sum_{a'} \exp\left(\boldsymbol{\phi}(s,a')^\top \boldsymbol{\theta}\right)} $$
结论：
$$ \nabla_\theta \log \pi_\theta(s,a) = \phi(s,a) - \mathbb{E}_{\pi_\theta}\big[\phi(s,\cdot)\big] $$
## Gaussian Policy（连续）

对于连续的动作空间，使用**高斯（正态）分布**来建模
$$\mu(s) = \phi(s)^T\theta$$
- $\phi(s)$：状态特征向量
- $\mu(s)$：当前状态最推荐的动作值

动作服从正态分布：
$$ a \sim \mathcal{N}\big(\mu(s), \sigma^2\big) $$
- action 在 $\mu(s)$ 附近概率最大
- $\sigma$ 用来衡量探索程度
结论：
$$ \nabla_\theta \log \pi_\theta(s,a) = \frac{\big(a - \mu(s)\big)\phi(s)}{\sigma^2} $$
# Policy Optimisation

## One-Step MDP

令 $d(s)$ 为初始状态分布：智能体一开始出现在各个状态 s 的概率
只执行一步 action，得到奖励 $r = \mathcal{R}_{s,a}$ 后结束
$$ \begin{aligned} J(\theta) &= \mathbb{E}_{\pi_\theta} [r] \\ &= \sum_{s\in\mathcal{S}} d(s) \sum_{a\in\mathcal{A}} \pi_\theta(s,a)\mathcal{R}_{s,a} \\[5pt] \nabla_\theta J(\theta) &= \sum_{s\in\mathcal{S}} d(s) \sum_{a\in\mathcal{A}} \pi_\theta(s,a)\nabla_\theta \log\pi_\theta(s,a)\mathcal{R}_{s,a} \\ &= \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log\pi_\theta(s,a)\,r\big] \end{aligned} $$
即：
$$ \nabla J = \mathbb{E}\big[\text{Score} \cdot \text{价值}\big] $$

**推广到 multi-step MDP: replace $r$ with $Q^\pi(s,a)$**

## MC Policy Gradient

1. Initialize $\theta$ arbitrarily 
2. For each episode $\{s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T\} \sim \pi_\theta$  
3. $v_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots + \gamma^{T-t-1} r_T$  // calculate $v_t$
4. $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(s_t, a_t) \, v_t$ 

## Actor-Critic

本质：将 TD 的价值估计引入到 Policy Gradient 中，通过预测的 value 来修正 $\theta$ 而非实际测量的 value

因为 MC Policy Gradient 具有 high variance，为了进行优化，Actor-Critic 使用 $Q_t(s,a,\textbf{w})$ 来代替 $v_t$ ，其中 $\textbf{w}$ 在学习中是不断被优化的
$$ \Delta \theta = \alpha \nabla_\theta \log \pi_\theta(s,a) \, Q_w(s,a) $$
![[QAC.png]]

### Advantage Function (Baseline)
$$ \begin{aligned} \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log\pi_\theta(s,a)\,B(s)\big] &= \sum_{s\in\mathcal{S}} d^{\pi_\theta}(s) \sum_{a} \nabla_\theta \pi_\theta(s,a)\,B(s) \\ &= \sum_{s\in\mathcal{S}} d^{\pi_\theta}(s)\,B(s)\,\nabla_\theta \sum_{a\in\mathcal{A}} \pi_\theta(s,a) \\ &= 0 \end{aligned} $$
So: 
$$\mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log\pi_\theta(s,a)\,)\,(Q^{\pi\theta}(s,a)\big] = \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log\pi_\theta(s,a)\,(Q^{\pi\theta}(s,a)-B(s)\big)]$$
合适的 $B(s)$ 的选取可以有效减少我们使用随机样本来近似真实梯度的方差

理论证明：$V^{\pi\theta}(s)$ 是一个比较好的 $B(s)$ ，它可以通过 MC/TD 得到

$$ \delta^{\pi_\theta} = Q^{\pi\theta}(s,a) - V^{\pi_\theta}(s) $$
