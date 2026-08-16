---
cssclasses:
  - 机器学习
  - 强化学习
date: 2026-05-06
---
Nobody tell the environment, but still manage to find the optimal way
Namely estimate the value function of an unknown MDP
# Monte-Carlo Learning

- 目的：找到 policy $\pi$ 对应的 value 函数 $v_\pi$ 
- 核心：用 empirical mean 替代 expected return
- 要求：行为必须 terminate
- 优势：不需要提前知道环境的转移概率和奖励函数，只需和环境交互并通过采样轨迹来学习

具体的算法原理见 [[数学原理#Return]]

对于 loop，有两种 evaluate 的方案：first-visit and every-visit
## 算法
$$ \begin{aligned} \mu_k &= \frac{1}{k} \sum_{j=1}^{k} x_j \\ &= \frac{1}{k} \left( x_k + \sum_{j=1}^{k-1} x_j \right) \\ &= \frac{1}{k} \left( x_k + (k-1)\mu_{k-1} \right) \\ &= \mu_{k-1} + \frac{1}{k} \left( x_k - \mu_{k-1} \right) \end{aligned} $$
只需要用旧均值和新数据，就能增量式更新，无需保存所有数据 

so：
$$ \begin{aligned} N(S_t) &\leftarrow N(S_t) + 1 \\ V(S_t) &\leftarrow V(S_t) + \frac{1}{N(S_t)} \left( G_t - V(S_t) \right) \end{aligned} $$

$G_t$ is unbiased estimate of $v_\pi(S_t)$ 
# Temporal-Difference Learning

- 直接从与环境交互得到的经验序列（episode）中学习
- 可以从不完整的 episode 中学习,不需要等到 episode 结束拿到完整回报 $G_t$
- 适用于 non-terminating environment

由于无法等到 episode 结束，因此TD学习采用 $α$ 来替代 $\frac{1}{N(S_t)}$ ，称为学习率
$$ V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right) $$
**新价值 = 旧价值 + 学习率 × 误差**

利用 estimated return $R_{t+1} + \gamma V(S_{t+1})$ 更新 $V(S_t)$ ：

$$ V(S_t) \leftarrow V(S_t) + \alpha \left( R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right) $$
其中 $R_{t+1}$ 是和环境交互直接拿到的真实数据，$V(S_{t+1})$ 是下一步状态的价值估计。它是通过当前的经验和已有的价值进行估计的（利用当前正在学习的价值表）
## 价值表 V

1. 初始化：所有 state 的 value 设置为0
2. 通过与 environment 的互动对表进行更新
3. 若遇到了新的 state：
   - 将其补充到表里，并设置 $V(S_{new}) = 0$
   - 根据表中相似的旧状态推测新状态的价值

初始时，estimated return 的偏差很大，但会随着学习减少，最终和真实回报一致。
## n-Step Prediction

进行 n 步后再进行 estimate
$n \to \infty$：MC
$$ \begin{aligned} G_t^{(1)} &= R_{t+1} + \gamma V(S_{t+1}) \\ G_t^{(2)} &= R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2}) \\ &\vdots \\ G_t^{(\infty)} &= R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-1} R_T \end{aligned} $$

- Online Update：每走 n 步就更新一次 value function，对学习率 α 敏感，更新频繁容易震荡
- Offline Update：回合结束后，再回头处理轨迹中的每个状态

两种方法得到的 $v(s)$ 大概率不一样，因为在 online 中，价值表 V 实时改变，因此 estimate value 会由于用到已经被修改过的 V而发生改变；对于 offline 轨迹还是用旧 V 生成的，直到批量更新后才会变。

若遇到 new state 是，令 $V(S_{new}) = 0$ ，Online 会因为新状态初值为 0，用很差的目标去更新前面状态，导致前期学习不稳定、效果较差
但由于 online 具有很高的学习效率，因此也被大量使用
优化方案：
- 减小学习率 $\alpha$
- 使用 n-step TD
- 使用更加合理的初始价值

![[NStep.png]]
**How to combine information from all time-steps: $\lambda-return$**  
use weight  $(1-\lambda)\lambda^{n-1}$ :
$$ G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)} $$
![[LambdaWeight.png]]
# Conclude

MC：
- 每次 $R_t$ 的测量都存在 noise，所以存在 high variance, but no bias
- good convergence property
- not sensitive to initial value
TD：
- more efficient
- low variance, some bias
- sensitive to initial value
# Build MDP based on known data

用最大似然估计来估计概率，从而利用轨迹数据估计出最贴合实际的MDP 的模型参数 $P$ 和 $R$ 

Solution to the MDP $\langle \mathcal{S}, \mathcal{A}, \hat{\mathcal{P}}, \hat{\mathcal{R}}, \gamma \rangle$ that best fits the data $$ \hat{\mathcal{P}}_{s,s'}^{a} = \frac{1}{N(s,a)} \sum_{k=1}^{K} \sum_{t=1}^{T_k} \mathbf{1}\left(s_t^k, a_t^k, s_{t+1}^k = s, a, s'\right) $$ $$ \hat{\mathcal{R}}_{s}^{a} = \frac{1}{N(s,a)} \sum_{k=1}^{K} \sum_{t=1}^{T_k} \mathbf{1}\left(s_t^k, a_t^k = s, a\right) r_t^k $$
# Dynamic Programming —— no sample

进行完全模拟（所有可能性）

![[PolicyEvaluation.png]]