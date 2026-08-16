---
cssclasses:
  - 机器学习
  - 强化学习
date: 2026-05-08
---
  # Introduction

Goal: **Optimise the value function of an unknown MDP**

 - On-Policy learning：给出policy，让机器通过学习决定 optimal action
 - Off-Policy learning：“Look over someone's shoulder”，数据来自另一个行为策略 $\mu$ 

核心：pick an action that maximize the value function

![[greedy.png]]
# Monte-Carlo Control
## $\epsilon$ - Greedy Exploration

在进过较多次的学习后，机器会选当前已知最好的动作（贪心动作），导致只进行很单一的 action，可能错过其他的较好的action

$$ \pi(a|s) = \begin{cases} 1 - \epsilon + \dfrac{\epsilon}{m} & \text{if } a = a^* = \arg\max_{a \in \mathcal{A}} Q(s,a) \\[1em] \dfrac{\epsilon}{m} & \text{otherwise} \end{cases} $$
m：动作空间的大小；a*：贪心动作；$\epsilon$：探索概率

目的：让所有动作都有被选中的非零概率，从而保证 policy improvement   
## Control Plan

每跑完一整局或几局，就用数据更新一次 Q 值，然后立刻改进一次策略 

All state-action pairs are explored infinitely many times
# TD Control

TD learning has several advantages over MC :
- lower variance
- online (update every time-step)
- incomplete sequences
$$ Q(S,A) \leftarrow Q(S,A) + \alpha \left( R + \gamma Q(S',A') - Q(S,A) \right) $$
要更新 $Q(S,A)$，你必须记录整条轨迹里：每一步在哪个状态 S，选了哪个动作 A，获得了什么奖励 R

![[TD control.png|697]]
## 收敛性定理

满足以下条件时，$Q(S,A)$ 会收敛到最优动作价值函数 $q_*(S,A)$
1. **Greedy in the Limit with Infinite Exploration：**
      - **无限探索**：在训练过程中，每个 $(s,a)$ 被采样到的次数是无限多的
      - **极限贪心**：随着训练次数 $t \to ∞$  策略 $π_t$​ 收敛到一个贪心策略（即完全根据 Q 选最优动作）
      - 常见策略：衰减的 $\epsilon - greedy$ ，前期 $\epsilon$ 较大，鼓励探索；随着训练进行，$\epsilon$ 逐渐衰减到 0，策略越来越贪心
2. **Robbins-Monro 步长序列：**
   学习率 $\alpha$ 满足：$$ \sum_{t=1}^{\infty} \alpha_t = \infty, \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty $$
   前者保证算法有足够的学习动力；后者保证后期更新足够稳定，避免在最优值附近震荡
## n-Step
### Forward View
$$ \begin{aligned} 
&Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( q_t^{(n)} - Q(S_t, A_t) \right)\\

&q_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}) \end{aligned} $$

通过 $\lambda$ combine 所有 n-step：

$$ \begin{aligned} q_t^\lambda &= (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} q_t^{(n)} \\ Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \alpha \left( q_t^\lambda - Q(S_t, A_t) \right) \end{aligned} $$

![[forward.png|337]]
### Backward View

Forward View 的缺点：必须**等未来的奖励和状态都出现了**，才能算出 $q_t^\lambda$​，然后更新 $Q(S_t​,A_t​)$，这导致效率很低

**引入 Eligibility Traces：** 将前向视角的 “未来信息加权”，变成了后向视角的 “过去状态加权”
$$ \begin{aligned} 
E_0(s,a) &= 0 \\[3pt]
E_t(s,a) &= \gamma \lambda E_{t-1}(s,a) + \mathbf{1}(S_t = s, A_t = a) 
\end{aligned} $$
其中：
$$\mathbf{1}(S_t = s, A_t = a) = 
\begin{cases} 
1 & S_t = s, A_t = a, \\ 0 & otherwise. 
\end{cases}$$
$E_t$ 记录了每个 (s,a) 对当前 TD 误差的贡献度
用当前的 TD 误差，按Eligibility Traces更新所有过去的状态:
$$ \begin{aligned} 
&\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \\[5pt] 
&Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E_t(s,a) \end{aligned} $$

![[backward.png|697]]
# Off-Policy Learning

On-Policy：用来生成数据的策略，就是要学习的策略本身。所以只需要一个 给 agent 提供一个 policy $\pi$，让其通过轨迹数据来更新 $\pi$

Off-Policy：用来生成数据的策略，和你要学习的策略，可以不同
- Behavior Policy $\mu$：负责环境交互、生成数据
- Target Policy $\pi$：最终想要学习、优化的策略。本身可以是贪心的（只选最优动作），不需要带探索，因为它不直接和环境交互

$\mu$ 可以通过学习其他 agent（like humans），也可以是其他 old policies，这也是 off-policy 的优势
## Importance Sampling

$$ \begin{aligned} \mathbb{E}_{X \sim P}\left[G_t(X)\right] &= \sum_{X} P(X) G_t(X) \\ &= \sum_{X} Q(X) \frac{P(X)}{Q(X)} G_t(X) \\ &= \mathbb{E}_{X \sim Q}\left[ \frac{P(X)}{Q(X)} G_t(X) \right] \end{aligned} $$
- $P(X)$：目标分布（Target Distribution），对应目标策略 $\pi(A|S)$
- $Q(X)$：行为分布，对应目标策略 $\mu(A|S)$
- $\frac{P(X)}{Q(X)}$：重要性权重（Importance Weight），describing the similarity between policies
## MC+Importance Sampling
$$ G_t^{\pi/\mu} = \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)} \frac{\pi(A_{t+1}|S_{t+1})}{\mu(A_{t+1}|S_{t+1})} \cdots \frac{\pi(A_T|S_T)}{\mu(A_T|S_T)} G_t $$

权重是轨迹的概率比，方差很大，无法使用
## TD+Importance Sampling
$$ V(S_t) \leftarrow V(S_t) + \alpha \left( \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)} \left( R_{t+1} + \gamma V(S_{t+1}) \right) - V(S_t) \right) $$
Much lower variance than Monte-Carlo importance sampling, but still very high
## Q-Learning

**No importance sampling is required**

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t) \right) $$

当处在 $S_t$ 时，通过行为策略 $\mu$ 决定 action $A_t$（会探索），到达下一个 state $S_{t+1}$，得到 $R_{t+1}$
$A'$ 是目标策略 $\pi$ 在 $S_{t+1}$ 下选择的贪心动作，与 $A_{t+1}$ 无关

Q-Learning 通过在更新时直接取目标策略的贪心 Q 值，绕开了对行为策略轨迹的重要性采样，因此方差更低、实现更简单

**Both behavior and target policies can be improved**
二者的 improve 使用的是同一张价值表

![[Q.png|697]]
