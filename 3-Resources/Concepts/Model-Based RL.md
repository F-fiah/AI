---
cssclasses:
  - 机器学习
  - 强化学习
date: 2026-05-16
---
Previous lesson: learn value function or policy from experience, **(Model-Free)

Model‑free 的缺陷：
- 真实环境交互成本很高
- 必须大量试错，样本效率极低

**Model-based:** 
1. 通过采样拟合出环境模型
2. 使用虚拟模型进行规划和训练
3. 在真实环境中执行 action

A model $\mathcal{M}$ is a representation of an MDP $\langle \mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R} \rangle$, parametrized by $\eta$

1. Learn a model from real experience
2. **Plan** value function (and/or policy) from simulated experience（由于难以遍历所有状态，所以只能在模型里随机生成大量虚拟样本，用 MC / SARSA / Q‑learning / 策略梯度 去近似学出价值和策略）
# Dyna --- Integrate Learning and Planning
 ![[dyna.png|350]]
## Dyna-Q
 ![[dyna_algorithm.png]]
 Attention:
 - n 不能太大，否则真实样本无法对偏差进行有效修正
 - 可以分别赋予真实样本和虚拟样本不同的权重
## Dyna-2

The agent stores two sets of feature weights
- Long-term memory: for Long-term memory
- Short-term memory: for simulated experience

长期记忆为全局知识，适用于所有回合
短期记忆为当下环境的局部临时知识
$$V(s) = V_{long}(s) + V_{short}(s)$$
# Simulation-Based

Dyna‑Q：提前训练全局 Q 表，所有状态价值都学好

Simulation-based search：向前模拟未来，选最好的一步
## MC

1. Given a model $\mathcal{M}_\nu$ and a simulation policy $\pi$ 
2. For each action $a \in \mathcal{A}$ ： 
	- Simulate $K$ episodes from current (real) state $s_t$ ：$$ \{s_t, a, R_{t+1}^k, S_{t+1}^k, A_{t+1}^k, \dots, S_T^k\}_{k=1}^K \sim \mathcal{M}_\nu, \pi $$
	   - Evaluate actions by mean return (Monte‑Carlo evaluation) $$ Q(s_t,a) = \frac{1}{K}\sum_{k=1}^K G_t \xrightarrow{P} q_\pi(s_t,a) $$
	   - Or build a search tree containing visited states and actions, and evaluate states (Q(s,a)) by mean return:$$ Q(s,a) = \frac{1}{N(s,a)}\sum_{k=1}^K\sum_{u=t}^T \mathbb{1}(S_u,A_u=s,a)\,G_u \xrightarrow{\text{收敛}} q_\pi(s,a) $$
3. Select current (real) action with maximum value 
$$ a_t = \underset{a\in\mathcal{A}}{\text{argmax}}\, Q(s_t,a) $$
## TD

- reduce variance
- more efficient
- but increase bias

1. Simulate episodes from the current (real) state $s_t$ 
2. Estimate action‑value function $Q(s, a)$ 
3. For each step of simulation, update action‑values by Sarsa $$ \Delta Q(S,A) = \alpha\big(R + \gamma Q(S',A') - Q(S,A)\big) $$
4. Select actions based on action‑values $Q(s,a)$ - e.g. $\epsilon$-greedy
5. May also use function approximation for $Q$

