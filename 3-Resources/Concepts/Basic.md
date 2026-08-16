---
cssclasses:
  - 深度学习
  - 生成模型
date: 2026-07-19
---
$$P(S_1 \cap S_2 \cap \dots \cap S_n) = P(S_1)P(S_2 \mid S_1)\cdots P(S_n \mid S_1 \cap \dots \cap S_{n-1})$$
$$p(S_1 \mid S_2) = \dfrac{p(S_1 \cap S_2)}{p(S_2)} = \dfrac{p(S_2 \mid S_1)p(S_1)}{p(S_2)}$$

Problem：如果采用链式准则，所需参数很大
$$ p(x_1,\dots,x_n) = p(x_1)p(x_2 \mid x_1)p(x_3 \mid x_1,x_2)\cdots p(x_n \mid x_1,\dots,x_{n-1}) $$
Soution 1：引入马尔可夫条件简化
$P(X_{i+1})$ 只与 $P(X_i)$ 有关
$$p(x_1, \dots, x_n) = p(x_1)p(x_2 \mid x_1)p(x_3 \mid x_2)\cdots p(x_n \mid x_{n-1})$$
But 过于简化（当前时刻不可能只与上一个时刻相关）
Soution 2：条件参数化
$$ p(x_1,\dots,x_n) = \prod_i p(x_i \mid X_{A_i}) $$
其中 $X_{A_i}$ 是 $X_i$ 的依赖子集，可以是前面像素、邻域像素、隐变量等
通过局部相关性进行简化
 ![[Bayesian_Network.png]]
 通过贝叶斯网络，改全局概率可简化为：
 $p(d,i,g,s,l) = p(d)p(i)p(g \mid i,d)p(s \mid i)p(l \mid g)$
