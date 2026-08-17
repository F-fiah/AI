---
cssclasses:
  - 深度学习
  - 生成模型
date: 2026-08-17
status: in-progress
---
使用两个神经网络，再零和博弈的框架下相互竞争

- 生成器网络：将隐变量转化成数据
- 判别器网络：评估数据的真假性

![[3518c8a8cd3f6b4519f91625377b0f05.jpg]]
训练顺序：
1. 使用真实数据更新 Discriminator
2. 使用虚假数据更新 Discriminator
3. 更新 Generator
