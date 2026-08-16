---
cssclasses:
  - 计算机视觉
  - 深度学习
date: 2026-07-13
---
1. Pretext
	生成虚拟标签
	- 把图片打乱拼图，让网络还原正确顺序
	- 图片随机旋转，让网络预测旋转角度
	- 遮住图片一块，让网络还原被遮挡区域
	- 同一张图做两种随机裁剪、变色、模糊，让网络判断两张生成图是否匹配
	使用这些数据自身变换出来的标签对模型进行训练，可以强制网络学习边缘、纹理、物体轮廓、空间位置、光影结构等图像信息
2. Downstream
	将上段训练完的完整 Encoder 直接复用
	- 主干卷积层不需要大规模改动
	- 在 Encoder 最后，新增专属预测层
	用带标签的数据进行训练
# Pretext

- 将图像旋转一个角度
- 将图像分割乘多个小块
	- 输入一个小块，让网络判断其再图像中的位置
	- 将小块的顺序打乱，让网络重新排序
- 遮住图像的一或多块，让网络进行补全
  损失函数为原图和生成图的距离
- Iamge coloring
  将图像变成灰度图，往网络上色

为了完成这些小游戏，模型**被迫学习高质量自然图像特征**，甚至学到物体类别的语义表征

如何判断 Pretext 的效果好不好：
1. Linear Evaluation
   把预训练好的 Encoder 完全冻结，用少量标注数据训练这个线性层，看分类精度
2. Clustering
   把所有图片的特征向量做无监督聚类，看同一类物体的图片特征是否聚成同一簇
3. t-SNE 可视化
   把高维特征压缩到 2D 平面画图，看同类物体的点是否聚集在一起
# Contrastive Representation Learning

对同一张原图做数据增强得到一对正样本，不同图片为负样本；训练编码器，正样本特征向量尽可能靠近、负样本特征向量尽可能远离，即：
$$ \mathrm{score}\big(f(x), f(x^+)\big) \gg \mathrm{score}\big(f(x), f(x^-)\big) $$
Goal：找到最优的 score function
Loss function：
$$ L = -\mathbb{E}_X\left[ \log \frac{\exp\big(s(f(x), f(x^+))\big)} {\exp\big(s(f(x), f(x^+))\big) + \sum_{j=1}^{N-1}\exp\big(s(f(x), f(x_j^-))\big)} \right] $$
## SimCLR

将正样本和负样本都进程 CNN/Transformer，得到特征向量 u，v
$$ s(u, v) = \frac{u^\top v}{\|u\| \|v\|} $$
Problem: It need large batvh size with lots of negatives, in order to get the model to converge to good features
## Momentum Contrastive Learning

分两部分：
- Query Encoder
- Momentum Key Encoder

1. 取无标签原图 x，两套随机变换，记为 $x^{query},x^{key}$，分别送入主编码器和动量编码器
2. $x^{query}$ 通过 CNN/Transformer 输出特征向量 q，$x^{key}$ 输出 k（无梯度运算，通过$\theta_k := m \cdot \theta_k + (1 - m) \cdot \theta_q$ 进行动量更新）
3. 将本次新生成的 k 存入队列尾部，最老旧的一批历史 k 自动弹出丢弃
4. 计算损失
	- 正样本对：q 和刚存入的 k
	- 全部负样本对：除了新进入的 k 外，队列中原来的 k
5. 反向传播更新 Query Encoder 网络，动量平滑更新动量编码器权重
## DINO

核心思路：不使用负样本，一张图两种不同裁剪 / 变色，自己教自己

1. 原图随机生成两种不同增强图 $x_1、x_2$
2. $x_1$ 进学生网络，算出分布 $p_1$
   $x_2$ 进教师网络，算出分布 $p_2$
3. $\mathcal{L} = -p_2 \log p_1$，反向传播仅更新学生网络，动量平滑更新教师权重
## Contrastive Predictive Coding

利用**时间先后顺序**构造正负样本
即用过去一段上下文，预测未来时刻的特征，做对比损失

1. 将过去一段时间的图像整合成一个 context vector
2. 正样本：未来一小段时间内的图像特征
   负样本：其他不匹配的时序特征
