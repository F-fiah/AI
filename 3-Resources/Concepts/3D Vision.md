---
cssclasses:
  - 计算机视觉
  - 深度学习
date: 2026-07-15
---
# Shape Representations
## Explicit

1. Point Clouds
	- onlu points, no connectivity
	- potentially noise
	- 难以进行一些图像操作
2. Polygonal Meshes（多边形网络）
	- 用大量平面多边形面片拼接、缝合，包裹出 3D 物体的立体表面
	- 可以支持大量复杂操作
3. Parametric Representation
	- 使用函数的方式表示表面
	- 足够光滑
	- more details

- Easy to sample
- Hard to judge whether a piont is inside or outside the object
## Implicit

Based on classifying points: points satisfy some specified relationship $f(x,y,z) = 0$

The relationship could be figure out by nerual network

- hard to sample
- Easy to judge whether a piont is inside or outside the object
- If the object has complex shaps, it's hard to get the relationship function

Voxels : 将空间的点二进制化（内部 / 外部）
# 3D Deep Learning

## Learning on Voxels

use 3D-CNN / 3D-GANS
## Learning on Points

Point cloud is a set of unorder points

1. 每一个三维坐标点，都送入同一个多层全连接网络h，把原始 3 维坐标映射成高维特征向量
2. 经过 h 变换后的高维特征，进入对称聚合层（置换不变性），输出唯一全局特征向量
3. 将全局特征向量用 MLP 进行处理

Problem : 点云是无序点集合，不能直接用像素 L2 损失（点顺序不一一对应）来衡量预测点云和真实点云的相似程度
Solution : 引入距离来衡量
- Chamfer Distance
  $d_{CD}(S_1,S_2) = \sum_{x\in S_1}\min_{y\in S_2}\|x-y\|_2 + \sum_{y\in S_2}\min_{x\in S_1}\|x-y\|_2$
- Earth Mover’s Distance
  要求两个点集点数必须完全相等
  $d_{EMD}(S_1,S_2) = \min_{\phi:S_1\to S_2}\sum_{x\in S_1}\|x - \phi(x)\|_2$
## Learning on Parametric

用多块 2D 平面参数面片，通过多个 MLP 映射形变到 3D 空间，分块拼接形成完整光滑物体曲面 