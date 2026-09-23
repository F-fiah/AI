---
date: 2026-08-19
cssclasses:
  - AIGC
status: in-progress
---
# 文献阅读

- text-span : [[INTERPRETING CLIP’S IMAGE REPRESENTATION VIA TEXT-BASED DECOMPOSITION]]
- ViT : [[Transformers for Image Recognition at Scale]]
- clip : [[Learning Transferable Visual Models From Natural Language Supervision]]
- [[Pruning the Paradox How CLIP's Most Informative Heads Enhance Performance While Amplifying Bias]]
- 泛化性：[[Forensic Rewiring Circuit-Level Shortcut Suppression for Generalizable Deepfake Detection]]
- 模型适配：[[C2P-CLIP Injecting Category Common Prompt in CLIP to Enhance  Generalization in Deepfake Detection]]
- 模型适配：[[Generalizable Deepfake Detection via Simplicity-Bias-Aware  CLIP Adaptation]]
# Concepts

![[54ba2ad98adb703873c6a4d0c16e610b.jpg]]
## ViT-L/14
| 参数                     | 数值                            |
| ---------------------- | ----------------------------- |
| **图像分辨率**              | 224 × 224（预训练）                |
| **Patch 尺寸**           | 14 × 14                       |
| **Transformer 层数 (L)** | **24 层**                      |
| **隐藏层维度 (D)**          | **1024**                      |
| **注意力头数 (H)**          | **16**                        |
| **Patch 数量 (N)**       | 256                           |
| **总 Token 数**          | 256（图像Patch）+ 1（CLS）= **257** |
| **最终嵌入维度**             | **768**（投影后）                  |
### Preprocess

1. **Resize：** 将图片的短边缩放到 224，长边按比例缩放
2. **CenterCrop：** 从缩放后的图片中心裁剪出 224×224 的正方形区域
### ViT

1. 对图像按 patch 进行切割，得到 256 个 `[14, 14, 3]` 的 patch
2. 将每个 patch 展平为向量 `[256, 588]`，并通过一个**可学习的线性投影矩阵** `E` 映射到模型的隐藏维度 `[256, 1024]`
3. 插入 cls 向量
4. Position Embedding：$z_0 = \left[x_{\text{class}}; x_1; \dots; x_{256}\right] + E_{\text{pos}}$
### Train

1. Batch 中的 N 张图片通过 ViT 得到 N 个 768 维图像向量；N 段文本通过 Text Transformer 得到 N 个 768 维文本向量
2. 计算所有 N×N 个图文对的余弦相似度
3. **对称交叉熵损失**：对角线上的 N 个配对应被拉近（相似度接近 1），非对角线上的 N²-N 个错误配对应被推远（相似度接近 0）
4. 反向传播更新参数
# Result Analysis
## Ablation
$$
M_{\text{image}}(I) = \underbrace{P[Z^0]_{\text{cls}}}_{\text{初始嵌入}} + \underbrace{\sum_{l=1}^{24} P[\text{MSA}^l]_{\text{cls}}}_{24\text{ 个 attention 项}} + \underbrace{\sum_{l=1}^{24} P[\text{MLP}^l]_{\text{cls}}}_{24\text{ 个 MLP 项}}
$$
其中：
$$
P[\text{MSA}^l]_{\text{cls}} = \sum_{h=1}^{16}\sum_{i=0}^{256} c_{i,l,h}
$$
ViT 是残差网络，**第 24 层的 CLS token 本身就包含了从第 0 层到第 24 层所有层累加贡献的总和**
这个公式本身就是在重建 CLS

消融的本质：把某几个加数从"真实值"换成"训练集均值"，其余加数保持不动，得到一个新的"伪 CLS 表示"
### 逐层消融

将 1~n 层的输出改为平均输出，计算分类精度

单层消融可能被其他层补偿（因为信息冗余），看不出重要性；累积消融无补偿机会（被消融的层越来越多，冗余空间被逐步压缩）

在训练阶段，记录每个组件在所有图像上的均值输出
- Attention 消融
	累积消融到第 k 层时，将前 k 层的 attention 输出都替换为其处在的层的 16 个 head 的平均输出之和
- MLP 消融
	累积消融到第 k 层时，将前 k 层的 mlp 输出替换为其在训练集上的平均输出
### 单头消融

将某个 head 的输出替换为其在训练集上的平均输出
## 线性可分

**如果只看单个注意力头（head）对 CLS token 的贡献，这个头内部是否独立包含了足以区分真伪的线性信息？**

使用每个头单独提取训练集特征，训练一个独立的线性探针。在测试集上，用该头自己训练的探针，评估该头特征的 ACC / AUC
## TextSpan 算法

Goal：给定一个注意力头在 $K$ 张图片上的输出向量 $C∈R^{K × d'}$，以及一个包含 M 个文本描述的候选池，找出最能解释该头 m 个文本描述

算法核心：用文本向量张成的子空间，去尽可能多地捕捉该头在图片间的变化
$$
V_{\text{explained}}(\mathcal{T}) = \frac{1}{K}\sum_{k=1}^{K}\left\|\text{Proj}_{\mathcal{T}}(c_k - c_{\text{avg}})\right\|_2^2
$$
## 特征重构

Goal：将 ViT 提取得到的 768 维向量 c 投影到由 m 个文本嵌入张成的子空间上
$$ \text{Proj}_{\mathcal{T}}(c) = R_{\mathcal{T}} \cdot \underbrace{\left(R_{\mathcal{T}}^T R_{\mathcal{T}}\right)^{-1} \cdot R_{\mathcal{T}}^T \cdot c}_{\text{权重系数 }\beta\text{ (m维)}} $$
本质：在由这 `m` 个文本方向张成的子空间里，找到离 `c` 最近的向量

单个 head 的 TextSPAN 过程：
```
原始头输出 C [N, 768]
    │
    ├─ SVD 降维 → 找到数据有效变化的主成分子空间
    │
    ├─ 文本嵌入投影到主成分子空间
    │
    ├─ 数据中心化（减均值）
    │
    ├─ 初始化重构为均值
    │
    └─ 贪心迭代 m 次：
          │
          ├─ 在残余空间中找方差最大的文本方向
          │
          ├─ 把该方向的贡献加到重构结果
          │
          ├─ 从残余中减去该方向（数据侧正交化）
          │
          └─ 从文本字典中减去该方向（字典侧正交化）
    │
    ▼
重构头输出 reconstruct [N, 768]
```
整体流程：
```
输入：
  384 个头的原始贡献          attns[l, h]  每个 [N, 768]
  25 个 MLP 块的原始贡献      mlps[b]      每个 [N, 768]
       │
       │ 每个头独立运行 TextSPAN
       ▼
中间层：
  384 个头的重构贡献          attns_recon[l, h]  每个 [N, 768]
  25 个 MLP 块的原始贡献      mlps[b]            每个 [N, 768]  （MLP 不重构）
       │
       │ 加总（aggregate）
       ▼
最终层：
  完整图像的重构特征          [N, 768]
```