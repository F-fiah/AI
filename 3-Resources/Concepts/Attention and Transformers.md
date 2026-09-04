---
cssclasses:
  - 深度学习
  - 计算机视觉
date: 2026-07-11
---
![[RNN-disad.png]]
RNN 的问题：对于任意数据集，RNN最终总结出来的向量c 的维度时固定的，当遇到很大的数据集时，c 的大小可能不够

Solution：look back at the whole input sequence on each step of the output
# Attention
## Cross-Attention

解码器每一步 t，动态计算一组权重，选择性聚焦编码器所有时间步的隐状态，生成专属当前步的上下文向量 $c_t$

1. 对齐分数 (Alignment Scores)：
   $e_{t,i} = f_{att}(s_{t-1},h_i)$，表示当前时间 t 要生成的词，与源语言第 i 个词的相关程度
2. 将 $e_{t,i}$ 通过 softmax 函数转化成概率 $a_{t,i}$
3. 生成上下文向量 (Context Vector)：
   $c_t = \sum_i a_{t,i} h_i$

常见处理方式：QAV

- $Q \in \mathbb{R}^{N_Q \times D_Q}$ Query 查询矩阵
    代表现在要检索的信息，对应解码器状态 $s_{t-1}$
- $X \in \mathbb{R}^{N_X \times D_X}$ 原始数据源矩阵
    对应编码器全部隐状态 $h_1,h_2...$，即原始待检索数据
- $W_K \in \mathbb{R}^{D_X \times D_Q}$
    把原始数据X映射到 Key 空间，维度对齐 Query 用于点积打分
- $W_V \in \mathbb{R}^{D_X \times D_V}$
    把原始数据X映射到 Value 空间，最终加权求和输出的载体
---
- $D_V$ 是人为设定的超参数，即最终 Context Vector 的大小，但必须等于下游解码网络输入维度
- $D_H,D_S$ 分别是编码器和解码器的维度，手动设置
---
1. $K = XW_K$，把原始数据映射到查询匹配空间
   $V = XW_V$，把原始数据映射到输出内容空间
2. 计算相似度矩阵 $E = \frac{QK^\top}{\sqrt{D_Q}}$，其中 $\div \sqrt{D_Q}$ 是为了防止数值爆炸
3. $A = \mathrm{softmax}(E,\ \mathrm{dim}=1)$，值越大，说明这条 Value 对输出贡献越高
4. 输出 $Y = AV$
![[Cross-Attention.jpg|392]]
## Self-Attention

Q、K、V 均全部来自同一份输入序列 X，自己和自己匹配

- $X \in \mathbb{R}^{N \times D_{in}}$
- $W_Q, W_K, W_V \in \mathbb{R}^{D_{in} \times D_{out}}$
$$ Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\quad E = \frac{QK^\top}{\sqrt{D_{out}}} $$
![[Self-Attention.png|272]]

**Problem:** Self-Attention does not know the order of the sequebse
**Solution:** Add positional encoding to each input，$X_i := X_i + \mathrm{PE}(i)$
## Masked Self-Attention

普通自注意力能看到序列全部位置，掩码自注意力强制只能看特定位置

计算相似度矩阵 $E$ 时，将所有 $j > i$ 的位置都赋值为 $−∞$，经过 softmax 后概率变为 0，从而实现屏蔽未来 token 的信息
## Multiheaded Self-Attention

单头注意力只有一套 QKV，它只能侧重一种关联

多头注意力则同时进行多个 QKV 操作，每个单独学习一套语义匹配规则，最终把所有结果起融合输出
- 模型总维度：$D$
- 头数量：$H$
- 单头维度：$D_H = D / H$ 

1. Concat 维度拼接（无参数，单纯堆叠） 
   每个注意力头独立输出 $Y_h \in [N,\ D_H]$ 将 $H$ 个头的输出在特征维度拼接： $$Y_{concat} = \text{Concat}(Y_1,Y_2,...,Y_H)$$把所有头捕捉到的多组语义信息合并到同一个向量
   2. 线性投影融合，投影矩阵 $W_O \in \mathbb{R}^{H D_H \times D}$ $$O = Y_{concat} W_O$$
# Decode
## 使用 RNN 来解码

- $h_i$：Encoder 编码器的隐状态
- $s_t$：Decoder 解码器的隐状态
$$ s_t = g_u(y_{t-1},\, s_{t-1},\, c_t) $$
## 使用 MLP

RNN 虽然能处理序列，但串行计算慢、长距离依赖弱，无法制作成大模型

MLP+位置编码也能处理序列，且全程并行、长文本更强，适合大数据、大模型
# Transformers

**单层 Transformer = 多头自注意力模块 + MLP 前馈模块**

1. 原始输入向量进行 Self-Attention
   $X_{\mathrm{att}} = \mathrm{Input} + \mathrm{MultiHeadAttention}(\mathrm{Input})$
2. 将输出进行归一化
3. 归一后的特征送入多头并行 MLP
	- 自注意力只有线性矩阵运算，模型表达能力不足；MLP 引入非线性，提升拟合能力
	- 注意力只完成「词和词之间信息交换」，MLP 专门独立加工单个 token 内部细粒度语义，细化词性、语义、属性
    - 多头并行 MLP 和多头注意力一一对应，分开处理每组注意力捕捉到的不同维度关联
	$X_{\mathrm{out}} = \mathrm{Norm}\left(X_{\mathrm{att}} + \mathrm{MLP}\left(X_{\mathrm{att}}\right)\right)$
4. 将输出进行标准化，作为下一层 Transformer 的输入

![[transformers.png]]
## Vision Transformers

将一张图像切成固定大小的图像块 Patch，并把每个 Patch 视为 NLP 中的一个 Token
**让这些Patch两两之间做全局自注意力**，从而可以学会 patch 之间的语义关系（不受距离限制）

1. 将图像分割为 patch，利用 CNN 卷积为向量
2. 加上位置编码
3. Transformer Encoder 前向传播，每个 Patch 都变成了蕴含全局信息的向量

Problem：选取哪一个向量作为 “整张图的代表” 去送进分类头
Solution：
- 将所有向量取平均：平均池化会严重稀释的极端特征，导致模型偏向于学习背景纹理
- 取最值：不稳定，聚焦于噪声
**Final solution：CLS**
通过自注意力机制，汇总所有 patch 的信息

