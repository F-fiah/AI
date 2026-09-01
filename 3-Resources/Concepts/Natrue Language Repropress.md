---
date: 2026-05-24
status: reviewed
---
# Tokenization

神经网络的处理和学习能力基于数学运算，这些都需要输入数值
So，需要将文本转换为 token

- character tokenization：减少了模型的参数数量，但单个字符往往缺乏显著的含义
- word tokenization：单词具有丰富的语义，但数量过多会导致训练过程很慢
- subword tokenization：词缀，a balance between the two
# 词嵌入 (word2vec)

独热向量：
- 维数过大，训练效率很低
- 独热编码中所有词元在向量空间中的相互距离相等，即词元之间没有相似性的概念

词嵌入：将单词压缩到地位的向量空间来降低单词，从而有效捕捉单词之间的上下文
利用词元出现
利用词元出现的上下文来学习他妈的嵌入，从而捕捉上下文信息
## 跳元模型

核心假设：一个词可以用来在文本序列中生成其周围的词

用 $\mathbf{v}_i \in \mathbb{R}^d$ 和 $\mathbf{u}_i \in \mathbb{R}^d$ 表示其用作中心词和上下文词时的两个向量
给定中心词 $w_c$（词典中的索引 $c$），生成任何上下文词 $w_o$（词典中的索引 $o$）的条件概率可以通过对向量点积的 $\mathrm{softmax}$ 来计算：
$$
P(w_o \mid w_c)=\frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{i\in\mathcal{V}}\exp(\mathbf{u}_i^\top \mathbf{v}_c)},
$$
若假设上下文词是在给定任何中心词的情况下独立生成的，则对于上下文窗口 $m$，跳元模型的似然函数是在给定任何中心词的情况下生成所有上下文词的概率：
$$
\prod_{t=1}^{T}\prod_{-m\le j\le m,\,j\ne 0} P(w^{(t+j)} \mid w^{(t)}),
$$
## 连续词袋模型

假设：中心词是基于文本序列中周围上下文的词生成的
$$
P(w_c \mid w_{o_1},\dots,w_{o_{2m}})=
\frac{\exp\left(\frac{1}{2m}\mathbf{u}_c^\top \left(\mathbf{v}_{o_1}+\dots+\mathbf{v}_{o_{2m}}\right)\right)}
{\sum_{i\in\mathcal{V}} \exp\left(\frac{1}{2m}\mathbf{u}_i^\top \left(\mathbf{v}_{o_1}+\dots+\mathbf{v}_{o_{2m}}\right)\right)}
$$
对于上下文窗口$m$，连续词袋模型的似然函数是在给定其上下文词的情况下生成所有中心词的概率：
$$
\prod_{t=1}^{T} P\big(w^{(t)} \mid w^{(t-m)},\dots,w^{(t-1)},w^{(t+1)},\dots,w^{(t+m)}\big).
$$
## GloVe

利用整个语料库中的统计信息进行词嵌入

用 $q_{ij}$ 表示词 $w_j$ 的条件概率 $P(w_j \mid w_i)$，在跳元模型中给定词 $w_i$，有：
$$
q_{ij} = \frac{\exp(\boldsymbol{u}_j^\top \boldsymbol{v}_i)}{\sum_{k \in \mathcal{V}} \exp(\boldsymbol{u}_k^\top \boldsymbol{v}_i)},
$$
词 $w_i$ 在整个语料库中可能出现多次，所有以 $w_i$ 为中心词的上下文词形成一个词索引的多重集，记多重集中的元素的重数为 $x_{ij}$ 

则损失函数优化为：
$$
-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij}\log q_{ij}
$$
# 字词嵌入

使用词的形态信息
## fastText

字词级跳元模型，每个中心词由其自此级向量之和表示
$$ \boldsymbol{v}_w = \sum_{g\in G_w} \boldsymbol{z}_g $$
$G_w$：在给定条件约束内，词 w 的所有字词的并集；$g$：词表 $G_w$ 的一个向量