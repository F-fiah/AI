---
date: 2026-08-27
status: in-progress
---
# n-gram

每个词的概率仅由前 n-1 个决定
# RNN

当前输入+包含前置序列全部信息的潜向量
参考 [[神经网络#循环神经网络（Reccurrent Neural Networks）]]
## LSTM

隐状态 $\boldsymbol{h}^{(t)}$ + 单元状态 $\boldsymbol{c}^{(t)}$（用于储存长距离信息）

- 遗忘门（forget gate），控制上一个单元状态中的哪些信息被保留，哪些信息被遗忘：
$$
\boldsymbol{f}^{(t)} = \sigma(\boldsymbol{W}_f \boldsymbol{h}^{(t-1)} + \boldsymbol{U}_f \boldsymbol{x}^{(t)} + \boldsymbol{b}_f)
$$
- 输入门（input gate），控制哪些信息被写入单元状态：
$$
\boldsymbol{i}^{(t)} = \sigma(\boldsymbol{W}_i \boldsymbol{h}^{(t-1)} + \boldsymbol{U}_i \boldsymbol{x}^{(t)} + \boldsymbol{b}_i)
$$
- 输出门（output gate），控制单元状态中的哪些信息被写入隐状态：
$$
\boldsymbol{o}^{(t)} = \sigma(\boldsymbol{W}_o \boldsymbol{h}^{(t-1)} + \boldsymbol{U}_o \boldsymbol{x}^{(t)} + \boldsymbol{b}_o)
$$
- 新的单元内容，即待写入单元的新信息：
$$
\tilde{\boldsymbol{c}}^{(t)} = \tanh(\boldsymbol{W}_c \boldsymbol{h}^{(t-1)} + \boldsymbol{U}_c \boldsymbol{x}^{(t)} + \boldsymbol{b}_c)
$$
- 单元状态，通过擦除（遗忘）上一个单元状态中的部分信息并写入部分新的信息而获得：
$$
\boldsymbol{c}^{(t)} = \boldsymbol{f}^{(t)} \odot \boldsymbol{c}^{(t-1)} + \boldsymbol{i}^{(t)} \odot \tilde{\boldsymbol{c}}^{(t)}
$$
- 隐状态，其内容是从单元状态中输出的一部分信息：
$$
\boldsymbol{h}^{(t)} = \boldsymbol{o}^{(t)} \odot \tanh \boldsymbol{c}^{(t)}
$$
# Transformer

每个词的概率通对前置所有词使用注意力机制得到
参考 [[3-Resources/Concepts/Attention and Transformers|Attention and Transformers]]

RNN 无法很好地建模长距离依赖
1. 如果序列中的第 $i$ 个词需要对第 $j$ 个词产生影响，需经过 $j-i$ 个计算步骤，而随着步数增加，第 $i$ 个词的信息会很快衰减
2. 每一步用来预测下一个词的隐状态都需要包含这个词左边所有词的信息，但隐状态的维度有限，所能表达的信息容量也有限
3. RNN 需要按顺序处理信息，无法并行运算，速度受到运算
---
**多头注意力**

将 $\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}$ 映射到不同维度的空间中，分别使用注意力机制
使得词之间可以通过多种不同的方式进行交互

---
Transformer：将 RNN 中相邻隐状态之间的连接完全去除，只保留注意力机制
## 缩放点积注意力
![[7b2e1520d67a1c8707ff27c485fe4b98.jpg|526]]
