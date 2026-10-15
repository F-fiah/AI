---
citekey: zhuTransferableDualDomainFeature2025
title: Transferable Dual-Domain Feature Importance Attack against AI-Generated Image Detector
authors: Weiheng Zhu, Gang Cao, Jing Liu, Lifang Yu, Shaowei Weng
date: 2025-11
DOI: 10.48550/arXiv.2511.15571
tags:
  - 文献笔记
期刊: ""
期刊简称: ""
类别: preprint
languages: en
url: http://arxiv.org/abs/2511.15571
status: todo
---
# Transferable Dual-Domain Feature Importance Attack against AI-Generated Image Detector

**PDF**: [bing-dual](file://C:\Users\17934\Zotero\storage\Y829GUPA\Zhu%20等%20-%202025%20-%20Transferable%20Dual-Domain%20Feature%20Importance%20Attack%20against%20AI-Generated%20Image%20Detector.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [bing-dual](zotero://select/library/items/Y829GUPA) · [在线打开](http://arxiv.org/abs/2511.15571)
## Notes

%% begin research_v1 %%

DuFIA 通过空间域插值梯度和频域随机扰动，分别估计代理检测器中间层的特征重要性，将两种重要性图融合后构造新的中间层攻击损失，再采用动量迭代梯度方法优化输入图像，以提高对抗样本在不同 AIGC 图像检测器之间的迁移能力。
### 背景与缺口

1. 已有对抗攻击的迁移性不足
2. 普通分类攻击主要关注最终输出层的交叉熵，但是最后的分类头往往与特定模型、训练数据以及任务决策边界绑定，中间层特征包含纹理、局部结构、语义等信息，其中一部分可能被不同架构共同利用。所以可以直接优化模型中间特征。
3. 现有研究只关注空间域，没有充分利用 AIGC 检测中的频谱信息
### 方法与设计

![[Pasted image 20261010170124.png]]
### 结果与意义

### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-10T16:55:32.062+08:00 %%
