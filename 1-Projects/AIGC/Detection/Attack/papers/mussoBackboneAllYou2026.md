---
citekey: mussoBackboneAllYou2026
title: "Backbone is All You Need: Assessing Vulnerabilities of Frozen Foundation Models in Synthetic Image Forensics"
authors: Chiara Musso, Joy Battocchio, Andrea Montibeller, Giulia Boato
date: 2026-06
DOI: 10.1145/3785353.3815075
tags:
  - 文献笔记
期刊: ""
期刊简称: ""
类别: conferencePaper
languages: en
url: https://dl.acm.org/doi/10.1145/3785353.3815075
status: todo
---
# Backbone is All You Need: Assessing Vulnerabilities of Frozen Foundation Models in Synthetic Image Forensics

**PDF**: [Backbone is All You Need-bing-dual](file://C:\Users\17934\Zotero\storage\53GL8JU2\Musso%20等%20-%202026%20-%20Backbone%20is%20All%20You%20Need%20Assessing%20Vulnerabilities%20of%20Frozen%20Foundation%20Models%20in%20Synthetic%20Image%20F.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [Backbone is All You Need-bing-dual](zotero://select/library/items/53GL8JU2) · [在线打开](https://dl.acm.org/doi/10.1145/3785353.3815075)
## Notes

%% begin research_v1 %%

### 速读

- 研究问题：如果攻击者只知道检测器所使用的预训练视觉骨干网络，是否足以构造有效的对抗图像？
- 核心方法：
- 关键结论：
  提出了 SIAA，一种针对使用冻结 Vision Transformer 骨干网络的合成图像检测器的灰盒攻击方法，通过构建视觉—文本特征对齐的代理头（FP-Head），在完全不知道目标检测器分类头参数的情况下，使用 PGD 生成小幅度像素扰动，使检测器产生错误判断。
- 精读决策：
	- [x] 是
	- [ ] 否  

### 核心拆解
#### 背景与缺口

当前许多 AIGC 图像检测器采用预训练视觉编码器加轻量分类头的架构，这个范式训练资源需求小，且可以利用基础模型强大的视觉表征能力
由于模型都是开源的，攻击者可直接获取与检测器相同的预训练骨干网络参数，这就提供了一条绕开未知分类头的攻击路径
#### 方法与设计

**1. FP-Head**
将图像特征与 CLIP 文本特征映射到同一个空间

使用两个固定的提示词：This image is real 和 This image is fake，将他们通过 CLIP 文本编码器后获得对于的向量，在经过一个预训练的文本投影层，得到两个类别锚点；同时将 ViT 的输出图像表征经过另一个图像投影层，得到图像在该空间上的向量

训练核心：让图像向量尽可能靠近对应类别的锚点，远离相反类别的文本锚点；且Real 和 Fake 两个文本锚点之间保持足够距离

**2. PGD 攻击算法**
保持 ViTal 骨干和 FP-Head 冻结，往图像中加入像素扰动
核心逻辑：根据损失函数对输入像素的梯度，反复决定每个像素应该增加还是减少一点点数值

对于一个 fake 图像 I，经过骨干和视觉投影层得到与文本向量相同的空间的表征，希望加入像素扰动使其远离 Fake 锚点并靠近 Real 锚点，即最大化：
$$\boxed{ \mathcal J(I) = \underbrace{\|v'-t'_1\|_2}_{\text{远离 Fake}} - \underbrace{\|v'-t'_0\|_2}_{\text{靠近 Real}} }$$
#### 结果与意义

实验结果证明，在某些基于公开预训练 ViT 的检测器中，攻击者即使不知道分类头，也能利用相同骨干的特征空间获得很高的攻击成功率

同时，作者还提出了提出带视觉—文本对齐的代理特征处理头：不直接在原始 ViT 特征上攻击，而是额外训练一个 Feature-Processing Head（FP-Head），将图像特征与 CLIP 文本特征映射到同一个空间。在该空间中，图像可以根据 Real/Fake 标签聚集到对应的文本语义锚点周围。
**这样，即使不知道目标分类头的决策函数，也能构造一个有明确攻击方向的代理目标**
#### 局限性与可质疑点


%% end research_v1 %%

## Annotations


%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-09T14:42:52.060+08:00 %%
