---
citekey: zhouBreakingLatentPrior
title: Breaking Latent Prior Bias in Detectors for Generalizable AIGC Image Detection
authors: Yue Zhou, Xinan He, Kaiqing Lin, Bin Fan, Feng Ding, Bin Li
date: ""
DOI: 10.52202/085713-1030
tags:
  - 文献笔记
期刊: ""
期刊简称: ""
类别: journalArticle
languages: en
url: ""
status: todo
---
# Breaking Latent Prior Bias in Detectors for Generalizable AIGC Image Detection

**PDF**: [bing-dual](file://C:\Users\17934\Zotero\storage\VZDNMZLV\Zhou%20等%20-%20Breaking%20Latent%20Prior%20Bias%20in%20Detectors%20for%20Generalizable%20AIGC%20Image%20Detection.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [bing-dual](zotero://select/library/items/VZDNMZLV) · [在线打开]()
## Notes

%% begin research_v1 %%

部分 AIGC 检测器会依赖由扩散模型初始随机噪声 $z_T$ 影响的非稳健特征；作者通过优化初始噪声来构造仍由真实生成流程产生的困难假样本，再使用这些样本进行流形上**对抗训练**（OMAT），显著提高 ResNet50 和 CLIP 检测器识别未知生成器的能力，而无需设计新的检测网络架构
### 背景与缺口

1. 传统 AIGC 检测存在 Shortcut Learning，泛化性较差
2. 即使固定生成模型的参数和文本条件，改变初始随机噪声也可能使检测器的判断翻转
### 方法与设计

OMAT 是一种训练策略，而不是新的图像检测网络，其包含两部分：
1. OMAT 是一种训练策略，而不是新的图像检测网络
2. 优化检测器参数，使这些困难假样本重新被正确判断为 Fake

---

1. 在某一数据集上训练普通检测器
2. **固定检测器与生成器，优化初始噪声**
	损失函数：$\mathcal{L}_{\mathrm{adv}}(z_T) = \ell\left(D_{\phi_0}\big(G_\theta(z_T, c)\big), 0\right)$
	其中：
	- $D_{\phi_0}$：冻结的检测器
	- $G_\theta$：冻结的 Stable Diffusion
	- $c$：固定文本提示词
	- $z_T$：唯一被优化的输入变量
	- $\ell$：BCEWithLogits 分类损失
	- 目标标签 $0$：Real
3. 生成困难假样本，并使用困难假样本进行 OMAT
### 结果与意义

### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-10T17:31:35.172+08:00 %%
