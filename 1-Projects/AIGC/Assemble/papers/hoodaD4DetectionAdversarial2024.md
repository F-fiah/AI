---
citekey: hoodaD4DetectionAdversarial2024
title: "D4: Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles"
authors: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash
date: 2024-01
DOI: 10.1109/WACV57701.2024.00377
tags:
  - 文献笔记
期刊: ""
期刊简称: ""
类别: conferencePaper
languages: en
url: https://ieeexplore.ieee.org/document/10483793/
status: todo
---
# D4: Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles

**PDF**: [D4-bing-dual](file://C:\Users\17934\Zotero\storage\6AURUSB8\Hooda%20等%20-%202024%20-%20D4%20Detection%20of%20Adversarial%20Diffusion%20Deepfakes%20Using%20Disjoint%20Ensembles.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [D4-bing-dual](zotero://select/library/items/6AURUSB8) · [在线打开](https://ieeexplore.ieee.org/document/10483793/)
## Notes

%% begin research_v1 %%

利用 AI 生成图像的判别信息在频域中具有冗余分布这一现象，将 DCT 频率系数划分成互不重叠的子集，分别训练多个检测器，再通过投票进行协作判断，从而降低黑盒攻击者利用同一组脆弱特征同时欺骗所有模型的可能性。

*D4 算法不是通过简单增加模型数量获得鲁棒性，而是主动限制每个模型能够使用的输入特征，试图从结构上减少模型之间共享的攻击脆弱性*
### 背景与缺口

1. 传统的 deepfake 检测器往往依赖少量对真假分类特别有效、但容易受到攻击的特征
    攻击者只要改变这些共同特征，就可能同时欺骗三个模型。所以简单集成不同网络结构，并不一定带来理想的安全收益
### 方法与设计

实验发现：将使用完整图像进行检测的结果与仅保留一部分 RGB 像素的结果进行对比，发现只保留约 6% 的频率系数，模型仍保留相当的真假判别能力；但 RGB 像素随机减少后，检测效果下降更明显

**这说明不同频率子集可能各自包含足够的真假判别信息**

因此，可以把频域划分成多份，让多个检测器分别学习，从而增强检测效果

**核心技巧：梯度正交法**

让每一个模型的损失函数的梯度相互正交，这意味着最能提高模型 1 损失的方向，不会直接提高模型 2 的损失，从而增加了攻击者同时欺骗两个模型的困难。
### 结果与意义

### 局限性与可质疑点

%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-13T15:26:52.720+08:00 %%
