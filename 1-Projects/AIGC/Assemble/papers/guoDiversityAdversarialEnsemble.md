---
citekey: "guoDiversityAdversarialEnsemble"
title: "On the Diversity of Adversarial Ensemble Learning"
authors: "Jun-Qi Guo, Meng-Zhang Qian, Wei Gao, Zhi-Hua Zhou"
date: ""
DOI: ""
tags:

- 文献笔记
期刊: ""
期刊简称: ""
类别: "journalArticle"
languages: "en"
url: "https://proceedings.mlr.press/v267/guo25g.html"
status: todo
---
# On the Diversity of Adversarial Ensemble Learning

**PDF**: [bing-dual](file://C:\Users\17934\Zotero\storage\YN5Z4YU4\Guo%20等%20-%20On%20the%20Diversity%20of%20Adversarial%20Ensemble%20Learning.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [bing-dual](zotero://select/library/items/YN5Z4YU4) · [在线打开](https://proceedings.mlr.press/v267/guo25g.html)
## Notes

%% begin research_v1 %%

作者证明精确计算神经网络对抗集成的多样性具有 NP-hard 计算复杂度，并在一阶泰勒近似下，把集成对抗损失分解为单模型平均对抗损失以及预测多样性、梯度多样性和交叉多样性，随后利用对抗样本上的预测正交正则化训练多个模型，从而提高集成系统的对抗鲁棒性。
### 背景与缺口

传统集成学习研究经常使用模型预测差异衡量多样性；对抗集成研究则经常使用梯度余弦相似度
如果两个梯度接近正交，通常意味着攻击一个模型的局部最陡上升方向与另一个模型不同，因此现有的方法大多尝试降低两个梯度的余弦相似度

本文对传统的观点提出质疑：如果想严谨理解集成系统的对抗鲁棒性，不能把模型的输出和模型的输入梯度割裂开来
### 方法与设计

纯理论推导，过于复杂
### 结果与意义

==**仅让不同模型的梯度方向不一致，并不足以完整描述集成鲁棒性**==
### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-14T20:10:04.034+08:00 %%
