---
citekey: zhouStealthDiffusionEvadingDiffusion2024
title: "StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Model"
authors: Ziyin Zhou, Ke Sun, Zhongxi Chen, Huafeng Kuang, Xiaoshuai Sun, Rongrong Ji
date: 2024-10
DOI: 10.1145/3664647.3681535
tags:
  - 文献笔记
期刊: ""
期刊简称: ""
类别: conferencePaper
languages: en
url: https://dl.acm.org/doi/10.1145/3664647.3681535
status: todo
---
# StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Model

**PDF**: [Full Text PDF](file://C:\Users\17934\Zotero\storage\DGE5MS7M\Zhou%20等%20-%202024%20-%20StealthDiffusion%20Towards%20Evading%20Diffusion%20Forensic%20Detection%20through%20Diffusion%20Model.pdf) · **Zotero**: [Full Text PDF](zotero://select/library/items/DGE5MS7M) · [在线打开](https://dl.acm.org/doi/10.1145/3664647.3681535)
## Notes

%% begin research_v1 %%

StealthDiffusion 提出一种面向 AI 生成图像取证检测器的扩散式对抗重建方法：首先通过 PGD 和潜在空间对抗优化（LAO）改变检测器敏感特征，再利用带有跨层特征融合的 Control-VAE 和真实图像噪声频谱原型损失（NPL），降低合成图像与真实图像之间的频谱差异，从而改善攻击迁移性并尽量保持图像视觉质量
### 背景与缺口

- 传统像素级攻击容易引入异常扰动
- 对抗攻击的泛化能力较弱
- 生成器容易在高频噪声残差中留下规律性的频谱结构
### 方法与设计

![[Pasted image 20261010160926.png]]

1. 对于一张合成的虚假图像，先使用 PGD 产生一个经过小幅扰动的图像$$x_{k+1}=\text{Clip}\big[x_k+\eta\,\text{sign}\big(\nabla_x \mathcal{L}(S(x_k),y_{\text{true}})\big)\big]$$
2. 将 PDG 处理后的图像经过 VAE Encoder，得到图像在更低空间分辨率的潜在表征
3. 加入随机噪声，然后调用一个已经预训练好的扩散模型，进行降噪
4.  Control-VAE 解码
### 结果与意义

### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-10T15:30:26.688+08:00 %%
