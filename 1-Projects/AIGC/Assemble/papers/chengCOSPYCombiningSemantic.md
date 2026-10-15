---
citekey: "chengCOSPYCombiningSemantic"
title: "CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI"
authors: "Siyuan Cheng, Lingjuan Lyu, Zhenting Wang, Xiangyu Zhang, Vikash Sehwag"
date: ""
DOI: "10.1109/CVPR52734.2025.01256"
tags:

- 文献笔记
期刊: ""
期刊简称: ""
类别: "journalArticle"
languages: "en"
url: "https://doi.org/10.1109/CVPR52734.2025.01256"
status: todo
---
# CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI

**PDF**: [bing-dual](file://C:\Users\17934\Zotero\storage\HRQDN4AR\Cheng%20等%20-%20CO-SPY%20Combining%20Semantic%20and%20Pixel%20Features%20to%20Detect%20Synthetic%20Images%20by%20AI.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [bing-dual](zotero://select/library/items/HRQDN4AR) · [在线打开](https://doi.org/10.1109/CVPR52734.2025.01256)
## Notes

%% begin research_v1 %%

CO-SPY 发现语义检测器与像素伪影检测器具有不同的泛化优势和弱点，因此分别使用大型预训练 CLIP、特征插值以及 VAE 重建残差增强两类特征，再通过可学习的动态权重融合，实现对未知生成器、未知图像内容和有损压缩图像更稳定的 AI 生成检测。
### 背景与缺口

1. 现有的检测器存在缺陷：
	- 语义特征检测器
	  使用预训练模型或深度视觉特征判断图像真假。不依赖于具体某个像素的精确值，较稳定；但可能学习了训练集中与 Fake 标签相关的内容或生成风格，而没有充分学到跨生成器稳定的来源特征
	- 伪影特征检测器
	  用生成过程引入的局部纹理和像素统计异常。具有较好的跨模型泛化能力，但容易受到 JPEG、缩放或模糊等操作影响
2. 不同检测器简单的融合只能获得有限的互补收益
### 方法与设计

**先提高两个分支各自的泛化能力，再让系统根据具体输入，动态调整两类特征的贡献，通过融合实现综合收益**

![[Pasted image 20261014174319.png]]
1. 基于 VAE 重建残差的伪影提取
    残差：原始图像和VAE重建图像之间的差异
    VAE 模型是预训练的，需要训练一个 CNN 模型，识别是真实图片的残差模式 or 生成图片的残差模式
2. Enhanced Semantic Features
    一些语义检测器容易记住训练集中常见的生成内容或特定生成器风格
    在 CLIP 特征空间进行插值，从而让模型学习两类特征之间的过渡区域
    这样做能够起到正则化作用，降低模型对某些特定样本的过度自信（过拟合）
3. Adaptive Fusion
    *经过 JPEG 压缩后，局部像素伪影可能受到很大破坏此时，系统应该有能力更加依赖相对稳定的语义表示；反过来，当图片来自未知生成器时，原有语义特征可能不再可靠，而某些生成器共有的残差统计仍然有用*
    
    希望找到一个可变的融合方式，根据输入调整两个检测方式各自的权重
    
    设计两个可学习的模块，根据分支的特征得出权重系数
    ![[Pasted image 20261014181148.png]]
    
### 结果与意义

### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-14T17:25:36.040+08:00 %%
