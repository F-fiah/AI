---
citekey: "tanAggregatingDiverseCue"
title: "Aggregating Diverse Cue Experts for AI-Generated Image Detection"
authors: "Lei Tan, Shuwei Li, Mohan Kankanhalli, Robby T Tan"
date: ""
DOI: ""
tags:

- 文献笔记
期刊: ""
期刊简称: ""
类别: "journalArticle"
languages: "en"
url: ""
status: todo
---
# Aggregating Diverse Cue Experts for AI-Generated Image Detection

**PDF**: [bing-dual](file://C:\Users\17934\Zotero\storage\PHUCGWDF\Tan%20等%20-%20Aggregating%20Diverse%20Cue%20Experts%20for%20AI-Generated%20Image%20Detection.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [bing-dual](zotero://select/library/items/PHUCGWDF) · [在线打开]()
## Notes

%% begin research_v1 %%

MCAN 同时提取原始图像、DWT 高频细节以及新提出的色度不一致性（Chromaticity Inconsistency, CI）三类取证线索，利用冻结 CLIP 骨干中的动态编码器专家适配器（MoEA）学习不同线索，再通过三个分类头的输出聚合，提高对未知生成模型的检测泛化能力。
### 背景与缺口

现有的 AIGC 检测器存在缺陷：过度依赖特定的生成器伪影，导致跨生成器泛化不足

常见的检测证据为：

| 取证线索   | 主要关注的信息          | 可能的问题             |
| ------ | ---------------- | ----------------- |
| 原始图像   | 图像内容、物体结构、整体视觉表现 | 容易受到图像内容或生成风格偏差影响 |
| 高频细节   | 边缘、局部纹理、部分生成伪影   | 会丢失大量低频语义信息       |
| 重建误差   | 原图与扩散重建图的差异      | 结果依赖具体的重建模型       |
| 色度不一致性 | 颜色通道比例中的局部不规则变化  | 作者希望借此捕捉真实成像噪声    |
**不同取证线索可能具有互补性**

**简单的多模型平均融合存在缺陷，所以提出在不同线索不仅在决策末端融合，还能在特征提取过程中，通过共享模型与专家适配器进行学习**
### 方法与设计

1. 原始图像线索
    将原始图像经过 ViT 得到其表征，再得到一个分类输出
2. 高频线索
    通过通过离散小波变换提取图像的高频成分（希望利用这些高频信息检测生成图像留下的边缘与局部纹理异常）
3. Chromaticity Inconsistency（色度不一致）
    真实照片通常经历的物理成像过程会引入随机噪声，而完全生成的图像并不一定包含与真实传感器相同的噪声形成机制。通过检测这个差异可以进行图像的鉴伪
4. **Mixture-of-Encoder Adapter**
    对进入适配器的每个 Token 计算不同编码专家的混合权重
    
### 结果与意义

### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-14T20:42:47.125+08:00 %%
