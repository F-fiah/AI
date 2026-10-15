---
citekey: chenFBA$^2$DFrequencybasedBlackbox2025
title: "FBA$^2$D: Frequency-based Black-box Attack for AI-generated Image Detection"
authors: Xiaojing Chen, Dan Li, Lijun Peng, Jun Yan Zhiqing Guo, Junyang Chen, Xiao Lan, Zhongjie Ba, Yunfeng Diao
date: 2025-01
DOI: 10.48550/ARXIV.2512.09264
tags:
  - 文献笔记
期刊: ""
期刊简称: ""
类别: preprint
languages: en
url: https://arxiv.org/abs/2512.09264
status: todo
---
# FBA$^2$D: Frequency-based Black-box Attack for AI-generated Image Detection

**PDF**: [FBA$^2$D-bing-dual](file://C:\Users\17934\Zotero\storage\ZEVUZ22M\Chen%20等%20-%202025%20-%20FBA$^2$D%20Frequency-based%20Black-box%20Attack%20for%20AI-generated%20Image%20Detection.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [FBA$^2$D-bing-dual](zotero://select/library/items/ZEVUZ22M) · [在线打开](https://arxiv.org/abs/2512.09264)
## Notes

%% begin research_v1 %%

FBA²D 根据真实图像与合成图像的频域差异选择不同的扰动子空间，并利用代理模型产生的 Adversarial Example Soup 改善初始化，从而在仅有二值预测标签和有限查询预算的情况下，寻找视觉扰动较小的对抗图像

---
**图像的频域表示**

图像的频率描述的是图像像素随着空间位置变化的快慢
$\text{空间频率越高} \Rightarrow \text{单位空间内明暗变化越快}$

对于一行像素，可以用一组不同快慢的正弦波、余弦波进行表示（傅里叶变换）
$$f(x)= A_0+ A_1\cos(2\pi\nu_1x+\phi_1) + A_2\cos(2\pi\nu_2x+\phi_2) +\cdots$$
$[A_0,A_1,A_2,\ldots]$ 称为频域

对于二维矩阵，则可以使用两个频域（水平方向+竖直方向）
==🟡任何一张固定大小的图像，都能由大量不同的二维余弦图案按一定系数相加得到==
例如：
$$I= c_{0,0}B_{0,0} +c_{1,0}B_{1,0} +c_{0,1}B_{0,1} +\cdots$$
其中：
- $B_{0,0}$：均匀明暗的图案
- $B_{1,0}$：沿某一方向缓慢变化的图案
- $B_{0,1}$：沿另一方向缓慢变化的图案
- $c_{u,v}$：每个图案的权重，组成 DCT 矩阵

---
### 背景与缺口

1. 现有的攻击对检测器的信息要求过高，有时候攻击者只能看到二值决策，无法得到预测分数，在这样情况下 [[xieTakeFakeReal]] 的 R$^2$BA 的优化方法就不再适用
   *只有二值决策，能否仍然高效地找到对抗扰动？*
2. 传统决策型攻击多在 ImageNet 等图像分类任务中研究，在 AIGC 领域内研究较少
### 方法与设计

设输入图像 $x$ 原本被正确分类，希望找到扰动 $\delta$，满足：
$$\min_{\delta}\|\delta\|_p \quad \mathrm{s.t.} \quad c(x+\delta)\neq c(x)$$
**核心思路：先找一个确定属于相反类别的样本，再不断缩短它与原始图像之间的距离，同时维持攻击成功**

1. 初始化一个成功对抗样本
    给定一个 fake 原始图像，寻找一个 real 图像
    - 直接使用 real 类别的图像
	    优点是容易获得一个位于目标类别区域的样本
	    但它可能在视觉上与原图完全不同，因此初始扰动距离很大
	- 使用 Adversarial Example Soup
		==利用代理模型生成比较接近原始图像的对抗候选，再验证它是否能够迁移到目标检测器==
2. 对原始图像和初始对抗图像进行 DCT
3. 根据经验选取特定的频率，加入扰动（扰动的变化程度要小于对抗图像和原始图像之间的距离）
4. 不断重复加入扰动，直到满足相应的停止条件

$$\boxed{ \text{成功对抗点} \xrightarrow{\text{频域几何搜索}} \text{更靠近原图的候选} \xrightarrow{\text{标签验证}} \text{接受或重新搜索} }$$
### 结果与意义

1. 有二值标签的 AIGC 检测器仍可能受到高效对抗攻击
2. 利用真实与合成图像在频率统计上的差异，可以缩小每轮攻击所探索的区域，从而在部分模型上显著提升攻击成功率
### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-10T12:36:48.374+08:00 %%
