---
citekey: xieTakeFakeReal
title: "Take Fake as Real: Realistic-like Robust Black-box Adversarial Attack to Evade AIGC Detection"
authors: Caiyun Xie, Dengpan Ye, Yunming Zhang, Long Tang, Yunna Lv, Jiacheng Deng, Jiawei Song
date: 2026-10-09
DOI: 10.1145/3785353.3815075
tags:
  - 文献笔记
期刊: IEEE Transactions on Circuits and Systems for Video Technology，2026，36(5)：6347–6360
期刊简称: TCSVT
类别: journalArticle
languages: en
url: https://dl.acm.org/doi/10.1145/3785353.3815075
status: todo
---
# Take Fake as Real: Realistic-like Robust Black-box Adversarial Attack to Evade AIGC Detection

**PDF**: [bing-dual](file://C:\Users\17934\Zotero\storage\AMGACM8V\Xie%20等%20-%20Take%20Fake%20as%20Real%20Realistic-like%20Robust%20Black-box%20Adversarial%20Attack%20to%20Evade%20AIGC%20Detection.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [bing-dual](zotero://select/library/items/AMGACM8V) · [在线打开]()
## Notes

%% begin research_v1 %%

### 速读

- 研究问题：
- 核心方法：
- 关键结论：
- 精读决策：
	- [x] 是
	- [ ] 否  
### 核心拆解

R²BA 将高斯模糊、JPEG 压缩、高斯噪声和光斑四类真实图像后处理组合成一个八维参数空间，通过带惯性衰减和随机扰动的粒子群优化（PSO），仅根据目标检测器输出的 Fake 概率寻找有效的后处理组合，从而使合成图像被误判为真实图像，并尽可能保持视觉质量和后续处理下的攻击有效性
#### 背景与缺口

1. 像素级扰动方法可能损伤图像视觉质量；即使能欺骗检测器，也不一定适合真实图像传播场景
2. 某些扰动对压缩、重采样等后续处理比较敏感，攻击图像经过平台重新编码后，效果可能明显下降
#### 方法与设计

实验发现，后处理之间存在非简单叠加关系，从而引出了引入参数融合优化的攻击方法
$$\mathbf p= (\alpha,\beta,\phi,\sigma, \lambda_x,\lambda_y,\theta,\gamma)$$

| 参数                    | 所属操作           | 含义           |
| --------------------- | -------------- | ------------ |
| $\alpha$              | Gaussian Blur  | 模糊核尺寸或半径相关参数 |
| $\beta$               | Gaussian Blur  | 高斯模糊标准差      |
| $\phi$                | JPEG           | 压缩质量参数       |
| $\sigma$              | Gaussian Noise | 高斯噪声方差       |
| $\lambda_x,\lambda_y$ | Light Spot     | 光斑中心位置       |
| $\theta$              | Light Spot     | 光斑亮度         |
| $\gamma$              | Light Spot     | 光斑半径         |
#### 结果与意义

1. 从任意像素扰动转向结构化后处理搜索，将攻击限制到四种自然图像变换构成的参数化空间，其特点是优化维度很小，并且每一种修改都具有明确的图像处理意义
2. 利用粒子群优化寻找多种后处理的有效组合
#### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-09T15:33:56.483+08:00 %%
