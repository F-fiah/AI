---
citekey: "dehdashtianPolyJuiceMakesIt"
title: "PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors"
authors: "Sepehr Dehdashtian, Mashrur M Morshed, Jacob H Seidman, Gaurav Bharaj, Vishnu Naresh Boddeti"
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
# PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors

**PDF**: [bing-dual](file://C:\Users\17934\Zotero\storage\WM243PMG\Dehdashtian%20等%20-%20PolyJuice%20Makes%20It%20Real%20Black-Box,%20Universal%20Red%20Teaming%20for%20Synthetic%20Image%20Detectors.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [bing-dual](zotero://select/library/items/WM243PMG) · [在线打开]()
## Notes

%% begin research_v1 %%

### 速读

> 核心判断：是否值得精读

- 研究问题：构造无需访问目标检测器内部参数、且不必为每张图像重新优化的通用对抗攻击方法
- 核心方法：通过学习生成模型潜空间中与检测器误判相关的方向，在图像生成过程中施加引导，使合成图像更容易被检测器错误识别为真实图像
  **不直接修改已经生成的图像，而是利用检测器历史误判样本揭示的潜空间规律，改变后续图像的生成轨迹**
- 关键结论：
- 精读决策：
	- [x] 是
	- [ ] 否  
### 核心拆解

PolyJuice 利用检测器的二值预测标签，通过监督主成分分析（SPCA）寻找生成模型潜在空间中与检测结果相关的方向，并在扩散或流匹配生成过程中施加这些方向，使更多合成图像被误判为真实图像
#### 背景与缺口

- 领域现状：现有的SID(合成图像检测器)对不同模型的泛化能力不稳定，且可能存在被系统性利用的误判区域
- Research Gap：
- 研究目标：寻找一种不依赖具体图像或文本提示词的通用潜在空间变换，可以使大量生成图像更容易绕过检测器
#### 方法与设计

**通过潜在空间与检测器预测标签之间的统计依赖关系寻找空间变换方向**

实验发现：将生成过程中的 latent 按检测结果分组后，被识别为 Fake 的样本与被误判为 Real 的样本在某些投影方向上呈现明显分布差异。这说明存在某个潜在空间方向，使生成图像的检测结果与沿该方向的变化存在统计关联。

---

定义一个文生图模型：
$$G:\mathcal Z\times\mathcal C\rightarrow\mathcal X$$
其输入包括：
- $z\sim p(z)$：随机潜在变量
- $c\sim p(c)$：文本提示词
- $x=G(z,c)$：最终生成的图像。

设目标检测器输出合成概率：$f:\mathcal X\rightarrow[0,1]$

阈值为 $\tau$，检测规则为：
$$\hat y(x)= \begin{cases} 0,&f(x)<\tau \quad(\text{Real})\\ 1,&f(x)\geq\tau \quad(\text{Fake}) \end{cases}$$
希望寻找潜在空间变换 $h_\Theta$，使经过变换的潜在变量更容易生成被判断为 Real 的图像，即：
$$\max_{\Theta}\; \mathbb E_{\substack{z\sim p(z)\\c\sim p(c)}} \left[ \mathbb 1\left\{ f(G(h_\Theta(z),c))<\tau \right\} \right] $$

---

#### 结果与讨论


%% end research_v1 %%

## Annotations

%% begin annotations %%


### Imported: 2026-10-08 5:34 下午


> [!quote]
> 红队测试通过识别和利用SIDs的失败模式(通 过误分类的合成图像)来提高其有效性 [jump to](zotero://open-pdf/library/items/WM243PMG?page=&annotation=CI7UMRQ4)



%% end annotations %%


%% Import Date: 2026-10-08T17:34:28.771+08:00 %%
