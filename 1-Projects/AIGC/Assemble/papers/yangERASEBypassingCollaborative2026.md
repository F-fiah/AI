---
citekey: "yangERASEBypassingCollaborative2026"
title: "ERASE: Bypassing Collaborative Detection of AI Counterfeit via Comprehensive Artifacts Elimination"
authors: "Qianyun Yang, Peizhuo Lv, Yingjiu Li, Shengzhi Zhang, Yuxuan Chen, Zhiwei Chen, Zixu Li, Yupeng Hu"
date: "2026-07"
DOI: "10.1109/TDSC.2026.3677794"
tags:

- 文献笔记
期刊: "IEEE Transactions on Dependable and Secure Computing"
期刊简称: "IEEE Trans. Dependable and Secure Comput."
类别: "journalArticle"
languages: "en"
url: "https://ieeexplore.ieee.org/document/11456581/"
status: todo
---
# ERASE: Bypassing Collaborative Detection of AI Counterfeit via Comprehensive Artifacts Elimination

**PDF**: [ERASE-bing-dual](file://C:\Users\17934\Zotero\storage\GD7IVJF6\Yang%20等%20-%202026%20-%20ERASE%20Bypassing%20Collaborative%20Detection%20of%20AI%20Counterfeit%20via%20Comprehensive%20Artifacts%20Elimination.no_watermark.zh-CN.LR_dual.pdf) · **Zotero**: [ERASE-bing-dual](zotero://select/library/items/GD7IVJF6) · [在线打开](https://ieeexplore.ieee.org/document/11456581/)
## Notes

%% begin research_v1 %%

ERASE 通过“代理模型梯度扰动—带概率门控的扩散重建—低高频解耦修正”三个连续模块，构造既保留原始图像外观、又能同时逃避不同取证原理检测器的 AI 生成图像，揭示**异构检测器联合判定仍存在可迁移的共同失效**。

### 背景与缺口

1. 已有的攻击方法难以同时绕过空间纹理、频谱、重建误差和 CLIP 特征等不同检测机制
2. 现有方法在机器逃避与人类视觉保真难以兼顾

|取证类型|可能使用的信息|单一攻击可能遗漏的问题|
|---|---|---|
|空间特征|局部纹理、CNN 判别特征|代理扰动可能不迁移|
|频域特征|频谱统计、生成器周期性伪影|普通 PGD 不保证频谱自然|
|重建特征|扩散重建误差及统计差异|普通后处理难以改变重建行为|
|基础模型特征|CLIP 等预训练视觉表示|局部频谱修正不保证欺骗语义特征|
不同检测器依赖的信息不同，所以攻击不应只针对一种特征，而应在多个层次上改变图像
### 方法与设计

1. Sensitive Feature Attack（SFA）
    类似于 PGD，往图像中加入扰动，使得 fake 图像被检测成 real 图像（需要知道检测器的参数等信息）
    $\boxed{ x_{k+1} = \mathcal C_\epsilon \left[ x_k- \eta\operatorname{sign} \left( \nabla_{x_k}\mathcal L_{\mathrm{CE}} (S_\theta(x_k),y_{\mathrm{real}}) \right) \right] }$
	每次迭代执行：
	1. 将当前图像输入代理检测器
	2. 计算它被判断为 Real 时的交叉熵损失
	3. 通过反向传播得到对输入像素的梯度
	4. 沿负梯度方向修改像素
	5. 将图像投影回允许的扰动范围
2. Diffusion Chain Attack（DCA）
    使用一个已经训练好的扩散模型，把 SFA 处理过的图片先加噪，再重新还原，并且故意干预还原过程
    1. 将图像进行 VAE 编码
    2. 前向加噪，逐步在潜向量中加入高斯噪声
    3. 反向重建，使用已训练好的去噪网络，把当前潜在状态恢复得更接近自然图像表示
3. Decoupled Frequency Domain Processing（DFDP）
    进一步处理 DCA 重建后的图像，让它在频率统计上更接近真实照片，同时尽量不破坏视觉内容
    扩散重建并不保证生成图像的频谱已经与真实图像一致，所以生成的图像可能存在：
    - 某些低频幅度统计与真实照片不同
	- 高频纹理过于规则或缺乏随机性
	- 某些频率异常仍能被 FreDect 等检测器识别
	训练一个新的轻量生成器。它通过像素损失、LPIPS 和真实图像平均低频谱损失修正图像；随后使用无需训练的 Haar 小波随机高频扰动
### 结果与意义

### 局限性与可质疑点


%% end research_v1 %%

## Annotations



%% begin annotations %%


%% end annotations %%


%% Import Date: 2026-10-14T15:35:37.007+08:00 %%
