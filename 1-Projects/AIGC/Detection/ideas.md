# AIGC 图像检测安全方向：初步研究 Idea 汇总

**研究主题：** AIGC 图像检测器的对抗脆弱性、泛化能力与系统安全性
## 真实图像的定向误报攻击（Real → Fake）

**研究内容：** 对一张原本被正确识别的真实照片添加微小扰动，使未知 AIGC 检测器错误地将其判为 AI 生成图像。初步考虑基于公开代理模型的迁移黑盒攻击，限制扰动幅度，重点评价攻击成功率、跨检测器迁移性、图像质量和成本。

**研究背景：** 现有攻击研究更多关注 Fake → Real，即合成图像逃避检测；而 Real → Fake 可能造成真实照片被错误质疑。但目前已有论文研究过双向攻击，因此简单反转已有攻击算法的目标标签，创新性不足。

**潜在创新点：** 重点研究两类攻击在决策边界、频率敏感性和跨模型迁移上的系统性非对称，分析真实照片被误报的独特机制，并在严格低误报阈值下验证。

**核心相关论文：**

- [FBA²D：Frequency-based Black-box Attack（2025，arXiv）](https://arxiv.org/abs/2512.09264)：双向频域黑盒攻击
- [RAID：Adversarial Robustness of AI-Generated Image Detectors（2025，arXiv）](https://arxiv.org/abs/2506.03988)：多检测器迁移攻击与鲁棒性评测
- [DuFIA：Dual-Domain Feature Importance Attack（IEEE SPL 2026）](https://arxiv.org/abs/2511.15571)：结合空间与频域特征重要性的迁移攻击。

## 异构 AIGC 检测器协作系统的对抗安全性

**研究内容：** 研究多个不同取证原理的检测器（CNN、CLIP、频域、扩散重建等）通过多数投票或置信度融合进行协作时，是否真正提高对抗鲁棒性。通过攻击完整协作系统，分析其共同失效机制，并探索更可靠的融合策略。

**研究背景：** 多检测器集成通常被认为比单模型安全，但不同架构不意味着攻击脆弱性独立。一种扰动可能同时影响多个取证系统，使多数投票失效。

**潜在创新点：** 研究模型错误相关性、异构取证证据的互补性，以及协作决策规则对整体安全性的影响。可进一步考虑部分模型未知、只能查询最终投票结果的真实部署场景。

**核心相关论文：**

- [RAID（2025，arXiv）](https://arxiv.org/abs/2506.03988)：通过多个检测器联合构造可迁移对抗样本
- [D4：Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles（WACV 2024）](https://ieeexplore.ieee.org/document/10483793/)：通过频域子空间划分构建集成防御。
- [On the Diversity of Adversarial Ensemble Learning（ICML 2025）](https://proceedings.mlr.press/v267/guo25g.html)：分析模型多样性与集成对抗鲁棒性的关系。
- [CO-SPY（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_CO-SPY_Combining_Semantic_and_Pixel_Features_to_Detect_Synthetic_Images_CVPR_2025_paper.html)：融合语义和像素取证证据。
    

*多模型联合攻击和集成防御已有较多研究，必须证明异构取证系统存在新的系统级安全问题，而不是简单复现 ensemble attack*

## 图像多轮传播下的对抗鲁棒性

**研究内容：** 研究一张图像经过多次 JPEG 压缩、裁剪、缩放、社交平台转发等处理后，AIGC 检测器的判断如何变化。进一步考虑攻击者能够控制部分传播版本或处理操作，研究多版本联合检测的安全性。

**研究背景：** 真实图像在互联网上通常经过多轮处理，而传统检测器大多针对单张图像进行判断。已有研究证明传播处理会造成检测性能下降，也提出了利用多个传播版本提高检测稳定性的方法。

**潜在创新点：** 将被动的传播鲁棒性评测扩展为主动对抗场景，研究攻击者选择传播路径、控制部分近重复版本时，现有联合取证方法是否仍然可靠。

**核心相关论文：**

- [RRDataset：Bridging the Gap Between Ideal and Real-world Evaluation（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.html)：真实场景、多轮社交传播与再数字化评测
- [QuAD：Quality-Aware Calibration for AI-Generated Image Detection in the Wild（CVPR Workshops 2026）](https://arxiv.org/abs/2604.15027)：利用近重复图像的质量信息进行联合检测
- [TrueFake（IJCNN 2025）](https://arxiv.org/abs/2504.20658)：真实社交平台处理版本的数据集

多轮传播、传播树及多版本融合都已有直接先例。仅研究多次 JPEG 压缩或叠加常规图像变换，难以形成充分创新。

## 基于因果分析的 AIGC 取证特征研究

**研究内容：** 分析现有检测器利用哪些信息判断图像真假，区分真正与生成过程有关的取证特征，以及图像内容、压缩格式、相机来源等造成的虚假相关性。进一步研究如何减少对这些捷径特征的依赖。

**研究背景：** 许多检测器在已知生成器上准确率很高，但面对未知生成器或真实场景时明显下降。已有研究表明，训练数据中的内容与频谱偏差可能导致模型学习错误的判别依据。

**潜在创新点：** 从经验性数据增强进一步发展到可控制变量的取证特征归因，通过反事实或受控实验分析不同因素的真实贡献，再据此改善跨生成器泛化。

**核心相关论文：**

- [B-Free：A Bias-Free Training Paradigm（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html)：利用语义匹配的 Real/Fake 数据减少训练偏差。
- [Dual Data Alignment（NeurIPS 2025）](https://papers.nips.cc/paper_files/paper/2025/hash/991c9f1799cfeebb4217baaacc462d86-Abstract-Conference.html)：同时考虑空间和频率域的数据对齐。
- [Breaking Latent Prior Bias（NeurIPS 2025）](https://papers.neurips.cc/paper_files/paper/2025/hash/2c2be72635379ec8600eea13df4743e3-Abstract-Conference.html)：利用潜在空间困难样本训练改善泛化。
