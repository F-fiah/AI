近几周主要参考 INTERPRETING CLIP’S IMAGE REPRESENTATION VIA TEXT-BASED DECOMPOSITION 文章的算法进行CLIP模型可解释性的分析，同时对MLLMs进行了初步的研究分析
# 核心科研进展与成果

## 实验

### CLIP

使用OpenCLIP ViT-L/14模型+线性探针，以GenImage SDV4 train为训练集训练探针，其余val数据集为测试集

1. 逐层累计消融
   将CLIP模型每层的MSA/MLP贡献替换为训练集均值，计算其性能变化
   核心结果：MLP层对最终决策的影响相对MHA较小，且最后一层（L23）的MHA是决策关键区域；模型在BigGAN数据集上的泛化能力较差（baseline=57.8%）
2. 单头消融
   将一个头的贡献替换为训练集均值，保留其他所有组件不变，观察模型性能变化
   核心结果：单头消融几乎不影响性能，这可能说明CLIP的判别信息是高度分布式冗余的
3. 单头线性探针
   仅用单个头的输出特征训练线性探针，在6个数据集上评估ACC/AUC
   核心结果：中层头（L10-L15）线性可分性极高，说明模型中层编码的判别信息较为丰富
4. TextSPAN语义分析
   通过textspan算法，提取每个头对应的语义信息
   核心结果：大部分头的Real/Fake文本描述高度重合
### MLLM

通过冻结视觉编码器 + 线性探针的范式，评估 InternVL2.5-8B 与 Qwen2.5-VL-7B 两个 LMM 的视觉编码器在 Deepfake 检测任务的性能

目前运行到的结果为：
InternVL2.5-8B

| 测试集            | N       | ACC(%) | AUC     |
| -------------- | ------- | ------ | ------- |
| sdv4/train     | 216,626 | 99.94  | 0.99999 |
| sdv4/val       | 10,022  | 99.83  | 0.99994 |
| ADM/val        | 12,000  | 58.16  | 0.947   |
| BigGAN/val     | 12,000  | 50.74  | 0.787   |
| glide/val      | 12,000  | 64.52  | 0.960   |
| Midjourney/val | 12,000  | 80.60  | 0.977   |
| VQDM/val       | 12,000  | 60.36  | 0.946   |
## 文献调研

- C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection
  向文本编码器注入类别通用提示（Category Common Prompt），以增强图像编码器的检测性能
- Generalizable Deepfake Detection via Simplicity-Bias-Aware CLIP Adaptation
  针对CLIP的简单性偏好（Simplicity Bias）进行适配

# 现存科研问题与卡点

1. CLIP中层头的线性可分性很高，但存在信息冗余，导致难以筛选出对泛化性影响较大的头
2. L23H13在各个实验结果中都存在特殊性，需要进一步分析
3. textspan算法提取到的语义信息在real/fake上高度重合
4. MLLMs实验中AUC 与 ACC存在较大偏差








