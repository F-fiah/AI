---
cssclasses:
  - 深度学习
  - 计算机视觉
date: 2026-07-08
---
# Nearest Neighbour

记录所有带标签的已知数据集，将代预测的数据集和已知数据集进行比较，找到最相近的一个
但由于数据集里可能存在噪音，所以通常计算最近的 k 个，找其中数量最大的

距离函数（$I_i^j$ 代表第 i 个图片的第 j 个像素点）：
- $d_1(I_1,I_2) = \sum_{p} \left| I_1^p - I_2^p \right|$
- $d_2(I_1,I_2) = \sqrt{\sum_{p} \big(I_1^p - I_2^p\big)^2}$

 如何确定 k 的值：k 折交叉法
# Linear Classifier

![[Linear-Classifier.png]]
## 正则化
$$ L = \frac{1}{N}\sum_{i=1}^{N} L_i + R(W) $$
![[正则化.png]]
## Optimization

记 $L(\theta)$ 为损失函数
- 批量梯度下降：$\theta_{t+1} = \theta_t - \eta \cdot \nabla L_{\text{full}}$
- 随机梯度下降：$\theta_{t+1} = \theta_t - \eta \cdot \nabla L_{\text{single}}$
- 小批量梯度下降：$\theta_{t+1} = \theta_t - \eta \cdot \nabla L_{\text{batch}}$
- 动量梯度下降：$\begin{cases} v_t = \gamma \cdot v_{t-1} + \nabla L_t \\ \theta_{t+1} = \theta_t - \eta \cdot v_t \end{cases}$
- AdaGrad：累积全部历史梯度平方，永久缩放学习率
  $\begin{cases} S_t = S_{t-1} + \big(\nabla L_t\big)^2 \\ \theta_{t+1} = \theta_t - \dfrac{\eta}{\sqrt{S_t + \varepsilon}} \nabla L_t \end{cases}$
- RMSProp：
  $\begin{cases} S_t = \gamma S_{t-1} + (1 - \gamma)\big(\nabla L_t\big)^2 \\ \theta_{t+1} = \theta_t - \dfrac{\eta}{\sqrt{S_t + \varepsilon}} \nabla L_t \end{cases}$
- Adam：Momentum + RMSProp
  $\begin{cases} m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla L_t \\ S_t = \beta_2 S_{t-1} + (1 - \beta_2)\big(\nabla L_t\big)^2 \\ \hat{m}_t = \dfrac{m_t}{1 - (\beta_1)^t},\quad \hat{S}_t = \dfrac{S_t}{1 - (\beta_2)^t} \\ \theta_{t+1} = \theta_t - \dfrac{\eta}{\sqrt{\hat{S}_t + \varepsilon}} \hat{m}_t \end{cases}$
  第三行的修正是防止在前期 $m_t$ 和 $S_t$ 过小
- AdamW：Adam + 正则化
  在 Adam 的基础上，最后一步改为：
  $\theta_t = \theta_{t-1} - \eta \cdot \dfrac{\hat{m}_t}{\sqrt{\hat{v}_t + \varepsilon}} - \eta \lambda \cdot \theta_{t-1}$

经过固定次数的迭代后，可以减小学习率 $\eta$，从而加快收敛速度
- 每次减小为原来的 0.1
- $\eta_t = \dfrac{1}{2}\eta_0 \big(1 + \cos\left(\frac{t\pi}{T}\right)\big)$，T：Total epoch number
- $\eta_t = \eta_0(1 - t/T)$
- $\eta_t = \eta_0/\sqrt{t}$
- 对于小批量梯度下降，如果增大 batch 的值，学习率也应当等比例增大

