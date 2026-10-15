> 核心参考文件："D:\DeepLearning\Tunnel-Detect\OrangePi\OrangePi_AI_Pro_20T_昇腾_用户手册_v1.1.pdf"

---

**华为昇腾生态下模型部署的核心流程：PC训练 -> ONNX -> ATC转OM -> NPU推理**
参考：[昇腾社区](https://www.hiascend.com/)
1. PC 端使用 PyTorch 训练好模型，生成 .pt 模型权重文件
2. 昇腾的 ATC 工具本身不直接支持 PyTorch 的`.pt`文件，所以需要先通过`torch.onnx.export`将 PyTorch 模型导出为通用的 ONNX 格式
3. OM（Offline Model）是昇腾 AI 处理器专用的离线模型格式。它经过了 ATC 工具的深度优化（如算子调度、内存优化等），能充分发挥NPU的硬件性能

---
# 核心部件

| 层次           | 内容                  | 作用                    |
| ------------ | ------------------- | --------------------- |
| **硬件层**      | 昇腾 310B4 NPU        | 执行 AI 计算的芯片           |
| **驱动/固件层**   | NPU 驱动、固件           | 让 Linux 能识别并控制 NPU    |
| **操作系统层**    | Ubuntu 22.04（Linux） | 基础运行环境                |
| **CANN 软件栈** | 运行时、编译器、算子库、工具链     | **连接 AI 框架与 NPU 的桥梁** |
| **AI 框架层**   | PyTorch / MindSpore | 训练和导出模型               |
| **应用层**      | 你的 YOLOv8 检测脚本      | 最终实现隧道缺陷检测            |
## CANN

CANN（Compute Architecture for Neural Networks）是昇腾 AI 处理器的软件栈，负责连接上层 AI 框架（PyTorch、MindSpore 等）和底层昇腾 NPU 硬件，让模型能够在 NPU 上高效执行
*CANN 之于昇腾 NPU，等同于 CUDA 之于 NVIDIA GPU*

| 组件              | 功能                                     |
| --------------- | -------------------------------------- |
| GE 图引擎          | 计算图编译和运行的控制中心，提供图优化、图编译、图执行等功能         |
| Ascend C 算子开发语言 | 支持 C/C++ 标准规范，用于开发高性能自定义算子             |
| AOL 算子加速库       | 提供深度优化的高性能算子，包括神经网络库、线性代数计算库等          |
| HCCL 集合通信库      | 提供单机多卡及多机多卡间的数据并行、模型并行集合通信方案           |
| Runtime 运行时     | 提供硬件资源管理、媒体数据预处理、模型推理等接口               |
| ATC 离线模型编译工具    | 将网络模型转换为昇腾 AI 处理器支持的 .om 格式离线模型        |
| AscendCL 应用开发接口 | 提供 C/C++ 和 Python 两种语言的 API，用于开发 AI 应用 |
