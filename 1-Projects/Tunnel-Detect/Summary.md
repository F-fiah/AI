# Basic Knowledge
## Linux

[[linux]]
## Orange Pi

[[Orange Pi]]
## Embedded AI

- [[[野火]《嵌入式AI应用开发实战指南—基于LubanCat-RK系列板卡》.pdf]]
  专门面向瑞芯微RK系列板卡的嵌入式AI实战指南
- [Radxa](https://docs.radxa.com/som/cm/cm4/app-dev/npu)
- "D:\DeepLearning\Tunnel-Detect\rknn3-model-zoo-main"
  YOLO、ResNet等模型的官方部署示例
- "D:\DeepLearning\Tunnel-Detect\custom_yolov5_npu_rk3588_orangepi-main"
  针对Orange Pi 5 Plus等开发板优化
## Projects

- "D:\DeepLearning\Tunnel-Detect\yolov8-crack-master"
  基建场景（如桥梁、隧道）的裂缝检测，代码结构简单
- [Smart-concrete-crack-detector](https://github.com/Escgot/Smart-concrete-crack-detector)
  做检测的同时，还实现了裂缝的物理宽度测量和严重程度分类，可以作为进阶任务
- "D:\DeepLearning\Tunnel-Detect\steel-defect-detection-main"
  以YOLOv8为引擎，采用前后端分离架构，支持图片、视频及摄像头多模式识别
  将一个CV模型包装成一个可用的工业系统，对项目最终的系统开发有很大启发
- [Wedtect-YOLOv8-OBB](https://github.com/amedo007-poly/Wedtect-YOLOv8-OBB)
- "D:\DeepLearning\Tunnel-Detect\ai-tunnel-beta-main"
  多类病害检测，与我们的项目目标很接近
# 数据处理

使用 Label Studio 进行数据标记

---

**检查 Label Studio 是否仍在运行**

香橙派 SSH 终端中执行：
```bash
sudo ss -lntp | grep ':8080'
```
如果出现类似：
```bash
LISTEN 0 128 0.0.0.0:8080 ...
```
说明 Label Studio 仍在监听端口；如果没有任何输出，说明目前没有程序监听 8080，服务可能已经退出

---
# Log
## 9.24-9.25

按照“硬件联网 → 环境验证 → 模型转换 → 推理验证”四个阶段，完整走通了昇腾 NPU 的部署链路
把训练好的 YOLOv8 模型成功部署到昇腾 NPU 上，并且跑通了完整的静态图片推理流程

**1. 硬件联网与远程环境搭建**
*   **启动与登录**：使用 Micro USB 调试串口线连接开发板与电脑，通过 MobaXterm 以 115200 波特率连接串口，成功登录系统（`HwHiAiUser` / `Mind@123`）
*   **网络配置**：通过 `nmcli` 命令让开发板连接手机热点，获取到局域网 IP（`10.148.36.232`），并成功通过 SSH 从电脑远程登录

**2. NPU 状态检查与官方样例验证**
*   **状态排查**：执行 `npu-smi info`，发现 NPU 状态显示为 `Alarm`（错误码 `80E3A203`，LPM 传感器电流检测异常）
*   **决策验证**：为了确认 NPU 计算核心是否可用，我们启动了 Jupyter Lab，运行了官方预置的离线推理样例。样例成功运行，证明了 **NPU 核心计算功能完全正常**，`Alarm` 只是电源监控传感器的误报，决定暂时忽略

**3. 模型准备与格式转换（PC → 开发板）**
*   **PC 端导出 ONNX**：在 WSL 训练环境中，将 YOLOv8 的 `.pt` 权重导出为通用的 `.onnx` 格式（`opset=11`，`imgsz=640`，`simplify=True`）
*   **上传至开发板**：使用 `scp` 命令将 `.onnx` 文件上传到开发板
*   **ATC 转换 OM 模型**：在开发板激活 CANN 环境（`source set_env.sh`），使用 ATC 工具将 ONNX 编译为昇腾专用的 `.om` 离线模型。关键参数：`--soc_version=Ascend310B4`，`--input_shape="images:1,3,640,640"`。成功获得 `best.om`

**4. 编写推理脚本并跑通（OM 推理）**
*   **编写推理脚本**：使用 `ais_bench` 的 `InferSession` 加载 OM 模型。实现了完整的预处理（Letterbox 保持纵横比、BGR→RGB 转换、归一化）、推理、置信度过滤、NMS 后处理，并将坐标映射回原图。
*   **结果输出**：成功处理 `crack_predict` 文件夹中的图片，输出了带检测框的可视化图片和 YOLO 格式的 txt 标签文件，结构与 PC 端的 `model.predict` 完全一致。

### 经验总结

**1. 硬件与环境**
*   **网络**：USB 调试串口只能单机操作，手机热点是临时过渡方案。后续团队协作必须将开发板接入实验室局域网（网线或 Wi-Fi），确保所有成员能通过 SSH 访问。
*   **散热与供电**：昇腾 310B4 高负载下发热明显，必须确保主动散热风扇正常工作，并使用官方 65W PD 适配器，否则会导致 NPU 降频甚至重启。

**2. 工具链与模型**
*   **参数必须严格对齐**：ATC 转换时的 `--input_shape` 必须与导出 ONNX 时的 `imgsz` 一致（这里是 `1,3,640,640`）；`--soc_version` 必须精确匹配芯片型号（`Ascend310B4`）。
*   **预处理与后处理必须对齐**：
    *   **预处理**：YOLOv8 训练用 RGB，OpenCV 读入是 BGR，必须做 `img[:, :, ::-1]` 转换；Letterbox 的 padding 值需要精确计算并用于坐标映射。
    *   **后处理**：YOLOv8 输出格式为 `[1, 84, 8400]`（通道优先），需要转置为 `(8400, 84)` 再取前 4 位为 `xywh`，后 80 位为类别分数，且**没有 objectness**。
*   **编码规范**：Linux 环境默认 UTF-8，Python 脚本中尽量避免中文注释和 `print`，否则可能触发 `SyntaxError: Non-UTF-8 code`

**3. 工程与协作层面**
*   **记录一切**：从 `Alarm` 状态排查到 ATC 转换报错，所有踩坑过程都是后续论文、专利和团队交接的宝贵素材
*   **先跑通，再优化**：我们先用官方样例验证 NPU，再跑通静态图片推理
### 后续推进方向

目前已经完成了**单张图片推理**的验证，接下来需要向**实时视频流**和**系统集成**推进

**1. 短期目标：接入摄像头，实现实时推理**
*   **硬件准备**：采购一个 USB 摄像头（建议星光级低照度，如 ELP-USD1080P03-H100），插入开发板 USB 3.0 接口。
*   **代码改造**：将 `predict_om.py` 中的图片读取改为 `cv2.VideoCapture(0)`，循环读取视频帧，逐帧调用 `infer_one`，并在画面上绘制检测框。
*   **性能优化**：引入多线程架构（采集线程、推理线程、显示线程分离），使用 `queue.Queue` 缓冲帧，避免因推理慢于采集导致画面卡顿。
*   **参考官方样例**：官方 `samples` 目录中有 `01-SSD` 等实时视频流样例，可以参考其 ACLLite 封装和摄像头读取方式。

**2. 中期目标：病害量化与系统集成**
*   **病害量化**：在检测框基础上，计算裂缝的长度/宽度（骨架提取或矩形邻域法）、渗漏水面积等物理尺寸（需要相机标定）。
*   **系统开发**：将推理模块封装为类，接入 Flask 后端（接收数据上报）、SQLite 数据库（本地存储）、MQTT/HTTP（云端上报），构建端到端的边缘检测系统。
*   **多人协作**：按照之前制定的分工，算法负责人继续优化模型（低光照增强、P2 检测层），部署负责人优化 NPU 推理性能（NMS 下沉、零拷贝），系统负责人搭建监控页面。

**3. 长期目标：算法优化与成果输出**
*   **低光照增强**：在预处理中加入 CLAHE 或自适应伽马校正，提升隧道低光照环境下的小目标裂缝检测能力。
*   **模型轻量化与量化补偿**：探索 INT8 量化后的精度损失补偿方案（如混合量化），寻找精度与速度的最优平衡。
*   **消融实验与论文/专利**：设计完整的消融实验矩阵（Baseline、增强、改进模型、完整方案），整理实验数据，撰写论文或申请专利。

**4. 团队协作**
*   **开发板接入实验室网络**：搬入实验室后，立即用网线将开发板接入路由器，获取固定 IP，确保团队三人均可通过 SSH 访问。
*   **代码仓库规范化**：将 `predict_om.py` 等脚本整理进 Git 仓库，编写 README 说明运行环境和步骤，方便团队成员复现和协作
## 9.25

使用 Tailscale，解决团队远程连接的问题

**OrangePi 的 Tailscale 虚拟 IP 地址：100.70.198.52**

OrangePi 需要保持联网状态，检测方式：
- 电脑的 PowerShell 中执行：`tailscale status` 或 `ping 100.70.198.52`
- 尝试 SSH 登录
- 香橙派串口本地检查
	```
	# 查看网络接口是否获取到 IP
	ip addr show wlan0   # 如果连的是 WiFi
	ip addr show eth0    # 如果连的是网线
	
	# 查看 NetworkManager 连接状态
	nmcli device status
	```

OrangePi 在校园网环境下的 IP 为：10.69.230.170