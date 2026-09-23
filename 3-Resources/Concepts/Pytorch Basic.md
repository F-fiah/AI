---
cssclasses:
  - 深度学习
date: 2026-09-09
status: in-progress
---
# 数据处理
## 张量

**可以在 gpu 或其它硬件加速器上运行的 ndarray**
### 张量的创建
```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```
默认情况下，张量是在 CPU 上创建的，需要使用 `.to` 方法将张量显式移动到加速器上
```python
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
```
### 张量的属性

- `tensor.shape` 形状
- `tensor.dtype` 数据类型
- `tensor.device` 存储设备
### 张量的操作

- 索引与切片：与 numpy 类似
- 张量拼接：`torch.cat([tensor,tensor], dim=1)`
## Dataset and DataLoader
### 加载数据集

`torch.utils.data.Dataset`，用于存储样本及其对应的标签

`TorchVision`：
- `root` 存储训练/测试数据的路径
- `train` 指定训练/测试数据集
- `download=True` 若 `root` 处没有数据，则从互联网下载
- `transform` 特征
- `target_transform` 标签

将 `Dataset` 作为参数传递给 `DataLoader`，可以将其封装一个可迭代对象，并支持自动批处理（batching）、采样（sampling）、洗牌（shuffling）和多进程数据加载
### 自定义数据集

自定义 `Dataset` 类，需实现`__init__`、`__len__` 和 `__getitem__` 三个函数

```python
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """"""
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
	    """样本数量"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """在给定的索引 `idx` 处加载并返回数据集中的一个样本"""
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```
### 准备数据

`Dataset` 一次检索一个样本的数据集特征和标签；`DataLoader` 则以小批量传递样本，在每个时期重新打乱数据以减少模型过拟合

`train_dataloader = DataLoader(training_data), batch_size=64, shuffle=True)`
## 数据转换

所有 TorchVision 数据集都具有两个参数：用于修改特征的 `transform` 和用于修改标签的 `target_transform`
```python
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)
```
- `v2.ToImage`：将 PIL 图像或 NumPy `ndarray` 转换为 `torchvision.tv_tensors.Image` 张量
- `v2.ToDtype`：转换为 `float32` 类型
# 模型
## 神经网络

通过继承 `nn.Module` 定义神经网络
```python
class NeuralNetwork(nn.Module):
    def __init__(self):
	    """定义网络层"""
        super().__init__()
        self.flatten = nn.Flatten() # 将矩阵转换为一维向量
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
	    """指定数据如何通过网络"""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

	
model = NeuralNetwork().to(device)
```
## `torch.autograd` 自动微分

默认情况下，所有设置了 `requires_grad=True` 的张量都会跟踪其计算历史并支持梯度计算

对于已经训练好的模型，只希望将数据输入到网络中进行正向计算，这可以通过 `torch.no_grad()` 块包裹计算代码来停止跟踪计算
## 损失函数 and 优化器

```python
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```
## 训练函数
```python
def train(dataloader, model, loss_fn, optimizer):
	"""对训练数据集（分批输入）进行预测，并反向传播预测误差以调整模型的参数"""

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
## 测试函数
```python
def test(dataloader, model, loss_fn):
	"""根据测试数据集检查模型的性能"""
	
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
```
## 模型保存与加载

`torch.save(model.state_dict(), "model.pth")`

```
model = NeuralNetwork().to(device)

model.load_state_dict(torch.load("model.pth", weights_only=True))
```
