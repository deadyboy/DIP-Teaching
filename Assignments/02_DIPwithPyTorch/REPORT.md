# Assignment 2 - DIP with PyTorch 作业报告

本次作业包含两个部分：**泊松图像编辑 (Poisson Image Editing)** 和 **Pix2Pix 图像翻译网络**。下面会用最通俗的语言解释每个部分做了什么、为什么这么做、以及你该怎么运行代码得到结果。

---

## 目录
1. [环境配置（第一步，必做）](#1-环境配置)
2. [Part 1：泊松图像编辑 (Poisson Image Editing)](#2-part-1泊松图像编辑)
   - [它是什么？](#21-它是什么)
   - [我填写了哪些代码？](#22-我填写了哪些代码)
   - [如何运行？](#23-如何运行)
   - [预期效果](#24-预期效果)
3. [Part 2：Pix2Pix 图像翻译网络](#3-part-2pix2pix-图像翻译网络)
   - [它是什么？](#31-它是什么)
   - [我填写了哪些代码？](#32-我填写了哪些代码)
   - [如何运行？](#33-如何运行)
   - [预期效果](#34-预期效果)
4. [文件清单](#4-文件清单)
5. [致谢](#5-致谢)

---

## 1. 环境配置

在运行任何代码之前，你需要先配置 Python 环境。推荐使用 [Miniconda](https://docs.anaconda.com/miniconda/)。

```bash
# 1. 创建一个新的 conda 环境（只需执行一次）
conda create -n dip python=3.10 -y
conda activate dip

# 2. 安装 PyTorch（如果你有 NVIDIA 显卡，用 GPU 版本会快很多）
# GPU 版本（推荐，需要 NVIDIA 显卡 + CUDA）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 如果没有显卡，用 CPU 版本：
# pip install torch torchvision

# 3. 安装其他依赖
pip install gradio pillow numpy opencv-python
```

> **小提示**：如果你不确定自己有没有 GPU，在终端输入 `nvidia-smi`，如果能看到显卡信息就说明有。

---

## 2. Part 1：泊松图像编辑

### 2.1 它是什么？

想象你要把一张照片里的某个物体（比如一只猫）"贴"到另一张照片上（比如一片草地）。如果直接复制粘贴，边缘会非常明显、不自然。

**泊松图像编辑**就是一种让"粘贴"看起来自然的技术。它的核心思想是：
- 不是直接复制像素值，而是复制**梯度**（即像素之间的变化关系）
- 让粘贴区域的"纹理变化"和原图一样，但颜色能自然地融入背景

打个比方：就像你用水彩笔在一张纸上画了图案，然后用水把边缘晕染开，让它和背景融为一体。

### 2.2 我填写了哪些代码？

在 `run_blending_gradio.py` 文件中填写了两个函数：

#### 函数 1：`create_mask_from_points()` （第 96-118 行）

**作用**：把用户在图上画的多边形变成一个"蒙版"（mask）。

```
蒙版是什么？就是一张黑白图：
- 白色（255）= 多边形内部（需要融合的区域）
- 黑色（0）  = 多边形外部（不需要处理的区域）
```

**实现方式**：使用 PIL 库的 `ImageDraw.polygon()` 函数，传入多边形的顶点坐标，自动填充内部为白色。

#### 函数 2：`cal_laplacian_loss()` （第 121-155 行）

**作用**：计算"拉普拉斯损失"，这是泊松融合的核心。

**通俗解释**：
1. **拉普拉斯算子**是一个 3×3 的小矩阵（卷积核），长这样：
   ```
   [0,  1, 0]
   [1, -4, 1]
   [0,  1, 0]
   ```
   它能提取图像中每个像素和周围像素的"差异"（梯度信息）。

2. 我们分别对**前景图**和**融合图**做拉普拉斯运算，得到各自的梯度。

3. 计算两者梯度的差异（均方误差），这就是损失值。

4. 优化器会不断调整融合图，让这个损失越来越小 → 融合图的梯度和前景图越来越像 → 融合效果越来越自然。

### 2.3 如何运行？

```bash
# 1. 确保已经激活 conda 环境
conda activate dip

# 2. 进入作业目录
cd Assignments/02_DIPwithPyTorch

# 3. 运行程序
python run_blending_gradio.py
```

运行后会在终端显示一个 URL（通常是 `http://127.0.0.1:7860`），用浏览器打开它。

### 2.4 预期效果

打开网页后你会看到一个交互界面：

1. **上传前景图**（左上角）：这是你想"剪切"物体的图片
2. **在前景图上画多边形**：点击图片上的多个点，圈出你想要的区域
3. **点击 "Close Polygon"**：关闭多边形
4. **上传背景图**（右上角）：这是你想把物体"粘贴"到的图片
5. **调整 Horizontal/Vertical Offset 滑条**：调整粘贴位置
6. **点击 "Blend Images"**：开始融合！

融合过程需要一些时间（5000步优化），终端会显示进度。完成后右下角会显示融合结果。

> **注意**：`data_poisson` 文件夹中提供了测试用的图片（equation、monolisa、water），你可以用这些图片来测试。如果没有 GPU，融合过程会比较慢（可能需要几分钟）。

---

## 3. Part 2：Pix2Pix 图像翻译网络

### 3.1 它是什么？

Pix2Pix 是一种用**深度学习**实现的"图像到图像翻译"技术。比如：
- 输入一张建筑物的语义分割图（每种颜色代表一种材质）→ 输出一张真实的建筑物照片
- 输入一张素描 → 输出一张彩色图片

本次作业使用**全卷积网络 (FCN)** 来实现一个简化版的 Pix2Pix。

### 3.2 我填写了哪些代码？

在 `Pix2Pix/FCN_network.py` 中实现了完整的 **U-Net 风格编码器-解码器网络**。

#### 网络结构图解

```
输入图像 (3通道, 256×256)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    编码器 (Encoder)                       │
│  逐步"压缩"图像，提取高层特征                              │
│                                                          │
│  conv1: 3→64 通道,    256×256 → 128×128  (LeakyReLU)    │
│  conv2: 64→128 通道,  128×128 → 64×64   (BN+LeakyReLU) │
│  conv3: 128→256 通道, 64×64   → 32×32   (BN+LeakyReLU) │
│  conv4: 256→512 通道, 32×32   → 16×16   (BN+LeakyReLU) │
│  conv5: 512→512 通道, 16×16   → 8×8     (BN+LeakyReLU) │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│               解码器 (Decoder) + 跳跃连接                 │
│  逐步"放大"回原始尺寸，同时利用编码器的细节信息              │
│                                                          │
│  deconv5: 512→512 通道,   8×8   → 16×16  (Dropout 0.5)  │
│  deconv4: 1024→256 通道,  16×16 → 32×32  (拼接 conv4, Dropout 0.5) │
│  deconv3: 512→128 通道,   32×32 → 64×64  (拼接 conv3, Dropout 0.5) │
│  deconv2: 256→64 通道,    64×64 → 128×128 (拼接 conv2)   │
│  deconv1: 128→3 通道,    128×128→ 256×256 (拼接 conv1)   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
输出图像 (3通道, 256×256, 经过Tanh激活函数，值域[-1,1])
```

#### 关键设计说明

- **编码器使用 LeakyReLU(0.2)**：相比 ReLU，允许负值梯度流动，防止神经元死亡
- **第一层编码器不使用 BatchNorm**：标准 pix2pix 做法，避免对输入数据过度归一化
- **解码器前三层使用 Dropout(0.5)**：作为正则化手段，防止过拟合，提升生成多样性
- **跳跃连接 (Skip Connection)**：解码器的每一层都会把对应编码层的输出"拼接"过来（通道维度拼接），这样能保留更多细节
- **大通道数 (64→512)**：相比之前的小通道数 (8→128)，大幅增加网络容量，能学习更复杂的映射
- **最后一层用 Tanh**：因为训练数据被归一化到 [-1, 1] 范围，所以输出也要在这个范围内

### 3.3 如何运行？

```bash
# 1. 确保已激活 conda 环境
conda activate dip

# 2. 进入 Pix2Pix 目录
cd Assignments/02_DIPwithPyTorch/Pix2Pix

# 3. 下载 Facades 数据集（建筑立面数据集）
bash download_facades_dataset.sh

# 4. 开始训练模型
python train.py
```

> **训练需要较长时间**：300个epoch，如果有 GPU 大约需要 30-60 分钟，CPU 上可能需要数小时。

### 3.4 预期效果

训练过程中，代码会自动保存结果图片：

- `train_results/epoch_X/` — 训练集上的效果（每5个epoch保存一次）
- `val_results/epoch_X/` — 验证集上的效果（每5个epoch保存一次）
- `checkpoints/` — 模型文件（每50个epoch保存一次）

每张结果图由 **三张图横向拼接** 组成：
```
[ 输入（语义分割图） | 目标（真实照片） | 模型输出（生成的照片） ]
```

随着训练的进行，模型输出会越来越接近目标图像。

> **作业提示**：README中提到需要使用[更多数据集](https://github.com/phillipi/pix2pix#datasets)来获得更好的泛化效果。你可以在结果良好后替换数据集来进一步提升。

---

## 4. 文件清单

| 文件 | 说明 | 是否修改 |
|------|------|---------|
| `run_blending_gradio.py` | 泊松图像编辑的完整程序（含Gradio界面） | ✅ 填写了2个函数 |
| `data_poisson/` | 泊松编辑的测试图片 | 未修改 |
| `Pix2Pix/FCN_network.py` | 全卷积网络定义 | ✅ 填写了完整的U-Net网络 |
| `Pix2Pix/train.py` | 训练脚本 | 未修改 |
| `Pix2Pix/facades_dataset.py` | 数据集加载 | 未修改 |
| `Pix2Pix/download_facades_dataset.sh` | 数据集下载脚本 | 未修改 |

---

## 5. 致谢

> 📋 感谢以下论文和工具：
> - [Poisson Image Editing (Pérez et al., 2003)](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) — 泊松图像编辑的原始论文
> - [Image-to-Image Translation with Conditional Adversarial Nets (Isola et al., 2017)](https://phillipi.github.io/pix2pix/) — Pix2Pix 原始论文
> - [Fully Convolutional Networks for Semantic Segmentation (Long et al., 2015)](https://arxiv.org/abs/1411.4038) — FCN 网络结构
> - [PyTorch](https://pytorch.org/) — 深度学习框架
> - [Gradio](https://www.gradio.app/) — 网页交互界面
