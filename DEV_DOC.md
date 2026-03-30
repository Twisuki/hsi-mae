# 🌈 HSI-MAE: 高光谱图像基础模型与地物分类系统

> A production-ready, reproducible implementation of **Hyperspectral Masked Autoencoder (HSI-MAE)** for representation learning and downstream land-cover classification.

---

# 📌 1. 项目简介

本项目旨在构建一个**工程化、可复现的高光谱基础模型（HSI Foundation Model）**，并验证其在**地物分类任务**中的效果。

核心思路：

```text
高光谱数据 → 自监督预训练（HSI-MAE）
           → 学习通用表征（encoder）
           → 下游任务（地物分类）
```

---

# 🧠 2. 原理概述

## 2.1 什么是 HSI-MAE

HSI-MAE（Hyperspectral Masked Autoencoder）是 MAE 在高光谱领域的扩展：

* 输入：高光谱数据 `[B, H, W]`
* 操作：mask 掉部分光谱 + 空间信息
* 目标：重建原始数据

👉 模型被迫学习：

* 光谱相关性（材料特性）
* 空间结构（地物形态）

---

## 2.2 核心机制

### Mask策略

* Spatial Mask：遮挡空间 patch
* Spectral Mask：遮挡波段
* Joint Mask：3D cube masking

---

### 训练目标

```math
L = MSE(x̂, x) + λ * SAM(x̂, x)
```

* MSE：数值重建
* SAM：光谱角一致性

---

## 2.3 为什么有效

HSI-MAE 能学习：

* 材料级别特征（spectral signature）
* 空间上下文
* 无监督表征（无需标签）

---

# 🏗️ 3. 工程结构

```text
hsi-mae/
├── pyproject.toml
├── README.md
├── src/
│   ├── datasets/
│   │   └── hsi_dataset.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── mae.py
│   │   └── classifier.py
│   ├── augmentations/
│   │   └── hsi_mask.py
│   ├── losses/
│   │   └── sam.py
│   ├── train/
│   │   ├── pretrain.py
│   │   └── finetune.py
│   └── utils/
│       └── config.py
└── scripts/
    ├── run_pretrain.py
    └── run_finetune.py
```

---

# ⚙️ 4. 环境配置（uv）

## 4.1 安装 uv

```bash
pip install uv
```

---

## 4.2 初始化项目

```bash
uv init
```

---

## 4.3 pyproject.toml

```toml
[project]
name = "hsi-mae"
version = "0.1.0"
description = "HSI Foundation Model via MAE"
requires-python = ">=3.10"

dependencies = [
    "torch==2.0.0",
    "torchvision",
    "numpy",
    "tqdm",
    "einops",
    "lightly",
    "scikit-learn",
]

[tool.uv]
dev-dependencies = [
    "pytest",
    "black",
    "ruff"
]
```

---

## 4.4 安装依赖

```bash
uv sync
```

⚠️ CUDA版本说明：

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu126
```

---

# 📦 5. 数据格式

输入：

```python
x: [B, H, W]   # 高光谱
y: [H, W]      # 标签（可选）
```

建议：

* 使用 `.npy` 或 `.mat`
* 统一归一化到 [0,1]

---

# 🧩 6. 核心模块实现

---

## 6.1 HSI Mask

```python
def mask_hsi(x, spectral_ratio=0.5, spatial_ratio=0.5):
    B, H, W = x.shape

    spectral_mask = torch.rand(B) > spectral_ratio
    x[~spectral_mask] = 0

    spatial_mask = torch.rand(H, W) > spatial_ratio
    x[:, ~spatial_mask] = 0

    return x
```

---

## 6.2 Encoder（Spectral + Spatial）

```python
class HSIEncoder(nn.Module):
    def __init__(self, bands, dim=256):
        super().__init__()
        self.spectral = nn.Conv2d(bands, dim, 1)
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        x = self.spectral(x)
        x = self.spatial(x)
        return x
```

---

## 6.3 MAE

```python
class HSIMAE(nn.Module):
    def __init__(self, encoder, bands):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Conv2d(256, bands, 1)

    def forward(self, x):
        x_masked = mask_hsi(x.clone())
        z = self.encoder(x_masked)
        return self.decoder(z)
```

---

## 6.4 SAM Loss

```python
def sam_loss(x_hat, x):
    x_hat = F.normalize(x_hat, dim=1)
    x = F.normalize(x, dim=1)
    return 1 - (x_hat * x).sum(dim=1).mean()
```

---

# 🚀 7. 训练流程

---

## 7.1 预训练（HSI-MAE）

```bash
bash scripts/run_pretrain.sh
```

核心逻辑：

```python
loss = mse(x_hat, x) + 0.1 * sam_loss(x_hat, x)
```

输出：

```text
encoder.pt  ← foundation model
```

---

## 7.2 下游任务（地物分类）

```bash
bash scripts/run_finetune.sh
```

---

### Linear Probe

```python
freeze(encoder)
train(classifier)
```

---

### Fine-tune

```python
train(encoder + classifier)
```

---

# 📊 8. 评估指标

* Overall Accuracy (OA)
* Average Accuracy (AA)
* Kappa
* F1-score

---

# 🧪 9. 实验建议

推荐数据集：

* Indian Pines
* Pavia University
* Houston

---

# 🔧 10. 工程化规范

## ✔ 必须做到

* 不提交缓存（.gitignore）
* 固定随机种子
* 配置分离（config）
* 日志记录（建议 wandb）

---

## ✔ 可扩展

* ViT Encoder
* Patch MAE
* 多模态（HSI + RGB）
* 分布式训练

---

# 🧭 11. Roadmap

* [ ] Patch-based MAE
* [ ] Spectral Transformer
* [ ] 多模态预训练
* [ ] 大规模数据预训练

---

# 📌 12. 一句话总结

> 本项目构建了一个工程化的 HSI-MAE 框架，实现了从无监督表征学习到地物分类的完整流程，是迈向“高光谱基础模型”的可复现实践。

---

# 🤝 13. License

Apache 2.0 License

