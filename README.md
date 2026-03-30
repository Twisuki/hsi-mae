# HSI-MAE: 高光谱图像基础模型

> 基于 Masked Autoencoder 的自监督表征学习与地物分类框架

---

## 项目结构

```
hsi-mae/
├── src/
│   ├── datasets/        hsi_dataset.py    # .npy / .mat 数据加载
│   ├── models/         encoder.py        # HSIEncoder
│   │                  mae.py            # HSIMAE
│   │                  classifier.py      # HSIClassifier / HSIFineTuner
│   ├── augmentations/  hsi_mask.py       # Spectral / Spatial / Joint / Random Mask
│   ├── losses/         sam.py            # SAM Loss + MSE+SAM 组合损失
│   ├── train/          pretrain.py       # PretrainEngine
│   │                  finetune.py        # FinetuneEngine
│   │                  augmentations.py   # TrainAugmentation / ValAugmentation
│   └── utils/         config.py          # TrainConfig / ConfigManager
│                      seed.py            # set_seed()
│                      logger.py         # get_logger()
│                      metrics.py        # OA / AA / Kappa / F1
│                      visualization.py  # 绘图工具
├── scripts/
│   ├── run_pretrain.py            # 预训练入口
│   ├── run_finetune.py            # 微调入口
│   └── generate_synthetic_hsi.py  # 生成合成数据
└── tests/                         # pytest 单元测试
```

---

## 快速开始

### 1. 安装依赖

```bash
uv sync
# 或手动安装 PyTorch (CUDA 12.6)
pip install torch==2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 2. 生成合成数据 (用于验证)

```bash
python scripts/generate_synthetic_hsi.py \
    --bands 200 --num-classes 8 --seed 42
```

### 3. 预训练

```bash
python scripts/run_pretrain.py \
    --data-path data/synthetic_hsi.npy \
    --bands 200 \
    --encoder-dim 256 \
    --encoder-layers 4 \
    --mask-ratio 0.75 \
    --epochs 50 \
    --batch-size 256 \
    --save-dir checkpoints
```

### 4. 微调

```bash
python scripts/run_finetune.py \
    --data-path data/synthetic_hsi.npy \
    --labels-path data/synthetic_hsi_gt.npy \
    --encoder-path checkpoints/best_encoder.pt \
    --num-classes 8 \
    --mode full \
    --epochs 50
```

---

## 核心概念

### 训练目标

```
L = MSE(x̂, x) + λ · SAM(x̂, x)
```

- **MSE**: 数值重建误差
- **SAM**: 光谱角映射, 确保材料光谱特征一致性
- `λ` 默认 0.1

### Mask 策略

| 策略 | 说明               |
|------|------------------|
| `SpectralMask` | 随机遮挡光谱波段         |
| `SpatialMask` | 随机遮挡空间位置         |
| `JointMask` | 3D 联合遮挡          |
| `RandomMask` | 元素级随机遮挡 (MAE 默认) |

### 微调模式

- `linear_probe`: 冻结 encoder, 仅训练分类头
- `full`: encoder + 分类头联合微调

---

## 评估指标

| 指标 | 说明 |
|------|------|
| OA | Overall Accuracy, 整体准确率 |
| AA | Average Accuracy, 各类别召回率均值 |
| Kappa | Cohen's Kappa 系数 |
| F1 | Macro F1 + Per-class F1 |

---

## 数据格式

| 文件类型 | 格式 | 说明 |
|----------|------|------|
| 数据 | `.npy` `[H, W, C]` | 每像素一行光谱向量 |
| 标签 | `.npy` `[H, W]` | 每像素一个类别 ID |
| `.mat` | `{"data": [...], "labels": [...]}` | 也可直接传入 |

推荐数据集: Indian Pines, Pavia University, Houston.

---

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码检查

```bash
ruff check .
ruff format .
```

### 完整验证流程

```bash
# 1. 生成数据
python scripts/generate_synthetic_hsi.py --bands 50 --num-classes 4

# 2. 预训练 (1 epoch 快速验证)
python scripts/run_pretrain.py \
    --data-path data/synthetic_hsi.npy \
    --bands 50 --encoder-dim 64 --epochs 1 --batch-size 256

# 3. 微调
python scripts/run_finetune.py \
    --data-path data/synthetic_hsi.npy \
    --labels-path data/synthetic_hsi_gt.npy \
    --encoder-path checkpoints/best_encoder.pt \
    --num-classes 4 --bands 50 --encoder-dim 64 --epochs 1
```

---

## License

Apache 2.0
