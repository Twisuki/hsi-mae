# HSI-MAE: 高光谱图像基础模型

> 基于 Masked Autoencoder 的自监督表征学习与地物分类框架

---

## 项目结构

```
hsi-mae/
├── main.py                 # 交互式入口
├── src/
│   ├── datasets/        hsi_dataset.py    # .npy / .mat 数据加载
│   ├── models/         encoder.py        # HSIEncoder
│   │                  mae.py            # HSIMAE
│   │                  classifier.py      # HSIClassifier / HSIFineTuner
│   ├── augmentations/  hsi_mask.py       # Spectral / Spatial / Joint / Random Mask
│   ├── losses/         sam.py            # SAM Loss + MSE+SAM 组合损失
│   ├── train/          pretrain.py        # PretrainEngine
│   │                  finetune.py        # FinetuneEngine
│   │                  augmentations.py   # TrainAugmentation / ValAugmentation
│   └── utils/         config.py          # TrainConfig / ConfigManager
│                      seed.py            # set_seed()
│                      logger.py         # get_logger()
│                      metrics.py        # OA / AA / Kappa / F1
│                      visualization.py   # 绘图工具
├── scripts/
│   ├── run_pretrain.py            # 预训练入口
│   ├── run_finetune.py            # 微调入口
│   └── generate_synthetic_hsi.py  # 生成合成数据
└── tests/                         # pytest 单元测试
```

---

## 环境变量

项目使用 `.env` 文件管理配置.

- `.env` - 示例配置文件, 提交到 git
- `.env.local` - 本地私有配置, 不提交到 git, 优先级高于 `.env`

加载优先级: 存在 `.env.local` 时使用它, 否则使用 `.env`

运行 `python main.py` 后选择 [7] 可交互式导出当前配置到 `.env.local`.

---

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 交互式入口

```bash
python main.py
```

菜单如下, 每个步骤均显示当前配置值作为默认值:

```
=== HSI-MAE 交互式入口 ===

请选择操作:

  [1] 生成合成数据
  [2] 预训练 HSI-MAE
  [3] 微调分类
  [4] 运行测试
  [5] 代码检查
  [6] 查看当前环境变量
  [7] 导出配置到 .env.local

  [0] 退出

> _
```

### 3. 直接运行脚本 (可选)

如需手动指定参数, 可直接运行脚本:

```bash
# 生成合成数据
python scripts/generate_synthetic_hsi.py \
    --bands 200 --num-classes 8 --seed 42

# 预训练
python scripts/run_pretrain.py \
    --data-path data/synthetic_hsi.npy \
    --bands 200 --encoder-dim 256 --epochs 50 --batch-size 256

# 微调
python scripts/run_finetune.py \
    --data-path data/synthetic_hsi.npy \
    --labels-path data/synthetic_hsi_gt.npy \
    --encoder-path checkpoints/best_encoder.pt \
    --num-classes 8 --mode full
```

---

## 核心概念

### 训练目标

```
L = MSE(xhat, x) + lambda * SAM(xhat, x)
```

- **MSE**: 数值重建误差
- **SAM**: 光谱角映射, 确保材料光谱特征一致性
- `lambda` 默认 0.1

### Mask 策略

| 策略 | 说明 |
|------|------|
| `SpectralMask` | 随机遮挡光谱波段 |
| `SpatialMask` | 随机遮挡空间位置 |
| `JointMask` | 3D 联合遮挡 |
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

---

## License

Apache 2.0
