# DEV_PLAN: HSI-MAE 开发计划

> 基于 `DEV_DOC.md` 规范，构建完整的高光谱图像基础模型框架

---

## 一、项目现状分析

| 维度 | 现状 | 差距 |
|------|------|------|
| 目录结构 | 仅 `main.py` 占位符 | 需创建 `src/` 全部模块 |
| 依赖配置 | `pyproject.toml` 已定义 | 需 `uv sync` 安装 |
| 模型实现 | DEV_DOC 有伪代码 | 需完整实现 |
| 数据加载 | 无 | 需实现 HSI Dataset |
| 训练流程 | 无 | 需实现预训练+微调 |
| 脚本入口 | 无 | 需 `scripts/` |

---

## 二、总体架构

```
HSI-MAE Framework
├── src/
│   ├── datasets/          # 数据加载
│   ├── models/             # 模型定义（Encoder / MAE / Classifier）
│   ├── augmentations/     # 数据增强（Mask策略）
│   ├── losses/             # 损失函数（MSE + SAM）
│   ├── train/              # 训练逻辑
│   └── utils/             # 工具（配置、日志）
└── scripts/               # 入口脚本
```

---

## 三、详细开发计划

### Phase 1：基础框架搭建

#### 1.1 目录结构创建 ✅
- [x] 创建 `src/datasets/`
- [x] 创建 `src/models/`
- [x] 创建 `src/augmentations/`
- [x] 创建 `src/losses/`
- [x] 创建 `src/train/`
- [x] 创建 `src/utils/`
- [x] 创建 `src/__init__.py`
- [x] 创建 `scripts/`
- [x] 创建 `tests/`
- [x] 更新 `.gitignore`

#### 1.2 配置管理系统 `src/utils/` ✅
- [x] `src/utils/__init__.py`
- [x] `src/utils/config.py` — 参数配置类（TrainConfig / ConfigManager）
- [x] `src/utils/seed.py` — 随机种子固定
- [x] `src/utils/logger.py` — 日志工具（loguru 封装）

#### 1.3 数据集加载 `src/datasets/` ✅
- [x] `src/datasets/__init__.py`
- [x] `src/datasets/hsi_dataset.py`
  - 支持 `.npy` / `.mat` 格式
  - 自动归一化到 [0, 1]
  - 支持 train/val split
  - 返回 `(B, H, W, C)` 或 `(B, C, H, W)`

---

### Phase 2：核心模型实现

#### 2.1 数据增强 `src/augmentations/` ✅
- [x] `src/augmentations/__init__.py`
- [x] `src/augmentations/hsi_mask.py`
  - `SpectralMask` — 随机遮挡光谱波段
  - `SpatialMask` — 随机遮挡空间位置
  - `JointMask` — 3D 联合 masking
  - `RandomMask` — 随机元素级 masking（MAE 默认）
  - 统一 `nn.Module` 接口 + 函数式 API

#### 2.2 损失函数 `src/losses/` ✅
- [x] `src/losses/__init__.py`
- [x] `src/losses/sam.py`
  - `sam_loss(pred, target)` — arccos/π 光谱角映射损失，范围 [0,1]
  - `mse_sam_loss(pred, target, lambda_sam=0.1)` — 组合损失
  - `MSELS` / `SAMLoss` / `MSESAMLoss` nn.Module 版本

#### 2.3 模型定义 `src/models/` ✅

##### Encoder
- [x] `src/models/__init__.py`
- [x] `src/models/encoder.py`
  - `HSIEncoder` — Conv2d 光谱投影 + 残差式空间编码
  - 支持 `bands`, `dim`, `num_layers`, `norm`（bn/gn/none）
  - `freeze()` / `unfreeze()` 接口

##### MAE
- [x] `src/models/mae.py`
  - `HSIMAE` — Encoder + Conv Decoder
  - `forward()` — mask → encode → decode，返回 `(recon, x_masked, mask)`
  - 暴露 `encode_no_mask()` / `decode_features()` 供下游复用

##### Classifier
- [x] `src/models/classifier.py`
  - `HSIClassifier` — linear / mlp 两种分类头
  - `HSIFineTuner` — Encoder + Classifier 组合，支持 freeze encoder

---

### Phase 3：训练流程

#### 3.1 预训练 `src/train/` ✅
- [x] `src/train/__init__.py`
- [x] `src/train/pretrain.py`
  - `PretrainEngine` — lr warmup+cosine、grad clip、encoder.pt 保存
  - 支持 `load_encoder(path)` 加载预训练权重

#### 3.2 微调 `src/train/` ✅
- [x] `src/train/finetune.py`
  - `FinetuneEngine` — linear_probe / full 两种模式
  - OA / AA / Kappa / F1 指标
  - `load_classifier(path)` 加载权重

#### 3.3 数据增强工厂 ✅
- [x] `src/train/augmentations.py`
  - `TrainAugmentation` — 光谱噪声/漂移/随机丢带/波段混排
  - `ValAugmentation` — 空操作（no-op）
  - `Compose` / `Normalize` / `ToTensor` 辅助

---

### Phase 4：入口脚本 ✅

#### 4.1 预训练脚本 ✅
- [x] `scripts/run_pretrain.py`
  - argparse 参数解析
  - 调用 `PretrainEngine`
  - set_seed + loguru 日志

#### 4.2 微调脚本 ✅
- [x] `scripts/run_finetune.py`
  - argparse 参数解析
  - 调用 `FinetuneEngine`
  - 训练后自动在 test 集输出 OA / AA / Kappa / F1

---

### Phase 5：评估与可视化 ✅

#### 5.1 评估工具 ✅
- [x] `src/utils/metrics.py`
  - `overall_accuracy()` / `average_accuracy()` / `kappa_coefficient()`
  - `f1_scores()` — macro + per-class
  - `classification_report()` — sklearn 格式报告

#### 5.2 可视化工具 ✅
- [x] `src/utils/visualization.py`
  - `plot_spectral_signature()` — 单像素/均值光谱曲线
  - `plot_confusion_matrix()` — 归一化热力图
  - `plot_classification_map()` — GT vs Prediction 并排
  - `plot_training_curve()` — Loss + Metric 曲线

---

### Phase 6：测试与文档

#### 6.1 单元测试
- [ ] `tests/__init__.py`
- [ ] `tests/test_sam_loss.py`
- [ ] `tests/test_encoder.py`
- [ ] `tests/test_mae.py`
- [ ] `tests/test_dataset.py`

#### 6.2 示例数据脚本
- [ ] `scripts/generate_synthetic_hsi.py` — 生成合成 HSI 数据用于快速验证

---

## 四、实现顺序（推荐）

```
Step 1:  目录结构 + __init__.py
Step 2:  utils/ (config, seed, logger)
Step 3:  losses/sam.py
Step 4:  datasets/hsi_dataset.py
Step 5:  augmentations/hsi_mask.py
Step 6:  models/encoder.py
Step 7:  models/mae.py
Step 8:  models/classifier.py
Step 9:  train/pretrain.py
Step 10: train/finetune.py
Step 11: scripts/run_pretrain.py
Step 12: scripts/run_finetune.py
Step 13: utils/metrics.py + visualization.py
Step 14: tests/
Step 15: synthetic data 验证脚本
```

---

## 五、关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 数据格式 | `.npy` / `.mat` | 通用且高效 |
| Mask 实现 | 随机 mask + 可配置比例 | 灵活可控 |
| SAM Loss | normalize 后求余弦距离 | 数值稳定 |
| Encoder | Conv2d 架构 | 适合 HSI 空间结构 |
| 日志 | loguru | 简洁无配置 |
| 实验跟踪 | wandb（可选） | 可插拔 |

---

## 六、验收标准

- [ ] `python scripts/run_pretrain.py --data_path <path> --epochs 10` 可正常运行
- [ ] `python scripts/run_finetune.py --encoder_path encoder.pt --data_path <path>` 可正常运行
- [ ] SAM Loss 在合成数据上下降
- [ ] Encoder 输出形状正确
- [ ] 所有核心模块有单元测试
- [ ] 无 lint 错误（ruff）

---

## 七、Roadmap（超出本次实现范围）

- Patch-based MAE with ViT
- Spectral Transformer
- 多模态预训练（HSI + RGB）
- 分布式训练支持
- ONNX 导出
