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

#### 1.1 目录结构创建
- [ ] 创建 `src/datasets/`
- [ ] 创建 `src/models/`
- [ ] 创建 `src/augmentations/`
- [ ] 创建 `src/losses/`
- [ ] 创建 `src/train/`
- [ ] 创建 `src/utils/`
- [ ] 创建 `src/__init__.py`
- [ ] 创建 `scripts/`
- [ ] 更新 `.gitignore`

#### 1.2 配置管理系统 `src/utils/`
- [ ] `src/utils/__init__.py`
- [ ] `src/utils/config.py` — 参数配置类（pretrain / finetune）
- [ ] `src/utils/seed.py` — 随机种子固定
- [ ] `src/utils/logger.py` — 日志工具（loguru 封装）
- [ ] `src/utils/__init__.py` 导出

#### 1.3 数据集加载 `src/datasets/`
- [ ] `src/datasets/__init__.py`
- [ ] `src/datasets/hsi_dataset.py`
  - 支持 `.npy` / `.mat` 格式
  - 自动归一化到 [0, 1]
  - 支持 train/val split
  - 返回 `(B, H, W, C)` 或 `(B, C, H, W)`

---

### Phase 2：核心模型实现

#### 2.1 数据增强 `src/augmentations/`
- [ ] `src/augmentations/__init__.py`
- [ ] `src/augmentations/hsi_mask.py`
  - `spectral_mask()` — 光谱维度随机 mask
  - `spatial_mask()` — 空间维度随机 mask
  - `joint_mask()` — 3D joint masking
  - 统一接口 `MaskStrategy.apply(x) → x_masked, mask`

#### 2.2 损失函数 `src/losses/`
- [ ] `src/losses/__init__.py`
- [ ] `src/losses/sam.py`
  - `sam_loss(pred, target)` — 光谱角映射损失
  - `mse_sam_loss(pred, target, lambda_sam=0.1)` — 组合损失
  - 支持 batch 和 spatial 维度

#### 2.3 模型定义 `src/models/`

##### Encoder
- [ ] `src/models/__init__.py`
- [ ] `src/models/encoder.py`
  - `HSIEncoder` — Conv2d 光谱 + 空间编码
  - 支持自定义 `bands`, `dim`, `num_layers`
  - 可选 `pretrained` 加载路径

##### MAE
- [ ] `src/models/mae.py`
  - `HSIMAE` — Encoder + Decoder
  - `forward(x)` — mask → encode → decode → reconstruct
  - 分离 `encode()` / `decode()` 方法

##### Classifier
- [ ] `src/models/classifier.py`
  - `HSIClassifier` — 线性分类头
  - `HSIFineTuner` — Encoder + Classifier 联合

---

### Phase 3：训练流程

#### 3.1 预训练 `src/train/`
- [ ] `src/train/__init__.py`
- [ ] `src/train/pretrain.py`
  - `PretrainEngine` 类
  - 支持 `mask_ratio` / `lr` / `epochs` / `device`
  - 日志输出（loss, lr, grad_norm）
  - 保存 `encoder.pt`

#### 3.2 微调 `src/train/`
- [ ] `src/train/finetune.py`
  - `FinetuneEngine` 类
  - 支持 `LinearProbe` / `FullFineTune` 两种模式
  - 计算 OA / AA / Kappa / F1
  - 保存 `classifier.pt`

#### 3.3 数据增强工厂
- [ ] `src/train/augmentations.py`
  - `TrainAugmentation` — 训练时增强策略
  - `ValAugmentation` — 验证时无增强

---

### Phase 4：入口脚本

#### 4.1 预训练脚本
- [ ] `scripts/run_pretrain.py`
  - argparse 参数解析
  - 调用 `PretrainEngine`
  - wandb / loguru 日志集成

#### 4.2 微调脚本
- [ ] `scripts/run_finetune.py`
  - argparse 参数解析
  - 调用 `FinetuneEngine`
  - 评估指标输出

---

### Phase 5：评估与可视化

#### 5.1 评估工具
- [ ] `src/utils/metrics.py`
  - `overall_accuracy()`
  - `average_accuracy()`
  - `kappa_coefficient()`
  - `f1_scores()`
  - `classification_report()`

#### 5.2 可视化工具
- [ ] `src/utils/visualization.py`
  - `plot_spectral_signature()` — 光谱曲线
  - `plot_confusion_matrix()` — 混淆矩阵
  - `plot_classification_map()` — 分类结果图

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
