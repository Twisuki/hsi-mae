# Loss=NaN 问题排查报告

## 问题描述

微调（Fine-tuning）阶段出现 loss=NaN，导致训练完全失败，无法完成分类任务。

## 排查过程

### 1. 定位来源

通过 `debug_nan.py` 诊断脚本逐步检查：
- 加载 `final_encoder.pt` 后，encoder state 已被污染
- encoder 所有参数均包含 NaN
- classifier 参数正常（未感染）
- 向前传播时 encoder 输出即为 NaN
- 反向传播时梯度也全为 NaN

### 2. 检查各 Checkpoint 健康状态

| Checkpoint | has_nan |
|------------|---------|
| `best_encoder.pt` | False |
| `last_encoder.pt` | False |
| `final_encoder.pt` | **True** |

`best_encoder.pt` 和 `last_encoder.pt` 状态健康，只有 `final_encoder.pt` 包含 NaN。

### 3. 追踪 root cause

`.env.local` 配置路径为：
```
ENCODER_PATH=checkpoints/indian_pines_encoder.pt
```

该文件不存在，fallback 逻辑降级到 `final_encoder.pt`：
```python
# run_finetune.py 原始 fallback 逻辑
if not encoder_path.exists():
    fallback = Path(args.save_dir) / "final_encoder.pt"
    if fallback.exists():
        encoder_path = fallback
```

而 `final_encoder.pt` 本身已被 NaN 污染。

### 4. 根本原因

预训练结束时 `run_pretrain.py` 保存最终模型的方式存在问题：

```python
# scripts/run_pretrain.py
torch.save({"encoder_state": engine.model.encoder.state_dict()}, final_path)
```

训练完成后的 `engine.model.encoder` 权重已经被后续优化步骤中的梯度污染（产生 NaN）。在保存 `final_encoder.pt` 之前，模型可能经历了某个产生 NaN 的 step，但该 NaN 没有被及时检测和跳过。

而 `best_encoder.pt` 和 `last_encoder.pt` 是在每个 epoch 结束前保存的快照，保存时模型状态健康，所以不包含 NaN。

## 解决方案

### 1. 修复 fallback 逻辑（`scripts/run_finetune.py`）

增加健康检查，优先使用不包含 NaN 的 checkpoint：

```python
encoder_path = Path(args.encoder_path)
if not encoder_path.exists():
    for candidate in ["best_encoder.pt", "last_encoder.pt", "final_encoder.pt"]:
        fallback = Path(args.save_dir) / candidate
        if fallback.exists():
            ckpt = torch.load(fallback, map_location="cpu", weights_only=False)
            state = ckpt["encoder_state"]
            has_nan = torch.isnan(torch.cat([v.flatten() for v in state.values()])).any().item()
            if not has_nan:
                encoder_path = fallback
                logger.info(f"Using healthy checkpoint: {encoder_path}")
                break
            else:
                logger.warning(f"Checkpoint {fallback} contains NaN, trying next...")
    else:
        raise FileNotFoundError(
            f"Encoder not found: {args.encoder_path} (no healthy checkpoints available)"
        )
```

### 2. 更新 `.env.local` 配置

```bash
ENCODER_PATH=checkpoints/best_encoder.pt
```

直接使用健康且验证过性能最佳的 `best_encoder.pt`。

## 修复验证

修复后运行 4 epochs 测试：

**预训练**：
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 4.0217 | 1.3024 |
| 2 | 0.4503 | 0.1480 |
| 3 | 0.0747 | 0.0438 |
| 4 | 0.0285 | 0.0220 |

**微调**：
| Epoch | Train Acc | Val OA | Val AA | Val Kappa |
|-------|----------|--------|--------|-----------|
| 1 | 64.03% | 42.77% | 25.92% | 0.2651 |
| 2 | 72.73% | 62.75% | 37.27% | 0.4642 |
| 3 | 75.30% | **66.56%** | **53.16%** | **0.5444** |
| 4 | 77.12% | 58.99% | 33.67% | 0.3540 |

**最终测试集**：OA=65.48%, AA=53.27%, Kappa=0.5363, F1-macro=0.4836

无 NaN，训练正常完成。

## 预防建议

1. **保存 checkpoint 前增加 NaN 检测**：在 `_save_checkpoint` 中对 `state_dict` 进行 NaN 检查，异常时跳过保存并报警
2. **统一使用 `best_encoder.pt`**：`run_pretrain.py` 预训练结束后直接复制 `best_encoder.pt` 到 `final_encoder.pt`，而非使用 `engine.model.encoder.state_dict()`
3. **fallback 时必须验证 checkpoint 健康性**：所有 fallback 路径都应检查 NaN，避免加载损坏的权重
4. **长期方案**：在 `PretrainEngine.fit()` 结束时统一保存，使用验证集最优模型作为最终输出