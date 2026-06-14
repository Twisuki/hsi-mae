#!/usr/bin/env python
"""Indian Pines 数据集预处理工具.

将 .mat 文件转换为 .npy 格式，与 src.datasets.HSIDataset 兼容.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indian Pines 数据集预处理")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "indian-pines"),
        help="原始 .mat 文件所在目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "indian-pines"),
        help="输出目录",
    )
    return parser.parse_args()


def find_mat_files(input_dir: Path) -> tuple[Path | None, Path | None]:
    """查找 .mat 文件."""
    data_files = list(input_dir.glob("*corrected.mat")) + list(input_dir.glob("*Corrected.mat"))
    gt_files = list(input_dir.glob("*gt.mat")) + list(input_dir.glob("*GT.mat"))

    data_path = data_files[0] if data_files else None
    gt_path = gt_files[0] if gt_files else None

    return data_path, gt_path


def load_raw_data(input_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """从 .mat 文件加载 Indian Pines 数据."""
    from scipy.io import loadmat

    data_path, gt_path = find_mat_files(input_dir)

    if data_path is None or gt_path is None:
        raise FileNotFoundError(
            f"找不到 .mat 文件，请确保以下文件存在于 {input_dir}:\n"
            "  - *corrected.mat (光谱数据)\n"
            "  - *gt.mat (标签数据)"
        )

    print(f"加载数据: {data_path.name}")
    data = loadmat(data_path)

    # 查找数据键
    data_key = None
    for k in data.keys():
        if not k.startswith("__"):
            data_key = k
            break

    if data_key is None:
        raise ValueError("无法从 .mat 文件中找到数据键")

    hsi_data = data[data_key]
    print(f"  形状: {hsi_data.shape}, 类型: {hsi_data.dtype}")

    print(f"加载标签: {gt_path.name}")
    gt_data = loadmat(gt_path)

    gt_key = None
    for k in gt_data.keys():
        if not k.startswith("__"):
            gt_key = k
            break

    if gt_key is None:
        raise ValueError("无法从 .mat 文件中找到标签键")

    labels = gt_data[gt_key]
    print(f"  形状: {labels.shape}, 类型: {labels.dtype}")

    # 验证类别数
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels) - (1 if 0 in unique_labels else 0)
    print(f"  类别数: {num_classes} (不含背景)")

    return hsi_data, labels


def preprocess_and_save(
    hsi_data: np.ndarray, labels: np.ndarray, output_dir: Path
) -> None:
    """预处理数据并保存为 .npy 格式.

    输出格式与 src.datasets.HSIDataset 兼容:
    - data: [H, W, C] float32, 归一化到 [0, 1]
    - labels: [H, W] int64, 类别索引 0-15 (原始 1-16 减 1)
    """
    print("\n预处理数据...")

    # 转换数据类型
    hsi_data = hsi_data.astype(np.float32)
    labels = labels.astype(np.int64)

    # 将标签从 1-16 转换为 0-15 (PyTorch CrossEntropyLoss 要求 0-indexed)
    # 0 保持为背景
    labels = np.where(labels > 0, labels - 1, labels)

    # 逐波段归一化到 [0, 1]
    print("归一化...")
    orig_shape = hsi_data.shape
    h, w, c = hsi_data.shape
    flat = hsi_data.reshape(-1, c)

    mins = flat.min(axis=0, keepdims=True)
    maxs = flat.max(axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (flat - mins) / ranges

    hsi_data = normalized.reshape(orig_shape).astype(np.float32)

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "indian_pines.npy"
    labels_path = output_dir / "indian_pines_gt.npy"

    np.save(data_path, hsi_data)
    np.save(labels_path, labels)

    print("\n数据已保存:")
    print(f"  数据: {data_path} (形状: {hsi_data.shape})")
    print(f"  标签: {labels_path} (形状: {labels.shape})")

    # 打印类别分布
    print("\n类别分布:")
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
        "Oats", "Soybeans-notill", "Soybeans-mintill", "Soybeans-clean",
        "Wheat", "Woods", "Buildings-grass-trees", "Stone-steel-towers",
    ]
    for u, c in zip(unique, counts):
        if u == 0:
            print(f"  背景 (0): {c} 像素")
        else:
            idx = u - 1
            name = class_names[idx] if idx < len(class_names) else f"Class_{u}"
            print(f"  {name} ({u}): {c} 像素")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=== Indian Pines 数据集预处理 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}\n")

    # 加载并预处理
    hsi_data, labels = load_raw_data(input_dir)
    preprocess_and_save(hsi_data, labels, output_dir)

    print("\n预处理完成!")
    print("\n现在可以使用以下命令进行预训练:")
    print(f"  python scripts/run_pretrain.py --data-path {output_dir / 'indian_pines.npy'}")
    print("\n或使用交互式入口:")
    print("  python example/indian-pines/main.py")


if __name__ == "__main__":
    main()
