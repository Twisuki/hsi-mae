#!/usr/bin/env python
"""Indian Pines 数据集交互式入口."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

# -------------------------------------------------------------------------
# Env loading (must be before other imports that need os.environ)
# -------------------------------------------------------------------------

_current_dir = Path(__file__).parent.resolve()
_root_dir = _current_dir.parent.parent  # 项目根目录


def _load_env() -> None:
    """加载环境变量.

    优先级（从高到低）:
    1. 本文件夹的 .env.local (example/indian-pines/.env.local)
    2. 项目根目录的 .env.local
    3. 项目根目录的 .env
    """
    env_local_self = _current_dir / ".env.local"
    env_local_root = _root_dir / ".env.local"
    env_file = _root_dir / ".env"

    if env_local_self.exists():
        _load_dotenv(env_local_self)
        print("[env] loaded: example/indian-pines/.env.local")
    elif env_local_root.exists():
        _load_dotenv(env_local_root)
        print("[env] loaded: .env.local")
    elif env_file.exists():
        _load_dotenv(env_file)
        print("[env] loaded: .env")
    else:
        print("[env] no .env files found, using defaults")


def _load_dotenv(path: Path) -> None:
    """解析 .env 文件并导出变量到 os.environ."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] in ('"', "'"):
                value = value[1:-1]
            os.environ[key] = value


_load_env()


def _run_script(script: str, extra_args: list[str] | None = None) -> int:
    """通过 subprocess 运行 Python 脚本."""
    cmd = [sys.executable, str(_root_dir / "scripts" / script), *(extra_args or [])]
    return subprocess.run(cmd, cwd=_root_dir).returncode


def _env(key: str) -> str:
    return os.environ.get(key, "")


# Indian Pines 数据集参数
BANDS = 200
NUM_CLASSES = 16
DATA_DIR = _root_dir / "data" / "indian-pines"

CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybeans-notill", "Soybeans-mintill", "Soybeans-clean",
    "Wheat", "Woods", "Buildings-grass-trees", "Stone-steel-towers",
]


def _get_defaults() -> dict[str, str]:
    """获取默认路径（在环境变量加载后调用）。"""
    return {
        "data_path": _env("DATA_PATH") or str(DATA_DIR / "indian_pines.npy"),
        "labels_path": _env("LABELS_PATH") or str(DATA_DIR / "indian_pines_gt.npy"),
        "encoder_path": _env("ENCODER_PATH") or str(_root_dir / "checkpoints" / "indian_pines_encoder.pt"),
        "save_dir": _env("SAVE_DIR") or str(_root_dir / "checkpoints"),
    }


MENU = """
=== Indian Pines 数据集入口 ===

请选择操作:

  [1] 下载 Indian Pines 数据集
  [2] 预处理 Indian Pines (mat → npy)
  [3] 预训练 HSI-MAE (Indian Pines)
  [4] 微调分类 (Indian Pines)
  [5] 查看可视化结果
  [6] 查看当前环境变量
  [7] 导出配置到 .env.local

  [0] 退出

> """


def action_download() -> None:
    print("\n--- 下载 Indian Pines 数据集 ---")
    script = _current_dir / "download.py"
    subprocess.run([sys.executable, str(script)], cwd=_root_dir)


def action_preprocess() -> None:
    print("\n--- 预处理 Indian Pines 数据集 ---")
    script = _current_dir / "preprocess.py"
    subprocess.run([sys.executable, str(script)], cwd=_root_dir)


def action_pretrain() -> None:
    print("\n--- 预训练 HSI-MAE (Indian Pines) ---")
    defaults = _get_defaults()

    data_path = input(f"数据路径 (默认 {defaults['data_path']}): ").strip()
    data_path = data_path or defaults["data_path"]

    dim = input(f"Encoder 维度 (默认 {_env('ENCODER_DIM')}): ").strip()
    dim = dim or _env("ENCODER_DIM")

    layers = input(f"Encoder 层数 (默认 {_env('ENCODER_LAYERS')}): ").strip()
    layers = layers or _env("ENCODER_LAYERS")

    mask = input(f"Mask 比例 (默认 {_env('MASK_RATIO')}): ").strip()
    mask = mask or _env("MASK_RATIO")

    epochs = input(f"训练轮数 (默认 {_env('EPOCHS')}): ").strip()
    epochs = epochs or _env("EPOCHS")

    batch = input(f"Batch size (默认 {_env('BATCH_SIZE')}): ").strip()
    batch = batch or _env("BATCH_SIZE")

    save_dir = input(f"保存目录 (默认 {defaults['save_dir']}): ").strip()
    save_dir = save_dir or defaults["save_dir"]

    device = input(f"设备 (cuda/cpu, 默认 {_env('DEVICE')}): ").strip()
    device = device or _env("DEVICE")

    args = [
        "--data-path",
        data_path,
        "--bands",
        str(BANDS),
        "--encoder-dim",
        dim,
        "--encoder-layers",
        layers,
        "--mask-ratio",
        mask,
        "--epochs",
        epochs,
        "--batch-size",
        batch,
        "--save-dir",
        save_dir,
        "--device",
        device,
    ]
    _run_script("run_pretrain.py", args)


def action_finetune() -> None:
    print("\n--- 微调分类 (Indian Pines) ---")
    defaults = _get_defaults()

    data_path = input(f"数据路径 (默认 {defaults['data_path']}): ").strip()
    data_path = data_path or defaults["data_path"]

    labels_path = input(f"标签路径 (默认 {defaults['labels_path']}): ").strip()
    labels_path = labels_path or defaults["labels_path"]

    encoder_path = input(f"预训练模型路径 (默认 {defaults['encoder_path']}): ").strip()
    encoder_path = encoder_path or defaults["encoder_path"]

    mode = input(
        f"微调模式 (linear_probe / full, 默认 {_env('FINETUNE_MODE')}): "
    ).strip()
    mode = mode or _env("FINETUNE_MODE")

    epochs = input(f"训练轮数 (默认 {_env('EPOCHS')}): ").strip()
    epochs = epochs or _env("EPOCHS")

    batch = input(f"Batch size (默认 {_env('BATCH_SIZE')}): ").strip()
    batch = batch or _env("BATCH_SIZE")

    save_dir = input(f"保存目录 (默认 {defaults['save_dir']}): ").strip()
    save_dir = save_dir or defaults["save_dir"]

    args = [
        "--data-path",
        data_path,
        "--labels-path",
        labels_path,
        "--encoder-path",
        encoder_path,
        "--bands",
        str(BANDS),
        "--num-classes",
        str(NUM_CLASSES),
        "--mode",
        mode,
        "--epochs",
        epochs,
        "--batch-size",
        batch,
        "--save-dir",
        save_dir,
        "--class-names",
        ",".join(CLASS_NAMES),
    ]
    _run_script("run_finetune.py", args)


def action_visualize() -> None:
    """查看可视化结果."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    defaults = _get_defaults()
    save_dir = Path(defaults["save_dir"])
    image_files = {
        "预训练损失曲线": save_dir / "training_curve.png",
        "预训练重建进度": save_dir / "reconstruction_progress.png",
        "微调训练曲线": save_dir / "finetune_training_curve.png",
        "混淆矩阵": save_dir / "confusion_matrix.png",
    }

    print("\n--- 可视化结果 ---")
    for name, path in image_files.items():
        if path.exists():
            print(f"  {name}: {path}")
        else:
            print(f"  {name}: (未找到)")

    print("\n输入要查看的图片编号 (1-4)，或按回车返回:")
    choice = input("> ").strip()

    if choice == "1" and image_files["预训练损失曲线"].exists():
        img = plt.imread(image_files["预训练损失曲线"])
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title("预训练损失曲线")
        plt.show()
    elif choice == "2" and image_files["预训练重建进度"].exists():
        img = plt.imread(image_files["预训练重建进度"])
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("预训练重建进度")
        plt.show()
    elif choice == "3" and image_files["微调训练曲线"].exists():
        img = plt.imread(image_files["微调训练曲线"])
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title("微调训练曲线")
        plt.show()
    elif choice == "4" and image_files["混淆矩阵"].exists():
        img = plt.imread(image_files["混淆矩阵"])
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.title("混淆矩阵")
        plt.show()


def action_env() -> None:
    print("\n--- 当前环境变量 ---")
    keys = sorted(os.environ)
    for k in keys:
        if any(
            k.startswith(p)
            for p in [
                "DATA",
                "BANDS",
                "ENCODER",
                "MASK",
                "NUM",
                "EPOCHS",
                "BATCH",
                "LR",
                "WEIGHT",
                "WARMUP",
                "DEVICE",
                "SAVE",
                "LOG",
                "SEED",
                "FINETUNE",
            ]
        ):
            print(f"  {k}={os.environ[k]}")


def action_export_env() -> None:
    print("\n--- 导出配置到 .env.local ---")
    print("将当前值写入本文件夹的 .env.local, 优先级最高, 不提交到 git")
    path = _current_dir / ".env.local"
    lines = ["# Indian Pines 数据集配置\n"]
    for key in [
        "DATA_PATH",
        "LABELS_PATH",
        "BANDS",
        "ENCODER_DIM",
        "ENCODER_LAYERS",
        "MASK_RATIO",
        "NUM_CLASSES",
        "EPOCHS",
        "BATCH_SIZE",
        "LR",
        "WEIGHT_DECAY",
        "WARMUP_EPOCHS",
        "ENCODER_PATH",
        "DEVICE",
        "NUM_WORKERS",
        "SAVE_DIR",
        "LOG_INTERVAL",
        "SEED",
        "FINETUNE_MODE",
    ]:
        current_val = _env(key) if _env(key) else ""
        current = input(f"{key} [{current_val}]: ").strip()
        val = current or _env(key)
        lines.append(f"{key}={val}\n")

    path.write_text("".join(lines), encoding="utf-8")
    print(f"已写入: {path}")
    print("下次运行 main.py 时会自动加载")


def main() -> None:
    actions: dict[str, Callable[[], None]] = {
        "1": action_download,
        "2": action_preprocess,
        "3": action_pretrain,
        "4": action_finetune,
        "5": action_visualize,
        "6": action_env,
        "7": action_export_env,
    }

    print("Indian Pines: 高光谱图像分类基准数据集")
    print(f"项目根目录: {_root_dir}")
    print(f"数据目录: {DATA_DIR}")

    while True:
        try:
            choice = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出.")
            break

        if choice == "0" or choice in ("q", "quit", "exit"):
            print("再见.")
            break

        if choice in actions:
            try:
                actions[choice]()
            except Exception as e:
                print(f"[error] {e}")
        else:
            print("无效选择, 请重试.")


if __name__ == "__main__":
    main()
