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
    """加载环境变量."""
    env_file = _root_dir / ".env"
    env_local = _root_dir / ".env.local"

    if env_local.exists():
        _load_dotenv(env_local)
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
DEFAULT_DATA_PATH = str(DATA_DIR / "indian_pines.npy")
DEFAULT_LABELS_PATH = str(DATA_DIR / "indian_pines_gt.npy")
DEFAULT_ENCODER_PATH = str(_root_dir / "checkpoints" / "indian_pines_encoder.pt")
DEFAULT_SAVE_DIR = str(_root_dir / "checkpoints")


MENU = """
=== Indian Pines 数据集入口 ===

请选择操作:

  [1] 下载 Indian Pines 数据集
  [2] 预处理 Indian Pines (mat → npy)
  [3] 预训练 HSI-MAE (Indian Pines)
  [4] 微调分类 (Indian Pines)
  [5] 查看当前环境变量
  [6] 导出配置到 .env.local

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

    data_path = input(f"数据路径 (默认 {DEFAULT_DATA_PATH}): ").strip()
    data_path = data_path or DEFAULT_DATA_PATH

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

    save_dir = input(f"保存目录 (默认 {DEFAULT_SAVE_DIR}): ").strip()
    save_dir = save_dir or DEFAULT_SAVE_DIR

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

    data_path = input(f"数据路径 (默认 {DEFAULT_DATA_PATH}): ").strip()
    data_path = data_path or DEFAULT_DATA_PATH

    labels_path = input(f"标签路径 (默认 {DEFAULT_LABELS_PATH}): ").strip()
    labels_path = labels_path or DEFAULT_LABELS_PATH

    encoder_path = input(f"预训练模型路径 (默认 {DEFAULT_ENCODER_PATH}): ").strip()
    encoder_path = encoder_path or DEFAULT_ENCODER_PATH

    mode = input(
        f"微调模式 (linear_probe / full, 默认 {_env('FINETUNE_MODE')}): "
    ).strip()
    mode = mode or _env("FINETUNE_MODE")

    epochs = input(f"训练轮数 (默认 {_env('EPOCHS')}): ").strip()
    epochs = epochs or _env("EPOCHS")

    batch = input(f"Batch size (默认 {_env('BATCH_SIZE')}): ").strip()
    batch = batch or _env("BATCH_SIZE")

    save_dir = input(f"保存目录 (默认 {DEFAULT_SAVE_DIR}): ").strip()
    save_dir = save_dir or DEFAULT_SAVE_DIR

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
    ]
    _run_script("run_finetune.py", args)


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
    print("将当前值写入 .env.local, 优先级高于 .env, 不提交到 git")
    path = _root_dir / ".env.local"
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
        "5": action_env,
        "6": action_export_env,
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