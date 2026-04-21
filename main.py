#!/usr/bin/env python
"""HSI-MAE 交互式入口."""

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
_env_file = _current_dir / ".env"
_env_local = _current_dir / ".env.local"


def _load_env() -> None:
    """Load environment variables from .env files.

    Priority: .env.local if exists, otherwise .env.
    .env is the committed example file.
    .env.local is ignored by git, for local overrides.
    """
    if _env_local.exists():
        _load_dotenv(_env_local)
        print("[env] loaded: .env.local")
    elif _env_file.exists():
        _load_dotenv(_env_file)
        print("[env] loaded: .env")
    else:
        print("[env] no .env files found, using defaults")


def _load_dotenv(path: Path) -> None:
    """Parse a .env file and export variables to os.environ."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] in ('"', "'"):
                value = value[1:-1]
            os.environ[key] = value


_load_env()


def _run_script(script: str, extra_args: list[str] | None = None) -> int:
    """Run a Python script via subprocess."""
    cmd = [sys.executable, str(_current_dir / "scripts" / script), *(extra_args or [])]
    return subprocess.run(cmd, cwd=_current_dir).returncode


def _env(key: str) -> str:
    return os.environ.get(key, "")


MENU = """
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

> """


def action_gen_data() -> None:
    print("\n--- 生成合成数据 ---")
    bands = input(f"光谱波段数 (默认 {_env('BANDS')}): ").strip()
    bands = bands or _env("BANDS")
    classes = input(f"类别数 (默认 {_env('NUM_CLASSES')}): ").strip()
    classes = classes or _env("NUM_CLASSES")
    seed = input(f"随机种子 (默认 {_env('SEED')}): ").strip()
    seed = seed or _env("SEED")
    height = input("图像高度 (默认 64): ").strip() or "64"
    width = input("图像宽度 (默认 64): ").strip() or "64"

    args = [
        "--bands",
        bands,
        "--num-classes",
        classes,
        "--seed",
        seed,
        "--height",
        height,
        "--width",
        width,
        "--output-dir",
        "data",
    ]
    _run_script("generate_synthetic_hsi.py", args)


def action_pretrain() -> None:
    print("\n--- 预训练 HSI-MAE ---")
    data_path = input(f"数据路径 (默认 {_env('DATA_PATH')}): ").strip()
    data_path = data_path or _env("DATA_PATH")

    bands = input(f"光谱波段数 (默认 {_env('BANDS')}): ").strip()
    bands = bands or _env("BANDS")

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

    save_dir = input(f"保存目录 (默认 {_env('SAVE_DIR')}): ").strip()
    save_dir = save_dir or _env("SAVE_DIR")

    device = input(f"设备 (cuda/cpu, 默认 {_env('DEVICE')}): ").strip()
    device = device or _env("DEVICE")

    args = [
        "--data-path",
        data_path,
        "--bands",
        bands,
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
    print("\n--- 微调分类 ---")
    data_path = input(f"数据路径 (默认 {_env('DATA_PATH')}): ").strip()
    data_path = data_path or _env("DATA_PATH")

    labels_path = input(f"标签路径 (默认 {_env('LABELS_PATH')}): ").strip()
    labels_path = labels_path or _env("LABELS_PATH")

    encoder_path = input(f"预训练模型路径 (默认 {_env('ENCODER_PATH')}): ").strip()
    encoder_path = encoder_path or _env("ENCODER_PATH")

    bands = input(f"光谱波段数 (默认 {_env('BANDS')}): ").strip()
    bands = bands or _env("BANDS")

    num_classes = input(f"类别数 (默认 {_env('NUM_CLASSES')}): ").strip()
    num_classes = num_classes or _env("NUM_CLASSES")

    mode = input(
        f"微调模式 (linear_probe / full, 默认 {_env('FINETUNE_MODE')}): "
    ).strip()
    mode = mode or _env("FINETUNE_MODE")

    epochs = input(f"训练轮数 (默认 {_env('EPOCHS')}): ").strip()
    epochs = epochs or _env("EPOCHS")

    batch = input(f"Batch size (默认 {_env('BATCH_SIZE')}): ").strip()
    batch = batch or _env("BATCH_SIZE")

    save_dir = input(f"保存目录 (默认 {_env('SAVE_DIR')}): ").strip()
    save_dir = save_dir or _env("SAVE_DIR")

    args = [
        "--data-path",
        data_path,
        "--labels-path",
        labels_path,
        "--encoder-path",
        encoder_path,
        "--bands",
        bands,
        "--num-classes",
        num_classes,
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


def action_tests() -> None:
    print("\n--- 运行测试 ---")
    pytest_args = ["pytest", str(_current_dir / "tests"), "-v"]
    subprocess.run(pytest_args, cwd=_current_dir)


def action_lint() -> None:
    print("\n--- 代码检查 ---")
    ruff_args = ["uv", "run", "ruff", "check", "."]
    subprocess.run(ruff_args, cwd=_current_dir)


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
    path = _current_dir / ".env.local"
    lines = ["# HSI-MAE 本地配置 (不提交到 git)\n"]
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


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    actions: dict[str, Callable[[], None]] = {
        "1": action_gen_data,
        "2": action_pretrain,
        "3": action_finetune,
        "4": action_tests,
        "5": action_lint,
        "6": action_env,
        "7": action_export_env,
    }

    print("HSI-MAE: 高光谱图像基础模型")
    print(f"工作目录: {_current_dir}")

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
