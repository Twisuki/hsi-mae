#!/usr/bin/env python
"""Indian Pines 数据集下载工具.

Indian Pines 是经典的高光谱图像分类基准数据集, 包含 145x145 像素, 200 个光谱波段, 16 个地物类别.

数据集下载地址: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indian Pines 数据集下载")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "indian-pines"),
        help="输出目录",
    )
    return parser.parse_args()


def download_indian_pines(output_dir: Path) -> None:
    """从网上下载 Indian Pines 数据集.

    数据来源: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
    """
    import ssl
    import urllib.request

    # 创建不验证 SSL 证书的上下文
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Indian Pines 数据集文件 URL
    # 来源: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
    urls = {
        "Indian_pines_corrected.mat": "https://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "Indian_pines_gt.mat": "https://www.ehu.es/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in urls.items():
        filepath = output_dir / filename
        if filepath.exists():
            print(f"已存在: {filename}")
            continue

        print(f"正在下载 {filename}...")
        print(f"来源: {url}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ssl_context) as response:
                with open(filepath, "wb") as f:
                    f.write(response.read())
            print(f"下载完成: {filename}")
        except Exception as e:
            print(f"下载失败: {e}")
            # 清理已下载的文件
            if filepath.exists():
                filepath.unlink()
            print("\n请手动下载数据集:")
            print(
                "  1. 访问 https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes"
            )
            print("  2. 下载 Indian Pines 数据集")
            print(f"  3. 将 .mat 文件放置到 {output_dir} 目录")
            sys.exit(1)

    # 列出下载的文件
    print("\n下载的文件:")
    for f in output_dir.iterdir():
        if f.suffix == ".mat":
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.2f} MB)")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=== Indian Pines 数据集下载 ===")
    print(f"输出目录: {output_dir}\n")

    download_indian_pines(output_dir)

    print("\n下载完成!")
    print("\n接下来运行预处理脚本:")
    print("  python example/indian-pines/preprocess.py")


if __name__ == "__main__":
    main()
