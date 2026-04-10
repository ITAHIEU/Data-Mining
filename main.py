"""
main.py - Entry point để chạy toàn bộ pipeline Data Mining.

Thứ tự chạy:
    1. Process.py      -> Tiền xử lý và merge dữ liệu
    2. run_eda.py      -> Phân tích khám phá dữ liệu (EDA)
    3. run_topic_analysis.py -> Chạy mô hình và phân tích
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(script_name: str) -> None:
    """Chạy một bước trong pipeline."""
    base_dir = Path(__file__).resolve().parent
    script_path = base_dir / script_name

    if not script_path.exists():
        print(f"[ERROR] File not found: {script_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  RUNNING: {script_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(base_dir),
    )

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[OK] {script_name} completed successfully.")


def main() -> None:
    """Chạy toàn bộ pipeline Data Mining."""
    print("=" * 60)
    print("  DATA MINING PIPELINE - AI JOB MARKET ANALYSIS")
    print("  Nhóm: Nguyễn Lê Đức Hiếu - Nguyễn Thị Thúy Nga - Nguyễn Huỳnh Thái Bảo")
    print("=" * 60)

    steps = [
        "Process.py",
        "run_eda.py",
        "run_topic_analysis.py",
    ]

    for step in steps:
        run_step(step)

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    print("  Check results/ folder for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
