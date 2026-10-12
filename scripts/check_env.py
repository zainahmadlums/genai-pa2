"""Print the environment facts the assignment asks for: package versions, GPUs, disk.

    python scripts/check_env.py
    python scripts/check_env.py --require_gpus 2     # exit with an error unless two GPUs are visible
"""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys


def version_of(module_name: str) -> str:
    try:
        return str(getattr(importlib.import_module(module_name), "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        return f"not available ({type(exc).__name__})"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--require_gpus", type=int, default=0)
    parser.add_argument("--working_dir", default=".")
    args = parser.parse_args(argv)

    print(f"python       {sys.version.split()[0]} ({platform.platform()})")
    for name in ("torch", "transformers", "datasets", "accelerate", "peft", "trl", "huggingface_hub",
                 "kaggle_benchmarks"):
        print(f"{name:<12} {version_of(name)}")

    gpu_count = 0
    try:
        import torch

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"cuda         available={torch.cuda.is_available()} build={torch.version.cuda} gpus={gpu_count}")
        for index in range(gpu_count):
            properties = torch.cuda.get_device_properties(index)
            print(f"  cuda:{index}     {properties.name} {properties.total_memory / 1024**3:.1f} GiB")
    except ImportError:
        print("cuda         torch not installed")

    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        print(result.stdout.strip() or result.stderr.strip())
    else:
        print("nvidia-smi   not found")

    usage = shutil.disk_usage(args.working_dir)
    print(f"disk         {usage.free / 1024**3:.1f} GiB free of {usage.total / 1024**3:.1f} GiB at {args.working_dir}")
    if gpu_count < args.require_gpus:
        raise SystemExit(f"Expected {args.require_gpus} GPUs but found {gpu_count}. In Kaggle choose Accelerator: GPU T4 x2.")


if __name__ == "__main__":
    main()
