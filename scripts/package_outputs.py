"""Zip the small, auditable artifacts of one or more directories under /kaggle/working.

    python scripts/package_outputs.py --source outputs/sft --zip_path /kaggle/working/sft_run_outputs.zip
    python scripts/package_outputs.py --source outputs/eval outputs/plots --zip_path /kaggle/working/eval_outputs.zip

Model weights and resume states (``*.safetensors``, ``*.bin``, ``*.pt``) are skipped by
default: they stay in the notebook output and zipping them would double disk use. Add
``--include_weights`` only when there is room.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="+", required=True, help="directories or files to archive")
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--include_weights", action="store_true")
    args = parser.parse_args(argv)

    zip_path = Path(args.zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count, skipped, total_bytes = 0, 0, 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in map(Path, args.source):
            files = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
            for path in files:
                if path.resolve() == zip_path.resolve():
                    continue
                if path.suffix in WEIGHT_SUFFIXES and not args.include_weights:
                    skipped += 1
                    continue
                archive.write(path, arcname=str(path))
                count += 1
                total_bytes += path.stat().st_size
    print(f"[package] {zip_path}: {count} files, {total_bytes / 1e6:.1f} MB before compression, "
          f"{skipped} weight files skipped")
    return zip_path


if __name__ == "__main__":
    main()
