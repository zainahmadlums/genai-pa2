"""Plot training curves for one or more run directories (each containing metrics.jsonl).

    python scripts/plot_metrics.py --run_dir outputs/sft outputs/dpo_hh outputs/dpo_ultrafeedback outputs/persona_lora \
        --out_dir outputs/plots

Writes ``<run>_curves.png`` and ``<run>_eval_table.csv`` per run. Run directories without
a metrics file are skipped with a message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.plotting import plot_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", nargs="+", required=True)
    parser.add_argument("--out_dir", default="outputs/plots")
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="smoothing factor for training curves")
    return parser


def main(argv=None) -> list[Path]:
    args = build_parser().parse_args(argv)
    written = []
    for run_dir in args.run_dir:
        if not (Path(run_dir) / "metrics.jsonl").exists():
            print(f"[plot] skipping {run_dir}: no metrics.jsonl")
            continue
        path = plot_run(run_dir, args.out_dir, args.ema_alpha)
        print(f"[plot] wrote {path}")
        written.append(path)
    return written


if __name__ == "__main__":
    main()
