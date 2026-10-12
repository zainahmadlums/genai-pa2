"""Supervised fine-tuning of Qwen/Qwen2.5-0.5B with the hand-written training loop.

Examples:
    python scripts/train_sft.py --config configs/sft.yaml --smoke
    python scripts/train_sft.py --config configs/sft.yaml
    python scripts/train_sft.py --config configs/sft.yaml --resume
    python scripts/train_sft.py --config configs/sft.yaml --override learning_rate=1e-5 output_dir=outputs/sft_lr1e-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.config import load_config, pick  # noqa: E402
from alignment.engine import LoopConfig, already_complete, remove_smoke_checkpoints, run_training  # noqa: E402
from alignment.sft_trainer import SFTTask, SFTTaskConfig  # noqa: E402
from alignment.train_utils import read_jsonl, resolve_device, write_json  # noqa: E402

SMOKE = {"train_size": 64, "val_size": 16, "max_steps": 4, "eval_every_steps": 2, "log_every_steps": 1,
         "resume_every_steps": 0, "grad_accum_steps": 4}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--resume", action="store_true", help="continue from <output_dir>/resume/state.pt if present")
    parser.add_argument("--force", action="store_true", help="retrain even if this run already completed")
    parser.add_argument("--smoke", action="store_true", help="tiny run (64 rows, 4 optimizer steps) into <output_dir>_smoke")
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.override)
    if args.smoke:
        config.update(SMOKE)
        config["output_dir"] = str(config["output_dir"]).rstrip("/") + "_smoke"
        config["run_name"] = config.get("run_name", "sft") + "_smoke"

    finished = already_complete(config["output_dir"])
    if finished and not args.force:
        print(f"[sft] {config['output_dir']} is already complete (best step {finished.get('best_step')}); use --force to retrain")
        return finished

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(config.get("device", "auto"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # Trainable weights stay in fp32; fp16 is applied only through autocast.
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float32).to(device)
    model.config.use_cache = False

    train_rows = read_jsonl(config["train_file"])[: int(config["train_size"])]
    val_rows = read_jsonl(config["val_file"])[: int(config["val_size"])]
    print(f"[sft] model={config['model_name']} device={device} train={len(train_rows)} val={len(val_rows)}")

    task_config = pick(config, SFTTaskConfig)
    loop_config = pick(config, LoopConfig)
    meta = {"base_model": config["model_name"], "train_file": config["train_file"], "val_file": config["val_file"],
            "train_rows": len(train_rows), "val_rows": len(val_rows), "seed": loop_config.seed}
    task = SFTTask(model, tokenizer, val_rows, task_config, device, loop_config.precision, checkpoint_meta=meta)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    write_json(Path(config["output_dir"]) / "run_config.json", {**config, "stage": "sft", "torch": torch.__version__})
    summary = run_training(loop_config, model, train_rows, task, device, resume=args.resume)
    if args.smoke:
        remove_smoke_checkpoints(config["output_dir"])
    return summary


if __name__ == "__main__":
    main()
