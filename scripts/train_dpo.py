"""From-scratch DPO starting from the SFT best checkpoint (policy trainable, reference frozen).

The same script runs both preference tracks; only the config differs.

Examples:
    python scripts/train_dpo.py --config configs/dpo_hh.yaml --smoke
    python scripts/train_dpo.py --config configs/dpo_hh.yaml
    python scripts/train_dpo.py --config configs/dpo_ultrafeedback.yaml --resume
    python scripts/train_dpo.py --config configs/dpo_hh.yaml --override optimizer=rmsprop output_dir=outputs/dpo_hh_rmsprop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.config import load_config, pick  # noqa: E402
from alignment.dpo_trainer import DPOTask, DPOTaskConfig  # noqa: E402
from alignment.engine import LoopConfig, already_complete, remove_smoke_checkpoints, run_training  # noqa: E402
from alignment.prompt_format import DEFAULT_TEMPLATE  # noqa: E402
from alignment.train_utils import load_checkpoint_meta, read_jsonl, resolve_device, write_json  # noqa: E402

SMOKE = {"train_size": 32, "val_size": 8, "max_steps": 4, "eval_every_steps": 2, "log_every_steps": 1,
         "resume_every_steps": 0, "grad_accum_steps": 4}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="configs/dpo_hh.yaml or configs/dpo_ultrafeedback.yaml")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--resume", action="store_true", help="continue from <output_dir>/resume/state.pt if present")
    parser.add_argument("--force", action="store_true", help="retrain even if this run already completed")
    parser.add_argument("--smoke", action="store_true", help="tiny run (32 pairs, 4 optimizer steps) into <output_dir>_smoke")
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.override)
    if args.smoke:
        config.update(SMOKE)
        config["output_dir"] = str(config["output_dir"]).rstrip("/") + "_smoke"
        config["run_name"] = config.get("run_name", "dpo") + "_smoke"

    finished = already_complete(config["output_dir"])
    if finished and not args.force:
        print(f"[dpo] {config['output_dir']} is already complete (best step {finished.get('best_step')}); use --force to retrain")
        return finished

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft_checkpoint = config["sft_checkpoint"]
    sft_meta = load_checkpoint_meta(sft_checkpoint)
    sft_template = sft_meta.get("template")
    if not config.get("template"):
        config["template"] = sft_template or DEFAULT_TEMPLATE
    elif sft_template and config["template"] != sft_template:
        raise SystemExit(
            f"Config template {config['template']!r} differs from the SFT checkpoint template {sft_template!r}. "
            "SFT, DPO and evaluation must share one prompt format."
        )

    policy_device = resolve_device(config.get("policy_device", "auto"))
    reference_device = resolve_device(config.get("reference_device", "cuda:1"))
    if reference_device.type != policy_device.type:
        reference_device = policy_device
    reference_dtype = {"fp32": torch.float32, "fp16": torch.float16}[config.get("reference_dtype", "fp32")]
    if reference_device.type != "cuda":
        reference_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    # Policy: trainable fp32 weights. Reference: a second, frozen copy of the same checkpoint.
    policy = AutoModelForCausalLM.from_pretrained(sft_checkpoint, torch_dtype=torch.float32).to(policy_device)
    reference = AutoModelForCausalLM.from_pretrained(sft_checkpoint, torch_dtype=reference_dtype).to(reference_device)
    policy.config.use_cache = False
    reference.config.use_cache = False

    train_rows = read_jsonl(config["train_file"])[: int(config["train_size"])]
    val_rows = read_jsonl(config["val_file"])[: int(config["val_size"])]
    print(f"[dpo] sft_checkpoint={sft_checkpoint} policy={policy_device} reference={reference_device} "
          f"({config.get('reference_dtype', 'fp32')}) train={len(train_rows)} val={len(val_rows)} template={config['template']}")

    task_config = pick(config, DPOTaskConfig)
    loop_config = pick(config, LoopConfig)
    meta = {"base_model": sft_meta.get("base_model"), "sft_checkpoint": str(sft_checkpoint),
            "reference_checkpoint": str(sft_checkpoint), "sft_checkpoint_step": sft_meta.get("step"),
            "train_file": config["train_file"], "val_file": config["val_file"],
            "train_rows": len(train_rows), "val_rows": len(val_rows), "seed": loop_config.seed}
    task = DPOTask(policy, reference, tokenizer, val_rows, task_config, policy_device, reference_device,
                   loop_config.precision, checkpoint_meta=meta)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    write_json(Path(config["output_dir"]) / "run_config.json",
               {**config, "stage": "dpo", "policy_device": str(policy_device), "reference_device": str(reference_device),
                "torch": torch.__version__})
    summary = run_training(loop_config, policy, train_rows, task, policy_device, resume=args.resume)
    if args.smoke:
        remove_smoke_checkpoints(config["output_dir"])
    return summary


if __name__ == "__main__":
    main()
