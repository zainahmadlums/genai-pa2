"""Persona track: TRL ``DPOTrainer`` + PEFT LoRA on HuggingFaceTB/SmolLM2-360M-Instruct.

This is the only part of the project that uses TRL / PEFT training code (the assignment
requires it here and forbids it for SFT and the two from-scratch DPO runs).

    python scripts/train_persona_lora.py --config configs/persona_lora.yaml --smoke
    python scripts/train_persona_lora.py --config configs/persona_lora.yaml

Written against ``trl==1.14.1``: ``DPOTrainer(model=, args=DPOConfig, train_dataset=,
eval_dataset=, processing_class=, peft_config=)``. With ``peft_config`` and no
``ref_model``, TRL uses the base model with the adapter disabled as the frozen reference.
Rows are converted to TRL's conversational preference format, so TRL applies the model's
own chat template.

Outputs under ``output_dir``: ``adapter/`` (LoRA weights + tokenizer), ``metrics.jsonl``,
``trainer_log_history.json``, ``data_split.json``, ``run_config.json``, ``TRAINING_COMPLETE.json``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# The Hugging Face Trainer wraps the model in DataParallel when it sees two GPUs, which
# is unnecessary for 220 pairs. Pin one GPU before torch is imported (override by
# exporting CUDA_VISIBLE_DEVICES yourself).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.config import load_config  # noqa: E402
from alignment.data_prep import load_eval_instruction_keys, normalize_text  # noqa: E402
from alignment.engine import already_complete  # noqa: E402
from alignment.prompt_format import CHAT_TEMPLATE  # noqa: E402
from alignment.train_utils import COMPLETE_MARKER, CHECKPOINT_META, append_jsonl, read_json, write_json  # noqa: E402

SMOKE = {"num_train_epochs": 1, "max_steps": 2, "eval_steps": 1, "logging_steps": 1, "val_size": 4, "train_limit": 8}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/persona_lora.yaml")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--force", action="store_true", help="retrain even if this run already completed")
    parser.add_argument("--smoke", action="store_true", help="2 optimizer steps on 8 pairs into <output_dir>_smoke")
    return parser


def to_conversational(row: dict) -> dict:
    """TRL conversational preference format with an explicit prompt."""

    return {
        "prompt": [{"role": "user", "content": row["prompt"].strip()}],
        "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
        "rejected": [{"role": "assistant", "content": row["rejected"].strip()}],
    }


def split_rows(rows: list[dict], val_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle; the first ``val_size`` rows are validation."""

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    val = [rows[index] for index in order[:val_size]]
    train = [rows[index] for index in order[val_size:]]
    return train, val


def selected_checkpoint_step(best_model_checkpoint: str | None, final_step: int) -> int:
    """Training step of the checkpoint the Trainer reloaded (``.../checkpoint-<step>``), else the final step."""

    if best_model_checkpoint:
        tail = Path(best_model_checkpoint).name.rsplit("-", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return final_step


def log_history_to_metrics(log_history: list[dict], metrics_path: Path) -> None:
    """Rewrite the Trainer log in the metrics.jsonl schema used by the custom runs."""

    if metrics_path.exists():
        metrics_path.unlink()
    for entry in log_history:
        step = entry.get("step")
        if "eval_loss" in entry:
            append_jsonl(metrics_path, {
                "type": "eval", "step": step, "epoch": entry.get("epoch"), "val_loss": entry["eval_loss"],
                "val_reward_accuracy": entry.get("eval_rewards/accuracies"),
                "val_reward_margin": entry.get("eval_rewards/margins"),
            })
        elif "loss" in entry:
            append_jsonl(metrics_path, {
                "type": "train", "step": step, "epoch": entry.get("epoch"), "loss": entry["loss"],
                "lr": entry.get("learning_rate"), "grad_norm": entry.get("grad_norm"),
                "reward_accuracy": entry.get("rewards/accuracies"), "reward_margin": entry.get("rewards/margins"),
            })


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.override)
    if args.smoke:
        config.update(SMOKE)
        config["output_dir"] = str(config["output_dir"]).rstrip("/") + "_smoke"
    output_dir = Path(config["output_dir"])

    finished = already_complete(output_dir)
    if finished and not args.force:
        print(f"[persona] {output_dir} is already complete; use --force to retrain")
        return finished

    import torch
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    seed = int(config.get("seed", 42))
    rows = read_json(config["data_file"])
    eval_keys = load_eval_instruction_keys([config.get("eval_prompts_file", "eval_sets/persona_eval_prompts.json")])
    overlapping = [row["prompt"] for row in rows if normalize_text(row["prompt"]) in eval_keys]
    if overlapping:
        raise SystemExit(f"{len(overlapping)} training prompts also appear in the persona evaluation set")
    train_rows, val_rows = split_rows(rows, int(config.get("val_size", 20)), seed)
    if config.get("train_limit"):
        train_rows = train_rows[: int(config["train_limit"])]
    train_dataset = Dataset.from_list([to_conversational(row) for row in train_rows])
    eval_dataset = Dataset.from_list([to_conversational(row) for row in val_rows])

    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    select_best = bool(config.get("select_best", True)) and not args.smoke
    eval_steps = int(config.get("eval_steps", 10))
    training_args = DPOConfig(
        output_dir=str(output_dir / "trainer"),
        beta=float(config.get("beta", 0.1)),
        max_length=int(config.get("max_length", 512)),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_train_epochs=float(config["num_train_epochs"]),
        max_steps=int(config.get("max_steps", -1)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        logging_steps=int(config.get("logging_steps", 5)),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps" if select_best else "no",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=select_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # T4 GPUs: fp16 autocast. TRL's own default would be bf16.
        fp16=use_cuda,
        bf16=False,
        gradient_checkpointing=False,
        report_to="none",
        seed=seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[persona] trl={trl.__version__} train={len(train_rows)} val={len(val_rows)} "
          f"trainable={trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", {**config, "stage": "persona_lora_dpo", "trl": trl.__version__,
                                                "torch": torch.__version__, "trainable_parameters": trainable,
                                                "total_parameters": total})
    write_json(output_dir / "data_split.json", {
        "seed": seed, "train_prompts": [row["prompt"] for row in train_rows],
        "val_prompts": [row["prompt"] for row in val_rows],
    })

    initial_eval = trainer.evaluate()
    train_result = trainer.train()
    final_step = trainer.state.global_step
    # The learning curve ends here. With load_best_model_at_end the weights in memory are now those of
    # the selected checkpoint, so any evaluation below describes THAT checkpoint, not the last training
    # step, and must not be added to the curve or reported as "final".
    training_log = list(trainer.state.log_history)
    selected_step = selected_checkpoint_step(trainer.state.best_model_checkpoint, final_step) if select_best else final_step
    selected_eval = trainer.evaluate()
    final_step_metrics = next((entry for entry in reversed(training_log)
                               if "eval_loss" in entry and entry.get("step") == final_step), None)
    if not select_best:
        final_step_metrics = selected_eval  # no checkpoint was reloaded: the saved adapter is the final step

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_json(adapter_dir / CHECKPOINT_META, {
        "stage": "persona_lora_dpo", "template": CHAT_TEMPLATE, "base_model": config["model_name"],
        "beta": float(config.get("beta", 0.1)), "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "step": selected_step, "final_training_step": final_step, "metrics": selected_eval,
    })
    write_json(output_dir / "trainer_log_history.json", training_log)
    # Curve = the evaluation before any update (step 0) followed by the evaluations made during training.
    log_history_to_metrics([{**initial_eval, "step": 0, "epoch": 0.0}, *training_log], output_dir / "metrics.jsonl")
    summary = {
        "run_name": config.get("run_name", "persona_lora"),
        "completed": True,
        "optimizer_steps": final_step,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_loss": train_result.training_loss,
        "initial_eval": initial_eval,
        # The saved adapter: which training step it comes from and its validation metrics.
        "selection": "lowest validation DPO loss (eval_loss)" if select_best else "last training step",
        "best_step": selected_step,
        "best_metrics": selected_eval,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        # Validation metrics of the LAST training step, only if an evaluation ran at that step (else null).
        "final_step": final_step,
        "final_metrics": final_step_metrics,
        "adapter": str(adapter_dir),
    }
    write_json(output_dir / COMPLETE_MARKER, summary)
    print(f"[persona] adapter saved to {adapter_dir} (step {selected_step} of {final_step}); "
          f"eval loss {initial_eval.get('eval_loss')} -> {selected_eval.get('eval_loss')}")
    return summary


if __name__ == "__main__":
    main()
