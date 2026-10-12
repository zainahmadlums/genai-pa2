"""Shared training utilities: seeding, AMP, schedules, JSONL I/O, and checkpoint writing."""

from __future__ import annotations

import math
import os
import random
import shutil
import time
from pathlib import Path

import torch

from alignment.io_utils import (  # noqa: F401 - re-exported for trainers and scripts
    CHECKPOINT_META,
    COMPLETE_MARKER,
    append_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def resolve_device(name: str | None = None) -> torch.device:
    """``auto``/``None`` -> cuda:0 when available, else cpu. Missing GPUs fall back to an existing device."""

    if name in (None, "", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            return torch.device("cuda:0")
    return device


class AmpSettings:
    """Mixed-precision policy: fp16/bf16 autocast on CUDA, plain fp32 elsewhere.

    Trainable weights always stay in fp32; only the forward computation is autocast.
    A GradScaler is used for fp16 (not needed for bf16 or fp32).
    """

    def __init__(self, device: torch.device, precision: str = "fp16"):
        if precision not in ("fp16", "bf16", "fp32"):
            raise ValueError("precision must be fp16, bf16 or fp32")
        self.device = torch.device(device)
        self.requested = precision
        self.enabled = precision != "fp32" and self.device.type == "cuda"
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.use_scaler = self.enabled and precision == "fp16"

    @property
    def effective(self) -> str:
        return self.requested if self.enabled else "fp32"

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.enabled)

    def make_scaler(self):
        try:
            return torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=self.use_scaler)


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    """Multiplier applied to the peak learning rate for 0-indexed optimizer step ``step``."""

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        return 1.0 - progress
    raise ValueError(f"Unknown schedule {schedule!r}; expected cosine, linear or constant")


def build_optimizer(model, name: str, learning_rate: float, weight_decay: float):
    """AdamW or RMSprop over trainable parameters; weight decay skips biases and norm weights."""

    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    groups = [group for group in groups if group["params"]]
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=learning_rate, alpha=0.99, eps=1e-8)
    raise ValueError(f"Unknown optimizer {name!r}; expected adamw or rmsprop")


def epoch_order(num_rows: int, seed: int, epoch: int) -> list[int]:
    """Deterministic shuffled row order for one epoch (reproducible across resumes)."""

    generator = torch.Generator().manual_seed(int(seed) + 1000 * int(epoch))
    return torch.randperm(num_rows, generator=generator).tolist()


def accumulation_groups(num_rows: int, batch_size: int, grad_accum_steps: int) -> list[list[tuple[int, int]]]:
    """Split one epoch into optimizer steps.

    Returns a list of groups; each group is a list of ``(start, end)`` slices into the
    epoch order, one per micro-batch. The last group may hold fewer micro-batches and the
    last micro-batch fewer rows; the trainer divides by the true group size.
    """

    slices = [(start, min(start + batch_size, num_rows)) for start in range(0, num_rows, batch_size)]
    return [slices[index:index + grad_accum_steps] for index in range(0, len(slices), grad_accum_steps)]


def save_checkpoint(model, tokenizer, out_dir, meta: dict | None = None) -> Path:
    """Write model (+ tokenizer) to ``out_dir`` atomically, replacing any previous copy.

    Hugging Face models are written with ``save_pretrained`` (fp32 safetensors). Plain
    ``nn.Module`` objects (unit tests) fall back to a ``state_dict`` file.
    """

    out_dir = Path(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(tmp_dir), safe_serialization=True)
    else:
        torch.save(model.state_dict(), tmp_dir / "pytorch_model.bin")
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(tmp_dir))
    payload = dict(meta or {})
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(tmp_dir / CHECKPOINT_META, payload)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)
    return out_dir


def load_checkpoint_meta(path) -> dict:
    meta_path = Path(path) / CHECKPOINT_META
    return read_json(meta_path) if meta_path.exists() else {}


def disk_free_gb(path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1024**3
    except OSError:
        return float("nan")
