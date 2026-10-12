"""Supervised fine-tuning task (response-only loss) for the custom training loop."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from alignment.batching import build_lm_batch
from alignment.losses import SFT_REDUCTION, forward_logits, token_logprobs_from_logits
from alignment.prompt_format import DEFAULT_TEMPLATE, format_prompt
from alignment.train_utils import AmpSettings, save_checkpoint


@dataclass
class SFTTaskConfig:
    max_length: int = 512
    max_prompt_length: int | None = None
    template: str = DEFAULT_TEMPLATE
    reduction: str = SFT_REDUCTION
    eval_batch_size: int = 2


class SFTTask:
    """Rows are dicts with ``instruction`` and ``response``."""

    def __init__(self, model, tokenizer, val_rows, config: SFTTaskConfig, device, precision: str = "fp16",
                 checkpoint_meta: dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.val_rows = list(val_rows)
        self.config = config
        self.device = torch.device(device)
        self.amp = AmpSettings(self.device, precision)
        self.checkpoint_meta = dict(checkpoint_meta or {})

    def collate(self, rows) -> dict[str, torch.Tensor]:
        prompts = [format_prompt(row["instruction"], self.config.template) for row in rows]
        responses = [row["response"] for row in rows]
        return build_lm_batch(
            self.tokenizer, prompts, responses, self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )

    def _token_logps(self, batch):
        logits = forward_logits(self.model, batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
        return token_logprobs_from_logits(logits, batch["labels"])

    def loss(self, batch):
        token_logps, mask = self._token_logps(batch)
        tokens = mask.sum()
        if self.config.reduction == "token_mean":
            loss = -token_logps.sum() / tokens.clamp_min(1)
        elif self.config.reduction == "example_sum_mean":
            loss = -token_logps.sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown SFT reduction {self.config.reduction!r}")
        return loss, {"response_tokens": float(tokens)}

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Validation loss as mean NLL per response token over the whole validation set."""

        self.model.eval()
        total_nll, total_tokens, total_rows = 0.0, 0, 0
        for start in range(0, len(self.val_rows), self.config.eval_batch_size):
            batch = self.collate(self.val_rows[start:start + self.config.eval_batch_size])
            with self.amp.autocast():
                token_logps, mask = self._token_logps(batch)
            total_nll += float(-token_logps.double().sum())
            total_tokens += int(mask.sum())
            total_rows += batch["input_ids"].shape[0]
        val_loss = total_nll / max(1, total_tokens)
        return {
            "val_loss": val_loss,
            "val_perplexity": math.exp(min(val_loss, 50.0)),
            "val_nll_per_example": total_nll / max(1, total_rows),
            "val_response_tokens": total_tokens,
            "val_rows": total_rows,
        }

    def selection_key(self, metrics: dict) -> tuple:
        return (-metrics["val_loss"],)

    def save(self, directory, meta: dict) -> None:
        payload = {**self.checkpoint_meta, **meta, "stage": "sft", "template": self.config.template,
                   "max_length": self.config.max_length, "sft_reduction": self.config.reduction}
        save_checkpoint(self.model, self.tokenizer, directory, payload)
