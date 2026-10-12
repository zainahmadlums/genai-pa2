"""From-scratch Direct Preference Optimization task for the custom training loop.

The trainable policy and the frozen reference both start from the same SFT checkpoint.
On Kaggle T4 x2 the policy lives on ``cuda:0`` and the reference on ``cuda:1``; with one
device (or CPU) both share it. The reference never receives gradients: its parameters
have ``requires_grad=False``, it stays in eval mode, and its forward pass runs under
``torch.no_grad()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from alignment.batching import build_dpo_batch, concat_lm_batches
from alignment.losses import (
    dpo_loss_from_logps,
    dpo_logits,
    dpo_metrics,
    forward_logits,
    sequence_logprobs_from_logits,
)
from alignment.prompt_format import DEFAULT_TEMPLATE, format_prompt
from alignment.train_utils import AmpSettings, save_checkpoint


@dataclass
class DPOTaskConfig:
    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int | None = 256
    template: str = DEFAULT_TEMPLATE
    eval_batch_size: int = 4
    single_forward: bool = True


def freeze_reference(reference_model):
    """Put the reference in eval mode and disable gradients for all of its parameters."""

    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    return reference_model


class DPOTask:
    """Rows are dicts with ``instruction``, ``chosen`` and ``rejected``."""

    def __init__(self, policy, reference, tokenizer, val_rows, config: DPOTaskConfig, policy_device,
                 reference_device=None, precision: str = "fp16", checkpoint_meta: dict | None = None):
        self.policy = policy
        self.reference = freeze_reference(reference)
        self.tokenizer = tokenizer
        self.val_rows = list(val_rows)
        self.config = config
        self.policy_device = torch.device(policy_device)
        self.reference_device = torch.device(reference_device if reference_device is not None else policy_device)
        self.amp = AmpSettings(self.policy_device, precision)
        self.reference_amp = AmpSettings(self.reference_device, precision)
        self.checkpoint_meta = dict(checkpoint_meta or {})
        self._reference_val_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def collate(self, rows) -> dict:
        prompts = [format_prompt(row["instruction"], self.config.template) for row in rows]
        return build_dpo_batch(
            self.tokenizer, prompts, [row["chosen"] for row in rows], [row["rejected"] for row in rows],
            self.config.max_length, max_prompt_length=self.config.max_prompt_length, same_width=True,
        )

    def _pair_logps(self, model, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed response log-probabilities for the chosen and rejected rows."""

        if self.config.single_forward:
            # One forward pass over [chosen rows; rejected rows]; padding is masked, so the
            # result matches two separate passes.
            joined = concat_lm_batches(batch["chosen"], batch["rejected"])
            logits = forward_logits(model, joined["input_ids"].to(device), joined["attention_mask"].to(device))
            logps = sequence_logprobs_from_logits(logits, joined["labels"])
            rows = batch["chosen"]["input_ids"].shape[0]
            return logps[:rows], logps[rows:]
        results = []
        for side in ("chosen", "rejected"):
            logits = forward_logits(model, batch[side]["input_ids"].to(device), batch[side]["attention_mask"].to(device))
            results.append(sequence_logprobs_from_logits(logits, batch[side]["labels"]))
        return results[0], results[1]

    @torch.no_grad()
    def reference_logps(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        self.reference.eval()
        with self.reference_amp.autocast():
            chosen, rejected = self._pair_logps(self.reference, batch, self.reference_device)
        return chosen.detach().float(), rejected.detach().float()

    def loss(self, batch):
        # The reference pass is launched first so that, with two GPUs, it overlaps with
        # the policy pass. Only the policy graph is kept.
        reference_chosen, reference_rejected = self.reference_logps(batch)
        policy_chosen, policy_rejected = self._pair_logps(self.policy, batch, self.policy_device)
        policy_chosen, policy_rejected = policy_chosen.float(), policy_rejected.float()
        loss = dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, self.config.beta)
        with torch.no_grad():
            metrics = dpo_metrics(policy_chosen, policy_rejected, reference_chosen.to(policy_chosen.device),
                                  reference_rejected.to(policy_chosen.device), self.config.beta)
        stats = {"reward_accuracy": float(metrics["reward_accuracy"]), "reward_margin": float(metrics["reward_margin"])}
        return loss, stats

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Validation DPO loss, reward accuracy (strict ``r_w > r_l``) and reward margin."""

        self.policy.eval()
        batches = [
            self.collate(self.val_rows[start:start + self.config.eval_batch_size])
            for start in range(0, len(self.val_rows), self.config.eval_batch_size)
        ]
        if self._reference_val_cache is None:
            # The reference is frozen, so its validation log-probabilities are computed once.
            self._reference_val_cache = [tuple(t.cpu() for t in self.reference_logps(batch)) for batch in batches]
        policy_chosen, policy_rejected, reference_chosen, reference_rejected = [], [], [], []
        for batch, (ref_chosen, ref_rejected) in zip(batches, self._reference_val_cache):
            with self.amp.autocast():
                chosen, rejected = self._pair_logps(self.policy, batch, self.policy_device)
            policy_chosen.append(chosen.float().cpu())
            policy_rejected.append(rejected.float().cpu())
            reference_chosen.append(ref_chosen)
            reference_rejected.append(ref_rejected)
        policy_chosen, policy_rejected = torch.cat(policy_chosen), torch.cat(policy_rejected)
        reference_chosen, reference_rejected = torch.cat(reference_chosen), torch.cat(reference_rejected)
        beta = self.config.beta
        logits = dpo_logits(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
        metrics = dpo_metrics(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
        return {
            "val_loss": float(dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)),
            "val_reward_accuracy": float(metrics["reward_accuracy"]),
            "val_reward_margin": float(metrics["reward_margin"]),
            "val_tie_fraction": float(logits.eq(0).float().mean()),
            "val_policy_chosen_logp": float(policy_chosen.mean()),
            "val_policy_rejected_logp": float(policy_rejected.mean()),
            "val_reference_chosen_logp": float(reference_chosen.mean()),
            "val_reference_rejected_logp": float(reference_rejected.mean()),
            "val_pairs": int(policy_chosen.numel()),
        }

    def selection_key(self, metrics: dict) -> tuple:
        # Best checkpoint = highest validation reward accuracy; reward margin breaks ties.
        return (metrics["val_reward_accuracy"], metrics["val_reward_margin"])

    def save(self, directory, meta: dict) -> None:
        payload = {**self.checkpoint_meta, **meta, "stage": "dpo", "template": self.config.template,
                   "max_length": self.config.max_length, "max_prompt_length": self.config.max_prompt_length,
                   "beta": self.config.beta}
        save_checkpoint(self.policy, self.tokenizer, directory, payload)
