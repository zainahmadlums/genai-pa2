"""Response-only log-probabilities, the SFT loss, and the DPO objective.

Causal alignment: the logits at position ``t`` predict the token at ``t + 1``, so
``logits[:, :-1]`` is scored against ``labels[:, 1:]``. Positions whose label is ``-100``
(prompt and padding) contribute nothing. A response token at position 0 (empty prompt)
has no preceding context and is therefore never scored.

SFT reduction (used everywhere in this project): mean negative log-likelihood over all
unmasked response tokens in the batch (``token_mean``). The alternative accepted by the
assignment, the mean of per-example summed NLL, is available as ``example_sum_mean``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from alignment.batching import IGNORE_INDEX, build_dpo_batch, build_lm_batch

SFT_REDUCTION = "token_mean"


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Run a causal LM and return ``(batch, seq, vocab)`` logits."""

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        outputs = model(input_ids)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, Mapping):
        return outputs["logits"]
    return outputs[0]


def token_logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position label log-probabilities and the scored-position mask.

    Returns ``(token_logps, mask)`` of shape ``(batch, seq - 1)``; masked positions are 0.
    """

    labels = labels.to(logits.device)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    if shift_logits.dtype in (torch.float16, torch.bfloat16):
        # Normalise over the vocabulary in fp32 even when the forward pass ran in half precision.
        shift_logits = shift_logits.float()
    mask = shift_labels.ne(IGNORE_INDEX)
    batch_size, steps, vocab_size = shift_logits.shape
    negative_logps = F.cross_entropy(
        shift_logits.reshape(batch_size * steps, vocab_size),
        shift_labels.reshape(batch_size * steps),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(batch_size, steps)
    return -negative_logps, mask


def sequence_logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Summed response-token log-probability per row, shape ``(batch,)``."""

    token_logps, _ = token_logprobs_from_logits(logits, labels)
    return token_logps.sum(dim=-1)


def sequence_logprobs(model, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Summed response log-probability per row for an LM batch, on the model's device."""

    device = model_device(model)
    logits = forward_logits(model, batch["input_ids"].to(device), batch["attention_mask"].to(device))
    return sequence_logprobs_from_logits(logits, batch["labels"])


def sft_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, reduction: str = SFT_REDUCTION) -> torch.Tensor:
    """Scalar response-only SFT loss.

    ``token_mean``: sum of response-token NLL divided by the number of scored tokens.
    ``example_sum_mean``: mean over rows of each row's summed response NLL.
    """

    token_logps, mask = token_logprobs_from_logits(logits, labels)
    if reduction == "token_mean":
        return -token_logps.sum() / mask.sum().clamp_min(1)
    if reduction == "example_sum_mean":
        return -token_logps.sum(dim=-1).mean()
    raise ValueError(f"Unknown SFT reduction {reduction!r}")


def sft_loss(model, batch: Mapping[str, torch.Tensor], reduction: str = SFT_REDUCTION) -> torch.Tensor:
    device = model_device(model)
    logits = forward_logits(model, batch["input_ids"].to(device), batch["attention_mask"].to(device))
    return sft_loss_from_logits(logits, batch["labels"], reduction=reduction)


def reference_sequence_logprobs(reference_model, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Reference log-probabilities with gradients disabled and the model in eval mode."""

    was_training = bool(getattr(reference_model, "training", False))
    if was_training:
        reference_model.eval()
    try:
        with torch.no_grad():
            return sequence_logprobs(reference_model, batch).detach()
    finally:
        if was_training:
            reference_model.train()


def _as_tensor(values, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    if like is not None:
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)
    return torch.as_tensor(values, dtype=torch.float32)


def dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """``beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))`` per example.

    Reference terms are treated as constants (detached) and moved to the policy device.
    """

    policy_chosen_logps = _as_tensor(policy_chosen_logps)
    policy_rejected_logps = _as_tensor(policy_rejected_logps, like=policy_chosen_logps)
    reference_chosen_logps = _as_tensor(reference_chosen_logps, like=policy_chosen_logps).detach()
    reference_rejected_logps = _as_tensor(reference_rejected_logps, like=policy_chosen_logps).detach()
    reference_chosen_logps = reference_chosen_logps.to(device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
    reference_rejected_logps = reference_rejected_logps.to(
        device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype
    )
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    return beta * (policy_gap - reference_gap)


def dpo_loss_from_logps(
    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
) -> torch.Tensor:
    """Scalar batch-mean DPO loss ``mean(-logsigmoid(beta * (policy_gap - reference_gap)))``.

    ``logsigmoid`` is used instead of ``log(sigmoid(.))`` so large gaps do not overflow.
    """

    logits = dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta)
    return -F.logsigmoid(logits).mean()


def dpo_rewards(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """Implicit rewards ``r_w = beta (log pi(y_w) - log pi_ref(y_w))`` and ``r_l`` likewise."""

    policy_chosen_logps = _as_tensor(policy_chosen_logps)
    reward_chosen = beta * (policy_chosen_logps - _as_tensor(reference_chosen_logps, like=policy_chosen_logps).to(policy_chosen_logps.device))
    policy_rejected_logps = _as_tensor(policy_rejected_logps, like=policy_chosen_logps)
    reward_rejected = beta * (
        policy_rejected_logps - _as_tensor(reference_rejected_logps, like=policy_chosen_logps).to(policy_chosen_logps.device)
    )
    return reward_chosen, reward_rejected


def dpo_metrics(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """Reward accuracy (strict ``r_w > r_l``; exact ties count as incorrect) and reward margin."""

    reward_chosen, reward_rejected = dpo_rewards(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )
    return {
        "reward_accuracy": (reward_chosen > reward_rejected).float().mean(),
        "reward_margin": (reward_chosen - reward_rejected).mean(),
    }


# ---------------------------------------------------------------------------
# Text-level entry points used by submission_adapter.py. They pad to ``max_length``.
# ---------------------------------------------------------------------------


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    batch = build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True)
    return sequence_logprobs(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length, reduction: str = SFT_REDUCTION) -> torch.Tensor:
    batch = build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True)
    return sft_loss(model, batch, reduction=reduction)


def compute_dpo_loss(
    policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length
) -> torch.Tensor:
    batch = build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to_max_length=True
    )
    policy_chosen_logps = sequence_logprobs(policy_model, batch["chosen"])
    policy_rejected_logps = sequence_logprobs(policy_model, batch["rejected"])
    reference_chosen_logps = reference_sequence_logprobs(reference_model, batch["chosen"])
    reference_rejected_logps = reference_sequence_logprobs(reference_model, batch["rejected"])
    return dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )
