"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.

The implementation lives in ``alignment/batching.py`` (batch construction) and
``alignment/losses.py`` (log-probabilities, SFT loss, DPO loss). The training loops in
``alignment/sft_trainer.py`` and ``alignment/dpo_trainer.py`` call the same functions.
"""

from __future__ import annotations

from alignment import batching, losses


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return batching.build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return losses.compute_response_logprobs(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    # Reduction: mean over all unmasked response tokens in the batch (losses.SFT_REDUCTION).
    return losses.compute_sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return batching.build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to_max_length=True
    )


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return losses.dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return losses.compute_dpo_loss(
        policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length
    )
