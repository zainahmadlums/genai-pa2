"""Numerical checks for response log-probabilities, the SFT loss and the DPO objective."""

from __future__ import annotations

import math

import pytest
import torch

import submission_adapter as adapter
from alignment import losses
from alignment.batching import IGNORE_INDEX
from tests.helpers import BigramLM, manual_bigram_logprob
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_logprobs_use_next_token_alignment():
    """Token t+1 is scored with the logits produced at position t (checked against a hand computation)."""

    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size, seed=3, dtype=torch.float64)
    prompts = ["Hi: ", "A much longer prompt: ", "x"]
    responses = ["yes", "no", "maybe so"]
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=40)
    assert logps.dtype == torch.float64
    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        expected = manual_bigram_logprob(
            model, tokenizer.encode(prompt), tokenizer.encode(response) + [tokenizer.eos_token_id])
        torch.testing.assert_close(logps[row], expected, rtol=1e-10, atol=1e-10)


def test_truncated_rows_score_only_the_surviving_response_tokens():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size, seed=5, dtype=torch.float64)
    logps = adapter.compute_response_logprobs(model, tokenizer, ["abc: "], ["defghij"], max_length=8)
    expected = manual_bigram_logprob(model, tokenizer.encode("abc: "), tokenizer.encode("def"))  # no EOS left
    torch.testing.assert_close(logps[0], expected, rtol=1e-10, atol=1e-10)


def test_empty_prompt_never_scores_the_first_token():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size, seed=7, dtype=torch.float64)
    logps = adapter.compute_response_logprobs(model, tokenizer, [""], ["abc"], max_length=10)
    expected = manual_bigram_logprob(model, [], tokenizer.encode("abc") + [tokenizer.eos_token_id])
    torch.testing.assert_close(logps[0], expected, rtol=1e-10, atol=1e-10)


def test_prompt_and_padding_do_not_change_response_scores():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size, seed=11, dtype=torch.float64)
    # Same last prompt token and same response: a bigram model must give the same score.
    short = adapter.compute_response_logprobs(model, tokenizer, ["z: "], ["same"], max_length=12)
    long = adapter.compute_response_logprobs(model, tokenizer, ["a long unrelated prefix z: "], ["same"], max_length=64)
    torch.testing.assert_close(short, long, rtol=1e-10, atol=1e-10)


def test_sft_loss_is_the_token_mean_and_matches_cross_entropy():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size, seed=13, dtype=torch.float64)
    prompts, responses = ["Do: ", "Say this: "], ["a", "much longer"]
    loss = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=32)
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=32)
    tokens = sum(len(tokenizer.encode(response)) + 1 for response in responses)
    torch.testing.assert_close(loss, -logps.sum() / tokens, rtol=1e-10, atol=1e-10)

    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=32)
    logits = model(batch["input_ids"]).logits
    reference = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), batch["labels"][:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
    torch.testing.assert_close(loss, reference, rtol=1e-10, atol=1e-10)

    summed = losses.compute_sft_loss(model, tokenizer, prompts, responses, 32, reduction="example_sum_mean")
    torch.testing.assert_close(summed, -logps.mean(), rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError):
        losses.compute_sft_loss(model, tokenizer, prompts, responses, 32, reduction="other")


def test_half_precision_logits_are_normalised_in_float32():
    logits = torch.randn(2, 5, 7).to(torch.float16)
    labels = torch.tensor([[IGNORE_INDEX, 1, 2, 3, IGNORE_INDEX], [IGNORE_INDEX, IGNORE_INDEX, 4, 5, 6]])
    token_logps, mask = losses.token_logprobs_from_logits(logits, labels)
    assert token_logps.dtype == torch.float32
    assert mask.sum() == 6
    expected = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    torch.testing.assert_close(token_logps[0, 0], expected[0, 0, 1])
    assert token_logps[0, 3] == 0  # masked position contributes nothing


def test_dpo_loss_matches_the_formula_numerically():
    policy_chosen = torch.tensor([-1.0, -4.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -1.0, -2.0])
    reference_chosen = torch.tensor([-2.0, -3.0, -5.0])
    reference_rejected = torch.tensor([-2.5, -2.0, -1.0])
    beta = 0.5
    expected_logits = [0.5 * ((-1 + 3) - (-2 + 2.5)), 0.5 * ((-4 + 1) - (-3 + 2)), 0.5 * ((-2 + 2) - (-5 + 1))]
    assert expected_logits == [0.75, -1.0, 2.0]
    expected = sum(math.log(1 + math.exp(-value)) for value in expected_logits) / 3
    loss = adapter.compute_dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    assert loss.ndim == 0
    assert float(loss) == pytest.approx(expected, rel=1e-6)

    # Plain Python lists are accepted too.
    listed = adapter.compute_dpo_loss_from_logps([-1.0, -4.0, -2.0], [-3.0, -1.0, -2.0], [-2.0, -3.0, -5.0],
                                                 [-2.5, -2.0, -1.0], beta)
    assert float(listed) == pytest.approx(expected, rel=1e-6)


def test_dpo_loss_is_stable_for_extreme_gaps():
    big = torch.tensor([1.0e4])
    zero = torch.tensor([0.0])
    winning = adapter.compute_dpo_loss_from_logps(big, zero, zero, zero, beta=1.0)
    losing = adapter.compute_dpo_loss_from_logps(zero, big, zero, zero, beta=1.0)
    assert float(winning) == 0.0
    assert torch.isfinite(losing) and float(losing) == pytest.approx(1.0e4)


def test_dpo_gradients_push_chosen_up_and_rejected_down_and_skip_the_reference():
    policy_chosen = torch.tensor([-2.0, -3.0], requires_grad=True)
    policy_rejected = torch.tensor([-2.5, -1.0], requires_grad=True)
    reference_chosen = torch.tensor([-2.0, -3.0], requires_grad=True)
    reference_rejected = torch.tensor([-2.5, -1.0], requires_grad=True)
    loss = adapter.compute_dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, 0.1)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))  # policy == reference
    loss.backward()
    assert (policy_chosen.grad < 0).all()   # gradient descent raises log pi(chosen)
    assert (policy_rejected.grad > 0).all()  # ... and lowers log pi(rejected)
    torch.testing.assert_close(policy_chosen.grad, torch.full((2,), -0.1 * 0.5 / 2))
    assert reference_chosen.grad is None and reference_rejected.grad is None


def test_reward_accuracy_uses_a_strict_comparison():
    tied = torch.tensor([-2.0, -5.0])
    metrics = losses.dpo_metrics(tied, tied, tied, tied, beta=0.1)
    assert float(metrics["reward_accuracy"]) == 0.0  # ties are not counted as correct
    assert float(metrics["reward_margin"]) == 0.0

    policy_chosen, policy_rejected = torch.tensor([-1.0, -4.0, -2.0]), torch.tensor([-3.0, -1.0, -2.0])
    reference_chosen, reference_rejected = torch.tensor([-2.0, -3.0, -2.0]), torch.tensor([-2.5, -2.0, -2.0])
    metrics = losses.dpo_metrics(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.5)
    # rewards: chosen = [0.5, -0.5, 0.0], rejected = [-0.25, 0.5, 0.0] -> correct, wrong, tie
    assert float(metrics["reward_accuracy"]) == pytest.approx(1 / 3)
    assert float(metrics["reward_margin"]) == pytest.approx((0.75 - 1.0 + 0.0) / 3)


def test_model_level_dpo_loss_equals_the_tensor_level_loss():
    tokenizer = ToyTokenizer()
    policy = BigramLM(tokenizer.vocab_size, seed=1, dtype=torch.float64)
    reference = BigramLM(tokenizer.vocab_size, seed=2, dtype=torch.float64)
    prompts, chosen, rejected = ["Pick: ", "Choose now: "], ["apple", "ok"], ["no", "banana split"]
    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.3, max_length=32)
    expected = adapter.compute_dpo_loss_from_logps(
        adapter.compute_response_logprobs(policy, tokenizer, prompts, chosen, 32),
        adapter.compute_response_logprobs(policy, tokenizer, prompts, rejected, 32),
        adapter.compute_response_logprobs(reference, tokenizer, prompts, chosen, 32),
        adapter.compute_response_logprobs(reference, tokenizer, prompts, rejected, 32),
        0.3,
    )
    torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-10)


def test_reference_model_is_frozen_inside_the_dpo_loss():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.3)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.8)
    reference.train()  # even a reference left in train mode with requires_grad=True gets no gradient
    before = reference.logit_bias.detach().clone()
    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, ["Pick: "], ["a"], ["b"], beta=0.2, max_length=12)
    loss.backward()
    assert reference.logit_bias.grad is None
    assert reference.training  # the caller's mode is restored
    torch.testing.assert_close(reference.logit_bias.detach(), before)
    assert policy.logit_bias.grad.abs().sum() > 0


def test_identical_policy_and_reference_give_log_two():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.6)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.6)
    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, ["Pick: ", "Go: "], ["a", "left"], ["b", "right"],
                                    beta=0.1, max_length=16)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
