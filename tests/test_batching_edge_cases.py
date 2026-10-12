"""Batch construction: truncation, EOS, padding, masks, row order, unusual tokenizers."""

from __future__ import annotations

import pytest
import torch

import submission_adapter as adapter
from alignment.batching import IGNORE_INDEX, build_dpo_batch, build_lm_batch, concat_lm_batches
from tests.helpers import (
    CallOnlyTokenizer,
    EncodeOnlyTokenizer,
    NoEosTokenizer,
    NoPadTokenizer,
    PadEqualsEosTokenizer,
)
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_eos_is_kept_when_the_sequence_fits_exactly():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["ab: "], ["xyz"], max_length=8)  # 4 + 3 + EOS = 8
    assert batch["input_ids"].shape == (1, 8)
    assert batch["input_ids"][0, -1] == tokenizer.eos_token_id
    assert batch["labels"][0, -1] == tokenizer.eos_token_id
    assert batch["attention_mask"][0].tolist() == [1] * 8


def test_eos_is_dropped_when_truncation_cuts_the_response():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["ab: "], ["xyzw"], max_length=8)  # 4 + 4 + EOS = 9 > 8
    expected = tokenizer.encode("ab: xyzw")
    assert batch["input_ids"][0].tolist() == expected
    assert tokenizer.eos_token_id not in batch["input_ids"][0].tolist()
    assert batch["labels"][0].tolist() == [IGNORE_INDEX] * 4 + expected[4:]
    assert batch["attention_mask"][0].tolist() == [1] * 8


def test_prompt_longer_than_max_length_leaves_no_response_tokens():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tokenizer, ["abcdefghij"], ["xyz"], max_length=6)
    assert batch["input_ids"][0].tolist() == tokenizer.encode("abcdef")
    assert batch["labels"].eq(IGNORE_INDEX).all()
    logps = adapter.compute_response_logprobs(model, tokenizer, ["abcdefghij"], ["xyz"], max_length=6)
    torch.testing.assert_close(logps, torch.zeros(1))
    loss = adapter.compute_sft_loss(model, tokenizer, ["abcdefghij"], ["xyz"], max_length=6)
    assert torch.isfinite(loss) and float(loss.detach()) == 0.0
    loss.backward()  # still differentiable


def test_adapter_pads_to_max_length_and_training_path_pads_to_longest():
    tokenizer = ToyTokenizer()
    prompts, responses = ["Task: ", "Q: "], ["ok", "yes"]
    padded = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=16)
    dynamic = build_lm_batch(tokenizer, prompts, responses, max_length=16)
    assert padded["input_ids"].shape == (2, 16)
    assert dynamic["input_ids"].shape == (2, 9)  # longest row: 6 + 2 + EOS
    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(padded[key][:, :9], dynamic[key])
    assert padded["attention_mask"][:, 9:].eq(0).all()
    assert padded["labels"][:, 9:].eq(IGNORE_INDEX).all()


def test_pad_equal_to_eos_does_not_corrupt_masks_or_labels():
    tokenizer = PadEqualsEosTokenizer()
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    prompts, responses = ["Task: ", "Q: "], ["ok", "yes"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=16)
    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        length = len(tokenizer.encode(prompt)) + len(tokenizer.encode(response)) + 1
        assert batch["attention_mask"][row].tolist() == [1] * length + [0] * (16 - length)
        # Exactly one supervised EOS (the real one); padding EOS ids are masked out.
        assert int(batch["labels"][row].eq(tokenizer.eos_token_id).sum()) == 1
        assert batch["labels"][row, length - 1] == tokenizer.eos_token_id
        assert batch["labels"][row, length:].eq(IGNORE_INDEX).all()
        assert batch["input_ids"][row, length:].eq(tokenizer.pad_token_id).all()

    # Scores are identical to a tokenizer with a distinct PAD id.
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.9)
    reference = adapter.compute_response_logprobs(model, ToyTokenizer(), prompts, responses, max_length=16)
    scored = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=16)
    torch.testing.assert_close(scored, reference)


def test_tokenizer_without_pad_token_falls_back_to_eos():
    tokenizer = NoPadTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["a: ", "b: "], ["x", "yyyy"], max_length=10)
    assert batch["input_ids"][0, 5:].eq(tokenizer.eos_token_id).all()
    assert batch["attention_mask"][0].tolist() == [1] * 5 + [0] * 5
    assert batch["labels"][0, 5:].eq(IGNORE_INDEX).all()


def test_tokenizer_without_eos_appends_nothing():
    tokenizer = NoEosTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["a: "], ["xy"], max_length=10)
    assert batch["attention_mask"][0].sum() == 5
    assert batch["labels"][0, :5].tolist() == [IGNORE_INDEX] * 3 + tokenizer.encode("xy")


@pytest.mark.parametrize("tokenizer_class", [EncodeOnlyTokenizer, CallOnlyTokenizer])
def test_minimal_tokenizers_without_hf_apis(tokenizer_class):
    tokenizer = tokenizer_class()
    reference = ToyTokenizer()
    prompts, responses = ["Do: ", "Say: "], ["a", "longer"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=16)
    expected = adapter.build_sft_batch(reference, prompts, responses, max_length=16)
    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(batch[key], expected[key])
    model = ToyCausalLM(reference.vocab_size, bias_scale=0.4)
    loss = adapter.compute_dpo_loss(model, ToyCausalLM(reference.vocab_size), tokenizer, prompts, responses,
                                    ["b", "short"], beta=0.1, max_length=16)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_row_order_is_preserved_under_permutation():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.0)
    prompts = ["One: ", "Second prompt: ", "3: ", "The fourth: "]
    responses = ["a", "bcdef", "gh", "ijklmnop"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=24)
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=24)
    assert len({round(value, 5) for value in logps.tolist()}) == 4  # rows are distinguishable

    order = [2, 0, 3, 1]
    permuted_batch = adapter.build_sft_batch(
        tokenizer, [prompts[i] for i in order], [responses[i] for i in order], max_length=24)
    permuted_logps = adapter.compute_response_logprobs(
        model, tokenizer, [prompts[i] for i in order], [responses[i] for i in order], max_length=24)
    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(permuted_batch[key], batch[key][order])
    torch.testing.assert_close(permuted_logps, logps[order])

    # Each row scored alone equals its value inside the mixed-length batch.
    for row in range(4):
        single = adapter.compute_response_logprobs(model, tokenizer, [prompts[row]], [responses[row]], max_length=24)
        torch.testing.assert_close(single[0], logps[row])


def test_single_string_inputs_are_treated_as_one_row():
    tokenizer = ToyTokenizer()
    batch = build_lm_batch(tokenizer, "Q: ", "yes", max_length=12)
    assert batch["input_ids"].shape[0] == 1


def test_mismatched_lengths_and_bad_max_length_raise():
    tokenizer = ToyTokenizer()
    with pytest.raises(ValueError):
        build_lm_batch(tokenizer, ["a", "b"], ["c"], max_length=8)
    with pytest.raises(ValueError):
        build_lm_batch(tokenizer, ["a"], ["c"], max_length=0)


def test_max_prompt_length_keeps_the_end_of_the_prompt():
    tokenizer = ToyTokenizer()
    batch = build_lm_batch(tokenizer, ["abcdefgh: "], ["ok"], max_length=32, max_prompt_length=4)
    assert batch["input_ids"][0].tolist() == tokenizer.encode("gh: ok") + [tokenizer.eos_token_id]
    assert batch["labels"][0].tolist() == [IGNORE_INDEX] * 4 + tokenizer.encode("ok") + [tokenizer.eos_token_id]


def test_dpo_batch_shares_prompts_and_can_be_concatenated():
    tokenizer = ToyTokenizer()
    prompts, chosen, rejected = ["Task: ", "Q: "], ["good answer", "yes"], ["bad", "no"]
    batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length=32, same_width=True)
    assert batch["chosen"]["input_ids"].shape == batch["rejected"]["input_ids"].shape
    for row, prompt in enumerate(prompts):
        length = len(tokenizer.encode(prompt))
        assert torch.equal(batch["chosen"]["input_ids"][row, :length], batch["rejected"]["input_ids"][row, :length])
        assert batch["chosen"]["labels"][row, :length].eq(IGNORE_INDEX).all()
        assert batch["rejected"]["labels"][row, :length].eq(IGNORE_INDEX).all()
    joined = concat_lm_batches(batch["chosen"], batch["rejected"])
    assert joined["input_ids"].shape[0] == 4
    assert torch.equal(joined["labels"][:2], batch["chosen"]["labels"])
    assert torch.equal(joined["labels"][2:], batch["rejected"]["labels"])

    separate = build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length=32)
    with pytest.raises(ValueError):
        concat_lm_batches(separate["chosen"], separate["rejected"])


def test_adapter_dpo_batch_truncates_both_sides_to_max_length():
    tokenizer = ToyTokenizer()
    batch = adapter.build_dpo_batch(tokenizer, ["Prompt: "], ["a very long chosen answer"], ["no"], max_length=12)
    assert batch["chosen"]["input_ids"].shape == (1, 12)
    assert batch["rejected"]["input_ids"].shape == (1, 12)
    assert tokenizer.eos_token_id not in batch["chosen"]["input_ids"][0].tolist()  # cut before EOS
    assert batch["rejected"]["labels"][0, 10] == tokenizer.eos_token_id  # 8 + 2 + EOS fits
