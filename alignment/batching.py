"""Batch construction for response-only language-model training.

Every sequence is ``prompt_ids + response_ids + [eos]`` (EOS appended when the tokenizer
has one), truncated on the right to ``max_length``, so EOS can be lost when the response
is cut. Labels equal the input ids on response positions and ``-100`` on prompt and
padding positions.

Padding and masks are derived from sequence lengths, never from ``input_ids == pad_id``,
because Qwen/Qwen2.5-0.5B uses the same id for PAD and EOS. Tokenisation only needs
``tokenizer.encode`` (or ``tokenizer(text)``) plus optional ``eos_token_id`` /
``pad_token_id`` attributes, so minimal toy tokenizers work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

IGNORE_INDEX = -100


def encode_text(tokenizer, text: str) -> list[int]:
    """Token ids for ``text`` without automatically added special tokens."""

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = encode(text, add_special_tokens=False)
        except TypeError:
            ids = encode(text)
    else:
        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer(text)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(token_id) for token_id in ids]


def get_eos_token_id(tokenizer):
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    return None if eos_token_id is None else int(eos_token_id)


def get_pad_token_id(tokenizer) -> int:
    """PAD id, falling back to EOS and then 0. The value never affects masks or labels."""

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad_token_id is None else int(pad_token_id)


def _as_list(values) -> list[str]:
    if isinstance(values, str):
        return [values]
    return list(values)


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int | None,
    max_prompt_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return ``input_ids``, ``attention_mask`` and response-only ``labels``.

    Rows keep the order of ``prompts`` / ``responses``.

    Args:
        max_length: right-truncation length applied after EOS is appended.
        max_prompt_length: optional cap on prompt tokens. Over-long prompts keep their
            *last* tokens so the response header of the template survives.
        pad_to_max_length: pad every row to ``max_length`` (adapter contract) instead of
            the longest row in the batch (training, to save compute).
        pad_to_width: pad to at least this width (used to align chosen/rejected batches).
    """

    prompts, responses = _as_list(prompts), _as_list(responses)
    if len(prompts) != len(responses):
        raise ValueError(f"Got {len(prompts)} prompts but {len(responses)} responses")
    if max_length is not None and int(max_length) < 1:
        raise ValueError("max_length must be at least 1")
    max_length = None if max_length is None else int(max_length)

    eos_token_id = get_eos_token_id(tokenizer)
    pad_token_id = get_pad_token_id(tokenizer)

    rows_ids: list[list[int]] = []
    rows_labels: list[list[int]] = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = encode_text(tokenizer, prompt)
        if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length:]
        response_ids = encode_text(tokenizer, response)
        if eos_token_id is not None:
            response_ids = response_ids + [eos_token_id]
        ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        if max_length is not None:
            ids, labels = ids[:max_length], labels[:max_length]
        rows_ids.append(ids)
        rows_labels.append(labels)

    width = max((len(ids) for ids in rows_ids), default=0)
    if pad_to_max_length and max_length is not None:
        width = max_length
    if pad_to_width is not None:
        width = max(width, int(pad_to_width))
    width = max(width, 1)

    batch_size = len(rows_ids)
    input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    labels = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, row_labels) in enumerate(zip(rows_ids, rows_labels)):
        length = len(ids)
        if length:
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
            labels[row, :length] = torch.tensor(row_labels, dtype=torch.long)
            attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dpo_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int | None,
    max_prompt_length: int | None = None,
    pad_to_max_length: bool = False,
    same_width: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return ``{"chosen": lm_batch, "rejected": lm_batch}`` sharing identical prompts.

    ``same_width=True`` pads both sides to a common width so they can be concatenated
    into one forward pass.
    """

    prompts = _as_list(prompts)
    chosen_responses, rejected_responses = _as_list(chosen_responses), _as_list(rejected_responses)
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must have equal length")

    common = dict(max_length=max_length, max_prompt_length=max_prompt_length, pad_to_max_length=pad_to_max_length)
    chosen = build_lm_batch(tokenizer, prompts, chosen_responses, **common)
    rejected = build_lm_batch(tokenizer, prompts, rejected_responses, **common)
    if same_width and chosen["input_ids"].shape[1] != rejected["input_ids"].shape[1]:
        width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
        chosen = build_lm_batch(tokenizer, prompts, chosen_responses, pad_to_width=width, **common)
        rejected = build_lm_batch(tokenizer, prompts, rejected_responses, pad_to_width=width, **common)
    return {"chosen": chosen, "rejected": rejected}


def concat_lm_batches(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Stack two equally wide LM batches along the batch dimension (first rows, then second)."""

    if first["input_ids"].shape[1] != second["input_ids"].shape[1]:
        raise ValueError("LM batches must have the same width; build them with same_width=True")
    return {key: torch.cat([first[key], second[key]], dim=0) for key in ("input_ids", "attention_mask", "labels")}
