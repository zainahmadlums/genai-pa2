"""Extra test doubles for the supplementary tests (the public toy_lm.py is left untouched)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from tests.toy_lm import ToyTokenizer


class PadEqualsEosTokenizer(ToyTokenizer):
    """Like Qwen/Qwen2.5-0.5B: PAD and EOS share one id."""

    def __init__(self):
        super().__init__()
        self.pad_token = self.eos_token
        self.pad_token_id = self.eos_token_id


class TemplateToyTokenizer(ToyTokenizer):
    """Toy tokenizer whose vocabulary also covers the characters used by the prompt templates."""

    def __init__(self):
        super().__init__()
        for char in "#\n<>|'\"()+=/":
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inv_vocab = {token_id: token for token, token_id in self.vocab.items()}

    def decode(self, token_ids, skip_special_tokens=True):
        special = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
        return "".join(self.inv_vocab[int(token_id)] for token_id in token_ids if int(token_id) not in special)


class _AttributeOverrideTokenizer(ToyTokenizer):
    """Toy tokenizer with the same vocabulary but some special-token attributes removed."""

    def __init__(self):
        super().__init__()
        self._plain = ToyTokenizer()

    def encode(self, text, add_special_tokens=False):
        return self._plain.encode(text, add_special_tokens=False)


class NoPadTokenizer(_AttributeOverrideTokenizer):
    def __init__(self):
        super().__init__()
        self.pad_token = None
        self.pad_token_id = None


class NoEosTokenizer(_AttributeOverrideTokenizer):
    def __init__(self):
        super().__init__()
        self.eos_token = None
        self.eos_token_id = None


class EncodeOnlyTokenizer:
    """Minimal tokenizer: positional ``encode(text)`` only, no kwargs, no ``__call__``."""

    eos_token_id = 1
    pad_token_id = 0

    def __init__(self):
        self._inner = ToyTokenizer()
        self.vocab_size = self._inner.vocab_size

    def encode(self, text):
        return self._inner.encode(text)


class CallOnlyTokenizer:
    """Minimal tokenizer: ``tokenizer(text)`` returning a dict, no ``encode`` method."""

    eos_token_id = 1
    pad_token_id = 0

    def __init__(self):
        self._inner = ToyTokenizer()
        self.vocab_size = self._inner.vocab_size

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._inner.encode(text)}


class BigramLM(nn.Module):
    """Causal LM whose logits at position t depend only on the token at position t.

    ``logits[b, t] = table[input_ids[b, t]]``, so the log-probability of token t+1 given
    token t can be computed by hand. Any off-by-one in label shifting changes the result.
    """

    def __init__(self, vocab_size: int, seed: int = 0, dtype=torch.float32):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, generator=generator, dtype=dtype))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(logits=self.table[input_ids])


def manual_bigram_logprob(model: BigramLM, prompt_ids: list[int], response_ids: list[int]) -> torch.Tensor:
    """Sum of log p(response token | previous token) computed without any batching code."""

    sequence = prompt_ids + response_ids
    log_probs = torch.log_softmax(model.table.detach(), dim=-1)
    total = torch.zeros((), dtype=log_probs.dtype)
    for position in range(len(prompt_ids), len(sequence)):
        if position == 0:
            continue  # no context for the very first token
        total = total + log_probs[sequence[position - 1], sequence[position]]
    return total
