"""Regression tests: build_pairs.py refuses unfair or unverifiable inputs, and generation cuts answers at
special turn-marker tokens before decoding removes them (chatml)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from alignment.generation import GenerationConfig, generate_batch, generate_file, generation_kwargs, stop_token_ids
from alignment.io_utils import read_json, read_jsonl, write_json
from alignment.prompt_format import trim_at_stop_strings
from tests.helpers import TemplateToyTokenizer
from tests.test_generation_and_scripts import EchoGenerator, _records
from tests.test_generation_provenance import FailingGenerator

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = dict(max_new_tokens=8, batch_size=2)


def load_cli():
    spec = importlib.util.spec_from_file_location("pa2_build_pairs_cli", ROOT / "scripts" / "build_pairs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(path, generator, model_fingerprint, template="alpaca", precision="fp16", model=None, **config):
    tokenizer = TemplateToyTokenizer()
    model = model or EchoGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id)
    identity = {"model": model_fingerprint, "adapter": None, "precision": precision, "seed": None}
    return generate_file(model, tokenizer, _records(4), path, generator, None, template,
                         GenerationConfig(**{**BASE_CONFIG, **config}), "cpu", model_identity=identity,
                         log=lambda *_: None)


def pair_args(tmp_path, *extra):
    return ["--baseline_path", str(tmp_path / "sft.json"), "--candidate_path", str(tmp_path / "dpo.json"),
            "--output_path", str(tmp_path / "pairs" / "dpo_vs_sft_pairs.jsonl"), "--comparison", "dpo_vs_sft", *extra]


# ---------------------------------------------------------------------------
# Finding 4: pairing guard
# ---------------------------------------------------------------------------


def test_complete_inputs_with_matching_settings_are_paired_and_the_settings_recorded(tmp_path):
    generate(tmp_path / "sft.json", "toy-sft", "checkpoint-a")
    generate(tmp_path / "dpo.json", "toy-dpo", "checkpoint-b", batch_size=1)  # other model, other batch size: fine
    pairs = load_cli().main(pair_args(tmp_path))
    assert len(pairs) == 4
    meta = read_json(tmp_path / "pairs" / "dpo_vs_sft_pairs.meta.json")
    assert meta["inputs_verified"] is True and meta["unverified_reasons"] == []
    assert meta["generation_settings"]["template"] == "alpaca" and meta["generation_settings"]["precision"] == "fp16"
    assert meta["generation_settings"]["decoding"]["repetition_penalty"] == 1.0


@pytest.mark.parametrize("candidate_change, message", [
    ({"repetition_penalty": 1.1}, r"decoding\.repetition_penalty: 1\.0 -> 1\.1"),
    ({"max_new_tokens": 16}, r"decoding\.max_new_tokens: 8 -> 16"),
    ({"no_repeat_ngram_size": 4}, r"decoding\.no_repeat_ngram_size: 0 -> 4"),
    ({"precision": "fp32"}, r"precision: 'fp16' -> 'fp32'"),
    ({"template": "chatml"}, r"template: 'alpaca' -> 'chatml'"),
])
def test_inputs_generated_with_different_settings_are_refused(tmp_path, candidate_change, message):
    """The scenario from the review: an old greedy SFT file next to a DPO file generated with other flags."""

    generate(tmp_path / "sft.json", "toy-sft", "checkpoint-a")
    generate(tmp_path / "dpo.json", "toy-dpo", "checkpoint-b", **candidate_change)
    with pytest.raises(SystemExit, match=message):
        load_cli().main(pair_args(tmp_path))
    assert not (tmp_path / "pairs").exists()  # nothing is written for a judge to consume


def test_incomplete_generation_is_refused(tmp_path):
    tokenizer = TemplateToyTokenizer()
    generate(tmp_path / "sft.json", "toy-sft", "checkpoint-a")
    with pytest.raises(RuntimeError, match="session lost"):
        generate(tmp_path / "dpo.json", "toy-dpo", "checkpoint-b",
                 model=FailingGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id, fail_on_call=2))
    assert len(read_json(tmp_path / "dpo.json")) == 2  # a partial file exists
    with pytest.raises(SystemExit, match=r"dpo\.json: generation is not complete \(status='in_progress'\)"):
        load_cli().main(pair_args(tmp_path))
    # A complete meta file that no longer matches the outputs (edited or truncated file) is refused too.
    generate(tmp_path / "dpo.json", "toy-dpo", "checkpoint-b")
    write_json(tmp_path / "dpo.json", read_json(tmp_path / "dpo.json")[:3])
    with pytest.raises(SystemExit, match="holds 3 outputs but its meta file records 4"):
        load_cli().main(pair_args(tmp_path))


def test_missing_or_legacy_metadata_needs_the_explicit_override(tmp_path):
    cli = load_cli()
    generate(tmp_path / "sft.json", "toy-sft", "checkpoint-a")
    generate(tmp_path / "dpo.json", "toy-dpo", "checkpoint-b")
    meta_path = tmp_path / "dpo.meta.json"
    write_json(meta_path, {key: value for key, value in read_json(meta_path).items() if key != "provenance"})
    with pytest.raises(SystemExit, match="no provenance record"):
        cli.main(pair_args(tmp_path))
    meta_path.unlink()
    with pytest.raises(SystemExit, match="generation settings are unknown"):
        cli.main(pair_args(tmp_path))

    pairs = cli.main(pair_args(tmp_path, "--allow_unverified_inputs"))
    meta = read_json(tmp_path / "pairs" / "dpo_vs_sft_pairs.meta.json")
    assert len(pairs) == 4 == len(read_jsonl(tmp_path / "pairs" / "dpo_vs_sft_pairs.jsonl"))
    assert meta["inputs_verified"] is False and meta["generation_settings"] is None
    assert any("generation settings are unknown" in reason for reason in meta["unverified_reasons"])


# ---------------------------------------------------------------------------
# Finding 5: special turn-marker tokens (chatml)
# ---------------------------------------------------------------------------


class ChatMLToyTokenizer(TemplateToyTokenizer):
    """Toy tokenizer in which the chatml markers are single special tokens, as in the Qwen tokenizer."""

    SPECIALS = ("<|im_start|>", "<|im_end|>")
    unk_token_id = None

    def __init__(self):
        super().__init__()
        for token in self.SPECIALS:
            self.vocab[token] = len(self.vocab)
        self.inv_vocab = {token_id: token for token, token_id in self.vocab.items()}
        self.special_ids = {self.vocab[token] for token in (*self.SPECIALS, self.eos_token, self.pad_token)}

    def encode(self, text, add_special_tokens=False):
        ids, index = [], 0
        while index < len(text):
            for token in (*self.SPECIALS, self.eos_token, self.pad_token):
                if text.startswith(token, index):
                    ids.append(self.vocab[token])
                    index += len(token)
                    break
            else:
                ids.append(self.vocab[text[index]])
                index += 1
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.inv_vocab[int(token_id)] for token_id in token_ids
                       if not (skip_special_tokens and int(token_id) in self.special_ids))

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)


class ScriptedGenerator(torch.nn.Module):
    """Appends a fixed continuation to every prompt and records the generate() arguments."""

    def __init__(self, continuation_ids):
        super().__init__()
        self.continuation = torch.tensor(continuation_ids, dtype=torch.long)
        self.kwargs = None

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.kwargs = kwargs
        return torch.cat([input_ids, self.continuation.unsqueeze(0).expand(input_ids.shape[0], -1)], dim=1)


INVENTED_TURN = "Hi there.<|im_end|>\n<|im_start|>user\nAnd another question?<|im_end|>\n<|im_start|>assistant\nSure."


def test_chatml_answer_is_cut_at_the_special_marker_token_not_after_decoding():
    tokenizer = ChatMLToyTokenizer()
    im_start, im_end = tokenizer.vocab["<|im_start|>"], tokenizer.vocab["<|im_end|>"]
    assert stop_token_ids(tokenizer, "chatml") == [im_end, im_start]

    continuation = tokenizer.encode(INVENTED_TURN)
    # Why a text-level trim cannot work: decoding drops the markers and the invented turn stays in the text.
    decoded = tokenizer.decode(continuation, skip_special_tokens=True)
    assert "<|im_end|>" not in decoded and "And another question?" in trim_at_stop_strings(decoded, "chatml")

    model = ScriptedGenerator(continuation + [tokenizer.eos_token_id])
    results = generate_batch(model, tokenizer, ["Say hi", "Say hello please"], "chatml", GenerationConfig(), "cpu")
    assert [result["output"] for result in results] == ["Hi there.", "Hi there."]
    assert all(result["finished"] and result["new_tokens"] == len("Hi there.") for result in results)
    # Generation itself is also told to stop at the markers.
    assert model.kwargs["eos_token_id"] == [tokenizer.eos_token_id, im_end, im_start]


def test_a_marker_as_the_first_generated_token_gives_an_empty_answer_not_an_invented_turn():
    tokenizer = ChatMLToyTokenizer()
    model = ScriptedGenerator(tokenizer.encode("<|im_start|>user\nWhat else?") + [tokenizer.eos_token_id])
    (result,) = generate_batch(model, tokenizer, ["Say hi"], "chatml", GenerationConfig(), "cpu")
    assert result == {"output": "", "new_tokens": 0, "finished": True}


def test_alpaca_template_and_plain_tokenizers_are_unaffected():
    chatml_tokenizer, plain_tokenizer = ChatMLToyTokenizer(), TemplateToyTokenizer()
    assert stop_token_ids(chatml_tokenizer, "alpaca") == []      # alpaca headers are ordinary text
    assert stop_token_ids(plain_tokenizer, "chatml") == []       # no convert_tokens_to_ids: nothing to look up
    assert stop_token_ids(chatml_tokenizer, None) == [] and stop_token_ids(chatml_tokenizer, "chat_template") == []
    for tokenizer in (chatml_tokenizer, plain_tokenizer):
        assert generation_kwargs(tokenizer, GenerationConfig(), "alpaca")["eos_token_id"] == tokenizer.eos_token_id
    model = ScriptedGenerator(chatml_tokenizer.encode("Hello.\n### Instruction:\nNext question")
                              + [chatml_tokenizer.eos_token_id])
    (result,) = generate_batch(model, chatml_tokenizer, ["Say hello"], "alpaca", GenerationConfig(), "cpu")
    assert result["output"] == "Hello." and result["finished"]


def test_stop_tokens_with_a_real_hugging_face_tokenizer():
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    characters = [chr(code) for code in range(32, 127)] + ["\n"]
    vocab = {"<eos>": 0, "<|im_start|>": 1, "<|im_end|>": 2, **{char: index + 3 for index, char in enumerate(characters)}}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", pad_token="<eos>",
                                        additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    assert tokenizer.unk_token_id is None  # like Qwen: unknown markers must not be mistaken for a token id
    ids = tokenizer.encode(INVENTED_TURN, add_special_tokens=False)
    assert ids.count(2) == 2 and ids.count(1) == 2
    assert "<|im_end|>" not in tokenizer.decode(ids, skip_special_tokens=True)  # the premise of the bug
    assert stop_token_ids(tokenizer, "chatml") == [2, 1] and stop_token_ids(tokenizer, "alpaca") == []
    model = ScriptedGenerator(ids + [tokenizer.eos_token_id])
    (result,) = generate_batch(model, tokenizer, ["Say hi"], "chatml", GenerationConfig(), "cpu")
    assert result["output"] == "Hi there." and result["finished"]
