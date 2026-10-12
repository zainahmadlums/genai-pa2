"""Regression tests: saved generations are reused only for the same checkpoint, template and decoding
settings, the provenance is on disk before the first batch, and replacing outputs needs an explicit overwrite."""

from __future__ import annotations

import shutil

import pytest

from alignment.generation import (
    GenerationCacheMismatch,
    GenerationConfig,
    checkpoint_fingerprint,
    generate_file,
    generation_provenance,
)
from alignment.io_utils import read_json, write_json
from tests.helpers import TemplateToyTokenizer
from tests.test_generation_and_scripts import EchoGenerator, _records

IDENTITY = {"model": "fingerprint-1", "adapter": None, "precision": "fp16", "seed": None}


def run(tmp_path, records, config=None, template="alpaca", identity=IDENTITY, model=None, **kwargs):
    tokenizer = TemplateToyTokenizer()
    model = model or EchoGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id)
    entries = generate_file(model, tokenizer, records, tmp_path / "gen.json", "toy-model", None, template,
                            config or GenerationConfig(max_new_tokens=8, batch_size=2), "cpu",
                            model_identity=identity, log=lambda *_: None, **kwargs)
    return entries, model


def generated_rows(model) -> int:
    return sum(call["shape"][0] for call in model.calls)


def test_changed_max_new_tokens_is_refused_instead_of_relabelling_old_answers(tmp_path):
    records = _records(5)
    first, _ = run(tmp_path, records)
    meta_before = read_json(tmp_path / "gen.meta.json")
    assert meta_before["status"] == "complete" and meta_before["decoding"]["max_new_tokens"] == 8
    assert meta_before["provenance"]["model_identity"] == IDENTITY

    tokenizer = TemplateToyTokenizer()
    model = EchoGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id)
    with pytest.raises(GenerationCacheMismatch, match=r"decoding\.max_new_tokens: 8 -> 256"):
        run(tmp_path, records, GenerationConfig(max_new_tokens=256, batch_size=2), model=model)
    # No generation, no rewritten outputs, and the metadata still says what really produced the file.
    assert model.calls == [] and read_json(tmp_path / "gen.json") == first
    assert read_json(tmp_path / "gen.meta.json") == meta_before

    entries, model = run(tmp_path, records, GenerationConfig(max_new_tokens=256, batch_size=2), overwrite=True)
    meta = read_json(tmp_path / "gen.meta.json")
    assert generated_rows(model) == 5 and all(call["kwargs"]["max_new_tokens"] == 256 for call in model.calls)
    assert meta["decoding"]["max_new_tokens"] == 256 and meta["provenance"]["decoding"]["max_new_tokens"] == 256
    assert meta["reused_outputs"] == 0 and meta["generated_this_run"] == 5 and len(entries) == 5


@pytest.mark.parametrize("change, expected", [
    ({"identity": {**IDENTITY, "model": "fingerprint-2"}}, "model_identity.model"),
    ({"identity": {**IDENTITY, "adapter": "adapter-fingerprint"}}, "model_identity.adapter"),
    ({"identity": {**IDENTITY, "precision": "fp32"}}, "model_identity.precision"),
    ({"template": "chatml"}, "template: 'alpaca' -> 'chatml'"),
    ({"config": GenerationConfig(max_new_tokens=8, batch_size=2, repetition_penalty=1.1)}, "repetition_penalty"),
    ({"config": GenerationConfig(max_new_tokens=8, batch_size=2, no_repeat_ngram_size=4)}, "no_repeat_ngram_size"),
    ({"config": GenerationConfig(max_new_tokens=8, batch_size=2, do_sample=True, temperature=0.7)}, "do_sample"),
])
def test_checkpoint_template_and_decoding_changes_block_reuse(tmp_path, change, expected):
    records = _records(4)
    first, _ = run(tmp_path, records)
    tokenizer = TemplateToyTokenizer()
    model = EchoGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id)
    with pytest.raises(GenerationCacheMismatch, match=expected):
        run(tmp_path, records, model=model, **change)
    assert model.calls == [] and read_json(tmp_path / "gen.json") == first


def test_same_provenance_resumes_and_batch_size_may_change(tmp_path):
    run(tmp_path, _records(3))
    entries, model = run(tmp_path, _records(5), GenerationConfig(max_new_tokens=8, batch_size=1))
    meta = read_json(tmp_path / "gen.meta.json")
    assert len(entries) == 5 and generated_rows(model) == 2  # only the two new prompts
    assert meta["reused_outputs"] == 3 and meta["generated_this_run"] == 2 and meta["status"] == "complete"
    assert generation_provenance("g", None, "alpaca", GenerationConfig(batch_size=1)) == generation_provenance(
        "g", None, "alpaca", GenerationConfig(batch_size=64))


def test_outputs_without_a_provenance_record_are_refused(tmp_path):
    records = _records(3)
    first, _ = run(tmp_path, records)
    meta_path = tmp_path / "gen.meta.json"
    legacy = {key: value for key, value in read_json(meta_path).items() if key != "provenance"}
    write_json(meta_path, legacy)  # a meta file written before provenance tracking
    tokenizer = TemplateToyTokenizer()
    model = EchoGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id)
    with pytest.raises(GenerationCacheMismatch, match="no provenance record"):
        run(tmp_path, records, model=model)
    meta_path.unlink()  # and an output file whose meta file is missing altogether
    with pytest.raises(GenerationCacheMismatch, match="no provenance record"):
        run(tmp_path, records, model=model)
    assert model.calls == [] and read_json(tmp_path / "gen.json") == first
    run(tmp_path, records, overwrite=True)
    assert read_json(meta_path)["provenance"]["model_identity"] == IDENTITY


class FailingGenerator(EchoGenerator):
    """Raises on the given (1-based) generate call, like a notebook disconnect in the middle of a file."""

    def __init__(self, eos_token_id, pad_token_id, fail_on_call):
        super().__init__(eos_token_id, pad_token_id)
        self.fail_on_call = fail_on_call

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if len(self.calls) + 1 == self.fail_on_call:
            raise RuntimeError("session lost")
        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def test_provenance_is_on_disk_before_the_first_batch_and_survives_a_crash(tmp_path):
    tokenizer = TemplateToyTokenizer()
    records = _records(5)
    with pytest.raises(RuntimeError, match="session lost"):
        run(tmp_path, records, model=FailingGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id, fail_on_call=1))
    meta = read_json(tmp_path / "gen.meta.json")
    assert meta["status"] == "in_progress" and meta["provenance"]["decoding"]["max_new_tokens"] == 8
    assert not (tmp_path / "gen.json").exists()

    with pytest.raises(RuntimeError, match="session lost"):
        run(tmp_path, records, model=FailingGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id, fail_on_call=2))
    partial = read_json(tmp_path / "gen.json")
    meta = read_json(tmp_path / "gen.meta.json")
    assert len(partial) == 2 and meta["status"] == "in_progress" and meta["provenance"]["model_identity"] == IDENTITY

    # After the crash a run with other settings is still refused; the same settings continue.
    with pytest.raises(GenerationCacheMismatch):
        run(tmp_path, records, GenerationConfig(max_new_tokens=64, batch_size=2))
    entries, model = run(tmp_path, records)
    meta = read_json(tmp_path / "gen.meta.json")
    assert len(entries) == 5 and generated_rows(model) == 3
    assert meta["status"] == "complete" and meta["reused_outputs"] == 2 and meta["generated_this_run"] == 3


def test_overwrite_discards_old_outputs_before_recording_the_new_provenance(tmp_path):
    tokenizer = TemplateToyTokenizer()
    records = _records(4)
    run(tmp_path, records)
    with pytest.raises(RuntimeError, match="session lost"):
        run(tmp_path, records, GenerationConfig(max_new_tokens=64, batch_size=2), overwrite=True,
            model=FailingGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id, fail_on_call=1))
    # Crash right after an explicit overwrite: the old answers are gone, they are not paired with the new settings.
    assert not (tmp_path / "gen.json").exists()
    assert read_json(tmp_path / "gen.meta.json")["decoding"]["max_new_tokens"] == 64


def test_checkpoint_fingerprint_follows_content_not_location(tmp_path):
    checkpoint = tmp_path / "run" / "best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"hidden_size": 32}', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(bytes(range(256)) * 16384)  # 4 MB: head and tail are sampled
    fingerprint = checkpoint_fingerprint(checkpoint)
    assert fingerprint and fingerprint == checkpoint_fingerprint(checkpoint)

    copy = tmp_path / "attached_input" / "best"
    shutil.copytree(checkpoint, copy)
    assert checkpoint_fingerprint(copy) == fingerprint  # same checkpoint attached under another path

    weights = bytearray((checkpoint / "model.safetensors").read_bytes())
    weights[100] ^= 0xFF  # retrained into the same path: same size, different content
    (checkpoint / "model.safetensors").write_bytes(bytes(weights))
    retrained = checkpoint_fingerprint(checkpoint)
    assert retrained != fingerprint
    weights[-5] ^= 0xFF
    (checkpoint / "model.safetensors").write_bytes(bytes(weights))
    assert checkpoint_fingerprint(checkpoint) != retrained
    (checkpoint / "config.json").write_text('{"hidden_size": 64}', encoding="utf-8")
    assert checkpoint_fingerprint(checkpoint) not in (fingerprint, retrained)

    assert checkpoint_fingerprint("Qwen/Qwen2.5-0.5B") is None  # hub ids are identified by name
    assert checkpoint_fingerprint(tmp_path / "does_not_exist") is None
