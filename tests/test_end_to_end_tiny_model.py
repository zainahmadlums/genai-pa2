"""End-to-end check of the real command-line entry points on a tiny random Qwen2 model (CPU, no downloads).

Exercises the Hugging Face code paths the toy tests cannot reach: ``from_pretrained`` /
``save_pretrained`` checkpoints, the policy + frozen reference DPO setup, left-padded batched
``generate``, pair building, judging with the mock backend, plotting and the summary tables.
Skipped when ``transformers`` / ``tokenizers`` are not installed. The numbers it produces are
meaningless (random 2-layer model) and are written to a temporary directory only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu",
         "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"pa2_script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path.as_posix()


def build_tiny_model(directory: Path) -> str:
    """Character-level tokenizer (PAD = EOS, like Qwen) and a random 2-layer Qwen2 model."""

    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    characters = [chr(code) for code in range(32, 127)] + ["\n"]
    vocab = {"<unk>": 0, "<eos>": 1, **{char: index + 2 for index, char in enumerate(characters)}}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>",
                                        pad_token="<eos>")
    assert tokenizer.encode("a b\n", add_special_tokens=False) == [vocab["a"], vocab[" "], vocab["b"], vocab["\n"]]
    assert tokenizer.decode(tokenizer.encode("a b\nc", add_special_tokens=False)) == "a b\nc"

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         bos_token_id=1, eos_token_id=1, pad_token_id=1)
    Qwen2ForCausalLM(config).save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return directory.as_posix()


def test_full_pipeline_with_the_real_scripts(tmp_path):
    model_dir = build_tiny_model(tmp_path / "tiny_qwen2")
    data = tmp_path / "data"
    sft_rows = [{"id": f"s{index}", "instruction": f"say {word}", "response": f"{word} {word}"}
                for index, word in enumerate(WORDS)]
    pair_rows = [{"id": f"p{index}", "instruction": f"pick {word}", "chosen": f"yes {word}", "rejected": "no"}
                 for index, word in enumerate(WORDS[:12])]
    prompts = [{"id": f"e{index}", "instruction": f"tell me about {word}", "dataset": "tiny_eval",
                "category": "even" if index % 2 == 0 else "odd"} for index, word in enumerate(WORDS[:5])]
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
    out = tmp_path / "outputs"
    common = ["max_length=96", "max_prompt_length=64", "log_every_steps=1", "learning_rate=1e-3", "precision=fp16"]

    # ---- SFT -------------------------------------------------------------------------------
    sft_dir = out / "sft"
    sft_summary = load_script("train_sft").main([
        "--config", str(ROOT / "configs" / "sft.yaml"), "--override", f"model_name={model_dir}",
        f"train_file={write_jsonl(data / 'sft_train.jsonl', sft_rows[:16])}",
        f"val_file={write_jsonl(data / 'sft_val.jsonl', sft_rows[16:])}", f"output_dir={sft_dir.as_posix()}",
        "train_size=16", "val_size=4", "batch_size=2", "grad_accum_steps=2", "eval_every_steps=2", "device=cpu",
        *common])
    assert sft_summary["completed"] and sft_summary["optimizer_steps"] == 4
    for name in ("best", "final"):
        assert (sft_dir / name / "model.safetensors").exists() and (sft_dir / name / "tokenizer.json").exists()
    sft_meta = json.loads((sft_dir / "best" / "alignment_meta.json").read_text(encoding="utf-8"))
    assert sft_meta["template"] == "alpaca" and sft_meta["stage"] == "sft" and sft_meta["base_model"] == model_dir
    # A finished run is detected and not repeated.
    again = load_script("train_sft").main([
        "--config", str(ROOT / "configs" / "sft.yaml"), "--override", f"output_dir={sft_dir.as_posix()}"])
    assert again["best_step"] == sft_summary["best_step"]

    # ---- DPO (policy and frozen reference both from the SFT best checkpoint) ------------------
    dpo_dir = out / "dpo_hh"
    dpo_summary = load_script("train_dpo").main([
        "--config", str(ROOT / "configs" / "dpo_hh.yaml"), "--override",
        f"sft_checkpoint={(sft_dir / 'best').as_posix()}",
        f"train_file={write_jsonl(data / 'hh_train.jsonl', pair_rows[:8])}",
        f"val_file={write_jsonl(data / 'hh_val.jsonl', pair_rows[8:])}", f"output_dir={dpo_dir.as_posix()}",
        "train_size=8", "val_size=4", "grad_accum_steps=4", "eval_every_steps=1", "policy_device=cpu",
        "reference_device=cpu", *common])
    assert dpo_summary["completed"] and dpo_summary["optimizer_steps"] == 2
    records = [json.loads(line) for line in (dpo_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    initial = next(record for record in records if record["type"] == "eval")
    # Identical policy and reference: every reward pair ties, so strict accuracy is exactly 0.
    assert initial["tag"] == "initial" and initial["val_reward_accuracy"] == 0.0
    assert initial["val_reward_margin"] == 0.0 and initial["val_tie_fraction"] == 1.0
    assert initial["val_loss"] == pytest.approx(0.6931472, rel=1e-5)
    later = [record for record in records if record["type"] == "eval" and record["step"] > 0]
    assert any(record["val_tie_fraction"] < 1.0 for record in later)  # the policy moved away from the reference
    dpo_meta = json.loads((dpo_dir / "best" / "alignment_meta.json").read_text(encoding="utf-8"))
    assert dpo_meta["stage"] == "dpo" and dpo_meta["template"] == "alpaca" and dpo_meta["beta"] == 0.1
    assert dpo_meta["reference_checkpoint"] == (sft_dir / "best").as_posix()
    # The SFT checkpoint used as the reference is untouched by DPO training.
    from safetensors.torch import load_file

    reference_weights = load_file(str(sft_dir / "best" / "model.safetensors"))
    policy_weights = load_file(str(dpo_dir / "final" / "model.safetensors"))
    assert any(not torch.equal(reference_weights[name], policy_weights[name]) for name in reference_weights)
    # A template that differs from the SFT checkpoint's is refused.
    with pytest.raises(SystemExit, match="differs from the SFT checkpoint template"):
        load_script("train_dpo").main([
            "--config", str(ROOT / "configs" / "dpo_hh.yaml"), "--override",
            f"sft_checkpoint={(sft_dir / 'best').as_posix()}", "template=chatml",
            f"output_dir={(out / 'dpo_bad').as_posix()}"])

    # ---- generation (greedy, same flags for both models) ---------------------------------------
    generate = load_script("generate")
    generations = out / "eval" / "generations"
    flags = ["--prompts_path", str(prompts_path), "--max_new_tokens", "12", "--batch_size", "2", "--device", "cpu"]
    smoke = generate.main(["--model_path", str(sft_dir / "best"), "--generator", "tiny-sft",
                           "--output_path", str(generations / "tiny_sft.json"), "--smoke", "2", *flags])
    assert len(smoke) == 2 and (generations / "tiny_sft.smoke.json").exists()
    assert not (generations / "tiny_sft.json").exists()  # the sanity check never writes the real file
    sft_out = generate.main(["--model_path", str(sft_dir / "best"), "--generator", "tiny-sft",
                             "--output_path", str(generations / "tiny_sft.json"), *flags])
    dpo_out = generate.main(["--model_path", str(dpo_dir / "best"), "--generator", "tiny-dpo",
                             "--output_path", str(generations / "tiny_dpo.json"), *flags])
    assert [entry["id"] for entry in sft_out] == [prompt["id"] for prompt in prompts] == [e["id"] for e in dpo_out]
    for entry in sft_out:
        assert {"id", "instruction", "output", "generator", "dataset"} <= set(entry)
        assert entry["generator"] == "tiny-sft" and entry["dataset"] == "tiny_eval" and isinstance(entry["output"], str)
    meta = json.loads((generations / "tiny_sft.meta.json").read_text(encoding="utf-8"))
    assert meta["template"] == "alpaca" and meta["decoding"]["do_sample"] is False
    # Batched (left-padded) greedy decoding gives the same text as decoding one prompt at a time.
    single = generate.main(["--model_path", str(sft_dir / "best"), "--generator", "tiny-sft",
                            "--output_path", str(generations / "single" / "tiny_sft.json"),
                            "--prompts_path", str(prompts_path), "--max_new_tokens", "12", "--batch_size", "1",
                            "--device", "cpu"])
    assert [entry["output"] for entry in single] == [entry["output"] for entry in sft_out]
    # Saved outputs are tied to the checkpoint contents and decoding settings that produced them.
    before = (generations / "tiny_sft.json").read_text(encoding="utf-8")
    assert meta["provenance"]["model_identity"]["model"] == generate.checkpoint_fingerprint(sft_dir / "best")
    with pytest.raises(SystemExit, match=r"decoding\.max_new_tokens: 12 -> 4"):
        generate.main(["--model_path", str(sft_dir / "best"), "--generator", "tiny-sft", "--output_path",
                       str(generations / "tiny_sft.json"), "--prompts_path", str(prompts_path),
                       "--max_new_tokens", "4", "--batch_size", "2", "--device", "cpu"])
    with pytest.raises(SystemExit, match=r"model_identity\.model"):  # another checkpoint under the same label
        generate.main(["--model_path", str(dpo_dir / "best"), "--generator", "tiny-sft",
                       "--output_path", str(generations / "tiny_sft.json"), *flags])
    assert (generations / "tiny_sft.json").read_text(encoding="utf-8") == before
    reused = generate.main(["--model_path", str(sft_dir / "best"), "--generator", "tiny-sft",
                            "--output_path", str(generations / "tiny_sft.json"), *flags])
    assert reused == sft_out
    assert json.loads((generations / "tiny_sft.meta.json").read_text(encoding="utf-8"))["reused_outputs"] == 5

    # ---- pairs, judging (mock backend: pipeline check only), plots, summary --------------------
    pairs_path = out / "eval" / "pairs" / "tiny_dpo_vs_sft_pairs.jsonl"
    pairs = load_script("build_pairs").main([
        "--baseline_path", str(generations / "tiny_sft.json"), "--candidate_path", str(generations / "tiny_dpo.json"),
        "--output_path", str(pairs_path), "--comparison", "tiny_dpo_vs_sft", "--profile", "general"])
    assert len(pairs) == 5 and {pair["hidden"]["candidate"] for pair in pairs} == {"tiny-dpo"}
    judged_path = out / "eval" / "judged" / "tiny_dpo_vs_sft_judged.jsonl"
    judge_args = ["--pairs_path", str(pairs_path), "--output_path", str(judged_path), "--backend", "mock",
                  "--judge_model", "mock", "--sleep_seconds", "0", "--reuse_judgments"]
    summary = load_script("judge_pairwise_kaggle").main(judge_args)
    assert summary["complete"] and summary["total_pairs"] == 5 and summary["failures"] == 0
    assert summary["candidate_wins"] + summary["baseline_wins"] + summary["ties"] == 5

    plots = load_script("plot_metrics").main(["--run_dir", str(sft_dir), str(dpo_dir), "--out_dir", str(out / "plots")])
    assert [path.name for path in plots] == ["sft_curves.png", "dpo_hh_curves.png"]
    report = load_script("summarize_results").main([
        "--runs", str(sft_dir), str(dpo_dir), "--eval_dir", str(out / "eval"), "--out_dir", str(out / "report")])
    assert [row["run"] for row in report["runs"]] == ["sft", "dpo_hh"] and all(row["completed"] for row in report["runs"])
    assert report["runs"][0]["effective_batch"] == 4 and report["runs"][1]["beta"] == 0.1
    assert len(report["judgments"]) == 1 and len(report["generations"]) == 2
    tables = (out / "report" / "report_tables.md").read_text(encoding="utf-8")
    assert "MOCK: not a result" in tables and "full fine-tuning" in tables
