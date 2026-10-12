"""Offline check of the persona track: TRL DPOTrainer + PEFT LoRA on a tiny random model (CPU).

Verifies that ``scripts/train_persona_lora.py`` drives the installed TRL / PEFT versions correctly
(conversational preference rows, LoRA adapter saving, metrics export) and that ``scripts/generate.py``
can load the adapter with the tokenizer chat template. Skipped when ``trl``, ``peft`` or ``datasets``
are missing. The scripts run as subprocesses with GPUs hidden, so the check behaves the same on a
laptop and on Kaggle. The numbers it produces are meaningless and stay in a temporary directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("peft")
pytest.importorskip("trl")

from tests.test_end_to_end_tiny_model import ROOT, WORDS, build_tiny_model, load_script  # noqa: E402

CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<' + message['role'] + '>\n' + message['content'] + '<eos>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<assistant>\n' }}{% endif %}"
)


def run_script(name: str, *args: str) -> str:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py"), *args], cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
                            check=False)
    assert result.returncode == 0, f"{name} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
    return result.stdout


def test_persona_row_conversion_and_split(monkeypatch):
    # The script pins one GPU through the environment when imported; keep that out of this process.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    module = load_script("train_persona_lora")
    rows = json.loads((ROOT / "data" / "persona_lora" / "ramsay_style_preferences.json").read_text(encoding="utf-8"))
    train_rows, val_rows = module.split_rows(rows, 20, 42)
    assert len(train_rows) == 200 and len(val_rows) == 20
    assert not {row["prompt"] for row in train_rows} & {row["prompt"] for row in val_rows}
    assert module.split_rows(rows, 20, 42) == (train_rows, val_rows)
    assert module.selected_checkpoint_step("outputs/persona_lora/trainer/checkpoint-40", 75) == 40
    assert module.selected_checkpoint_step(None, 75) == 75 and module.selected_checkpoint_step("odd-name", 75) == 75
    assert module.to_conversational(rows[0]) == {
        "prompt": [{"role": "user", "content": rows[0]["prompt"]}],
        "chosen": [{"role": "assistant", "content": rows[0]["chosen"]}],
        "rejected": [{"role": "assistant", "content": rows[0]["rejected"]}],
    }


def test_persona_lora_training_and_adapter_generation(tmp_path):
    from transformers import AutoTokenizer

    model_dir = build_tiny_model(tmp_path / "tiny_instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(model_dir)

    data_file = tmp_path / "persona.json"
    data_file.write_text(json.dumps([
        {"prompt": f"How do I cook {word}?", "chosen": f"Listen, you donkey, sear the {word} properly!",
         "rejected": f"You can cook {word} in a pan."} for word in WORDS[:12]]), encoding="utf-8")
    out_dir = tmp_path / "persona_lora"

    run_script(
        "train_persona_lora", "--config", str(ROOT / "configs" / "persona_lora.yaml"), "--override",
        f"model_name={model_dir}", f"data_file={data_file.as_posix()}", f"output_dir={out_dir.as_posix()}",
        "val_size=4", "num_train_epochs=1", "per_device_train_batch_size=2", "gradient_accumulation_steps=1",
        "eval_steps=2", "logging_steps=1", "max_length=128", "lora_r=4", "lora_alpha=8", "learning_rate=1e-3")

    summary = json.loads((out_dir / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    assert summary["completed"] and summary["optimizer_steps"] == 4
    assert summary["train_rows"] == 8 and summary["val_rows"] == 4
    assert 0 < summary["trainable_parameters"] < summary["total_parameters"]  # only the LoRA matrices train
    adapter = out_dir / "adapter"
    assert (adapter / "adapter_config.json").exists() and (adapter / "adapter_model.safetensors").exists()
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["r"] == 4 and adapter_config["lora_alpha"] == 8
    assert set(adapter_config["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                     "down_proj"}
    assert json.loads((adapter / "alignment_meta.json").read_text(encoding="utf-8"))["template"] == "chat_template"

    records = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    evals = [record for record in records if record["type"] == "eval"]
    trains = [record for record in records if record["type"] == "train"]
    assert trains and evals
    assert all(record["val_loss"] is not None and record["val_reward_accuracy"] is not None
               and record["val_reward_margin"] is not None for record in evals)
    assert all(record["loss"] is not None for record in trains)
    # The curve holds real training-time evaluations only: steps 0, 2 and 4, each once. The evaluation of
    # the reloaded best checkpoint (run after training) must not appear as a second point at the last step.
    assert [record["step"] for record in evals] == [0, 2, 4]
    by_step = {record["step"]: record for record in evals}
    # The saved adapter is labelled with the step it really comes from, and "final" means the last step.
    assert summary["selection"].startswith("lowest validation DPO loss")
    assert summary["final_step"] == 4 and summary["best_step"] in (2, 4)
    assert summary["best_step"] == min((2, 4), key=lambda step: by_step[step]["val_loss"])
    assert summary["best_metrics"]["eval_loss"] == pytest.approx(by_step[summary["best_step"]]["val_loss"], rel=1e-4)
    assert summary["final_metrics"]["eval_loss"] == pytest.approx(by_step[4]["val_loss"], rel=1e-6)
    assert summary["final_metrics"]["step"] == 4
    adapter_meta = json.loads((adapter / "alignment_meta.json").read_text(encoding="utf-8"))
    assert adapter_meta["step"] == summary["best_step"] and adapter_meta["final_training_step"] == 4
    assert str(summary["best_model_checkpoint"]).endswith(f"checkpoint-{summary['best_step']}")
    # Before any update the adapter is a no-op, so policy == reference and the DPO loss is log 2.
    assert summary["initial_eval"]["eval_loss"] == pytest.approx(0.6931, abs=2e-3)
    split = json.loads((out_dir / "data_split.json").read_text(encoding="utf-8"))
    assert len(split["train_prompts"]) == 8 and len(split["val_prompts"]) == 4

    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps([
        {"id": f"c{index}", "instruction": f"How do I roast {word}?", "dataset": "persona_style_eval",
         "category": "cooking"} for index, word in enumerate(WORDS[12:15])]), encoding="utf-8")
    common = ["--model_path", model_dir, "--prompts_path", str(prompts_path), "--max_new_tokens", "8",
              "--batch_size", "2", "--device", "cpu"]
    run_script("generate", *common, "--generator", "tiny-base", "--template", "chat_template",
               "--output_path", str(tmp_path / "gen" / "persona_base.json"))
    run_script("generate", *common, "--generator", "tiny-persona", "--adapter_path", str(adapter),
               "--output_path", str(tmp_path / "gen" / "persona_lora.json"))
    base = json.loads((tmp_path / "gen" / "persona_base.json").read_text(encoding="utf-8"))
    tuned = json.loads((tmp_path / "gen" / "persona_lora.json").read_text(encoding="utf-8"))
    assert len(base) == len(tuned) == 3
    assert {entry["generator"] for entry in tuned} == {"tiny-persona"}
    meta = json.loads((tmp_path / "gen" / "persona_lora.meta.json").read_text(encoding="utf-8"))
    assert meta["template"] == "chat_template" and meta["adapter_path"] == str(adapter)

    # The saved metrics plot like the custom runs (loss, reward accuracy, reward margin panels).
    png = load_script("plot_metrics").main(["--run_dir", str(out_dir), "--out_dir", str(tmp_path / "plots")])
    assert [path.name for path in png] == ["persona_lora_curves.png"]
