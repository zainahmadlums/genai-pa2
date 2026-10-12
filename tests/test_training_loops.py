"""Custom training loop: accumulation, schedules, checkpoints, resume, and DPO reference freezing."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from alignment.dpo_trainer import DPOTask, DPOTaskConfig
from alignment.engine import LoopConfig, already_complete, run_training
from alignment.losses import dpo_loss_from_logps, sequence_logprobs
from alignment.sft_trainer import SFTTask, SFTTaskConfig
from alignment.train_utils import (
    AmpSettings,
    accumulation_groups,
    build_optimizer,
    epoch_order,
    load_checkpoint_meta,
    lr_factor,
    read_json,
    read_jsonl,
)
from tests.helpers import BigramLM, TemplateToyTokenizer
from tests.toy_lm import ToyCausalLM

SFT_ROWS = [{"instruction": f"say {word}", "response": word} for word in
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]]
DPO_ROWS = [{"instruction": f"pick {index}", "chosen": "aaaa" + "a" * (index % 3), "rejected": "bbbb" + "b" * (index % 2)}
            for index in range(10)]


def loop_config(tmp_path, **overrides) -> LoopConfig:
    values = dict(output_dir=str(tmp_path / "run"), run_name="test", epochs=1, batch_size=1, grad_accum_steps=4,
                  learning_rate=0.05, weight_decay=0.0, optimizer="adamw", schedule="constant", warmup_ratio=0.0,
                  max_grad_norm=0.0, precision="fp16", eval_every_steps=1, eval_at_start=True, log_every_steps=1,
                  resume_every_steps=0, seed=7)
    values.update(overrides)
    return LoopConfig(**values)


def quiet(*_args, **_kwargs):
    pass


def test_accumulation_groups_handle_partial_micro_batches_and_groups():
    groups = accumulation_groups(num_rows=11, batch_size=2, grad_accum_steps=4)
    assert [len(group) for group in groups] == [4, 2]           # 6 micro-batches -> 4 + 2
    assert groups[0] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert groups[1] == [(8, 10), (10, 11)]                     # last micro-batch holds a single row
    assert len(accumulation_groups(30000, 2, 16)) == 938        # SFT guide config
    assert len(accumulation_groups(15000, 1, 32)) == 469        # DPO guide config


def test_learning_rate_schedule_shapes():
    total, warmup = 938, math.ceil(0.03 * 938)
    assert warmup == 29
    assert lr_factor(0, total, warmup, "cosine") == pytest.approx(1 / 29)
    assert lr_factor(28, total, warmup, "cosine") == pytest.approx(1.0)
    assert lr_factor(29, total, warmup, "cosine") == pytest.approx(1.0)
    midpoint = warmup + (total - warmup) // 2
    assert 0.45 < lr_factor(midpoint, total, warmup, "cosine") < 0.55
    assert lr_factor(total, total, warmup, "cosine") == pytest.approx(0.0, abs=1e-12)
    assert all(lr_factor(step, 469, 0, "constant") == 1.0 for step in (0, 100, 468))
    assert lr_factor(50, 100, 0, "linear") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        lr_factor(1, 10, 0, "unknown")


def test_amp_is_disabled_on_cpu_and_weights_stay_fp32():
    amp = AmpSettings(torch.device("cpu"), "fp16")
    assert not amp.enabled and not amp.use_scaler and amp.effective == "fp32"
    scaler = amp.make_scaler()
    assert not scaler.is_enabled()
    model = ToyCausalLM(8)
    with amp.autocast():
        assert model.logit_bias.dtype == torch.float32


def test_weight_decay_skips_one_dimensional_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(model, "adamw", 1e-3, 0.1)
    decay = [group for group in optimizer.param_groups if group["weight_decay"] > 0]
    no_decay = [group for group in optimizer.param_groups if group["weight_decay"] == 0]
    assert all(parameter.ndim >= 2 for group in decay for parameter in group["params"])
    assert all(parameter.ndim < 2 for group in no_decay for parameter in group["params"])
    assert isinstance(build_optimizer(model, "rmsprop", 1e-3, 0.0), torch.optim.RMSprop)
    with pytest.raises(ValueError):
        build_optimizer(model, "sgd", 1e-3, 0.0)


def _dpo_task(tokenizer, policy, reference, val_rows, beta=0.5):
    config = DPOTaskConfig(beta=beta, max_length=32, max_prompt_length=16, template="alpaca", eval_batch_size=3)
    return DPOTask(policy, reference, tokenizer, val_rows, config, torch.device("cpu"))


def test_gradient_accumulation_matches_manual_group_means_including_the_partial_group(tmp_path):
    """6 pairs, batch 1, accumulation 4 -> two optimizer steps over 4 and then 2 pairs."""

    tokenizer = TemplateToyTokenizer()
    torch.manual_seed(0)
    start = BigramLM(tokenizer.vocab_size, seed=21)
    reference = BigramLM(tokenizer.vocab_size, seed=22)
    rows = DPO_ROWS[:6]
    config = loop_config(tmp_path, grad_accum_steps=4, eval_at_start=False, eval_every_steps=0, learning_rate=0.01)

    policy = copy.deepcopy(start)
    task = _dpo_task(tokenizer, policy, copy.deepcopy(reference), rows[:2])
    summary = run_training(config, policy, rows, task, torch.device("cpu"), log=quiet)
    assert summary["optimizer_steps"] == 2 and summary["examples_seen"] == 6

    order = epoch_order(len(rows), config.seed, 0)

    def manual_run(divide_by_group_size: bool) -> torch.Tensor:
        manual = copy.deepcopy(start)
        manual_task = _dpo_task(tokenizer, manual, copy.deepcopy(reference), rows[:2])
        optimizer = build_optimizer(manual, "adamw", 0.01, 0.0)
        for group in (order[:4], order[4:]):
            optimizer.zero_grad()
            total = 0.0
            for index in group:
                batch = manual_task.collate([rows[index]])
                reference_chosen = sequence_logprobs(reference, batch["chosen"]).detach()
                reference_rejected = sequence_logprobs(reference, batch["rejected"]).detach()
                total = total + dpo_loss_from_logps(
                    sequence_logprobs(manual, batch["chosen"]), sequence_logprobs(manual, batch["rejected"]),
                    reference_chosen, reference_rejected, 0.5)
            (total / (len(group) if divide_by_group_size else 4)).backward()
            optimizer.step()
        return manual.table.detach()

    # Correct: every group is a true mean, including the short final group of 2.
    torch.testing.assert_close(policy.table.detach(), manual_run(divide_by_group_size=True), rtol=1e-5, atol=1e-6)
    # The common bug (always dividing by the nominal accumulation of 4) gives different weights.
    assert not torch.allclose(policy.table.detach(), manual_run(divide_by_group_size=False), rtol=1e-5, atol=1e-6)


def test_gradient_clipping_limits_the_update_norm(tmp_path):
    tokenizer = TemplateToyTokenizer()
    policy = BigramLM(tokenizer.vocab_size, seed=31)
    reference = BigramLM(tokenizer.vocab_size, seed=32)
    task = _dpo_task(tokenizer, policy, reference, DPO_ROWS[:2], beta=5.0)
    config = loop_config(tmp_path, max_grad_norm=1e-3, eval_at_start=False, eval_every_steps=0, max_steps=1)
    run_training(config, policy, DPO_ROWS[:4], task, torch.device("cpu"), log=quiet)
    records = [record for record in read_jsonl(tmp_path / "run" / "metrics.jsonl") if record["type"] == "train"]
    assert records[0]["grad_norm"] > 1e-3  # the logged norm is the pre-clip norm


def test_sft_training_reduces_validation_loss_and_writes_best_and_final(tmp_path):
    tokenizer = TemplateToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    task_config = SFTTaskConfig(max_length=64, max_prompt_length=48, template="alpaca", eval_batch_size=4)
    task = SFTTask(model, tokenizer, SFT_ROWS[8:], task_config, torch.device("cpu"), checkpoint_meta={"base_model": "toy"})
    config = loop_config(tmp_path, batch_size=2, grad_accum_steps=2, epochs=3, learning_rate=0.1, eval_every_steps=2)
    summary = run_training(config, model, SFT_ROWS[:8], task, torch.device("cpu"), log=quiet)

    run_dir = tmp_path / "run"
    records = read_jsonl(run_dir / "metrics.jsonl")
    evals = [record for record in records if record["type"] == "eval"]
    assert evals[0]["tag"] == "initial" and evals[0]["is_best"] is False
    assert evals[-1]["val_loss"] < evals[0]["val_loss"]
    assert summary["completed"] and summary["optimizer_steps"] == 6  # 8 rows / 2 / 2 = 2 steps x 3 epochs
    assert summary["best_metrics"]["val_loss"] == min(record["val_loss"] for record in evals[1:])
    for name in ("best", "final"):
        assert (run_dir / name / "pytorch_model.bin").exists()
        meta = load_checkpoint_meta(run_dir / name)
        assert meta["template"] == "alpaca" and meta["stage"] == "sft" and meta["base_model"] == "toy"
    assert load_checkpoint_meta(run_dir / "best")["step"] == summary["best_step"]
    assert already_complete(run_dir)["best_step"] == summary["best_step"]
    assert not (run_dir / "resume" / "state.pt").exists()
    assert not list(run_dir.glob("*.tmp"))
    loop = read_json(run_dir / "loop_config.json")
    assert loop["total_steps"] == 6 and loop["precision_effective"] == "fp32"


def test_dpo_training_starts_tied_learns_the_preference_and_never_touches_the_reference(tmp_path):
    tokenizer = TemplateToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    reference.train()
    reference_before = reference.logit_bias.detach().clone()
    policy_before = policy.logit_bias.detach().clone()
    task = _dpo_task(tokenizer, policy, reference, DPO_ROWS[6:])
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())

    config = loop_config(tmp_path, batch_size=1, grad_accum_steps=3, epochs=4, learning_rate=0.05)
    summary = run_training(config, policy, DPO_ROWS[:6], task, torch.device("cpu"), log=quiet)

    evals = [record for record in read_jsonl(tmp_path / "run" / "metrics.jsonl") if record["type"] == "eval"]
    initial = evals[0]
    # Policy == reference before the first update: all rewards tie, strict accuracy is exactly 0.
    assert initial["val_reward_accuracy"] == 0.0
    assert initial["val_reward_margin"] == 0.0
    assert initial["val_tie_fraction"] == 1.0
    assert initial["val_loss"] == pytest.approx(math.log(2.0), rel=1e-6)

    assert evals[-1]["val_reward_accuracy"] == 1.0
    assert evals[-1]["val_reward_margin"] > 0
    assert evals[-1]["val_loss"] < math.log(2.0)
    assert summary["best_metrics"]["val_reward_accuracy"] == 1.0
    assert torch.equal(reference.logit_bias.detach(), reference_before)
    assert reference.logit_bias.grad is None
    assert not torch.equal(policy.logit_bias.detach(), policy_before)
    meta = load_checkpoint_meta(tmp_path / "run" / "best")
    assert meta["stage"] == "dpo" and meta["beta"] == 0.5


def test_single_forward_and_two_forward_dpo_paths_agree():
    tokenizer = TemplateToyTokenizer()
    policy, reference = BigramLM(tokenizer.vocab_size, seed=41), BigramLM(tokenizer.vocab_size, seed=42)
    joined = _dpo_task(tokenizer, policy, reference, DPO_ROWS[:2])
    separate = _dpo_task(tokenizer, policy, reference, DPO_ROWS[:2])
    separate.config.single_forward = False
    batch = joined.collate(DPO_ROWS[:5])
    loss_joined, stats_joined = joined.loss(batch)
    loss_separate, stats_separate = separate.loss(batch)
    torch.testing.assert_close(loss_joined, loss_separate)
    assert stats_joined == pytest.approx(stats_separate)


def test_interrupted_run_resumes_to_the_same_weights(tmp_path):
    tokenizer = TemplateToyTokenizer()

    def fresh():
        policy = BigramLM(tokenizer.vocab_size, seed=51)
        reference = BigramLM(tokenizer.vocab_size, seed=51)
        return policy, _dpo_task(tokenizer, policy, reference, DPO_ROWS[8:])

    settings = dict(batch_size=1, grad_accum_steps=2, epochs=2, learning_rate=0.02, schedule="cosine", warmup_ratio=0.25)
    policy_full, task_full = fresh()
    full = run_training(loop_config(tmp_path / "full", **settings), policy_full, DPO_ROWS[:8], task_full,
                        torch.device("cpu"), log=quiet)
    assert full["optimizer_steps"] == 8

    # First session: the time budget (0 minutes) stops the run after one optimizer step.
    policy_part, task_part = fresh()
    interrupted = loop_config(tmp_path / "part", max_minutes=0.0, **settings)
    partial = run_training(interrupted, policy_part, DPO_ROWS[:8], task_part, torch.device("cpu"), log=quiet)
    run_dir = tmp_path / "part" / "run"
    assert partial["completed"] is False and partial["optimizer_steps"] == 1
    assert (run_dir / "resume" / "state.pt").exists() and (run_dir / "INCOMPLETE.json").exists()
    assert already_complete(run_dir) is None

    # Second session: new process state (fresh model objects), resume and finish.
    policy_resumed, task_resumed = fresh()
    resumed = run_training(loop_config(tmp_path / "part", **settings), policy_resumed, DPO_ROWS[:8], task_resumed,
                           torch.device("cpu"), resume=True, log=quiet)
    assert resumed["completed"] and resumed["optimizer_steps"] == 8 and resumed["examples_seen"] == 16
    torch.testing.assert_close(policy_resumed.table, policy_full.table, rtol=1e-6, atol=1e-7)
    assert not (run_dir / "resume" / "state.pt").exists() and not (run_dir / "INCOMPLETE.json").exists()
    steps = [record["step"] for record in read_jsonl(run_dir / "metrics.jsonl") if record["type"] == "train"]
    assert steps == list(range(1, 9))  # metrics continue without duplicates or gaps


def test_restart_without_resume_starts_clean(tmp_path):
    tokenizer = TemplateToyTokenizer()
    for _ in range(2):
        model = ToyCausalLM(tokenizer.vocab_size)
        task = SFTTask(model, tokenizer, SFT_ROWS[8:], SFTTaskConfig(max_length=64), torch.device("cpu"))
        run_training(loop_config(tmp_path, batch_size=2, grad_accum_steps=2), model, SFT_ROWS[:8], task,
                     torch.device("cpu"), log=quiet)
    records = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert [record["step"] for record in records if record["type"] == "train"] == [1, 2]


def test_training_curves_are_plotted_from_real_run_files(tmp_path):
    pytest.importorskip("matplotlib")
    from alignment.plotting import ema, load_run, plot_run

    tokenizer = TemplateToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    task = _dpo_task(tokenizer, policy, reference, DPO_ROWS[6:])
    run_training(loop_config(tmp_path, grad_accum_steps=3, epochs=2), policy, DPO_ROWS[:6], task, torch.device("cpu"),
                 log=quiet)
    run = load_run(tmp_path / "run")
    assert run["kind"] == "dpo" and len(run["train"]) == 4 and len(run["eval"]) == 5  # initial + 3 periodic + final

    png = plot_run(tmp_path / "run", tmp_path / "plots")
    assert png.exists() and png.stat().st_size > 10_000
    table = (tmp_path / "plots" / "run_eval_table.csv").read_text(encoding="utf-8").splitlines()
    assert table[0].startswith("step,tag,is_best,val_loss") and len(table) == 6

    assert ema([1.0, 1.0, 1.0], 0.1) == pytest.approx([1.0, 1.0, 1.0])  # bias-corrected: no start-up dip
    smoothed = ema([0.0, 10.0], 0.5)
    assert smoothed[0] == 0.0 and smoothed[1] == pytest.approx(5.0 / 0.75)
