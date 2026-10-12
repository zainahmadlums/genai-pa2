"""Hand-written optimisation loop shared by the SFT and DPO trainers.

No Hugging Face ``Trainer`` or TRL code is used. The loop provides:

* gradient accumulation with a correct final partial group (each micro-batch loss is
  divided by the number of micro-batches actually in its group);
* fp16 autocast with fp32 weights and a ``GradScaler``; gradients are unscaled before
  clipping;
* warmup + cosine / linear / constant learning-rate schedules stepped per optimizer step;
* periodic validation, ``metrics.jsonl`` appended and fsynced after every record;
* ``best/`` (by the task's selection key) and ``final/`` checkpoints only;
* an optional single resume file (``<resume_dir>/state.pt``: model, optimizer, scheduler,
  scaler and counters) that is replaced on every save and deleted on success, so disk use
  stays bounded. It is off by default because it is about 6 GB for the 0.5B model.

A *task* object supplies the model-specific pieces::

    task.collate(rows) -> batch
    task.loss(batch) -> (scalar loss, {name: float})   # called under autocast
    task.evaluate() -> {name: float}
    task.selection_key(metrics) -> comparable, higher is better
    task.save(directory, meta) -> None
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from alignment.io_utils import log_line, write_jsonl
from alignment.train_utils import (
    COMPLETE_MARKER,
    AmpSettings,
    accumulation_groups,
    append_jsonl,
    build_optimizer,
    disk_free_gb,
    epoch_order,
    load_checkpoint_meta,
    lr_factor,
    read_json,
    set_seed,
    write_json,
)


@dataclass
class LoopConfig:
    output_dir: str
    run_name: str = "run"
    epochs: int = 1
    batch_size: int = 2
    grad_accum_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    optimizer: str = "adamw"
    schedule: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    precision: str = "fp16"
    eval_every_steps: int = 100
    eval_at_start: bool = True
    log_every_steps: int = 10
    resume_every_steps: int = 0
    resume_dir: str | None = None  # default <output_dir>/resume; on Kaggle point this at /kaggle/temp
    keep_resume_state: bool = False
    max_steps: int | None = None
    max_minutes: float | None = None
    seed: int = 42


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _trim_metrics(metrics_path: Path, step: int) -> int:
    """Drop metric records written after ``step`` (atomic rewrite); return how many were removed.

    A torn last line left by a crash in the middle of an append is dropped as well.
    """

    if not metrics_path.exists():
        return 0
    kept, removed = [], 0
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                removed += 1
                continue
            if record.get("step", 0) > step:
                removed += 1
            else:
                kept.append(record)
    if removed:
        write_jsonl(metrics_path, kept)
    return removed


def _reconcile_best_checkpoint(out_dir: Path, state: dict, task) -> dict:
    """Make the restored best-checkpoint bookkeeping agree with the ``best/`` directory on disk.

    ``best/`` is replaced whenever an evaluation improves, also between two snapshots. After
    rolling back to a snapshot it can therefore hold a checkpoint from a later step than the
    snapshot knows about. Those weights and the validation metrics stored with them are
    genuine (same data order, same code), and the checkpoint the snapshot refers to no longer
    exists, so the on-disk checkpoint is adopted as the current best and the fact is recorded.
    Without usable metadata the best tracking is reset; the next evaluation then sets it.
    """

    meta = load_checkpoint_meta(out_dir / "best")
    disk_step, disk_metrics = meta.get("step"), meta.get("metrics")
    note = {"best_step_in_snapshot": state["best_step"], "best_checkpoint_step": disk_step}
    if disk_step == state["best_step"]:
        note["best_checkpoint"] = "consistent"
    elif disk_step is not None and isinstance(disk_metrics, dict):
        state["best_key"] = list(task.selection_key(disk_metrics))
        state["best_step"], state["best_metrics"] = disk_step, disk_metrics
        note["best_checkpoint"] = "adopted_from_disk"
    else:
        state["best_key"], state["best_step"], state["best_metrics"] = None, None, None
        note["best_checkpoint"] = "reset"
    return note


def run_training(config: LoopConfig, model, rows, task, device, resume: bool = False, log=log_line) -> dict:
    """Train ``model`` on ``rows`` and return a summary dict (also written to disk)."""

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    resume_path = (Path(config.resume_dir) if config.resume_dir else out_dir / "resume") / "state.pt"
    if config.resume_every_steps or config.max_minutes is not None:
        resume_path.parent.mkdir(parents=True, exist_ok=True)  # fail now, not at the first save
    device = torch.device(device)
    amp = AmpSettings(device, config.precision)
    set_seed(config.seed)

    groups_per_epoch = accumulation_groups(len(rows), config.batch_size, config.grad_accum_steps)
    steps_per_epoch = len(groups_per_epoch)
    total_steps = steps_per_epoch * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, int(config.max_steps))
    if total_steps < 1:
        raise ValueError("Nothing to train: no optimizer steps (check dataset size and max_steps)")
    warmup_steps = math.ceil(config.warmup_ratio * total_steps) if config.warmup_ratio > 0 else 0

    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup_steps, config.schedule)
    )
    scaler = amp.make_scaler()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    # "step" counts ATTEMPTED optimizer steps (one per accumulation group): it fixes the position
    # in the data, the evaluation cadence and the end of training. "applied_steps" counts the
    # updates that really changed the weights; "skipped_steps" the groups whose update was
    # dropped (no finite loss, or non-finite gradients). The LR schedule advances per applied step.
    state = {
        "step": 0,
        "applied_steps": 0,
        "skipped_steps": 0,
        "examples_seen": 0,
        "best_key": None,
        "best_step": None,
        "best_metrics": None,
        "elapsed_seconds": 0.0,
        "skipped_nonfinite": 0,
        "last_eval_step": None,
        "last_eval_metrics": None,
    }

    resumed = False
    if resume and resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        state.update(payload["state"])
        if "applied_steps" not in payload["state"]:  # snapshot written before applied/skipped were tracked
            state["applied_steps"] = state["step"]
        resumed = True
        # The snapshot may be older than the last lines of metrics.jsonl (the run went on after it was
        # written and then died). Those steps are trained again, so their old records are removed.
        trimmed = _trim_metrics(metrics_path, state["step"])
        best_note = _reconcile_best_checkpoint(out_dir, state, task)
        append_jsonl(metrics_path, {
            "type": "resume", "step": state["step"], "trimmed_records": trimmed, **best_note,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        log(f"[{config.run_name}] resumed from {resume_path} at optimizer step {state['step']}; removed {trimmed} "
            f"metric records written after that step; best checkpoint: {best_note['best_checkpoint']}")
    else:
        if metrics_path.exists():
            metrics_path.unlink()
        if (out_dir / COMPLETE_MARKER).exists():
            (out_dir / COMPLETE_MARKER).unlink()

    write_json(
        out_dir / "loop_config.json",
        {**asdict(config), "total_steps": total_steps, "warmup_steps": warmup_steps,
         "steps_per_epoch": steps_per_epoch, "train_rows": len(rows), "precision_effective": amp.effective,
         "grad_scaler": amp.use_scaler, "device": str(device), "resumed": resumed},
    )
    log(
        f"[{config.run_name}] rows={len(rows)} batch={config.batch_size} accum={config.grad_accum_steps} "
        f"optimizer_steps={total_steps} warmup={warmup_steps} precision={amp.effective} device={device}"
    )

    session_start = time.time()
    elapsed_before = state["elapsed_seconds"]

    def elapsed() -> float:
        return elapsed_before + (time.time() - session_start)

    def run_eval(tag: str, eligible_for_best: bool) -> dict:
        metrics = task.evaluate()
        model.train()
        record = {"type": "eval", "tag": tag, "step": state["step"], "examples_seen": state["examples_seen"],
                  "elapsed_seconds": round(elapsed(), 1), **metrics}
        is_best = False
        if eligible_for_best:
            key = task.selection_key(metrics)
            if state["best_key"] is None or key > tuple(state["best_key"]):
                is_best = True
                state["best_key"], state["best_step"], state["best_metrics"] = list(key), state["step"], metrics
                task.save(out_dir / "best", {"kind": "best", "step": state["step"], "metrics": metrics})
        record["is_best"] = is_best
        append_jsonl(metrics_path, record)
        state["last_eval_step"], state["last_eval_metrics"] = state["step"], metrics
        shown = " ".join(f"{name}={value:.4f}" for name, value in metrics.items() if isinstance(value, float))
        log(f"[{config.run_name}] eval step {state['step']}/{total_steps} {shown}{' *best*' if is_best else ''}")
        return metrics

    def save_resume() -> None:
        state["elapsed_seconds"] = elapsed()
        _atomic_torch_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "state": dict(state)},
            resume_path,
        )
        log(f"[{config.run_name}] resume state saved at step {state['step']} (free disk {disk_free_gb(out_dir):.1f} GB)")

    if config.eval_at_start and state["step"] == 0 and not resumed:
        # Baseline numbers before any update. Not eligible for "best": for DPO the policy
        # equals the reference here, so reward accuracy is exactly 0 by the strict tie rule.
        run_eval("initial", eligible_for_best=False)

    model.train()
    window = {"loss": 0.0, "groups": 0, "stats": {}, "grad_norm": 0.0, "norm_groups": 0}
    stopped_early = False
    start_epoch, start_group = divmod(state["step"], steps_per_epoch)

    for epoch in range(start_epoch, config.epochs):
        if state["step"] >= total_steps or stopped_early:
            break
        order = epoch_order(len(rows), config.seed, epoch)
        first_group = start_group if epoch == start_epoch else 0
        for group_index in range(first_group, steps_per_epoch):
            if state["step"] >= total_steps:
                break
            group = groups_per_epoch[group_index]
            group_size = len(group)
            optimizer.zero_grad(set_to_none=True)
            group_loss, group_stats, used = 0.0, {}, 0
            for start, end in group:
                batch = task.collate([rows[index] for index in order[start:end]])
                with amp.autocast():
                    loss, stats = task.loss(batch)
                state["examples_seen"] += end - start
                if not torch.isfinite(loss):
                    state["skipped_nonfinite"] += 1
                    continue
                # Divide by the true number of micro-batches in this group so a short
                # final group is still an average, not a down-weighted partial sum.
                scaler.scale(loss / group_size).backward()
                group_loss += float(loss.detach())
                used += 1
                for name, value in stats.items():
                    group_stats[name] = group_stats.get(name, 0.0) + float(value)
            applied, grad_norm = False, float("nan")
            if used:
                scaler.unscale_(optimizer)
                max_norm = config.max_grad_norm if config.max_grad_norm and config.max_grad_norm > 0 else float("inf")
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm))
                finite = math.isfinite(grad_norm)
                if scaler.is_enabled():
                    # An enabled GradScaler skips optimizer.step() itself when it saw inf/nan gradients
                    # and then lowers its scale in update(); that drop is how the skip is observed.
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    applied = finite and scaler.get_scale() >= scale_before
                elif finite:
                    scaler.step(optimizer)
                    scaler.update()
                    applied = True
                # else: no scaler guards the weights, so non-finite gradients are simply not applied.
            current_lr = optimizer.param_groups[0]["lr"]
            state["step"] += 1
            if applied:
                state["applied_steps"] += 1
                scheduler.step()  # the schedule follows real updates, not skipped ones
            else:
                state["skipped_steps"] += 1
            # The group is finished (applied or dropped), so its gradients are released now instead of at
            # the start of the next group: evaluation and the resume save below then run without a second,
            # weight-sized copy of every parameter on the device.
            optimizer.zero_grad(set_to_none=True)

            if used:
                window["loss"] += group_loss / used
                window["groups"] += 1
                for name, value in group_stats.items():
                    window["stats"][name] = window["stats"].get(name, 0.0) + value / used
            if applied:
                # Only finite norms of applied steps are averaged; a skipped step adds neither a 0 nor a count.
                window["grad_norm"] += grad_norm
                window["norm_groups"] += 1

            is_last = state["step"] >= total_steps
            if state["step"] % max(1, config.log_every_steps) == 0 or is_last:
                count = window["groups"]
                record = {"type": "train", "step": state["step"], "epoch": epoch,
                          "examples_seen": state["examples_seen"],
                          "loss": window["loss"] / count if count else None,
                          "lr": current_lr,
                          "grad_norm": window["grad_norm"] / window["norm_groups"] if window["norm_groups"] else None,
                          "applied_steps": state["applied_steps"], "skipped_steps": state["skipped_steps"],
                          "skipped_nonfinite": state["skipped_nonfinite"], "elapsed_seconds": round(elapsed(), 1)}
                record.update({name: value / count for name, value in window["stats"].items() if count})
                append_jsonl(metrics_path, record)
                shown = {name: "n/a" if record[name] is None else f"{record[name]:.4f}" for name in ("loss", "grad_norm")}
                log(f"[{config.run_name}] step {state['step']}/{total_steps} loss={shown['loss']} "
                    f"lr={current_lr:.2e} grad_norm={shown['grad_norm']} skipped_steps={state['skipped_steps']} "
                    f"elapsed={record['elapsed_seconds']:.0f}s")
                window = {"loss": 0.0, "groups": 0, "stats": {}, "grad_norm": 0.0, "norm_groups": 0}

            if config.eval_every_steps and state["step"] % config.eval_every_steps == 0 and not is_last:
                run_eval("periodic", eligible_for_best=True)
            if config.resume_every_steps and state["step"] % config.resume_every_steps == 0 and not is_last:
                save_resume()
            if config.max_minutes is not None and not is_last and elapsed() / 60.0 >= config.max_minutes:
                log(f"[{config.run_name}] time budget of {config.max_minutes} min reached; saving resume state")
                save_resume()
                stopped_early = True
                break

    completed = state["step"] >= total_steps
    summary = {
        "run_name": config.run_name,
        "completed": completed,
        "optimizer_steps": state["applied_steps"],   # updates that were really applied
        "attempted_steps": state["step"],
        "skipped_steps": state["skipped_steps"],     # attempted - applied
        "total_steps": total_steps,
        "examples_seen": state["examples_seen"],
        "skipped_nonfinite": state["skipped_nonfinite"],  # micro-batches dropped for a non-finite loss
        "elapsed_seconds": round(elapsed(), 1),
        "precision": amp.effective,
    }
    if not completed:
        summary.update({"best_step": state["best_step"], "best_metrics": state["best_metrics"]})
        write_json(out_dir / "INCOMPLETE.json", summary)
        log(f"[{config.run_name}] stopped at step {state['step']}/{total_steps}; rerun with --resume to continue")
        return summary

    final_metrics = (
        state["last_eval_metrics"] if state["last_eval_step"] == state["step"] else run_eval("final", eligible_for_best=True)
    )
    task.save(out_dir / "final", {"kind": "final", "step": state["step"], "metrics": final_metrics})
    if state["best_step"] is None:
        state["best_step"], state["best_metrics"] = state["step"], final_metrics
        task.save(out_dir / "best", {"kind": "best", "step": state["step"], "metrics": final_metrics})
    summary.update({
        "best_step": state["best_step"],
        "best_metrics": state["best_metrics"],
        "final_metrics": final_metrics,
        "best_checkpoint": str(out_dir / "best"),
        "final_checkpoint": str(out_dir / "final"),
    })
    write_json(out_dir / COMPLETE_MARKER, summary)
    if (out_dir / "INCOMPLETE.json").exists():
        (out_dir / "INCOMPLETE.json").unlink()
    if resume_path.exists() and not config.keep_resume_state:
        resume_path.unlink()
    log(f"[{config.run_name}] done: best step {state['best_step']} -> {out_dir / 'best'}; final -> {out_dir / 'final'}")
    return summary


def remove_smoke_checkpoints(output_dir, log=log_line) -> None:
    """Delete the weights written by a ``--smoke`` run (about 2 GB each); metrics are kept.

    Only directories whose name ends in ``_smoke`` are touched.
    """

    import shutil

    output_dir = Path(output_dir)
    if not output_dir.name.endswith("_smoke"):
        return
    for name in ("best", "final"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    marker = output_dir / COMPLETE_MARKER
    if marker.exists():
        marker.unlink()  # a later smoke run should train again instead of being skipped
    log(f"[smoke] finished; smoke checkpoints under {output_dir} removed to save disk (metrics kept)")


def already_complete(output_dir) -> dict | None:
    """Return the saved summary when this run finished earlier, else ``None``."""

    marker = Path(output_dir) / COMPLETE_MARKER
    return read_json(marker) if marker.exists() else None
