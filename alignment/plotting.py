"""Training-curve figures from ``metrics.jsonl``.

One measure per panel (no dual axes): SFT gets a loss panel; DPO and persona runs get
stacked loss / reward-accuracy / reward-margin panels that share the step axis.
Training curves are exponentially smoothed and the smoothing factor is stated in the
legend; validation points are shown unsmoothed. The selected (best) checkpoint is marked.
A CSV of the validation rows is written next to every figure.
"""

from __future__ import annotations

import csv
from pathlib import Path

from alignment.io_utils import COMPLETE_MARKER, read_json, read_jsonl

# Light-surface palette: categorical slot 1 (training) and slot 2 (validation), neutral ink for text.
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

PANELS = {
    "sft": [("loss", "val_loss", "Loss (mean NLL per response token)", None)],
    "dpo": [
        ("loss", "val_loss", "DPO loss", None),
        ("reward_accuracy", "val_reward_accuracy", "Reward accuracy", 0.5),
        ("reward_margin", "val_reward_margin", "Reward margin", 0.0),
    ],
}


def ema(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average with bias correction (first value is not over-weighted)."""

    smoothed, running = [], 0.0
    for index, value in enumerate(values, start=1):
        running = alpha * value + (1 - alpha) * running
        smoothed.append(running / (1 - (1 - alpha) ** index))
    return smoothed


def load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    records = read_jsonl(run_dir / "metrics.jsonl")
    marker = run_dir / COMPLETE_MARKER
    summary = read_json(marker) if marker.exists() else {}
    train = [record for record in records if record.get("type") == "train"]
    evals = [record for record in records if record.get("type") == "eval"]
    kind = "dpo" if any("val_reward_accuracy" in record for record in evals) else "sft"
    return {"name": run_dir.name, "dir": run_dir, "train": train, "eval": evals, "summary": summary, "kind": kind}


def _series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(record["step"], record[key]) for record in records if record.get(key) is not None]
    return [step for step, _ in points], [value for _, value in points]


def plot_run(run_dir, out_dir, alpha: float = 0.1, title: str | None = None) -> Path:
    """Write ``<out_dir>/<run>_curves.png`` and ``<run>_eval_table.csv``; return the PNG path."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = load_run(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = PANELS[run["kind"]]
    best_step = run["summary"].get("best_step")
    if best_step is None:
        flagged = [record["step"] for record in run["eval"] if record.get("is_best")]
        best_step = flagged[-1] if flagged else None

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    figure, axes = plt.subplots(len(panels), 1, figsize=(7.0, 2.7 * len(panels) + 0.5), sharex=True, squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    for axis, (train_key, val_key, label, reference_line) in zip(axes[:, 0], panels):
        axis.set_facecolor(SURFACE)
        steps, values = _series(run["train"], train_key)
        if steps:
            axis.plot(steps, values, color=TRAIN_COLOR, linewidth=0.8, alpha=0.3, label="train (logged)")
            axis.plot(steps, ema(values, alpha), color=TRAIN_COLOR, linewidth=2.0, label=f"train (EMA, alpha={alpha:g})")
        steps, values = _series(run["eval"], val_key)
        if steps:
            axis.plot(steps, values, color=VAL_COLOR, linewidth=2.0, marker="o", markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label="validation (unsmoothed)")
        if reference_line is not None:
            axis.axhline(reference_line, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        if best_step is not None:
            axis.axvline(best_step, color=INK_MUTED, linewidth=1.0, linestyle=(0, (2, 2)), zorder=0)
        axis.set_ylabel(label, color=INK_SECONDARY)
        axis.grid(True, axis="y", color=GRID, linewidth=0.6)
        axis.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=3)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS)
    top = axes[0, 0]
    if best_step is not None:
        # Label sits on the side of the marker line that has room inside the axes.
        low, high = top.get_xlim()
        on_right_half = best_step > (low + high) / 2
        top.annotate(f"selected checkpoint (step {best_step})", xy=(best_step, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(-4 if on_right_half else 4, -10), textcoords="offset points",
                     ha="right" if on_right_half else "left", fontsize=8, color=INK_SECONDARY)
    handles, labels = top.get_legend_handles_labels()
    if handles:
        # Legend above the plot area so it never covers data or the checkpoint label.
        top.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, fontsize=8,
                   borderaxespad=0.0, labelcolor=INK_SECONDARY)
    axes[-1, 0].set_xlabel("Optimizer step", color=INK_SECONDARY)
    figure.suptitle(title or f"{run['name']}: training curves", color=INK, fontsize=11, x=0.02, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = out_dir / f"{run['name']}_curves.png"
    figure.savefig(png_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)

    columns = ["step", "tag", "is_best"] + sorted(
        {key for record in run["eval"] for key in record if key.startswith("val_")}
    )
    with open(out_dir / f"{run['name']}_eval_table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in run["eval"]:
            writer.writerow(record)
    return png_path
