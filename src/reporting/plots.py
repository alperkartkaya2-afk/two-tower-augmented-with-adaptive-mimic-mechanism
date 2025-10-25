"""Plotting helpers for experiment artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

# Force a non-interactive backend for headless environments (CI, servers, etc.).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def save_loss_curves(
    loss_history: Mapping[str, Sequence[float]],
    *,
    output_path: Path | str,
    xlabel: str = "Epoch",
    ylabel: str = "BCE Loss",
    title: str = "Training / Validation / Test Loss",
) -> Path:
    """
    Save line plots for multiple loss series on the same axes.

    Parameters
    ----------
    loss_history:
        Mapping of series label to the loss values ordered by epoch (1..N).
    output_path:
        Target image path. Directories are created automatically.
    xlabel, ylabel, title:
        Axis and chart annotations for readability.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    epochs = None
    for label, values in loss_history.items():
        if not values:
            continue
        has_data = True
        epochs = range(1, len(values) + 1) if epochs is None else range(1, len(values) + 1)
        ax.plot(
            epochs,
            values,
            marker="o",
            linestyle="-",
            label=label,
        )

    if not has_data:
        plt.close(fig)
        raise ValueError("Loss history is empty; nothing to plot.")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path
