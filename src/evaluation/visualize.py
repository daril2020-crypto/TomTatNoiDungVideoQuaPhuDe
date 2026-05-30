import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path


METRIC_LABELS = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "bleu": "BLEU-4",
}

METHOD_COLORS = {
    "TextRank": "#4C72B0",
    "BART": "#DD8452",
    "PEGASUS": "#55A868",
    "Flan-T5": "#C44E52",
}


def plot_comparison_bar(
    results: dict[str, dict],
    metrics: list[str] | None = None,
    save_path: str | None = None,
    title: str = "Model Comparison",
) -> plt.Figure:
    """Bar chart comparing multiple methods across multiple metrics.

    Args:
        results: {method_name: {metric_name: value, ...}, ...}
        metrics: which metrics to show (defaults to ROUGE-1/2/L + BERTScore F1)
        save_path: if given, save figure to this path
    """
    if metrics is None:
        metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]

    methods = list(results.keys())
    n_metrics = len(metrics)
    n_methods = len(methods)
    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        values = [results[method].get(m, 0) for m in metrics]
        bars = ax.bar(
            x + i * width - (n_methods - 1) * width / 2,
            values,
            width,
            label=method,
            color=METHOD_COLORS.get(method, None),
            alpha=0.85,
            edgecolor="white",
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved to {save_path}")

    return fig


def plot_training_curves(
    log_history: list[dict],
    metric: str = "rougeL",
    save_path: str | None = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """Plot train loss and validation metric across epochs.

    log_history: list of dicts from trainer.state.log_history
    """
    train_steps, train_losses = [], []
    eval_steps, eval_metrics = [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if f"eval_{metric}" in entry:
            eval_steps.append(entry["step"])
            eval_metrics.append(entry[f"eval_{metric}"])

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(train_steps, train_losses, color="#4C72B0", label="Train Loss", linewidth=1.5)
    ax2.plot(
        eval_steps,
        eval_metrics,
        color="#DD8452",
        label=f"Val {METRIC_LABELS.get(metric, metric)}",
        linewidth=1.5,
        marker="o",
        markersize=5,
    )

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss", color="#4C72B0")
    ax2.set_ylabel(METRIC_LABELS.get(metric, metric), color="#DD8452")
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved to {save_path}")

    return fig


def load_results(results_dir: str = "results/scores") -> dict[str, dict]:
    """Load all *_results.json files from a directory into a dict."""
    results = {}
    for path in sorted(Path(results_dir).glob("*_results.json")):
        method = path.stem.replace("_results", "").upper()
        with open(path) as f:
            data = json.load(f)
        results[method] = data.get("scores", data)
    return results
