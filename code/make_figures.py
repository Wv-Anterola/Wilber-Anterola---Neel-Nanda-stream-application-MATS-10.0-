#!/usr/bin/env python3
"""
Render plots from the stored results.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    with open(INPUT_DIR / "exp2_suppress_reasoning.json", "r", encoding="utf-8") as f:
        exp2_data = json.load(f)
    with open(INPUT_DIR / "head_level_results.json", "r", encoding="utf-8") as f:
        head_data = json.load(f)
    with open(INPUT_DIR / "quality_assessment.json", "r", encoding="utf-8") as f:
        quality_data = json.load(f)
    return exp2_data, head_data, quality_data


def plot_layer_heatmap(exp2_data):
    all_layers = sorted([int(k) for k in exp2_data[0]["layer_results"].keys()])
    n_prompts = len(exp2_data)

    matrix = np.zeros((n_prompts, len(all_layers)))
    for i, result in enumerate(exp2_data):
        for j, layer in enumerate(all_layers):
            layer_str = str(layer)
            matrix[i, j] = result["layer_results"][layer_str]["token_change"]

    min_val = float(np.min(matrix))
    max_val = float(np.max(matrix))
    max_abs = max(abs(min_val), abs(max_val))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(range(len(all_layers)))
    ax.set_xticklabels(all_layers)
    ax.set_yticks(range(n_prompts))
    ax.set_yticklabels([f"P{i+1}" for i in range(n_prompts)], fontsize=8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Prompt")
    ax.set_title("Token Change by Layer (Negative = Suppression)")

    plt.colorbar(im, ax=ax, label="Token Change (neg = suppression)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layer_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_head_effectiveness(head_data):
    head_avg = {}
    for layer_str, layer_results in head_data.items():
        layer = int(layer_str)
        if not layer_results:
            continue
        head_keys = layer_results[0]["head_results"].keys()
        head_count = max(int(k) for k in head_keys) + 1

        for head_idx in range(head_count):
            head_key = str(head_idx)
            changes = []
            for result in layer_results:
                if head_key in result["head_results"]:
                    changes.append(result["head_results"][head_key]["token_change"])
            if changes:
                head_avg[(layer, head_idx)] = float(np.mean(changes))

    sorted_heads = sorted(head_avg.items(), key=lambda x: x[1])[:10]
    labels = [f"L{l}H{h}" for (l, h), _ in sorted_heads]
    values = [avg for _, avg in sorted_heads]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if v < -20 else "#ff7f0e" if v < -10 else "#1f77b4" for v in values]
    ax.barh(range(len(labels)), values, color=colors)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Avg Token Change (neg = suppression)")
    ax.set_title("Top 10 Heads by Suppression (head-level analysis, baseline prompt)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "head_effectiveness.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_quality_comparison(quality_data):
    results = quality_data["detailed_results"]

    baseline_tokens = [r["baseline"]["tokens"] for r in results]
    intervened_tokens = [r["intervened"]["tokens"] for r in results]
    baseline_markers = [r["baseline"]["reasoning_markers"] for r in results]
    intervened_markers = [r["intervened"]["reasoning_markers"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(results))
    width = 0.35

    ax1.bar(x - width / 2, baseline_tokens, width, label="Baseline", color="#1f77b4")
    ax1.bar(x + width / 2, intervened_tokens, width, label="Intervened", color="#ff7f0e")
    ax1.set_ylabel("Tokens")
    ax1.set_xlabel("Problem")
    ax1.set_title("Output Length")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"P{i+1}" for i in range(len(results))], fontsize=8)
    ax1.legend()

    ax2.bar(x - width / 2, baseline_markers, width, label="Baseline", color="#1f77b4")
    ax2.bar(x + width / 2, intervened_markers, width, label="Intervened", color="#ff7f0e")
    ax2.set_ylabel("Reasoning Markers")
    ax2.set_xlabel("Problem")
    ax2.set_title("Reasoning Markers")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P{i+1}" for i in range(len(results))], fontsize=8)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quality_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy(quality_data):
    summary = quality_data["summary"]

    categories = ["Baseline", "Intervened"]
    accuracies = [
        summary["baseline_accuracy"] * 100,
        summary["intervened_accuracy"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(categories, accuracies, color=["#1f77b4", "#ff7f0e"], alpha=0.7, edgecolor="black")

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.0f}%",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Answer Correctness (n=25)")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150

    exp2_data, head_data, quality_data = load_results()

    plot_layer_heatmap(exp2_data)
    print("Saved layer_heatmap.png")

    plot_head_effectiveness(head_data)
    print("Saved head_effectiveness.png")

    plot_quality_comparison(quality_data)
    print("Saved quality_comparison.png")

    plot_accuracy(quality_data)
    print("Saved accuracy.png")

    print(f"\nOK All figures saved to {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("READY FOR WRITEUP")
    print("=" * 70)


if __name__ == "__main__":
    main()
