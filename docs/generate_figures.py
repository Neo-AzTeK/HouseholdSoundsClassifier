"""Regenerate this note's figures from the selected public training clips.

Run from the project root: python3 docs/generate_figures.py
The script uses .venv for the exact model preprocessing and the system Python
for Matplotlib, which is installed outside the model environment here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
DATA = ASSETS / "spectrogram_examples_data.npz"
EXAMPLES = (
    ("104017", "Mechanical fan"),
    ("104001", "Wooden knocks"),
    ("119913", "Footsteps"),
)


def extract() -> None:
    """Run in the model venv so figure features use the production code."""
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from household_sounds.model import (  # noqa: PLC0415
        candidate_starts, fixed_window, read_mono, simulate_thingy_stream, spectrogram,
    )

    rows = {row["id"]: row for row in (
        json.loads(line) for line in (ROOT / "data/manifest.jsonl").read_text().splitlines()
    )}
    features = {}
    for clip_id, _ in EXAMPLES:
        row = rows[clip_id]
        if row["split"] != "train":
            raise ValueError(f"Expected a training clip: {clip_id}")
        audio, sample_rate = read_mono(str(ROOT / row["path"]))
        stream = simulate_thingy_stream(audio, sample_rate)
        window = fixed_window(stream, candidate_starts(stream)[0])
        features[clip_id] = spectrogram(window)[0]
    ASSETS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, **features)


def render() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    with np.load(DATA) as arrays:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
        for ax, (clip_id, title) in zip(axes, EXAMPLES):
            feature = arrays[clip_id]
            shown = ax.imshow(
                feature, origin="lower", aspect="auto", cmap="magma",
                extent=(0, 1, 0, 4031.25 / 2000), vmin=0,
                vmax=float(np.quantile(feature, 0.99)),
            )
            ax.set_title(f"{title} · FSD50K {clip_id}", loc="left")
            ax.set_ylabel("Frequency (kHz)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2.016)
            fig.colorbar(shown, ax=ax, label="Log magnitude", shrink=0.83)
        axes[-1].set_xlabel("Time within selected one-second window (s)")
        fig.suptitle("Actual model input: loudest window of three public training clips", fontsize=14)
        fig.savefig(ASSETS / "spectrogram_examples.png", dpi=170)
        plt.close(fig)

    stages = (
        ("Audio Window", "4,031 samples\n~1.0 s @ 4,031 Hz", "#dceaf4"),
        ("STFT + Log", "1 × 129 × 48\nSpectrogram", "#dceaf4"),
        ("Conv Blocks", "16 → 32 → 64 → 96\nConv → BN → ReLU → Pool", "#d9eddf"),
        ("Dual Pooling", "AvgPool (96) + MaxPool (96)\n→ 192 features", "#f6e7cb"),
        ("Linear Head", "Dense 192 → 3\nLogits", "#f6e7cb"),
        ("Sigmoids", "3 Independent Scores:\nVoices, Machines, Impacts", "#fad2d2"),
    )
    fig, ax = plt.subplots(figsize=(15, 3), layout="constrained")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for index, (title, detail, color) in enumerate(stages):
        x = 0.1 + index * 2.5
        ax.add_patch(FancyBboxPatch(
            (x, 0.55), 2.15, 1.8, boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.3, edgecolor="#31516b", facecolor=color,
        ))
        ax.text(x + 1.075, 1.78, title, ha="center", va="center", weight="bold", fontsize=11)
        ax.text(x + 1.075, 1.12, detail, ha="center", va="center", fontsize=8.5)
        if index < len(stages) - 1:
            ax.annotate("", xy=(x + 2.48, 1.45), xytext=(x + 2.22, 1.45),
                        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#31516b"})
    fig.savefig(ASSETS / "model_pipeline.png", dpi=170)
    plt.close(fig)

    # Detailed Model Architecture Diagram (tensor shapes, layers, dual pooling)
    fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(7, 8.55, "SoundCNN Neural Network Architecture & Tensor Transformations", 
            ha="center", va="center", weight="bold", fontsize=14, color="#1b2a47")
    ax.text(7, 8.2, "Input [1, 129, 48] · 4 Conv Blocks · Dual Statistical Pooling (Mean+Max) · 79,683 Parameters · 313 KB ONNX", 
            ha="center", va="center", fontsize=9.5, color="#556b82")

    blocks = [
        ("Input Spectrogram", "1 × 129 × 48\nNormalized Log STFT\n129 freq × 48 time", "#eaf2f8", 0.6, 5.0, 2.0, 2.3),
        ("Block 1 (192 params)", "Conv2d(1→16, 3×3)\nBatchNorm2d(16)\nReLU + MaxPool2d(2)\n→ 16 × 64 × 24", "#d4efdf", 3.1, 5.0, 2.2, 2.3),
        ("Block 2 (4,704 params)", "Conv2d(16→32, 3×3)\nBatchNorm2d(32)\nReLU + MaxPool2d(2)\n→ 32 × 32 × 12", "#d4efdf", 5.8, 5.0, 2.2, 2.3),
        ("Block 3 (18,624 params)", "Conv2d(32→64, 3×3)\nBatchNorm2d(64)\nReLU + MaxPool2d(2)\n→ 64 × 16 × 6", "#d4efdf", 8.5, 5.0, 2.2, 2.3),
        ("Block 4 (55,584 params)", "Conv2d(64→96, 3×3)\nBatchNorm2d(96)\nReLU + MaxPool2d(2)\n→ 96 × 8 × 3", "#d4efdf", 11.2, 5.0, 2.2, 2.3),
    ]

    for title, detail, col, x, y, w, h in blocks:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1",
                                    linewidth=1.4, edgecolor="#2471a3" if "Input" in title else "#1e8449", facecolor=col))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", weight="bold", fontsize=10, color="#1b2a47")
        ax.text(x + w / 2, y + (h - 0.4) / 2, detail, ha="center", va="center", fontsize=8.5, color="#2c3e50")

    # Sequential arrows between blocks
    for i in range(len(blocks) - 1):
        x1 = blocks[i][3] + blocks[i][5]
        x2 = blocks[i + 1][3]
        y_mid = 5.0 + 2.3 / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#1b2a47"})

    # Dual Pooling Branch (splitting from Block 4 at x=12.3, y=5.0 to lower area)
    ax.annotate("", xy=(10.5, 3.8), xytext=(12.3, 5.0),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#8e44ad", "connectionstyle": "arc3,rad=0.2"})
    ax.annotate("", xy=(10.5, 2.2), xytext=(12.3, 5.0),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#8e44ad", "connectionstyle": "arc3,rad=-0.2"})

    # Avg Pool & Max Pool boxes
    ax.add_patch(FancyBboxPatch((8.0, 3.2), 2.4, 1.2, boxstyle="round,pad=0.08,rounding_size=0.1",
                                linewidth=1.4, edgecolor="#8e44ad", facecolor="#f4ecf7"))
    ax.text(9.2, 4.05, "AdaptiveAvgPool2d(1)", ha="center", va="center", weight="bold", fontsize=9.5, color="#4a235a")
    ax.text(9.2, 3.55, "Global Average (96)\nSustained drones & formants", ha="center", va="center", fontsize=8, color="#512e5f")

    ax.add_patch(FancyBboxPatch((8.0, 1.6), 2.4, 1.2, boxstyle="round,pad=0.08,rounding_size=0.1",
                                linewidth=1.4, edgecolor="#8e44ad", facecolor="#f4ecf7"))
    ax.text(9.2, 2.45, "AdaptiveMaxPool2d(1)", ha="center", va="center", weight="bold", fontsize=9.5, color="#4a235a")
    ax.text(9.2, 1.95, "Global Max (96)\nTransient impulse peaks", ha="center", va="center", fontsize=8, color="#512e5f")

    # Concat Box
    ax.annotate("", xy=(5.8, 2.9), xytext=(8.0, 3.8),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#8e44ad", "connectionstyle": "arc3,rad=0.1"})
    ax.annotate("", xy=(5.8, 2.5), xytext=(8.0, 2.2),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#8e44ad", "connectionstyle": "arc3,rad=-0.1"})

    ax.add_patch(FancyBboxPatch((3.8, 2.0), 1.9, 1.6, boxstyle="round,pad=0.08,rounding_size=0.1",
                                linewidth=1.4, edgecolor="#d35400", facecolor="#fdebd0"))
    ax.text(4.75, 3.25, "torch.cat(dim=1)", ha="center", va="center", weight="bold", fontsize=9.5, color="#7e5109")
    ax.text(4.75, 2.65, "Statistical Pooling\nVector:\n192 Features", ha="center", va="center", fontsize=8.5, color="#6e2c00")

    # Linear Head & Sigmoid Box
    ax.annotate("", xy=(2.4, 2.7), xytext=(3.8, 2.7),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#c0392b"})

    ax.add_patch(FancyBboxPatch((0.5, 1.8), 1.8, 1.9, boxstyle="round,pad=0.08,rounding_size=0.1",
                                linewidth=1.4, edgecolor="#c0392b", facecolor="#fadbd8"))
    ax.text(1.4, 3.35, "Linear(192→3)", ha="center", va="center", weight="bold", fontsize=9.5, color="#78281f")
    ax.text(1.4, 2.75, "3 Logits\n+\nIndependent\nSigmoids", ha="center", va="center", fontsize=8.5, color="#922b21")

    # Final Output Pills
    outputs = [
        ("Voices (0.62 Cutoff)", 4.0, "#ebdef0", "#6c3483"),
        ("Machines (0.68 Cutoff)", 2.8, "#d1f2eb", "#117864"),
        ("Impacts (0.51 Cutoff)", 1.6, "#fcf3cf", "#b7950b"),
    ]
    for lbl, y_pos, bg, fg in outputs:
        ax.annotate("", xy=(11.5, y_pos), xytext=(2.3, 2.7),
                    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": fg, "connectionstyle": "arc3,rad=0.0"})
        ax.add_patch(FancyBboxPatch((11.5, y_pos - 0.35), 2.0, 0.7, boxstyle="round,pad=0.06,rounding_size=0.1",
                                    linewidth=1.2, edgecolor=fg, facecolor=bg))
        ax.text(12.5, y_pos, lbl, ha="center", va="center", weight="bold", fontsize=8.5, color=fg)

    fig.savefig(ASSETS / "model_architecture.png", dpi=180)
    plt.close(fig)

    # The numbers here illustrate the operation; the trained weights differ.
    patch = np.array([[0, 0, 1], [0, 2, 1], [0, 1, 0]])
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), layout="constrained")
    for ax, grid, title, cmap, low, high in (
        (axes[0], patch, "3×3 input patch", "Blues", 0, 2),
        (axes[1], kernel, "One learned 3×3 filter", "RdBu_r", -1, 1),
    ):
        ax.imshow(grid, cmap=cmap, vmin=low, vmax=high)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(grid[row, col]), ha="center", va="center", fontsize=14)
    axes[2].axis("off")
    axes[2].text(0.5, 0.68, "Multiply matching cells\nand add them", ha="center", va="center", fontsize=11)
    axes[2].text(0.5, 0.34, "1 + 1 + 0 = 2", ha="center", va="center", fontsize=16, weight="bold")
    axes[3].axis("off")
    axes[3].text(0.5, 0.65, "Repeat at every position\nwith 16 different filters", ha="center", va="center", fontsize=11)
    axes[3].text(0.5, 0.32, "→ 16 feature maps", ha="center", va="center", fontsize=14, weight="bold")
    fig.suptitle("First convolution: illustrative values, not checkpoint weights", fontsize=13)
    fig.savefig(ASSETS / "first_convolution.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--extract":
        extract()
    elif len(sys.argv) == 1:
        subprocess.run([str(ROOT / ".venv/bin/python"), str(__file__), "--extract"], check=True)
        try:
            render()
        finally:
            DATA.unlink(missing_ok=True)
        print(f"Wrote figures to {ASSETS}")
    else:
        raise SystemExit("Usage: python3 docs/generate_figures.py")
