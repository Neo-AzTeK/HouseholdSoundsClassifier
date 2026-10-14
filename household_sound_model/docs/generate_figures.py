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
    from audio_model import (  # noqa: PLC0415
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
        ("Audio", "4,031 samples\n~1 second", "#dceaf4"),
        ("STFT + log", "1 × 129 × 48\nfeature map", "#dceaf4"),
        ("Block 1", "16 × 64 × 24\nConv → BN → ReLU → Pool", "#d9eddf"),
        ("Blocks 2–4", "96 × 8 × 3\nat block 4 output", "#d9eddf"),
        ("Average + head", "96 summaries\n→ 5 logits", "#f6e7cb"),
        ("Sigmoid", "5 independent\nscores", "#f6e7cb"),
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
