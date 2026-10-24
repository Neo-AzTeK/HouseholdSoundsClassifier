"""Train and evaluate the household sound model from data/manifest.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy import signal
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from household_sounds.loss import AsymmetricLoss
from household_sounds.model import (
    LABELS,
    PREPROCESSING,
    SILENCE_RMS,
    STREAM_RATE,
    WINDOW_SAMPLES,
    SoundCNN,
    candidate_starts,
    fixed_window,
    read_mono,
    simulate_thingy_stream,
    spectrogram,
)

ROOT = Path(__file__).resolve().parent.parent

# 80 Hz 1st-order high-pass filter coefficients at 4,031.25 Hz stream rate
HP_B = np.array([0.9412423, -0.9412423], dtype=np.float32)
HP_A = np.array([1.0, -0.8824846], dtype=np.float32)


def apply_spec_augment(
    spec: torch.Tensor,
    max_freq_mask: int = 16,
    max_time_mask: int = 8,
    freq_masks: int = 2,
    time_masks: int = 2,
) -> torch.Tensor:
    """SpecAugment: Randomly masks frequency and time bands.
    spec shape: [channels, freq_bins, time_frames] e.g. [1, 129, 48].
    """
    spec = spec.clone()
    _, f_bins, t_frames = spec.shape
    for _ in range(freq_masks):
        f = random.randint(1, max_freq_mask)
        f0 = random.randint(0, max(0, f_bins - f))
        spec[:, f0:f0 + f, :] = 0.0
    for _ in range(time_masks):
        t = random.randint(1, max_time_mask)
        t0 = random.randint(0, max(0, t_frames - t))
        spec[:, :, t0:t0 + t] = 0.0
    return spec


class Clips(Dataset):
    def __init__(
        self,
        rows: list[dict],
        train: bool,
        mix_probability: float = 0.0,
        silence_probability: float = 0.0,
    ) -> None:
        self.rows = rows
        self.train = train
        self.mix_probability = mix_probability
        self.silence_probability = silence_probability
        self.indices_by_label = {
            label: [i for i, row in enumerate(rows) if row["labels"][label]]
            for label in LABELS
        }
        self.cache = ROOT / "data" / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        if self.train:
            # Oversample machines and ambient/music negatives to balance with voices and impacts
            expanded: list[int] = []
            for i, r in enumerate(rows):
                expanded.append(i)
                if r["labels"].get("machines", 0):
                    # Add 1 additional copy of machine clips
                    expanded.append(i)
                elif sum(r["labels"].values()) == 0:
                    # Add 1 additional copy of ambient noise / music clips
                    expanded.append(i)
            self.sample_indices = expanded
        else:
            self.sample_indices = list(range(len(rows)))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def window(self, index: int) -> np.ndarray:
        row = self.rows[index]
        cache_file = self.cache / f"{row['source_partition']}_{row['id']}.npy"
        if cache_file.exists():
            audio = np.load(cache_file)
        else:
            audio, sample_rate = read_mono(str(ROOT / row["path"]))
            audio = simulate_thingy_stream(audio, sample_rate)
            np.save(cache_file, audio)
        starts = candidate_starts(audio)
        start = random.choice(starts) if self.train else starts[0]
        return fixed_window(audio, start)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_idx = self.sample_indices[index]
        row = self.rows[row_idx]
        if self.train and random.random() < self.silence_probability:
            # Multi-level ambient background noise:
            # 50% faint room floor (1e-4 to 2e-3)
            # 50% loud continuous room HVAC, fan whoosh, and 50/100/120Hz hum (5e-3 to 3.5e-2)
            t = np.arange(WINDOW_SAMPLES) / STREAM_RATE
            hum = 0.5 * np.sin(2 * np.pi * 50.0 * t) + 0.3 * np.sin(2 * np.pi * 100.0 * t) + 0.2 * np.sin(2 * np.pi * 120.0 * t)
            white = np.random.normal(0, 1.0, WINDOW_SAMPLES)
            pink = signal.lfilter([0.15], [1.0, -0.85], white)
            if random.random() < 0.5:
                rms = random.uniform(1e-4, 2e-3)
            else:
                # Loud ambient room noise / HVAC drone labeled strictly as all-zero negative
                rms = random.uniform(5e-3, 3.5e-2)
            window = (rms * (0.35 * hum + 0.65 * pink)).astype(np.float32)
            window = signal.lfilter(HP_B, HP_A, window).astype(np.float32)
            target = np.zeros(len(LABELS), dtype=np.float32)
            return torch.from_numpy(spectrogram(window)), torch.from_numpy(target)

        window = self.window(row_idx)
        labels = dict(row["labels"])

        if self.train and random.random() < self.mix_probability:
            # Calibrated multi-source superposition:
            # Prefer pairing with different active class
            other_labels = [label for label in LABELS if not labels[label] and self.indices_by_label[label]]
            if other_labels:
                chosen_label = random.choice(other_labels)
                other_index = random.choice(self.indices_by_label[chosen_label])
                other_window = self.window(other_index)

                # Loudness-calibrated mixing so neither sound drowns out the other
                rms1 = max(float(np.sqrt(np.mean(window * window))), 1e-4)
                rms2 = max(float(np.sqrt(np.mean(other_window * other_window))), 1e-4)
                scale2 = (rms1 / rms2) * random.uniform(0.6, 1.4)
                window = window + other_window * scale2
                for label, value in self.rows[other_index]["labels"].items():
                    if value:
                        labels[label] = 1

        if self.train:
            # Gain augmentation: randomly scale between 0.2x and 2.5x (-14 dB to +8 dB)
            gain = random.uniform(0.2, 2.5)
            window = window * gain

            # If clip is an ambient/music negative (all labels 0), ensure it has realistic volume
            # so the model actively learns to reject loud music/background noise as machines
            if sum(labels.values()) == 0:
                current_rms = max(float(np.sqrt(np.mean(window * window))), 1e-5)
                target_neg_rms = random.uniform(0.008, 0.04)
                window = window * (target_neg_rms / current_rms)

            # Ambient room noise injection (30% chance): faint room hum + pink noise
            if random.random() < 0.3:
                t = np.arange(WINDOW_SAMPLES) / STREAM_RATE
                hum = 0.5 * np.sin(2 * np.pi * 50.0 * t) + 0.3 * np.sin(2 * np.pi * 100.0 * t)
                noise = signal.lfilter([0.15], [1.0, -0.85], np.random.normal(0, 1.0, WINDOW_SAMPLES))
                noise_rms = random.uniform(2e-4, 1.5e-3)
                window = window + (noise_rms * (0.3 * hum + 0.7 * noise)).astype(np.float32)

            window = np.clip(window, -1.0, 1.0).astype(np.float32)

        window = signal.lfilter(HP_B, HP_A, window).astype(np.float32)
        spec = torch.from_numpy(spectrogram(window))
        if self.train and random.random() < 0.5:
            spec = apply_spec_augment(spec)
        target = np.array([labels[label] for label in LABELS], dtype=np.float32)
        return spec, torch.from_numpy(target)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, ground_truth = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        ground_truth.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(ground_truth)


def best_thresholds(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    steps = np.linspace(0.1, 0.9, 81)
    thresholds = []
    for index in range(len(LABELS)):
        p = probabilities[:, index]
        t = truth[:, index]
        best_f1, best_t = -1.0, 0.5
        for threshold in steps:
            pred = p >= threshold
            tp = int(np.logical_and(pred, t).sum())
            fp = int(np.logical_and(pred, np.logical_not(t)).sum())
            fn = int(np.logical_and(np.logical_not(pred), t).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(threshold)
        thresholds.append(best_t)
    return np.array(thresholds, dtype=np.float32)


def metrics(probabilities: np.ndarray, truth: np.ndarray, thresholds: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for index, label in enumerate(LABELS):
        pred = probabilities[:, index] >= thresholds[index]
        t = truth[:, index]
        tp = int(np.logical_and(pred, t).sum())
        fp = int(np.logical_and(pred, np.logical_not(t)).sum())
        fn = int(np.logical_and(np.logical_not(pred), t).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[label] = {
            "threshold": round(float(thresholds[index]), 3),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(t.sum()),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_manifest = (
        ROOT / "data" / "manifest_combined.jsonl"
        if (ROOT / "data" / "manifest_combined.jsonl").is_file()
        else ROOT / "data" / "manifest.jsonl"
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mix-probability",
        type=float,
        default=0.55,
        help="Chance of mixing a second clip into a training window",
    )
    parser.add_argument(
        "--silence-probability",
        type=float,
        default=0.15,
        help="Chance of replacing a training window with silence or faint noise",
    )
    parser.add_argument(
        "--loss",
        choices=["asl", "bce"],
        default="asl",
        help="Loss function: 'asl' (Asymmetric Loss) or 'bce' (Binary Cross Entropy)",
    )
    parser.add_argument(
        "--asl-gamma-neg",
        type=float,
        default=1.8,
        help="ASL gamma_neg focusing parameter",
    )
    parser.add_argument(
        "--asl-clip",
        type=float,
        default=0.0,
        help="ASL margin clip for negative probabilities",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "household_sound.pt")
    args = parser.parse_args(argv)

    if not 0.0 <= args.mix_probability <= 1.0:
        parser.error("--mix-probability must be between 0 and 1")
    if not 0.0 <= args.silence_probability <= 1.0:
        parser.error("--silence-probability must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}", flush=True)

    content = args.manifest.read_bytes()
    all_rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    missing_files = [r["path"] for r in all_rows if not (ROOT / r["path"]).is_file()]
    if missing_files:
        raise SystemExit(
            f"Missing {len(missing_files)} of {len(all_rows)} manifest clips; "
            f"first missing: {missing_files[0]}. Run: uv run sound-dataset status"
        )
    groups = {
        split: [r for r in all_rows if r["split"] == split]
        for split in ("train", "val", "test")
    }
    for split, rows in groups.items():
        print(f"{split}: {len(rows)} present clips", flush=True)
    if not groups["train"] or not groups["val"] or not groups["test"]:
        raise SystemExit("Need downloaded train, val, and test clips.")

    train_data = Clips(groups["train"], True, args.mix_probability, args.silence_probability)
    loaders = {
        split: DataLoader(Clips(groups[split], False), batch_size=args.batch_size, shuffle=False)
        for split in ("val", "test")
    }
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    if args.loss == "asl":
        loss_fn = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=1.0, clip=args.asl_clip)
        print(f"Using AsymmetricLoss (gamma_neg={args.asl_gamma_neg}, clip={args.asl_clip})", flush=True)
    else:
        counts = np.array([sum(r["labels"][label] for r in groups["train"]) for label in LABELS])
        negatives = len(groups["train"]) - counts
        pos_weight = torch.tensor(
            np.clip(negatives / np.maximum(counts, 1), 1, 15),
            dtype=torch.float32,
            device=device,
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using BCEWithLogitsLoss with pos_weight", flush=True)
    model = SoundCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-5)
    best_score, stale = -1.0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        probs, truth = predict_loader(model, loaders["val"], device)
        thresholds = best_thresholds(probs, truth)
        report = metrics(probs, truth, thresholds)
        score = float(np.mean([item["f1"] for item in report.values()]))
        print(f"epoch {epoch:02d}: loss={np.mean(losses):.4f} val_macro_f1={score:.4f}", flush=True)
        if score > best_score:
            best_score, stale = score, 0
            torch.save({
                "state_dict": model.state_dict(),
                "labels": LABELS,
                "preprocessing": PREPROCESSING,
                "thresholds": thresholds.tolist(),
                "manifest_sha256": hashlib.sha256(content).hexdigest(),
                "validation_metrics": report,
                "epoch": epoch,
                "mix_probability": args.mix_probability,
                "silence_probability": args.silence_probability,
                "silence_rms": SILENCE_RMS,
                "loss": args.loss,
            }, args.output)
        else:
            stale += 1
            if stale >= 10:
                print("Early stopping", flush=True)
                break

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    probs, truth = predict_loader(model, loaders["test"], device)
    report = metrics(probs, truth, np.array(checkpoint["thresholds"]))
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "checkpoint": str(args.output),
        "manifest_sha256": checkpoint["manifest_sha256"],
        "epoch": checkpoint["epoch"],
        "validation": checkpoint["validation_metrics"],
        "test": report,
        "inference_silence_rms": SILENCE_RMS,
    }, indent=2))
    print(f"Saved {args.output} and {metrics_path}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
