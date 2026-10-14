"""Train and evaluate the household sound model from data/manifest.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audio_model import (
    LABELS, PREPROCESSING, SILENCE_RMS, WINDOW_SAMPLES, SoundCNN, candidate_starts, fixed_window, read_mono,
    simulate_thingy_stream, spectrogram,
)

ROOT = Path(__file__).resolve().parent


class Clips(Dataset):
    def __init__(self, rows: list[dict], train: bool, mix_probability: float = 0.0,
                 silence_probability: float = 0.0) -> None:
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

    def __len__(self) -> int:
        return len(self.rows)

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
        row = self.rows[index]
        if self.train and random.random() < self.silence_probability:
            # Quiet inputs otherwise sit outside the public clips' loud-window training distribution.
            window = (np.random.normal(0, 2e-4, WINDOW_SAMPLES).astype(np.float32)
                      if random.random() < 0.5 else np.zeros(WINDOW_SAMPLES, dtype=np.float32))
            target = np.zeros(len(LABELS), dtype=np.float32)
        else:
            window = self.window(index)
            target = np.array([row["labels"][label] for label in LABELS], dtype=np.float32)
            possible = [label for label in LABELS if not row["labels"][label] and self.indices_by_label[label]]
            if self.train and possible and random.random() < self.mix_probability:
                # Prefer speech/music mixtures when either source contains only one of them.
                if row["labels"]["voices"] and not row["labels"]["music"]:
                    partner_label = "music"
                elif row["labels"]["music"] and not row["labels"]["voices"]:
                    partner_label = "voices"
                else:
                    partner_label = random.choice(possible)
                partner = random.choice(self.indices_by_label[partner_label])
                other = self.window(partner)
                # Balance source loudness before summing, so both remain audible.
                a = window / max(float(np.sqrt(np.mean(window ** 2))), 0.002)
                b = other / max(float(np.sqrt(np.mean(other ** 2))), 0.002)
                window = np.clip((a + b) * (0.03 / np.sqrt(2)), -1.0, 1.0)
                target = np.maximum(target, [self.rows[partner]["labels"][label] for label in LABELS])
        if self.train:
            window = np.clip(window * random.uniform(0.6, 1.4), -1.0, 1.0)
        x = torch.from_numpy(spectrogram(window))
        y = torch.tensor(target, dtype=torch.float32)
        return x, y


def metrics(probs: np.ndarray, truth: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = probs >= thresholds[None, :]
    actual = truth > 0.5
    tp = np.sum(pred & actual, axis=0)
    fp = np.sum(pred & ~actual, axis=0)
    fn = np.sum(~pred & actual, axis=0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        label: {"precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4), "support": int(actual[:, i].sum())}
        for i, label in enumerate(LABELS)
    }


def best_thresholds(probs: np.ndarray, truth: np.ndarray) -> np.ndarray:
    found = []
    for index in range(len(LABELS)):
        best_t, best_f1 = 0.5, -1.0
        for threshold in np.arange(0.1, 0.91, 0.05):
            pred = probs[:, index] >= threshold
            actual = truth[:, index] > 0.5
            tp = np.sum(pred & actual)
            fp = np.sum(pred & ~actual)
            fn = np.sum(~pred & actual)
            score = float(2 * tp / max(2 * tp + fp + fn, 1))
            if score > best_f1:
                best_t, best_f1 = float(threshold), score
        found.append(best_t)
    return np.array(found, dtype=np.float32)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.jsonl")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mix-probability", type=float, default=0.3,
                        help="Chance of mixing a second, different-label public clip into a training window")
    parser.add_argument("--silence-probability", type=float, default=0.0,
                        help="Chance of replacing a training window with silence or faint noise")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "household_sound.pt")
    args = parser.parse_args()
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
            f"first missing: {missing_files[0]}. Run: python dataset.py status"
        )
    groups = {split: [r for r in all_rows if r["split"] == split]
              for split in ("train", "val", "test")}
    for split, rows in groups.items():
        print(f"{split}: {len(rows)} present clips", flush=True)
    if not groups["train"] or not groups["val"] or not groups["test"]:
        raise SystemExit("Need downloaded train, val, and test clips. Run: python dataset.py status")
    for split in ("train", "val", "test"):
        missing = [label for label in LABELS if not any(r["labels"][label] for r in groups[split])]
        if missing:
            raise SystemExit(f"{split} has no positive examples for: {', '.join(missing)}")

    train_data = Clips(groups["train"], True, args.mix_probability, args.silence_probability)
    loaders = {split: DataLoader(Clips(groups[split], False), batch_size=args.batch_size, shuffle=False)
               for split in ("val", "test")}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    counts = np.array([sum(r["labels"][label] for r in groups["train"]) for label in LABELS])
    negatives = len(groups["train"]) - counts
    pos_weight = torch.tensor(np.clip(negatives / np.maximum(counts, 1), 1, 15),
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = SoundCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
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
            }, args.output)
        else:
            stale += 1
            if stale >= 6:
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
        "limitations": "FSD50K has clip-level labels; these are proxy window metrics, not room performance.",
    }, indent=2))
    print(f"Saved {args.output} and {metrics_path}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
