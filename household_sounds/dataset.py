"""Select a reproducible, expandable five-label subset of FSD50K."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

from household_sounds.model import LABELS

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "metadata"
MANIFEST = ROOT / "data" / "manifest.jsonl"
BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main"

TAGS = {
    "voices": {"Speech", "Conversation", "Whispering", "Yell", "Shout"},
    "machines": {"Mechanical_fan", "Microwave_oven", "Printer", "Power_tool", "Drill"},
    "impacts": {"Slam", "Knock", "Shatter", "Coin_(dropping)", "Chink_and_clink", "Walk_and_footsteps"},
}
NEGATIVE_TAGS = {"Rain", "Wind", "Water", "Dog", "Bird", "Insect"}
META_FILES = {
    "dev.csv": "labels/dev.csv",
    "eval.csv": "labels/eval.csv",
    "dev_info.json": "metadata/dev_clips_info_FSD50K.json",
    "eval_info.json": "metadata/eval_clips_info_FSD50K.json",
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_name(dest.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "household-sounds/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        if tmp.stat().st_size == 0:
            raise RuntimeError(f"Empty response: {url}")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def fetch_metadata() -> None:
    for name, remote in META_FILES.items():
        path = META / name
        if not path.exists():
            print(f"Downloading {remote}", flush=True)
        fetch(f"{BASE}/{remote}", path)


def source_rows() -> list[dict]:
    rows: list[dict] = []
    for partition in ("dev", "eval"):
        info = json.loads((META / f"{partition}_info.json").read_text())
        with (META / f"{partition}.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                clip_id = row["fname"]
                source_tags = set(row["labels"].split(","))
                mapped = {label: int(bool(source_tags & TAGS[label])) for label in LABELS}
                details = info.get(clip_id, {})
                split = "test" if partition == "eval" else ("val" if row["split"] == "val" else "train")
                rows.append({
                    "id": clip_id,
                    "source": "FSD50K",
                    "source_partition": partition,
                    "source_url": f"https://freesound.org/s/{clip_id}/",
                    "audio_url": f"{BASE}/clips/{partition}/{clip_id}.wav",
                    "path": f"data/audio/{partition}/{clip_id}.wav",
                    "split": split,
                    "source_labels": sorted(source_tags),
                    "labels": mapped,
                    "license": details.get("license", ""),
                    "uploader": details.get("uploader", ""),
                    "title": details.get("title", ""),
                })
    return rows


def select(rows: list[dict], per_class: dict[str, int], negatives: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen: list[dict] = []
    for split in ("train", "val", "test"):
        pool = [row for row in rows if row["split"] == split]
        rng.shuffle(pool)
        counts = Counter()
        used: set[str] = set()
        for row in pool:
            positive = [label for label, value in row["labels"].items() if value]
            if not positive or not any(counts[label] < per_class[split] for label in positive):
                continue
            chosen.append(row)
            used.add(row["id"])
            counts.update(positive)
            if all(counts[label] >= per_class[split] for label in LABELS):
                break
        for row in pool:
            if counts["negative"] >= negatives[split]:
                break
            if row["id"] in used or any(row["labels"].values()):
                continue
            if not (set(row["source_labels"]) & NEGATIVE_TAGS):
                continue
            chosen.append(row)
            counts["negative"] += 1
        print(f"{split}: {dict(counts)}", flush=True)
    chosen.sort(key=lambda r: (r["split"], str(r["id"])))
    return chosen


def save_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("select", help="Fetch metadata and create a small manifest")
    choose.add_argument("--train-per-class", type=int, default=200)
    choose.add_argument("--val-per-class", type=int, default=40)
    choose.add_argument("--test-per-class", type=int, default=40)
    choose.add_argument("--train-negatives", type=int, default=200)
    choose.add_argument("--val-negatives", type=int, default=40)
    choose.add_argument("--test-negatives", type=int, default=40)
    choose.add_argument("--seed", type=int, default=7)
    download = sub.add_parser("download", help="Download only manifest-listed audio")
    download.add_argument("--max-gb", type=float, default=5.0)
    sub.add_parser("status", help="Count present and missing clips")
    args = parser.parse_args(argv)
    if args.command == "select":
        fetch_metadata()
        rows = select(
            source_rows(),
            {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class},
            {"train": args.train_negatives, "val": args.val_negatives, "test": args.test_negatives},
            args.seed,
        )
        save_manifest(rows, MANIFEST)
        print(f"Wrote {len(rows)} rows to {MANIFEST}")
    elif args.command == "download":
        rows = load_manifest(MANIFEST)
        budget = int(args.max_gb * 1024**3)
        total = sum((ROOT / row["path"]).stat().st_size for row in rows
                    if (ROOT / row["path"]).is_file())
        for index, row in enumerate(rows, 1):
            path = ROOT / row["path"]
            if not path.exists():
                if total >= budget:
                    print(f"Audio budget of {args.max_gb} GiB reached with {index - 1} clips checked")
                    break
                fetch(row["audio_url"], path)
                total += path.stat().st_size
                if total > budget:
                    print(f"Audio budget of {args.max_gb} GiB exceeded by the last clip; stopping after {index} clips")
                    break
            if index % 100 == 0:
                print(f"Checked {index}/{len(rows)} clips", flush=True)
    else:
        rows = load_manifest(MANIFEST)
        present = [row for row in rows if (ROOT / row["path"]).is_file()]
        counts = Counter(row["split"] for row in present)
        print(json.dumps({"listed": len(rows), "present": len(present), "by_split": counts}, indent=2))


if __name__ == "__main__":
    main()
