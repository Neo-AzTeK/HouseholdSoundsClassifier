"""Build clean multi-dataset fusion manifest combining ESC-50 and verified FSD50K."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ESC50_META = ROOT / "data" / "esc50" / "meta" / "esc50.csv"
ESC50_AUDIO = ROOT / "data" / "esc50" / "audio"
FSD_MANIFEST = ROOT / "data" / "manifest.jsonl"
OUT_MANIFEST = ROOT / "data" / "manifest_fusion.jsonl"


def build_fusion_manifest() -> Path:
    random.seed(42)
    rows: list[dict] = []

    # 1. Process ESC-50
    if not ESC50_META.is_file():
        raise FileNotFoundError(f"ESC-50 metadata not found at {ESC50_META}")

    with ESC50_META.open(newline="", encoding="utf-8") as f:
        esc_reader = list(csv.DictReader(f))

    for item in esc_reader:
        cat = item["category"]
        filename = item["filename"]
        fold = int(item["fold"])

        # Folds 1, 2, 3 -> train; Fold 4 -> val; Fold 5 -> test
        split = "test" if fold == 5 else ("val" if fold == 4 else "train")

        labels = {
            "voices": 0,
            "machines": 0,
            "impacts": 0,
        }

        # Clean category mappings from ESC-50
        is_used = True
        if cat in ("footsteps", "door_wood_knock", "glass_breaking", "can_opening", "door_wood_creaks"):
            labels["impacts"] = 1
        elif cat in ("washing_machine", "vacuum_cleaner"):
            labels["machines"] = 1
        elif cat in ("crying_baby", "laughing", "coughing", "sneezing"):
            labels["voices"] = 1
        elif cat in ("rain", "wind", "water_drops", "sea_waves", "crickets"):
            # Clean all-zero ambient negative
            pass
        else:
            is_used = False

        if is_used:
            rows.append({
                "id": f"esc50_{item['src_file']}_{item['take']}",
                "source": "ESC-50",
                "source_partition": "esc50",
                "path": f"data/esc50/audio/{filename}",
                "split": split,
                "labels": labels,
                "source_labels": [cat],
                "title": f"ESC50 {cat} {filename}",
            })

    print(f"Added {len(rows)} clean clips from ESC-50")

    # 2. Process clean, verified subset from FSD50K
    with FSD_MANIFEST.open(encoding="utf-8") as f:
        fsd_rows = [json.loads(line) for line in f]

    # Strict exclusionary tags to filter out all ambiguous/conflicting noise
    bad_tags = {
        "Drum", "Cymbal", "Crash_cymbal", "Percussion", "Trash_can",
        "Vehicle", "Motor_vehicle_(road)", "Car", "Engine", "Engine_idling",
        "Singing", "Female_singing", "Male_singing",
        "Screaming", "Bicycle", "Bicycle_bell",
    }

    instruments = {
        "Piano", "Acoustic_guitar", "Guitar", "Electric_guitar",
        "Keyboard_(musical)", "Violin_and_fiddle", "Accordion",
        "Harmonica", "Flute", "Plucked_string_instrument", "Bowed_string_instrument"
    }

    fsd_added = 0
    for r in fsd_rows:
        src = set(r.get("source_labels", []))
        if src & bad_tags:
            continue

        is_voice = bool(src & {"Speech", "Conversation", "Whispering", "Yell", "Shout"})
        is_machine = bool(src & {"Mechanical_fan", "Microwave_oven", "Printer"})
        is_impact = bool(src & {"Coin_(dropping)", "Chink_and_clink", "Shatter", "Walk_and_footsteps"}) and not (src & {"Door", "Slam", "Water", "Liquid"})
        is_music = bool(src & instruments) and ("Music" in src or "Musical_instrument" in src)

        # 3 target classes: voices, machines, impacts
        matches = [is_voice, is_machine, is_impact]
        if sum(matches) == 1 and not is_music:
            labels = {
                "voices": int(is_voice),
                "machines": int(is_machine),
                "impacts": int(is_impact),
            }
            rows.append({
                "id": f"fsd_{r['id']}",
                "source": "FSD50K_clean",
                "source_partition": r["source_partition"],
                "path": r["path"],
                "split": r["split"],
                "labels": labels,
                "source_labels": list(src),
                "title": r.get("title", ""),
            })
            fsd_added += 1
        elif is_music and sum(matches) == 0:
            # Music without voice/machine/impact -> negative ambient clutter
            labels = {
                "voices": 0,
                "machines": 0,
                "impacts": 0,
            }
            rows.append({
                "id": f"fsd_{r['id']}",
                "source": "FSD50K_clean",
                "source_partition": r["source_partition"],
                "path": r["path"],
                "split": r["split"],
                "labels": labels,
                "source_labels": list(src),
                "title": r.get("title", ""),
            })
            fsd_added += 1

    print(f"Added {fsd_added} verified clean clips from FSD50K")
    print(f"Total fusion dataset size: {len(rows)} clips")

    # Stats per class
    counts = Counter()
    for r in rows:
        for k, v in r["labels"].items():
            if v:
                counts[k] += 1
        if sum(r["labels"].values()) == 0:
            counts["all_zero_ambient"] += 1

    print("\nClass distribution in Fusion Dataset:")
    for k, count in sorted(counts.items()):
        print(f"  {k:18s}: {count:4d}")

    # Split breakdown
    split_counts = Counter(r["split"] for r in rows)
    print("\nSplit breakdown:")
    for split, count in split_counts.items():
        print(f"  {split:10s}: {count:4d}")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nManifest successfully written to {OUT_MANIFEST}")
    return OUT_MANIFEST


if __name__ == "__main__":
    build_fusion_manifest()
