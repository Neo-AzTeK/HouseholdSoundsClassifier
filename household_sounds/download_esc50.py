"""Download and extract the ESC-50 environmental sound dataset."""

from __future__ import annotations

import csv
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ESC50_DIR = ROOT / "data" / "esc50"
ESC50_ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def download_esc50(dest_dir: Path = ESC50_DIR) -> Path:
    """Download and extract ESC-50 repository."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = dest_dir / "meta" / "esc50.csv"
    audio_dir = dest_dir / "audio"

    if meta_csv.is_file() and audio_dir.is_dir() and len(list(audio_dir.glob("*.wav"))) >= 2000:
        print(f"ESC-50 already fully extracted at {dest_dir}")
        return dest_dir

    print(f"Downloading ESC-50 from {ESC50_ZIP_URL}...")
    zip_path = dest_dir / "esc50.zip"
    request = urllib.request.Request(ESC50_ZIP_URL, headers={"User-Agent": "household-sounds/0.2"})

    with urllib.request.urlopen(request, timeout=300) as response, zip_path.open("wb") as out:
        shutil.copyfileobj(response, out)

    print("Extracting ESC-50 archive...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            # Strip the leading 'ESC-50-master/' folder
            parts = Path(member).parts
            if len(parts) > 1 and parts[1] in ("audio", "meta"):
                target = dest_dir / Path(*parts[1:])
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

    zip_path.unlink(missing_ok=True)
    print(f"ESC-50 successfully extracted to {dest_dir}")
    return dest_dir


if __name__ == "__main__":
    download_esc50()
