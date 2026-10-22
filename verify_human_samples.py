"""
Verify human_samples/ for HAS-Forensics Step 4.

Checks file count, manifest consistency, and basic image validity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
HUMAN_DIR = PROJECT_ROOT / "human_samples"
AI_DIR = PROJECT_ROOT / "ai_batch"


def main() -> int:
    manifest_path = HUMAN_DIR / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    png_files = sorted(HUMAN_DIR.glob("human_*.png"))

    print("Step 4 verification — human_samples/")
    print(f"  PNG count:           {len(png_files)}")
    print(f"  Manifest entries:    {len(manifest)}")
    print(f"  Match:               {len(png_files) == len(manifest)}")

    ai_count = len(list(AI_DIR.glob("ai_*.png"))) if AI_DIR.is_dir() else 0
    if ai_count:
        print(f"  ai_batch PNG count:  {ai_count} (for parity reference)")

    missing = []
    bad = []
    sizes = []
    for entry in manifest:
        path = HUMAN_DIR / entry["file"]
        if not path.is_file():
            missing.append(entry["file"])
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                sizes.append(im.size)
                mode = im.mode
            if mode not in ("L", "RGB", "RGBA"):
                bad.append(f"{entry['file']}: unexpected mode {mode}")
        except Exception as exc:
            bad.append(f"{entry['file']}: {exc}")

    if missing:
        print(f"\nERROR: {len(missing)} manifest files missing on disk:", file=sys.stderr)
        for name in missing[:5]:
            print(f"  - {name}", file=sys.stderr)
        return 1

    if bad:
        print(f"\nERROR: {len(bad)} invalid images:", file=sys.stderr)
        for msg in bad[:5]:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    if sizes:
        widths = [w for w, _ in sizes]
        heights = [h for _, h in sizes]
        print(f"  Width range:         {min(widths)}–{max(widths)} px")
        print(f"  Height range:        {min(heights)}–{max(heights)} px")

    if manifest:
        sample = manifest[0]
        print(f"  Sample entry:        {sample['file']}")
        if sample.get("text"):
            text_preview = sample["text"][:60] + ("..." if len(sample["text"]) > 60 else "")
            print(f"  Sample transcription: {text_preview!r}")

    print("\nOK — human_samples/ verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
