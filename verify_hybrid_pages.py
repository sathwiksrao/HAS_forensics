"""
Verify hybrid_pages/ output for HAS-Forensics Step 5.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
HYBRID_DIR = PROJECT_ROOT / "hybrid_pages"

CANVAS_WIDTH = 1240
CANVAS_HEIGHT = 1754
LINES_PER_PAGE = 6


def bbox_within_canvas(bbox: dict, width: int, height: int) -> bool:
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    return x >= 0 and y >= 0 and x + w <= width and y + h <= height


def main() -> int:
    manifest_path = HYBRID_DIR / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("Step 5 verification — hybrid_pages/")
    print(f"  Pages in manifest:   {len(manifest)}")

    ratio_counts = Counter(entry["ai_ratio_target"] for entry in manifest)
    print(f"  AI ratio buckets:    {len(ratio_counts)}")
    for ratio in sorted(ratio_counts):
        print(f"    {ratio:>5.0%}: {ratio_counts[ratio]} pages")

    errors: list[str] = []
    for entry in manifest:
        page_id = entry["page_id"]
        png_path = HYBRID_DIR / entry["file"]
        json_path = HYBRID_DIR / entry["sidecar"]

        if not png_path.is_file():
            errors.append(f"{page_id}: missing PNG")
            continue
        if not json_path.is_file():
            errors.append(f"{page_id}: missing sidecar JSON")
            continue

        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        regions = sidecar.get("regions", [])

        if len(regions) != LINES_PER_PAGE:
            errors.append(f"{page_id}: expected {LINES_PER_PAGE} regions, got {len(regions)}")

        try:
            with Image.open(png_path) as im:
                if im.size != (CANVAS_WIDTH, CANVAS_HEIGHT):
                    errors.append(f"{page_id}: canvas size {im.size}, expected {(CANVAS_WIDTH, CANVAS_HEIGHT)}")
        except Exception as exc:
            errors.append(f"{page_id}: invalid PNG ({exc})")

        ai_count = sum(1 for r in regions if r.get("label") == "ai")
        if ai_count != sidecar.get("ai_line_count"):
            errors.append(
                f"{page_id}: ai_line_count mismatch ({sidecar.get('ai_line_count')} vs {ai_count})"
            )

        for region in regions:
            bbox = region.get("bbox", {})
            if not bbox_within_canvas(bbox, CANVAS_WIDTH, CANVAS_HEIGHT):
                errors.append(f"{page_id}: bbox out of canvas for region {region.get('region_id')}")

    if errors:
        print(f"\nERROR: {len(errors)} issue(s):", file=sys.stderr)
        for msg in errors[:10]:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print(f"  Canvas size:         {CANVAS_WIDTH}x{CANVAS_HEIGHT}")
    print(f"  Lines per page:      {LINES_PER_PAGE}")
    print("\nOK — hybrid_pages/ verified.")
    print("Run visualize_hybrid_pages.py --one-per-ratio and inspect hybrid_pages/preview/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
