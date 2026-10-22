"""
Draw ground-truth bounding boxes onto hybrid pages for visual verification (Step 5).

Hybrid pages intentionally use the same black ink for human and AI lines (real
forgeries blend in). This preview adds colored overlays so you can verify labels
without changing the training images.

Green tint = human, red tint = AI. Saves copies under hybrid_pages/preview/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "hybrid_pages"
PREVIEW_DIR_NAME = "preview"

COLOR_HUMAN = (0, 170, 0)
COLOR_AI = (230, 30, 30)
OVERLAY_ALPHA = 165  # strong enough to see on a white page
STRIPE_WIDTH = 24  # solid color bar on the left of each region


def load_sidecar(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    try:
        title = ImageFont.truetype("arial.ttf", 16)
        small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        title = ImageFont.load_default()
        small = title
    return title, small


def annotate_page(page_path: Path, sidecar: dict, out_path: Path) -> None:
    with Image.open(page_path) as page:
        base = page.convert("RGBA")

    title_font, small_font = load_fonts()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for region in sidecar["regions"]:
        bbox = region["bbox"]
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]
        label = region["label"]
        color = COLOR_AI if label == "ai" else COLOR_HUMAN

        overlay_draw.rectangle(
            [x, y, x + w, y + h],
            fill=(*color, OVERLAY_ALPHA),
        )
        stripe_right = min(x + w, x + STRIPE_WIDTH)
        overlay_draw.rectangle(
            [x, y, stripe_right, y + h],
            fill=(*color, 255),
        )

    composed = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)

    for region in sidecar["regions"]:
        bbox = region["bbox"]
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]
        label = region["label"]
        color = COLOR_AI if label == "ai" else COLOR_HUMAN

        draw.rectangle([x, y, x + w, y + h], outline=color, width=4)

        tag = f"{label.upper()} line {region['region_id']}"
        source = region.get("source_file", "")
        tag_y = max(48, y - 40)
        draw.rectangle([x, tag_y, x + 460, tag_y + 38], fill=(255, 255, 255, 240), outline=color, width=3)
        draw.text((x + 8, tag_y + 3), tag, fill=color, font=title_font)
        if source:
            draw.text((x + 8, tag_y + 20), source, fill=(30, 30, 30), font=small_font)

    ratio = sidecar["ai_ratio_target"]
    header = (
        f"{sidecar['page_id']} | AI ratio={ratio:.0%} "
        f"({sidecar['ai_line_count']}/{sidecar['layout']['lines_per_page']} AI lines) | "
        "RED band/tint = AI   GREEN band/tint = human"
    )
    draw.rectangle([0, 0, composed.width, 48], fill=(255, 255, 255, 255))
    draw.text((8, 6), header, fill=(0, 0, 0), font=title_font)
    draw.text(
        (8, 26),
        "Preview only — training pages (page_*.png) stay plain black ink.",
        fill=(60, 60, 60),
        font=small_font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(out_path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize hybrid page ground truth.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing page_*.png and sidecars.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max pages to annotate (0 = all).",
    )
    parser.add_argument(
        "--one-per-ratio",
        action="store_true",
        help="Annotate only the first page from each AI ratio bucket.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preview_dir = input_dir / PREVIEW_DIR_NAME

    entries = manifest
    if args.one_per_ratio:
        seen: set[float] = set()
        filtered = []
        for entry in manifest:
            ratio = entry["ai_ratio_target"]
            if ratio in seen:
                continue
            seen.add(ratio)
            filtered.append(entry)
        entries = filtered

    if args.limit > 0:
        entries = entries[: args.limit]

    if not entries:
        print("ERROR: no pages to annotate.", file=sys.stderr)
        return 1

    print(f"Annotating {len(entries)} page(s) -> {preview_dir}")
    for entry in entries:
        page_id = entry["page_id"]
        page_path = input_dir / entry["file"]
        sidecar_path = input_dir / entry["sidecar"]
        if not page_path.is_file() or not sidecar_path.is_file():
            print(f"ERROR: missing files for {page_id}", file=sys.stderr)
            return 1

        sidecar = load_sidecar(sidecar_path)
        out_path = preview_dir / f"{page_id}_annotated.png"
        annotate_page(page_path, sidecar, out_path)
        ai_lines = [r["region_id"] for r in sidecar["regions"] if r["label"] == "ai"]
        human_lines = [r["region_id"] for r in sidecar["regions"] if r["label"] == "human"]
        print(
            f"  {out_path.resolve()}  |  AI lines: {ai_lines or 'none'}  "
            f"|  human lines: {human_lines or 'none'}"
        )

    print(
        "\nOpen the *_annotated.png files under hybrid_pages/preview/ "
        "(NOT the plain page_*.png files in hybrid_pages/)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
