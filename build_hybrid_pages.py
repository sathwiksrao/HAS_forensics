"""
Build hybrid handwritten pages with ground-truth region labels (Step 5).

Composites human and AI line images onto A4 canvases (1240x1754 @ 150 DPI),
stacked horizontally like a real handwritten page. Each page gets a JSON sidecar
with per-region bounding boxes and human/ai labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
AI_BATCH_DIR = PROJECT_ROOT / "ai_batch"
HUMAN_SAMPLES_DIR = PROJECT_ROOT / "human_samples"
OUTPUT_DIR = PROJECT_ROOT / "hybrid_pages"

CANVAS_WIDTH = 1240
CANVAS_HEIGHT = 1754
CANVAS_DPI = 150
LINES_PER_PAGE = 6

MARGIN_X = 80
MARGIN_Y = 120

AI_RATIO_TARGETS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

LABEL_HUMAN = "human"
LABEL_AI = "ai"


def load_manifest(directory: Path, manifest_name: str = "manifest.json") -> list[dict]:
    path = directory / manifest_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ai_line_count(target_ratio: float, lines_per_page: int = LINES_PER_PAGE) -> int:
    if target_ratio <= 0.0:
        return 0
    if target_ratio >= 1.0:
        return lines_per_page
    return min(lines_per_page, max(1, round(lines_per_page * target_ratio)))


def scale_to_fit_width(img: Image.Image, max_width: int) -> Image.Image:
    if img.width <= max_width:
        return img.copy()
    scale = max_width / img.width
    new_size = (max_width, max(1, round(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def fit_in_slot(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scaled = scale_to_fit_width(img, max_width)
    if scaled.height <= max_height:
        return scaled
    scale = max_height / scaled.height
    new_size = (max(1, round(scaled.width * scale)), max_height)
    return scaled.resize(new_size, Image.Resampling.LANCZOS)


def build_page(
    page_index: int,
    ai_ratio_target: float,
    human_pool: list[dict],
    ai_pool: list[dict],
    human_cursor: int,
    ai_cursor: int,
    rng: random.Random,
    lines_per_page: int = LINES_PER_PAGE,
) -> tuple[Image.Image, dict, int, int]:
    content_width = CANVAS_WIDTH - 2 * MARGIN_X
    content_height = CANVAS_HEIGHT - 2 * MARGIN_Y
    slot_height = content_height // lines_per_page

    n_ai = ai_line_count(ai_ratio_target, lines_per_page)
    ai_slots = set(rng.sample(range(lines_per_page), n_ai)) if n_ai else set()

    canvas = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), color=255)
    regions: list[dict] = []

    for slot in range(lines_per_page):
        use_ai = slot in ai_slots
        if use_ai:
            entry = ai_pool[ai_cursor % len(ai_pool)]
            ai_cursor += 1
            source_dir = AI_BATCH_DIR
            label = LABEL_AI
        else:
            entry = human_pool[human_cursor % len(human_pool)]
            human_cursor += 1
            source_dir = HUMAN_SAMPLES_DIR
            label = LABEL_HUMAN

        source_path = source_dir / entry["file"]
        with Image.open(source_path) as raw:
            line_img = raw.convert("L")

        fitted = fit_in_slot(line_img, content_width, slot_height)
        slot_top = MARGIN_Y + slot * slot_height
        slot_left = MARGIN_X

        paste_x = slot_left + (content_width - fitted.width) // 2
        paste_y = slot_top + (slot_height - fitted.height) // 2

        canvas.paste(fitted, (paste_x, paste_y))

        regions.append(
            {
                "region_id": slot,
                "line_slot": slot,
                "label": label,
                "bbox": {
                    "x": paste_x,
                    "y": paste_y,
                    "width": fitted.width,
                    "height": fitted.height,
                },
                "source_file": entry["file"],
            }
        )

    page_id = f"page_{page_index:04d}"
    metadata = {
        "page_id": page_id,
        "file": f"{page_id}.png",
        "canvas": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "dpi": CANVAS_DPI,
        },
        "layout": {
            "style": "stacked_horizontal_lines",
            "lines_per_page": lines_per_page,
            "margin_x": MARGIN_X,
            "margin_y": MARGIN_Y,
            "scale_policy": "fit_width_preserve_aspect",
        },
        "ai_ratio_target": ai_ratio_target,
        "ai_line_count": n_ai,
        "human_line_count": lines_per_page - n_ai,
        "regions": regions,
    }
    return canvas, metadata, human_cursor, ai_cursor


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hybrid pages with ground truth.")
    parser.add_argument(
        "--pages-per-ratio",
        type=int,
        default=4,
        help="Pages generated for each AI ratio bucket (default: 4).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: hybrid_pages/).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing page PNG/JSON files in output dir before building.",
    )
    args = parser.parse_args()

    if args.pages_per_ratio < 1:
        print("ERROR: --pages-per-ratio must be >= 1", file=sys.stderr)
        return 1

    try:
        human_pool = load_manifest(HUMAN_SAMPLES_DIR)
        ai_pool = load_manifest(AI_BATCH_DIR)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not human_pool or not ai_pool:
        print("ERROR: human_samples/ or ai_batch/ is empty.", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for pattern in ("page_*.png", "page_*.json"):
            for path in output_dir.glob(pattern):
                path.unlink(missing_ok=True)
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            manifest_path.unlink(missing_ok=True)

    rng = random.Random(args.seed)
    rng.shuffle(human_pool)
    rng.shuffle(ai_pool)

    total_pages = args.pages_per_ratio * len(AI_RATIO_TARGETS)
    human_cursor = 0
    ai_cursor = 0
    page_manifest: list[dict] = []

    print(f"Building {total_pages} hybrid pages -> {output_dir}")
    print(
        f"  Canvas: {CANVAS_WIDTH}x{CANVAS_HEIGHT} @ {CANVAS_DPI} DPI, "
        f"{LINES_PER_PAGE} stacked lines"
    )

    page_index = 1
    for ratio in AI_RATIO_TARGETS:
        for _ in range(args.pages_per_ratio):
            canvas, metadata, human_cursor, ai_cursor = build_page(
                page_index=page_index,
                ai_ratio_target=ratio,
                human_pool=human_pool,
                ai_pool=ai_pool,
                human_cursor=human_cursor,
                ai_cursor=ai_cursor,
                rng=rng,
            )

            png_path = output_dir / metadata["file"]
            json_path = output_dir / f"{metadata['page_id']}.json"

            canvas.save(png_path, format="PNG", dpi=(CANVAS_DPI, CANVAS_DPI))
            json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            page_manifest.append(
                {
                    "page_id": metadata["page_id"],
                    "file": metadata["file"],
                    "sidecar": json_path.name,
                    "ai_ratio_target": ratio,
                    "ai_line_count": metadata["ai_line_count"],
                    "region_count": len(metadata["regions"]),
                }
            )

            if page_index % 6 == 0 or page_index == total_pages:
                print(f"  {page_index}/{total_pages} pages saved")

            page_index += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(page_manifest, indent=2), encoding="utf-8")

    png_count = len(list(output_dir.glob("page_*.png")))
    json_count = len(list(output_dir.glob("page_*.json")))

    print()
    print("Build complete.")
    print(f"  Page PNGs:        {png_count}")
    print(f"  Page sidecars:    {json_count}")
    print(f"  Index manifest:   {manifest_path}")
    print(f"  AI ratio buckets: {AI_RATIO_TARGETS}")

    if png_count != total_pages or json_count != total_pages:
        print(
            f"\nERROR: expected {total_pages} pages, "
            f"found {png_count} PNGs and {json_count} sidecars.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
