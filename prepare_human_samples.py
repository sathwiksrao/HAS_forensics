"""
Prepare human handwriting line images for HAS-Forensics (Step 4).

Sources (pick one):
  huggingface  — Teklia/IAM-line on Hugging Face (no IAM registration; default)
  iam-local    — official IAM lines/ tree after you extract data/lines.tgz

Uses huggingface_hub + pyarrow (not the datasets package) for Windows compatibility.

Outputs PNG line crops + manifest.json under human_samples/, mirroring ai_batch/.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
AI_BATCH_DIR = PROJECT_ROOT / "ai_batch"
OUTPUT_DIR = PROJECT_ROOT / "human_samples"


def default_count() -> int:
    if AI_BATCH_DIR.is_dir():
        n = len(list(AI_BATCH_DIR.glob("ai_*.png")))
        if n > 0:
            return n
    return 200


def sanitize_id(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", text).strip("_") or "line"


def existing_indices(output_dir: Path) -> set[int]:
    indices: set[int] = set()
    for path in output_dir.glob("human_*.png"):
        match = re.match(r"human_(\d{4})_", path.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def save_line_image(src: Image.Image, dest: Path) -> tuple[int, tuple[int, int]]:
    """Save as grayscale PNG; return (bytes, (width, height))."""
    if src.mode != "L":
        img = src.convert("L")
    else:
        img = src
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG")
    size = dest.stat().st_size
    return size, img.size


def collect_iam_local_lines(iam_lines_dir: Path) -> list[dict]:
    if not iam_lines_dir.is_dir():
        raise FileNotFoundError(f"IAM lines directory not found: {iam_lines_dir}")

    records: list[dict] = []
    for png in sorted(iam_lines_dir.rglob("*.png")):
        rel = png.relative_to(iam_lines_dir).as_posix()
        records.append(
            {
                "source": "iam-local",
                "source_id": rel,
                "path": png,
                "text": "",
                "split": "",
            }
        )
    if not records:
        raise RuntimeError(f"No PNG files found under {iam_lines_dir}")
    return records


HF_IAM_LINE_SPLITS = (
    ("train", "data/train.parquet"),
    ("validation", "data/validation.parquet"),
    ("test", "data/test.parquet"),
)


class HuggingFaceLinePool:
    """Lazy access to Teklia/IAM-line via parquet (avoids the datasets/pandas stack)."""

    def __init__(self) -> None:
        self._tables: dict[str, object] = {}
        self.records: list[dict] = []

    def load(self) -> None:
        print(
            "Loading Teklia/IAM-line from Hugging Face "
            "(first run downloads ~100 MB; cached after that)..."
        )
        for split_name, relpath in HF_IAM_LINE_SPLITS:
            local_path = hf_hub_download(
                "Teklia/IAM-line", relpath, repo_type="dataset"
            )
            table = pq.read_table(local_path)
            self._tables[split_name] = table
            for row_idx in range(table.num_rows):
                text = table.column("text")[row_idx].as_py() or ""
                self.records.append(
                    {
                        "source": "huggingface",
                        "source_id": f"iam_line_{split_name}_{row_idx:05d}",
                        "split": split_name,
                        "table_key": split_name,
                        "row_index": row_idx,
                        "text": text,
                    }
                )

    def open_image(self, record: dict) -> Image.Image:
        table = self._tables[record["table_key"]]
        img_struct = table.column("image")[record["row_index"]].as_py()
        return Image.open(BytesIO(img_struct["bytes"]))


def collect_huggingface_lines() -> HuggingFaceLinePool:
    pool = HuggingFaceLinePool()
    pool.load()
    return pool


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare human handwriting line images.")
    parser.add_argument(
        "--source",
        choices=("huggingface", "iam-local"),
        default="huggingface",
        help="Data source (default: huggingface — no IAM registration required).",
    )
    parser.add_argument(
        "--iam-lines-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "iam" / "lines",
        help="Path to extracted IAM lines/ tree (for --source iam-local).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=default_count(),
        help="Number of line images to export (defaults to ai_batch PNG count, else 200).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1000,
        help="Reject outputs smaller than this (likely blank or corrupt).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip indices that already exist in human_samples/.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing human_*.png and manifest.json before exporting.",
    )
    args = parser.parse_args()

    if args.count < 1:
        print("ERROR: --count must be >= 1", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for path in OUTPUT_DIR.glob("human_*.png"):
            path.unlink(missing_ok=True)
        manifest_old = OUTPUT_DIR / "manifest.json"
        if manifest_old.is_file():
            manifest_old.unlink(missing_ok=True)

    hf_pool: HuggingFaceLinePool | None = None
    try:
        if args.source == "iam-local":
            pool_records = collect_iam_local_lines(args.iam_lines_dir.resolve())
        else:
            hf_pool = collect_huggingface_lines()
            pool_records = hf_pool.records
    except Exception as exc:
        print(f"ERROR: failed to load source '{args.source}': {exc}", file=sys.stderr)
        return 1

    print(f"Source pool size: {len(pool_records)} lines")
    if len(pool_records) < args.count:
        print(
            f"ERROR: pool has only {len(pool_records)} lines but --count={args.count}.",
            file=sys.stderr,
        )
        return 1

    random.seed(args.seed)
    chosen = random.sample(pool_records, args.count)

    manifest: list[dict] = []
    if args.resume and (OUTPUT_DIR / "manifest.json").is_file():
        manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))

    done_indices = existing_indices(OUTPUT_DIR) if args.resume else set()
    target_indices = [i for i in range(1, args.count + 1) if i not in done_indices]

    print(f"Exporting {args.count} images -> {OUTPUT_DIR}")
    if done_indices:
        print(f"Resuming: {len(done_indices)} images already present")

    for idx in target_indices:
        record = chosen[idx - 1]
        source_id = sanitize_id(record["source_id"])
        out_name = f"human_{idx:04d}_{source_id}.png"
        out_path = OUTPUT_DIR / out_name

        if args.source == "iam-local":
            src_img = Image.open(record["path"])
        else:
            assert hf_pool is not None
            src_img = hf_pool.open_image(record)

        try:
            nbytes, (width, height) = save_line_image(src_img, out_path)
        except Exception as exc:
            print(f"ERROR: failed to save {out_name}: {exc}", file=sys.stderr)
            return 1

        if nbytes < args.min_bytes:
            out_path.unlink(missing_ok=True)
            print(
                f"ERROR: {out_name} too small ({nbytes} bytes); check source data.",
                file=sys.stderr,
            )
            return 1

        entry = {
            "file": out_name,
            "source": record["source"],
            "source_id": record["source_id"],
            "text": record.get("text", ""),
            "split": record.get("split", ""),
            "width": width,
            "height": height,
            "bytes": nbytes,
        }
        manifest = [m for m in manifest if m.get("file") != out_name]
        manifest.append(entry)
        done_indices.add(idx)

        if len(done_indices) % 10 == 0 or len(done_indices) == args.count:
            print(f"  {len(done_indices)}/{args.count} saved")

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(sorted(manifest, key=lambda m: m["file"]), indent=2),
        encoding="utf-8",
    )

    png_count = len(list(OUTPUT_DIR.glob("human_*.png")))
    print()
    print("Export complete.")
    print(f"  PNG files in human_samples/: {png_count}")
    print(f"  Manifest entries:           {len(manifest)}")
    print(f"  Manifest path:              {manifest_path}")
    print(f"  Source:                     {args.source}")

    if png_count != args.count:
        print(
            f"\nERROR: file count mismatch (expected {args.count}, found {png_count}).",
            file=sys.stderr,
        )
        return 1

    if len(manifest) != args.count:
        print(
            f"\nERROR: manifest mismatch (expected {args.count}, found {len(manifest)}).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
