"""
Generate a batch of AI handwriting line images for HAS-Forensics (Step 3).

Calls synthesize.py via subprocess (with env=os.environ.copy()) for each sample,
varying checkpoint and bias for visual diversity. Outputs renamed PNGs to ai_batch/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TOOLKIT_DIR = PROJECT_ROOT / "pytorch-handwriting-synthesis-toolkit"
SYNTHESIZE_SCRIPT = TOOLKIT_DIR / "synthesize.py"
OUTPUT_DIR = PROJECT_ROOT / "ai_batch"

CHECKPOINTS = ["Epoch_46", "Epoch_50", "Epoch_52", "Epoch_56"]

# Long enough lines to produce visible strokes (short words like "test" can be blank).
LINE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Forensic analysis requires careful attention to detail",
    "Handwritten documents may contain mixed authentic content",
    "Region level detection identifies localized forgeries",
    "Machine learning models can synthesize realistic handwriting",
    "Auditors examine contracts for signs of tampering",
    "Each line of text carries its own authorship signal",
    "Hybrid pages blend human and artificial ink strokes",
    "Document verification remains a critical security task",
    "Neural networks learn stroke patterns from training data",
    "The signature at the bottom must match the reference",
    "Evidence presented in court must be carefully validated",
    "Optical scanning preserves the texture of pen on paper",
    "Localization maps reveal which regions are synthetic",
    "Whole page classifiers fail when AI content is sparse",
    "Bounding boxes annotate each region on the page",
    "Training data diversity improves generalization performance",
    "Mobile networks classify small cropped image patches",
    "Edge consistency helps refine segmentation boundaries",
    "Research interns document every step of the pipeline",
    "Python scripts orchestrate batch synthesis on CPU hardware",
    "Checkpoints capture different stages of model convergence",
    "Bias controls the tradeoff between clarity and diversity",
    "Ground truth labels enable supervised metric evaluation",
    "Intersection over union measures localization accuracy",
    "Precision and recall summarize classifier performance",
    "Held out test pages prevent optimistic leakage estimates",
    "Canvas compositing builds hybrid pages from line crops",
    "Connected components segment ink regions without training",
    "Sobel filters highlight discontinuities at region edges",
    "Comparison tables contrast page level and region methods",
    "Novel task framing combines forgery localization ideas",
    "Human writers show natural variation in letter forms",
    "Synthetic lines may appear unusually uniform or clean",
    "Multiple epochs produce subtly different stroke styles",
    "Random sampling explores the generative model space",
    "Manifest files record parameters for each generated image",
    "Image forgery research inspires handwriting localization",
    "Typed text hybrid detection differs from ink analysis",
    "Court examiners need to know where forgery occurred",
]


def sanitize_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text)


def run_synthesis(
    python_exe: Path,
    checkpoint: str,
    text: str,
    bias: float,
    samples_dir: Path,
) -> subprocess.CompletedProcess[str]:
    model_path = f"checkpoints/{checkpoint}"
    cmd = [
        str(python_exe),
        str(SYNTHESIZE_SCRIPT),
        model_path,
        text,
        "-b",
        str(bias),
        "--samples_dir",
        str(samples_dir),
        "--trials",
        "1",
    ]
    return subprocess.run(
        cmd,
        cwd=str(TOOLKIT_DIR),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )


def expected_output_path(samples_dir: Path, text: str) -> Path:
    return samples_dir / f"{sanitize_text(text)}_1.png"


def wait_for_output(samples_dir: Path, text: str, retries: int = 10, delay_s: float = 0.25) -> Path | None:
    """Wait for synthesize.py to finish writing the PNG (avoids rare Windows race)."""
    expected = expected_output_path(samples_dir, text)
    for _ in range(retries):
        if expected.is_file():
            return expected
        time.sleep(delay_s)

    matches = sorted(samples_dir.glob("*_1.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return None


def existing_indices(output_dir: Path) -> set[int]:
    indices: set[int] = set()
    for path in output_dir.glob("ai_*.png"):
        match = re.match(r"ai_(\d{4})_", path.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-generate AI handwriting line images.")
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of images to generate (recommended: 150-300).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=3000,
        help="Reject outputs smaller than this (likely blank or degenerate).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip indices that already exist in ai_batch/ (ai_XXXX_*.png).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing ai_*.png and manifest.json before generating.",
    )
    args = parser.parse_args()

    if not (150 <= args.count <= 300):
        print(f"Warning: count={args.count} is outside 150-300; proceeding anyway.", file=sys.stderr)

    if not SYNTHESIZE_SCRIPT.is_file():
        print(f"ERROR: synthesize.py not found at {SYNTHESIZE_SCRIPT}", file=sys.stderr)
        return 1

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for path in OUTPUT_DIR.glob("ai_*.png"):
            path.unlink(missing_ok=True)
        manifest_old = OUTPUT_DIR / "manifest.json"
        if manifest_old.is_file():
            manifest_old.unlink(missing_ok=True)

    temp_dir = OUTPUT_DIR / "_tmp_synth"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for leftover in temp_dir.glob("*"):
        leftover.unlink(missing_ok=True)

    python_exe = Path(sys.executable)
    manifest: list[dict] = []
    if args.resume and (OUTPUT_DIR / "manifest.json").is_file():
        manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))

    done_indices = existing_indices(OUTPUT_DIR) if args.resume else set()
    attempts = 0
    max_attempts = args.count * 8

    print(f"Generating {args.count} images -> {OUTPUT_DIR}")
    print(f"Python: {python_exe}")
    print(f"Toolkit: {TOOLKIT_DIR}")
    if done_indices:
        print(f"Resuming: {len(done_indices)} images already present")

    while len(done_indices) < args.count and attempts < max_attempts:
        attempts += 1
        idx = next((i for i in range(1, args.count + 1) if i not in done_indices), None)
        if idx is None:
            break

        checkpoint = random.choice(CHECKPOINTS)
        bias = round(random.uniform(0.1, 1.0), 2)
        base_line = random.choice(LINE_TEXTS)
        text = f"{base_line} n{idx:04d}"

        result = run_synthesis(python_exe, checkpoint, text, bias, temp_dir)
        if result.returncode != 0:
            print(f"\nERROR: synthesis failed (attempt {attempts}, sample {idx})", file=sys.stderr)
            print(f"  checkpoint={checkpoint} bias={bias}", file=sys.stderr)
            print(f"  text={text!r}", file=sys.stderr)
            print(f"stdout:\n{result.stdout}", file=sys.stderr)
            print(f"stderr:\n{result.stderr}", file=sys.stderr)
            return 1

        raw_path = wait_for_output(temp_dir, text)
        if raw_path is None:
            print(
                f"WARNING: output missing for sample {idx}; retrying synthesis...",
                file=sys.stderr,
            )
            continue

        size = raw_path.stat().st_size
        if size < args.min_bytes:
            print(
                f"Skipping blank/tiny output ({size} bytes) for sample {idx}; retrying...",
                file=sys.stderr,
            )
            raw_path.unlink(missing_ok=True)
            continue

        out_name = f"ai_{idx:04d}_{checkpoint}_b{bias:.2f}.png"
        out_path = OUTPUT_DIR / out_name
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        raw_path.rename(out_path)

        entry = {
            "file": out_name,
            "checkpoint": checkpoint,
            "bias": bias,
            "text": text,
            "bytes": size,
        }
        manifest = [m for m in manifest if m.get("file") != out_name]
        manifest.append(entry)
        done_indices.add(idx)
        count_so_far = len(done_indices)
        if count_so_far % 10 == 0 or count_so_far == args.count:
            print(f"  {count_so_far}/{args.count} saved")
            manifest_path = OUTPUT_DIR / "manifest.json"
            manifest_path.write_text(json.dumps(sorted(manifest, key=lambda m: m["file"]), indent=2), encoding="utf-8")

    for leftover in temp_dir.glob("*"):
        leftover.unlink(missing_ok=True)
    temp_dir.rmdir()

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(sorted(manifest, key=lambda m: m["file"]), indent=2),
        encoding="utf-8",
    )

    png_count = len(list(OUTPUT_DIR.glob("ai_*.png")))
    print()
    print("Batch complete.")
    print(f"  PNG files in ai_batch/: {png_count}")
    print(f"  Manifest entries:     {len(manifest)}")
    print(f"  Manifest path:        {manifest_path}")

    if len(done_indices) < args.count:
        print(
            f"\nERROR: only generated {len(done_indices)}/{args.count} after {attempts} attempts.",
            file=sys.stderr,
        )
        return 1

    if png_count != args.count:
        print(
            f"\nERROR: file count mismatch (expected {args.count}, found {png_count}).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
