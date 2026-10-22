"""
Prepare region-level dataset for training (Step 7).

Extracts region crops from hybrid pages, matches to ground truth via IoU,
and splits by PAGE (not region) to avoid data leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
HYBRID_PAGES_DIR = PROJECT_ROOT / "hybrid_pages"
OUTPUT_DIR = PROJECT_ROOT / "region_dataset"

# Reuse segmentation logic
def load_page_image(page_path: Path) -> np.ndarray:
    img = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {page_path}")
    return img


def horizontal_projection(img: np.ndarray) -> np.ndarray:
    inverted = 255 - img
    proj = np.sum(inverted, axis=1)
    return proj


def find_line_boundaries(
    projection: np.ndarray,
    min_gap: int = 30,
    min_content: int = 100,
    margin_top: int = 100,
    margin_bottom: int = 100,
) -> List[Tuple[int, int]]:
    height = len(projection)
    valid_start = margin_top
    valid_end = height - margin_bottom
    content_rows = projection[valid_start:valid_end] > min_content
    transitions = np.diff(content_rows.astype(int))
    line_starts = np.where(transitions == 1)[0] + valid_start
    line_ends = np.where(transitions == -1)[0] + valid_start
    
    if len(line_starts) == 0 and len(line_ends) == 0:
        return []
    
    if len(line_starts) == 0 or (len(line_ends) > 0 and line_ends[0] < line_starts[0]):
        line_starts = np.insert(line_starts, 0, valid_start)
    
    if len(line_ends) == 0 or (len(line_starts) > 0 and line_starts[-1] > line_ends[-1]):
        line_ends = np.append(line_ends, valid_end)
    
    boundaries = []
    for start, end in zip(line_starts, line_ends):
        if end > start:
            boundaries.append((int(start), int(end)))
    
    return boundaries


def extract_regions(
    img: np.ndarray,
    boundaries: List[Tuple[int, int]],
    margin_x: int = 60,
    min_width: int = 50,
) -> List[dict]:
    height, width = img.shape
    regions = []
    
    for idx, (top, bottom) in enumerate(boundaries):
        left = margin_x
        right = width - margin_x
        
        if right <= left:
            continue
        
        crop = img[top:bottom, left:right]
        
        if crop.shape[1] < min_width:
            continue
        
        col_proj = np.sum(255 - crop, axis=0)
        content_cols = np.where(col_proj > 50)[0]
        
        if len(content_cols) > 0:
            actual_left = content_cols[0]
            actual_right = content_cols[-1] + 1
            crop = crop[:, actual_left:actual_right]
            left += actual_left
            right = left + crop.shape[1]
        
        regions.append({
            "region_id": idx,
            "bbox": {
                "x": int(left),
                "y": int(top),
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0]),
            },
            "crop": crop,
        })
    
    return regions


def compute_iou(box1: dict, box2: dict) -> float:
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_regions_to_labels(
    detected: List[dict],
    ground_truth: dict,
    iou_threshold: float = 0.5,
) -> List[dict]:
    """Match detected regions to ground truth labels."""
    gt_regions = ground_truth["regions"]
    
    matched = []
    unmatched_detected = list(range(len(detected)))
    unmatched_gt = list(range(len(gt_regions)))
    
    while unmatched_detected and unmatched_gt:
        best_iou = 0.0
        best_pair = (None, None)
        
        for det_idx in unmatched_detected:
            for gt_idx in unmatched_gt:
                iou = compute_iou(detected[det_idx]["bbox"], gt_regions[gt_idx]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (det_idx, gt_idx)
        
        if best_iou >= iou_threshold:
            det_idx, gt_idx = best_pair
            matched.append({
                "detected_idx": det_idx,
                "gt_idx": gt_idx,
                "iou": best_iou,
                "label": gt_regions[gt_idx]["label"],
            })
            unmatched_detected.remove(det_idx)
            unmatched_gt.remove(gt_idx)
        else:
            break
    
    # Build labeled regions list
    labeled = []
    for match in matched:
        det_region = detected[match["detected_idx"]].copy()
        det_region["label"] = match["label"]
        det_region["iou"] = match["iou"]
        labeled.append(det_region)
    
    return labeled


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare region dataset for training.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Ratio of pages for training (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of pages for validation (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: region_dataset/).",
    )
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print("ERROR: train_ratio + val_ratio must be < 1.0", file=sys.stderr)
        return 1
    
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    
    # Load manifest
    manifest_path = HYBRID_PAGES_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return 1
    
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    # Split pages by ID (not regions) to avoid leakage
    rng = random.Random(args.seed)
    rng.shuffle(manifest)
    
    n_total = len(manifest)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    
    train_pages = manifest[:n_train]
    val_pages = manifest[n_train:n_train + n_val]
    test_pages = manifest[n_train + n_val:]
    
    print(f"Split {n_total} pages into:")
    print(f"  Train: {len(train_pages)} pages")
    print(f"  Val: {len(val_pages)} pages")
    print(f"  Test: {len(test_pages)} pages")
    
    # Create output directories
    output_dir = args.output_dir.resolve()
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    def process_split(pages: List[dict], split_dir: Path, split_name: str) -> List[dict]:
        region_records = []
        
        for page_info in pages:
            page_id = page_info["page_id"]
            png_path = HYBRID_PAGES_DIR / page_info["file"]
            json_path = HYBRID_PAGES_DIR / page_info["sidecar"]
            
            # Load image and ground truth
            img = load_page_image(png_path)
            ground_truth = json.loads(json_path.read_text(encoding="utf-8"))
            
            # Segment
            projection = horizontal_projection(img)
            boundaries = find_line_boundaries(projection)
            detected = extract_regions(img, boundaries)
            
            # Match to labels
            labeled = match_regions_to_labels(detected, ground_truth)
            
            # Save crops
            for region in labeled:
                crop_img = Image.fromarray(region["crop"])
                region_filename = f"{page_id}_region_{region['region_id']:03d}.png"
                crop_path = split_dir / region_filename
                crop_img.save(crop_path)
                
                region_records.append({
                    "page_id": page_id,
                    "region_id": region["region_id"],
                    "file": region_filename,
                    "label": region["label"],
                    "iou": region["iou"],
                    "bbox": region["bbox"],
                })
        
        print(f"  {split_name}: {len(region_records)} regions extracted")
        return region_records
    
    train_records = process_split(train_pages, train_dir, "train")
    val_records = process_split(val_pages, val_dir, "val")
    test_records = process_split(test_pages, test_dir, "test")
    
    # Save manifests
    def save_manifest(records: List[dict], path: Path):
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    
    save_manifest(train_records, output_dir / "train_manifest.json")
    save_manifest(val_records, output_dir / "val_manifest.json")
    save_manifest(test_records, output_dir / "test_manifest.json")
    
    # Print label distribution
    def count_labels(records: List[dict]) -> Tuple[int, int]:
        human = sum(1 for r in records if r["label"] == "human")
        ai = sum(1 for r in records if r["label"] == "ai")
        return human, ai
    
    train_h, train_ai = count_labels(train_records)
    val_h, val_ai = count_labels(val_records)
    test_h, test_ai = count_labels(test_records)
    
    print("\nLabel distribution:")
    print(f"  Train: {train_h} human, {train_ai} AI")
    print(f"  Val: {val_h} human, {val_ai} AI")
    print(f"  Test: {test_h} human, {test_ai} AI")
    
    print(f"\nDataset saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
