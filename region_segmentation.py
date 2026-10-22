"""
Region segmentation for hybrid handwritten pages (Step 6).

Uses horizontal projection to detect line boundaries and extract region crops.
No training required - works purely on image structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
HYBRID_PAGES_DIR = PROJECT_ROOT / "hybrid_pages"
OUTPUT_DIR = PROJECT_ROOT / "segmentation_output"


def load_page_image(page_path: Path) -> np.ndarray:
    """Load page as grayscale numpy array."""
    img = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {page_path}")
    return img


def horizontal_projection(img: np.ndarray) -> np.ndarray:
    """Compute horizontal projection (sum of pixel values per row)."""
    # Invert so ink is high values (255), background is low (0)
    inverted = 255 - img
    # Sum across columns for each row
    proj = np.sum(inverted, axis=1)
    return proj


def find_line_boundaries(
    projection: np.ndarray,
    min_gap: int = 30,
    min_content: int = 100,
    margin_top: int = 100,
    margin_bottom: int = 100,
) -> List[Tuple[int, int]]:
    """
    Find line boundaries from horizontal projection.
    
    Args:
        projection: 1D array of row-wise sums
        min_gap: minimum rows of whitespace to consider a line break
        min_content: minimum sum threshold to consider a row as having content
        margin_top: ignore top margin rows
        margin_bottom: ignore bottom margin rows
    
    Returns:
        List of (top, bottom) row indices for each detected line
    """
    height = len(projection)
    
    # Mask out margins
    valid_start = margin_top
    valid_end = height - margin_bottom
    
    # Find rows with content
    content_rows = projection[valid_start:valid_end] > min_content
    
    # Find transitions from no-content to content and vice versa
    transitions = np.diff(content_rows.astype(int))
    
    line_starts = np.where(transitions == 1)[0] + valid_start
    line_ends = np.where(transitions == -1)[0] + valid_start
    
    # Handle edge cases
    if len(line_starts) == 0 and len(line_ends) == 0:
        return []
    
    # If content starts at the very beginning
    if len(line_starts) == 0 or (len(line_ends) > 0 and line_ends[0] < line_starts[0]):
        line_starts = np.insert(line_starts, 0, valid_start)
    
    # If content extends to the end
    if len(line_ends) == 0 or (len(line_starts) > 0 and line_starts[-1] > line_ends[-1]):
        line_ends = np.append(line_ends, valid_end)
    
    # Pair starts and ends
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
    """
    Extract region crops from detected line boundaries.
    
    Args:
        img: Full page image
        boundaries: List of (top, bottom) row indices
        margin_x: Horizontal margin to trim from each region
        min_width: Minimum width for a valid region
    
    Returns:
        List of region dicts with bbox and crop
    """
    height, width = img.shape
    regions = []
    
    for idx, (top, bottom) in enumerate(boundaries):
        # Trim horizontal margins
        left = margin_x
        right = width - margin_x
        
        if right <= left:
            continue
        
        # Extract crop
        crop = img[top:bottom, left:right]
        
        # Skip if too narrow
        if crop.shape[1] < min_width:
            continue
        
        # Find actual content bounds (trim vertical whitespace)
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


def visualize_segmentation(
    img: np.ndarray,
    detected_regions: List[dict],
    ground_truth: dict | None = None,
) -> Image.Image:
    """
    Visualize detected regions against optional ground truth.
    
    Green boxes = detected regions
    Red boxes = ground truth regions (if provided)
    """
    # Convert to RGB for visualization
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    vis_img = Image.fromarray(vis)
    draw = ImageDraw.Draw(vis_img)
    
    # Draw detected regions in green
    for region in detected_regions:
        bbox = region["bbox"]
        rect = [
            (bbox["x"], bbox["y"]),
            (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
        ]
        draw.rectangle(rect, outline=(0, 255, 0), width=3)
    
    # Draw ground truth in red if provided
    if ground_truth:
        for gt_region in ground_truth["regions"]:
            bbox = gt_region["bbox"]
            rect = [
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
            ]
            draw.rectangle(rect, outline=(255, 0, 0), width=2)
    
    return vis_img


def compute_iou(box1: dict, box2: dict) -> float:
    """Compute IoU between two bounding boxes."""
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


def match_regions_to_ground_truth(
    detected: List[dict],
    ground_truth: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Match detected regions to ground truth regions via IoU.
    
    Returns:
        Dict with match statistics
    """
    gt_regions = ground_truth["regions"]
    
    matches = []
    unmatched_detected = list(range(len(detected)))
    unmatched_gt = list(range(len(gt_regions)))
    
    # Greedy matching by highest IoU
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
            matches.append({
                "detected_idx": det_idx,
                "gt_idx": gt_idx,
                "iou": best_iou,
                "detected_label": "detected",
                "gt_label": gt_regions[gt_idx]["label"],
            })
            unmatched_detected.remove(det_idx)
            unmatched_gt.remove(gt_idx)
        else:
            break
    
    return {
        "total_detected": len(detected),
        "total_gt": len(gt_regions),
        "matches": len(matches),
        "unmatched_detected": len(unmatched_detected),
        "unmatched_gt": len(unmatched_gt),
        "match_details": matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Region segmentation for hybrid pages.")
    parser.add_argument(
        "--page-id",
        type=str,
        default=None,
        help="Specific page ID to process (e.g., page_0001). If not set, processes all.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images comparing detected vs ground truth.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save individual region crops to disk.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for visualizations and crops.",
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest_path = HYBRID_PAGES_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return 1
    
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    # Filter pages to process
    if args.page_id:
        pages_to_process = [p for p in manifest if p["page_id"] == args.page_id]
        if not pages_to_process:
            print(f"ERROR: Page {args.page_id} not found in manifest")
            return 1
    else:
        pages_to_process = manifest
    
    print(f"Processing {len(pages_to_process)} page(s)...")
    
    total_matches = 0
    total_gt = 0
    total_detected = 0
    
    for page_info in pages_to_process:
        page_id = page_info["page_id"]
        png_path = HYBRID_PAGES_DIR / page_info["file"]
        json_path = HYBRID_PAGES_DIR / page_info["sidecar"]
        
        print(f"\nProcessing {page_id}...")
        
        # Load image and ground truth
        img = load_page_image(png_path)
        ground_truth = json.loads(json_path.read_text(encoding="utf-8"))
        
        # Segment
        projection = horizontal_projection(img)
        boundaries = find_line_boundaries(projection)
        detected = extract_regions(img, boundaries)
        
        # Match to ground truth
        match_stats = match_regions_to_ground_truth(detected, ground_truth)
        
        total_matches += match_stats["matches"]
        total_gt += match_stats["total_gt"]
        total_detected += match_stats["total_detected"]
        
        print(f"  Detected: {match_stats['total_detected']} regions")
        print(f"  Ground truth: {match_stats['total_gt']} regions")
        print(f"  Matches (IoU >= 0.5): {match_stats['matches']}")
        print(f"  Unmatched detected: {match_stats['unmatched_detected']}")
        print(f"  Unmatched GT: {match_stats['unmatched_gt']}")
        
        # Visualize
        if args.visualize:
            vis = visualize_segmentation(img, detected, ground_truth)
            vis_path = output_dir / f"{page_id}_segmentation.png"
            vis.save(vis_path)
            print(f"  Visualization saved to {vis_path}")
        
        # Save crops
        if args.save_crops:
            crop_dir = output_dir / page_id / "crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            
            for region in detected:
                crop_img = Image.fromarray(region["crop"])
                crop_path = crop_dir / f"region_{region['region_id']:03d}.png"
                crop_img.save(crop_path)
            
            # Save detected bboxes as JSON
            detected_json = output_dir / page_id / "detected_regions.json"
            detected_data = [
                {
                    "region_id": r["region_id"],
                    "bbox": r["bbox"],
                }
                for r in detected
            ]
            detected_json.write_text(json.dumps(detected_data, indent=2), encoding="utf-8")
            print(f"  Crops saved to {crop_dir}")
    
    print("\n" + "=" * 50)
    print("OVERALL STATISTICS")
    print("=" * 50)
    print(f"Total pages processed: {len(pages_to_process)}")
    print(f"Total detected regions: {total_detected}")
    print(f"Total ground truth regions: {total_gt}")
    print(f"Total matches (IoU >= 0.5): {total_matches}")
    if total_gt > 0:
        print(f"Match rate: {100 * total_matches / total_gt:.1f}%")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
