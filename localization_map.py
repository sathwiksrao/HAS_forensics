"""
Generate page-level localization maps from region predictions (Step 8).

Stitches per-region classifier predictions into page-level masks,
applies Sobel edge-consistency for boundary refinement, and evaluates
against ground truth masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
HYBRID_PAGES_DIR = PROJECT_ROOT / "hybrid_pages"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "localization_output"

LABEL_MAP = {"human": 0, "ai": 1}
LABEL_MAP_INV = {0: "human", 1: "ai"}


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


def create_model(num_classes: int = 2) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_regions(
    model: nn.Module,
    regions: List[dict],
    transform: transforms.Compose,
    device: torch.device,
) -> List[dict]:
    """Run classifier on region crops, return regions with predictions."""
    model.eval()
    
    with torch.no_grad():
        for region in regions:
            # Convert crop to RGB and apply transform
            crop_img = Image.fromarray(region["crop"]).convert("RGB")
            input_tensor = transform(crop_img).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
            
            region["prediction"] = LABEL_MAP_INV[pred_class]
            region["confidence"] = confidence
    
    return regions


def stitch_mask(
    regions: List[dict],
    canvas_shape: Tuple[int, int],
    label_to_mask: str = "ai",
) -> np.ndarray:
    """
    Stitch region predictions into a page-level binary mask.
    
    Args:
        regions: List of region dicts with bbox and prediction
        canvas_shape: (height, width) of the page
        label_to_mask: Which label to mask (default: "ai")
    
    Returns:
        Binary mask where 1 = predicted as label_to_mask, 0 = otherwise
    """
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    
    for region in regions:
        if region["prediction"] == label_to_mask:
            bbox = region["bbox"]
            y1 = bbox["y"]
            y2 = y1 + bbox["height"]
            x1 = bbox["x"]
            x2 = x1 + bbox["width"]
            mask[y1:y2, x1:x2] = 1
    
    return mask


def apply_sobel_edge_consistency(
    mask: np.ndarray,
    original_img: np.ndarray,
    edge_threshold: float = 0.3,
) -> np.ndarray:
    """
    Refine mask boundaries using Sobel edge consistency.
    
    The idea: mask boundaries should align with strong edges in the original image.
    We use Sobel edges to sharpen the mask boundaries.
    """
    # Compute Sobel edges on original image
    sobel_x = cv2.Sobel(original_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(original_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize edge magnitude to [0, 1]
    if sobel_magnitude.max() > 0:
        sobel_magnitude = sobel_magnitude / sobel_magnitude.max()
    
    # Dilate the mask slightly to capture boundary regions
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    boundary_region = dilated_mask - mask
    
    # In boundary regions, keep mask where edges are strong
    if boundary_region.sum() > 0:
        edge_mask = (sobel_magnitude > edge_threshold).astype(np.uint8)
        refined = mask.copy()
        refined[boundary_region > 0] = edge_mask[boundary_region > 0]
        return refined
    
    return mask


def generate_ground_truth_mask(
    ground_truth: dict,
    canvas_shape: Tuple[int, int],
    label_to_mask: str = "ai",
) -> np.ndarray:
    """Generate ground truth mask from JSON sidecar."""
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    
    for region in ground_truth["regions"]:
        if region["label"] == label_to_mask:
            bbox = region["bbox"]
            y1 = bbox["y"]
            y2 = y1 + bbox["height"]
            x1 = bbox["x"]
            x2 = x1 + bbox["width"]
            mask[y1:y2, x1:x2] = 1
    
    return mask


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_pixel_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute pixel-level F1 score."""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Handle edge case: both masks are empty (all human or all AI pages)
    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
        return 1.0  # Perfect agreement on "no AI regions"
    
    return f1_score(gt_flat, pred_flat, average="binary", zero_division=0)


def visualize_comparison(
    original_img: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    page_id: str,
) -> Image.Image:
    """Create side-by-side comparison visualization."""
    # Convert original to RGB
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    
    # Create colored masks
    pred_color = np.zeros_like(original_rgb)
    pred_color[pred_mask > 0] = [255, 0, 0]  # Red for AI prediction
    
    gt_color = np.zeros_like(original_rgb)
    gt_color[gt_mask > 0] = [0, 255, 0]  # Green for ground truth
    
    # Blend masks with original
    alpha = 0.4
    pred_blend = cv2.addWeighted(original_rgb, 1 - alpha, pred_color, alpha, 0)
    gt_blend = cv2.addWeighted(original_rgb, 1 - alpha, gt_color, alpha, 0)
    
    # Stack horizontally
    combined = np.hstack([original_rgb, pred_blend, gt_blend])
    
    # Add labels
    combined_img = Image.fromarray(combined)
    draw = ImageDraw.Draw(combined_img)
    
    h = original_img.shape[0]
    w = original_img.shape[1]
    
    # Simple text labels (using default font)
    try:
        draw.text((10, 10), "Original", fill=(255, 255, 255))
        draw.text((w + 10, 10), "Prediction (Red=AI)", fill=(255, 255, 255))
        draw.text((2 * w + 10, 10), "Ground Truth (Green=AI)", fill=(255, 255, 255))
    except:
        pass  # Font not available, skip labels
    
    return combined_img


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate localization maps.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to process (default: test).",
    )
    parser.add_argument(
        "--apply-edge-refinement",
        action="store_true",
        help="Apply Sobel edge-consistency refinement.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory.",
    )
    args = parser.parse_args()
    
    # Load model
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_model(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    transform = get_transform()
    
    # Load test manifest
    test_manifest_path = PROJECT_ROOT / "region_dataset" / f"{args.split}_manifest.json"
    if not test_manifest_path.exists():
        print(f"ERROR: Manifest not found at {test_manifest_path}")
        return 1
    
    with open(test_manifest_path, encoding="utf-8") as f:
        test_records = json.load(f)
    
    # Get unique page IDs
    page_ids = sorted(set(r["page_id"] for r in test_records))
    print(f"Processing {len(page_ids)} pages from {args.split} split...")
    
    # Create output directory
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each page
    all_ious = []
    all_f1s = []
    
    for page_id in page_ids:
        print(f"\nProcessing {page_id}...")
        
        # Load page and ground truth
        png_path = HYBRID_PAGES_DIR / f"{page_id}.png"
        json_path = HYBRID_PAGES_DIR / f"{page_id}.json"
        
        img = load_page_image(png_path)
        ground_truth = json.loads(json_path.read_text(encoding="utf-8"))
        
        # Segment regions
        projection = horizontal_projection(img)
        boundaries = find_line_boundaries(projection)
        regions = extract_regions(img, boundaries)
        
        # Predict
        regions = predict_regions(model, regions, transform, device)
        
        # Stitch mask
        pred_mask = stitch_mask(regions, img.shape, label_to_mask="ai")
        
        # Apply edge refinement if requested
        if args.apply_edge_refinement:
            pred_mask = apply_sobel_edge_consistency(pred_mask, img)
        
        # Generate ground truth mask
        gt_mask = generate_ground_truth_mask(ground_truth, img.shape, label_to_mask="ai")
        
        # Compute metrics
        iou = compute_iou(pred_mask, gt_mask)
        f1 = compute_pixel_f1(pred_mask, gt_mask)
        
        all_ious.append(iou)
        all_f1s.append(f1)
        
        print(f"  IoU: {iou:.4f}")
        print(f"  Pixel F1: {f1:.4f}")
        
        # Visualize
        vis = visualize_comparison(img, pred_mask, gt_mask, page_id)
        vis_path = output_dir / f"{page_id}_localization.png"
        vis.save(vis_path)
        print(f"  Visualization saved to {vis_path}")
        
        # Save masks
        cv2.imwrite(str(output_dir / f"{page_id}_pred_mask.png"), pred_mask * 255)
        cv2.imwrite(str(output_dir / f"{page_id}_gt_mask.png"), gt_mask * 255)
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Pages processed: {len(page_ids)}")
    print(f"Mean IoU: {np.mean(all_ious):.4f}")
    print(f"Mean Pixel F1: {np.mean(all_f1s):.4f}")
    print(f"IoU Std: {np.std(all_ious):.4f}")
    print(f"F1 Std: {np.std(all_f1s):.4f}")
    
    # Save results
    results = {
        "split": args.split,
        "num_pages": len(page_ids),
        "mean_iou": float(np.mean(all_ious)),
        "std_iou": float(np.std(all_ious)),
        "mean_f1": float(np.mean(all_f1s)),
        "std_f1": float(np.std(all_f1s)),
        "per_page_iou": [float(x) for x in all_ious],
        "per_page_f1": [float(x) for x in all_f1s],
    }
    
    results_path = output_dir / f"{args.split}_localization_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
