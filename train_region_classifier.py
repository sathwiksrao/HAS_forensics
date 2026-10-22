"""
Train region-level human/AI classifier (Step 7).

Fine-tunes EfficientNet-B0 for binary classification on handwriting region crops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "region_dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

LABEL_MAP = {"human": 0, "ai": 1}
LABEL_MAP_INV = {0: "human", 1: "ai"}


class RegionDataset(Dataset):
    """PyTorch Dataset for handwriting region crops."""
    
    def __init__(self, data_dir: Path, manifest_path: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        with open(manifest_path, encoding="utf-8") as f:
            self.records = json.load(f)
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int):
        record = self.records[idx]
        img_path = self.data_dir / record["file"]
        
        # Load as grayscale, convert to RGB for EfficientNet
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        label = LABEL_MAP[record["label"]]
        
        return img, label, record["file"]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get image transforms for training or evaluation."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_model(num_classes: int = 2) -> nn.Module:
    """Create EfficientNet-B0 with pretrained weights."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Replace final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model, return metrics dict."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train region classifier.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Checkpoint directory.",
    )
    args = parser.parse_args()
    
    # Check dataset
    train_manifest = DATASET_DIR / "train_manifest.json"
    val_manifest = DATASET_DIR / "val_manifest.json"
    test_manifest = DATASET_DIR / "test_manifest.json"
    
    if not all(p.exists() for p in [train_manifest, val_manifest, test_manifest]):
        print("ERROR: Dataset manifests not found. Run prepare_region_dataset.py first.")
        return 1
    
    # Create datasets
    train_dataset = RegionDataset(
        DATASET_DIR / "train",
        train_manifest,
        transform=get_transforms(train=True),
    )
    val_dataset = RegionDataset(
        DATASET_DIR / "val",
        val_manifest,
        transform=get_transforms(train=False),
    )
    test_dataset = RegionDataset(
        DATASET_DIR / "test",
        test_manifest,
        transform=get_transforms(train=False),
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = create_model(num_classes=2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_f1 = 0.0
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        
        # Save best checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_metrics["f1"],
            }, checkpoint_path)
            print(f"  -> Saved best checkpoint (F1: {best_val_f1:.4f})")
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating best model on test set...")
    print("-" * 60)
    
    checkpoint_path = checkpoint_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    
    # Save test results
    results_path = checkpoint_dir / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nTest results saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
