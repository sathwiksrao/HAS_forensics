# HAS-Forensics

HAS-Forensics is a Python-based project for generating synthetic handwriting, mixing it with human handwriting, building hybrid pages, and training models to detect and localize AI-generated handwriting regions.

## Project Overview

The pipeline includes these major stages:

1. Generate AI handwriting samples
2. Prepare real human handwriting samples
3. Build hybrid pages with ground-truth labels
4. Segment pages into regions
5. Train a region-level classifier
6. Train a baseline whole-page classifier
7. Generate localization maps
8. Compare the two methods

## Requirements

- Python 3.10+
- Windows PowerShell or Command Prompt
- A virtual environment is recommended

## Setup

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\pytorch-handwriting-synthesis-toolkit\requirements.txt
python -m pip install opencv-python scikit-learn pyarrow huggingface-hub
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Run the Web Dashboard

Start the website interface:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The dashboard lets you run the pipeline steps from the browser.

## Run the Pipeline Manually

From the project root:

```powershell
python generate_ai_batch.py
python prepare_human_samples.py
python build_hybrid_pages.py --pages-per-ratio 4
python region_segmentation.py --visualize
python train_region_classifier.py --epochs 20 --batch-size 8
python train_baseline_classifier.py --epochs 20 --batch-size 8
python localization_map.py
python compare_methods.py --split test
```

## Output Folders

- `ai_batch/` - generated AI handwriting images
- `human_samples/` - prepared human handwriting samples
- `hybrid_pages/` - hybrid pages with JSON labels
- `region_dataset/` - region-based training dataset
- `checkpoints/` and `baseline_checkpoints/` - trained model weights
- `segmentation_output/`, `localization_output/`, `comparison_results/` - analysis outputs

## Notes

- The synthetic handwriting generator is located in `pytorch-handwriting-synthesis-toolkit/`.
- Some steps may take a while depending on your machine and dataset download size.
- If you use the web dashboard, ensure the same Python environment is active when launching the app.
