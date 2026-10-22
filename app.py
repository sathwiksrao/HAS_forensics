"""
HAS-Forensics Web Dashboard

Flask app providing browser UI for running the 9-step pipeline and inference.
All scripts run locally via subprocess with proper environment handling.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request, Response, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXECUTABLE = sys.executable
STATIC_DIR = PROJECT_ROOT / "static"
INFERENCE_DIR = STATIC_DIR / "inference_results"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Step definitions with prerequisites
STEPS = [
    {
        "id": "step1",
        "name": "Environment Setup",
        "description": "Create venv and install dependencies",
        "script": None,
        "command": None,
        "prerequisite_folders": [],
    },
    {
        "id": "step2",
        "name": "Clone Handwriting Generator",
        "description": "Clone and verify pytorch-handwriting-synthesis-toolkit",
        "script": None,
        "command": None,
        "prerequisite_folders": [],
    },
    {
        "id": "step3",
        "name": "Generate AI Batch",
        "description": "Generate 150-300 AI handwriting line images",
        "script": "generate_ai_batch.py",
        "command": [PYTHON_EXECUTABLE, "generate_ai_batch.py"],
        "prerequisite_folders": [],
    },
    {
        "id": "step4",
        "name": "Prepare Human Samples",
        "description": "Set up IAM Handwriting Database",
        "script": "prepare_human_samples.py",
        "command": [PYTHON_EXECUTABLE, "prepare_human_samples.py"],
        "prerequisite_folders": [],
    },
    {
        "id": "step5",
        "name": "Build Hybrid Pages",
        "description": "Create hybrid pages with ground truth",
        "script": "build_hybrid_pages.py",
        "command": [PYTHON_EXECUTABLE, "build_hybrid_pages.py", "--pages-per-ratio", "4"],
        "prerequisite_folders": ["ai_batch", "human_samples"],
    },
    {
        "id": "step6",
        "name": "Region Segmentation",
        "description": "Segment pages into regions",
        "script": "region_segmentation.py",
        "command": [PYTHON_EXECUTABLE, "region_segmentation.py", "--visualize"],
        "prerequisite_folders": ["hybrid_pages"],
    },
    {
        "id": "step7",
        "name": "Train Region Classifier",
        "description": "Train EfficientNet-B0 on region crops",
        "script": "train_region_classifier.py",
        "command": [PYTHON_EXECUTABLE, "train_region_classifier.py", "--epochs", "20", "--batch-size", "8"],
        "is_training": True,
        "prerequisite_folders": ["region_dataset"],
    },
    {
        "id": "step8",
        "name": "Localization Maps",
        "description": "Generate page-level localization maps",
        "script": "localization_map.py",
        "command": [PYTHON_EXECUTABLE, "localization_map.py", "--split", "test"],
        "prerequisite_folders": ["checkpoints"],
    },
    {
        "id": "step9",
        "name": "Baseline Comparison",
        "description": "Train baseline and compare methods",
        "script": "compare_methods.py",
        "command": [PYTHON_EXECUTABLE, "compare_methods.py", "--split", "test"],
        "is_training": True,
        "prerequisite_folders": ["checkpoints"],
    },
]

# Global state for tracking step status
step_status: Dict[str, Dict] = {
    step["id"]: {
        "status": "pending",  # pending, running, completed, failed
        "output": "",
        "error": None,
        "start_time": None,
        "end_time": None,
    }
    for step in STEPS
}

running_processes: Dict[str, subprocess.Popen] = {}
running_threads: Dict[str, threading.Thread] = {}


def check_prerequisites(step_id: str) -> Dict[str, bool]:
    """Check if prerequisite folders exist and are non-empty."""
    step = next(s for s in STEPS if s["id"] == step_id)
    prereq_status = {}
    
    for folder in step.get("prerequisite_folders", []):
        folder_path = PROJECT_ROOT / folder
        exists = folder_path.exists()
        non_empty = exists and any(folder_path.iterdir())
        prereq_status[folder] = non_empty
    
    return prereq_status


def are_prerequisites_met(step_id: str) -> bool:
    """Check if all prerequisites are met."""
    prereq_status = check_prerequisites(step_id)
    return all(prereq_status.values())


def run_step(step_id: str):
    """Run a step in a background thread."""
    step = next(s for s in STEPS if s["id"] == step_id)
    status = step_status[step_id]
    
    status["status"] = "running"
    status["start_time"] = time.time()
    status["output"] = ""
    status["error"] = None
    
    if step["command"] is None:
        # Manual step
        status["status"] = "completed"
        status["output"] = "This step requires manual execution. See project documentation."
        status["end_time"] = time.time()
        return
    
    try:
        # Run subprocess with proper environment (pitfall #2)
        env = os.environ.copy()
        process = subprocess.Popen(
            step["command"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        running_processes[step_id] = process
        
        # Stream output
        for line in process.stdout:
            status["output"] += line
        
        process.wait()
        
        if process.returncode == 0:
            status["status"] = "completed"
        else:
            status["status"] = "failed"
            status["error"] = f"Process exited with code {process.returncode}"
        
        status["end_time"] = time.time()
        
    except Exception as e:
        status["status"] = "failed"
        status["error"] = str(e)
        status["end_time"] = time.time()
    
    finally:
        if step_id in running_processes:
            del running_processes[step_id]


def run_all_automated_steps():
    """Run all automated pipeline steps sequentially in the background."""
    automated_step_ids = [step["id"] for step in STEPS if step["id"] not in {"step1", "step2"} and step["command"] is not None]

    for step_id in automated_step_ids:
        if step_status[step_id]["status"] == "completed":
            continue

        run_step(step_id)

        if step_status[step_id]["status"] == "failed":
            break


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/inference")
def inference():
    """Inference/Demo page."""
    return render_template("inference.html")


@app.route("/api/steps")
def get_steps():
    """Get current status of all steps including prerequisite status."""
    steps_with_prereqs = []
    for step in STEPS:
        prereq_status = check_prerequisites(step["id"])
        prereqs_met = are_prerequisites_met(step["id"])
        
        steps_with_prereqs.append({
            **step,
            "prerequisites_met": prereqs_met,
            "prerequisite_status": prereq_status,
        })
    
    return jsonify({
        "steps": steps_with_prereqs,
        "status": step_status,
    })


@app.route("/api/run/<step_id>", methods=["POST"])
def run_step_api(step_id: str):
    """Run a specific step."""
    if step_id not in step_status:
        return jsonify({"error": "Invalid step ID"}), 400
    
    status = step_status[step_id]
    
    if status["status"] == "running":
        return jsonify({"error": "Step already running"}), 400
    
    # Run in background thread
    thread = threading.Thread(target=run_step, args=(step_id,))
    thread.start()
    running_threads[step_id] = thread
    
    return jsonify({"status": "started"})


@app.route("/api/stop/<step_id>", methods=["POST"])
def stop_step(step_id: str):
    """Stop a running step."""
    if step_id not in running_processes:
        return jsonify({"error": "Step not running"}), 400
    
    process = running_processes[step_id]
    process.terminate()
    
    step_status[step_id]["status"] = "failed"
    step_status[step_id]["error"] = "Stopped by user"
    step_status[step_id]["end_time"] = time.time()
    
    del running_processes[step_id]
    
    return jsonify({"status": "stopped"})


@app.route("/api/output/<step_id>")
def stream_output(step_id: str):
    """Stream output from a running step."""
    if step_id not in step_status:
        return jsonify({"error": "Invalid step ID"}), 400
    
    status = step_status[step_id]
    
    def generate():
        while status["status"] == "running":
            yield f"data: {json.dumps({'output': status['output']})}\n\n"
            time.sleep(0.5)
        
        # Final update
        yield f"data: {json.dumps({'output': status['output'], 'status': status['status'], 'error': status['error']})}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/reset/<step_id>", methods=["POST"])
def reset_step(step_id: str):
    """Reset a step's status to pending."""
    if step_id not in step_status:
        return jsonify({"error": "Invalid step ID"}), 400
    
    step_status[step_id] = {
        "status": "pending",
        "output": "",
        "error": None,
        "start_time": None,
        "end_time": None,
    }
    
    return jsonify({"status": "reset"})


@app.route("/api/run-all", methods=["POST"])
def run_all_api():
    """Trigger the full automated pipeline from the web UI."""
    if any(step_status[step_id]["status"] == "running" for step_id in step_status):
        return jsonify({"error": "A pipeline step is already running"}), 400

    thread = threading.Thread(target=run_all_automated_steps, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


# ============================================================================
# INFERENCE ENDPOINTS
# ============================================================================

LABEL_MAP_INV = {0: "human", 1: "ai"}


def create_model(num_classes: int = 2) -> nn.Module:
    """Create EfficientNet-B0 model."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_transform() -> transforms.Compose:
    """Get image transform for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@app.route("/api/inference/upload", methods=["POST"])
def inference_upload():
    """Upload and process a handwriting image for inference."""
    # Import modules locally to avoid startup failures
    import region_segmentation
    import localization_map
    
    # Check if checkpoint exists
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        return jsonify({
            "error": "Model checkpoint not found. Please train the region classifier first (Step 7)."
        }), 400
    
    # Check if file was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file type
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Invalid file type. Please upload PNG or JPG."}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Create inference results directory
        INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded image
        upload_path = INFERENCE_DIR / f"{job_id}_original.png"
        file.save(str(upload_path))
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(num_classes=2)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        
        transform = get_transform()
        
        # Load and segment image
        img = region_segmentation.load_page_image(upload_path)
        projection = region_segmentation.horizontal_projection(img)
        boundaries = region_segmentation.find_line_boundaries(projection)
        regions = region_segmentation.extract_regions(img, boundaries)

        mode = "region_level"
        if not regions:
            regions = [{
                "region_id": 0,
                "bbox": {
                    "x": 0,
                    "y": 0,
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                },
                "crop": img,
            }]
            mode = "whole_image_fallback"
        
        # Predict on regions
        with torch.no_grad():
            for region in regions:
                crop_img = Image.fromarray(region["crop"]).convert("RGB")
                input_tensor = transform(crop_img).unsqueeze(0).to(device)
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
                
                region["prediction"] = LABEL_MAP_INV[pred_class]
                region["confidence"] = confidence
        
        # Stitch mask for compatibility with the existing workflow
        localization_map.stitch_mask(regions, img.shape, label_to_mask="ai")
        
        # Create visualization with color-coded boxes
        original_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        vis_img = Image.fromarray(original_rgb)
        draw = ImageDraw.Draw(vis_img)
        font = ImageFont.load_default()
        
        ai_count = 0
        human_count = 0
        
        for region in regions:
            bbox = region["bbox"]
            rect = [
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
            ]
            
            if region["prediction"] == "ai":
                color = (255, 0, 0)  # Red for AI
                ai_count += 1
            else:
                color = (0, 255, 0)  # Green for human
                human_count += 1
            
            draw.rectangle(rect, outline=color, width=3)

            if mode == "whole_image_fallback":
                label = f"{region['prediction'].upper()} {region['confidence'] * 100:.1f}%"
                text_position = (max(4, bbox["x"] + 4), max(4, bbox["y"] + 4))
                draw.text(text_position, label, fill=color, font=font)
        
        # Save visualization
        vis_path = INFERENCE_DIR / f"{job_id}_result.png"
        vis_img.save(vis_path)
        
        # Calculate overall verdict
        total_regions = len(regions)
        ai_ratio = ai_count / total_regions if total_regions > 0 else 0
        
        if ai_ratio > 0.5:
            overall_verdict = "AI-generated (majority)"
            verdict = "AI-generated"
        elif ai_ratio > 0:
            overall_verdict = "Hybrid (mixed)"
            verdict = "Hybrid"
        else:
            overall_verdict = "Human-generated"
            verdict = "Human-written"
        
        # Prepare region data for response
        regions_data = []
        for region in regions:
            regions_data.append({
                "region_id": region["region_id"],
                "prediction": region["prediction"],
                "confidence": round(region["confidence"], 4),
                "bbox": region["bbox"],
            })
        
        return jsonify({
            "job_id": job_id,
            "mode": mode,
            "result_image_url": f"/static/inference_results/{job_id}_result.png",
            "overall_verdict": overall_verdict,
            "verdict": verdict,
            "confidence": round(max(region["confidence"] for region in regions), 4),
            "ai_ratio": round(ai_ratio, 4),
            "total_regions": total_regions,
            "ai_regions": ai_count,
            "human_regions": human_count,
            "regions": regions_data,
        })
        
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@app.route("/static/inference_results/<filename>")
def serve_inference_result(filename):
    """Serve inference result images."""
    return send_from_directory(str(INFERENCE_DIR), filename)


if __name__ == "__main__":
    threading.Thread(target=run_all_automated_steps, daemon=True).start()
    app.run(debug=True, host="127.0.0.1", port=5000)
