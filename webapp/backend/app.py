"""
HAS-Forensics Web Application Backend

Flask server providing API endpoints for the 9-step pipeline.
Preserves existing logic - wraps CLI scripts as background tasks.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, request, send_file, Response

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

app = Flask(__name__)

# Step definitions
STEPS = [
    {
        "id": "step1",
        "name": "Environment Setup",
        "description": "Create venv and install dependencies",
        "script": None,
        "command": None,
        "manual": True,
    },
    {
        "id": "step2",
        "name": "Clone Handwriting Generator",
        "description": "Clone and verify pytorch-handwriting-synthesis-toolkit",
        "script": None,
        "command": None,
        "manual": True,
    },
    {
        "id": "step3",
        "name": "Generate AI Batch",
        "description": "Generate 150-300 AI handwriting line images",
        "script": "generate_ai_batch.py",
        "command": ["python", "generate_ai_batch.py"],
    },
    {
        "id": "step4",
        "name": "Prepare Human Samples",
        "description": "Set up IAM Handwriting Database",
        "script": "prepare_human_samples.py",
        "command": ["python", "prepare_human_samples.py"],
    },
    {
        "id": "step5",
        "name": "Build Hybrid Pages",
        "description": "Create hybrid pages with ground truth",
        "script": "build_hybrid_pages.py",
        "command": ["python", "build_hybrid_pages.py", "--pages-per-ratio", "4"],
    },
    {
        "id": "step6",
        "name": "Region Segmentation",
        "description": "Segment pages into regions",
        "script": "region_segmentation.py",
        "command": ["python", "region_segmentation.py", "--visualize"],
    },
    {
        "id": "step7",
        "name": "Train Region Classifier",
        "description": "Train EfficientNet-B0 on region crops",
        "script": "train_region_classifier.py",
        "command": ["python", "train_region_classifier.py", "--epochs", "20", "--batch-size", "8"],
        "is_training": True,
    },
    {
        "id": "step8",
        "name": "Localization Maps",
        "description": "Generate page-level localization maps",
        "script": "localization_map.py",
        "command": ["python", "localization_map.py", "--split", "test"],
    },
    {
        "id": "step9",
        "name": "Baseline Comparison",
        "description": "Train baseline and compare methods",
        "script": "compare_methods.py",
        "command": ["python", "compare_methods.py", "--split", "test"],
        "is_training": True,
    },
]

# Global state
step_status: Dict[str, Dict] = {
    step["id"]: {
        "status": "pending",
        "output": "",
        "error": None,
        "start_time": None,
        "end_time": None,
        "last_updated": None,
    }
    for step in STEPS
}

running_processes: Dict[str, subprocess.Popen] = {}
running_threads: Dict[str, threading.Thread] = {}
run_history: list = []


def run_step(step_id: str):
    """Run a step in a background thread."""
    step = next(s for s in STEPS if s["id"] == step_id)
    status = step_status[step_id]
    
    status["status"] = "running"
    status["start_time"] = time.time()
    status["output"] = ""
    status["error"] = None
    status["last_updated"] = datetime.now().isoformat()
    
    if step.get("manual", False):
        status["status"] = "completed"
        status["output"] = "This step requires manual execution. See project documentation."
        status["end_time"] = time.time()
        status["last_updated"] = datetime.now().isoformat()
        return
    
    try:
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
        
        for line in process.stdout:
            status["output"] += line
            status["last_updated"] = datetime.now().isoformat()
        
        process.wait()
        
        if process.returncode == 0:
            status["status"] = "completed"
            # Add to history
            run_history.append({
                "step_id": step_id,
                "step_name": step["name"],
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "duration": time.time() - status["start_time"],
            })
        else:
            status["status"] = "failed"
            status["error"] = f"Process exited with code {process.returncode}"
            run_history.append({
                "step_id": step_id,
                "step_name": step["name"],
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": status["error"],
                "duration": time.time() - status["start_time"],
            })
        
        status["end_time"] = time.time()
        status["last_updated"] = datetime.now().isoformat()
        
    except Exception as e:
        status["status"] = "failed"
        status["error"] = str(e)
        status["end_time"] = time.time()
        status["last_updated"] = datetime.now().isoformat()
        run_history.append({
            "step_id": step_id,
            "step_name": step["name"],
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e),
            "duration": time.time() - status["start_time"],
        })
    
    finally:
        if step_id in running_processes:
            del running_processes[step_id]


# API Routes

@app.route("/api/steps")
def get_steps():
    """Get all steps and their status."""
    return jsonify({
        "steps": STEPS,
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
    step_status[step_id]["last_updated"] = datetime.now().isoformat()
    
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
            yield f"data: {json.dumps({'output': status['output'], 'last_updated': status['last_updated']})}\n\n"
            time.sleep(0.5)
        
        yield f"data: {json.dumps({'output': status['output'], 'status': status['status'], 'error': status['error'], 'last_updated': status['last_updated']})}\n\n"
    
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
        "last_updated": datetime.now().isoformat(),
    }
    
    return jsonify({"status": "reset"})


@app.route("/api/history")
def get_history():
    """Get run history."""
    return jsonify({"history": run_history})


@app.route("/api/results")
def get_results():
    """Get all available results."""
    results = {}
    
    # Check for classifier results
    classifier_results = PROJECT_ROOT / "checkpoints" / "test_results.json"
    if classifier_results.exists():
        results["classifier"] = json.loads(classifier_results.read_text(encoding="utf-8"))
    
    # Check for localization results
    localization_results = PROJECT_ROOT / "localization_output" / "test_localization_results.json"
    if localization_results.exists():
        results["localization"] = json.loads(localization_results.read_text(encoding="utf-8"))
    
    # Check for comparison results
    comparison_results = PROJECT_ROOT / "comparison_results" / "method_comparison.json"
    if comparison_results.exists():
        results["comparison"] = json.loads(comparison_results.read_text(encoding="utf-8"))
    
    return jsonify(results)


@app.route("/api/download/<file_type>", methods=["GET"])
def download_results(file_type: str):
    """Download results as ZIP file."""
    zip_path = PROJECT_ROOT / f"results_{file_type}.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        if file_type == "all":
            # Add all results
            for dir_name in ["checkpoints", "localization_output", "comparison_results", "segmentation_output"]:
                dir_path = PROJECT_ROOT / dir_name
                if dir_path.exists():
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file():
                            zipf.write(file_path, file_path.relative_to(PROJECT_ROOT))
        elif file_type == "localization":
            # Only localization outputs
            dir_path = PROJECT_ROOT / "localization_output"
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(PROJECT_ROOT))
    
    return send_file(zip_path, as_attachment=True, download_name=f"has_forensics_{file_type}.zip")


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
