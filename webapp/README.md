# HAS-Forensics Web Application

A modern web interface for the HAS-Forensics pipeline, replacing CLI interaction with a clean browser-based UI.

## Project Structure

```
webapp/
├── backend/
│   └── app.py          # Flask backend with API endpoints
├── frontend/
│   ├── index.html      # Main UI
│   ├── styles.css      # Modern styling
│   └── app.js          # Frontend JavaScript
└── README.md
```

## Features

- **Modern UI**: Clean, responsive interface with step cards
- **Real-time Status**: Live updates via Server-Sent Events (SSE)
- **Run History**: Track all pipeline executions
- **Results Display**: Automatic rendering of metrics and comparisons
- **Download**: Export all results as ZIP file
- **No Terminal**: Complete browser-based interaction

## API Endpoints

- `GET /api/steps` - Get all steps and their status
- `POST /api/run/<step_id>` - Run a specific step
- `POST /api/stop/<step_id>` - Stop a running step
- `GET /api/output/<step_id>` - Stream output from running step
- `POST /api/reset/<step_id>` - Reset step status
- `GET /api/history` - Get run history
- `GET /api/results` - Get all available results
- `GET /api/download/<file_type>` - Download results as ZIP
- `GET /api/health` - Health check

## Installation

1. Install Flask (if not already installed):
```bash
pip install flask
```

2. Navigate to project root:
```bash
cd C:\Users\SATHWIK S RAO\Desktop\HAS-Forensics
```

## Running the Web App

1. Start the Flask backend:
```bash
python webapp/backend/app.py
```

2. Open `webapp/frontend/index.html` in a browser

The app will be available at http://127.0.0.1:5000

## Usage

1. **Run Steps**: Click "Run" on any automated step (3-9)
2. **Monitor Progress**: Watch real-time output in the output panel
3. **Stop Execution**: Click "Stop" to halt a running step
4. **Reset**: Click "Reset" to clear status and output
5. **View Results**: Results automatically appear in the Results section
6. **Download**: Click "Download All Results" to export everything

## Manual Steps

Steps 1-2 (Environment Setup, Clone Generator) are marked as "Manual" and require terminal execution:

```bash
# Step 1: Environment
python -m venv venv
venv\Scripts\activate
pip install torch torchvision numpy pillow opencv-python matplotlib h5py svgwrite requests

# Step 2: Clone generator
git clone https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit.git
cd pytorch-handwriting-synthesis-toolkit
python synthesize.py checkpoints/Epoch_46 "test" -b 1 --samples_dir=samples --trials=1
cd ..
```

## Backend Logic Preservation

The web app preserves all existing CLI logic:
- Same Python scripts are executed via subprocess
- Same command-line arguments are used
- Same output format is captured and displayed
- No modifications to core algorithms

## Design

- **Color Scheme**: Purple gradient header, clean white cards
- **Typography**: System fonts for native feel
- **Responsiveness**: Grid layout adapts to screen size
- **Loading States**: Animated spinners and progress bars
- **Error Handling**: User-friendly error messages
