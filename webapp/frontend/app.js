// HAS-Forensics Web App - Frontend JavaScript

const API_BASE = 'http://127.0.0.1:5000/api';
let eventSources = {};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadSteps();
    loadResults();
    loadHistory();
    
    // Poll for updates every 3 seconds
    setInterval(() => {
        loadSteps();
        loadResults();
        loadHistory();
    }, 3000);
});

// Load steps and their status
async function loadSteps() {
    try {
        const response = await fetch(`${API_BASE}/steps`);
        const data = await response.json();
        
        renderSteps(data.steps, data.status);
        updateStats(data.status);
    } catch (error) {
        console.error('Failed to load steps:', error);
    }
}

// Render step cards
function renderSteps(steps, status) {
    const grid = document.getElementById('steps-grid');
    grid.innerHTML = '';
    
    steps.forEach((step, index) => {
        const stepStatus = status[step.id];
        const card = document.createElement('div');
        card.className = 'step-card';
        card.id = `card-${step.id}`;
        
        const isManual = step.manual || false;
        const isRunning = stepStatus.status === 'running';
        
        card.innerHTML = `
            <div class="step-header">
                <div class="step-number">${index + 1}</div>
                <div class="step-status status-${stepStatus.status}" id="status-${step.id}">
                    ${stepStatus.status}
                </div>
            </div>
            <div class="step-name">${step.name}</div>
            <div class="step-description">${step.description}</div>
            
            <div class="step-actions">
                ${isManual ? 
                    '<button class="btn btn-primary" disabled>Manual</button>' :
                    `<button class="btn btn-primary" onclick="runStep('${step.id}')" id="btn-run-${step.id}" ${isRunning ? 'style="display:none;"' : ''}>
                        Run
                    </button>
                `}
                ${!isManual ? `
                    <button class="btn btn-danger" onclick="stopStep('${step.id}')" id="btn-stop-${step.id}" ${isRunning ? '' : 'style="display:none;"'}>
                        Stop
                    </button>
                ` : ''}
                <button class="btn btn-secondary" onclick="resetStep('${step.id}')">
                    Reset
                </button>
            </div>

            <div class="progress-bar" id="progress-${step.id}" ${isRunning ? '' : 'style="display:none;"'}>
                <div class="progress-fill running"></div>
            </div>

            <div class="output-panel" id="output-${step.id}" ${stepStatus.status === 'completed' || stepStatus.status === 'failed' ? '' : 'style="display:none;"'}>
                <div class="output-text" id="output-text-${step.id}">${escapeHtml(stepStatus.output)}</div>
                ${stepStatus.error ? `<div class="error-text">Error: ${escapeHtml(stepStatus.error)}</div>` : ''}
            </div>
        `;
        
        grid.appendChild(card);
        
        // Start streaming if running
        if (isRunning && !eventSources[step.id]) {
            startStreaming(step.id);
        }
    });
}

// Update statistics
function updateStats(status) {
    let completed = 0;
    let running = 0;
    let pending = 0;
    let failed = 0;
    
    for (const stepId in status) {
        const s = status[stepId].status;
        if (s === 'completed') completed++;
        else if (s === 'running') running++;
        else if (s === 'pending') pending++;
        else if (s === 'failed') failed++;
    }
    
    document.getElementById('stat-completed').textContent = completed;
    document.getElementById('stat-running').textContent = running;
    document.getElementById('stat-pending').textContent = pending;
    document.getElementById('stat-failed').textContent = failed;
}

// Run a step
async function runStep(stepId) {
    try {
        const response = await fetch(`${API_BASE}/run/${stepId}`, { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'started') {
            updateStepUI(stepId, 'running');
            startStreaming(stepId);
        }
    } catch (error) {
        console.error('Failed to run step:', error);
        alert('Failed to run step. Check console for details.');
    }
}

// Stop a step
async function stopStep(stepId) {
    try {
        const response = await fetch(`${API_BASE}/stop/${stepId}`, { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'stopped') {
            updateStepUI(stepId, 'failed');
            if (eventSources[stepId]) {
                eventSources[stepId].close();
                delete eventSources[stepId];
            }
        }
    } catch (error) {
        console.error('Failed to stop step:', error);
    }
}

// Reset a step
async function resetStep(stepId) {
    try {
        const response = await fetch(`${API_BASE}/reset/${stepId}`, { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'reset') {
            updateStepUI(stepId, 'pending');
            const outputText = document.getElementById(`output-text-${stepId}`);
            if (outputText) outputText.textContent = '';
            const outputPanel = document.getElementById(`output-${stepId}`);
            if (outputPanel) outputPanel.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to reset step:', error);
    }
}

// Update step UI
function updateStepUI(stepId, status) {
    const statusEl = document.getElementById(`status-${stepId}`);
    const runBtn = document.getElementById(`btn-run-${stepId}`);
    const stopBtn = document.getElementById(`btn-stop-${stepId}`);
    const progressEl = document.getElementById(`progress-${stepId}`);
    const outputEl = document.getElementById(`output-${stepId}`);
    
    if (statusEl) {
        statusEl.className = `step-status status-${status}`;
        statusEl.textContent = status;
    }
    
    if (runBtn) {
        runBtn.style.display = status === 'running' ? 'none' : 'inline-flex';
    }
    
    if (stopBtn) {
        stopBtn.style.display = status === 'running' ? 'inline-flex' : 'none';
    }
    
    if (progressEl) {
        progressEl.style.display = status === 'running' ? 'block' : 'none';
    }
    
    if (outputEl) {
        outputEl.style.display = (status === 'completed' || status === 'failed') ? 'block' : 'none';
    }
}

// Start streaming output
function startStreaming(stepId) {
    const eventSource = new EventSource(`${API_BASE}/output/${stepId}`);
    eventSources[stepId] = eventSource;
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        const outputText = document.getElementById(`output-text-${stepId}`);
        
        if (outputText && data.output) {
            outputText.textContent = data.output;
            // Auto-scroll to bottom
            const outputPanel = document.getElementById(`output-${stepId}`);
            if (outputPanel) {
                outputPanel.scrollTop = outputPanel.scrollHeight;
            }
        }
        
        if (data.status) {
            updateStepUI(stepId, data.status);
            eventSource.close();
            delete eventSources[stepId];
            
            if (data.error) {
                const outputPanel = document.getElementById(`output-${stepId}`);
                if (outputPanel) {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error-text';
                    errorDiv.textContent = `Error: ${data.error}`;
                    outputPanel.appendChild(errorDiv);
                }
            }
            
            // Reload results after completion
            if (data.status === 'completed') {
                setTimeout(() => {
                    loadResults();
                }, 1000);
            }
        }
    };
    
    eventSource.onerror = function() {
        eventSource.close();
        delete eventSources[stepId];
    };
}

// Load results
async function loadResults() {
    try {
        const response = await fetch(`${API_BASE}/results`);
        const data = await response.json();
        
        renderResults(data);
    } catch (error) {
        console.error('Failed to load results:', error);
    }
}

// Render results
function renderResults(data) {
    // Classifier results
    const classifierEl = document.getElementById('result-classifier');
    if (data.classifier) {
        classifierEl.innerHTML = `
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>${(data.classifier.accuracy * 100).toFixed(1)}%</td></tr>
                <tr><td>Precision</td><td>${(data.classifier.precision * 100).toFixed(1)}%</td></tr>
                <tr><td>Recall</td><td>${(data.classifier.recall * 100).toFixed(1)}%</td></tr>
                <tr><td>F1 Score</td><td>${(data.classifier.f1 * 100).toFixed(1)}%</td></tr>
            </table>
        `;
        classifierEl.previousElementSibling.querySelector('.badge').className = 'badge badge-success';
        classifierEl.previousElementSibling.querySelector('.badge').textContent = 'Ready';
    }
    
    // Localization results
    const localizationEl = document.getElementById('result-localization');
    if (data.localization) {
        localizationEl.innerHTML = `
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Mean IoU</td><td>${(data.localization.mean_iou * 100).toFixed(1)}%</td></tr>
                <tr><td>Mean Pixel F1</td><td>${(data.localization.mean_f1 * 100).toFixed(1)}%</td></tr>
                <tr><td>IoU Std</td><td>${(data.localization.iou_std * 100).toFixed(2)}%</td></tr>
                <tr><td>F1 Std</td><td>${(data.localization.f1_std * 100).toFixed(2)}%</td></tr>
            </table>
        `;
        localizationEl.previousElementSibling.querySelector('.badge').className = 'badge badge-success';
        localizationEl.previousElementSibling.querySelector('.badge').textContent = 'Ready';
    }
    
    // Comparison results
    const comparisonEl = document.getElementById('result-comparison');
    if (data.comparison && data.comparison.methods) {
        const region = data.comparison.methods.region_level;
        const baseline = data.comparison.methods.whole_page_baseline;
        
        comparisonEl.innerHTML = `
            <table>
                <tr><th>Metric</th><th>Region-Level</th><th>Baseline</th></tr>
                <tr><td>Accuracy</td><td>${(region.test_accuracy * 100).toFixed(1)}%</td><td>${(baseline.test_accuracy * 100).toFixed(1)}%</td></tr>
                <tr><td>F1 Score</td><td>${(region.test_f1 * 100).toFixed(1)}%</td><td>${(baseline.test_f1 * 100).toFixed(1)}%</td></tr>
                <tr><td>IoU</td><td>${region.localization_iou ? (region.localization_iou * 100).toFixed(1) + '%' : 'N/A'}</td><td>N/A</td></tr>
                <tr><td>Pixel F1</td><td>${region.localization_pixel_f1 ? (region.localization_pixel_f1 * 100).toFixed(1) + '%' : 'N/A'}</td><td>N/A</td></tr>
                <tr><td>Inference Time</td><td>${region.mean_inference_time_sec.toFixed(3)}s</td><td>${baseline.mean_inference_time_sec.toFixed(3)}s</td></tr>
            </table>
        `;
        comparisonEl.previousElementSibling.querySelector('.badge').className = 'badge badge-success';
        comparisonEl.previousElementSibling.querySelector('.badge').textContent = 'Ready';
    }
}

// Load history
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/history`);
        const data = await response.json();
        
        renderHistory(data.history);
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// Render history
function renderHistory(history) {
    const list = document.getElementById('history-list');
    
    if (history.length === 0) {
        list.innerHTML = '<p class="placeholder">No runs yet</p>';
        return;
    }
    
    list.innerHTML = history.map(item => `
        <div class="history-item">
            <div>
                <div class="history-step">${item.step_name}</div>
                <div class="history-time">${new Date(item.timestamp).toLocaleString()}</div>
            </div>
            <div class="history-status ${item.status}">${item.status}</div>
        </div>
    `).join('');
}

// Download results
async function downloadResults(fileType) {
    try {
        showLoading();
        
        const response = await fetch(`${API_BASE}/download/${fileType}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `has_forensics_${fileType}.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            alert('Failed to download results');
        }
    } catch (error) {
        console.error('Failed to download:', error);
        alert('Failed to download results');
    } finally {
        hideLoading();
    }
}

// Show loading overlay
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
