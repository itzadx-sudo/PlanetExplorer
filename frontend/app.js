let currentAnalysisData = null;
let currentFileName = '';
let allDataRows = [];
let currentRowIndex = 0;
const loadingStages = [
    "Initializing models...",
    "Loading Kepler dataset...",
    "Calculating orbital parameters...",
    "Finalizing results..."
];
let currentStageIndex = 0;

function createParticles() {
    const background = document.getElementById('animatedBg');
    const particleCount = 50;

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        const size = Math.random() * 3 + 1;
        const startX = Math.random() * window.innerWidth;
        const duration = Math.random() * 25 + 15;
        const delay = Math.random() * 15;
        const colors = ['0, 212, 255', '123, 97, 255', '255, 107, 157'];
        const randomColor = colors[Math.floor(Math.random() * colors.length)];

        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        particle.style.left = startX + 'px';
        particle.style.background = `rgba(${randomColor}, 0.6)`;
        particle.style.color = `rgba(${randomColor}, 0.8)`;
        particle.style.animationDuration = duration + 's';
        particle.style.animationDelay = delay + 's';
        background.appendChild(particle);
    }
}

async function analyzeDataset() {
    showLoading();

    try {
        const response = await fetch('/predict-dataset');

        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
            throw new Error(err.error || `HTTP error ${response.status}`);
        }

        const jsonData = await response.json();

        if (jsonData.success) {
            allDataRows = jsonData.predictions.map(pred => ({
                exoplanet_detected: pred.prediction === "CONFIRMED",
                confidence: pred.confidence,
                prediction_label: pred.prediction,
                confidence_level: pred.confidence_level,
                margin: pred.margin,
                row: pred.row,
                timestamp: new Date().toISOString()
            }));

            currentRowIndex = 0;
            currentFileName = jsonData.dataset || 'kepler_test.csv';

            setTimeout(() => {
                displayResults(allDataRows[currentRowIndex], currentFileName);
            }, 3000);
        } else {
            throw new Error(jsonData.error || 'Analysis failed');
        }

    } catch (error) {
        hideLoading();
        showError(error.message);
        console.error('Error details:', error);
    }
}

function displayResults(data, fileName) {
    currentAnalysisData = data;
    hideLoading();

    document.getElementById('resultsFilename').textContent = fileName;

    const statusBadge = document.getElementById('statusBadge');
    const statusText = document.getElementById('statusText');

    if (data.exoplanet_detected) {
        statusBadge.className = 'status-badge detected';
        statusText.textContent = 'Exoplanet Detected';
    } else {
        statusBadge.className = 'status-badge not-detected';
        statusText.textContent = 'No Exoplanet Detected';
    }

    const confidence = Math.round(data.confidence * 100);
    document.getElementById('confidenceValue').textContent = confidence + '%';

    const circumference = 2 * Math.PI * 90;
    const offset = circumference - (confidence / 100) * circumference;
    document.getElementById('confidenceCircle').style.strokeDashoffset = offset;

    if (allDataRows.length > 0) {
        const indicator = document.getElementById('rowIndicator');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');

        if (indicator) indicator.textContent = `Row ${currentRowIndex + 1} of ${allDataRows.length}`;
        if (prevBtn) prevBtn.disabled = currentRowIndex === 0;
        if (nextBtn) nextBtn.disabled = currentRowIndex === allDataRows.length - 1;
    }

    showSection('resultsSection');
}

function showLoading() {
    document.getElementById('loadingScreen').classList.add('active');
    currentStageIndex = 0;
    updateLoadingStage();
}

function hideLoading() {
    document.getElementById('loadingScreen').classList.remove('active');
}

function updateLoadingStage() {
    const stageElement = document.getElementById('loadingStage');
    if (currentStageIndex < loadingStages.length) {
        stageElement.textContent = loadingStages[currentStageIndex];
        currentStageIndex++;
        setTimeout(updateLoadingStage, 1200);
    }
}

function nextRow() {
    if (currentRowIndex < allDataRows.length - 1) {
        currentRowIndex++;
        displayResults(allDataRows[currentRowIndex], currentFileName);
    }
}

function previousRow() {
    if (currentRowIndex > 0) {
        currentRowIndex--;
        displayResults(allDataRows[currentRowIndex], currentFileName);
    }
}

function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function analyzeAnother() {
    showSection('homeSection');
}

function showError(message) {
    document.getElementById('errorText').textContent = message;
    document.getElementById('errorBanner').classList.add('active');
}

function closeError() {
    document.getElementById('errorBanner').classList.remove('active');
}

function downloadReport() {
    if (!currentAnalysisData) {
        alert('No analysis data available');
        return;
    }

    const reportContent = `
PLANETEXPLORER - EXOPLANET ANALYSIS REPORT
============================================
Generated: ${new Date().toLocaleString()}
Dataset: ${currentFileName}

DETECTION STATUS
================
Exoplanet Detected: ${currentAnalysisData.exoplanet_detected ? 'YES' : 'NO'}
Prediction: ${currentAnalysisData.prediction_label}
Confidence: ${(currentAnalysisData.confidence * 100).toFixed(2)}%
Confidence Level: ${currentAnalysisData.confidence_level}
Margin: ${(currentAnalysisData.margin * 100).toFixed(2)}%
Row Index: ${currentAnalysisData.row}
Timestamp: ${currentAnalysisData.timestamp || 'N/A'}

============================================
Report generated by PlanetExplorer
NASA Space Apps Challenge 2025
============================================
`;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `planetexplorer_report_row${currentAnalysisData.row}_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function viewRawData() {
    if (!currentAnalysisData) {
        alert('No analysis data available');
        return;
    }

    const jsonDisplay = document.getElementById('jsonDisplay');
    jsonDisplay.innerHTML = syntaxHighlight(JSON.stringify(currentAnalysisData, null, 2));
    document.getElementById('rawDataModal').classList.add('active');
}

function closeRawDataModal() {
    document.getElementById('rawDataModal').classList.remove('active');
}

function syntaxHighlight(json) {
    const getClass = (match) => {
        if (!/^"/.test(match)) {
            if (/true|false/.test(match)) return 'json-boolean';
            if (/null/.test(match)) return 'json-null';
            return 'json-number';
        }
        return /:$/.test(match) ? 'json-key' : 'json-string';
    };
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        (match) => `<span class="${getClass(match)}">${match}</span>`
    );
}

document.getElementById('rawDataModal').addEventListener('click', (e) => {
    if (e.target.id === 'rawDataModal') closeRawDataModal();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeRawDataModal();
});

window.addEventListener('DOMContentLoaded', createParticles);
