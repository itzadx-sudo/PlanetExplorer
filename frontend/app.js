let currentAnalysisData = null;
let currentFileName = '';
let allDataRows = [];
let filteredRows = [];
let currentRowIndex = 0;
let activeFilter = 'all';

const loadingStages = [
    "Initializing AI models...",
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
        const colors = ['0, 212, 255', '123, 97, 255', '255, 107, 157'];
        const c = colors[Math.floor(Math.random() * colors.length)];
        particle.style.cssText = `
            width:${size}px; height:${size}px;
            left:${Math.random() * window.innerWidth}px;
            background:rgba(${c},0.6);
            color:rgba(${c},0.8);
            animation-duration:${Math.random() * 25 + 15}s;
            animation-delay:${Math.random() * 15}s;
        `;
        background.appendChild(particle);
    }
}

/* ── API call ─────────────────────────────────────────────── */
async function analyzeDataset() {
    showLoading();
    try {
        const response = await fetch('/predict-dataset');
        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
            throw new Error(err.error || `HTTP error ${response.status}`);
        }
        const json = await response.json();
        if (!json.success) throw new Error(json.error || 'Analysis failed');

        allDataRows = json.predictions.map(pred => ({
            exoplanet_detected: pred.prediction === 'CONFIRMED',
            confidence: pred.confidence,
            prediction_label: pred.prediction,
            confidence_level: pred.confidence_level,
            margin: pred.margin,
            row: pred.row,
            timestamp: new Date().toISOString()
        }));

        currentFileName = json.dataset || 'kepler_test.csv';
        activeFilter = 'all';
        filteredRows = [...allDataRows];
        currentRowIndex = 0;

        updateSummaryCounts();
        setFilterPillActive('all');

        setTimeout(() => displayResults(filteredRows[0], currentFileName), 3000);

    } catch (error) {
        hideLoading();
        showError(error.message);
        console.error('Error:', error);
    }
}

/* ── Display a single row ─────────────────────────────────── */
function displayResults(data, fileName) {
    if (!data) return;
    currentAnalysisData = data;
    hideLoading();

    /* filename */
    document.getElementById('resultsFilename').textContent = fileName;

    /* confidence circle */
    const pct = Math.round(data.confidence * 100);
    document.getElementById('confidenceValue').textContent = pct + '%';
    const circumference = 2 * Math.PI * 90;
    document.getElementById('confidenceCircle').style.strokeDashoffset =
        circumference - (pct / 100) * circumference;

    /* confidence level badge */
    const lvlBadge = document.getElementById('confidenceLevelBadge');
    if (lvlBadge) {
        const lvl = data.confidence_level || 'Medium';
        lvlBadge.textContent = lvl;
        lvlBadge.className = 'confidence-level-badge ' + levelClass(lvl);
    }

    /* prediction badge */
    const predBadge = document.getElementById('predictionBadge');
    if (predBadge) {
        const pred = data.prediction_label || '—';
        predBadge.textContent = pred;
        predBadge.className = 'prediction-badge ' + predClass(pred);
    }

    /* meta cells */
    setText('marginValue',    data.margin     != null ? (data.margin * 100).toFixed(1) + '%' : '—');
    setText('levelValue',     data.confidence_level || '—');
    setText('rowValue',       data.row != null ? `#${data.row}` : '—');
    setText('timestampValue', data.timestamp
        ? new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        : '—');

    /* description */
    const descEl = document.getElementById('predictionDescription');
    if (descEl) descEl.textContent = predDescription(data.prediction_label);

    /* navigation */
    const total = filteredRows.length;
    setText('rowIndicator', `${currentRowIndex + 1} / ${total}`);
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    if (prevBtn) prevBtn.disabled = currentRowIndex === 0;
    if (nextBtn) nextBtn.disabled = currentRowIndex === total - 1;

    showSection('resultsSection');
}

/* ── Summary counts (run once after data loads) ───────────── */
function updateSummaryCounts() {
    const confirmed  = allDataRows.filter(r => r.prediction_label === 'CONFIRMED').length;
    const candidate  = allDataRows.filter(r => r.prediction_label === 'CANDIDATE').length;
    const fp         = allDataRows.filter(r => r.prediction_label === 'FALSE POSITIVE').length;

    setText('confirmedCount', confirmed);
    setText('candidateCount', candidate);
    setText('fpCount',        fp);
    setText('totalCount',     allDataRows.length);
}

/* ── Filtering ────────────────────────────────────────────── */
function setFilter(filter) {
    activeFilter = filter;
    filteredRows = filter === 'all'
        ? [...allDataRows]
        : allDataRows.filter(r => r.prediction_label === filter);

    currentRowIndex = 0;
    setFilterPillActive(filter);

    if (filteredRows.length > 0) {
        displayResults(filteredRows[0], currentFileName);
    } else {
        setText('rowIndicator', '0 / 0');
    }
}

function setFilterPillActive(filter) {
    document.querySelectorAll('.filter-pill').forEach(pill => {
        pill.classList.remove('active');
        if (pill.dataset.filter === filter) pill.classList.add('active');
    });
}

/* ── Navigation ───────────────────────────────────────────── */
function nextRow() {
    if (currentRowIndex < filteredRows.length - 1) {
        currentRowIndex++;
        displayResults(filteredRows[currentRowIndex], currentFileName);
    }
}

function previousRow() {
    if (currentRowIndex > 0) {
        currentRowIndex--;
        displayResults(filteredRows[currentRowIndex], currentFileName);
    }
}

function jumpToRow() {
    const input = document.getElementById('jumpInput');
    const n = parseInt(input.value, 10);
    if (!n || n < 1 || n > filteredRows.length) {
        showError(`Enter a number between 1 and ${filteredRows.length}`);
        return;
    }
    currentRowIndex = n - 1;
    displayResults(filteredRows[currentRowIndex], currentFileName);
    input.value = '';
}

/* ── Section switching ────────────────────────────────────── */
function showSection(id) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function analyzeAnother() {
    showSection('homeSection');
}

/* ── Loading ──────────────────────────────────────────────── */
function showLoading() {
    document.getElementById('loadingScreen').classList.add('active');
    currentStageIndex = 0;
    updateLoadingStage();
}

function hideLoading() {
    document.getElementById('loadingScreen').classList.remove('active');
}

function updateLoadingStage() {
    const el = document.getElementById('loadingStage');
    if (currentStageIndex < loadingStages.length) {
        el.textContent = loadingStages[currentStageIndex++];
        setTimeout(updateLoadingStage, 1200);
    }
}

/* ── Error banner ─────────────────────────────────────────── */
function showError(msg) {
    document.getElementById('errorText').textContent = msg;
    document.getElementById('errorBanner').classList.add('active');
}

function closeError() {
    document.getElementById('errorBanner').classList.remove('active');
}

/* ── Downloads ────────────────────────────────────────────── */
function downloadReport() {
    if (!currentAnalysisData) { showError('No analysis data available'); return; }
    const d = currentAnalysisData;
    const content = `LUMINA — EXOPLANET ANALYSIS REPORT
NASA Space Apps Challenge 2025
===============================================
Generated : ${new Date().toLocaleString()}
Dataset   : ${currentFileName}

CLASSIFICATION
===============
Prediction       : ${d.prediction_label}
Confidence       : ${(d.confidence * 100).toFixed(2)}%
Confidence Level : ${d.confidence_level}
Margin           : ${d.margin != null ? (d.margin * 100).toFixed(2) + '%' : 'N/A'}
Row Index        : ${d.row}
Analyzed At      : ${d.timestamp ? new Date(d.timestamp).toLocaleString() : 'N/A'}

===============================================
Report generated by Lumina / Five Guys
`;
    downloadText(content, `lumina_report_row${d.row}_${Date.now()}.txt`);
}

function downloadAllReport() {
    if (!allDataRows.length) { showError('No analysis data available'); return; }
    const header = 'row,prediction,confidence_pct,confidence_level,margin_pct\n';
    const rows = allDataRows.map(r =>
        `${r.row},${r.prediction_label},${(r.confidence * 100).toFixed(2)},${r.confidence_level},${r.margin != null ? (r.margin * 100).toFixed(2) : ''}`
    ).join('\n');
    downloadText(header + rows, `lumina_all_predictions_${Date.now()}.csv`);
}

function downloadText(content, filename) {
    const blob = new Blob([content], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement('a'), { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

/* ── Raw data modal ───────────────────────────────────────── */
function viewRawData() {
    if (!currentAnalysisData) { showError('No analysis data available'); return; }
    document.getElementById('jsonDisplay').innerHTML =
        syntaxHighlight(JSON.stringify(currentAnalysisData, null, 2));
    document.getElementById('rawDataModal').classList.add('active');
}

function closeRawDataModal() {
    document.getElementById('rawDataModal').classList.remove('active');
}

function syntaxHighlight(json) {
    const getClass = m => {
        if (!/^"/.test(m)) return /true|false/.test(m) ? 'json-boolean' : /null/.test(m) ? 'json-null' : 'json-number';
        return /:$/.test(m) ? 'json-key' : 'json-string';
    };
    return json
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
            m => `<span class="${getClass(m)}">${m}</span>`);
}

/* ── Helpers ──────────────────────────────────────────────── */
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function levelClass(lvl) {
    return lvl === 'High' ? 'level-high' : lvl === 'Medium' ? 'level-medium' : 'level-low';
}

function predClass(pred) {
    return pred === 'CONFIRMED' ? 'badge-confirmed'
         : pred === 'CANDIDATE' ? 'badge-candidate'
         : 'badge-fp';
}

function predDescription(pred) {
    if (pred === 'CONFIRMED')
        return 'This Kepler Object of Interest has been classified as a confirmed exoplanet with high statistical certainty based on orbital and photometric parameters.';
    if (pred === 'CANDIDATE')
        return 'This KOI is a viable exoplanet candidate. Additional follow-up observations are recommended to confirm its planetary nature.';
    return 'This signal has been classified as a false positive — likely caused by stellar variability, an eclipsing binary, or instrumental artefacts.';
}

/* ── Event listeners ──────────────────────────────────────── */
document.getElementById('rawDataModal').addEventListener('click', e => {
    if (e.target.id === 'rawDataModal') closeRawDataModal();
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeRawDataModal();
    if (e.key === 'ArrowRight' && document.getElementById('resultsSection').classList.contains('active')) nextRow();
    if (e.key === 'ArrowLeft'  && document.getElementById('resultsSection').classList.contains('active')) previousRow();
});

document.getElementById('jumpInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') jumpToRow();
});

window.addEventListener('DOMContentLoaded', createParticles);
