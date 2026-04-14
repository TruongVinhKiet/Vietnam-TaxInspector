/**
 * fraud.js – Phân tích Rủi ro Gian lận Thuế (v3.0)
 * =====================================================
 * Chế độ 1: Tra cứu đơn lẻ (Single Query)
 * Chế độ 2: Phân tích Lô CSV (Batch Dashboard)
 *
 * v3.0 Upgrades:
 *   - Toast notifications thay thế alert()
 *   - Phân trang 10 dòng/trang
 *   - Biểu đồ Radar (hồ sơ vi phạm)
 *   - Biểu đồ Donut (cơ cấu rủi ro)
 *   - Việt hóa 100%
 */

// ===================================================================
// TOAST NOTIFICATION SYSTEM
// ===================================================================

const TOAST_ICONS = {
    success: 'check_circle', error: 'error', warning: 'warning', info: 'info'
};

function showToast(title, message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="material-symbols-outlined text-xl" style="font-variation-settings:'FILL' 1;">${TOAST_ICONS[type] || 'info'}</span>
        <div class="flex-1">
            <p class="text-sm font-bold">${title}</p>
            <p class="text-xs opacity-80 mt-0.5">${message}</p>
        </div>
        <button onclick="this.parentElement.classList.add('hide');setTimeout(()=>this.parentElement.remove(),400)" class="opacity-60 hover:opacity-100 transition-opacity">
            <span class="material-symbols-outlined text-lg">close</span>
        </button>`;
    container.appendChild(toast);

    // Auto-dismiss
    setTimeout(() => {
        if (toast.parentElement) {
            toast.classList.add('hide');
            setTimeout(() => toast.remove(), 400);
        }
    }, duration);
}


// ===================================================================
// ECHARTS INSTANCE MANAGEMENT (prevents memory leaks)
// ===================================================================

// Registry of all active ECharts instances for proper lifecycle management
const _chartInstances = new Map();

/**
 * Safely initialize an ECharts instance on a container.
 * Disposes any existing instance on the same DOM element first,
 * preventing memory leaks from repeated re-renders.
 */
function safeInitChart(container) {
    if (!container) return null;
    // Dispose existing instance on this DOM node
    const existing = echarts.getInstanceByDom(container);
    if (existing) {
        existing.dispose();
    }
    const chart = echarts.init(container);
    // Track in registry for bulk disposal if needed
    _chartInstances.set(container.id || container, chart);
    return chart;
}

// Centralized resize handler – runs once on window resize for ALL charts
let _resizeRAF = null;
window.addEventListener('resize', () => {
    if (_resizeRAF) cancelAnimationFrame(_resizeRAF);
    _resizeRAF = requestAnimationFrame(() => {
        _chartInstances.forEach((chart) => {
            if (chart && !chart.isDisposed()) chart.resize();
        });
    });
});


// ===================================================================
// TAB SWITCHING
// ===================================================================

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tab}-btn`).classList.add('active');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
}


// ===================================================================
// RISK LEVEL MAPPING (Vietnamese)
// ===================================================================

function getRiskLabel(level) {
    const map = {
        'critical': 'Rủi ro rất cao', 'high': 'Rủi ro cao',
        'medium': 'Trung bình', 'low': 'An toàn'
    };
    return map[level] || level;
}

function getRiskBadgeClass(level) {
    const map = {
        'critical': 'risk-critical', 'high': 'risk-high',
        'medium': 'risk-medium', 'low': 'risk-low'
    };
    return map[level] || 'risk-low';
}


// ===================================================================
// MODE 1: SINGLE QUERY (Real-time)
// ===================================================================

async function checkFraudRisk() {
    const taxCode = document.getElementById('fraud-mst').value.trim();
    if (!taxCode) {
        showToast('Thiếu thông tin', 'Vui lòng nhập Mã số thuế hoặc Tên doanh nghiệp.', 'warning');
        return;
    }

    const btn = document.getElementById('fraud-btn');
    btn.innerHTML = '<div class="loader" style="width:20px;height:20px;border-width:2px"></div> Đang phân tích...';
    btn.disabled = true;

    try {
        const response = await secureFetch(`${API_BASE}/ai/single-query/${taxCode}`, { method: 'POST' });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Không thể phân tích MST này');
        }

        const data = await response.json();
        renderSingleResult(data);
        showToast('Phân tích hoàn tất', `Điểm rủi ro DN ${data.company_name || taxCode}: ${data.risk_score}`, data.risk_score >= 60 ? 'error' : 'success');

    } catch (error) {
        console.error(error);
        showToast('Lỗi phân tích', error.message || 'Có lỗi xảy ra khi gọi API AI.', 'error');
    } finally {
        btn.innerHTML = '<span class="material-symbols-outlined text-[18px]">psychology</span> Phân tích AI';
        btn.disabled = false;
    }
}


function renderSingleResult(data) {
    window._currentSingleData = data;
    const resultDiv = document.getElementById('fraud-result');
    resultDiv.classList.remove('hidden');
    resultDiv.animate([
        { opacity: 0, transform: 'translateY(40px)' },
        { opacity: 1, transform: 'translateY(0)' }
    ], { duration: 800, easing: 'cubic-bezier(0.16, 1, 0.3, 1)', fill: 'both' });

    document.getElementById('result-company-name').textContent = data.company_name || 'Không rõ';
    document.getElementById('result-meta').textContent = `MST: ${data.tax_code} • Ngành: ${data.industry || '---'}`;
    document.getElementById('result-revenue').textContent = formatVND(data.revenue);
    document.getElementById('result-expenses').textContent = formatVND(data.total_expenses);
    document.getElementById('result-year').textContent = data.year || '---';

    animateRiskScore(data.risk_score || 0, data);
    renderRedFlags(data.red_flags || []);
    renderShapExplanation(data.shap_explanation || []);

    // Single Query Charts (Trend + Radar)
    const chartsRow = document.getElementById('single-charts-row');
    if (chartsRow) {
        chartsRow.style.display = 'grid';
        renderSingleTrendChart(data.yearly_history || []);
        renderSingleRadarChart(data);
    }

    // Peer Comparison Chart
    const peerRow = document.getElementById('single-peer-row');
    if (peerRow) {
        peerRow.style.display = 'block';
        renderPeerComparison(data);
    }

    // Model Confidence Badge
    const confRow = document.getElementById('confidence-row');
    const confDisplay = document.getElementById('model-confidence-display');
    if (confRow && confDisplay && data.model_confidence) {
        confRow.style.display = 'block';
        confDisplay.textContent = data.model_confidence + '%';
    }

    // What-If Sandbox
    const whatifBox = document.getElementById('whatif-sandbox');
    if (whatifBox) {
        whatifBox.style.display = 'block';
        window._whatifTaxCode = data.tax_code;
        window._whatifOriginalScore = data.risk_score;
    }
}


function animateRiskScore(targetScore, data) {
    const scoreEl = document.getElementById('risk-score');
    const circleEl = document.getElementById('risk-circle');
    const badgeEl = document.getElementById('risk-badge');
    const recEl = document.getElementById('risk-recommendation');
    const anomalyEl = document.getElementById('anomaly-display');
    const levelEl = document.getElementById('risk-level-display');

    let strokeColor = '#16a34a', badgeClass = 'risk-low', badgeText = 'AN TOÀN';
    let recommendation = 'Doanh nghiệp hoạt động bình thường, không có dấu hiệu bất thường.';

    if (targetScore >= 80) {
        strokeColor = '#dc2626'; badgeClass = 'risk-critical'; badgeText = 'RỦI RO RẤT CAO';
        recommendation = 'Đề xuất đưa vào danh sách kiểm tra trọng điểm và yêu cầu giải trình.';
    } else if (targetScore >= 60) {
        strokeColor = '#ea580c'; badgeClass = 'risk-high'; badgeText = 'RỦI RO CAO';
        recommendation = 'Cần kiểm tra kỹ báo cáo tài chính và hồ sơ hoá đơn đầu vào.';
    } else if (targetScore >= 40) {
        strokeColor = '#eab308'; badgeClass = 'risk-medium'; badgeText = 'TRUNG BÌNH';
        recommendation = 'Theo dõi biến động tài chính trong các quý tới.';
    }

    circleEl.setAttribute('stroke', strokeColor);
    badgeEl.className = `py-2 px-4 rounded-lg inline-block mx-auto text-xs font-black tracking-widest ${badgeClass}`;
    badgeEl.textContent = badgeText;
    recEl.textContent = recommendation;

    anomalyEl.textContent = data.anomaly_score ? (data.anomaly_score * 100).toFixed(1) + '%' : '---';
    levelEl.textContent = getRiskLabel(data.risk_level);
    levelEl.className = `text-sm font-bold ${targetScore >= 60 ? 'text-error' : targetScore >= 40 ? 'text-yellow-600' : 'text-emerald-600'}`;

    scoreEl.textContent = '0';
    // Disable CSS transition to fully control via JS
    circleEl.style.transition = 'none';

    const durationMs = 2000;
    const startTime = performance.now();

    requestAnimationFrame(function animateNumbers(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / durationMs, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        
        // Animate text
        scoreEl.textContent = Math.round(easeOut * targetScore);
        
        // Animate circle ring
        const currentScore = easeOut * targetScore;
        const currentOffset = 552.92 - (552.92 * (currentScore / 100));
        circleEl.style.strokeDashoffset = currentOffset;

        if (progress < 1) requestAnimationFrame(animateNumbers);
        else {
            scoreEl.textContent = Math.round(targetScore);
            circleEl.style.strokeDashoffset = 552.92 - (552.92 * (targetScore / 100));
        }
    });
}


function renderRedFlags(flags) {
    const container = document.getElementById('red-flags-list');
    const countEl = document.getElementById('red-flags-count');
    container.innerHTML = '';
    countEl.textContent = `${flags.length} CẢNH BÁO${flags.length > 0 ? ' NGHIÊM TRỌNG' : ''}`;

    if (flags.length === 0) {
        container.innerHTML = `
            <div class="bg-emerald-50 p-5 rounded-xl border-l-4 border-emerald-500 flex items-start gap-4">
                <span class="material-symbols-outlined text-emerald-600" style="font-variation-settings:'FILL' 1;">check_circle</span>
                <div>
                    <p class="text-sm font-bold text-emerald-800">Không phát hiện dấu hiệu bất thường</p>
                    <p class="text-xs text-slate-600 mt-1">Các chỉ số tài chính nằm trong ngưỡng bình thường.</p>
                </div>
            </div>`;
        return;
    }

    flags.forEach((flag, i) => {
        const div = document.createElement('div');
        div.className = 'group bg-error-container/20 p-5 rounded-xl border-l-4 border-error flex items-start gap-4 transition-all hover:translate-x-1';
        div.style.animationDelay = `${i * 0.1}s`;
        div.innerHTML = `
            <span class="material-symbols-outlined text-error" style="font-variation-settings:'FILL' 1;">${flag.icon || 'warning'}</span>
            <div>
                <p class="text-sm font-bold text-on-error-container">${flag.title}</p>
                <p class="text-xs text-slate-600 mt-1">${flag.description || ''}</p>
            </div>`;
        container.appendChild(div);
    });
}


function renderShapExplanation(shap) {
    const container = document.getElementById('shap-container');
    const barsDiv = document.getElementById('shap-bars');
    if (!shap || shap.length === 0) { container.style.display = 'none'; return; }

    container.style.display = 'block';
    barsDiv.innerHTML = '';
    const maxImp = Math.max(...shap.map(s => Math.abs(s.importance || s.shap_value || 0)));

    shap.slice(0, 6).forEach(s => {
        const imp = Math.abs(s.importance || s.shap_value || 0);
        const pct = maxImp > 0 ? (imp / maxImp * 100) : 0;
        const isRisk = s.direction === 'risk' || s.shap_value > 0;
        const barColor = isRisk ? 'bg-red-400' : 'bg-emerald-400';
        const dirLabel = isRisk ? '▲ Tăng rủi ro' : '▼ Giảm rủi ro';
        const dirColor = isRisk ? 'text-red-500' : 'text-emerald-500';
        const shapVal = s.shap_value !== undefined ? s.shap_value.toFixed(4) : imp.toFixed(4);
        const div = document.createElement('div');
        div.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="text-[10px] font-bold text-slate-500 uppercase tracking-wider">${formatFeatureName(s.feature)}</span>
                <div class="flex items-center gap-2">
                    <span class="text-[9px] font-bold ${dirColor}">${dirLabel}</span>
                    <span class="text-[10px] font-mono text-slate-400">${shapVal}</span>
                </div>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
                <div class="${barColor} h-full rounded-full transition-all duration-1000" style="width: ${pct}%"></div>
            </div>`;
        barsDiv.appendChild(div);
    });
}


// ===================================================================
// MODE 2: BATCH UPLOAD & DASHBOARD
// ===================================================================

let currentBatchId = null;
let pollingInterval = null;

function handleDragOver(e) { e.preventDefault(); e.currentTarget.classList.add('dragover'); }
function handleDragLeave(e) { e.currentTarget.classList.remove('dragover'); }

function handleDrop(e) {
    e.preventDefault(); e.currentTarget.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) uploadCSV(e.dataTransfer.files[0]);
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) uploadCSV(e.target.files[0]);
}


async function uploadCSV(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('Định dạng không hợp lệ', 'Chỉ chấp nhận file CSV (.csv)', 'error');
        return;
    }

    document.getElementById('batch-upload-zone').classList.add('hidden');
    document.getElementById('batch-progress-section').classList.remove('hidden');
    document.getElementById('batch-dashboard').classList.add('hidden');
    document.getElementById('progress-title').textContent = `Đang upload: ${file.name}`;
    document.getElementById('progress-subtitle').textContent = `Kích thước: ${(file.size / 1024 / 1024).toFixed(2)} MB`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await secureFetch(`${API_BASE}/ai/batch-upload`, { method: 'POST', body: formData });
        if (!response.ok) { const err = await response.json(); throw new Error(err.detail || 'Upload thất bại'); }

        const data = await response.json();
        currentBatchId = data.batch_id;
        document.getElementById('progress-title').textContent = 'Hệ thống AI đang xử lý...';
        document.getElementById('progress-subtitle').textContent = `Lô #${data.batch_id} • ${file.name}`;
        showToast('Upload thành công', `File ${file.name} đã được nhận. Đang phân tích...`, 'info');
        startPolling(data.batch_id);

    } catch (error) {
        console.error(error);
        showToast('Lỗi upload', error.message || 'Có lỗi xảy ra khi upload file.', 'error');
        resetBatchUI();
    }
}


function startPolling(batchId) {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(async () => {
        try {
            const response = await secureFetch(`${API_BASE}/ai/batch-status/${batchId}`);
            if (!response.ok) return;
            const status = await response.json();

            const pct = status.progress_percent || 0;
            document.getElementById('progress-bar').style.width = `${pct}%`;
            document.getElementById('progress-percent').textContent = `${pct}%`;
            document.getElementById('progress-detail').textContent = `Đã phân tích ${status.processed_rows || 0} / ${status.total_rows || '?'} doanh nghiệp`;

            if (status.status === 'done') {
                clearInterval(pollingInterval); pollingInterval = null;
                showToast('Phân tích hoàn tất!', `Đã xử lý thành công ${status.processed_rows} doanh nghiệp.`, 'success', 6000);
                loadBatchResults(batchId);
            } else if (status.status === 'failed') {
                clearInterval(pollingInterval); pollingInterval = null;
                showToast('Phân tích thất bại', status.error_message || 'Lỗi không xác định', 'error', 8000);
                resetBatchUI();
            }
        } catch (err) { console.error('Polling error:', err); }
    }, 1500);
}


async function loadBatchResults(batchId) {
    try {
        const response = await secureFetch(`${API_BASE}/ai/batch-results/${batchId}`);
        if (!response.ok) throw new Error('Failed to load results');
        const data = await response.json();

        document.getElementById('batch-progress-section').classList.add('hidden');
        document.getElementById('batch-dashboard').classList.remove('hidden');
        renderBatchDashboard(data);
    } catch (error) {
        console.error(error);
        showToast('Lỗi tải kết quả', 'Không thể tải kết quả phân tích.', 'error');
    }
}


function resetBatchUI() {
    document.getElementById('batch-upload-zone').classList.remove('hidden');
    document.getElementById('batch-progress-section').classList.add('hidden');
    document.getElementById('batch-dashboard').classList.add('hidden');
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-percent').textContent = '0%';
    if (pollingInterval) { clearInterval(pollingInterval); pollingInterval = null; }
}


// ===================================================================
// BATCH DASHBOARD RENDERING (ECharts)
// ===================================================================

let batchData = null;
let sortColumn = 'risk_score';
let sortAsc = false;
let currentPage = 1;
const ROWS_PER_PAGE = 10;
let filteredAssessments = [];

function renderBatchDashboard(data) {
    batchData = data;
    const stats = data.statistics || {};
    const summary = stats.summary || {};

    // Summary cards with counting animation
    animateNumber('stat-records', summary.csv_total_rows || data.total_records || 0);
    animateNumber('stat-total', summary.total_companies || 0);
    animateNumber('stat-critical', summary.critical_count || 0);
    animateNumber('stat-high', summary.high_count || 0);
    animateNumber('stat-avg', summary.avg_risk || 0, true);

    // All charts + new features
    if (stats.year_trend) renderBatchTrendChart(stats.year_trend);
    renderScatterPlot(stats.scatter_data || [], stats.contour_data || null);
    renderHistogram(stats.risk_distribution || []);
    renderRadarChart(stats, data.assessments || []);
    renderDonutChart(summary);
    renderIndustryChart(stats.industry_stats || []);
    renderCorrelationMatrix(stats.correlation_matrix || {});
    renderBoxPlot(stats.box_plot_data || []);
    renderMapChart(stats.province_stats || []);
    // Use AI global feature importance if available, fallback to rule-based
    if (stats.global_feature_importance && stats.global_feature_importance.length > 0) {
        renderGlobalFeatureImportance(stats.global_feature_importance);
    } else {
        renderKeyDrivers(stats.key_drivers || []);
    }

    // Populate industry filter
    populateIndustryFilter(stats.industry_stats || []);

    // Data table with pagination
    filteredAssessments = data.assessments || [];
    currentPage = 1;
    renderPaginatedTable();
}


function animateNumber(elementId, target, isDecimal = false) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const duration = 1500;
    const start = performance.now();
    const startVal = 0;

    requestAnimationFrame(function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        const current = startVal + (target - startVal) * ease;
        el.textContent = isDecimal ? current.toFixed(1) : Math.round(current).toLocaleString();
        if (progress < 1) requestAnimationFrame(tick);
    });
}


// ---- PCA SCATTER PLOT (Clustering 2D) with Contour Overlay ----
function renderScatterPlot(scatterData, contourData) {
    const container = document.getElementById('chart-scatter');
    if (!container || !scatterData || scatterData.length === 0) return;
    const chart = safeInitChart(container);

    const getColor = (score) => {
        if (score >= 80) return '#dc2626';
        if (score >= 60) return '#ea580c';
        if (score >= 40) return '#eab308';
        return '#16a34a';
    };

    const seriesData = scatterData.map(d => ({
        value: [d.pc1, d.pc2],
        name: d.company_name,
        itemStyle: {
            color: getColor(d.risk_score),
            opacity: d.risk_score >= 60 ? 0.9 : 0.5,
            shadowBlur: d.risk_score >= 60 ? 8 : 0,
            shadowColor: getColor(d.risk_score)
        },
        symbolSize: d.risk_score >= 60 ? Math.max(10, d.risk_score / 6) : 6,
        _meta: d,
    }));

    const seriesList = [];

    // Contour heatmap overlay (Isolation Forest decision boundary)
    if (contourData && contourData.x && contourData.y && contourData.z) {
        const heatData = [];
        for (let j = 0; j < contourData.y.length; j++) {
            for (let i = 0; i < contourData.x.length; i++) {
                heatData.push([contourData.x[i], contourData.y[j], contourData.z[j][i]]);
            }
        }
        seriesList.push({
            name: 'Anomaly Density',
            type: 'heatmap',
            data: heatData,
            emphasis: { itemStyle: { borderColor: '#333', borderWidth: 1 } },
            itemStyle: { opacity: 0.35 },
            progressive: 1000,
            z: 0,
        });
    }

    // Scatter points on top
    seriesList.push({
        name: 'Doanh nghiệp',
        type: 'scatter',
        data: seriesData,
        z: 2,
    });

    const option = {
        animationDuration: 2000,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'item',
            formatter: p => {
                if (p.seriesName === 'Anomaly Density') return null;
                const d = p.data._meta;
                if (!d) return '';
                return `<b>${d.company_name}</b><br>`
                    + `MST: ${d.tax_code}<br>`
                    + `Ngành: ${d.industry}<br>`
                    + `Doanh thu: ${formatVND(d.revenue)}<br>`
                    + `Điểm rủi ro: <b style="color:${getColor(d.risk_score)}">${d.risk_score}</b>`;
            }
        },
        grid: { left: '10%', right: '5%', top: '5%', bottom: '20%' },
        xAxis: {
            name: 'PC1 (Trục Chính 1)', nameLocation: 'center', nameGap: 24,
            type: 'value', axisLabel: { fontSize: 9 },
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } },
        },
        yAxis: {
            name: 'PC2 (Trục Chính 2)',
            type: 'value', axisLabel: { fontSize: 9 },
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } },
        },
        series: seriesList,
    };

    // Add visualMap for contour if present
    if (contourData) {
        option.visualMap = [
            {
                show: false, min: 0, max: 1, seriesIndex: 0, calculable: false,
                inRange: {
                    color: ['rgba(22,163,74,0.05)', 'rgba(234,179,8,0.15)', 'rgba(234,88,12,0.25)', 'rgba(220,38,38,0.4)']
                },
            }
        ];
    }

    chart.setOption(option);
}


// ---- BATCH TREND CHART (Yearly) ----
function renderBatchTrendChart(trendData) {
    const container = document.getElementById('chart-batch-trend');
    if (!container || !trendData || trendData.length === 0) {
        if(container) container.innerHTML = '<div class="text-slate-400 text-xs flex items-center justify-center h-full">Không đủ dữ liệu nhiều năm để phân tích xu hướng</div>';
        return;
    }

    const chart = safeInitChart(container);
    const years = trendData.map(d => d.year);
    const avgRisks = trendData.map(d => d.avg_risk);
    const highRiskCounts = trendData.map(d => d.high_risk_count);

    chart.setOption({
        animationDuration: 1500,
        animationEasing: 'cubicOut',
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { data: ['Điểm Rủi ro TB', 'Số DN Rủi ro cao'], bottom: 0, textStyle: { fontSize: 10 } },
        grid: { left: '5%', right: '5%', top: '15%', bottom: '15%' },
        xAxis: { type: 'category', data: years, axisLabel: { fontSize: 10, fontWeight: 'bold' } },
        yAxis: [
            {
                type: 'value', name: 'Điểm Rủi ro TB', position: 'left',
                axisLabel: { color: '#002147', fontWeight: 'bold' },
                splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } }
            },
            {
                type: 'value', name: 'Số DN', position: 'right',
                axisLabel: { color: '#dc2626', fontWeight: 'bold' },
                splitLine: { show: false }
            }
        ],
        series: [
            {
                name: 'Điểm Rủi ro TB', type: 'line', yAxisIndex: 0, data: avgRisks,
                smooth: true, symbol: 'circle', symbolSize: 10,
                lineStyle: { width: 4, color: '#002147' },
                itemStyle: { color: '#002147', borderWidth: 2, borderColor: '#fff' },
                areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [
                    { offset: 0, color: 'rgba(0,33,71,0.2)' },
                    { offset: 1, color: 'rgba(0,33,71,0.01)' }
                ]}}
            },
            {
                name: 'Số DN Rủi ro cao', type: 'line', yAxisIndex: 1, data: highRiskCounts,
                smooth: true, symbol: 'rect', symbolSize: 8,
                lineStyle: { width: 3, color: '#dc2626', type: 'dashed' },
                itemStyle: { color: '#dc2626' }
            }
        ],
    });
}


// ---- HISTOGRAM ----
function renderHistogram(distribution) {
    const chart = safeInitChart(document.getElementById('chart-histogram'));
    chart.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: { left: '12%', right: '5%', top: '10%', bottom: '15%' },
        xAxis: { type: 'category', data: distribution.map(d => d.range), axisLabel: { fontSize: 10, rotate: 30 } },
        yAxis: { type: 'value', name: 'Số DN' },
        series: [{
            type: 'bar',
            data: distribution.map((d, i) => ({
                value: d.count,
                itemStyle: { color: i >= 8 ? '#dc2626' : i >= 6 ? '#ea580c' : i >= 4 ? '#eab308' : '#002147', borderRadius: [4, 4, 0, 0] },
            })),
            barWidth: '60%',
            label: { show: true, position: 'top', fontSize: 10, color: '#44474e' },
        }],
    });

}


// ---- RADAR CHART (Fraud Profile) ----
function renderRadarChart(stats, assessments) {
    const chart = safeInitChart(document.getElementById('chart-radar'));

    // Calculate averages for fraud vs normal groups
    const features = ['f1_divergence', 'f2_ratio_limit', 'f3_vat_structure', 'f4_peer_comparison'];
    const featureLabels = ['F1: Lệch pha\nTăng trưởng', 'F2: Tỷ lệ\nChi phí/DT', 'F3: Cấu trúc\nVAT', 'F4: So sánh\nNgành'];

    const highRisk = assessments.filter(a => a.risk_score >= 60);
    const lowRisk = assessments.filter(a => a.risk_score < 40);

    function avgFeature(group, feat) {
        if (!group.length) return 0;
        const sum = group.reduce((s, a) => s + Math.abs(a[feat] || 0), 0);
        return parseFloat((sum / group.length).toFixed(3));
    }

    const highValues = features.map(f => avgFeature(highRisk, f));
    const lowValues = features.map(f => avgFeature(lowRisk, f));

    // Scale to 0-1 range for radar
    const maxVal = Math.max(...highValues, ...lowValues, 0.01);
    const highScaled = highValues.map(v => parseFloat((v / maxVal).toFixed(3)));
    const lowScaled = lowValues.map(v => parseFloat((v / maxVal).toFixed(3)));

    chart.setOption({
        tooltip: {},
        legend: { data: ['Nhóm Rủi ro cao', 'Nhóm An toàn'], bottom: 0, textStyle: { fontSize: 10 } },
        radar: {
            indicator: featureLabels.map(name => ({ name, max: 1 })),
            shape: 'polygon',
            splitNumber: 4,
            axisName: { fontSize: 10, color: '#44474e' },
            splitArea: { areaStyle: { color: ['rgba(0,33,71,0.02)', 'rgba(0,33,71,0.04)'] } },
        },
        series: [{
            type: 'radar',
            data: [
                {
                    value: highScaled, name: 'Nhóm Rủi ro cao',
                    areaStyle: { color: 'rgba(220,38,38,0.15)' },
                    lineStyle: { color: '#dc2626', width: 2 },
                    itemStyle: { color: '#dc2626' },
                },
                {
                    value: lowScaled, name: 'Nhóm An toàn',
                    areaStyle: { color: 'rgba(0,33,71,0.08)' },
                    lineStyle: { color: '#002147', width: 2 },
                    itemStyle: { color: '#002147' },
                },
            ],
        }],
    });

}


// ---- DONUT CHART (Risk Distribution) ----
function renderDonutChart(summary) {
    const chart = safeInitChart(document.getElementById('chart-donut'));
    const legendContainer = document.getElementById('donut-legend');

    const data = [
        { value: summary.critical_count || 0, name: 'Rủi ro rất cao', color: '#dc2626', icon: 'gpp_bad' },
        { value: summary.high_count || 0, name: 'Rủi ro cao', color: '#ea580c', icon: 'warning' },
        { value: summary.medium_count || 0, name: 'Trung bình', color: '#eab308', icon: 'remove_moderator' },
        { value: summary.low_count || 0, name: 'An toàn', color: '#16a34a', icon: 'verified_user' },
    ];
    const total = data.reduce((s, d) => s + d.value, 0) || 1;

    chart.setOption({
        tooltip: { trigger: 'item', formatter: p => `<b>${p.name}</b><br>${p.value} DN (${p.percent}%)` },
        series: [{
            type: 'pie', radius: ['55%', '82%'], center: ['50%', '50%'],
            avoidLabelOverlap: false, padAngle: 3,
            itemStyle: { borderRadius: 6 },
            label: {
                show: true, position: 'center',
                formatter: () => `{big|${total}}\n{small|Tổng DN}`,
                rich: {
                    big: { fontSize: 28, fontWeight: 800, color: '#002147', lineHeight: 36 },
                    small: { fontSize: 10, color: '#999', lineHeight: 16 },
                },
            },
            emphasis: { label: { show: true, fontSize: 14, fontWeight: 'bold' } },
            data: data.map(d => ({ ...d, itemStyle: { color: d.color } })),
        }],
    });

    // Custom legend
    legendContainer.innerHTML = '';
    data.forEach(d => {
        const pct = ((d.value / total) * 100).toFixed(1);
        legendContainer.innerHTML += `
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background:${d.color}15">
                    <span class="material-symbols-outlined text-[16px]" style="color:${d.color};font-variation-settings:'FILL' 1;">${d.icon}</span>
                </div>
                <div>
                    <p class="text-xs font-bold text-on-surface">${d.name}</p>
                    <p class="text-[10px] text-slate-400">${d.value.toLocaleString()} DN • ${pct}%</p>
                </div>
            </div>`;
    });


}


// ---- INDUSTRY BAR CHART ----
function renderIndustryChart(industryStats) {
    const chart = safeInitChart(document.getElementById('chart-industry'));
    const sorted = [...industryStats].sort((a, b) => b.avg_risk - a.avg_risk);

    chart.setOption({
        tooltip: { trigger: 'axis', formatter: p => {
            const val = p[0].value !== undefined ? p[0].value : (p[0].data && p[0].data.value ? p[0].data.value : 0);
            return `<b>${p[0].name}</b><br>Điểm TB: ${val.toFixed(1)}<br>Số DN: ${sorted[p[0].dataIndex].company_count}`;
        }},
        grid: { left: '35%', right: '10%', top: '5%', bottom: '5%' },
        xAxis: { type: 'value', max: 100 },
        yAxis: { type: 'category', data: sorted.map(d => d.industry), axisLabel: { fontSize: 10 }, inverse: true },
        series: [{
            type: 'bar',
            data: sorted.map(d => ({
                value: Math.round(d.avg_risk * 100) / 100,
                itemStyle: { color: d.avg_risk >= 60 ? '#dc2626' : d.avg_risk >= 40 ? '#eab308' : '#002147', borderRadius: [0, 4, 4, 0] },
            })),
            barWidth: '50%',
            label: { show: true, position: 'right', fontSize: 10 },
        }],
    });

}


// ---- CORRELATION HEATMAP ----
function renderCorrelationMatrix(corrData) {
    const chart = safeInitChart(document.getElementById('chart-correlation'));
    if (!corrData.columns || !corrData.values) {
        chart.setOption({ title: { text: 'Không đủ dữ liệu', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } } });
        return;
    }

    const cols = corrData.columns.map(c => formatFeatureName(c));
    const data = [];
    corrData.values.forEach((row, i) => { row.forEach((val, j) => { data.push([j, i, parseFloat(val.toFixed(2))]); }); });

    chart.setOption({
        tooltip: { formatter: p => `${cols[p.data[0]]} × ${cols[p.data[1]]}<br>r = <b>${p.data[2]}</b>` },
        grid: { left: '22%', right: '15%', top: '5%', bottom: '22%' },
        xAxis: { type: 'category', data: cols, axisLabel: { fontSize: 8, rotate: 45 } },
        yAxis: { type: 'category', data: cols, axisLabel: { fontSize: 8 } },
        visualMap: { min: -1, max: 1, calculable: true, orient: 'vertical', right: 0, top: 'center', inRange: { color: ['#002147', '#f8f9fa', '#dc2626'] }, textStyle: { fontSize: 10 } },
        series: [{ type: 'heatmap', data, label: { show: true, fontSize: 9 }, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } } }],
    });

}


// ===================================================================
// DATA TABLE WITH PAGINATION
// ===================================================================

function populateIndustryFilter(industryStats) {
    const select = document.getElementById('table-industry-filter');
    select.innerHTML = '<option value="">Tất cả ngành</option>';
    industryStats.forEach(ind => {
        select.innerHTML += `<option value="${ind.industry}">${ind.industry} (${ind.company_count})</option>`;
    });
}


function filterAndPaginate() {
    if (!batchData || !batchData.assessments) return;

    const searchTerm = (document.getElementById('table-search').value || '').toLowerCase();
    const industryFilter = document.getElementById('table-industry-filter').value;

    filteredAssessments = batchData.assessments.filter(a => {
        const matchSearch = !searchTerm || `${a.tax_code} ${a.company_name}`.toLowerCase().includes(searchTerm);
        const matchIndustry = !industryFilter || a.industry === industryFilter;
        return matchSearch && matchIndustry;
    });

    currentPage = 1;
    renderPaginatedTable();
}


function renderPaginatedTable() {
    const tbody = document.getElementById('results-table-body');
    const infoEl = document.getElementById('pagination-info');
    const controlsEl = document.getElementById('pagination-controls');

    const total = filteredAssessments.length;
    const totalPages = Math.max(1, Math.ceil(total / ROWS_PER_PAGE));
    if (currentPage > totalPages) currentPage = totalPages;

    const startIdx = (currentPage - 1) * ROWS_PER_PAGE;
    const endIdx = Math.min(startIdx + ROWS_PER_PAGE, total);
    const pageData = filteredAssessments.slice(startIdx, endIdx);

    // Render rows
    tbody.innerHTML = '';
    pageData.forEach((a, idx) => {
        const globalIdx = startIdx + idx + 1;
        const levelClass = getRiskBadgeClass(a.risk_level);
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-surface-container-low/50 transition-colors cursor-pointer';
        tr.innerHTML = `
            <td class="px-4 py-3 font-mono text-slate-400">${globalIdx}</td>
            <td class="px-4 py-3 font-mono font-bold">${a.tax_code}</td>
            <td class="px-4 py-3 font-medium max-w-[200px] truncate">${a.company_name || '---'}</td>
            <td class="px-4 py-3 text-slate-500">${a.industry || '---'}</td>
            <td class="px-4 py-3 font-mono">${formatVND(a.revenue)}</td>
            <td class="px-4 py-3 font-black text-lg">${a.risk_score.toFixed(1)}</td>
            <td class="px-4 py-3 font-mono text-slate-500">${(a.anomaly_score || 0).toFixed(3)}</td>
            <td class="px-4 py-3">
                <span class="px-2 py-1 rounded text-[9px] font-black uppercase tracking-wider ${levelClass}">
                    ${getRiskLabel(a.risk_level)}
                </span>
            </td>`;
        tbody.appendChild(tr);
    });

    // Pagination info
    infoEl.textContent = total > 0
        ? `Hiển thị ${startIdx + 1} – ${endIdx} / ${total.toLocaleString()} doanh nghiệp`
        : 'Không có kết quả phù hợp';

    // Pagination controls
    controlsEl.innerHTML = '';
    const btnBase = 'px-3 py-1.5 text-[10px] font-bold rounded-lg transition-all';
    const btnActive = `${btnBase} bg-primary-container text-white shadow-md`;
    const btnInactive = `${btnBase} bg-surface-container text-on-surface-variant hover:bg-surface-container-high`;
    const btnDisabled = `${btnBase} bg-surface-container-low text-slate-300 cursor-not-allowed`;

    // Previous
    controlsEl.innerHTML += `<button class="${currentPage === 1 ? btnDisabled : btnInactive}" ${currentPage === 1 ? 'disabled' : ''} onclick="goToPage(${currentPage - 1})">
        <span class="material-symbols-outlined text-[14px]">chevron_left</span>
    </button>`;

    // Page numbers (show max 7 buttons)
    let startPage = Math.max(1, currentPage - 3);
    let endPage = Math.min(totalPages, startPage + 6);
    if (endPage - startPage < 6) startPage = Math.max(1, endPage - 6);

    if (startPage > 1) {
        controlsEl.innerHTML += `<button class="${btnInactive}" onclick="goToPage(1)">1</button>`;
        if (startPage > 2) controlsEl.innerHTML += `<span class="px-1 text-slate-300">...</span>`;
    }

    for (let i = startPage; i <= endPage; i++) {
        controlsEl.innerHTML += `<button class="${i === currentPage ? btnActive : btnInactive}" onclick="goToPage(${i})">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) controlsEl.innerHTML += `<span class="px-1 text-slate-300">...</span>`;
        controlsEl.innerHTML += `<button class="${btnInactive}" onclick="goToPage(${totalPages})">${totalPages}</button>`;
    }

    // Next
    controlsEl.innerHTML += `<button class="${currentPage === totalPages ? btnDisabled : btnInactive}" ${currentPage === totalPages ? 'disabled' : ''} onclick="goToPage(${currentPage + 1})">
        <span class="material-symbols-outlined text-[14px]">chevron_right</span>
    </button>`;
}


function goToPage(page) {
    const totalPages = Math.max(1, Math.ceil(filteredAssessments.length / ROWS_PER_PAGE));
    if (page < 1 || page > totalPages) return;
    currentPage = page;
    renderPaginatedTable();
    // Smooth scroll to table top
    document.getElementById('results-table-body').closest('.bg-surface-container-lowest').scrollIntoView({ behavior: 'smooth', block: 'start' });
}


function sortTable(column) {
    if (!batchData || !batchData.assessments) return;
    if (sortColumn === column) { sortAsc = !sortAsc; } else { sortColumn = column; sortAsc = false; }

    filteredAssessments.sort((a, b) => {
        const valA = a[column] || 0, valB = b[column] || 0;
        return sortAsc ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
    });

    currentPage = 1;
    renderPaginatedTable();
}


// ===================================================================
// UTILITIES
// ===================================================================

function formatVND(amount) {
    if (!amount || amount === 0) return '0 đ';
    if (amount >= 1e12) return (amount / 1e12).toFixed(1) + ' Nghìn tỷ';
    if (amount >= 1e9) return (amount / 1e9).toFixed(1) + ' Tỷ';
    if (amount >= 1e6) return (amount / 1e6).toFixed(1) + ' Triệu';
    return amount.toLocaleString() + ' đ';
}


function formatFeatureName(feature) {
    const map = {
        'f1_divergence': 'F1: Lệch pha', 'f2_ratio_limit': 'F2: Tỷ lệ CP/DT',
        'f3_vat_structure': 'F3: Cấu trúc VAT', 'f4_peer_comparison': 'F4: So sánh ngành',
        'revenue_log': 'Log(Doanh thu)', 'expense_log': 'Log(Chi phí)',
        'profit_margin': 'Biên LN', 'revenue_growth_rate': 'Tăng trưởng DT',
        'expense_growth_rate': 'Tăng trưởng CP', 'vat_net_ratio': 'VAT ròng',
    };
    return map[feature] || feature;
}


// ===================================================================
// SINGLE QUERY CHARTS (Trend Line + Individual Radar)
// ===================================================================

function renderSingleTrendChart(yearlyHistory) {
    const container = document.getElementById('chart-single-trend');
    if (!container || !yearlyHistory || yearlyHistory.length === 0) return;

    const chart = safeInitChart(container);
    const years = yearlyHistory.map(d => String(d.year));
    const revenues = yearlyHistory.map(d => d.revenue);
    const expenses = yearlyHistory.map(d => d.total_expenses);

    // Dynamic forecast: next year = max(years) + 1
    const latestYear = Math.max(...years.map(y => parseInt(y)));
    const forecastYear = latestYear + 1;
    let forecastRevenue = null, forecastExpense = null;
    if (revenues.length >= 2) {
        const rGrowth = revenues[revenues.length - 1] / Math.max(revenues[revenues.length - 2], 1);
        const eGrowth = expenses[expenses.length - 1] / Math.max(expenses[expenses.length - 2], 1);
        forecastRevenue = Math.round(revenues[revenues.length - 1] * rGrowth);
        forecastExpense = Math.round(expenses[expenses.length - 1] * eGrowth);
    }

    // Add forecast year
    const allYears = [...years];
    const revActual = [...revenues];
    const expActual = [...expenses];
    const revForecast = new Array(years.length).fill(null);
    const expForecast = new Array(years.length).fill(null);

    if (forecastRevenue !== null) {
        allYears.push(`${forecastYear} (Dự báo)`);
        // Connect forecast line to last actual point
        revForecast[revenues.length - 1] = revenues[revenues.length - 1];
        expForecast[expenses.length - 1] = expenses[expenses.length - 1];
        revActual.push(null);
        expActual.push(null);
        revForecast.push(forecastRevenue);
        expForecast.push(forecastExpense);
    }

    chart.setOption({
        animationDuration: 1500,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'axis',
            formatter: params => {
                let s = `<b>${params[0].axisValue}</b><br>`;
                params.forEach(p => {
                    if (p.value !== null && p.value !== undefined) {
                        s += `${p.marker} ${p.seriesName}: <b>${formatVND(p.value)}</b><br>`;
                    }
                });
                return s;
            }
        },
        legend: {
            data: ['Doanh thu', 'Tổng Chi phí', `DT Dự báo ${forecastYear}`, `CP Dự báo ${forecastYear}`],
            bottom: 0, textStyle: { fontSize: 9 }
        },
        grid: { left: '15%', right: '5%', top: '10%', bottom: '22%' },
        xAxis: { type: 'category', data: allYears, axisLabel: { fontSize: 10 } },
        yAxis: { type: 'value', axisLabel: { fontSize: 9, formatter: v => formatVND(v) } },
        series: [
            {
                name: 'Doanh thu', type: 'line', data: revActual,
                smooth: true, symbol: 'circle', symbolSize: 8,
                lineStyle: { width: 3, color: '#002147' },
                itemStyle: { color: '#002147' },
                areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [
                    { offset: 0, color: 'rgba(0,33,71,0.15)' },
                    { offset: 1, color: 'rgba(0,33,71,0.01)' }
                ]}},
                connectNulls: false,
            },
            {
                name: 'Tổng Chi phí', type: 'line', data: expActual,
                smooth: true, symbol: 'diamond', symbolSize: 8,
                lineStyle: { width: 3, color: '#dc2626', type: 'dashed' },
                itemStyle: { color: '#dc2626' },
                areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [
                    { offset: 0, color: 'rgba(220,38,38,0.1)' },
                    { offset: 1, color: 'rgba(220,38,38,0.01)' }
                ]}},
                connectNulls: false,
            },
            {
                name: `DT Dự báo ${forecastYear}`, type: 'line', data: revForecast,
                smooth: true, symbol: 'emptyCircle', symbolSize: 10,
                lineStyle: { width: 2, color: '#002147', type: 'dotted' },
                itemStyle: { color: '#002147', borderWidth: 2 },
                connectNulls: true,
            },
            {
                name: `CP Dự báo ${forecastYear}`, type: 'line', data: expForecast,
                smooth: true, symbol: 'emptyDiamond', symbolSize: 10,
                lineStyle: { width: 2, color: '#dc2626', type: 'dotted' },
                itemStyle: { color: '#dc2626', borderWidth: 2 },
                connectNulls: true,
            },
        ],
    });

}


function renderSingleRadarChart(data) {
    const container = document.getElementById('chart-single-radar');
    if (!container) return;

    const chart = safeInitChart(container);

    // Safe thresholds (values below these are "normal")
    const safeThresholds = {
        f1_divergence: 0.3,      // |divergence| < 0.3 is safe
        f2_ratio_limit: 0.95,    // ratio < 0.95 is safe
        f3_vat_structure: 0.90,  // vat ratio < 0.90 is safe
        f4_peer_comparison: 0.08 // |comparison| < 0.08 is safe
    };

    // Normalize to 0-1 scale (1 = most dangerous)
    const dnValues = [
        Math.min(Math.abs(data.f1_divergence || 0) / 1.0, 1),           // F1
        Math.min(Math.abs(data.f2_ratio_limit || 0) / 1.05, 1),        // F2
        Math.min(Math.abs(data.f3_vat_structure || 0) / 1.2, 1),       // F3
        Math.min(Math.abs(data.f4_peer_comparison || 0) / 0.3, 1),     // F4
    ];
    const safeValues = [
        safeThresholds.f1_divergence / 1.0,
        safeThresholds.f2_ratio_limit / 1.05,
        safeThresholds.f3_vat_structure / 1.2,
        safeThresholds.f4_peer_comparison / 0.3,
    ];

    chart.setOption({
        animationDuration: 1500,
        animationEasing: 'cubicOut',
        tooltip: {},
        legend: { data: [data.company_name || 'Doanh nghiệp', 'Ngưỡng An toàn'], bottom: 0, textStyle: { fontSize: 9 } },
        radar: {
            indicator: [
                { name: 'F1: Lệch pha\nTăng trưởng', max: 1 },
                { name: 'F2: Tỷ lệ\nChi phí/DT', max: 1 },
                { name: 'F3: Cấu trúc\nVAT', max: 1 },
                { name: 'F4: So sánh\nNgành', max: 1 },
            ],
            shape: 'polygon', splitNumber: 4,
            axisName: { fontSize: 9, color: '#44474e' },
            splitArea: { areaStyle: { color: ['rgba(0,33,71,0.02)', 'rgba(0,33,71,0.05)'] } },
        },
        series: [{
            type: 'radar',
            data: [
                {
                    value: dnValues, name: data.company_name || 'Doanh nghiệp',
                    areaStyle: { color: 'rgba(220,38,38,0.2)' },
                    lineStyle: { color: '#dc2626', width: 2 },
                    itemStyle: { color: '#dc2626' },
                },
                {
                    value: safeValues, name: 'Ngưỡng An toàn',
                    areaStyle: { color: 'rgba(22,163,74,0.1)' },
                    lineStyle: { color: '#16a34a', width: 2, type: 'dashed' },
                    itemStyle: { color: '#16a34a' },
                },
            ],
        }],
    });

}


// ===================================================================
// BATCH CHARTS: BOX PLOT + KEY DRIVERS
// ===================================================================

function renderBoxPlot(boxPlotData) {
    const container = document.getElementById('chart-boxplot');
    if (!container || !boxPlotData || boxPlotData.length === 0) return;

    const chart = safeInitChart(container);
    const industries = boxPlotData.map(d => d.industry);
    const boxData = boxPlotData.map(d => [d.min, d.q1, d.median, d.q3, d.max]);

    // Collect outliers
    const outlierData = [];
    boxPlotData.forEach((d, i) => {
        (d.outliers || []).forEach(v => outlierData.push([i, v]));
    });

    chart.setOption({
        animationDuration: 1500,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'item',
            formatter: p => {
                if (p.seriesIndex === 0) {
                    const d = boxPlotData[p.dataIndex];
                    return `<b>${d.industry}</b> (${d.count} DN)<br>`
                         + `Min: ${d.min.toFixed(1)}<br>Q1: ${d.q1}<br>`
                         + `Trung vị: <b>${d.median}</b><br>Q3: ${d.q3}<br>`
                         + `Max: ${d.max.toFixed(1)}`;
                }
                return `Điểm dị biệt: <b>${p.data[1].toFixed(1)}</b>`;
            }
        },
        grid: { left: '30%', right: '8%', top: '5%', bottom: '8%' },
        xAxis: { type: 'value', name: 'Điểm Rủi Ro', max: 100 },
        yAxis: {
            type: 'category', data: industries,
            axisLabel: { fontSize: 9, width: 120, overflow: 'truncate' },
            inverse: true,
        },
        series: [
            {
                name: 'Phân bổ', type: 'boxplot',
                data: boxData,
                itemStyle: {
                    color: 'rgba(0,33,71,0.08)',
                    borderColor: '#002147', borderWidth: 1.5,
                },
                emphasis: { itemStyle: { borderColor: '#dc2626', borderWidth: 2 } },
            },
            {
                name: 'Điểm dị biệt', type: 'scatter',
                data: outlierData,
                symbolSize: 6,
                itemStyle: { color: '#dc2626', opacity: 0.7 },
            },
        ],
    });

}


// ---- GLOBAL FEATURE IMPORTANCE (XGBoost AI-native) ----
function renderGlobalFeatureImportance(featureData) {
    const container = document.getElementById('chart-keydrivers');
    if (!container || !featureData || featureData.length === 0) return;

    const chart = safeInitChart(container);
    const driverColors = ['#dc2626', '#ea580c', '#eab308', '#002147'];

    chart.setOption({
        animationDuration: 1500,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'axis', axisPointer: { type: 'shadow' },
            formatter: p => {
                const d = featureData[p[0].dataIndex];
                return `<b>${d.label}</b><br>`
                     + `Trọng số (Gain): <b>${d.importance.toFixed(4)}</b><br>`
                     + `Tỷ trọng: <b>${d.importance_pct}%</b>`;
            }
        },
        grid: { left: '5%', right: '15%', top: '5%', bottom: '5%' },
        xAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
        yAxis: {
            type: 'category',
            data: featureData.map(d => d.label),
            axisLabel: { fontSize: 9, width: 130, overflow: 'break' },
            inverse: true,
        },
        series: [{
            type: 'bar',
            data: featureData.map((d, i) => ({
                value: d.importance_pct,
                itemStyle: {
                    color: driverColors[i % driverColors.length],
                    borderRadius: [0, 6, 6, 0],
                },
            })),
            barWidth: '55%',
            label: {
                show: true, position: 'right', fontSize: 10, fontWeight: 'bold',
                formatter: p => `${p.value}%`,
                color: '#1e293b',
            },
        }],
    });

}

// Fallback: rule-based key drivers (kept for backward compatibility)
function renderKeyDrivers(keyDrivers) {
    const container = document.getElementById('chart-keydrivers');
    if (!container || !keyDrivers || keyDrivers.length === 0) return;

    const chart = safeInitChart(container);
    const driverColors = ['#dc2626', '#ea580c', '#eab308', '#002147'];

    chart.setOption({
        animationDuration: 1500,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'axis', axisPointer: { type: 'shadow' },
            formatter: p => {
                const d = keyDrivers[p[0].dataIndex];
                return `<b>${d.label}</b><br>`
                     + `${d.triggered_count} DN vi phạm (${d.triggered_percent}%)<br>`
                     + `Giá trị TB: ${d.avg_value}`;
            }
        },
        grid: { left: '5%', right: '15%', top: '5%', bottom: '5%' },
        xAxis: { type: 'value', max: 100, axisLabel: { formatter: '{value}%' } },
        yAxis: {
            type: 'category',
            data: keyDrivers.map(d => d.label),
            axisLabel: { fontSize: 10, width: 140, overflow: 'break' },
            inverse: true,
        },
        series: [{
            type: 'bar',
            data: keyDrivers.map((d, i) => ({
                value: d.triggered_percent,
                itemStyle: {
                    color: driverColors[i % driverColors.length],
                    borderRadius: [0, 6, 6, 0],
                },
            })),
            barWidth: '50%',
            label: {
                show: true, position: 'right', fontSize: 11, fontWeight: 'bold',
                formatter: p => `${p.value}%`,
            },
        }],
    });

}


// ===================================================================
// SINGLE QUERY: PEER COMPARISON CHART
// ===================================================================

function renderPeerComparison(data) {
    const container = document.getElementById('chart-peer-comparison');
    if (!container) return;

    const chart = safeInitChart(container);

    const profitMargin = data.revenue > 0
        ? ((data.revenue - data.total_expenses) / data.revenue * 100)
        : 0;
    const industryAvg = (data.f4_peer_comparison !== undefined && data.revenue > 0)
        ? (profitMargin - (data.f4_peer_comparison * 100))
        : 10;

    const companyName = data.company_name || 'Doanh nghiệp';
    const isBelow = profitMargin < industryAvg;

    chart.setOption({
        animationDuration: 1500,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'axis', axisPointer: { type: 'shadow' },
            formatter: p => {
                return p.map(item =>
                    `${item.marker} ${item.seriesName}: <b>${item.value.toFixed(1)}%</b>`
                ).join('<br>');
            }
        },
        legend: { data: [companyName, `TB Ngành ${data.industry || ''}`], bottom: 0, textStyle: { fontSize: 10 } },
        grid: { left: '5%', right: '5%', top: '10%', bottom: '22%' },
        xAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
        yAxis: {
            type: 'category',
            data: ['Biên Lợi Nhuận'],
            axisLabel: { fontSize: 11, fontWeight: 'bold' },
        },
        series: [
            {
                name: companyName, type: 'bar',
                data: [{ value: Math.round(profitMargin * 10) / 10 }],
                barWidth: 28,
                itemStyle: {
                    color: isBelow ? '#dc2626' : '#002147',
                    borderRadius: [0, 6, 6, 0],
                },
                label: {
                    show: true, position: 'right', fontSize: 12, fontWeight: 'bold',
                    formatter: p => `${p.value}%`,
                    color: isBelow ? '#dc2626' : '#002147',
                },
            },
            {
                name: `TB Ngành ${data.industry || ''}`, type: 'bar',
                data: [{ value: Math.round(industryAvg * 10) / 10 }],
                barWidth: 28,
                itemStyle: {
                    color: '#16a34a',
                    borderRadius: [0, 6, 6, 0],
                },
                label: {
                    show: true, position: 'right', fontSize: 12, fontWeight: 'bold',
                    formatter: p => `${p.value}%`,
                    color: '#16a34a',
                },
            },
        ],
    });

}


// ===================================================================
// ACTION BUTTONS (Thanh tra)
// ===================================================================

function actionSendNotice() {
    const companyName = document.getElementById('result-company-name')?.textContent || 'Doanh nghiệp';
    const taxCode = document.getElementById('fraud-mst')?.value || '';
    if (!taxCode || companyName === 'Không rõ') {
        showToast('Chưa có doanh nghiệp', 'Vui lòng thực hiện Phân tích AI trước khi gửi giải trình.', 'warning');
        return;
    }
    showToast(
        'Đã gửi Thông báo Giải trình',
        `Yêu cầu giải trình đã được gửi đến ${companyName} (MST: ${taxCode}). Hệ thống sẽ theo dõi phản hồi trong 15 ngày.`,
        'success',
        6000
    );
}

function actionCreateReport() {
    const companyName = document.getElementById('result-company-name')?.textContent || 'Doanh nghiệp';
    const taxCode = document.getElementById('fraud-mst')?.value || '';
    const riskScore = document.getElementById('risk-score')?.textContent || '0';
    if (!taxCode || companyName === 'Không rõ') {
        showToast('Chưa có doanh nghiệp', 'Vui lòng thực hiện Phân tích AI trước khi lập biên bản.', 'warning');
        return;
    }
    showToast(
        'Đã lập Biên bản Vi phạm',
        `Biên bản vi phạm hành chính thuế cho ${companyName} (MST: ${taxCode}) với Điểm rủi ro ${riskScore}/100 đã được tạo thành công.`,
        'success',
        6000
    );
}


// ===================================================================
// WHAT-IF AI SANDBOX (Scenario Simulation)
// ===================================================================

let _whatifDebounceTimer = null;

function onWhatIfChange() {
    const revSlider = document.getElementById('whatif-revenue');
    const expSlider = document.getElementById('whatif-expenses');
    const revLabel = document.getElementById('whatif-revenue-label');
    const expLabel = document.getElementById('whatif-expenses-label');
    if (!revSlider || !expSlider) return;

    const revPct = parseInt(revSlider.value);
    const expPct = parseInt(expSlider.value);

    revLabel.textContent = `${revPct > 0 ? '+' : ''}${revPct}%`;
    expLabel.textContent = `${expPct > 0 ? '+' : ''}${expPct}%`;

    // Color the labels based on direction
    revLabel.className = revPct < 0 ? 'text-red-500 font-bold' : revPct > 0 ? 'text-emerald-600 font-bold' : 'text-blue-600';
    expLabel.className = expPct > 0 ? 'text-red-500 font-bold' : expPct < 0 ? 'text-emerald-600 font-bold' : 'text-blue-600';

    // Debounce API call
    if (_whatifDebounceTimer) clearTimeout(_whatifDebounceTimer);
    if (revPct === 0 && expPct === 0) {
        renderWhatIfIdle();
        return;
    }
    _whatifDebounceTimer = setTimeout(() => runWhatIfSimulation(revPct, expPct), 400);
}


function resetWhatIf() {
    const revSlider = document.getElementById('whatif-revenue');
    const expSlider = document.getElementById('whatif-expenses');
    if (revSlider) revSlider.value = 0;
    if (expSlider) expSlider.value = 0;
    document.getElementById('whatif-revenue-label').textContent = '0%';
    document.getElementById('whatif-revenue-label').className = 'text-blue-600';
    document.getElementById('whatif-expenses-label').textContent = '0%';
    document.getElementById('whatif-expenses-label').className = 'text-blue-600';
    renderWhatIfIdle();
}


function renderWhatIfIdle() {
    const resultDiv = document.getElementById('whatif-result');
    if (!resultDiv) return;
    resultDiv.innerHTML = `
        <div class="flex items-center gap-3">
            <span class="material-symbols-outlined text-slate-300 text-2xl">psychology_alt</span>
            <p class="text-[10px] text-slate-400">Di chuyển thanh trượt để bắt đầu mô phỏng...</p>
        </div>`;
}


async function runWhatIfSimulation(revPct, expPct) {
    const taxCode = window._whatifTaxCode;
    if (!taxCode) return;

    const resultDiv = document.getElementById('whatif-result');
    resultDiv.innerHTML = `
        <div class="flex items-center gap-3">
            <span class="material-symbols-outlined text-blue-400 text-xl animate-spin">autorenew</span>
            <p class="text-[10px] text-blue-500 font-bold">AI đang tính toán kịch bản mới...</p>
        </div>`;

    try {
        const adjustments = {};
        if (revPct !== 0) adjustments.revenue = revPct;
        if (expPct !== 0) adjustments.total_expenses = expPct;

        const response = await secureFetch(`${API_BASE}/ai/what-if/${taxCode}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(adjustments),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Mô phỏng thất bại');
        }

        const data = await response.json();
        renderWhatIfResult(data);

    } catch (error) {
        resultDiv.innerHTML = `
            <div class="flex items-center gap-3">
                <span class="material-symbols-outlined text-red-400 text-xl">error</span>
                <p class="text-[10px] text-red-500">${error.message}</p>
            </div>`;
    }
}


function renderWhatIfResult(data) {
    const resultDiv = document.getElementById('whatif-result');
    if (!resultDiv) return;

    const origScore = data.original_risk_score || window._whatifOriginalScore || 0;
    const simScore = data.simulated_risk_score || 0;
    const delta = data.delta_risk || (simScore - origScore);
    const deltaColor = delta > 0 ? 'text-red-600' : delta < 0 ? 'text-emerald-600' : 'text-slate-500';
    const deltaIcon = delta > 0 ? 'trending_up' : delta < 0 ? 'trending_down' : 'trending_flat';
    const deltaSign = delta > 0 ? '+' : '';

    const getRiskColor = (score) => {
        if (score >= 80) return '#dc2626';
        if (score >= 60) return '#ea580c';
        if (score >= 40) return '#eab308';
        return '#16a34a';
    };

    resultDiv.innerHTML = `
        <div class="flex items-center justify-between w-full gap-4">
            <div class="flex items-center gap-4">
                <div class="text-center">
                    <p class="text-[8px] text-slate-400 uppercase tracking-wider mb-1">Gốc</p>
                    <span class="text-xl font-black" style="color:${getRiskColor(origScore)}">${origScore.toFixed(1)}</span>
                </div>
                <span class="material-symbols-outlined text-slate-300">arrow_forward</span>
                <div class="text-center">
                    <p class="text-[8px] text-slate-400 uppercase tracking-wider mb-1">Mô phỏng</p>
                    <span class="text-xl font-black" style="color:${getRiskColor(simScore)}">${simScore.toFixed(1)}</span>
                </div>
            </div>
            <div class="flex items-center gap-2 px-3 py-2 rounded-lg ${delta > 0 ? 'bg-red-50' : delta < 0 ? 'bg-emerald-50' : 'bg-slate-50'}">
                <span class="material-symbols-outlined ${deltaColor} text-lg">${deltaIcon}</span>
                <span class="text-sm font-black ${deltaColor}">${deltaSign}${delta.toFixed(1)}</span>
            </div>
        </div>`;
}

// ===================================================================
// VIETNAM GEOGRAPHIC RISK HEATMAP (ECharts Map)
// ===================================================================

// Province name mapping: mock data names -> GeoJSON names
const PROVINCE_NAME_MAP = {
    'TP.HCM': 'Hồ Chí Minh city',
    'Hồ Chí Minh': 'Hồ Chí Minh city',
    'Bà Rịa-VT': 'Bà Rịa - Vũng Tàu',
    'Bà Rịa - Vũng Tàu': 'Bà Rịa - Vũng Tàu',
    'Thừa Thiên Huế': 'Thừa Thiên - Huế',
    'Đắk Lắk': 'Đắk Lắk',
    'Đăk Nông': 'Đăk Nông',
    'Hà Nội': 'Hà Nội',
    'Đà Nẵng': 'Đà Nẵng',
    'Hải Phòng': 'Hải Phòng',
    'Cần Thơ': 'Cần Thơ',
    'Bình Dương': 'Bình Dương',
    'Đồng Nai': 'Đồng Nai',
    'Bắc Ninh': 'Bắc Ninh',
    'Quảng Ninh': 'Quảng Ninh',
    'Khánh Hòa': 'Khánh Hòa',
    'Long An': 'Long An',
    'Lâm Đồng': 'Lâm Đồng',
    'Nghệ An': 'Nghệ An',
    'Thanh Hóa': 'Thanh Hóa',
};

let _vietnamGeoLoaded = false;

async function renderMapChart(provinceStats) {
    const container = document.getElementById('chart-geo-map');
    if (!container) return;
    const chart = safeInitChart(container);

    if (!provinceStats || provinceStats.length === 0) {
        chart.setOption({
            title: { text: 'Không có dữ liệu tỉnh thành', left: 'center', top: 'center',
                     textStyle: { color: '#999', fontSize: 14 } }
        });
        return;
    }

    // Load GeoJSON if not already loaded
    if (!_vietnamGeoLoaded) {
        try {
            const resp = await fetch('../json/vietnam.json');
            const geoData = await resp.json();
            echarts.registerMap('vietnam', geoData);
            _vietnamGeoLoaded = true;
        } catch (err) {
            console.error('[GeoMap] Failed to load vietnam.json:', err);
            chart.setOption({
                title: { text: 'Lỗi tải bản đồ GeoJSON', left: 'center', top: 'center',
                         textStyle: { color: '#dc2626', fontSize: 14 } }
            });
            return;
        }
    }

    // Map province stats -> GeoJSON-compatible data array
    const mapData = provinceStats.map(p => {
        const geoName = PROVINCE_NAME_MAP[p.province] || p.province;
        return {
            name: geoName,
            value: Math.round(p.avg_risk * 100) / 100,
            companyCount: p.company_count,
            originalName: p.province,
        };
    });

    const maxRisk = Math.max(...mapData.map(d => d.value), 60);

    chart.setOption({
        title: {
            text: 'BẢN ĐỒ CẢNH BÁO RỦI RO THUẾ VIỆT NAM',
            left: 'center',
            top: 8,
            textStyle: { color: '#002147', fontSize: 15, fontWeight: 900, letterSpacing: 2 },
        },
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(0,33,71,0.92)',
            borderColor: '#465f88',
            textStyle: { color: '#fff', fontSize: 12 },
            formatter: params => {
                if (!params.data || params.data.value === undefined) {
                    return `<b>${params.name}</b><br/>Không có dữ liệu`;
                }
                const d = params.data;
                const riskColor = d.value >= 60 ? '#ff6b6b' : d.value >= 40 ? '#ffd93d' : '#6bcb77';
                return `<div style="min-width:180px">
                    <b style="font-size:13px">${d.originalName || params.name}</b>
                    <hr style="border-color:rgba(255,255,255,.2);margin:6px 0">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span>Điểm rủi ro TB</span>
                        <b style="color:${riskColor}">${d.value.toFixed(1)}</b>
                    </div>
                    <div style="display:flex;justify-content:space-between">
                        <span>Số doanh nghiệp</span>
                        <b>${d.companyCount.toLocaleString()}</b>
                    </div>
                </div>`;
            }
        },
        visualMap: {
            min: 0,
            max: maxRisk,
            left: 16,
            bottom: 20,
            text: ['Rủi ro cao', 'Rủi ro thấp'],
            textStyle: { color: '#44474e', fontSize: 10, fontWeight: 700 },
            inRange: {
                color: ['#e8f5e9', '#fff9c4', '#ffe0b2', '#ffab91', '#ef5350', '#b71c1c']
            },
            calculable: true,
            orient: 'vertical',
            itemWidth: 14,
            itemHeight: 120,
        },
        series: [{
            name: 'Rủi ro Thuế',
            type: 'map',
            map: 'vietnam',
            roam: true,
            zoom: 1.2,
            center: [106.5, 16.5],
            scaleLimit: { min: 0.8, max: 5 },
            label: {
                show: true,
                fontSize: 7,
                color: '#333',
                formatter: p => {
                    // Show abbreviated name for readability
                    const short = p.name.replace(' city', '').replace('Thành phố ', '');
                    return short.length > 8 ? short.substring(0, 7) + '…' : short;
                }
            },
            emphasis: {
                label: { show: true, fontSize: 12, fontWeight: 'bold', color: '#002147' },
                itemStyle: { areaColor: '#aec7f6', shadowBlur: 20, shadowColor: 'rgba(0,33,71,0.4)', borderWidth: 2, borderColor: '#002147' }
            },
            select: {
                label: { show: true, fontSize: 12, fontWeight: 'bold' },
                itemStyle: { areaColor: '#d6e3ff' }
            },
            itemStyle: {
                borderColor: '#fff',
                borderWidth: 1,
                areaColor: '#f5f5f5',
            },
            data: mapData,
            animationDurationUpdate: 800,
            animationEasingUpdate: 'cubicInOut',
        }]
    });
}


// ===================================================================
// PDF / PRINT EXPORT
// ===================================================================

async function exportPDF() {
    const data = window._currentSingleData;
    const modal = document.getElementById('report-preview-modal');
    const paper = document.getElementById('report-paper-content');

    if (!data || !modal || !paper) {
        showToast('Lỗi', 'Không tìm thấy dữ liệu để xuất báo cáo.', 'error');
        return;
    }

    const btn = document.getElementById('btn-export-pdf');
    const oldBtnContent = btn.innerHTML;
    btn.innerHTML = 'Đang tạo báo cáo...';
    btn.disabled = true;

    // Small delay to allow UI to update button
    await new Promise(r => setTimeout(r, 100));

    try {
        // Collect chart images
        const chartImages = { radar: '', trend: '', shap: '' };
        try {
            const radarChart = echarts.getInstanceByDom(document.getElementById('chart-single-radar'));
            if (radarChart) chartImages.radar = radarChart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
            
            const trendChart = echarts.getInstanceByDom(document.getElementById('chart-single-trend'));
            if (trendChart) chartImages.trend = trendChart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
            
            const shapChart = echarts.getInstanceByDom(document.getElementById('chart-single-radar')); // Using radar as placeholder for visual if SHAP chart is not an echarts instance. But wait, SHAP is HTML bars.
            // Since SHAP is standard HTML, we can just print the HTML or skip the image for SHAP. We will skip SHAP image and only show radar + trend.
        } catch (err) {
            console.error('Lỗi khi lấy ảnh biểu đồ', err);
        }

        // Prepare texts
        const riskTextColor = data.risk_score >= 60 ? '#dc2626' : (data.risk_score >= 40 ? '#ea580c' : '#16a34a');
        const anomalyPercent = ((data.anomaly_score || 0)*100).toFixed(1);
        
        let riskConclusion = 'Doanh nghiệp <b>chưa có dấu hiệu rủi ro rõ ràng</b>, đề xuất đưa vào diện <b>Theo Dõi Định Kỳ</b>.';
        if (data.risk_score >= 80) {
            riskConclusion = 'Doanh nghiệp thuộc nhóm <b>RỦI RO RẤT CAO</b>, cấu trúc tài chính có dấu hiệu gian lận nghiêm trọng. Kính đề xuất cấp thẩm quyền đưa vào diện <b>Thanh Tra Đột Xuất</b> và yêu cầu giải trình ngay lập tức.';
        } else if (data.risk_score >= 60) {
            riskConclusion = 'Doanh nghiệp thuộc nhóm <b>RỦI RO CAO</b>. Kính đề xuất <b>Kiểm Tra Hồ Sơ Chuyên Sâu</b> đối với các tờ khai VAT và báo cáo tài chính trong 3 năm gần nhất.';
        } else if (data.risk_score >= 40) {
            riskConclusion = 'Doanh nghiệp có dấu hiệu <b>RỦI RO BẬC TRUNG</b>. Đề xuất tiếp tục <b>Theo Dõi Chặt Chẽ</b> biến động dòng tiền trong năm tài chính tiếp theo.';
        }

        const flagsHtml = (data.red_flags || []).map(f => {
            return `<li><b>Phát hiện bất thường về ${f.feature}:</b> ${f.reason} <i>(Chỉ số hiện tại ghi nhận: ${f.actual_value})</i>. Trí Tuệ Nhân Tạo (XGBoost) đánh giá đây là mắt xích trọng yếu có khả năng liên đới tới hành vi trục lợi thuế.</li>`;
        }).join('');

        const today = new Date();

        const htmlTemplate = `
            <div style="font-family: 'Times New Roman', Times, serif; font-size: 13pt; line-height: 1.5; color: #000;">
                <!-- Header -->
                <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                    <div style="text-align: center; width: 40%;">
                        <div style="font-size: 12pt;">BỘ TÀI CHÍNH</div>
                        <div style="font-weight: bold; font-size: 12pt; border-bottom: 1.5px solid #000; display: inline-block; padding-bottom: 2px;">TỔNG CỤC THUẾ</div>
                        <div style="margin-top: 5px; font-size: 11pt;">Số: ${Math.floor(Math.random() * 900) + 100}/TB-TCT</div>
                    </div>
                    <div style="text-align: center; width: 60%;">
                        <div style="font-weight: bold; font-size: 12pt;">CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM</div>
                        <div style="font-weight: bold; font-size: 13pt; border-bottom: 1.5px solid #000; display: inline-block; padding-bottom: 2px;">Độc lập - Tự do - Hạnh phúc</div>
                        <div style="margin-top: 5px; font-style: italic; font-size: 12pt;">Hà Nội, ngày ${today.getDate()} tháng ${today.getMonth() + 1} năm ${today.getFullYear()}</div>
                    </div>
                </div>

                <!-- Title -->
                <div style="text-align: center; margin-top: 40px; margin-bottom: 25px;">
                    <div style="font-weight: bold; font-size: 14pt; margin-bottom: 5px;">THÔNG BÁO</div>
                    <div style="font-weight: bold; font-size: 13pt;">Về việc cảnh báo rủi ro gian lận thuế và đề nghị giải trình<br>đối với hành vi kê khai tài chính bất thường</div>
                </div>

                <!-- Content -->
                <div style="text-align: justify; text-indent: 30px; margin-bottom: 15px;">
                    Triển khai chuyên đề giám sát trọng điểm đối với các rủi ro hoàn thuế, quản lý hóa đơn và chống thất thu ngân sách nhà nước, Tổng cục Thuế đã đưa vào vận hành Hệ thống Giám sát Tự động TaxInspector có ứng dụng Trí tuệ nhân tạo (Mô hình điểm dị biệt Isolation Forest và Phân lớp Gradient Boosting).
                    Qua phân tích định kỳ bộ dữ liệu tài chính khai báo đến thời điểm hiện tại, Tổng cục Thuế thông báo kết quả giám định rà soát đối với doanh nghiệp có tên sau đây:
                </div>

                <div style="margin-left: 30px; margin-bottom: 15px;">
                    - Ký danh: <b>${data.company_name || 'Không rõ'}</b><br>
                    - Mã số doanh nghiệp (MST): <b>${data.tax_code || '---'}</b><br>
                    - Ngành nghề kê khai: ${data.industry || '---'}<br>
                    - Quy mô doanh thu/năm: ${data.revenue ? data.revenue.toLocaleString() + ' tỷ VNĐ' : '---'}<br>
                    - Thống kê chi phí vận hành: ${data.total_expenses ? data.total_expenses.toLocaleString() + ' tỷ VNĐ' : '---'}
                </div>

                <div style="text-align: justify; text-indent: 30px; margin-bottom: 15px;">
                    Căn cứ vào việc đối soát chuỗi chỉ số cơ bản của đối tượng với hàng nghìn thực thể khác trên cùng hệ sinh thái ngành, hệ thống đặc biệt đưa ra xếp loại rủi ro <b>${getRiskLabel(data.risk_level)}</b> (điểm tổng hợp: <b>${data.risk_score} / 100</b>). 
                    Đồng thời, cấu trúc tài chính phát sinh mức phân tán <b>${anomalyPercent}%</b> so với biên độ an toàn cho phép. 
                </div>

                <div style="text-align: justify; text-indent: 30px; margin-bottom: 10px;">
                    Các luận điểm chuyên sâu từ hệ thống máy học (Red Flags) ghi nhận được:
                </div>
                <ul style="margin-top: 0; padding-left: 60px; margin-bottom: 15px; text-align: justify;">
                    ${flagsHtml || '<li>Mô hình tính toán không ghi nhận đặc trưng rủi ro nào ở mức cảnh báo cao.</li>'}
                </ul>

                <div style="text-align: justify; text-indent: 30px; margin-bottom: 15px;">
                    ${riskConclusion} 
                    Tổng cục Thuế đề nghị các Cơ quan thuế địa phương rà soát lại dữ liệu hóa đơn, hồ sơ khai thuế để kịp thời triển khai các biện pháp nghiệp vụ thanh tra theo đúng quy định pháp luật.
                </div>
                
                <div style="text-align: justify; text-indent: 30px; margin-bottom: 15px;">
                    Tổng cục Thuế thông báo để các cục nắm được và phối hợp thực hiện./.
                </div>

                <!-- Footer Signatures -->
                <div style="display: flex; justify-content: space-between; margin-top: 30px; line-height: 1.2;">
                    <div style="width: 50%;">
                        <div style="font-weight: bold; font-style: italic; font-size: 11pt;">Nơi nhận:</div>
                        <div style="font-size: 10pt; margin-left: -5px;">
                            - PTCTrg Đặng Ngọc Minh (để b/c);<br>
                            - Cục Thuế các Tỉnh/Thành phố trực thuộc TW;<br>
                            - Lưu: VT, CNTT.
                        </div>
                    </div>
                    <div style="width: 50%; text-align: center;">
                        <div style="font-weight: bold; font-size: 12pt;">TL. TỔNG CỤC TRƯỞNG</div>
                        <div style="font-weight: bold; font-size: 12pt;">CỤC TRƯỞNG CỤC CÔNG NGHỆ THÔNG TIN</div>
                        <br><br><br><br><br>
                        <div style="font-weight: bold; font-size: 13pt;">Phạm Quang Toàn</div>
                    </div>
                </div>

                <!-- Page 2 Charts -->
                <div style="page-break-before: always; padding-top: 20px;">
                    <div style="font-weight: bold; font-size: 12pt; margin-bottom: 20px; text-decoration: underline;">PHỤ LỤC: BẰNG CHỨNG HÌNH ẢNH TRỰC QUAN MÔ HÌNH HÓA</div>
                    <div style="display: flex; gap: 20px; justify-content: center; align-items: start;">
                        ${chartImages.trend ? `
                        <div style="flex: 1; text-align: center;">
                            <img src="${chartImages.trend}" alt="Trend Chart" style="max-width: 100%; border: 1px solid #ccc;">
                            <div style="font-style: italic; font-size: 11pt; margin-top: 8px;">Mô phỏng 1: Khẩu độ Doanh thu/Chi phí</div>
                        </div>` : ''}
                        ${chartImages.radar ? `
                        <div style="flex: 1; text-align: center;">
                            <img src="${chartImages.radar}" alt="Radar Chart" style="max-width: 100%; border: 1px solid #ccc;">
                            <div style="font-style: italic; font-size: 11pt; margin-top: 8px;">Mô phỏng 2: Biểu đồ Radar đa góc</div>
                        </div>` : ''}
                    </div>
                </div>
            </div>
        `;
        
        paper.innerHTML = htmlTemplate;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden'; // prevent background scrolling

    } catch (e) {
        console.error("Lỗi xuất PDF", e);
        showToast('Lỗi', 'Không thể tạo báo cáo PDF.', 'error');
    } finally {
        btn.innerHTML = oldBtnContent;
        btn.disabled = false;
    }
}

function closeReportModal() {
    const modal = document.getElementById('report-preview-modal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}
