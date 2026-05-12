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
// SINGLE QUERY: STRATEGIC CHARTS
// ===================================================================

function renderSingleRiskTierSankey(sankeyData) {
    const container = document.getElementById('chart-single-risk-tier-sankey');
    if (!container) return;

    const nodes = Array.isArray(sankeyData.nodes) ? sankeyData.nodes : [];
    const links = Array.isArray(sankeyData.links) ? sankeyData.links : [];

    if (nodes.length < 2 || !links.length) {
        renderSingleTrendChartMessage(container, 'Cần tối thiểu 2 năm dữ liệu để hiển thị luồng chuyển nhóm rủi ro.');
        return;
    }

    const chart = safeInitChart(container);
    const tierLabel = {
        low: 'An toàn',
        medium: 'Trung bình',
        high: 'Rủi ro cao',
        critical: 'Rất cao',
    };
    const tierColor = {
        low: '#16a34a',
        medium: '#eab308',
        high: '#ea580c',
        critical: '#dc2626',
    };

    const formattedNodes = nodes.map((node) => {
        const nodeName = String(node.name || `${node.year || '-'}:${node.tier || 'low'}`);
        const [rawYear, rawTier] = nodeName.split(':');
        const year = Number(node.year || rawYear) || rawYear || '-';
        const tier = String(node.tier || rawTier || 'low');

        return {
            name: nodeName,
            itemStyle: { color: tierColor[tier] || '#64748b' },
            label: {
                formatter: `${year}\n${tierLabel[tier] || tier}`,
                fontSize: 9,
                color: '#334155',
            },
        };
    });

    const formattedLinks = links
        .filter((link) => Number(link.value || 0) > 0)
        .map((link) => ({
            source: String(link.source || ''),
            target: String(link.target || ''),
            value: Number(link.value || 0),
        }));

    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        tooltip: {
            trigger: 'item',
            formatter: (params) => {
                if (params.dataType === 'edge') {
                    return `${params.data.source} -> ${params.data.target}<br><b>${params.data.value} kỳ</b>`;
                }
                return `${params.name}`;
            },
        },
        series: [
            {
                type: 'sankey',
                data: formattedNodes,
                links: formattedLinks,
                nodeWidth: 18,
                nodeGap: 14,
                orient: 'horizontal',
                lineStyle: {
                    color: 'source',
                    curveness: 0.42,
                    opacity: 0.45,
                },
                emphasis: {
                    focus: 'adjacency',
                },
            },
        ],
    });
}


function renderSingleCumulativeRiskCurve(curveData) {
    const container = document.getElementById('chart-single-cumulative-risk');
    if (!container) return;

    const points = Array.isArray(curveData.points) ? [...curveData.points] : [];
    if (!points.length) {
        renderSingleTrendChartMessage(container, 'Chưa có dữ liệu lũy kế rủi ro theo chuỗi năm.');
        return;
    }

    points.sort((a, b) => Number(a.period_count || 0) - Number(b.period_count || 0));
    const seriesData = points.map((p) => [
        Number(p.percent_periods ?? p.percent_companies ?? 0),
        Number(p.percent_risk || 0),
    ]);
    const diagonal = [[0, 0], [100, 100]];

    const chart = safeInitChart(container);

    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        tooltip: {
            trigger: 'axis',
            formatter: (params) => {
                const point = params[0] && params[0].data ? params[0].data : [0, 0];
                return `Top <b>${Number(point[0]).toFixed(1)}%</b> giai đoạn đang chứa <b>${Number(point[1]).toFixed(1)}%</b> tổng rủi ro`;
            },
        },
        legend: {
            data: ['Lũy kế rủi ro', 'Đường cân bằng'],
            bottom: 0,
            textStyle: { fontSize: 9 },
        },
        grid: { left: '12%', right: '6%', top: '8%', bottom: '18%' },
        xAxis: {
            type: 'value',
            min: 0,
            max: 100,
            name: '% Giai đoạn',
            axisLabel: { formatter: '{value}%' },
        },
        yAxis: {
            type: 'value',
            min: 0,
            max: 100,
            name: '% Tổng rủi ro',
            axisLabel: { formatter: '{value}%' },
        },
        series: [
            {
                name: 'Lũy kế rủi ro',
                type: 'line',
                smooth: true,
                data: seriesData,
                lineStyle: { width: 3, color: '#dc2626' },
                itemStyle: { color: '#dc2626' },
                areaStyle: { color: 'rgba(220,38,38,0.12)' },
                markPoint: {
                    symbolSize: 44,
                    data: [
                        {
                            name: 'Top 10%',
                            coord: [10, Number(curveData.top_10pct_risk_share || 0)],
                            value: `${Number(curveData.top_10pct_risk_share || 0).toFixed(1)}%`,
                        },
                        {
                            name: 'Top 20%',
                            coord: [20, Number(curveData.top_20pct_risk_share || 0)],
                            value: `${Number(curveData.top_20pct_risk_share || 0).toFixed(1)}%`,
                        },
                    ],
                    label: { fontSize: 8, color: '#334155' },
                },
            },
            {
                name: 'Đường cân bằng',
                type: 'line',
                data: diagonal,
                lineStyle: { width: 2, color: '#64748b', type: 'dashed' },
                showSymbol: false,
                tooltip: { show: false },
            },
        ],
    });
}


function renderSingleMarginDistribution(distributionData) {
    const container = document.getElementById('chart-single-margin-distribution');
    if (!container) return;

    const summaryEl = document.getElementById('single-margin-distribution-summary');
    const bins = Array.isArray(distributionData.bins) ? distributionData.bins : [];
    if (!bins.length) {
        renderSingleTrendChartMessage(container, 'Chưa có dữ liệu phân phối biên lợi nhuận theo ngành.');
        if (summaryEl) {
            summaryEl.textContent = 'Dữ liệu tham chiếu ngành chưa đủ để tính phân vị biên lợi nhuận.';
        }
        return;
    }

    const labels = bins.map((b) => String(b.label || `${b.start || 0}% đến ${b.end || 0}%`));
    const counts = bins.map((b) => Number(b.count || 0));
    const companyMargin = Number(distributionData.company_margin);
    const percentile = Number(distributionData.percentile);
    const sampleSize = Number(distributionData.sample_size || 0);
    const companyBinIndex = Number.isInteger(distributionData.company_bin_index)
        ? Number(distributionData.company_bin_index)
        : -1;

    if (summaryEl) {
        const marginLabel = Number.isFinite(companyMargin) ? `${companyMargin.toFixed(2)}%` : 'Không có';
        const percentileLabel = Number.isFinite(percentile) ? `${percentile.toFixed(1)}%` : 'Không có';
        summaryEl.textContent = `Biên lợi nhuận DN: ${marginLabel} | Phân vị: ${percentileLabel} | Mẫu so sánh: ${sampleSize.toLocaleString()} DN`;
    }

    const chart = safeInitChart(container);
    const barData = counts.map((count, idx) => ({
        value: count,
        itemStyle: {
            color: idx === companyBinIndex ? '#dc2626' : '#1d4ed8',
            borderRadius: [4, 4, 0, 0],
        },
    }));

    const scatterSeries = companyBinIndex >= 0 && companyBinIndex < counts.length
        ? [{
            name: 'Vị trí DN',
            type: 'scatter',
            data: [[companyBinIndex, counts[companyBinIndex]]],
            symbolSize: 16,
            itemStyle: { color: '#dc2626' },
            tooltip: {
                formatter: () => {
                    const marginLabel = Number.isFinite(companyMargin) ? `${companyMargin.toFixed(2)}%` : 'Không có';
                    const percentileLabel = Number.isFinite(percentile) ? `${percentile.toFixed(1)}%` : 'Không có';
                    return `<b>Biên lợi nhuận DN</b><br>${marginLabel}<br>Phân vị: ${percentileLabel}`;
                },
            },
        }]
        : [];

    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        tooltip: {
            trigger: 'item',
            formatter: (params) => {
                if (params.seriesType === 'scatter') {
                    const marginLabel = Number.isFinite(companyMargin) ? `${companyMargin.toFixed(2)}%` : 'Không có';
                    const percentileLabel = Number.isFinite(percentile) ? `${percentile.toFixed(1)}%` : 'Không có';
                    return `<b>Biên lợi nhuận DN</b><br>${marginLabel}<br>Phân vị: ${percentileLabel}`;
                }
                const idx = Number(params.dataIndex || 0);
                const label = labels[idx] || 'Không có';
                const count = Number(counts[idx] || 0);
                return `<b>${escapeHtml(label)}</b><br>Số DN: <b>${count.toLocaleString()}</b>`;
            },
        },
        grid: { left: '8%', right: '4%', top: '8%', bottom: '22%' },
        xAxis: {
            type: 'category',
            data: labels,
            axisLabel: {
                interval: 0,
                rotate: 26,
                fontSize: 9,
            },
        },
        yAxis: {
            type: 'value',
            name: 'Số DN',
            axisLabel: { fontSize: 10 },
        },
        series: [
            {
                name: 'Phân phối ngành',
                type: 'bar',
                data: barData,
                barWidth: '72%',
            },
            ...scatterSeries,
        ],
    });
}


function renderSingleRedFlagsTimeline(timelineData) {
    const container = document.getElementById('chart-single-redflags-timeline');
    if (!container) return;

    const yearPoints = Array.isArray(timelineData.year_points) ? timelineData.year_points : [];
    if (!yearPoints.length) {
        renderSingleTrendChartMessage(container, 'Chưa có dữ liệu timeline red flags theo năm.');
        return;
    }

    const years = yearPoints.map((p) => String(p.year || '---'));
    const counts = yearPoints.map((p) => Number(p.flag_count || 0));
    const flagByYear = yearPoints.map((p) => Array.isArray(p.flag_ids) ? p.flag_ids : []);

    const chart = safeInitChart(container);
    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        tooltip: {
            trigger: 'axis',
            formatter: (params) => {
                const idx = params[0] ? Number(params[0].dataIndex || 0) : 0;
                const flags = flagByYear[idx] || [];
                const list = flags.length ? flags.map((f) => `• ${escapeHtml(String(f))}`).join('<br>') : 'Không có';
                return `Năm <b>${escapeHtml(years[idx] || '---')}</b><br>Số cờ đỏ: <b>${Number(counts[idx] || 0)}</b><br><br>${list}`;
            },
        },
        grid: { left: '8%', right: '4%', top: '10%', bottom: '12%' },
        xAxis: {
            type: 'category',
            data: years,
            axisLabel: { fontSize: 10, fontWeight: 600 },
        },
        yAxis: {
            type: 'value',
            minInterval: 1,
            name: 'Số cờ đỏ',
            axisLabel: { fontSize: 10 },
        },
        series: [
            {
                name: 'Red flags theo năm',
                type: 'bar',
                data: counts,
                barWidth: '48%',
                itemStyle: {
                    color: '#f97316',
                    borderRadius: [6, 6, 0, 0],
                },
                label: {
                    show: true,
                    position: 'top',
                    fontSize: 10,
                    fontWeight: 'bold',
                },
            },
            {
                name: 'Xu hướng',
                type: 'line',
                data: counts,
                smooth: true,
                symbolSize: 7,
                lineStyle: { width: 2, color: '#dc2626' },
                itemStyle: { color: '#dc2626' },
            },
        ],
    });
}
// ===================================================================

const TOAST_ICONS = {
    success: 'check_circle', error: 'error', warning: 'warning', info: 'info'
};

function readMotionToken(name) {
    try {
        const raw = getComputedStyle(document.documentElement).getPropertyValue(name);
        return (raw || '').toString().trim();
    } catch (_err) {
        return '';
    }
}

function readMotionDurationMs(name, fallbackMs) {
    const raw = readMotionToken(name);
    if (!raw) return fallbackMs;

    if (/^-?\d+(\.\d+)?$/.test(raw)) {
        const parsed = Number(raw);
        return Number.isFinite(parsed) ? parsed : fallbackMs;
    }

    if (raw.endsWith('ms')) {
        const parsed = Number(raw.slice(0, -2));
        return Number.isFinite(parsed) ? parsed : fallbackMs;
    }

    if (raw.endsWith('s')) {
        const parsed = Number(raw.slice(0, -1));
        return Number.isFinite(parsed) ? parsed * 1000 : fallbackMs;
    }

    return fallbackMs;
}

function readMotionEase(fallback) {
    const raw = readMotionToken('--motion-ease-emphasis');
    return raw || fallback;
}

const PREFERS_REDUCED_MOTION = !!(
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
);
const MOTION_EASE_EMPHASIS = readMotionEase('cubic-bezier(0.16, 1, 0.3, 1)');
const MOTION_DURATION_CHART = PREFERS_REDUCED_MOTION ? 0 : readMotionDurationMs('--motion-duration-chart-ms', 900);
const MOTION_DURATION_SCORE = PREFERS_REDUCED_MOTION ? 0 : readMotionDurationMs('--motion-duration-score-ms', 850);
const MOTION_DURATION_RESULT = PREFERS_REDUCED_MOTION ? 0 : readMotionDurationMs('--motion-duration-result-ms', 600);
const MOTION_DURATION_TOAST_OUT = PREFERS_REDUCED_MOTION ? 0 : readMotionDurationMs('--motion-duration-toast-out-ms', 400);

let _fraudPageBindingsInitialized = false;
let _modalLastFocusedElement = null;
let _modalKeydownListener = null;
let _pollingRequestToken = 0;
let _pollingWatchdog = null;
let _singleResultVisibleBeforeLoading = false;
const WHATIF_HEATMAP_DEFAULT_REVENUE_STEPS = [-30, -20, -10, 0, 10, 20, 30];
const WHATIF_HEATMAP_DEFAULT_EXPENSE_STEPS = [30, 20, 10, 0, -10, -20, -30];
let _singleSensitivityHeatmapRequestToken = 0;
const _singleSensitivityHeatmapCache = new Map();

function dismissToast(toast) {
    if (!toast || !toast.parentElement) return;
    toast.classList.add('hide');
    setTimeout(() => {
        if (toast.parentElement) toast.remove();
    }, MOTION_DURATION_TOAST_OUT || 1);
}

function showToast(title, message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
    toast.setAttribute('aria-live', type === 'error' ? 'assertive' : 'polite');

    const icon = document.createElement('span');
    icon.className = 'material-symbols-outlined text-xl';
    icon.style.fontVariationSettings = "'FILL' 1";
    icon.textContent = TOAST_ICONS[type] || 'info';

    const content = document.createElement('div');
    content.className = 'flex-1';

    const titleEl = document.createElement('p');
    titleEl.className = 'text-sm font-bold';
    titleEl.textContent = title || '';

    const messageEl = document.createElement('p');
    messageEl.className = 'text-xs opacity-80 mt-0.5';
    messageEl.textContent = message || '';

    content.appendChild(titleEl);
    content.appendChild(messageEl);

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'opacity-60 hover:opacity-100 transition-opacity';
    closeBtn.setAttribute('aria-label', 'Đóng thông báo');

    const closeIcon = document.createElement('span');
    closeIcon.className = 'material-symbols-outlined text-lg';
    closeIcon.textContent = 'close';
    closeBtn.appendChild(closeIcon);
    closeBtn.addEventListener('click', () => dismissToast(toast));

    toast.appendChild(icon);
    toast.appendChild(content);
    toast.appendChild(closeBtn);
    container.appendChild(toast);

    // Auto-dismiss
    setTimeout(() => {
        dismissToast(toast);
    }, duration);
}


function escapeHtml(value) {
    const str = value === null || value === undefined ? '' : String(value);
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
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
    container.innerHTML = '';
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
    const tabBtn = document.getElementById(`tab-${tab}-btn`);
    if (tabBtn) {
        tabBtn.classList.add('active');
    }
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const tabContent = document.getElementById(`tab-${tab}`);
    if (tabContent) {
        tabContent.classList.add('active');
    }

    // Lazy-load company directory only when opening the dedicated CSDL tab.
    if (tab === 'directory') {
        initSingleCompanyDirectory();
    }
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


function formatSplitGateTimestamp(raw) {
    if (!raw) return '--';
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) return '--';
    return dt.toLocaleString('vi-VN', { hour12: false });
}


function normalizeSplitTriggerStatus(rawStatus) {
    const payload = rawStatus && typeof rawStatus === 'object' ? rawStatus : {};
    const hasPayload = Object.keys(payload).length > 0;

    if (!hasPayload) {
        return {
            ready: false,
            schemaReady: false,
            readinessScore: 0,
            enabledRules: 0,
            passedRules: 0,
            blockedTracks: [],
            reason: 'Chạy Phân tích AI để tải trạng thái split-trigger cho doanh nghiệp này.',
            generatedAt: '',
        };
    }

    const schemaReady = payload.schema_ready === true;
    const ready = schemaReady && payload.ready === true;

    const readinessScoreRaw = Number(payload.readiness_score);
    const readinessScore = Number.isFinite(readinessScoreRaw)
        ? Math.max(0, Math.min(100, readinessScoreRaw))
        : 0;

    const totals = payload.totals && typeof payload.totals === 'object' ? payload.totals : {};
    const enabledRules = Number.isFinite(Number(totals.enabled_rules)) ? Number(totals.enabled_rules) : 0;
    const passedRules = Number.isFinite(Number(totals.passed_rules)) ? Number(totals.passed_rules) : 0;

    const trackStatus = payload.track_status && typeof payload.track_status === 'object' ? payload.track_status : {};
    const blockedTracks = Object.entries(trackStatus)
        .filter(([, value]) => !(value && value.ready_for_split))
        .map(([trackName, value]) => {
            const blockingRuleCount = Number(value && value.blocking_rule_count ? value.blocking_rule_count : 0);
            const normalizedTrack = String(trackName || '').replaceAll('_', ' ').toUpperCase();
            return blockingRuleCount > 0
                ? `${normalizedTrack} (${blockingRuleCount} quy tắc chặn)`
                : normalizedTrack;
        });

    let reason = String(payload.reason || '').trim();
    if (!reason) {
        if (!schemaReady) {
            reason = 'Schema KPI chưa sẵn sàng để mở quản trị split-trigger.';
        } else if (ready) {
            reason = 'Split-trigger đã sẵn sàng, có thể mở hành động thanh tra.';
        } else {
            reason = 'Split-trigger chưa sẵn sàng, cần cải thiện KPI trên các track đang bị chặn.';
        }
    }

    return {
        ready,
        schemaReady,
        readinessScore,
        enabledRules,
        passedRules,
        blockedTracks,
        reason,
        generatedAt: payload.generated_at || '',
    };
}


function setFraudActionButtonState(button, disabled) {
    if (!button) return;
    button.disabled = Boolean(disabled);
    if (disabled) {
        button.classList.add('opacity-50', 'cursor-not-allowed');
        button.classList.remove('hover:bg-white/20');
    } else {
        button.classList.remove('opacity-50', 'cursor-not-allowed');
        button.classList.add('hover:bg-white/20');
    }
}


function renderFraudSplitTriggerGate(rawStatus) {
    const gate = normalizeSplitTriggerStatus(rawStatus);
    window._fraudSplitGateState = gate;

    const panel = document.getElementById('fraud-split-gate-panel');
    const badge = document.getElementById('fraud-split-gate-badge');
    const summary = document.getElementById('fraud-split-gate-summary');
    const updated = document.getElementById('fraud-split-gate-updated');

    let badgeLabel = 'SẴN SÀNG';
    let badgeToneClass = 'bg-emerald-500/20 text-emerald-100';
    if (!gate.schemaReady) {
        badgeLabel = 'THIẾU SCHEMA';
        badgeToneClass = 'bg-slate-500/20 text-slate-100';
    } else if (!gate.ready) {
        badgeLabel = 'ĐANG KHÓA';
        badgeToneClass = 'bg-amber-500/20 text-amber-100';
    }

    if (panel) {
        panel.classList.remove('hidden');
    }
    if (badge) {
        badge.className = `text-[10px] px-2 py-0.5 rounded font-black uppercase tracking-wider ${badgeToneClass}`;
        badge.textContent = badgeLabel;
    }
    if (summary) {
        const coreSummary = `Độ sẵn sàng ${gate.readinessScore.toFixed(1)}% • Quy tắc ${gate.passedRules}/${gate.enabledRules}`;
        const blockedText = gate.blockedTracks.length
            ? ` • Đang chặn: ${gate.blockedTracks.join(', ')}`
            : '';
        summary.textContent = gate.ready
            ? `${coreSummary}. ${gate.reason}`
            : `${coreSummary}${blockedText}. ${gate.reason}`;
    }
    if (updated) {
        updated.textContent = `Cập nhật: ${formatSplitGateTimestamp(gate.generatedAt)}`;
    }

    const lockActions = !gate.ready;
    const sendNoticeBtn = document.getElementById('action-send-notice-btn');
    const createReportBtn = document.getElementById('action-create-report-btn');
    setFraudActionButtonState(sendNoticeBtn, lockActions);
    setFraudActionButtonState(createReportBtn, lockActions);

    const lockReason = gate.reason || 'Split-trigger chưa sẵn sàng.';
    if (sendNoticeBtn) {
        sendNoticeBtn.title = lockActions ? lockReason : 'Gửi thông báo giải trình';
    }
    if (createReportBtn) {
        createReportBtn.title = lockActions ? lockReason : 'Lập biên bản vi phạm';
    }
}


// ===================================================================
// MODE 1: SINGLE QUERY (Real-time)
// ===================================================================

function showSingleResultSkeleton() {
    const skeleton = document.getElementById('single-result-skeleton');
    const resultDiv = document.getElementById('fraud-result');
    if (!skeleton || !resultDiv) return;

    _singleResultVisibleBeforeLoading = !resultDiv.classList.contains('hidden');
    resultDiv.classList.add('hidden');
    skeleton.classList.remove('hidden');
    skeleton.setAttribute('aria-busy', 'true');
}


function hideSingleResultSkeleton({ restorePrevious = false } = {}) {
    const skeleton = document.getElementById('single-result-skeleton');
    const resultDiv = document.getElementById('fraud-result');

    if (skeleton) {
        skeleton.classList.add('hidden');
        skeleton.setAttribute('aria-busy', 'false');
    }

    if (restorePrevious && resultDiv && _singleResultVisibleBeforeLoading) {
        resultDiv.classList.remove('hidden');
    }

    _singleResultVisibleBeforeLoading = false;
}

async function checkFraudRisk() {
    const taxCode = document.getElementById('fraud-mst').value.trim();
    if (!taxCode) {
        showToast('Thiếu thông tin', 'Vui lòng nhập Mã số thuế hoặc Tên doanh nghiệp.', 'warning');
        return;
    }

    showSingleResultSkeleton();

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
        hideSingleResultSkeleton({ restorePrevious: true });
        showToast('Lỗi phân tích', error.message || 'Có lỗi xảy ra khi gọi API AI.', 'error');
    } finally {
        btn.innerHTML = '<span class="material-symbols-outlined text-[18px]">psychology</span> Phân tích AI';
        btn.disabled = false;
    }
}


function buildWorkspaceDeepLink(targetPage, options = {}) {
    const params = new URLSearchParams();
    params.set('source', 'fraud');
    params.set('window_days', String(options.windowDays || 90));
    params.set('top_k', String(options.topK || 50));

    const focus = String(options.focus || '').trim().toLowerCase();
    if (focus) {
        params.set('focus', focus);
    }

    const taxCode = String(options.taxCode || '').trim();
    if (taxCode) {
        params.set('tax_code', taxCode);
    }

    return `${targetPage}?${params.toString()}`;
}


function syncFraudWorkspaceLinks(data) {
    const taxCode = String(data?.tax_code || '').trim();

    const decisionLink = document.getElementById('decision-open-intervention-link');
    if (decisionLink) {
        decisionLink.href = buildWorkspaceDeepLink('intervention.html', {
            focus: 'actions',
            taxCode,
        });
    }

    const vatLink = document.getElementById('vat-open-specialized-link');
    if (vatLink) {
        vatLink.href = buildWorkspaceDeepLink('specialized.html', {
            focus: 'vat',
            taxCode,
        });
    }

    const auditLink = document.getElementById('audit-open-specialized-link');
    if (auditLink) {
        auditLink.href = buildWorkspaceDeepLink('specialized.html', {
            focus: 'audit',
            taxCode,
        });
    }
}


function renderSingleResult(data) {
    hideSingleResultSkeleton();

    window._currentSingleData = data;
    const resultDiv = document.getElementById('fraud-result');
    resultDiv.classList.remove('hidden');
    if (!PREFERS_REDUCED_MOTION) {
        resultDiv.animate([
            { opacity: 0, transform: 'translateY(40px)' },
            { opacity: 1, transform: 'translateY(0)' }
        ], { duration: MOTION_DURATION_RESULT, easing: MOTION_EASE_EMPHASIS, fill: 'both' });
    }

    document.getElementById('result-company-name').textContent = data.company_name || 'Không rõ';
    document.getElementById('result-meta').textContent = `MST: ${data.tax_code} • Ngành: ${data.industry || '---'}`;
    document.getElementById('result-revenue').textContent = formatVND(data.revenue);
    document.getElementById('result-expenses').textContent = formatVND(data.total_expenses);
    document.getElementById('result-year').textContent = data.year || '---';
    syncFraudWorkspaceLinks(data);
    renderFraudSplitTriggerGate(data.split_trigger_status || null);

    animateRiskScore(data.risk_score || 0, data);
    renderRedFlags(data.red_flags || []);
    const interventionPayload = normalizeInterventionUplift(data.intervention_uplift || null, data);
    renderDecisionIntelligence(data.decision_intelligence || null, data, interventionPayload);
    renderVatRefundSignals(data.vat_refund_signals || null, data);
    renderAuditValue(data.audit_value || null, data);
    renderShapExplanation(data.shap_explanation || []);
    renderForensicArgumentation(data);

    // Single Query Charts (Trend + Radar)
    const chartsRow = document.getElementById('single-charts-row');
    if (chartsRow) {
        chartsRow.style.display = 'grid';
        renderSingleTrendChart(data.yearly_history || [], {
            historySource: data.history_source || 'unavailable',
            historyYearCount: Number(data.history_year_count) || 0,
        });
        renderSingleRadarChart(data);
    }

    // Peer Comparison Chart
    const peerRow = document.getElementById('single-peer-row');
    if (peerRow) {
        peerRow.style.display = 'block';
        renderPeerComparison(data);
    }

    // Advanced Single Charts (Feature timeline + Waterfall deltas)
    const advancedRow = document.getElementById('single-advanced-row');
    if (advancedRow) {
        const hasFeatureSnapshot = ['risk_score', 'f1_divergence', 'f2_ratio_limit', 'f3_vat_structure', 'f4_peer_comparison']
            .some((key) => Number.isFinite(Number(data[key])));
        const hasAdvancedData = (
            (Array.isArray(data.yearly_feature_scores) && data.yearly_feature_scores.length > 0)
            || (data.previous_year_features && typeof data.previous_year_features === 'object')
            || (data.feature_deltas && typeof data.feature_deltas === 'object' && Object.keys(data.feature_deltas).length > 0)
            || hasFeatureSnapshot
        );

        if (hasAdvancedData) {
            advancedRow.style.display = 'grid';
            renderSingleFeatureTrendChart(data.yearly_feature_scores || []);
            renderSingleFeatureDeltaWaterfall(data);
            renderSingleFeaturePercentileBullet(data);
            renderSingleSensitivityHeatmap(data);
        } else {
            advancedRow.style.display = 'none';
        }
    }

    // Strategic Single Charts (Sankey/Cumulative/Distribution/Timeline)
    const strategicRow = document.getElementById('single-strategic-row');
    if (strategicRow) {
        const sankeyData = (data.single_risk_tier_sankey && typeof data.single_risk_tier_sankey === 'object')
            ? data.single_risk_tier_sankey
            : {};
        const cumulativeData = (data.single_cumulative_risk_curve && typeof data.single_cumulative_risk_curve === 'object')
            ? data.single_cumulative_risk_curve
            : {};
        const marginDistribution = (data.single_margin_distribution && typeof data.single_margin_distribution === 'object')
            ? data.single_margin_distribution
            : {};
        const redFlagTimeline = (data.single_red_flags_timeline && typeof data.single_red_flags_timeline === 'object')
            ? data.single_red_flags_timeline
            : {};

        const hasStrategicData = (
            (Array.isArray(sankeyData.nodes) && sankeyData.nodes.length > 0)
            || (Array.isArray(cumulativeData.points) && cumulativeData.points.length > 0)
            || (Array.isArray(marginDistribution.bins) && marginDistribution.bins.length > 0)
            || (Array.isArray(redFlagTimeline.year_points) && redFlagTimeline.year_points.length > 0)
        );

        if (hasStrategicData) {
            strategicRow.style.display = 'grid';
            renderSingleRiskTierSankey(sankeyData);
            renderSingleCumulativeRiskCurve(cumulativeData);
            renderSingleMarginDistribution(marginDistribution);
            renderSingleRedFlagsTimeline(redFlagTimeline);
        } else {
            strategicRow.style.display = 'none';
        }
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


function buildFallbackInterventionUplift(data) {
    const score = Math.max(0, Math.min(100, Number(data?.risk_score || 0)));
    const anomaly = Math.max(0, Number(data?.anomaly_score || 0));
    const confidenceRaw = Math.max(0, Math.min(100, Number(data?.model_confidence || 0)));
    const redFlags = Array.isArray(data?.red_flags) ? data.red_flags : [];

    const highFlagCount = redFlags.filter((flag) => {
        const severity = String(flag?.severity || '').toLowerCase();
        return severity === 'high' || severity === 'critical';
    }).length;

    const yearlyFeatureScores = Array.isArray(data?.yearly_feature_scores) ? data.yearly_feature_scores : [];
    let trendDelta = 0;
    if (yearlyFeatureScores.length >= 2) {
        const sorted = [...yearlyFeatureScores].sort((a, b) => Number(a?.year || 0) - Number(b?.year || 0));
        const latest = Number(sorted[sorted.length - 1]?.risk_score || 0);
        const previous = Number(sorted[sorted.length - 2]?.risk_score || 0);
        trendDelta = latest - previous;
    }

    const priorityRaw = (
        score * 0.60
        + Math.min(20, highFlagCount * 8)
        + Math.min(12, redFlags.length * 2)
        + Math.min(10, Math.max(0, trendDelta) * 0.8)
        + Math.min(15, Math.max(0, (anomaly - 0.55) * 35))
    );
    const priorityScore = Math.max(0, Math.min(100, Math.round(priorityRaw + (confidenceRaw > 0 && confidenceRaw < 55 ? 5 : 0))));

    let action = 'monitor';
    if (score >= 82 || highFlagCount >= 2 || anomaly >= 0.90 || priorityScore >= 86) {
        action = 'escalated_enforcement';
    } else if (score >= 65 || highFlagCount >= 1 || priorityScore >= 68) {
        action = 'field_audit';
    } else if (score >= 45 || redFlags.length > 0 || priorityScore >= 50) {
        action = 'structured_outreach';
    } else if (score >= 25 || anomaly >= 0.60) {
        action = 'auto_reminder';
    }

    const impactRatioMap = {
        monitor: 0.05,
        auto_reminder: 0.12,
        structured_outreach: 0.20,
        field_audit: 0.30,
        escalated_enforcement: 0.38,
    };
    const impactRatio = impactRatioMap[action] || 0.12;

    const revenue = Math.max(0, Number(data?.revenue || 0));
    const totalExpenses = Math.max(0, Number(data?.total_expenses || 0));
    const estimatedPenaltyExposure = Math.max(0, (totalExpenses - revenue) * 0.03) + highFlagCount * 2500000 + redFlags.length * 800000;
    const estimatedCollectionBase = Math.max(0, revenue * 0.015 + totalExpenses * 0.008);

    const confidence = confidenceRaw >= 80 ? 'high' : confidenceRaw >= 55 ? 'medium' : 'low';
    const rationale = [
        `Điểm rủi ro hiện tại ${score.toFixed(1)}/100.`,
        highFlagCount > 0 ? `Ghi nhận ${highFlagCount} red flag mức cao.` : null,
        trendDelta >= 8 ? `Xu hướng điểm rủi ro tăng ${trendDelta.toFixed(1)} điểm.` : null,
        anomaly >= 0.70 ? 'Điểm dị biệt ở mức cần can thiệp nghiệp vụ.' : null,
    ].filter(Boolean).join(' ');

    const nextStepsByAction = {
        monitor: [
            'Duy trì giám sát định kỳ theo chu kỳ quý.',
            'Tự động cảnh báo khi điểm vượt ngưỡng 45 trong kỳ mới.',
        ],
        auto_reminder: [
            'Gửi nhắc hạn giải trình cho doanh nghiệp trong 72 giờ.',
            'Ràng buộc đối soát bổ sung nếu điểm dị biệt tiếp tục tăng.',
        ],
        structured_outreach: [
            'Mở phiên làm việc có cấu trúc với doanh nghiệp theo checklist F1-F4.',
            'Đối chiếu tờ khai VAT và báo cáo tài chính của 3 kỳ gần nhất.',
        ],
        field_audit: [
            'Phân công tổ nghiệp vụ kiểm tra tại chỗ theo hồ sơ ưu tiên.',
            'Yêu cầu bộ chứng từ giải trình và đối soát nguồn hóa đơn liên quan.',
        ],
        escalated_enforcement: [
            'Kích hoạt quy trình xử lý cưỡng chế theo thẩm quyền.',
            'Phối hợp liên đơn vị để thu hồi nghĩa vụ tồn đọng trong 30 ngày.',
        ],
    };

    return {
        recommended_action: action,
        priority_score: priorityScore,
        expected_risk_reduction_pp: Number((score * impactRatio).toFixed(1)),
        expected_penalty_saving: Number((estimatedPenaltyExposure * Math.min(0.9, impactRatio + 0.15)).toFixed(2)),
        expected_collection_uplift: Number((estimatedCollectionBase * impactRatio).toFixed(2)),
        confidence,
        rationale: rationale || 'Hệ thống chưa ghi nhận động lực can thiệp mạnh ở chu kỳ hiện tại.',
        next_steps: nextStepsByAction[action] || nextStepsByAction.monitor,
    };
}


function normalizeInterventionUplift(interventionUplift, data) {
    const fallback = buildFallbackInterventionUplift(data || {});
    if (!interventionUplift || typeof interventionUplift !== 'object') {
        return fallback;
    }

    const actionRaw = String(interventionUplift.recommended_action || '').trim();
    const allowedActions = new Set(['monitor', 'auto_reminder', 'structured_outreach', 'field_audit', 'escalated_enforcement']);
    const recommendedAction = allowedActions.has(actionRaw) ? actionRaw : fallback.recommended_action;

    const priorityScore = Number.isFinite(Number(interventionUplift.priority_score))
        ? Math.max(0, Math.min(100, Math.round(Number(interventionUplift.priority_score))))
        : fallback.priority_score;

    const expectedRiskReduction = Number.isFinite(Number(interventionUplift.expected_risk_reduction_pp))
        ? Math.max(0, Number(interventionUplift.expected_risk_reduction_pp))
        : fallback.expected_risk_reduction_pp;

    const expectedPenaltySaving = Number.isFinite(Number(interventionUplift.expected_penalty_saving))
        ? Math.max(0, Number(interventionUplift.expected_penalty_saving))
        : fallback.expected_penalty_saving;

    const expectedCollectionUplift = Number.isFinite(Number(interventionUplift.expected_collection_uplift))
        ? Math.max(0, Number(interventionUplift.expected_collection_uplift))
        : fallback.expected_collection_uplift;

    const confidenceRaw = String(interventionUplift.confidence || '').toLowerCase();
    const confidence = ['low', 'medium', 'high'].includes(confidenceRaw) ? confidenceRaw : fallback.confidence;

    const rationale = (typeof interventionUplift.rationale === 'string' && interventionUplift.rationale.trim())
        ? interventionUplift.rationale.trim()
        : fallback.rationale;

    const nextSteps = Array.isArray(interventionUplift.next_steps)
        ? interventionUplift.next_steps.map((step) => String(step || '').trim()).filter(Boolean).slice(0, 5)
        : [];

    return {
        recommended_action: recommendedAction,
        priority_score: priorityScore,
        expected_risk_reduction_pp: expectedRiskReduction,
        expected_penalty_saving: expectedPenaltySaving,
        expected_collection_uplift: expectedCollectionUplift,
        confidence,
        rationale,
        next_steps: nextSteps.length ? nextSteps : fallback.next_steps,
    };
}


function mapDecisionFromIntervention(interventionPayload) {
    const action = String(interventionPayload?.recommended_action || 'monitor');
    if (action === 'escalated_enforcement') {
        return { recommended_action: 'urgent_audit', action_label: 'Xử lý cưỡng chế nâng cao', action_deadline_days: 7 };
    }
    if (action === 'field_audit') {
        return { recommended_action: 'targeted_review', action_label: 'Kiểm tra tại chỗ', action_deadline_days: 14 };
    }
    if (action === 'structured_outreach') {
        return { recommended_action: 'enhanced_monitoring', action_label: 'Can thiệp có cấu trúc', action_deadline_days: 21 };
    }
    if (action === 'auto_reminder') {
        return { recommended_action: 'enhanced_monitoring', action_label: 'Nhắc hạn tự động', action_deadline_days: 30 };
    }
    return { recommended_action: 'periodic_monitoring', action_label: 'Theo dõi định kỳ', action_deadline_days: 60 };
}


function harmonizeDecisionWithIntervention(decisionPayload, interventionPayload) {
    const base = decisionPayload && typeof decisionPayload === 'object' ? { ...decisionPayload } : {};
    const intervention = normalizeInterventionUplift(interventionPayload || null, null);
    const mapped = mapDecisionFromIntervention(intervention);
    const shouldEscalate = intervention.recommended_action === 'field_audit'
        || intervention.recommended_action === 'escalated_enforcement'
        || Number(intervention.priority_score || 0) >= 75;

    const topSignals = Array.isArray(base.top_signals) ? [...base.top_signals] : [];
    if (!topSignals.some((signal) => String(signal?.key || '') === 'intervention_uplift')) {
        topSignals.push({
            key: 'intervention_uplift',
            label: 'Can thiệp/Uplift',
            value: Number(intervention.priority_score || 0),
            severity: shouldEscalate ? 'high' : Number(intervention.priority_score || 0) >= 45 ? 'medium' : 'low',
            summary: intervention.rationale || 'Hành động can thiệp đã được đồng bộ với Trí tuệ quyết định.',
        });
    }

    return {
        ...base,
        recommended_action: mapped.recommended_action,
        action_label: mapped.action_label,
        action_deadline_days: mapped.action_deadline_days,
        priority_score: Number(intervention.priority_score || 0),
        rationale: intervention.rationale || base.rationale || 'Không có ghi chú giải thích bổ sung.',
        next_steps: Array.isArray(intervention.next_steps) && intervention.next_steps.length ? intervention.next_steps : (base.next_steps || []),
        top_signals: topSignals.slice(0, 4),
        should_escalate: shouldEscalate,
    };
}


function buildFallbackDecisionIntelligence(data) {
    const score = Math.max(0, Math.min(100, Number(data?.risk_score || 0)));
    const anomaly = Math.max(0, Number(data?.anomaly_score || 0));
    const redFlags = Array.isArray(data?.red_flags) ? data.red_flags : [];
    const yearlyFeatureScores = Array.isArray(data?.yearly_feature_scores) ? data.yearly_feature_scores : [];

    let trendDelta = 0;
    if (yearlyFeatureScores.length >= 2) {
        const sorted = [...yearlyFeatureScores].sort((a, b) => Number(a?.year || 0) - Number(b?.year || 0));
        const latest = Number(sorted[sorted.length - 1]?.risk_score || 0);
        const previous = Number(sorted[sorted.length - 2]?.risk_score || 0);
        trendDelta = latest - previous;
    }

    const highFlagCount = redFlags.filter((flag) => {
        const severity = String(flag?.severity || '').toLowerCase();
        return severity === 'high' || severity === 'critical';
    }).length;

    const priorityRaw = (
        score * 0.62
        + Math.min(20, highFlagCount * 9)
        + Math.min(12, redFlags.length * 2)
        + Math.min(14, Math.max(0, trendDelta) * 0.9)
        + Math.min(15, Math.max(0, (anomaly - 0.5) * 35))
    );
    const priorityScore = Math.max(0, Math.min(100, Math.round(priorityRaw)));

    let recommendedAction = 'periodic_monitoring';
    let actionLabel = 'Theo dõi định kỳ';
    let actionDeadlineDays = 60;
    let shouldEscalate = false;

    if (score >= 80 || highFlagCount >= 2 || anomaly >= 0.90 || priorityScore >= 85) {
        recommendedAction = 'urgent_audit';
        actionLabel = 'Thanh tra đột xuất';
        actionDeadlineDays = 7;
        shouldEscalate = true;
    } else if (score >= 60 || highFlagCount >= 1 || priorityScore >= 65) {
        recommendedAction = 'targeted_review';
        actionLabel = 'Kiểm tra chuyên sâu';
        actionDeadlineDays = 15;
        shouldEscalate = priorityScore >= 75;
    } else if (score >= 40 || redFlags.length > 0 || priorityScore >= 45) {
        recommendedAction = 'enhanced_monitoring';
        actionLabel = 'Giám sát tăng cường';
        actionDeadlineDays = 30;
    }

    const nextStepsByAction = {
        urgent_audit: [
            'Khởi tạo hồ sơ thanh tra đột xuất trong 24 giờ.',
            'Yêu cầu doanh nghiệp giải trình hóa đơn VAT trong 7 ngày.',
            'Đối soát giao dịch liên quan với nhóm đối tác có rủi ro cao.',
        ],
        targeted_review: [
            'Mở đợt kiểm tra chuyên đề theo nhóm chỉ số vượt ngưỡng.',
            'Ưu tiên đối soát tờ khai VAT và điều chỉnh bổ sung gần nhất.',
            'Đánh giá lại sau khi tiếp nhận bộ chứng từ giải trình.',
        ],
        enhanced_monitoring: [
            'Đưa doanh nghiệp vào danh sách giám sát tăng cường theo tháng.',
            'Theo dõi biến động điểm và kích hoạt cảnh báo nếu điểm vượt 60.',
            'Cập nhật bộ dữ liệu trước kỳ đánh giá tiếp theo.',
        ],
        periodic_monitoring: [
            'Duy trì theo dõi định kỳ theo quý.',
            'Cập nhật dữ liệu tài chính mới trước lần đánh giá tiếp theo.',
            'Tự động nâng cấp mức cảnh báo nếu xuất hiện red flag mới.',
        ],
    };

    const topSignals = [
        {
            key: 'risk_score',
            label: 'Tổng điểm rủi ro',
            value: score,
            severity: score >= 60 ? 'high' : score >= 40 ? 'medium' : 'low',
            summary: `Điểm tổng hợp hiện tại ${score.toFixed(1)}/100.`,
        },
    ];
    if (anomaly >= 0.6) {
        topSignals.push({
            key: 'anomaly_score',
            label: 'Độ dị biệt',
            value: anomaly,
            severity: anomaly >= 0.85 ? 'high' : 'medium',
            summary: 'Độ dị biệt tài chính đang cao hơn mức an toàn.',
        });
    }
    if (redFlags.length > 0) {
        topSignals.push({
            key: 'red_flags',
            label: 'Cụm cảnh báo',
            value: redFlags.length,
            severity: highFlagCount > 0 || redFlags.length >= 3 ? 'high' : 'medium',
            summary: `Phát hiện ${redFlags.length} red flag trong kết quả AI.`,
        });
    }
    if (trendDelta >= 8) {
        topSignals.push({
            key: 'risk_trend',
            label: 'Xu hướng điểm rủi ro',
            value: trendDelta,
            severity: trendDelta >= 15 ? 'high' : 'medium',
            summary: 'Điểm rủi ro đang tăng so với kỳ trước.',
        });
    }

    const rationale = [
        `Điểm rủi ro hiện tại ${score.toFixed(1)}/100.`,
        redFlags.length > 0 ? `Có ${redFlags.length} red flag đang kích hoạt.` : null,
        trendDelta >= 8 ? `Xu hướng điểm rủi ro tăng ${trendDelta.toFixed(1)} điểm.` : null,
        anomaly >= 0.7 ? 'Độ dị biệt tài chính ở mức cảnh báo.' : null,
    ].filter(Boolean).join(' ');

    return {
        recommended_action: recommendedAction,
        action_label: actionLabel,
        action_deadline_days: actionDeadlineDays,
        priority_score: priorityScore,
        rationale: rationale || 'Hệ thống khuyến nghị theo dõi định kỳ đối với doanh nghiệp này.',
        next_steps: nextStepsByAction[recommendedAction] || nextStepsByAction.periodic_monitoring,
        top_signals: topSignals.slice(0, 4),
        should_escalate: shouldEscalate,
    };
}


function normalizeDecisionIntelligence(decisionIntelligence, data, interventionPayload = null) {
    const fallback = buildFallbackDecisionIntelligence(data || {});
    const normalizedIntervention = normalizeInterventionUplift(interventionPayload || data?.intervention_uplift || null, data || {});

    if (!decisionIntelligence || typeof decisionIntelligence !== 'object') {
        return harmonizeDecisionWithIntervention(fallback, normalizedIntervention);
    }

    const rawAction = String(decisionIntelligence.recommended_action || '').trim();
    const allowedActions = new Set(['periodic_monitoring', 'enhanced_monitoring', 'targeted_review', 'urgent_audit']);
    const recommendedAction = allowedActions.has(rawAction) ? rawAction : fallback.recommended_action;

    const actionLabel = (typeof decisionIntelligence.action_label === 'string' && decisionIntelligence.action_label.trim())
        ? decisionIntelligence.action_label.trim()
        : fallback.action_label;

    const actionDeadlineDays = Number.isFinite(Number(decisionIntelligence.action_deadline_days))
        ? Math.max(1, Math.round(Number(decisionIntelligence.action_deadline_days)))
        : fallback.action_deadline_days;

    const priorityScore = Number.isFinite(Number(decisionIntelligence.priority_score))
        ? Math.max(0, Math.min(100, Math.round(Number(decisionIntelligence.priority_score))))
        : fallback.priority_score;

    const rationale = (typeof decisionIntelligence.rationale === 'string' && decisionIntelligence.rationale.trim())
        ? decisionIntelligence.rationale.trim()
        : fallback.rationale;

    const nextSteps = Array.isArray(decisionIntelligence.next_steps)
        ? decisionIntelligence.next_steps
            .map((step) => String(step || '').trim())
            .filter(Boolean)
            .slice(0, 5)
        : [];

    const topSignals = Array.isArray(decisionIntelligence.top_signals)
        ? decisionIntelligence.top_signals
            .filter((signal) => signal && typeof signal === 'object')
            .map((signal) => {
                const severity = String(signal.severity || '').toLowerCase();
                return {
                    key: String(signal.key || 'signal'),
                    label: String(signal.label || 'Tín hiệu'),
                    value: Number.isFinite(Number(signal.value)) ? Number(signal.value) : null,
                    severity: severity === 'high' || severity === 'medium' || severity === 'low' ? severity : 'medium',
                    summary: String(signal.summary || '').trim(),
                };
            })
            .slice(0, 4)
        : [];

    const normalizedDecision = {
        recommended_action: recommendedAction,
        action_label: actionLabel,
        action_deadline_days: actionDeadlineDays,
        priority_score: priorityScore,
        rationale,
        next_steps: nextSteps.length ? nextSteps : fallback.next_steps,
        top_signals: topSignals.length ? topSignals : fallback.top_signals,
        should_escalate: Boolean(decisionIntelligence.should_escalate),
    };

    return harmonizeDecisionWithIntervention(normalizedDecision, normalizedIntervention);
}


function getDecisionBadgeClass(action) {
    if (action === 'urgent_audit') return 'bg-red-100 text-red-700 border border-red-200';
    if (action === 'targeted_review') return 'bg-orange-100 text-orange-700 border border-orange-200';
    if (action === 'enhanced_monitoring') return 'bg-amber-100 text-amber-700 border border-amber-200';
    return 'bg-emerald-100 text-emerald-700 border border-emerald-200';
}


function formatDecisionSignalValue(signal) {
    const value = Number(signal?.value);
    if (!Number.isFinite(value)) return 'Không có';

    const key = String(signal?.key || '');
    if (key === 'risk_score') return `${value.toFixed(1)}/100`;
    if (key === 'anomaly_score') return `${(value * 100).toFixed(1)}%`;
    if (key === 'risk_trend') return `${value > 0 ? '+' : ''}${value.toFixed(1)} điểm`;
    if (key === 'red_flags') return `${Math.round(value)} cảnh báo`;
    return value.toFixed(3);
}


function getInterventionActionLabel(action) {
    const map = {
        monitor: 'Giám sát',
        auto_reminder: 'Nhắc hạn tự động',
        structured_outreach: 'Can thiệp có cấu trúc',
        field_audit: 'Kiểm tra tại chỗ',
        escalated_enforcement: 'Cưỡng chế nâng cao',
    };
    return map[String(action || '').toLowerCase()] || 'Giám sát';
}


function renderDecisionIntelligence(decisionIntelligence, data, interventionPayload = null) {
    const panel = document.getElementById('decision-intelligence-panel');
    if (!panel) return;

    const actionBadge = document.getElementById('decision-action-badge');
    const priorityEl = document.getElementById('decision-priority-score');
    const rationaleEl = document.getElementById('decision-rationale');
    const interventionActionEl = document.getElementById('decision-intervention-action');
    const interventionRiskReductionEl = document.getElementById('decision-intervention-risk-reduction');
    const interventionCollectionUpliftEl = document.getElementById('decision-intervention-collection-uplift');
    const stepsEl = document.getElementById('decision-next-steps');
    const signalsEl = document.getElementById('decision-signals');

    if (!actionBadge || !priorityEl || !rationaleEl || !stepsEl || !signalsEl) {
        panel.style.display = 'none';
        return;
    }

    const normalizedIntervention = normalizeInterventionUplift(interventionPayload || data?.intervention_uplift || null, data || {});
    const payload = normalizeDecisionIntelligence(decisionIntelligence, data || {}, normalizedIntervention);
    panel.style.display = 'block';

    const suffix = payload.should_escalate ? ' • Chuyển cấp' : '';
    actionBadge.className = `px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider ${getDecisionBadgeClass(payload.recommended_action)}`;
    actionBadge.textContent = `${payload.action_label}${suffix}`;
    priorityEl.textContent = String(payload.priority_score);
    rationaleEl.textContent = payload.rationale || 'Không có ghi chú giải thích bổ sung.';

    if (interventionActionEl) {
        interventionActionEl.textContent = getInterventionActionLabel(normalizedIntervention.recommended_action || 'monitor');
    }
    if (interventionRiskReductionEl) {
        interventionRiskReductionEl.textContent = `${Number(normalizedIntervention.expected_risk_reduction_pp || 0).toFixed(1)} pp`;
    }
    if (interventionCollectionUpliftEl) {
        interventionCollectionUpliftEl.textContent = formatVND(Number(normalizedIntervention.expected_collection_uplift || 0));
    }

    stepsEl.innerHTML = '';
    (Array.isArray(payload.next_steps) ? payload.next_steps : []).forEach((step) => {
        const li = document.createElement('li');
        li.className = 'flex items-start gap-2';

        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined text-[14px] text-primary-container mt-0.5';
        icon.textContent = 'task_alt';

        const text = document.createElement('span');
        text.textContent = step;

        li.appendChild(icon);
        li.appendChild(text);
        stepsEl.appendChild(li);
    });

    if (!stepsEl.children.length) {
        const li = document.createElement('li');
        li.className = 'text-slate-500 italic';
        li.textContent = 'Chưa có hành động khuyến nghị.';
        stepsEl.appendChild(li);
    }

    signalsEl.innerHTML = '';
    const toneBySeverity = {
        high: 'bg-red-50 border-red-200 text-red-700',
        medium: 'bg-amber-50 border-amber-200 text-amber-700',
        low: 'bg-emerald-50 border-emerald-200 text-emerald-700',
    };

    (Array.isArray(payload.top_signals) ? payload.top_signals : []).forEach((signal) => {
        const tone = toneBySeverity[String(signal?.severity || 'medium')] || toneBySeverity.medium;
        const card = document.createElement('div');
        card.className = `rounded-lg border px-3 py-2 ${tone}`;

        const title = document.createElement('p');
        title.className = 'text-[10px] font-black uppercase tracking-wider';
        title.textContent = `${signal.label || 'Tín hiệu'}: ${formatDecisionSignalValue(signal)}`;

        const summary = document.createElement('p');
        summary.className = 'text-[10px] mt-1 leading-relaxed';
        summary.textContent = signal.summary || 'Tín hiệu được đưa vào danh sách ưu tiên kiểm tra.';

        card.appendChild(title);
        card.appendChild(summary);
        signalsEl.appendChild(card);
    });

    if (!signalsEl.children.length) {
        const placeholder = document.createElement('div');
        placeholder.className = 'rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[10px] text-slate-500 italic';
        placeholder.textContent = 'Chưa ghi nhận tín hiệu ưu tiên bổ sung.';
        signalsEl.appendChild(placeholder);
    }
}


function buildFallbackVatRefundSignals(data) {
    const score = Math.max(0, Math.min(100, Number(data?.risk_score || 0)));
    const anomaly = Math.max(0, Number(data?.anomaly_score || 0));
    const f2 = Math.max(0, Number(data?.f2_ratio_limit || 0));
    const f3 = Math.max(0, Number(data?.f3_vat_structure || 0));
    const revenue = Math.max(0, Number(data?.revenue || 0));
    const totalExpenses = Math.max(0, Number(data?.total_expenses || 0));

    const vatOutput = revenue * 0.1;
    const vatInput = totalExpenses * 0.75 * 0.1;
    const ratio = vatOutput > 0 ? vatInput / vatOutput : (vatInput > 0 ? 2 : 0);
    const refundGap = Math.max(0, vatInput - vatOutput);

    const redFlags = Array.isArray(data?.red_flags) ? data.red_flags : [];
    const vatFlagCount = redFlags.filter((flag) => {
        const blob = [
            flag?.feature,
            flag?.title,
            flag?.reason,
            flag?.description,
            flag?.actual_value,
        ].map((v) => String(v || '').toLowerCase()).join(' ');
        return blob.includes('vat') || blob.includes('hoa don') || blob.includes('hoan thue') || blob.includes('invoice');
    }).length;

    const yearlyFeatureScores = Array.isArray(data?.yearly_feature_scores) ? data.yearly_feature_scores : [];
    let f3Trend = 0;
    if (yearlyFeatureScores.length >= 2) {
        const sorted = [...yearlyFeatureScores].sort((a, b) => Number(a?.year || 0) - Number(b?.year || 0));
        f3Trend = Number(sorted[sorted.length - 1]?.f3_vat_structure || 0) - Number(sorted[sorted.length - 2]?.f3_vat_structure || 0);
    }

    const indicators = [];
    if (f3 >= 0.7) {
        indicators.push({ key: 'f3_vat_structure', label: 'Cấu trúc VAT bất thường', value: f3, severity: 'high', summary: 'F3 VAT vượt ngưỡng cao cần đối soát.' });
    } else if (f3 >= 0.55) {
        indicators.push({ key: 'f3_vat_structure', label: 'Cấu trúc VAT bất thường', value: f3, severity: 'medium', summary: 'F3 VAT vượt ngưỡng theo dõi.' });
    }
    if (ratio >= 1.15) {
        indicators.push({ key: 'vat_input_output_ratio', label: 'Tỷ lệ VAT đầu vào/đầu ra', value: ratio, severity: 'high', summary: 'VAT đầu vào vượt đáng kể VAT đầu ra.' });
    } else if (ratio >= 1.0) {
        indicators.push({ key: 'vat_input_output_ratio', label: 'Tỷ lệ VAT đầu vào/đầu ra', value: ratio, severity: 'medium', summary: 'Tỷ lệ VAT cần được theo dõi sát.' });
    }
    if (refundGap > 0) {
        indicators.push({ key: 'estimated_refund_gap', label: 'Chênh lệch VAT ước tính', value: refundGap, severity: refundGap > vatOutput * 0.3 ? 'high' : 'medium', summary: 'Chênh lệch VAT dương cần đối chiếu chứng từ.' });
    }
    if (vatFlagCount > 0) {
        indicators.push({ key: 'vat_related_flags', label: 'Red-flag liên quan VAT', value: vatFlagCount, severity: vatFlagCount >= 2 ? 'high' : 'medium', summary: `Ghi nhận ${vatFlagCount} red-flag liên quan VAT/hóa đơn.` });
    }
    if (anomaly >= 0.75) {
        indicators.push({ key: 'anomaly_score', label: 'Độ dị biệt tài chính', value: anomaly, severity: anomaly >= 0.9 ? 'high' : 'medium', summary: 'Độ dị biệt tài chính vượt ngưỡng theo dõi hoàn thuế.' });
    }
    if (f3Trend >= 0.08) {
        indicators.push({ key: 'f3_trend', label: 'Xu hướng F3 VAT', value: f3Trend, severity: f3Trend >= 0.15 ? 'high' : 'medium', summary: 'F3 VAT tăng so với kỳ trước.' });
    }

    const priorityRaw = (
        f3 * 45
        + Math.min(25, Math.max(0, (ratio - 1) * 60))
        + Math.min(15, Math.max(0, refundGap / (vatOutput + 1)) * 10)
        + Math.min(20, vatFlagCount * 8)
        + Math.min(10, Math.max(0, anomaly - 0.6) * 30)
        + Math.min(8, Math.max(0, f3Trend) * 40)
        + (score >= 60 ? 6 : 0)
        + (f2 >= 1.2 ? 5 : 0)
    );
    const signalScore = Math.max(0, Math.min(100, Math.round(priorityRaw)));

    let queue = 'monitor';
    let level = 'low';
    let hasSignal = false;
    if (signalScore >= 80 || (f3 >= 0.7 && ratio >= 1.2) || vatFlagCount >= 2) {
        queue = 'priority_refund_audit';
        level = 'critical';
        hasSignal = true;
    } else if (signalScore >= 60 || (f3 >= 0.65 && ratio >= 1.05)) {
        queue = 'priority_refund_audit';
        level = 'high';
        hasSignal = true;
    } else if (signalScore >= 40 || f3 >= 0.55 || ratio >= 1.0) {
        queue = 'refund_watchlist';
        level = 'medium';
        hasSignal = true;
    }

    const rationaleParts = [];
    if (hasSignal) rationaleParts.push(`Mức ưu tiên hoàn thuế VAT: ${level.toUpperCase()} (${signalScore}/100).`);
    if (f3 >= 0.55) rationaleParts.push(`F3 VAT = ${f3.toFixed(2)} vượt ngưỡng theo dõi.`);
    if (ratio >= 1.0) rationaleParts.push(`Tỷ lệ VAT đầu vào/đầu ra = ${ratio.toFixed(2)}.`);
    if (refundGap > 0) rationaleParts.push(`Chênh lệch VAT đầu vào-đầu ra ước tính ${refundGap.toFixed(0)}.`);
    if (vatFlagCount > 0) rationaleParts.push(`Có ${vatFlagCount} red-flag liên quan VAT/hóa đơn.`);
    if (!rationaleParts.length) rationaleParts.push('Chưa ghi nhận tín hiệu hoàn thuế VAT bất thường trong kỳ hiện tại.');

    const checksMap = {
        priority_refund_audit: [
            'Đối chiếu hóa đơn VAT đầu vào theo chuỗi nhà cung cấp trước phê duyệt hoàn.',
            'Kiểm tra tính hợp lệ bộ chứng từ khấu trừ VAT của 3 kỳ gần nhất.',
            'Bổ sung xác minh giao dịch có VAT đầu vào giá trị lớn.',
        ],
        refund_watchlist: [
            'Đưa hồ sơ vào danh sách theo dõi hoàn thuế theo chu kỳ tháng/quý.',
            'Yêu cầu bổ sung giải trình cho hóa đơn đầu vào giá trị cao.',
            'Cảnh báo nếu F3 VAT tiếp tục tăng trong kỳ tiếp theo.',
        ],
        monitor: [
            'Duy trì giám sát thường xuyên, chưa cần kích hoạt kiểm tra chuyên đề.',
            'Cập nhật dữ liệu VAT kỳ mới để đánh giá lại xu hướng.',
        ],
    };

    return {
        has_signal: hasSignal,
        queue,
        level,
        score: signalScore,
        rationale: rationaleParts.join(' '),
        vat_input_output_ratio: Number.isFinite(ratio) ? ratio : null,
        estimated_refund_gap: Number.isFinite(refundGap) ? refundGap : 0,
        recommended_checks: checksMap[queue] || checksMap.monitor,
        indicators: indicators.slice(0, 5),
    };
}


function normalizeVatRefundSignals(vatRefundSignals, data) {
    const fallback = buildFallbackVatRefundSignals(data || {});
    if (!vatRefundSignals || typeof vatRefundSignals !== 'object') {
        return fallback;
    }

    const queue = ['monitor', 'refund_watchlist', 'priority_refund_audit'].includes(String(vatRefundSignals.queue || ''))
        ? String(vatRefundSignals.queue)
        : fallback.queue;
    const level = ['low', 'medium', 'high', 'critical'].includes(String(vatRefundSignals.level || ''))
        ? String(vatRefundSignals.level)
        : fallback.level;
    const score = Number.isFinite(Number(vatRefundSignals.score))
        ? Math.max(0, Math.min(100, Math.round(Number(vatRefundSignals.score))))
        : fallback.score;
    const ratio = Number.isFinite(Number(vatRefundSignals.vat_input_output_ratio))
        ? Number(vatRefundSignals.vat_input_output_ratio)
        : fallback.vat_input_output_ratio;
    const gap = Number.isFinite(Number(vatRefundSignals.estimated_refund_gap))
        ? Math.max(0, Number(vatRefundSignals.estimated_refund_gap))
        : fallback.estimated_refund_gap;
    const rationale = (typeof vatRefundSignals.rationale === 'string' && vatRefundSignals.rationale.trim())
        ? vatRefundSignals.rationale.trim()
        : fallback.rationale;
    const recommendedChecks = Array.isArray(vatRefundSignals.recommended_checks)
        ? vatRefundSignals.recommended_checks.map((it) => String(it || '').trim()).filter(Boolean).slice(0, 5)
        : [];
    const indicators = Array.isArray(vatRefundSignals.indicators)
        ? vatRefundSignals.indicators
            .filter((item) => item && typeof item === 'object')
            .map((item) => {
                const severity = String(item.severity || '').toLowerCase();
                return {
                    key: String(item.key || 'signal'),
                    label: String(item.label || 'Tín hiệu VAT'),
                    value: Number.isFinite(Number(item.value)) ? Number(item.value) : null,
                    severity: ['low', 'medium', 'high'].includes(severity) ? severity : 'medium',
                    summary: String(item.summary || '').trim(),
                };
            })
            .slice(0, 5)
        : [];

    return {
        has_signal: Boolean(vatRefundSignals.has_signal) || queue !== 'monitor' || level !== 'low',
        queue,
        level,
        score,
        rationale,
        vat_input_output_ratio: ratio,
        estimated_refund_gap: gap,
        recommended_checks: recommendedChecks.length ? recommendedChecks : fallback.recommended_checks,
        indicators: indicators.length ? indicators : fallback.indicators,
    };
}


function getVatRefundQueueClass(queue) {
    if (queue === 'priority_refund_audit') return 'bg-red-100 text-red-700 border border-red-200';
    if (queue === 'refund_watchlist') return 'bg-amber-100 text-amber-700 border border-amber-200';
    return 'bg-blue-100 text-blue-700 border border-blue-200';
}


function formatVatRefundIndicatorValue(indicator) {
    const value = Number(indicator?.value);
    if (!Number.isFinite(value)) return 'Không có';
    const key = String(indicator?.key || '');
    if (key === 'vat_input_output_ratio') return `${value.toFixed(2)}x`;
    if (key === 'estimated_refund_gap') return formatVND(value);
    if (key === 'vat_related_flags') return `${Math.round(value)} cảnh báo`;
    if (key === 'anomaly_score') return `${(value * 100).toFixed(1)}%`;
    if (key === 'f3_trend') return `${value > 0 ? '+' : ''}${value.toFixed(3)}`;
    return value.toFixed(3);
}


function renderVatRefundSignals(vatRefundSignals, data) {
    const panel = document.getElementById('vat-refund-signals-panel');
    if (!panel) return;

    const queueBadge = document.getElementById('vat-refund-queue-badge');
    const scoreEl = document.getElementById('vat-refund-score');
    const ratioEl = document.getElementById('vat-refund-ratio');
    const gapEl = document.getElementById('vat-refund-gap');
    const rationaleEl = document.getElementById('vat-refund-rationale');
    const checksEl = document.getElementById('vat-refund-checks');
    const indicatorsEl = document.getElementById('vat-refund-indicators');

    if (!queueBadge || !scoreEl || !ratioEl || !gapEl || !rationaleEl || !checksEl || !indicatorsEl) {
        panel.style.display = 'none';
        return;
    }

    const payload = normalizeVatRefundSignals(vatRefundSignals, data || {});
    panel.style.display = 'block';

    const queueLabelMap = {
        priority_refund_audit: 'Kiểm tra hoàn thuế ưu tiên',
        refund_watchlist: 'Danh sách theo dõi hoàn thuế',
        monitor: 'Giám sát',
    };

    queueBadge.className = `px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider ${getVatRefundQueueClass(payload.queue)}`;
    queueBadge.textContent = `${queueLabelMap[payload.queue] || 'Giám sát'} • ${String(payload.level || 'low').toUpperCase()}`;
    scoreEl.textContent = String(payload.score || 0);

    ratioEl.textContent = Number.isFinite(Number(payload.vat_input_output_ratio))
        ? `${Number(payload.vat_input_output_ratio).toFixed(2)}x`
        : '--';
    gapEl.textContent = Number.isFinite(Number(payload.estimated_refund_gap))
        ? formatVND(Number(payload.estimated_refund_gap))
        : '--';
    rationaleEl.textContent = payload.rationale || 'Chưa ghi nhận tín hiệu hoàn thuế VAT bất thường.';

    checksEl.innerHTML = '';
    (Array.isArray(payload.recommended_checks) ? payload.recommended_checks : []).forEach((check) => {
        const li = document.createElement('li');
        li.className = 'flex items-start gap-2';

        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined text-[14px] text-primary-container mt-0.5';
        icon.textContent = 'task_alt';

        const text = document.createElement('span');
        text.textContent = check;

        li.appendChild(icon);
        li.appendChild(text);
        checksEl.appendChild(li);
    });
    if (!checksEl.children.length) {
        const li = document.createElement('li');
        li.className = 'text-slate-500 italic';
        li.textContent = 'Chưa có kiểm tra bổ sung được đề xuất.';
        checksEl.appendChild(li);
    }

    indicatorsEl.innerHTML = '';
    const toneBySeverity = {
        high: 'bg-red-50 border-red-200 text-red-700',
        medium: 'bg-amber-50 border-amber-200 text-amber-700',
        low: 'bg-emerald-50 border-emerald-200 text-emerald-700',
    };

    (Array.isArray(payload.indicators) ? payload.indicators : []).forEach((indicator) => {
        const tone = toneBySeverity[String(indicator?.severity || 'medium')] || toneBySeverity.medium;
        const card = document.createElement('div');
        card.className = `rounded-lg border px-3 py-2 ${tone}`;

        const title = document.createElement('p');
        title.className = 'text-[10px] font-black uppercase tracking-wider';
        title.textContent = `${indicator.label || 'Tín hiệu VAT'}: ${formatVatRefundIndicatorValue(indicator)}`;

        const summary = document.createElement('p');
        summary.className = 'text-[10px] mt-1 leading-relaxed';
        summary.textContent = indicator.summary || 'Chỉ báo VAT cần được xem xét bổ sung.';

        card.appendChild(title);
        card.appendChild(summary);
        indicatorsEl.appendChild(card);
    });

    if (!indicatorsEl.children.length) {
        const placeholder = document.createElement('div');
        placeholder.className = 'rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[10px] text-slate-500 italic';
        placeholder.textContent = 'Không có tín hiệu VAT ưu tiên tại thời điểm đánh giá.';
        indicatorsEl.appendChild(placeholder);
    }
}


function buildFallbackAuditValue(data) {
    const score = Math.max(0, Math.min(100, Number(data?.risk_score || 0)));
    const anomaly = Math.max(0, Number(data?.anomaly_score || 0));
    const confidenceRaw = Math.max(0, Math.min(100, Number(data?.model_confidence || 0)));
    const rev = Math.max(0, Number(data?.revenue || 0));
    const exp = Math.max(0, Number(data?.total_expenses || 0));

    const redFlags = Array.isArray(data?.red_flags) ? data.red_flags : [];
    const highRedFlagCount = redFlags.filter((flag) => {
        const severity = String(flag?.severity || '').toLowerCase();
        return severity === 'high' || severity === 'critical';
    }).length;

    const yearlyFeatureScores = Array.isArray(data?.yearly_feature_scores) ? data.yearly_feature_scores : [];
    let trendDelta = 0;
    if (yearlyFeatureScores.length >= 2) {
        const sorted = [...yearlyFeatureScores].sort((a, b) => Number(a?.year || 0) - Number(b?.year || 0));
        trendDelta = Number(sorted[sorted.length - 1]?.risk_score || 0) - Number(sorted[sorted.length - 2]?.risk_score || 0);
    }

    const marginGap = Math.max(0, exp - rev);
    const exposureBase = (
        marginGap * 0.08
        + rev * 0.012
        + redFlags.length * 900000
        + highRedFlagCount * 2200000
        + Math.max(0, anomaly - 0.55) * 25000000
        + Math.max(0, trendDelta) * 450000
    );

    let recoverability = (
        0.18
        + (score / 100) * 0.45
        + Math.min(0.15, Math.max(0, anomaly - 0.5) * 0.45)
        + Math.min(0.14, highRedFlagCount * 0.04)
        + (trendDelta >= 10 ? 0.05 : 0)
    );
    if (confidenceRaw > 0 && confidenceRaw < 55) recoverability -= 0.06;
    recoverability = Math.max(0.08, Math.min(0.86, recoverability));

    const estimatedRecovery = Math.max(0, exposureBase) * recoverability;
    const auditHoursEstimate = Math.max(
        8,
        Math.min(120, 14 + score * 0.62 + highRedFlagCount * 4.5 + Math.min(18, Math.max(0, trendDelta) * 0.7)),
    );
    const estimatedAuditCost = auditHoursEstimate * 650000;
    const expectedNetRecovery = Math.max(0, estimatedRecovery - estimatedAuditCost);

    const netMillion = expectedNetRecovery / 1000000;
    let priorityScore = Math.round(
        score * 0.55
        + Math.min(24, netMillion * 0.75)
        + Math.min(12, highRedFlagCount * 3)
        + (recoverability >= 0.6 ? 5 : 0),
    );
    priorityScore = Math.max(0, Math.min(100, priorityScore));

    let recommendedLane = 'monitor';
    if (expectedNetRecovery >= 2000000000 || priorityScore >= 80) {
        recommendedLane = 'priority_audit';
    } else if (expectedNetRecovery >= 800000000 || priorityScore >= 62) {
        recommendedLane = 'targeted_audit';
    } else if (expectedNetRecovery >= 200000000 || priorityScore >= 45) {
        recommendedLane = 'desk_review';
    }

    let confidence = 'low';
    if (confidenceRaw >= 80) confidence = 'high';
    else if (confidenceRaw >= 55) confidence = 'medium';
    if (confidence === 'low' && yearlyFeatureScores.length >= 4) confidence = 'medium';

    const rationaleParts = [
        `Giá trị truy thu kỳ vọng ước tính ${formatVND(estimatedRecovery)}.`,
        `Giá trị thuần sau chi phí ước tính ${formatVND(expectedNetRecovery)}.`,
        `Điểm rủi ro hiện tại ${score.toFixed(1)}/100 với khả năng thu hồi ${(recoverability * 100).toFixed(1)}%.`,
    ];
    if (highRedFlagCount > 0) rationaleParts.push(`Phát hiện ${highRedFlagCount} red flag mức cao cần ưu tiên hồ sơ.`);
    if (trendDelta >= 8) rationaleParts.push(`Xu hướng rủi ro tăng ${trendDelta.toFixed(1)} điểm so với kỳ trước.`);

    return {
        estimated_recovery: estimatedRecovery,
        expected_net_recovery: expectedNetRecovery,
        recoverability_ratio: recoverability,
        audit_hours_estimate: auditHoursEstimate,
        estimated_audit_cost: estimatedAuditCost,
        priority_score: priorityScore,
        recommended_lane: recommendedLane,
        confidence,
        rationale: rationaleParts.join(' '),
        drivers: [
            {
                key: 'expected_recovery',
                label: 'Giá trị truy thu ước tính',
                value: estimatedRecovery,
                impact: estimatedRecovery >= 1000000000 ? 'high' : 'medium',
                summary: 'Tổng giá trị có thể truy thu nếu mở thanh tra hồ sơ này.',
            },
            {
                key: 'recoverability_ratio',
                label: 'Tỷ lệ thu hồi',
                value: recoverability,
                impact: recoverability >= 0.6 ? 'high' : 'medium',
                summary: 'Ước tính khả năng biến rủi ro thành giá trị thu hồi thực tế.',
            },
            {
                key: 'audit_cost',
                label: 'Chi phí thanh tra',
                value: estimatedAuditCost,
                impact: estimatedAuditCost <= 60000000 ? 'low' : 'medium',
                summary: 'Nguồn lực cán bộ ước tính để hoàn tất một chu kỳ xử lý.',
            },
            {
                key: 'risk_momentum',
                label: 'Động lực rủi ro',
                value: trendDelta,
                impact: trendDelta >= 12 ? 'high' : (trendDelta >= 4 ? 'medium' : 'low'),
                summary: 'Tốc độ tăng điểm rủi ro theo chuỗi năm phân tích.',
            },
        ],
    };
}


function normalizeAuditValue(auditValue, data) {
    const fallback = buildFallbackAuditValue(data || {});
    if (!auditValue || typeof auditValue !== 'object') return fallback;

    const lane = String(auditValue.recommended_lane || '').trim();
    const allowedLane = new Set(['monitor', 'desk_review', 'targeted_audit', 'priority_audit']);
    const confidenceRaw = String(auditValue.confidence || '').trim().toLowerCase();
    const allowedConfidence = new Set(['low', 'medium', 'high']);

    const drivers = Array.isArray(auditValue.drivers)
        ? auditValue.drivers
            .filter((driver) => driver && typeof driver === 'object')
            .map((driver) => {
                const impact = String(driver.impact || '').toLowerCase();
                return {
                    key: String(driver.key || 'driver'),
                    label: String(driver.label || 'Yếu tố'),
                    value: Number.isFinite(Number(driver.value)) ? Number(driver.value) : null,
                    impact: impact === 'high' || impact === 'medium' || impact === 'low' ? impact : 'medium',
                    summary: String(driver.summary || '').trim(),
                };
            })
            .slice(0, 6)
        : [];

    return {
        estimated_recovery: Number.isFinite(Number(auditValue.estimated_recovery))
            ? Math.max(0, Number(auditValue.estimated_recovery))
            : Number(fallback.estimated_recovery || 0),
        expected_net_recovery: Number.isFinite(Number(auditValue.expected_net_recovery))
            ? Math.max(0, Number(auditValue.expected_net_recovery))
            : Number(fallback.expected_net_recovery || 0),
        recoverability_ratio: Number.isFinite(Number(auditValue.recoverability_ratio))
            ? Math.max(0, Math.min(1, Number(auditValue.recoverability_ratio)))
            : Number(fallback.recoverability_ratio || 0),
        audit_hours_estimate: Number.isFinite(Number(auditValue.audit_hours_estimate))
            ? Math.max(0, Number(auditValue.audit_hours_estimate))
            : Number(fallback.audit_hours_estimate || 0),
        estimated_audit_cost: Number.isFinite(Number(auditValue.estimated_audit_cost))
            ? Math.max(0, Number(auditValue.estimated_audit_cost))
            : Number(fallback.estimated_audit_cost || 0),
        priority_score: Number.isFinite(Number(auditValue.priority_score))
            ? Math.max(0, Math.min(100, Math.round(Number(auditValue.priority_score))))
            : Number(fallback.priority_score || 0),
        recommended_lane: allowedLane.has(lane) ? lane : String(fallback.recommended_lane || 'monitor'),
        confidence: allowedConfidence.has(confidenceRaw) ? confidenceRaw : String(fallback.confidence || 'medium'),
        rationale: (typeof auditValue.rationale === 'string' && auditValue.rationale.trim())
            ? auditValue.rationale.trim()
            : String(fallback.rationale || ''),
        drivers: drivers.length ? drivers : (Array.isArray(fallback.drivers) ? fallback.drivers : []),
    };
}


function getAuditValueLaneClass(lane) {
    if (lane === 'priority_audit') return 'bg-red-100 text-red-700 border border-red-200';
    if (lane === 'targeted_audit') return 'bg-orange-100 text-orange-700 border border-orange-200';
    if (lane === 'desk_review') return 'bg-blue-100 text-blue-700 border border-blue-200';
    return 'bg-emerald-100 text-emerald-700 border border-emerald-200';
}


function formatAuditDriverValue(driver) {
    const value = Number(driver?.value);
    if (!Number.isFinite(value)) return 'Không có';
    const key = String(driver?.key || '');

    if (key === 'recoverability_ratio') return `${(value * 100).toFixed(1)}%`;
    if (key === 'risk_momentum') return `${value >= 0 ? '+' : ''}${value.toFixed(1)} điểm`;
    if (key.includes('cost') || key.includes('recovery')) return formatVND(value);
    return value.toFixed(2);
}


function renderAuditValue(auditValue, data) {
    const panel = document.getElementById('audit-value-panel');
    if (!panel) return;

    const laneBadge = document.getElementById('audit-value-lane-badge');
    const priorityEl = document.getElementById('audit-value-priority');
    const recoveryEl = document.getElementById('audit-value-recovery');
    const netRecoveryEl = document.getElementById('audit-value-net-recovery');
    const recoverabilityEl = document.getElementById('audit-value-recoverability');
    const auditHoursEl = document.getElementById('audit-value-audit-hours');
    const costEl = document.getElementById('audit-value-cost');
    const rationaleEl = document.getElementById('audit-value-rationale');
    const driversEl = document.getElementById('audit-value-drivers');

    if (!laneBadge || !priorityEl || !recoveryEl || !netRecoveryEl || !recoverabilityEl || !auditHoursEl || !costEl || !rationaleEl || !driversEl) {
        panel.style.display = 'none';
        return;
    }

    const payload = normalizeAuditValue(auditValue, data || {});
    panel.style.display = 'block';

    const laneLabelMap = {
        priority_audit: 'Thanh tra ưu tiên',
        targeted_audit: 'Thanh tra mục tiêu',
        desk_review: 'Rà soát hồ sơ',
        monitor: 'Giám sát',
    };

    laneBadge.className = `px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider ${getAuditValueLaneClass(payload.recommended_lane)}`;
    laneBadge.textContent = `${laneLabelMap[payload.recommended_lane] || 'Giám sát'} • ${String(payload.confidence || 'medium').toUpperCase()}`;
    priorityEl.textContent = String(payload.priority_score || 0);

    recoveryEl.textContent = formatVND(Number(payload.estimated_recovery || 0));
    netRecoveryEl.textContent = formatVND(Number(payload.expected_net_recovery || 0));
    recoverabilityEl.textContent = `${(Number(payload.recoverability_ratio || 0) * 100).toFixed(1)}%`;
    auditHoursEl.textContent = `${Number(payload.audit_hours_estimate || 0).toFixed(1)} h`;
    costEl.textContent = formatVND(Number(payload.estimated_audit_cost || 0));
    rationaleEl.textContent = payload.rationale || 'Chưa có dữ liệu ước tính giá trị thanh tra cho hồ sơ này.';

    driversEl.innerHTML = '';
    const toneByImpact = {
        high: 'bg-red-50 border-red-200 text-red-700',
        medium: 'bg-amber-50 border-amber-200 text-amber-700',
        low: 'bg-emerald-50 border-emerald-200 text-emerald-700',
    };

    (Array.isArray(payload.drivers) ? payload.drivers : []).forEach((driver) => {
        const tone = toneByImpact[String(driver?.impact || 'medium')] || toneByImpact.medium;
        const card = document.createElement('div');
        card.className = `rounded-lg border px-3 py-2 ${tone}`;

        const title = document.createElement('p');
        title.className = 'text-[10px] font-black uppercase tracking-wider';
        title.textContent = `${driver.label || 'Yếu tố giá trị'}: ${formatAuditDriverValue(driver)}`;

        const summary = document.createElement('p');
        summary.className = 'text-[10px] mt-1 leading-relaxed';
        summary.textContent = driver.summary || 'Yếu tố này được sử dụng để xếp hạng giá trị thanh tra.';

        card.appendChild(title);
        card.appendChild(summary);
        driversEl.appendChild(card);
    });

    if (!driversEl.children.length) {
        const placeholder = document.createElement('div');
        placeholder.className = 'rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[10px] text-slate-500 italic';
        placeholder.textContent = 'Không có yếu tố giá trị bổ sung trong kỳ đánh giá.';
        driversEl.appendChild(placeholder);
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

    const finalOffset = 552.92 - (552.92 * (targetScore / 100));
    if (PREFERS_REDUCED_MOTION) {
        scoreEl.textContent = Math.round(targetScore);
        circleEl.style.strokeDashoffset = finalOffset;
        return;
    }

    const durationMs = MOTION_DURATION_SCORE;
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
            circleEl.style.strokeDashoffset = finalOffset;
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
        const safeIcon = /^[a-z0-9_]+$/i.test(String(flag.icon || '')) ? String(flag.icon) : 'warning';
        const safeTitle = flag.title ? String(flag.title) : 'Dấu hiệu bất thường';
        const safeDescription = flag.description ? String(flag.description) : '';

        const div = document.createElement('div');
        div.className = 'red-flag-item group bg-error-container/20 p-5 rounded-xl border-l-4 border-error flex items-start gap-4 transition-all hover:translate-x-1';
        div.style.animationDelay = `${i * 0.1}s`;

        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined text-error';
        icon.style.fontVariationSettings = "'FILL' 1";
        icon.textContent = safeIcon;

        const textWrap = document.createElement('div');

        const titleEl = document.createElement('p');
        titleEl.className = 'text-sm font-bold text-on-error-container';
        titleEl.textContent = safeTitle;

        const descEl = document.createElement('p');
        descEl.className = 'text-xs text-slate-600 mt-1';
        descEl.textContent = safeDescription;

        textWrap.appendChild(titleEl);
        textWrap.appendChild(descEl);
        div.appendChild(icon);
        div.appendChild(textWrap);
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
        const featureLabel = escapeHtml(formatFeatureName(s.feature));
        const safeDirLabel = escapeHtml(dirLabel);
        const safeShapVal = escapeHtml(shapVal);
        const div = document.createElement('div');
        div.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="text-[10px] font-bold text-slate-500 uppercase tracking-wider">${featureLabel}</span>
                <div class="flex items-center gap-2">
                    <span class="text-[9px] font-bold ${dirColor}">${safeDirLabel}</span>
                    <span class="text-[10px] font-mono text-slate-400">${safeShapVal}</span>
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
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload thất bại');
        }

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
    stopPollingLoop();
    const pollingToken = ++_pollingRequestToken;
    let pollingInFlight = false;

    const pollOnce = async () => {
        if (pollingInFlight || pollingToken !== _pollingRequestToken || document.hidden) {
            return;
        }

        pollingInFlight = true;
        try {
            const response = await secureFetch(`${API_BASE}/ai/batch-status/${batchId}`);
            if (pollingToken !== _pollingRequestToken || !response.ok) return;
            const status = await response.json();
            if (pollingToken !== _pollingRequestToken || !status) return;
        


            const pct = status.progress_percent || 0;
            const normalizedStatus = String(status.status || '').toLowerCase();
            document.getElementById('progress-bar').style.width = `${pct}%`;
            document.getElementById('progress-percent').textContent = `${pct}%`;
            const baseProgressText = `Đã phân tích ${status.processed_rows || 0} / ${status.total_rows || '?'} doanh nghiệp`;
            const isFinalizingTail = normalizedStatus === 'finalizing' || (pct >= 100 && normalizedStatus === 'processing');
            document.getElementById('progress-detail').textContent = isFinalizingTail
                ? `${baseProgressText} • đang đồng bộ kết quả vào hệ thống`
                : baseProgressText;

            if (isFinalizingTail) {
                document.getElementById('progress-title').textContent = 'Đang hoàn tất lưu kết quả...';
            }
            const progressTrack = document.getElementById('progress-track');
            if (progressTrack) {
                progressTrack.setAttribute('aria-valuenow', String(pct));
                progressTrack.setAttribute('aria-valuetext', `${pct}%`);
            }

            if (normalizedStatus === 'done') {
                stopPollingLoop();
                showToast('Phân tích hoàn tất!', `Đã xử lý thành công ${status.processed_rows} doanh nghiệp.`, 'success', 6000);
                loadBatchResults(batchId);
            } else if (normalizedStatus === 'failed') {
                stopPollingLoop();
                showToast('Phân tích thất bại', status.error_message || 'Lỗi không xác định', 'error', 8000);
                resetBatchUI();
            }
        } catch (err) {
            console.error('Polling error:', err);
        } finally {
            pollingInFlight = false;
        }
    };

    pollingInterval = setInterval(() => {
        void pollOnce();
    }, 1500);

    void pollOnce();
}


function stopPollingLoop() {
    _pollingRequestToken += 1;
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
    if (_pollingWatchdog) {
        clearTimeout(_pollingWatchdog);
        _pollingWatchdog = null;
    }
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
    const progressTrack = document.getElementById('progress-track');
    if (progressTrack) {
        progressTrack.setAttribute('aria-valuenow', '0');
        progressTrack.setAttribute('aria-valuetext', '0%');
    }
    stopPollingLoop();
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

let singleCompanyMode = 'all';
let singleCompanyPage = 1;
let singleCompanyTotal = 0;
const SINGLE_COMPANY_PAGE_SIZE = 10;
let singleCompanyRows = [];
let singleCompanySearchDebounce = null;

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
    renderRevenueRiskScatter(stats.revenue_risk_scatter || stats.scatter_data || [], data.assessments || []);
    renderHistogram(stats.risk_distribution || []);
    renderRadarChart(stats, data.assessments || []);
    renderDonutChart(summary);
    renderIndustryChart(stats.industry_stats || []);
    renderCorrelationMatrix(stats.correlation_matrix || {});
    renderBoxPlot(stats.box_plot_data || []);
    renderMapChart(stats.province_stats || []);
    renderCohortRiskFunnel(stats.cohort_transition_sankey || {});
    renderVatAnomalyHeatmap(stats.vat_anomaly_heatmap || {});
    renderCumulativeRiskCurve(stats.cumulative_risk_curve || {});
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
        animationDuration: MOTION_DURATION_CHART,
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


// ---- REVENUE vs RISK SCATTER (Business-friendly view) ----
function renderRevenueRiskScatter(revenueRiskData, assessments = []) {
    const container = document.getElementById('chart-revenue-risk');
    if (!container) return;

    const source = Array.isArray(revenueRiskData) && revenueRiskData.length
        ? revenueRiskData
        : (Array.isArray(assessments) ? assessments : []);

    const normalized = source
        .map((row) => {
            const revenue = Number(row.revenue || 0);
            const totalExpenses = Number(row.total_expenses || 0);
            const ratio = Number(row.expense_ratio);
            const expenseRatio = Number.isFinite(ratio)
                ? ratio
                : (revenue > 0 ? totalExpenses / Math.max(revenue, 1) : 0);

            return {
                tax_code: String(row.tax_code || ''),
                company_name: String(row.company_name || ''),
                industry: String(row.industry || 'Khác'),
                revenue,
                risk_score: Number(row.risk_score || 0),
                expense_ratio: Number.isFinite(expenseRatio) ? expenseRatio : 0,
                log_revenue: Math.log10(Math.max(0, revenue) + 1),
            };
        })
        .filter((row) => Number.isFinite(row.log_revenue) && Number.isFinite(row.risk_score));

    if (!normalized.length) {
        renderSingleTrendChartMessage(container, 'Không đủ dữ liệu để vẽ biểu đồ phân tán Doanh thu vs Rủi ro.');
        return;
    }

    const chart = safeInitChart(container);
    const industries = [...new Set(normalized.map((d) => d.industry))];
    const palette = ['#002147', '#dc2626', '#ea580c', '#16a34a', '#7c3aed', '#0891b2', '#be123c', '#854d0e', '#1d4ed8', '#4d7c0f'];

    const series = industries.map((industry, idx) => {
        const points = normalized.filter((d) => d.industry === industry);
        return {
            name: industry,
            type: 'scatter',
            symbolSize: (value) => Math.max(7, Math.min(24, Number(value[2] || 0.2) * 14)),
            itemStyle: {
                color: palette[idx % palette.length],
                opacity: 0.75,
            },
            data: points.map((d) => ({
                value: [d.log_revenue, d.risk_score, d.expense_ratio],
                _meta: d,
            })),
            emphasis: {
                itemStyle: {
                    opacity: 1,
                    shadowBlur: 10,
                    shadowColor: 'rgba(15, 23, 42, 0.3)',
                },
            },
        };
    });

    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        animationEasing: 'cubicOut',
        legend: {
            type: 'scroll',
            bottom: 0,
            textStyle: { fontSize: 9 },
        },
        grid: { left: '12%', right: '5%', top: '8%', bottom: '22%' },
        tooltip: {
            trigger: 'item',
            formatter: (params) => {
                const d = params.data && params.data._meta ? params.data._meta : null;
                if (!d) return '';
                return `<b>${escapeHtml(d.company_name || '---')}</b><br>`
                    + `MST: ${escapeHtml(d.tax_code || '---')}<br>`
                    + `Ngành: ${escapeHtml(d.industry || '---')}<br>`
                    + `Doanh thu: <b>${formatVND(d.revenue)}</b><br>`
                    + `Điểm rủi ro: <b>${d.risk_score.toFixed(1)}</b><br>`
                    + `Tỷ lệ Chi phí/DT: <b>${(d.expense_ratio * 100).toFixed(1)}%</b>`;
            },
        },
        xAxis: {
            type: 'value',
            name: 'Log10(Doanh thu)',
            nameLocation: 'center',
            nameGap: 28,
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } },
            axisLabel: { fontSize: 10 },
        },
        yAxis: {
            type: 'value',
            name: 'Điểm rủi ro',
            min: 0,
            max: 100,
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } },
            axisLabel: { fontSize: 10 },
        },
        series,
        dataZoom: [
            { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
            { type: 'slider', xAxisIndex: 0, height: 16, bottom: 28 },
        ],
    });
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
        animationDuration: MOTION_DURATION_CHART,
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
    if (!select) return;
    select.innerHTML = '';

    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Tất cả ngành';
    select.appendChild(defaultOption);

    industryStats.forEach(ind => {
        const option = document.createElement('option');
        option.value = String(ind.industry || '');
        option.textContent = `${ind.industry || '---'} (${Number(ind.company_count || 0)})`;
        select.appendChild(option);
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
        const rawTaxCode = String(a.tax_code || '');
        const safeTaxCode = escapeHtml(rawTaxCode || '---');
        const safeCompanyName = escapeHtml(a.company_name || '---');
        const safeIndustry = escapeHtml(a.industry || '---');
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-surface-container-low/50 transition-colors cursor-pointer';
        tr.innerHTML = `
            <td class="px-4 py-3 font-mono text-slate-400">${globalIdx}</td>
            <td class="px-4 py-3 font-mono font-bold">${safeTaxCode}</td>
            <td class="px-4 py-3 font-medium max-w-[200px] truncate">${safeCompanyName}</td>
            <td class="px-4 py-3 text-slate-500">${safeIndustry}</td>
            <td class="px-4 py-3 font-mono">${formatVND(a.revenue)}</td>
            <td class="px-4 py-3 font-black text-lg">${a.risk_score.toFixed(1)}</td>
            <td class="px-4 py-3 font-mono text-slate-500">${(a.anomaly_score || 0).toFixed(3)}</td>
            <td class="px-4 py-3">
                <span class="px-2 py-1 rounded text-[9px] font-black uppercase tracking-wider ${levelClass}">
                    ${getRiskLabel(a.risk_level)}
                </span>
            </td>
            <td class="px-4 py-3 text-right">
                <div class="inline-flex items-center gap-2">
                    <button type="button" class="batch-copy-btn px-2.5 py-1.5 rounded-lg bg-surface-container text-on-surface-variant text-[9px] font-black uppercase tracking-wider hover:bg-surface-container-high" data-tax-code="${safeTaxCode}">Copy MST</button>
                    <button type="button" class="batch-analyze-btn px-2.5 py-1.5 rounded-lg bg-primary-container text-white text-[9px] font-black uppercase tracking-wider hover:bg-primary" data-tax-code="${safeTaxCode}">Phân tích</button>
                </div>
            </td>`;
        tbody.appendChild(tr);
    });

    tbody.querySelectorAll('.batch-copy-btn').forEach((btn) => {
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            const taxCode = btn.getAttribute('data-tax-code') || '';
            copyTaxCodeToClipboard(taxCode);
        });
    });

    tbody.querySelectorAll('.batch-analyze-btn').forEach((btn) => {
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            const taxCode = btn.getAttribute('data-tax-code') || '';
            analyzeSingleCompanyFromBatch(taxCode);
        });
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
    controlsEl.innerHTML += `<button type="button" class="${currentPage === 1 ? btnDisabled : btnInactive}" ${currentPage === 1 ? 'disabled' : ''} data-page="${currentPage - 1}" aria-label="Trang trước">
        <span class="material-symbols-outlined text-[14px]">chevron_left</span>
    </button>`;

    // Page numbers (show max 7 buttons)
    let startPage = Math.max(1, currentPage - 3);
    let endPage = Math.min(totalPages, startPage + 6);
    if (endPage - startPage < 6) startPage = Math.max(1, endPage - 6);

    if (startPage > 1) {
        controlsEl.innerHTML += `<button type="button" class="${btnInactive}" data-page="1">1</button>`;
        if (startPage > 2) controlsEl.innerHTML += `<span class="px-1 text-slate-300">...</span>`;
    }

    for (let i = startPage; i <= endPage; i++) {
        controlsEl.innerHTML += `<button type="button" class="${i === currentPage ? btnActive : btnInactive}" data-page="${i}">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) controlsEl.innerHTML += `<span class="px-1 text-slate-300">...</span>`;
        controlsEl.innerHTML += `<button type="button" class="${btnInactive}" data-page="${totalPages}">${totalPages}</button>`;
    }

    // Next
    controlsEl.innerHTML += `<button type="button" class="${currentPage === totalPages ? btnDisabled : btnInactive}" ${currentPage === totalPages ? 'disabled' : ''} data-page="${currentPage + 1}" aria-label="Trang sau">
        <span class="material-symbols-outlined text-[14px]">chevron_right</span>
    </button>`;

    controlsEl.querySelectorAll('button[data-page]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const page = Number(btn.getAttribute('data-page'));
            if (Number.isFinite(page)) goToPage(page);
        });
    });
}


function goToPage(page) {
    const totalPages = Math.max(1, Math.ceil(filteredAssessments.length / ROWS_PER_PAGE));
    if (page < 1 || page > totalPages) return;
    currentPage = page;
    renderPaginatedTable();
    // Smooth scroll to table top
    document.getElementById('results-table-body').closest('.bg-surface-container-lowest').scrollIntoView({ behavior: 'smooth', block: 'start' });
}


function analyzeSingleCompanyFromBatch(taxCode) {
    if (!taxCode) return;
    analyzeSingleCompanyFromDirectory(taxCode);
}


async function copyTaxCodeToClipboard(taxCode) {
    if (!taxCode) return;

    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(taxCode);
        } else {
            const tempInput = document.createElement('textarea');
            tempInput.value = taxCode;
            tempInput.setAttribute('readonly', 'readonly');
            tempInput.style.position = 'fixed';
            tempInput.style.left = '-9999px';
            document.body.appendChild(tempInput);
            tempInput.select();
            document.execCommand('copy');
            document.body.removeChild(tempInput);
        }
        showToast('Đã copy MST', `MST ${taxCode} đã được sao chép.`, 'success');
    } catch (error) {
        console.error(error);
        showToast('Copy thất bại', 'Không thể sao chép MST vào clipboard.', 'warning');
    }
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
// SINGLE-QUERY COMPANY DIRECTORY (All DB vs Assessed)
// ===================================================================

function updateSingleCompanyModeButtons() {
    const allBtn = document.getElementById('single-company-mode-all');
    const assessedBtn = document.getElementById('single-company-mode-assessed');
    if (!allBtn || !assessedBtn) return;

    if (singleCompanyMode === 'all') {
        allBtn.className = 'px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-wider bg-primary-container text-white';
        assessedBtn.className = 'px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-wider bg-surface-container text-on-surface-variant border border-outline-variant/20';
    } else {
        allBtn.className = 'px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-wider bg-surface-container text-on-surface-variant border border-outline-variant/20';
        assessedBtn.className = 'px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-wider bg-primary-container text-white';
    }
}


function setSingleCompanyMode(mode) {
    if (!['all', 'assessed'].includes(mode)) return;
    if (singleCompanyMode === mode) return;
    singleCompanyMode = mode;
    singleCompanyPage = 1;
    updateSingleCompanyModeButtons();
    loadSingleCompanyDirectory();
}


function formatDirectoryTimestamp(raw) {
    if (!raw) return '--';
    const date = new Date(raw);
    if (Number.isNaN(date.getTime())) return '--';
    return date.toLocaleString('vi-VN', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}


function getDirectoryInterventionFallback(score) {
    const normalizedScore = Math.max(0, Math.min(100, Number(score || 0)));
    if (normalizedScore >= 82) return { action: 'escalated_enforcement', priority: Math.round(normalizedScore) };
    if (normalizedScore >= 65) return { action: 'field_audit', priority: Math.round(normalizedScore) };
    if (normalizedScore >= 45) return { action: 'structured_outreach', priority: Math.round(normalizedScore) };
    if (normalizedScore >= 25) return { action: 'auto_reminder', priority: Math.round(normalizedScore) };
    return { action: 'monitor', priority: Math.round(normalizedScore) };
}


function normalizeDirectoryIntervention(row) {
    const fallback = getDirectoryInterventionFallback(row?.risk_score);
    const actionRaw = String(row?.intervention_action || '').trim().toLowerCase();
    const allowedActions = new Set(['monitor', 'auto_reminder', 'structured_outreach', 'field_audit', 'escalated_enforcement']);
    const action = allowedActions.has(actionRaw) ? actionRaw : fallback.action;
    const priority = Number.isFinite(Number(row?.intervention_priority))
        ? Math.max(0, Math.min(100, Math.round(Number(row.intervention_priority))))
        : fallback.priority;

    return { action, priority };
}


function getDirectoryInterventionLabel(action) {
    if (action === 'escalated_enforcement') return 'Cưỡng chế';
    if (action === 'field_audit') return 'Kiểm tra tại chỗ';
    if (action === 'structured_outreach') return 'Can thiệp có cấu trúc';
    if (action === 'auto_reminder') return 'Nhắc hạn tự động';
    return 'Giám sát';
}


function getDirectoryInterventionBadgeClass(action) {
    if (action === 'escalated_enforcement') return 'bg-rose-50 text-rose-700 border border-rose-100';
    if (action === 'field_audit') return 'bg-orange-50 text-orange-700 border border-orange-100';
    if (action === 'structured_outreach') return 'bg-amber-50 text-amber-700 border border-amber-100';
    if (action === 'auto_reminder') return 'bg-blue-50 text-blue-700 border border-blue-100';
    return 'bg-slate-100 text-slate-600 border border-slate-200';
}


function normalizeDelinquencyKeyword(value) {
    return String(value || '').trim().slice(0, 80);
}


function openDelinquencyWithInterventionFilter(action, searchKeyword = '') {
    const actionRaw = String(action || '').trim().toLowerCase();
    const allowedActions = new Set(['monitor', 'auto_reminder', 'structured_outreach', 'field_audit', 'escalated_enforcement']);
    const normalizedAction = allowedActions.has(actionRaw) ? actionRaw : 'monitor';
    const normalizedKeyword = normalizeDelinquencyKeyword(searchKeyword);

    const targetUrl = new URL('delinquency.html', window.location.href);
    targetUrl.searchParams.set('intervention_filter', normalizedAction);
    targetUrl.searchParams.set('sort_by', 'intervention_priority_desc');
    if (normalizedKeyword) {
        targetUrl.searchParams.set('q', normalizedKeyword);
    } else {
        targetUrl.searchParams.delete('q');
    }
    targetUrl.searchParams.delete('page');

    window.location.href = targetUrl.toString();
}


function renderSingleCompanyDirectoryTable() {
    const tbody = document.getElementById('single-company-table-body');
    const summary = document.getElementById('single-company-summary');
    if (!tbody || !summary) return;

    if (!singleCompanyRows.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-8 text-center text-slate-400 text-xs italic">Không tìm thấy doanh nghiệp phù hợp.</td></tr>';
        summary.textContent = 'Không có dữ liệu theo bộ lọc hiện tại.';
        return;
    }

    tbody.innerHTML = singleCompanyRows
        .map((row) => {
            const taxCode = escapeHtml(row.tax_code || '');
            const name = escapeHtml(row.name || '---');
            const industry = escapeHtml(row.industry || '---');
            const score = Number(row.risk_score || 0);
            const intervention = normalizeDirectoryIntervention(row);
            const interventionLabel = getDirectoryInterventionLabel(intervention.action);
            const interventionClass = getDirectoryInterventionBadgeClass(intervention.action);
            const scoreClass = score >= 80
                ? 'text-red-600'
                : score >= 60
                    ? 'text-orange-600'
                    : score >= 40
                        ? 'text-yellow-600'
                        : 'text-emerald-600';
            const updatedAt = formatDirectoryTimestamp(row.latest_assessment_at);
            const assessedTag = row.assessed
                ? '<span class="px-2 py-0.5 rounded bg-blue-50 text-blue-700 text-[9px] font-black uppercase tracking-wider">AI</span>'
                : '<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-500 text-[9px] font-black uppercase tracking-wider">CHƯA CHẤM</span>';

            return `
                <tr class="group hover:bg-slate-50 transition-all duration-300">
                    <td class="px-4 py-3 font-mono font-bold text-primary-container">${taxCode}</td>
                    <td class="px-4 py-3">
                        <div class="font-semibold text-on-surface text-xs">${name}</div>
                        <div class="mt-1">${assessedTag}</div>
                    </td>
                    <td class="px-4 py-3 text-slate-500">${industry}</td>
                    <td class="px-4 py-3 font-black ${scoreClass}">${score.toFixed(1)}</td>
                    <td class="px-4 py-3">
                        <div class="inline-flex flex-col gap-1.5">
                            <button
                                type="button"
                                class="single-company-intervention-btn px-2 py-0.5 rounded text-[9px] font-black uppercase tracking-wider transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_8px_16px_rgba(0,33,71,0.12)] ${interventionClass}"
                                data-intervention-action="${escapeHtml(intervention.action)}"
                                title="Mở trang Nợ đọng với bộ lọc can thiệp tương ứng"
                            >${escapeHtml(interventionLabel)}</button>
                            <span class="text-[9px] font-bold text-slate-500">P${intervention.priority}</span>
                        </div>
                    </td>
                    <td class="px-4 py-3 text-slate-500 text-[10px]">${updatedAt}</td>
                    <td class="px-4 py-3 text-right">
                        <button class="single-company-analyze-btn px-3 py-1.5 rounded-lg bg-primary-container text-white text-[10px] font-black uppercase tracking-wider hover:bg-primary transition-colors" data-tax-code="${taxCode}">Phân tích</button>
                    </td>
                </tr>
            `;
        })
        .join('');

    summary.textContent = singleCompanyMode === 'all'
        ? 'Đang hiển thị toàn bộ danh mục doanh nghiệp trong CSDL.'
        : 'Đang hiển thị những doanh nghiệp đã có kết quả chấm điểm AI.';

    tbody.querySelectorAll('.single-company-analyze-btn').forEach((button) => {
        if (button.getAttribute('data-bound') === 'true') return;
        button.setAttribute('data-bound', 'true');
        button.addEventListener('click', () => {
            analyzeSingleCompanyFromDirectory(button.getAttribute('data-tax-code') || '');
        });
    });

    tbody.querySelectorAll('.single-company-intervention-btn').forEach((button) => {
        if (button.getAttribute('data-bound') === 'true') return;
        button.setAttribute('data-bound', 'true');
        button.addEventListener('click', () => {
            const searchKeyword = document.getElementById('single-company-search')?.value || '';
            openDelinquencyWithInterventionFilter(button.getAttribute('data-intervention-action') || '', searchKeyword);
        });
    });
}


function renderSingleCompanyPagination() {
    const info = document.getElementById('single-company-pagination-info');
    const controls = document.getElementById('single-company-pagination-controls');
    if (!info || !controls) return;

    const totalPages = Math.max(1, Math.ceil(singleCompanyTotal / SINGLE_COMPANY_PAGE_SIZE));
    const startIdx = singleCompanyTotal === 0 ? 0 : (singleCompanyPage - 1) * SINGLE_COMPANY_PAGE_SIZE + 1;
    const endIdx = Math.min(singleCompanyPage * SINGLE_COMPANY_PAGE_SIZE, singleCompanyTotal);

    info.textContent = `Hiển thị ${startIdx} - ${endIdx} / ${singleCompanyTotal.toLocaleString()} doanh nghiệp`;

    const btnBase = 'px-3 py-1.5 text-[10px] font-bold rounded-lg transition-all';
    const btnActive = `${btnBase} bg-primary-container text-white shadow-sm`;
    const btnInactive = `${btnBase} bg-surface-container text-on-surface-variant hover:bg-surface-container-high`;
    const btnDisabled = `${btnBase} bg-surface-container-low text-slate-300 cursor-not-allowed`;

    controls.innerHTML = '';
    controls.innerHTML += `<button type="button" class="${singleCompanyPage === 1 ? btnDisabled : btnInactive}" ${singleCompanyPage === 1 ? 'disabled' : ''} data-single-page="${singleCompanyPage - 1}" aria-label="Trang trước">&lt;</button>`;

    let startPage = Math.max(1, singleCompanyPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);
    if (endPage - startPage < 4) startPage = Math.max(1, endPage - 4);

    for (let p = startPage; p <= endPage; p += 1) {
        controls.innerHTML += `<button type="button" class="${p === singleCompanyPage ? btnActive : btnInactive}" data-single-page="${p}">${p}</button>`;
    }

    controls.innerHTML += `<button type="button" class="${singleCompanyPage >= totalPages ? btnDisabled : btnInactive}" ${singleCompanyPage >= totalPages ? 'disabled' : ''} data-single-page="${singleCompanyPage + 1}" aria-label="Trang sau">&gt;</button>`;

    controls.querySelectorAll('button[data-single-page]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const page = Number(btn.getAttribute('data-single-page'));
            if (Number.isFinite(page)) goToSingleCompanyPage(page);
        });
    });
}


function goToSingleCompanyPage(page) {
    const totalPages = Math.max(1, Math.ceil(singleCompanyTotal / SINGLE_COMPANY_PAGE_SIZE));
    if (page < 1 || page > totalPages) return;
    singleCompanyPage = page;
    loadSingleCompanyDirectory();
}


function analyzeSingleCompanyFromDirectory(taxCode) {
    const input = document.getElementById('fraud-mst');
    if (!input || !taxCode) return;

    switchTab('single');
    input.value = taxCode;
    input.focus();
    checkFraudRisk();

    const singleTab = document.getElementById('tab-single');
    if (singleTab) singleTab.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


async function loadSingleCompanyDirectory() {
    const tbody = document.getElementById('single-company-table-body');
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-8 text-center text-slate-400 text-xs italic">Đang tải dữ liệu doanh nghiệp...</td></tr>';

    const q = (document.getElementById('single-company-search')?.value || '').trim();
    const params = new URLSearchParams({
        mode: singleCompanyMode,
        page: String(singleCompanyPage),
        page_size: String(SINGLE_COMPANY_PAGE_SIZE),
        sort_by: 'risk_score',
        sort_order: 'desc',
    });
    if (q) params.set('q', q);

    try {
        const response = await secureFetch(`${API_BASE}/ai/companies?${params.toString()}`);
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Không thể tải danh sách doanh nghiệp.');
        }

        const payload = await response.json();
        singleCompanyRows = Array.isArray(payload.results) ? payload.results : [];
        singleCompanyTotal = Number(payload.total) || 0;
        singleCompanyPage = Number(payload.page) || 1;

        renderSingleCompanyDirectoryTable();
        renderSingleCompanyPagination();
    } catch (error) {
        console.error(error);
        tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-8 text-center text-error text-xs font-bold">Lỗi tải dữ liệu doanh nghiệp.</td></tr>';
        const summary = document.getElementById('single-company-summary');
        if (summary) summary.textContent = 'Không tải được danh sách doanh nghiệp ở thời điểm hiện tại.';
        showToast('Lỗi danh sách doanh nghiệp', error.message || 'Không thể tải dữ liệu CSDL.', 'warning');
    }
}


function initSingleCompanyDirectory() {
    const search = document.getElementById('single-company-search');
    const allBtn = document.getElementById('single-company-mode-all');
    const assessedBtn = document.getElementById('single-company-mode-assessed');

    if (!search || !allBtn || !assessedBtn) return;
    if (search.getAttribute('data-initialized') === 'true') return;

    search.setAttribute('data-initialized', 'true');

    allBtn.addEventListener('click', () => setSingleCompanyMode('all'));
    assessedBtn.addEventListener('click', () => setSingleCompanyMode('assessed'));
    search.addEventListener('input', () => {
        if (singleCompanySearchDebounce) clearTimeout(singleCompanySearchDebounce);
        singleCompanySearchDebounce = setTimeout(() => {
            singleCompanyPage = 1;
            loadSingleCompanyDirectory();
        }, 280);
    });

    updateSingleCompanyModeButtons();
    loadSingleCompanyDirectory();
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

function normalizeYearlyHistory(yearlyHistory) {
    if (!Array.isArray(yearlyHistory)) return [];

    return yearlyHistory
        .map((row) => {
            if (!row || typeof row !== 'object') return null;
            const year = Number(row.year);
            const revenue = Number(row.revenue);
            const totalExpenses = Number(row.total_expenses);
            if (!Number.isFinite(year) || year <= 0) return null;
            return {
                year: Math.trunc(year),
                revenue: Number.isFinite(revenue) ? revenue : 0,
                total_expenses: Number.isFinite(totalExpenses) ? totalExpenses : 0,
            };
        })
        .filter(Boolean)
        .sort((a, b) => a.year - b.year);
}


function renderSingleTrendChartMessage(container, message) {
    if (!container) return;
    const existing = echarts.getInstanceByDom(container);
    if (existing) existing.dispose();

    container.innerHTML = `
        <div class="h-full flex items-center justify-center text-center px-4">
            <p class="text-xs text-slate-400 italic leading-relaxed">${escapeHtml(message)}</p>
        </div>
    `;
}


function mapHistorySourceLabel(historySource) {
    const normalized = String(historySource || '').trim();
    const sourceMap = {
        cache: 'Bộ nhớ đệm',
        tax_returns: 'Tờ khai thuế',
        tax_returns_aggregation: 'Tờ khai thuế',
        assessment_history: 'Lịch sử đánh giá',
        unavailable: 'Chưa có',
    };
    return sourceMap[normalized] || 'Không rõ';
}


function updateSingleTrendSubtitle(yearCount, historySource) {
    const subtitle = document.getElementById('single-trend-subtitle');
    if (!subtitle) return;

    const normalizedYearCount = Number.isFinite(Number(yearCount))
        ? Math.max(0, Math.trunc(Number(yearCount)))
        : 0;
    const sourceLabel = mapHistorySourceLabel(historySource);
    subtitle.textContent = `(${normalizedYearCount} năm • ${sourceLabel})`;
}


function setSingleTrendWarning(message) {
    const warningEl = document.getElementById('single-trend-warning');
    if (!warningEl) return;

    if (!message) {
        warningEl.classList.add('hidden');
        warningEl.textContent = '';
        return;
    }

    warningEl.textContent = message;
    warningEl.classList.remove('hidden');
}


function renderSingleTrendChart(yearlyHistory, options = {}) {
    const container = document.getElementById('chart-single-trend');
    if (!container) return;

    const normalized = normalizeYearlyHistory(yearlyHistory);
    const historyYearCount = Number(options.historyYearCount || 0);
    const historySource = String(options.historySource || 'unavailable');
    const effectiveYearCount = normalized.length || (Number.isFinite(historyYearCount) ? historyYearCount : 0);
    updateSingleTrendSubtitle(effectiveYearCount, historySource);

    if (!normalized.length) {
        const hint = historySource === 'unavailable'
            ? 'Không tìm thấy chuỗi doanh thu/chi phí theo năm trong dữ liệu hiện tại.'
            : `Không đủ điểm dữ liệu trend hợp lệ (nguồn: ${historySource}, số năm: ${historyYearCount}).`;
        setSingleTrendWarning('Chưa đủ dữ liệu lịch sử để tạo dự báo 1 năm tiếp theo.');
        renderSingleTrendChartMessage(container, hint);
        return;
    }

    setSingleTrendWarning('');
    container.innerHTML = '';

    const chart = safeInitChart(container);
    if (!chart) return;

    const years = normalized.map(d => String(d.year));
    const revenues = normalized.map(d => d.revenue);
    const expenses = normalized.map(d => d.total_expenses);

    // Dynamic forecast: next year = max(years) + 1
    const latestYear = Math.max(...normalized.map((d) => d.year));
    const forecastYear = latestYear + 1;
    let forecastRevenue = null, forecastExpense = null;
    if (revenues.length >= 2) {
        const rGrowth = revenues[revenues.length - 1] / Math.max(revenues[revenues.length - 2], 1);
        const eGrowth = expenses[expenses.length - 1] / Math.max(expenses[expenses.length - 2], 1);
        forecastRevenue = Math.round(revenues[revenues.length - 1] * rGrowth);
        forecastExpense = Math.round(expenses[expenses.length - 1] * eGrowth);
    } else {
        setSingleTrendWarning(`Dữ liệu hiện tại chỉ có ${revenues.length} năm (nguồn: ${mapHistorySourceLabel(historySource)}), hệ thống chỉ hiển thị trend thực tế và không tạo dự báo.`);
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

    const seriesData = [
        {
            name: 'Doanh thu', type: 'line', data: revActual,
            smooth: true, symbol: 'circle', symbolSize: 8,
            lineStyle: { width: 3, color: '#002147' },
            itemStyle: { color: '#002147' },
            areaStyle: {
                color: {
                    type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [
                        { offset: 0, color: 'rgba(0,33,71,0.15)' },
                        { offset: 1, color: 'rgba(0,33,71,0.01)' },
                    ],
                },
            },
            connectNulls: false,
        },
        {
            name: 'Tổng Chi phí', type: 'line', data: expActual,
            smooth: true, symbol: 'diamond', symbolSize: 8,
            lineStyle: { width: 3, color: '#dc2626', type: 'dashed' },
            itemStyle: { color: '#dc2626' },
            areaStyle: {
                color: {
                    type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [
                        { offset: 0, color: 'rgba(220,38,38,0.1)' },
                        { offset: 1, color: 'rgba(220,38,38,0.01)' },
                    ],
                },
            },
            connectNulls: false,
        },
    ];

    const legendData = ['Doanh thu', 'Tổng Chi phí'];
    if (forecastRevenue !== null) {
        legendData.push(`DT Dự báo ${forecastYear}`, `CP Dự báo ${forecastYear}`);
        seriesData.push(
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
            }
        );
    }

    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
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
            data: legendData,
            bottom: 0, textStyle: { fontSize: 9 }
        },
        grid: { left: '15%', right: '5%', top: '10%', bottom: '22%' },
        xAxis: { type: 'category', data: allYears, axisLabel: { fontSize: 10 } },
        yAxis: { type: 'value', axisLabel: { fontSize: 9, formatter: v => formatVND(v) } },
        series: seriesData,
    });

}


function normalizeYearlyFeatureScores(yearlyFeatureScores) {
    if (!Array.isArray(yearlyFeatureScores)) return [];

    return yearlyFeatureScores
        .map((row) => {
            if (!row || typeof row !== 'object') return null;
            const year = Number(row.year);
            if (!Number.isFinite(year) || year <= 0) return null;

            const f1 = Number(row.f1_divergence);
            const f2 = Number(row.f2_ratio_limit);
            const f3 = Number(row.f3_vat_structure);
            const f4 = Number(row.f4_peer_comparison);

            return {
                year: Math.trunc(year),
                f1_divergence: Number.isFinite(f1) ? f1 : 0,
                f2_ratio_limit: Number.isFinite(f2) ? f2 : 0,
                f3_vat_structure: Number.isFinite(f3) ? f3 : 0,
                f4_peer_comparison: Number.isFinite(f4) ? f4 : 0,
            };
        })
        .filter(Boolean)
        .sort((a, b) => a.year - b.year);
}


function renderSingleFeatureTrendChart(yearlyFeatureScores) {
    const container = document.getElementById('chart-single-feature-trend');
    if (!container) return false;

    const normalized = normalizeYearlyFeatureScores(yearlyFeatureScores);
    if (!normalized.length) {
        renderSingleTrendChartMessage(container, 'Chưa có dữ liệu F1-F4 theo năm để vẽ biểu đồ tiến triển đặc trưng.');
        return false;
    }

    const years = normalized.map((d) => String(d.year));
    const f1Series = normalized.map((d) => d.f1_divergence);
    const f2Series = normalized.map((d) => d.f2_ratio_limit);
    const f3Series = normalized.map((d) => d.f3_vat_structure);
    const f4Series = normalized.map((d) => d.f4_peer_comparison);

    const maxAbs = Math.max(
        0.2,
        ...f1Series.map((v) => Math.abs(v)),
        ...f2Series.map((v) => Math.abs(v)),
        ...f3Series.map((v) => Math.abs(v)),
        ...f4Series.map((v) => Math.abs(v)),
    );
    const yBound = Number((Math.ceil(maxAbs * 5) / 5).toFixed(2));

    const chart = safeInitChart(container);
    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'axis',
            formatter: (params) => {
                let html = `<b>${params[0] ? params[0].axisValue : ''}</b><br>`;
                params.forEach((p) => {
                    const value = Number(p.value || 0);
                    html += `${p.marker} ${p.seriesName}: <b>${value.toFixed(4)}</b><br>`;
                });
                return html;
            },
        },
        legend: {
            data: ['F1: Lệch pha', 'F2: Tỷ lệ CP/DT', 'F3: VAT', 'F4: So sánh ngành'],
            bottom: 0,
            textStyle: { fontSize: 9 },
        },
        grid: { left: '12%', right: '5%', top: '8%', bottom: '22%' },
        xAxis: {
            type: 'category',
            data: years,
            axisLabel: { fontSize: 10 },
        },
        yAxis: {
            type: 'value',
            min: -yBound,
            max: yBound,
            axisLabel: { fontSize: 9 },
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.35 } },
        },
        series: [
            {
                name: 'F1: Lệch pha',
                type: 'line',
                smooth: true,
                data: f1Series,
                symbol: 'circle',
                lineStyle: { width: 2, color: '#0f766e' },
                itemStyle: { color: '#0f766e' },
            },
            {
                name: 'F2: Tỷ lệ CP/DT',
                type: 'line',
                smooth: true,
                data: f2Series,
                symbol: 'diamond',
                lineStyle: { width: 2, color: '#0369a1' },
                itemStyle: { color: '#0369a1' },
            },
            {
                name: 'F3: VAT',
                type: 'line',
                smooth: true,
                data: f3Series,
                symbol: 'triangle',
                lineStyle: { width: 2, color: '#b45309' },
                itemStyle: { color: '#b45309' },
            },
            {
                name: 'F4: So sánh ngành',
                type: 'line',
                smooth: true,
                data: f4Series,
                symbol: 'rect',
                lineStyle: { width: 2, color: '#7c3aed' },
                itemStyle: { color: '#7c3aed' },
            },
        ],
    });

    return true;
}


function deriveFeatureDeltas(singleData) {
    const featureMeta = [
        { key: 'f1_divergence', label: 'F1' },
        { key: 'f2_ratio_limit', label: 'F2' },
        { key: 'f3_vat_structure', label: 'F3' },
        { key: 'f4_peer_comparison', label: 'F4' },
    ];

    const fromPayload = singleData && typeof singleData.feature_deltas === 'object'
        ? singleData.feature_deltas
        : {};
    const previous = singleData && typeof singleData.previous_year_features === 'object'
        ? singleData.previous_year_features
        : null;

    return featureMeta
        .map((meta) => {
            let delta = Number(fromPayload[meta.key]);

            if (!Number.isFinite(delta) && previous) {
                const currentVal = Number(singleData[meta.key]);
                const previousVal = Number(previous[meta.key]);
                if (Number.isFinite(currentVal) && Number.isFinite(previousVal)) {
                    delta = currentVal - previousVal;
                }
            }

            if (!Number.isFinite(delta)) return null;
            return {
                key: meta.key,
                label: meta.label,
                value: Number(delta.toFixed(4)),
            };
        })
        .filter(Boolean);
}


function renderSingleFeatureDeltaWaterfall(singleData) {
    const container = document.getElementById('chart-single-feature-waterfall');
    if (!container) return false;

    const deltas = deriveFeatureDeltas(singleData || {});
    if (!deltas.length) {
        renderSingleTrendChartMessage(container, 'Chưa đủ dữ liệu năm trước để tính waterfall biến động đặc trưng.');
        return false;
    }

    const categories = deltas.map((d) => d.label).concat('Tổng');
    const assist = [];
    const increase = [];
    const decrease = [];
    const rawValues = [];

    let cumulative = 0;
    deltas.forEach((step) => {
        const value = Number(step.value || 0);
        if (value >= 0) {
            assist.push(Number(cumulative.toFixed(4)));
            increase.push(Number(value.toFixed(4)));
            decrease.push('-');
        } else {
            assist.push(Number((cumulative + value).toFixed(4)));
            increase.push('-');
            decrease.push(Number(Math.abs(value).toFixed(4)));
        }
        cumulative += value;
        rawValues.push(Number(value.toFixed(4)));
    });

    assist.push(0);
    rawValues.push(Number(cumulative.toFixed(4)));
    if (cumulative >= 0) {
        increase.push(Number(cumulative.toFixed(4)));
        decrease.push('-');
    } else {
        increase.push('-');
        decrease.push(Number(Math.abs(cumulative).toFixed(4)));
    }

    const chart = safeInitChart(container);
    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: (params) => {
                const active = params.find((p) => p.seriesName !== 'Nền') || params[0];
                const idx = active ? active.dataIndex : 0;
                const value = Number(rawValues[idx] || 0);
                const sign = value > 0 ? '+' : '';
                if (idx === categories.length - 1) {
                    return `<b>${categories[idx]}</b><br>Tổng biến động: <b>${sign}${value.toFixed(4)}</b>`;
                }
                return `<b>${categories[idx]}</b><br>Độ lệch: <b>${sign}${value.toFixed(4)}</b>`;
            },
        },
        grid: { left: '12%', right: '6%', top: '8%', bottom: '18%' },
        xAxis: {
            type: 'category',
            data: categories,
            axisLabel: { fontSize: 10 },
        },
        yAxis: {
            type: 'value',
            axisLabel: { fontSize: 9 },
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.35 } },
        },
        series: [
            {
                name: 'Nền',
                type: 'bar',
                stack: 'total',
                data: assist,
                itemStyle: { color: 'rgba(0,0,0,0)' },
                emphasis: { disabled: true },
                tooltip: { show: false },
            },
            {
                name: 'Tăng',
                type: 'bar',
                stack: 'total',
                data: increase,
                itemStyle: {
                    color: (params) => params.dataIndex === categories.length - 1 ? '#002147' : '#16a34a',
                },
                label: {
                    show: true,
                    position: 'top',
                    fontSize: 9,
                    formatter: (params) => (params.value === '-' ? '' : `+${Number(params.value).toFixed(3)}`),
                },
            },
            {
                name: 'Giảm',
                type: 'bar',
                stack: 'total',
                data: decrease,
                itemStyle: {
                    color: (params) => params.dataIndex === categories.length - 1 ? '#002147' : '#dc2626',
                },
                label: {
                    show: true,
                    position: 'bottom',
                    fontSize: 9,
                    formatter: (params) => (params.value === '-' ? '' : `-${Number(params.value).toFixed(3)}`),
                },
            },
        ],
    });

    return true;
}


function computePercentileRank(values, targetValue) {
    const numericValues = Array.isArray(values)
        ? values.map((v) => Number(v)).filter((v) => Number.isFinite(v))
        : [];
    const numericTarget = Number(targetValue);

    if (!numericValues.length || !Number.isFinite(numericTarget)) return null;
    if (numericValues.length < 2) return 50;

    const lessOrEqual = numericValues.filter((v) => v <= numericTarget).length;
    return Number(((lessOrEqual / numericValues.length) * 100).toFixed(1));
}


function toFeatureRiskMagnitude(featureKey, rawValue) {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) return null;

    if (featureKey === 'f1_divergence' || featureKey === 'f4_peer_comparison') {
        return Math.abs(value);
    }
    if (featureKey === 'f2_ratio_limit' || featureKey === 'f3_vat_structure') {
        return Math.max(0, value);
    }
    if (featureKey === 'risk_score') {
        return Math.max(0, value);
    }
    return value;
}


function buildSingleFeaturePercentileMetrics(singleData) {
    const yearlyPoints = normalizeYearlyFeatureScores(singleData && singleData.yearly_feature_scores);
    const metrics = [];

    const riskHistory = (singleData && Array.isArray(singleData.yearly_feature_scores)
        ? singleData.yearly_feature_scores
        : [])
        .map((row) => Number(row && row.risk_score))
        .filter((v) => Number.isFinite(v));
    const currentRisk = toFeatureRiskMagnitude('risk_score', singleData && singleData.risk_score);
    const riskPercentile = computePercentileRank(riskHistory, currentRisk);
    if (riskPercentile !== null && Number.isFinite(currentRisk)) {
        metrics.push({
            key: 'risk_score',
            label: 'Rủi ro',
            percentile: riskPercentile,
            benchmarkPercentile: 50,
            currentDisplay: Number(currentRisk).toFixed(1),
            benchmarkDisplay: 'P50',
        });
    }

    const featureMeta = [
        { key: 'f1_divergence', label: 'F1', safeThreshold: 0.30 },
        { key: 'f2_ratio_limit', label: 'F2', safeThreshold: 0.95 },
        { key: 'f3_vat_structure', label: 'F3', safeThreshold: 0.90 },
        { key: 'f4_peer_comparison', label: 'F4', safeThreshold: 0.08 },
    ];

    featureMeta.forEach((meta) => {
        const historyMagnitude = yearlyPoints
            .map((point) => toFeatureRiskMagnitude(meta.key, point[meta.key]))
            .filter((v) => Number.isFinite(v));

        let currentRaw = Number(singleData && singleData[meta.key]);
        if (!Number.isFinite(currentRaw) && yearlyPoints.length) {
            currentRaw = Number(yearlyPoints[yearlyPoints.length - 1][meta.key]);
        }

        const currentMagnitude = toFeatureRiskMagnitude(meta.key, currentRaw);
        const percentile = computePercentileRank(historyMagnitude, currentMagnitude);
        if (percentile === null || !Number.isFinite(currentMagnitude)) return;

        const safeMagnitude = toFeatureRiskMagnitude(meta.key, meta.safeThreshold);
        let benchmarkPercentile = computePercentileRank(historyMagnitude, safeMagnitude);
        if (benchmarkPercentile === null) benchmarkPercentile = 50;

        metrics.push({
            key: meta.key,
            label: meta.label,
            percentile,
            benchmarkPercentile: Number(benchmarkPercentile.toFixed(1)),
            currentDisplay: Number(currentRaw).toFixed(4),
            benchmarkDisplay: Number(meta.safeThreshold).toFixed(2),
        });
    });

    return metrics;
}


function classifyPercentileLevel(percentile) {
    const value = Number(percentile);
    if (!Number.isFinite(value)) return { code: 'unknown', label: 'Không rõ', color: '#64748b' };
    if (value >= 75) return { code: 'high', label: 'Cao', color: '#dc2626' };
    if (value >= 40) return { code: 'medium', label: 'Trung bình', color: '#d97706' };
    return { code: 'low', label: 'Thấp', color: '#16a34a' };
}


function getFeaturePercentileNarrative(featureKey, levelCode) {
    const narratives = {
        risk_score: {
            high: 'Tổng điểm rủi ro đang nằm ở nhóm cao so với lịch sử doanh nghiệp, cần ưu tiên kiểm tra kỳ gần nhất.',
            medium: 'Tổng điểm đang tăng so với nền an toàn, nên tăng cường giám sát và đối soát chứng từ liên quan.',
            low: 'Tổng điểm đang ở vùng thấp so với lịch sử, tạm thời chưa có dấu hiệu cảnh báo mạnh.',
        },
        f1_divergence: {
            high: 'F1 lệch pha lớn, cần đối chiếu tăng trưởng doanh thu, lợi nhuận và thời điểm ghi nhận.',
            medium: 'F1 có biến động trung bình, nên theo dõi xu hướng giữa các kỳ khai báo liên tiếp.',
            low: 'F1 ổn định, chưa thấy dấu hiệu lệch pha đáng kể trong chuỗi dữ liệu.',
        },
        f2_ratio_limit: {
            high: 'F2 vượt ngưỡng cao, tỷ lệ chi phí/doanh thu bất thường và cần rà soát hóa đơn đầu vào.',
            medium: 'F2 ở mức cảnh báo trung bình, nên kiểm tra các khoản mục chi phí tăng đột biến.',
            low: 'F2 nằm trong vùng an toàn tương đối, cấu trúc chi phí/doanh thu đang cân bằng.',
        },
        f3_vat_structure: {
            high: 'F3 cao, cấu trúc VAT có dấu hiệu bất thường; cần đối soát VAT đầu vào/đầu ra theo kỳ.',
            medium: 'F3 tăng nhẹ, cần theo dõi biến động VAT và đối chiếu với chu kỳ kinh doanh.',
            low: 'F3 ổn định, cấu trúc VAT gần với nền lịch sử thông thường.',
        },
        f4_peer_comparison: {
            high: 'F4 lệch xa nhóm đồng ngành, cần so sánh thêm biên lợi nhuận và hiệu suất hoạt động.',
            medium: 'F4 có độ lệch trung bình so với nhóm ngành, nên tiếp tục giám sát để phát hiện xu hướng.',
            low: 'F4 gần nhóm đồng ngành, không có lệch chuẩn lớn tại thời điểm hiện tại.',
        },
    };

    const featureNarrative = narratives[featureKey] || narratives.risk_score;
    return featureNarrative[levelCode] || featureNarrative.low;
}


function renderSingleFeaturePercentileBullet(singleData) {
    const container = document.getElementById('chart-single-feature-percentile');
    if (!container) return false;

    const metrics = buildSingleFeaturePercentileMetrics(singleData || {});
    if (!metrics.length) {
        renderSingleTrendChartMessage(container, 'Chưa đủ dữ liệu lịch sử để tính thước phân vị cho Rủi ro/F1-F4.');
        return false;
    }

    const categories = metrics.map((m) => m.label);
    const lowBand = categories.map(() => 40);
    const mediumBand = categories.map(() => 35);
    const highBand = categories.map(() => 25);
    const currentPercentiles = metrics.map((m) => Number(m.percentile || 0));
    const benchmarkData = metrics.map((m) => [Number(m.benchmarkPercentile || 50), m.label]);

    const chart = safeInitChart(container);
    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        animationEasing: 'cubicOut',
        tooltip: {
            trigger: 'item',
            formatter: (params) => {
                const idx = Number(params.dataIndex || 0);
                const metric = metrics[idx];
                if (!metric) return '';

                const riskTier = classifyPercentileLevel(metric.percentile);
                const narrative = getFeaturePercentileNarrative(metric.key, riskTier.code);

                return `<b>${metric.label}</b><br>`
                    + `Phân vị hiện tại: <b>${Number(metric.percentile).toFixed(1)}%</b><br>`
                    + `Mốc tham chiếu: <b>P${Number(metric.benchmarkPercentile).toFixed(1)}</b><br>`
                    + `Giá trị hiện tại: <b>${metric.currentDisplay}</b><br>`
                    + `Ngưỡng an toàn: <b>${metric.benchmarkDisplay}</b><br>`
                    + `Mức cảnh báo: <b style="color:${riskTier.color}">${riskTier.label}</b><br>`
                    + `<span style="color:#475569">${escapeHtml(narrative)}</span>`;
            },
        },
        legend: {
            data: ['Phân vị hiện tại', 'Mốc tham chiếu'],
            bottom: 0,
            textStyle: { fontSize: 9 },
        },
        grid: { left: '18%', right: '8%', top: '8%', bottom: '20%' },
        xAxis: {
            type: 'value',
            min: 0,
            max: 100,
            axisLabel: { formatter: '{value}%', fontSize: 9 },
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } },
        },
        yAxis: {
            type: 'category',
            data: categories,
            inverse: true,
            axisLabel: { fontSize: 10, fontWeight: 700 },
        },
        series: [
            {
                name: 'Vùng thấp',
                type: 'bar',
                stack: 'band',
                data: lowBand,
                barWidth: 14,
                itemStyle: { color: 'rgba(22,163,74,0.16)' },
                silent: true,
                emphasis: { disabled: true },
                tooltip: { show: false },
            },
            {
                name: 'Vùng trung bình',
                type: 'bar',
                stack: 'band',
                data: mediumBand,
                barWidth: 14,
                itemStyle: { color: 'rgba(234,179,8,0.22)' },
                silent: true,
                emphasis: { disabled: true },
                tooltip: { show: false },
            },
            {
                name: 'Vùng cao',
                type: 'bar',
                stack: 'band',
                data: highBand,
                barWidth: 14,
                itemStyle: { color: 'rgba(220,38,38,0.22)' },
                silent: true,
                emphasis: { disabled: true },
                tooltip: { show: false },
            },
            {
                name: 'Phân vị hiện tại',
                type: 'bar',
                data: currentPercentiles,
                barWidth: 8,
                barGap: '-75%',
                itemStyle: { color: '#002147', borderRadius: [0, 5, 5, 0] },
                z: 10,
            },
            {
                name: 'Mốc tham chiếu',
                type: 'scatter',
                data: benchmarkData,
                symbol: 'diamond',
                symbolSize: 12,
                itemStyle: { color: '#1d4ed8', borderColor: '#fff', borderWidth: 1.5 },
                z: 20,
            },
        ],
    });

    return true;
}


function clampNumber(value, min, max) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return min;
    return Math.max(min, Math.min(max, numeric));
}


function classifyRiskTierFromScore(score) {
    const numeric = Number(score);
    if (!Number.isFinite(numeric)) return { key: 'unknown', label: 'Không rõ' };
    if (numeric >= 80) return { key: 'critical', label: 'Rất cao' };
    if (numeric >= 60) return { key: 'high', label: 'Cao' };
    if (numeric >= 40) return { key: 'medium', label: 'Trung bình' };
    return { key: 'low', label: 'Thấp' };
}


function getSensitivityNarrative(riskTierKey, deltaRisk) {
    const delta = Number(deltaRisk || 0);
    if (riskTierKey === 'critical' || delta >= 12) {
        return 'Kịch bản stress cao: cần ưu tiên thanh tra và đối soát nhanh các khoản mục phát sinh đột biến.';
    }
    if (riskTierKey === 'high' || delta >= 5) {
        return 'Kịch bản cảnh báo cao: nên mở rộng theo dõi, kiểm tra hóa đơn đầu vào và biến động VAT.';
    }
    if (riskTierKey === 'medium' || delta >= 1) {
        return 'Kịch bản trung bình: tiếp tục giám sát kỳ tiếp theo và đối chiếu với ngưỡng ngành.';
    }
    return 'Kịch bản ổn định: biến động hiện tại chưa cho thấy áp lực rủi ro lớn trên điểm tổng hợp.';
}


function buildLocalSensitivityHeatmapData(singleData) {
    const revenueAdjustments = [...WHATIF_HEATMAP_DEFAULT_REVENUE_STEPS];
    const expenseAdjustments = [...WHATIF_HEATMAP_DEFAULT_EXPENSE_STEPS];

    const currentRisk = clampNumber(singleData && singleData.risk_score, 0, 100);
    const f1 = Math.abs(Number(singleData && singleData.f1_divergence || 0));
    const f2 = Math.max(0, Number(singleData && singleData.f2_ratio_limit || 0));
    const f3 = Math.max(0, Number(singleData && singleData.f3_vat_structure || 0));
    const f4 = Math.abs(Number(singleData && singleData.f4_peer_comparison || 0));

    const f1Norm = clampNumber(f1 / 1.0, 0, 1);
    const f2Norm = clampNumber(f2 / 1.2, 0, 1);
    const f3Norm = clampNumber(f3 / 1.1, 0, 1);
    const f4Norm = clampNumber(f4 / 0.2, 0, 1);

    const baseSensitivity = 0.55 + (f2Norm * 0.20) + (f3Norm * 0.15) + (f1Norm * 0.10) + (f4Norm * 0.08);
    const interactionGain = 0.04 + (f2Norm * 0.03) + (f3Norm * 0.03);

    const values = [];
    for (let yIdx = 0; yIdx < expenseAdjustments.length; yIdx += 1) {
        const expAdj = expenseAdjustments[yIdx];
        for (let xIdx = 0; xIdx < revenueAdjustments.length; xIdx += 1) {
            const revAdj = revenueAdjustments[xIdx];

            const revenueEffect = (-revAdj) * (0.24 + (f1Norm * 0.05));
            const expenseEffect = expAdj * (0.27 + (f2Norm * 0.06) + (f3Norm * 0.05));
            const interactionEffect = (expAdj - revAdj) * interactionGain;

            const deltaRisk = (revenueEffect + expenseEffect + interactionEffect) * baseSensitivity;
            const projectedRisk = clampNumber(currentRisk + deltaRisk, 0, 100);

            values.push([
                xIdx,
                yIdx,
                Number(projectedRisk.toFixed(2)),
                Number(deltaRisk.toFixed(2)),
            ]);
        }
    }

    return {
        source: 'local_fallback',
        revenueAdjustments,
        expenseAdjustments,
        currentRisk,
        values,
    };
}


function normalizeWhatIfHeatmapSteps(rawSteps, fallbackSteps) {
    const source = Array.isArray(rawSteps) ? rawSteps : fallbackSteps;
    return source
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v));
}


function normalizeWhatIfHeatmapValues(rawValues, revenueAdjustments, expenseAdjustments) {
    if (!Array.isArray(rawValues)) return [];

    const xMax = Math.max(0, revenueAdjustments.length - 1);
    const yMax = Math.max(0, expenseAdjustments.length - 1);

    return rawValues
        .map((row) => {
            if (!Array.isArray(row) || row.length < 4) return null;

            const xIndex = Number(row[0]);
            const yIndex = Number(row[1]);
            const simulatedRisk = Number(row[2]);
            const deltaRisk = Number(row[3]);

            if (!Number.isFinite(xIndex) || !Number.isFinite(yIndex)) return null;
            if (!Number.isFinite(simulatedRisk) || !Number.isFinite(deltaRisk)) return null;

            const x = Math.trunc(xIndex);
            const y = Math.trunc(yIndex);
            if (x < 0 || y < 0 || x > xMax || y > yMax) return null;

            return [
                x,
                y,
                clampNumber(simulatedRisk, 0, 100),
                deltaRisk,
            ];
        })
        .filter(Boolean);
}


function getSingleHeatmapCacheKey(singleData) {
    const taxCode = String(singleData && singleData.tax_code || window._whatifTaxCode || '').trim();
    const riskScore = Number(singleData && singleData.risk_score || window._whatifOriginalScore || 0);
    return `${taxCode}|${Number.isFinite(riskScore) ? riskScore.toFixed(2) : '0.00'}`;
}


async function fetchSingleSensitivityHeatmapData(singleData) {
    const taxCode = String(singleData && singleData.tax_code || window._whatifTaxCode || '').trim();
    if (!taxCode) {
        throw new Error('Không có MST để tải sensitivity heatmap.');
    }

    const cacheKey = getSingleHeatmapCacheKey(singleData || {});
    if (_singleSensitivityHeatmapCache.has(cacheKey)) {
        return _singleSensitivityHeatmapCache.get(cacheKey);
    }

    const response = await secureFetch(`${API_BASE}/ai/what-if-grid/${encodeURIComponent(taxCode)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            revenue_steps: WHATIF_HEATMAP_DEFAULT_REVENUE_STEPS,
            expense_steps: WHATIF_HEATMAP_DEFAULT_EXPENSE_STEPS,
        }),
    });

    if (!response.ok) {
        let detail = 'Không thể tải dữ liệu sensitivity từ backend.';
        try {
            const err = await response.json();
            detail = err && err.detail ? String(err.detail) : detail;
        } catch {
            // ignore JSON parse failure
        }
        throw new Error(detail);
    }

    const payload = await response.json();
    const revenueAdjustments = normalizeWhatIfHeatmapSteps(
        payload && payload.revenue_steps,
        WHATIF_HEATMAP_DEFAULT_REVENUE_STEPS,
    );
    const expenseAdjustments = normalizeWhatIfHeatmapSteps(
        payload && payload.expense_steps,
        WHATIF_HEATMAP_DEFAULT_EXPENSE_STEPS,
    );
    const values = normalizeWhatIfHeatmapValues(
        payload && payload.values,
        revenueAdjustments,
        expenseAdjustments,
    );

    const originalRiskFromApi = Number(payload && payload.original_risk_score);
    const fallbackRisk = Number(singleData && singleData.risk_score);
    const currentRisk = Number.isFinite(originalRiskFromApi)
        ? clampNumber(originalRiskFromApi, 0, 100)
        : clampNumber(fallbackRisk, 0, 100);

    if (!revenueAdjustments.length || !expenseAdjustments.length || !values.length) {
        throw new Error('Payload sensitivity từ backend không đầy đủ.');
    }

    const normalized = {
        source: 'what_if_backend',
        revenueAdjustments,
        expenseAdjustments,
        currentRisk,
        values,
    };
    _singleSensitivityHeatmapCache.set(cacheKey, normalized);
    return normalized;
}


function snapToSliderValue(rawValue, sliderEl) {
    const value = Number(rawValue);
    const minValue = Number(sliderEl && sliderEl.min);
    const maxValue = Number(sliderEl && sliderEl.max);
    const stepValue = Number(sliderEl && sliderEl.step);

    const safeMin = Number.isFinite(minValue) ? minValue : -80;
    const safeMax = Number.isFinite(maxValue) ? maxValue : 250;
    const safeStep = Number.isFinite(stepValue) && stepValue > 0 ? stepValue : 1;

    const clamped = clampNumber(value, safeMin, safeMax);
    return Math.round(clamped / safeStep) * safeStep;
}


function applyHeatmapScenarioToWhatIfSliders(revenuePct, expensePct) {
    const revenueSlider = document.getElementById('whatif-revenue');
    const expenseSlider = document.getElementById('whatif-expenses');
    if (!revenueSlider || !expenseSlider) return;

    const revenueValue = snapToSliderValue(revenuePct, revenueSlider);
    const expenseValue = snapToSliderValue(expensePct, expenseSlider);

    revenueSlider.value = String(revenueValue);
    expenseSlider.value = String(expenseValue);

    const whatifBox = document.getElementById('whatif-sandbox');
    if (whatifBox && whatifBox.style.display === 'none') {
        whatifBox.style.display = 'block';
    }

    onWhatIfChange();
    showToast(
        'Đã áp dụng kịch bản heatmap',
        `Cập nhật slider: Doanh thu ${revenueValue > 0 ? '+' : ''}${revenueValue}% | Chi phí ${expenseValue > 0 ? '+' : ''}${expenseValue}%`,
        'info',
        2200,
    );
}


function renderSingleSensitivityHeatmapChart(container, heatmapData) {
    if (!container) return false;

    const xLabels = heatmapData.revenueAdjustments.map((v) => `DT ${v > 0 ? '+' : ''}${v}%`);
    const yLabels = heatmapData.expenseAdjustments.map((v) => `CP ${v > 0 ? '+' : ''}${v}%`);
    const baseX = heatmapData.revenueAdjustments.indexOf(0);
    const baseY = heatmapData.expenseAdjustments.indexOf(0);
    const sourceLabel = heatmapData.source === 'what_if_backend' ? 'Mô phỏng từ backend' : 'Mô phỏng cục bộ';

    const chart = safeInitChart(container);
    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        animationEasing: 'cubicOut',
        tooltip: {
            formatter: (params) => {
                const payload = params.data || [];
                const xIndex = Number(payload[0] || 0);
                const yIndex = Number(payload[1] || 0);
                const projectedRisk = Number(payload[2] || 0);
                const deltaRisk = Number(payload[3] || 0);
                const tier = classifyRiskTierFromScore(projectedRisk);
                const narrative = getSensitivityNarrative(tier.key, deltaRisk);

                const revAdj = heatmapData.revenueAdjustments[xIndex] || 0;
                const expAdj = heatmapData.expenseAdjustments[yIndex] || 0;
                const deltaSign = deltaRisk > 0 ? '+' : '';

                return `<b>Kịch bản mô phỏng</b><br>`
                    + `Doanh thu: <b>${revAdj > 0 ? '+' : ''}${revAdj}%</b><br>`
                    + `Chi phí: <b>${expAdj > 0 ? '+' : ''}${expAdj}%</b><br>`
                    + `Rủi ro ước tính: <b>${projectedRisk.toFixed(1)}</b><br>`
                    + `Độ lệch rủi ro: <b>${deltaSign}${deltaRisk.toFixed(1)}</b><br>`
                    + `Mức cảnh báo: <b>${tier.label}</b><br>`
                    + `Nguồn: <b>${sourceLabel}</b><br>`
                    + `<span style="color:#475569">${escapeHtml(narrative)}</span>`;
            },
        },
        grid: { left: '13%', right: '16%', top: '8%', bottom: '12%' },
        xAxis: {
            type: 'category',
            data: xLabels,
            splitArea: { show: true },
            axisLabel: { fontSize: 9, interval: 0 },
        },
        yAxis: {
            type: 'category',
            data: yLabels,
            splitArea: { show: true },
            axisLabel: { fontSize: 9, interval: 0 },
        },
        visualMap: {
            min: 0,
            max: 100,
            calculable: true,
            orient: 'vertical',
            right: 0,
            top: 'middle',
            text: ['Rủi ro cao', 'Rủi ro thấp'],
            inRange: {
                color: ['#dcfce7', '#bbf7d0', '#fde68a', '#fdba74', '#f87171', '#b91c1c'],
            },
            textStyle: { fontSize: 9 },
        },
        series: [
            {
                name: 'Độ nhạy',
                type: 'heatmap',
                data: heatmapData.values,
                label: {
                    show: true,
                    fontSize: 8,
                    formatter: (params) => Number(params.data[2] || 0).toFixed(0),
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.35)',
                    },
                },
            },
            {
                name: 'Hiện tại',
                type: 'scatter',
                data: [[baseX, baseY, heatmapData.currentRisk]],
                symbol: 'pin',
                symbolSize: 22,
                itemStyle: {
                    color: '#002147',
                    borderColor: '#ffffff',
                    borderWidth: 1.5,
                },
                label: {
                    show: true,
                    formatter: 'Hiện tại',
                    color: '#ffffff',
                    fontSize: 8,
                    position: 'inside',
                },
                z: 30,
                tooltip: {
                    formatter: () => `Trạng thái hiện tại\nRủi ro: ${heatmapData.currentRisk.toFixed(1)}`,
                },
            },
        ],
    });

    chart.off('click');
    chart.on('click', (params) => {
        if (!params || params.seriesName !== 'Sensitivity' || !Array.isArray(params.data)) return;

        const xIndex = Math.trunc(Number(params.data[0]));
        const yIndex = Math.trunc(Number(params.data[1]));
        const revenuePct = Number(heatmapData.revenueAdjustments[xIndex]);
        const expensePct = Number(heatmapData.expenseAdjustments[yIndex]);

        if (!Number.isFinite(revenuePct) || !Number.isFinite(expensePct)) return;
        applyHeatmapScenarioToWhatIfSliders(revenuePct, expensePct);
    });

    return true;
}


function renderSingleSensitivityHeatmap(singleData) {
    const container = document.getElementById('chart-single-sensitivity-heatmap');
    if (!container) return false;

    const currentRisk = Number(singleData && singleData.risk_score);
    if (!Number.isFinite(currentRisk)) {
        renderSingleTrendChartMessage(container, 'Không có điểm rủi ro hiện tại để tạo sensitivity heatmap.');
        return false;
    }

    const requestToken = ++_singleSensitivityHeatmapRequestToken;
    renderSingleTrendChartMessage(container, 'Đang tải bản đồ nhiệt độ nhạy từ backend mô phỏng giả định...');

    fetchSingleSensitivityHeatmapData(singleData || {})
        .then((heatmapData) => {
            if (requestToken !== _singleSensitivityHeatmapRequestToken) return;
            renderSingleSensitivityHeatmapChart(container, heatmapData);
        })
        .catch((error) => {
            if (requestToken !== _singleSensitivityHeatmapRequestToken) return;

            console.warn('[SensitivityHeatmap] Backend grid failed, fallback local simulation:', error);
            const fallbackData = buildLocalSensitivityHeatmapData(singleData || {});
            renderSingleSensitivityHeatmapChart(container, fallbackData);
        });

    return true;
}


// ---- COHORT RISK PROGRESSION (Sankey) ----
function renderCohortRiskFunnel(sankeyData) {
    const container = document.getElementById('chart-cohort-funnel');
    if (!container) return;

    const nodes = Array.isArray(sankeyData.nodes) ? sankeyData.nodes : [];
    const links = Array.isArray(sankeyData.links) ? sankeyData.links : [];

    if (!nodes.length || !links.length) {
        renderSingleTrendChartMessage(container, 'Chưa có dữ liệu chuyển dịch nhóm rủi ro giữa các năm.');
        return;
    }

    const chart = safeInitChart(container);
    const tierLabel = {
        low: 'An toàn',
        medium: 'Trung bình',
        high: 'Rủi ro cao',
        critical: 'Rất cao',
    };
    const tierColor = {
        low: '#16a34a',
        medium: '#eab308',
        high: '#ea580c',
        critical: '#dc2626',
    };

    const formattedNodes = nodes.map((node) => {
        const name = String(node.name || 'unknown');
        const parts = name.split(':');
        const year = parts[0] || '-';
        const tier = parts[1] || 'low';
        return {
            name,
            itemStyle: { color: tierColor[tier] || '#64748b' },
            label: {
                formatter: `${year}\n${tierLabel[tier] || tier}`,
                fontSize: 9,
                color: '#334155',
            },
        };
    });

    const formattedLinks = links
        .filter((link) => Number(link.value || 0) > 0)
        .map((link) => ({
            source: String(link.source || ''),
            target: String(link.target || ''),
            value: Number(link.value || 0),
        }));

    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        tooltip: {
            trigger: 'item',
            formatter: (params) => {
                if (params.dataType === 'edge') {
                    return `${params.data.source} -> ${params.data.target}<br><b>${params.data.value.toLocaleString()} DN</b>`;
                }
                return `${params.name}`;
            },
        },
        series: [
            {
                type: 'sankey',
                data: formattedNodes,
                links: formattedLinks,
                nodeWidth: 18,
                nodeGap: 14,
                orient: 'horizontal',
                lineStyle: {
                    color: 'source',
                    curveness: 0.45,
                    opacity: 0.45,
                },
                emphasis: {
                    focus: 'adjacency',
                },
            },
        ],
    });
}


// ---- VAT ANOMALY HEATMAP (Industry x Year) ----
function renderVatAnomalyHeatmap(heatmapData) {
    const container = document.getElementById('chart-vat-heatmap');
    if (!container) return;

    const years = Array.isArray(heatmapData.years) ? heatmapData.years : [];
    const industries = Array.isArray(heatmapData.industries) ? heatmapData.industries : [];
    const values = Array.isArray(heatmapData.values) ? heatmapData.values : [];
    const counts = Array.isArray(heatmapData.counts) ? heatmapData.counts : [];

    if (!years.length || !industries.length || !values.length) {
        renderSingleTrendChartMessage(container, 'Chưa có dữ liệu F3 VAT anomaly theo ngành-năm.');
        return;
    }

    const chart = safeInitChart(container);
    const countsMap = new Map();
    counts.forEach((cell) => {
        if (!Array.isArray(cell) || cell.length < 4) return;
        countsMap.set(`${cell[0]}-${cell[1]}`, { anomaly: cell[2], total: cell[3] });
    });

    const maxRate = Math.max(10, ...values.map((v) => Number(v[2] || 0)));

    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        tooltip: {
            formatter: (params) => {
                const x = params.data[0];
                const y = params.data[1];
                const rate = Number(params.data[2] || 0);
                const metric = countsMap.get(`${x}-${y}`) || { anomaly: 0, total: 0 };
                return `<b>${escapeHtml(String(industries[y] || '---'))}</b><br>`
                    + `Năm: ${escapeHtml(String(years[x] || '---'))}<br>`
                    + `Tỉ lệ bất thường F3: <b>${rate.toFixed(2)}%</b><br>`
                    + `Số bản ghi anomaly: <b>${Number(metric.anomaly || 0).toLocaleString()}</b>/<b>${Number(metric.total || 0).toLocaleString()}</b>`;
            },
        },
        grid: { left: '20%', right: '16%', top: '6%', bottom: '10%' },
        xAxis: {
            type: 'category',
            data: years,
            splitArea: { show: true },
            axisLabel: { fontSize: 10 },
        },
        yAxis: {
            type: 'category',
            data: industries,
            splitArea: { show: true },
            axisLabel: { fontSize: 9, width: 110, overflow: 'truncate' },
        },
        visualMap: {
            min: 0,
            max: maxRate,
            calculable: true,
            orient: 'vertical',
            right: 0,
            top: 'middle',
            text: ['Cao', 'Thấp'],
            inRange: {
                color: ['#ecfeff', '#bae6fd', '#fef08a', '#fdba74', '#f87171', '#b91c1c'],
            },
            textStyle: { fontSize: 10 },
        },
        series: [
            {
                name: 'VAT anomaly rate',
                type: 'heatmap',
                data: values,
                label: {
                    show: true,
                    fontSize: 8,
                    formatter: (params) => `${Number(params.data[2] || 0).toFixed(1)}%`,
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.3)',
                    },
                },
            },
        ],
    });
}


// ---- CUMULATIVE RISK CURVE (Concentration) ----
function renderCumulativeRiskCurve(curveData) {
    const container = document.getElementById('chart-cumulative-risk');
    if (!container) return;

    const points = Array.isArray(curveData.points) ? curveData.points : [];
    if (!points.length) {
        renderSingleTrendChartMessage(container, 'Chưa có dữ liệu đường cong rủi ro tích lũy.');
        return;
    }

    const chart = safeInitChart(container);
    const seriesData = points.map((p) => [Number(p.percent_companies || 0), Number(p.percent_risk || 0)]);
    const diagonal = [[0, 0], [100, 100]];

    chart.setOption({
        animationDuration: MOTION_DURATION_CHART,
        tooltip: {
            trigger: 'axis',
            formatter: (params) => {
                const point = params[0] && params[0].data ? params[0].data : [0, 0];
                return `Nhóm top <b>${Number(point[0]).toFixed(1)}%</b> DN đang chứa <b>${Number(point[1]).toFixed(1)}%</b> tổng rủi ro`;
            },
        },
        legend: {
            data: ['Rủi ro tích lũy', 'Đường cân bằng'],
            bottom: 0,
            textStyle: { fontSize: 9 },
        },
        grid: { left: '12%', right: '6%', top: '8%', bottom: '18%' },
        xAxis: {
            type: 'value',
            min: 0,
            max: 100,
            name: '% Doanh nghiệp',
            axisLabel: { formatter: '{value}%' },
        },
        yAxis: {
            type: 'value',
            min: 0,
            max: 100,
            name: '% Tổng rủi ro',
            axisLabel: { formatter: '{value}%' },
        },
        series: [
            {
                name: 'Rủi ro tích lũy',
                type: 'line',
                smooth: true,
                data: seriesData,
                lineStyle: { width: 3, color: '#dc2626' },
                itemStyle: { color: '#dc2626' },
                areaStyle: { color: 'rgba(220,38,38,0.12)' },
                markPoint: {
                    symbolSize: 44,
                    data: [
                        {
                            name: 'Top 10%',
                            coord: [10, Number(curveData.top_10pct_risk_share || 0)],
                            value: `${Number(curveData.top_10pct_risk_share || 0).toFixed(1)}%`,
                        },
                        {
                            name: 'Top 20%',
                            coord: [20, Number(curveData.top_20pct_risk_share || 0)],
                            value: `${Number(curveData.top_20pct_risk_share || 0).toFixed(1)}%`,
                        },
                    ],
                    label: { fontSize: 8, color: '#334155' },
                },
            },
            {
                name: 'Đường cân bằng',
                type: 'line',
                data: diagonal,
                lineStyle: { width: 2, color: '#64748b', type: 'dashed' },
                showSymbol: false,
                tooltip: { show: false },
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
        animationDuration: MOTION_DURATION_CHART,
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
        animationDuration: MOTION_DURATION_CHART,
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
        animationDuration: MOTION_DURATION_CHART,
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
        animationDuration: MOTION_DURATION_CHART,
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
        animationDuration: MOTION_DURATION_CHART,
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
    const gateState = window._fraudSplitGateState;
    if (!gateState || !gateState.ready) {
        showToast('Cổng đang khóa', gateState?.reason || 'Split-trigger chưa sẵn sàng cho hành động thanh tra.', 'warning');
        return;
    }

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
    const gateState = window._fraudSplitGateState;
    if (!gateState || !gateState.ready) {
        showToast('Cổng đang khóa', gateState?.reason || 'Split-trigger chưa sẵn sàng cho hành động thanh tra.', 'warning');
        return;
    }

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
        const safeMessage = escapeHtml(error?.message || 'Mô phỏng thất bại');
        resultDiv.innerHTML = `
            <div class="flex items-center gap-3">
                <span class="material-symbols-outlined text-red-400 text-xl">error</span>
                <p class="text-[10px] text-red-500">${safeMessage}</p>
            </div>`;
    }
}


function renderWhatIfResult(data) {
    const resultDiv = document.getElementById('whatif-result');
    if (!resultDiv) return;

    const origScore = Number.isFinite(Number(data.original_risk_score))
        ? Number(data.original_risk_score)
        : Number(window._whatifOriginalScore || 0);
    const simScore = Number.isFinite(Number(data.simulated_risk_score))
        ? Number(data.simulated_risk_score)
        : 0;
    const delta = Number.isFinite(Number(data.delta_risk))
        ? Number(data.delta_risk)
        : (simScore - origScore);

    const appliedAdjustmentsRaw = (data && typeof data === 'object' && data.applied_adjustments && typeof data.applied_adjustments === 'object')
        ? data.applied_adjustments
        : {};
    const adjustmentLabels = {
        revenue: 'Doanh thu',
        total_expenses: 'Tổng chi phí',
    };
    const adjustmentChips = Object.entries(appliedAdjustmentsRaw)
        .map(([field, value]) => {
            const pct = Number(value);
            if (!Number.isFinite(pct)) return '';
            const sign = pct > 0 ? '+' : '';
            const label = adjustmentLabels[field] || field;
            const toneClass = pct > 0 ? 'bg-red-50 text-red-600' : pct < 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-500';
            return `<span class="px-2 py-1 rounded ${toneClass} text-[9px] font-black uppercase tracking-wider">${escapeHtml(label)} ${sign}${pct.toFixed(0)}%</span>`;
        })
        .filter(Boolean)
        .join('');

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
        <div class="flex flex-col gap-3 w-full">
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
            </div>
            <div class="flex flex-wrap items-center gap-1.5">
                ${adjustmentChips || '<span class="px-2 py-1 rounded bg-slate-100 text-slate-500 text-[9px] font-black uppercase tracking-wider">Không có thay đổi</span>'}
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
        const safeCompanyName = escapeHtml(data.company_name || 'Không rõ');
        const safeTaxCode = escapeHtml(data.tax_code || '---');
        const safeIndustry = escapeHtml(data.industry || '---');
        const safeRiskLevel = escapeHtml(getRiskLabel(data.risk_level));
        const revenueText = Number.isFinite(Number(data.revenue)) ? `${Number(data.revenue).toLocaleString()} tỷ VNĐ` : '---';
        const expensesText = Number.isFinite(Number(data.total_expenses)) ? `${Number(data.total_expenses).toLocaleString()} tỷ VNĐ` : '---';
        const riskScoreText = Number.isFinite(Number(data.risk_score)) ? Number(data.risk_score).toFixed(1) : '0.0';
        const safeAnomalyPercent = escapeHtml(anomalyPercent);

        const interventionPayload = normalizeInterventionUplift(data.intervention_uplift || null, data);
        const decisionPayload = normalizeDecisionIntelligence(data.decision_intelligence || null, data, interventionPayload);
        const actionCatalog = {
            urgent_audit: 'Thanh tra đột xuất',
            targeted_review: 'Kiểm tra chuyên sâu',
            enhanced_monitoring: 'Giám sát tăng cường',
            periodic_monitoring: 'Theo dõi định kỳ',
        };
        const decisionActionText = actionCatalog[decisionPayload.recommended_action] || decisionPayload.action_label || 'Theo dõi định kỳ';
        const decisionPriorityScore = Number.isFinite(Number(decisionPayload.priority_score))
            ? Math.max(0, Math.min(100, Math.round(Number(decisionPayload.priority_score))))
            : 0;
        const decisionDeadline = Number.isFinite(Number(decisionPayload.action_deadline_days))
            ? Math.max(1, Math.round(Number(decisionPayload.action_deadline_days)))
            : 30;
        const safeDecisionActionText = escapeHtml(decisionActionText);
        const safeDecisionRationale = escapeHtml(decisionPayload.rationale || 'Hệ thống chưa ghi nhận căn cứ khuyến nghị bổ sung.');
        const safeDecisionEscalation = decisionPayload.should_escalate
            ? 'Có (đề nghị chuyển cấp xử lý theo thẩm quyền)'
            : 'Không';
        const interventionActionCatalog = {
            monitor: 'Giám sát',
            auto_reminder: 'Nhắc hạn tự động',
            structured_outreach: 'Can thiệp có cấu trúc',
            field_audit: 'Kiểm tra tại chỗ',
            escalated_enforcement: 'Cưỡng chế nâng cao',
        };
        const interventionActionText = interventionActionCatalog[interventionPayload.recommended_action] || 'Giám sát';
        const safeInterventionActionText = escapeHtml(interventionActionText);
        const safeInterventionRiskReduction = escapeHtml(`${Number(interventionPayload.expected_risk_reduction_pp || 0).toFixed(1)} pp`);
        const safeInterventionPenaltySaving = escapeHtml(formatVND(Number(interventionPayload.expected_penalty_saving || 0)));
        const safeInterventionCollectionUplift = escapeHtml(formatVND(Number(interventionPayload.expected_collection_uplift || 0)));
        const safeInterventionConfidence = escapeHtml(String(interventionPayload.confidence || 'medium').toUpperCase());

        const decisionSignalsHtml = (Array.isArray(decisionPayload.top_signals) ? decisionPayload.top_signals : [])
            .slice(0, 4)
            .map((signal) => {
                const label = escapeHtml(String(signal?.label || 'Tín hiệu'));
                const summary = escapeHtml(String(signal?.summary || 'Không có diễn giải bổ sung.'));
                const severityRaw = String(signal?.severity || 'medium').toLowerCase();
                const severityLabel = severityRaw === 'high' ? 'Cao' : severityRaw === 'low' ? 'Thấp' : 'Trung bình';
                const safeSeverityLabel = escapeHtml(severityLabel);
                const valueText = escapeHtml(formatDecisionSignalValue(signal));
                return `<li><b>${label}</b> - Mức độ ${safeSeverityLabel}; Giá trị: ${valueText}. Nội dung đánh giá: ${summary}</li>`;
            })
            .join('');

        const decisionStepsHtml = (Array.isArray(decisionPayload.next_steps) ? decisionPayload.next_steps : [])
            .slice(0, 5)
            .map((step) => `<li>${escapeHtml(String(step || ''))}</li>`)
            .join('');

        const vatRefundPayload = normalizeVatRefundSignals(data.vat_refund_signals || null, data);
        const vatQueueCatalog = {
            priority_refund_audit: 'Kiểm tra hoàn thuế ưu tiên',
            refund_watchlist: 'Theo dõi hoàn thuế',
            monitor: 'Giám sát thường xuyên',
        };
        const vatQueueText = vatQueueCatalog[vatRefundPayload.queue] || 'Giám sát thường xuyên';
        const vatRatioText = Number.isFinite(Number(vatRefundPayload.vat_input_output_ratio))
            ? `${Number(vatRefundPayload.vat_input_output_ratio).toFixed(2)}x`
            : 'Không có';
        const vatGapText = Number.isFinite(Number(vatRefundPayload.estimated_refund_gap))
            ? formatVND(Number(vatRefundPayload.estimated_refund_gap))
            : 'Không có';
        const safeVatQueueText = escapeHtml(vatQueueText);
        const safeVatLevelText = escapeHtml(String(vatRefundPayload.level || 'low').toUpperCase());
        const safeVatRationale = escapeHtml(vatRefundPayload.rationale || 'Chưa ghi nhận tín hiệu hoàn thuế VAT bất thường tại thời điểm đánh giá.');
        const safeVatRatioText = escapeHtml(vatRatioText);
        const safeVatGapText = escapeHtml(vatGapText);

        const vatChecksHtml = (Array.isArray(vatRefundPayload.recommended_checks) ? vatRefundPayload.recommended_checks : [])
            .slice(0, 5)
            .map((step) => `<li>${escapeHtml(String(step || ''))}</li>`)
            .join('');

        const vatIndicatorsHtml = (Array.isArray(vatRefundPayload.indicators) ? vatRefundPayload.indicators : [])
            .slice(0, 5)
            .map((indicator) => {
                const label = escapeHtml(String(indicator?.label || 'Tín hiệu VAT'));
                const summary = escapeHtml(String(indicator?.summary || 'Không có diễn giải bổ sung.'));
                const valueText = escapeHtml(formatVatRefundIndicatorValue(indicator));
                const severityRaw = String(indicator?.severity || 'medium').toLowerCase();
                const severityLabel = severityRaw === 'high' ? 'Cao' : severityRaw === 'low' ? 'Thấp' : 'Trung bình';
                return `<li><b>${label}</b> - Mức độ ${escapeHtml(severityLabel)}; Giá trị: ${valueText}. Nội dung đánh giá: ${summary}</li>`;
            })
            .join('');
        
        let riskConclusion = 'Doanh nghiệp <b>chưa ghi nhận dấu hiệu rủi ro ở mức phải kiểm tra đột xuất</b>; kiến nghị đưa vào diện <b>theo dõi định kỳ</b> theo quy trình quản lý rủi ro.';
        if (data.risk_score >= 80) {
            riskConclusion = 'Doanh nghiệp thuộc nhóm <b>RỦI RO RẤT CAO</b>. Kiến nghị áp dụng biện pháp <b>thanh tra đột xuất</b>, yêu cầu giải trình toàn diện và thực hiện đối soát hồ sơ thuế theo quy định hiện hành.';
        } else if (data.risk_score >= 60) {
            riskConclusion = 'Doanh nghiệp thuộc nhóm <b>RỦI RO CAO</b>. Kiến nghị triển khai <b>kiểm tra chuyên sâu hồ sơ</b>, ưu tiên các tờ khai VAT và báo cáo tài chính trong 03 năm gần nhất.';
        } else if (data.risk_score >= 40) {
            riskConclusion = 'Doanh nghiệp có dấu hiệu <b>RỦI RO MỨC TRUNG BÌNH</b>. Kiến nghị thực hiện <b>giám sát tăng cường</b> đối với biến động dòng tiền trong kỳ đánh giá tiếp theo.';
        }

        if (decisionPayload.recommended_action === 'urgent_audit') {
            riskConclusion = 'Trên cơ sở kết quả Trí tuệ quyết định, hồ sơ được xác định thuộc diện <b>THANH TRA ĐỘT XUẤT</b>. Đề nghị trình cấp có thẩm quyền phê duyệt kế hoạch kiểm tra khẩn và ban hành yêu cầu giải trình trong thời hạn quy định.';
        } else if (decisionPayload.recommended_action === 'targeted_review') {
            riskConclusion = 'Theo kết quả Trí tuệ quyết định, hồ sơ thuộc diện <b>KIỂM TRA CHUYÊN SÂU</b> có trọng tâm. Đề nghị ưu tiên đối soát tờ khai VAT, chứng từ đầu vào/đầu ra và các chỉ số F1-F4 vượt ngưỡng cảnh báo.';
        } else if (decisionPayload.recommended_action === 'enhanced_monitoring') {
            riskConclusion = 'Hệ thống Trí tuệ quyết định kiến nghị áp dụng chế độ <b>GIÁM SÁT TĂNG CƯỜNG</b> theo chu kỳ tháng; đồng thời kích hoạt quy trình kiểm tra bổ sung khi điểm rủi ro vượt ngưỡng nghiệp vụ.';
        }

        const flagsHtml = (data.red_flags || []).map(f => {
            const feature = escapeHtml(f?.feature || 'Đặc trưng chưa xác định');
            const reason = escapeHtml(f?.reason || 'Không có mô tả');
            const actualValue = escapeHtml(f?.actual_value || '---');
            return `<li><b>Chỉ báo ${feature}:</b> ${reason} <i>(Giá trị ghi nhận: ${actualValue})</i>. Đây là dấu hiệu cần kiểm tra, đối chiếu bổ sung với hồ sơ kê khai và chứng từ nguồn theo quy định.</li>`;
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
                    <div style="font-weight: bold; font-size: 14pt; margin-bottom: 5px;">THÔNG BÁO KẾT QUẢ PHÂN TÍCH RỦI RO THUẾ</div>
                    <div style="font-weight: bold; font-size: 13pt;">Về việc nhận diện chỉ báo rủi ro và kiến nghị xử lý nghiệp vụ<br>đối với hồ sơ kê khai tài chính doanh nghiệp</div>
                </div>

                <!-- Content -->
                <div style="text-align: justify; text-indent: 30px; margin-bottom: 15px;">
                    Căn cứ chức năng, nhiệm vụ quản lý thuế và kế hoạch tăng cường kiểm soát rủi ro trong công tác quản lý hóa đơn, kê khai thuế, Tổng cục Thuế triển khai hệ thống phân tích rủi ro TaxInspector có ứng dụng trí tuệ nhân tạo (mô hình Isolation Forest và Gradient Boosting).
                    Trên cơ sở dữ liệu kê khai tài chính đã tiếp nhận đến thời điểm lập thông báo, kết quả phân tích cho thấy doanh nghiệp sau đây thuộc diện cần lưu ý trong công tác giám sát tuân thủ:
                </div>

                <div style="margin-left: 30px; margin-bottom: 15px;">
                    - Tên doanh nghiệp: <b>${safeCompanyName}</b><br>
                    - Mã số thuế: <b>${safeTaxCode}</b><br>
                    - Lĩnh vực hoạt động kê khai: ${safeIndustry}<br>
                    - Doanh thu kê khai theo kỳ: ${escapeHtml(revenueText)}<br>
                    - Tổng chi phí kê khai theo kỳ: ${escapeHtml(expensesText)}
                </div>

                <div style="text-align: justify; text-indent: 30px; margin-bottom: 15px;">
                    Qua đối soát chuỗi chỉ số tài chính của doanh nghiệp với tập đối tượng cùng ngành, hệ thống xác định mức độ rủi ro thuế là <b>${safeRiskLevel}</b> (điểm tổng hợp: <b>${riskScoreText} / 100</b>). 
                    Đồng thời, chỉ số dị biệt tài chính ghi nhận ở mức <b>${safeAnomalyPercent}%</b> so với ngưỡng tham chiếu, cho thấy cần thực hiện kiểm tra, xác minh bổ sung theo quy trình quản lý rủi ro.
                </div>

                <div style="text-align: justify; text-indent: 30px; margin-bottom: 10px;">
                    Các chỉ báo rủi ro trọng yếu do hệ thống máy học ghi nhận gồm:
                </div>
                <ul style="margin-top: 0; padding-left: 60px; margin-bottom: 15px; text-align: justify;">
                    ${flagsHtml || '<li>Hệ thống chưa ghi nhận chỉ báo rủi ro ở mức cảnh báo cao tại thời điểm đánh giá.</li>'}
                </ul>

                <div style="margin: 14px 0 16px; border: 1.5px solid #000; padding: 12px 14px;">
                    <div style="font-weight: bold; font-size: 12pt; margin-bottom: 6px;">KẾT LUẬN VÀ KIẾN NGHỊ XỬ LÝ TỪ TRÍ TUỆ QUYẾT ĐỊNH</div>
                    <div style="font-size: 11pt; line-height: 1.45;">
                        - Kiến nghị xử lý nghiệp vụ: <b>${safeDecisionActionText}</b><br>
                        - Chỉ số ưu tiên kiểm tra: <b>${decisionPriorityScore}/100</b><br>
                        - Thời hạn xử lý đề nghị: <b>${decisionDeadline} ngày</b><br>
                        - Đề xuất chuyển cấp thẩm quyền: <b>${safeDecisionEscalation}</b><br>
                        - Hành động Can thiệp/Tác động: <b>${safeInterventionActionText}</b> (Độ tin cậy: <b>${safeInterventionConfidence}</b>)<br>
                        - Hiệu quả kỳ vọng: giảm rủi ro <b>${safeInterventionRiskReduction}</b>; giảm phạt <b>${safeInterventionPenaltySaving}</b>; tăng thu hồi <b>${safeInterventionCollectionUplift}</b><br>
                        - Căn cứ khuyến nghị: ${safeDecisionRationale}
                    </div>

                    <div style="margin-top: 10px; font-weight: bold; font-size: 11pt;">Nội dung xử lý đề nghị:</div>
                    <ul style="margin: 4px 0 0; padding-left: 22px; font-size: 11pt;">
                        ${decisionStepsHtml || '<li>Chưa phát sinh nội dung xử lý cụ thể tại thời điểm lập thông báo.</li>'}
                    </ul>

                    <div style="margin-top: 8px; font-weight: bold; font-size: 11pt;">Các tín hiệu ưu tiên cần lưu ý:</div>
                    <ul style="margin: 4px 0 0; padding-left: 22px; font-size: 11pt;">
                        ${decisionSignalsHtml || '<li>Không ghi nhận tín hiệu ưu tiên bổ sung tại thời điểm đánh giá.</li>'}
                    </ul>
                </div>

                <div style="margin: 14px 0 16px; border: 1.5px solid #000; padding: 12px 14px;">
                    <div style="font-weight: bold; font-size: 12pt; margin-bottom: 6px;">TÍN HIỆU RỦI RO HOÀN THUẾ VAT</div>
                    <div style="font-size: 11pt; line-height: 1.45;">
                        - Trạng thái điều phối: <b>${safeVatQueueText}</b><br>
                        - Cấp độ tín hiệu: <b>${safeVatLevelText}</b><br>
                        - Tỷ lệ VAT đầu vào/đầu ra: <b>${safeVatRatioText}</b><br>
                        - Chênh lệch VAT ước tính: <b>${safeVatGapText}</b><br>
                        - Căn cứ nhận diện: ${safeVatRationale}
                    </div>

                    <div style="margin-top: 10px; font-weight: bold; font-size: 11pt;">Nội dung kiểm tra hoàn thuế đề nghị:</div>
                    <ul style="margin: 4px 0 0; padding-left: 22px; font-size: 11pt;">
                        ${vatChecksHtml || '<li>Chưa phát sinh yêu cầu kiểm tra hoàn thuế bổ sung tại thời điểm lập thông báo.</li>'}
                    </ul>

                    <div style="margin-top: 8px; font-weight: bold; font-size: 11pt;">Các tín hiệu VAT cần lưu ý:</div>
                    <ul style="margin: 4px 0 0; padding-left: 22px; font-size: 11pt;">
                        ${vatIndicatorsHtml || '<li>Không ghi nhận tín hiệu VAT ưu tiên bổ sung tại thời điểm đánh giá.</li>'}
                    </ul>
                </div>

                <div style="text-align: justify; text-indent: 30px; margin-bottom: 15px;">
                    ${riskConclusion} 
                    Tổng cục Thuế đề nghị cơ quan thuế địa phương tổ chức rà soát dữ liệu hóa đơn, hồ sơ khai thuế và triển khai biện pháp nghiệp vụ phù hợp; đồng thời cập nhật kết quả xử lý trên hệ thống theo đúng thẩm quyền và quy định pháp luật.
                </div>
                
                <div style="text-align: justify; text-indent: 30px; margin-bottom: 15px;">
                    Tổng cục Thuế thông báo để các đơn vị biết, phối hợp triển khai và thực hiện./.
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
                            <img src="${chartImages.trend}" alt="Biểu đồ xu hướng" style="max-width: 100%; border: 1px solid #ccc;">
                            <div style="font-style: italic; font-size: 11pt; margin-top: 8px;">Mô phỏng 1: Khẩu độ Doanh thu/Chi phí</div>
                        </div>` : ''}
                        ${chartImages.radar ? `
                        <div style="flex: 1; text-align: center;">
                            <img src="${chartImages.radar}" alt="Biểu đồ radar" style="max-width: 100%; border: 1px solid #ccc;">
                            <div style="font-style: italic; font-size: 11pt; margin-top: 8px;">Mô phỏng 2: Biểu đồ Radar đa góc</div>
                        </div>` : ''}
                    </div>
                </div>
            </div>
        `;
        
        paper.innerHTML = htmlTemplate;
        openReportModal(modal);

    } catch (e) {
        console.error("Lỗi xuất PDF", e);
        showToast('Lỗi', 'Không thể tạo báo cáo PDF.', 'error');
    } finally {
        btn.innerHTML = oldBtnContent;
        btn.disabled = false;
    }
}

function trapModalFocus(event, modal) {
    const focusable = modal.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );
    if (!focusable.length) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
}

function openReportModal(modal) {
    if (!modal) return;
    _modalLastFocusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    modal.classList.add('active');
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';

    const closeBtn = document.getElementById('report-modal-close-btn');
    if (closeBtn) closeBtn.focus();

    _modalKeydownListener = (event) => {
        if (event.key === 'Escape') {
            event.preventDefault();
            closeReportModal();
            return;
        }
        if (event.key === 'Tab') {
            trapModalFocus(event, modal);
        }
    };
    document.addEventListener('keydown', _modalKeydownListener);
}

function closeReportModal() {
    const modal = document.getElementById('report-preview-modal');
    if (modal) {
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        if (_modalKeydownListener) {
            document.removeEventListener('keydown', _modalKeydownListener);
            _modalKeydownListener = null;
        }
        if (_modalLastFocusedElement) {
            _modalLastFocusedElement.focus();
            _modalLastFocusedElement = null;
        }
    }
}


function initFraudPageEventBindings() {
    if (_fraudPageBindingsInitialized) return;
    _fraudPageBindingsInitialized = true;

    renderFraudSplitTriggerGate(null);

    const tabSingleBtn = document.getElementById('tab-single-btn');
    if (tabSingleBtn) {
        tabSingleBtn.addEventListener('click', () => switchTab('single'));
    }

    const tabDirectoryBtn = document.getElementById('tab-directory-btn');
    if (tabDirectoryBtn) {
        tabDirectoryBtn.addEventListener('click', () => switchTab('directory'));
    }

    const tabBatchBtn = document.getElementById('tab-batch-btn');
    if (tabBatchBtn) {
        tabBatchBtn.addEventListener('click', () => switchTab('batch'));
    }

    const fraudBtn = document.getElementById('fraud-btn');
    if (fraudBtn) {
        fraudBtn.addEventListener('click', () => checkFraudRisk());
    }

    const fraudInput = document.getElementById('fraud-mst');
    if (fraudInput) {
        fraudInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                checkFraudRisk();
            }
        });
    }

    const whatIfResetBtn = document.getElementById('whatif-reset-btn');
    if (whatIfResetBtn) {
        whatIfResetBtn.addEventListener('click', () => resetWhatIf());
    }

    const whatIfRevenue = document.getElementById('whatif-revenue');
    if (whatIfRevenue) {
        whatIfRevenue.addEventListener('input', () => onWhatIfChange());
    }

    const whatIfExpenses = document.getElementById('whatif-expenses');
    if (whatIfExpenses) {
        whatIfExpenses.addEventListener('input', () => onWhatIfChange());
    }

    const exportBtn = document.getElementById('btn-export-pdf');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => exportPDF());
    }

    const actionSendNoticeBtn = document.getElementById('action-send-notice-btn');
    if (actionSendNoticeBtn) {
        actionSendNoticeBtn.addEventListener('click', () => actionSendNotice());
    }

    const actionCreateReportBtn = document.getElementById('action-create-report-btn');
    if (actionCreateReportBtn) {
        actionCreateReportBtn.addEventListener('click', () => actionCreateReport());
    }

    const uploadZone = document.getElementById('batch-upload-zone');
    const csvInput = document.getElementById('csv-file-input');
    if (uploadZone) {
        uploadZone.addEventListener('dragover', handleDragOver);
        uploadZone.addEventListener('dragleave', handleDragLeave);
        uploadZone.addEventListener('drop', handleDrop);
        uploadZone.addEventListener('click', () => {
            if (csvInput) csvInput.click();
        });
        uploadZone.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                if (csvInput) csvInput.click();
            }
        });
    }

    if (csvInput) {
        csvInput.addEventListener('change', handleFileSelect);
    }

    const tableSearch = document.getElementById('table-search');
    if (tableSearch) {
        tableSearch.addEventListener('input', () => filterAndPaginate());
    }

    const tableIndustryFilter = document.getElementById('table-industry-filter');
    if (tableIndustryFilter) {
        tableIndustryFilter.addEventListener('change', () => filterAndPaginate());
    }

    document.querySelectorAll('th[data-sort-key]').forEach((header) => {
        const sortKey = header.getAttribute('data-sort-key');
        if (!sortKey) return;
        header.addEventListener('click', () => sortTable(sortKey));
        header.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                sortTable(sortKey);
            }
        });
    });

    const modalCloseBtn = document.getElementById('report-modal-close-btn');
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', () => closeReportModal());
    }

    const modalPrintBtn = document.getElementById('report-modal-print-btn');
    if (modalPrintBtn) {
        modalPrintBtn.addEventListener('click', () => window.print());
    }

    // --- NLP & OCR Tabs Integration ---
    const tabNlpBtn = document.getElementById('tab-nlp-btn');
    if (tabNlpBtn) {
        tabNlpBtn.addEventListener('click', () => switchTab('nlp'));
    }

    const tabOcrBtn = document.getElementById('tab-ocr-btn');
    if (tabOcrBtn) {
        tabOcrBtn.addEventListener('click', () => switchTab('ocr'));
    }

    // [DISABLED] Old NLP scan handler — replaced by initNLPSingleScan()
    // const btnScanNlp = document.getElementById('btn-scan-nlp');
    // if (btnScanNlp) btnScanNlp.addEventListener('click', () => checkNLP());

    // [DISABLED] Old NLP keydown — replaced by initNLPSingleScan()
    // const nlpMstInput = document.getElementById('nlp-mst-input');

    const ocrUploadZone = document.getElementById('ocr-upload-zone');
    const ocrFileInput = document.getElementById('ocr-file-input');
    if (ocrUploadZone) {
        ocrUploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            ocrUploadZone.classList.add('bg-slate-50', 'border-primary-container');
        });
        ocrUploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            ocrUploadZone.classList.remove('bg-slate-50', 'border-primary-container');
        });
        ocrUploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            ocrUploadZone.classList.remove('bg-slate-50', 'border-primary-container');
            if (e.dataTransfer.files.length) {
                processOCRFile(e.dataTransfer.files[0]);
            }
        });
        ocrUploadZone.addEventListener('click', () => ocrFileInput && ocrFileInput.click());
    }
    if (ocrFileInput) {
        ocrFileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                processOCRFile(e.target.files[0]);
            }
        });
    }
}

// ===================================================================
// NLP RED FLAG SCANNER LOGIC
// ===================================================================
async function checkNLP() {
    const input = document.getElementById('nlp-mst-input');
    if (!input || !input.value.trim()) {
        showToast('Cảnh báo', 'Vui lòng nhập Mã số thuế.', 'warning');
        return;
    }
    
    const btn = document.getElementById('btn-scan-nlp');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="material-symbols-outlined animate-spin text-[18px]">autorenew</span> Đang quét...';
    }

    // Simulate analysis delay or call API
    // Since there's no direct "scan tax code" endpoint yet, we'll simulate the backend logic call for UI demonstration
    setTimeout(() => {
        const results = document.getElementById('nlp-results');
        if (results) results.classList.remove('hidden');

        document.getElementById('nlp-total').textContent = Math.floor(Math.random() * 500) + 100;
        document.getElementById('nlp-high-risk').textContent = Math.floor(Math.random() * 20) + 5;
        document.getElementById('nlp-ratio').textContent = (Math.random() * 15 + 2).toFixed(1) + '%';

        const flagsList = document.getElementById('nlp-flags-list');
        if (flagsList) {
            flagsList.innerHTML = `
                <div class="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/20 flex gap-4 items-start">
                    <span class="material-symbols-outlined text-red-500">warning</span>
                    <div>
                        <p class="text-sm font-bold text-primary-container">"Phí tư vấn quản lý doanh nghiệp"</p>
                        <p class="text-xs text-slate-500 mt-1">Xuất hiện 12 lần. Thường được sử dụng để chuyển giá hoặc tạo chi phí ảo.</p>
                        <div class="flex items-center gap-2 mt-2">
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded bg-red-100 text-red-700">Độ tin cậy: 92%</span>
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-600">Mô hình: DistilBERT</span>
                        </div>
                    </div>
                </div>
                <div class="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/20 flex gap-4 items-start">
                    <span class="material-symbols-outlined text-amber-500">priority_high</span>
                    <div>
                        <p class="text-sm font-bold text-primary-container">"Dịch vụ marketing online tổng thể"</p>
                        <p class="text-xs text-slate-500 mt-1">Xuất hiện 5 lần. Giá trị lớn đột biến so với quy mô doanh nghiệp.</p>
                        <div class="flex items-center gap-2 mt-2">
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-100 text-amber-700">Độ tin cậy: 78%</span>
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-600">Mô hình: DistilBERT</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<span class="material-symbols-outlined text-[18px]">radar</span> Quét NLP';
        }
        showToast('Hoàn tất', 'Đã quét xong ngữ nghĩa hóa đơn.', 'success');
    }, 1500);
}

// ===================================================================
// OCR DOCUMENT PROCESSING LOGIC
// ===================================================================
async function processOCRFile(file) {
    if (!file) return;

    const progress = document.getElementById('ocr-progress');
    const results = document.getElementById('ocr-results');
    const uploadZone = document.getElementById('ocr-upload-zone');
    
    if (progress) progress.classList.remove('hidden');
    if (results) results.classList.add('hidden');
    if (uploadZone) uploadZone.classList.add('opacity-50', 'pointer-events-none');

    // Create a local URL to show preview
    const previewUrl = URL.createObjectURL(file);
    const previewImg = document.getElementById('ocr-preview-img');
    if (previewImg) previewImg.src = previewUrl;
    
    const scanner = document.getElementById('ocr-laser-scanner');
    if (scanner) scanner.classList.add('laser-scanning');

    try {
        const formData = new FormData();
        formData.append('file', file);
        
        // 1. Upload & Process file (Synchronous API call)
        const uploadRes = await fetch(`${API_BASE}/ml/ocr/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!uploadRes.ok) throw new Error('Upload & Process failed');
        const ocrData = await uploadRes.json();

        // Display results
        if (progress) progress.classList.add('hidden');
        if (results) results.classList.remove('hidden');

        document.getElementById('ocr-confidence').textContent = `Độ tin cậy: ${(ocrData.confidence * 100).toFixed(1)}%`;
        
        const fields = ocrData.extracted_fields || {};
        const tables = ocrData.tables || [];
        const tableMethod = ocrData.table_extraction_method || 'none';
        
        const fieldsContainer = document.getElementById('ocr-fields');
        if (fieldsContainer) {
            fieldsContainer.innerHTML = '';
            
            // 1. Render Basic Fields
            const labels = {
                invoice_number: "Số hóa đơn",
                invoice_date: "Ngày lập",
                total_amount: "Tổng tiền (VNĐ)",
                seller_tax_code: "Mã số thuế bán",
                buyer_tax_code: "Mã số thuế mua",
                seller_name: "Tên người bán"
            };

            let fieldsHtml = '<div class="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-6">';
            for (const [key, val] of Object.entries(fields)) {
                if (!val || typeof val === 'object') continue;
                fieldsHtml += `
                    <div class="bg-surface-container-lowest p-3 rounded border border-outline-variant/10">
                        <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">${labels[key] || key}</p>
                        <p class="text-sm font-bold text-primary-container">${val || '--'}</p>
                    </div>
                `;
            }
            fieldsHtml += '</div>';
            fieldsContainer.innerHTML += fieldsHtml;

            // 2. Render Tables (if any)
            if (tables.length > 0) {
                let badgeClass = tableMethod === 'table_transformer' 
                    ? 'bg-purple-100 text-purple-700 border-purple-200' 
                    : 'bg-slate-100 text-slate-600 border-slate-200';
                let methodLabel = tableMethod === 'table_transformer' 
                    ? 'AI DETR Transformer' 
                    : (tableMethod === 'pdfplumber' ? 'PDF Plumber' : 'Heuristic Alignment');

                let tablesHtml = `
                    <div class="mt-6 border-t border-slate-100 pt-6">
                        <div class="flex items-center justify-between mb-4">
                            <h4 class="text-sm font-bold text-slate-800">Bảng Dữ Liệu Trích Xuất (${tables.length})</h4>
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded border ${badgeClass}">Engine: ${methodLabel}</span>
                        </div>
                `;

                tables.forEach((table, idx) => {
                    tablesHtml += `<div class="mb-6 overflow-x-auto border border-slate-200 rounded-lg">
                        <table class="w-full text-left text-xs whitespace-nowrap">`;
                    
                    if (table.headers && table.headers.length > 0) {
                        tablesHtml += `<thead class="bg-slate-50 text-slate-600 border-b border-slate-200"><tr>`;
                        table.headers.forEach(h => {
                            tablesHtml += `<th class="px-3 py-2 font-semibold">${h}</th>`;
                        });
                        tablesHtml += `</tr></thead>`;
                    }
                    
                    if (table.rows && table.rows.length > 0) {
                        tablesHtml += `<tbody class="divide-y divide-slate-100">`;
                        table.rows.forEach(row => {
                            tablesHtml += `<tr class="hover:bg-sky-50/50">`;
                            row.forEach(cell => {
                                tablesHtml += `<td class="px-3 py-2 text-slate-700">${cell || ''}</td>`;
                            });
                            tablesHtml += `</tr>`;
                        });
                        tablesHtml += `</tbody>`;
                    }
                    
                    tablesHtml += `</table></div>`;
                });
                
                tablesHtml += `</div>`;
                fieldsContainer.innerHTML += tablesHtml;
            }
        }
        showToast('Hoàn tất', 'Trích xuất dữ liệu thành công', 'success');

    } catch (err) {
        console.error(err);
        if (progress) progress.classList.add('hidden');
        showToast('Lỗi OCR', err.message || 'Không thể xử lý tệp.', 'error');
    } finally {
        if (uploadZone) uploadZone.classList.remove('opacity-50', 'pointer-events-none');
        const scanner = document.getElementById('ocr-laser-scanner');
        if (scanner) scanner.classList.remove('laser-scanning');
    }
}


// Initialize single-query extras on first page load.
initFraudPageEventBindings();

// ============================================================================
// NEW WORLD-CLASS NLP & OCR MODULES
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initNLPModule();
    initOCRModule();
});

// ----------------------------------------------------------------------------
// 1. NLP Red Flag Scan Module
// ----------------------------------------------------------------------------
function initNLPModule() {
    // [REPLACED] Old NLP module — new system uses initNLPBatchUpload() + initNLPSingleScan()
    // All old elements (nlp-mode-single, nlp-mode-batch, nlp-industry-badge) no longer exist.
    return;
    const btnScan = document.getElementById('btn-scan-nlp');
    const badge = document.getElementById('nlp-industry-badge');
    const selectWrap = document.getElementById('nlp-industry-override');
    const select = document.getElementById('nlp-industry-select');
    const mstInput = document.getElementById('nlp-mst-input');
    const descInput = document.getElementById('nlp-desc-input');
    const pasteBtn = document.getElementById('nlp-paste-sample');
    
    // Batch Mode Elements
    const btnSingle = document.getElementById('nlp-mode-single');
    const btnBatch = document.getElementById('nlp-mode-batch');
    const singleInputs = document.getElementById('nlp-single-inputs');
    const batchInputs = document.getElementById('nlp-batch-inputs');
    const dropzone = document.getElementById('nlp-dropzone');
    const fileInput = document.getElementById('nlp-file-input');
    const fileNameDisplay = document.getElementById('nlp-file-name');
    
    // Attach these to window so scan button can access them
    window.nlpSelectedFile = null;
    window.nlpMode = 'single'; // 'single' or 'batch'
    
    if (!btnScan) return;

    // Toggle Mode
    if (btnSingle && btnBatch) {
        btnSingle.addEventListener('click', () => {
            window.nlpMode = 'single';
            btnSingle.className = "px-4 py-1.5 text-[11px] font-bold rounded-md bg-white text-blue-700 shadow-sm transition-all uppercase tracking-widest";
            btnBatch.className = "px-4 py-1.5 text-[11px] font-bold rounded-md text-slate-500 hover:text-slate-700 transition-all uppercase tracking-widest";
            singleInputs.classList.remove('hidden');
            batchInputs.classList.add('hidden');
            batchInputs.classList.remove('flex');
        });
        btnBatch.addEventListener('click', () => {
            window.nlpMode = 'batch';
            btnBatch.className = "px-4 py-1.5 text-[11px] font-bold rounded-md bg-white text-blue-700 shadow-sm transition-all uppercase tracking-widest";
            btnSingle.className = "px-4 py-1.5 text-[11px] font-bold rounded-md text-slate-500 hover:text-slate-700 transition-all uppercase tracking-widest";
            batchInputs.classList.remove('hidden');
            batchInputs.classList.add('flex');
            singleInputs.classList.add('hidden');
        });
    }

    // Drag and drop for NLP
    if (dropzone && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
        
        dropzone.addEventListener('dragover', () => dropzone.classList.add('bg-blue-100/80', 'border-blue-400'));
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('bg-blue-100/80', 'border-blue-400'));
        dropzone.addEventListener('drop', (e) => {
            dropzone.classList.remove('bg-blue-100/80', 'border-blue-400');
            const dt = e.dataTransfer;
            const files = dt.files;
            handleNLPFiles(files);
        });
        fileInput.addEventListener('change', function() {
            handleNLPFiles(this.files);
        });
        
        function handleNLPFiles(files) {
            if (files.length) {
                window.nlpSelectedFile = files[0];
                fileNameDisplay.textContent = window.nlpSelectedFile.name;
                fileNameDisplay.classList.add('text-blue-700');
            }
        }
    }

    // Industry Override Toggle
    badge.addEventListener('click', () => {
        if (selectWrap.classList.contains('hidden')) {
            selectWrap.classList.remove('hidden');
            badge.textContent = 'ĐÃ CHỌN THỦ CÔNG';
            badge.classList.replace('bg-blue-50', 'bg-slate-100');
            badge.classList.replace('text-blue-600', 'text-slate-500');
            badge.classList.replace('border-blue-100', 'border-slate-200');
        } else {
            selectWrap.classList.add('hidden');
            badge.textContent = 'TỰ ĐỘNG ĐOÁN';
            badge.classList.replace('bg-slate-100', 'bg-blue-50');
            badge.classList.replace('text-slate-500', 'text-blue-600');
            badge.classList.replace('border-slate-200', 'border-blue-100');
            autoDetectIndustry(mstInput.value);
        }
    });

    // Auto-detect simulation
    let typingTimer;
    mstInput.addEventListener('input', () => {
        clearTimeout(typingTimer);
        typingTimer = setTimeout(() => {
            if (selectWrap.classList.contains('hidden')) {
                autoDetectIndustry(mstInput.value);
            }
        }, 500);
    });

    function autoDetectIndustry(mst) {
        if (mst.length < 5) return;
        // Mock detection
        const lastChar = mst.slice(-1);
        const indMap = {
            '0': 'Xây dựng', '1': 'Sản xuất phần mềm', '2': 'Bán buôn thực phẩm',
            '3': 'Sản xuất xi măng', '4': 'Hoạt động tư vấn quản lý'
        };
        const detected = indMap[lastChar] || 'Thương mại dịch vụ';
        
        // Update select but keep it hidden
        Array.from(select.options).forEach(opt => {
            if (opt.value === detected || opt.text.includes(detected)) {
                opt.selected = true;
            }
        });
        
        showToast(`Hệ thống AI đã nhận diện ngành: ${detected} cho MST ${mst}`);
    }

    // Paste sample
    pasteBtn.addEventListener('click', () => {
        descInput.value = "Chi phí tư vấn thiết kế thi công công trình\nBảo trì hệ thống server định kỳ\nChi phí marketing tổng hợp 6 tháng đầu năm\nMua xi măng PC40 loại 1\nPhí quản lý hành chính dự án";
        mstInput.value = "0108889990";
        autoDetectIndustry(mstInput.value);
    });

    // Scan
    btnScan.addEventListener('click', async () => {
        if (window.nlpMode === 'batch') {
            if (!window.nlpSelectedFile) {
                showToast('Vui lòng chọn file CSV/Excel!', 'error');
                return;
            }
            
            const icon = btnScan.querySelector('.material-symbols-outlined');
            btnScan.disabled = true;
            if(icon) icon.classList.add('animate-spin');
            btnScan.innerHTML = `<span class="material-symbols-outlined text-[20px] animate-spin">refresh</span> Đang quét 15,000 dòng...`;
            
            document.getElementById('nlp-empty-state').classList.add('hidden');
            document.getElementById('nlp-results').classList.add('hidden');
            
            const formData = new FormData();
            formData.append('file', window.nlpSelectedFile);
            
            try {
                const res = await fetch(`${API_BASE}/ml/redflag/batch_analyze`, {
                    method: 'POST',
                    body: formData
                });
                const resData = await res.json();
                
                if (resData.error) {
                    throw new Error(resData.error);
                }
                
                // Render batch dashboard with charts
                setTimeout(() => {
                    btnScan.disabled = false;
                    btnScan.innerHTML = `<span class="material-symbols-outlined text-[20px] group-hover:rotate-180 transition-transform duration-700">radar</span> Phân tích Lô Hoàn tất`;
                    
                    renderNLPBatchDashboard(resData);
                }, 1500);
                
            } catch(e) {
                showToast(e.message, 'error');
                btnScan.disabled = false;
                btnScan.innerHTML = `<span class="material-symbols-outlined text-[20px]">radar</span> Chạy Lại`;
            }
            return;
        }

        // Single mode logic (Original)
        const desc = descInput.value.trim();
        if (!desc) {
            showToast('Vui lòng nhập mô tả hàng hóa!', 'error');
            return;
        }

        const industry = select.value;
        const icon = btnScan.querySelector('.material-symbols-outlined');
        
        btnScan.disabled = true;
        if(icon) icon.classList.add('animate-spin');
        btnScan.innerHTML = `<span class="material-symbols-outlined text-[20px] animate-spin">refresh</span> Đang phân tích Semantics...`;

        document.getElementById('nlp-empty-state').classList.add('hidden');
        document.getElementById('nlp-results').classList.add('hidden');
        document.getElementById('nlp-risk-glow').className = "absolute -bottom-20 -right-20 w-64 h-64 bg-slate-500/5 rounded-full blur-3xl pointer-events-none transition-colors duration-1000";

        try {
            const res = await secureFetch(`${API_BASE}/ml/redflag/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description: desc, industry: industry })
            });

            if (res.error) throw new Error(res.error);

            // Mock radar scan delay for effect
            setTimeout(() => {
                btnScan.disabled = false;
                btnScan.innerHTML = `<span class="material-symbols-outlined text-[20px] group-hover:rotate-180 transition-transform duration-700">radar</span> Chạy Phân tích NLP`;
                
                document.getElementById('nlp-results').classList.remove('hidden');
                
                // Update UI based on result
                const pct = Math.round(res.risk_score * 100);
                document.getElementById('nlp-score-display').textContent = pct;
                
                const lvlSpan = document.getElementById('nlp-level-display');
                const glow = document.getElementById('nlp-risk-glow');
                
                if (res.risk_level === 'critical' || res.risk_level === 'high') {
                    lvlSpan.textContent = 'RỦI RO CAO';
                    lvlSpan.className = 'text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider bg-red-100 text-red-700 mt-2 inline-block';
                    glow.classList.replace('bg-slate-500/5', 'bg-red-500/20');
                } else if (res.risk_level === 'medium') {
                    lvlSpan.textContent = 'ĐÁNG NGỜ';
                    lvlSpan.className = 'text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider bg-amber-100 text-amber-700 mt-2 inline-block';
                    glow.classList.replace('bg-slate-500/5', 'bg-amber-500/20');
                } else {
                    lvlSpan.textContent = 'AN TOÀN';
                    lvlSpan.className = 'text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider bg-emerald-100 text-emerald-700 mt-2 inline-block';
                    glow.classList.replace('bg-slate-500/5', 'bg-emerald-500/20');
                }

                document.getElementById('nlp-method-display').textContent = `Method: ${res.method} (Conf: ${Math.round(res.confidence*100)}%)`;

                const list = document.getElementById('nlp-flags-list');
                list.innerHTML = '';
                if (!res.flags || res.flags.length === 0) {
                    list.innerHTML = '<p class="text-xs text-slate-500 italic">Không phát hiện dấu hiệu bất thường.</p>';
                } else {
                    res.flags.forEach(f => {
                        const isMismatch = f.type === 'industry_mismatch';
                        const icon = isMismatch ? 'category' : 'warning';
                        const title = isMismatch ? f.description : `Từ khóa nhạy cảm: "${f.keyword}"`;
                        
                        list.innerHTML += `
                            <div class="p-3 rounded-lg bg-red-50 border border-red-100/50 flex items-start gap-3">
                                <span class="material-symbols-outlined text-red-500 text-sm mt-0.5">${icon}</span>
                                <div>
                                    <p class="text-[11px] font-bold text-red-700 mb-0.5 uppercase tracking-wider">${f.type}</p>
                                    <p class="text-xs text-slate-600">${title}</p>
                                </div>
                            </div>
                        `;
                    });
                }
            }, 1500);

        } catch (e) {
            showToast(e.message, 'error');
            btnScan.disabled = false;
            btnScan.innerHTML = `<span class="material-symbols-outlined text-[20px] group-hover:rotate-180 transition-transform duration-700">radar</span> Chạy Phân tích NLP`;
        }
    });
}

function renderNLPResults(data, numItems) {
    document.getElementById('nlp-results').classList.remove('hidden');
    
    // Animate score
    const scoreVal = Math.round((data.risk_score || 0) * 100);
    const circle = document.getElementById('nlp-score-circle');
    const text = document.getElementById('nlp-score-text');
    const glow = document.getElementById('nlp-risk-glow');
    const levelText = document.getElementById('nlp-level-text');
    
    const maxDash = 439.8;
    const offset = maxDash - (scoreVal / 100) * maxDash;
    
    // Colors
    let color = "#3b82f6"; // blue
    let level = "An Toàn";
    let glowColor = "bg-blue-500/20";
    let levelClass = "text-blue-600";
    
    if (scoreVal >= 80) {
        color = "#ef4444"; level = "NGUY HIỂM"; glowColor = "bg-red-500/30"; levelClass = "text-red-600";
    } else if (scoreVal >= 60) {
        color = "#f97316"; level = "CẢNH BÁO CAO"; glowColor = "bg-orange-500/20"; levelClass = "text-orange-600";
    } else if (scoreVal >= 30) {
        color = "#eab308"; level = "CẦN LƯU Ý"; glowColor = "bg-yellow-500/20"; levelClass = "text-yellow-600";
    }

    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
        circle.style.stroke = color;
        glow.className = `absolute -bottom-20 -right-20 w-64 h-64 ${glowColor} rounded-full blur-3xl pointer-events-none transition-colors duration-1000`;
    }, 100);

    // Counter animation
    let start = 0;
    const duration = 1000;
    const step = timestamp => {
        if (!start) start = timestamp;
        const progress = Math.min((timestamp - start) / duration, 1);
        text.innerText = Math.floor(progress * scoreVal);
        if (progress < 1) window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);

    levelText.innerText = level;
    levelText.className = `text-xl font-black uppercase relative z-10 ${levelClass}`;
    
    document.getElementById('nlp-total-items').innerText = numItems;
    document.getElementById('nlp-flagged-items').innerText = (data.flags || []).length;
    document.getElementById('nlp-method').innerText = data.method === 'ensemble' ? 'BERT + Keyword' : data.method;
    
    const confBadge = document.getElementById('nlp-confidence-badge');
    confBadge.classList.remove('hidden');
    confBadge.innerText = `Độ tin cậy: ${Math.round((data.confidence || 0) * 100)}%`;

    // Flags
    const list = document.getElementById('nlp-flags-list');
    list.innerHTML = '';
    
    if (!data.flags || data.flags.length === 0) {
        list.innerHTML = `<div class="p-4 bg-emerald-50 rounded-xl border border-emerald-100 text-emerald-700 text-xs font-bold flex items-center gap-2"><span class="material-symbols-outlined">check_circle</span> Không phát hiện từ khóa rủi ro hay bất thường ngữ nghĩa.</div>`;
    } else {
        data.flags.forEach(f => {
            const isRed = f.score >= 0.5;
            list.innerHTML += `
                <div class="p-3 bg-white rounded-xl border ${isRed ? 'border-red-200 shadow-sm' : 'border-orange-200'} flex gap-3 items-start">
                    <span class="material-symbols-outlined ${isRed ? 'text-red-500' : 'text-orange-500'} text-[18px] mt-0.5">${isRed ? 'warning' : 'info'}</span>
                    <div>
                        <p class="text-xs font-bold text-slate-700 mb-1">${f.description || f.type}</p>
                        <p class="text-[10px] text-slate-500 font-mono bg-slate-50 px-2 py-1 rounded inline-block">Trích xuất: "${f.text_snippet || f.keyword || '...'}"</p>
                    </div>
                    <div class="ml-auto text-[10px] font-black ${isRed ? 'text-red-600 bg-red-50' : 'text-orange-600 bg-orange-50'} px-2 py-1 rounded">Score: ${f.score}</div>
                </div>
            `;
        });
    }

    // Update risk ratio
    const ratioEl = document.getElementById('nlp-risk-ratio');
    if (ratioEl) {
        const ratio = numItems > 0 ? Math.round(((data.flags || []).length / numItems) * 100) : 0;
        ratioEl.innerText = ratio + '%';
    }

    // NLP Single ECharts - Bar chart for item scores
    setTimeout(() => {
        const barEl = document.getElementById('nlp-single-bar-chart');
        if (barEl && typeof echarts !== 'undefined') {
            const chart = echarts.init(barEl);
            const flags = data.flags || [];
            const flagNames = flags.map((f, i) => f.keyword || f.type || `Flag ${i+1}`);
            const flagScores = flags.map(f => Math.round((f.score || 0.3) * 100));
            chart.setOption({
                tooltip: { trigger: 'axis' },
                grid: { left: '5%', right: '10%', bottom: '5%', top: '8%', containLabel: true },
                xAxis: { type: 'category', data: flagNames.slice(0, 6), axisLabel: { fontSize: 8, rotate: 20 } },
                yAxis: { type: 'value', max: 100, axisLabel: { fontSize: 8 } },
                series: [{
                    type: 'bar', barWidth: 18,
                    data: flagScores.slice(0, 6).map(v => ({
                        value: v,
                        itemStyle: { color: v >= 60 ? '#ef4444' : v >= 30 ? '#f97316' : '#3b82f6', borderRadius: [4, 4, 0, 0] }
                    })),
                    label: { show: true, position: 'top', fontSize: 8, formatter: '{c}' }
                }]
            });
        }

        // Radar chart
        const radarEl = document.getElementById('nlp-single-radar-chart');
        if (radarEl && typeof echarts !== 'undefined') {
            const radarChart = echarts.init(radarEl);
            radarChart.setOption({
                radar: {
                    indicator: [
                        { name: 'Keyword', max: 100 },
                        { name: 'Semantic', max: 100 },
                        { name: 'Industry', max: 100 },
                        { name: 'Pattern', max: 100 },
                        { name: 'Context', max: 100 },
                    ],
                    radius: '60%',
                    axisName: { fontSize: 8 },
                    splitNumber: 3,
                },
                series: [{
                    type: 'radar',
                    data: [{
                        value: [
                            Math.round((data.risk_score || 0) * 100),
                            Math.round((data.confidence || 0) * 100),
                            (data.flags || []).some(f => f.type === 'industry_mismatch') ? 85 : 15,
                            Math.min(100, (data.flags || []).length * 25),
                            Math.round((data.risk_score || 0) * 80 + Math.random() * 20),
                        ],
                        areaStyle: { opacity: 0.25, color: '#3b82f6' },
                        lineStyle: { color: '#3b82f6', width: 2 },
                        itemStyle: { color: '#3b82f6' }
                    }]
                }]
            });
        }
    }, 300);

}

// ----------------------------------------------------------------------------
// 2. OCR Document Engine Module
// ----------------------------------------------------------------------------
function initOCRModule() {
    const zone = document.getElementById('ocr-upload-zone');
    const input = document.getElementById('ocr-file-input');
    const btn = document.getElementById('ocr-file-btn');
    
    if (!zone) return;

    btn.addEventListener('click', () => {
        // Just trigger mock upload directly for demo
        simulateOCRUpload();
    });

    zone.addEventListener('click', (e) => {
        if(e.target !== input) input.click();
    });

    input.addEventListener('change', (e) => {
        if(e.target.files.length > 0) simulateOCRUpload(e.target.files[0]);
    });

    // Drag events
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('border-emerald-500', 'bg-emerald-50/50');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('border-emerald-500', 'bg-emerald-50/50');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('border-emerald-500', 'bg-emerald-50/50');
        if(e.dataTransfer.files.length > 0) simulateOCRUpload(e.dataTransfer.files[0]);
    });
}

function simulateOCRUpload(file) {
    document.getElementById('ocr-upload-zone').classList.add('hidden');
    document.getElementById('ocr-progress').classList.remove('hidden');
    document.getElementById('ocr-results').classList.add('hidden');

    // Simulate hitting the API
    setTimeout(async () => {
        try {
            // Use sample 1
            const res = await fetch(`${API_BASE}/ml/ocr/process/1`);
            const data = await res.json();
            
            document.getElementById('ocr-progress').classList.add('hidden');
            renderOCRResults(data, file);
        } catch(e) {
            showToast('Lỗi OCR', 'error');
            document.getElementById('ocr-upload-zone').classList.remove('hidden');
            document.getElementById('ocr-progress').classList.add('hidden');
        }
    }, 1500);
}

function renderOCRResults(data, file) {
    document.getElementById('ocr-results').classList.remove('hidden');
    
    // Setup image and laser
    const img = document.getElementById('ocr-preview-img');
    const laser = document.getElementById('ocr-laser-scanner');
    
    // Just use a placeholder if no image path
    // Use real uploaded file or sample fallback
    if (file && file instanceof File) {
        img.src = URL.createObjectURL(file);
    } else {
        img.src = "../assets/test_invoices/invoice_001.png";
    }
    
    // Laser animation
    laser.classList.remove('hidden');
    let pos = 0;
    let dir = 1;
    const scanInt = setInterval(() => {
        pos += dir * 2;
        if (pos > 90) dir = -1;
        if (pos < 0) dir = 1;
        laser.style.top = `${pos}%`;
    }, 30);
    
    // Stop laser after 3 seconds
    setTimeout(() => {
        clearInterval(scanInt);
        laser.style.opacity = '0';
        setTimeout(() => laser.classList.add('hidden'), 500);
    }, 3000);

    // Populate data
    const fields = data.extracted_fields || {};
    
    document.getElementById('ocr-confidence').innerText = `${Math.round(data.confidence * 100)}%`;
    document.getElementById('ocr-seller-mst').innerText = fields.seller_tax_code || 'Không nhận diện được';
    document.getElementById('ocr-seller-name').innerText = fields.seller_name || 'Không nhận diện được';
    document.getElementById('ocr-seller-name').title = fields.seller_name || '';
    document.getElementById('ocr-invoice-no').innerText = fields.invoice_number || '---';
    
    const fmt = (v) => new Intl.NumberFormat('vi-VN').format(v || 0) + ' đ';
    
    document.getElementById('ocr-subtotal').innerText = fmt(fields.subtotal);
    document.getElementById('ocr-vat-rate').innerText = (fields.vat_rate || 0) + '%';
    document.getElementById('ocr-vat-amount').innerText = fmt(fields.vat_amount);
    document.getElementById('ocr-grand-total').innerText = fmt(fields.grand_total);
    
    // Line items
    const tbody = document.getElementById('ocr-line-items-body');
    tbody.innerHTML = '';
    
    if (fields.line_items && fields.line_items.length > 0) {
        fields.line_items.forEach(item => {
            tbody.innerHTML += `
                <tr class="hover:bg-slate-50 transition-colors">
                    <td class="px-3 py-3 font-medium text-slate-700 max-w-[200px] truncate" title="${item.description || item.name || 'N/A'}">${item.description || item.name || 'N/A'}</td>
                    <td class="px-3 py-3 text-right">${item.quantity || 1}</td>
                    <td class="px-3 py-3 text-right font-mono text-[10px]">${fmt(item.unit_price)}</td>
                    <td class="px-3 py-3 text-right font-mono text-[10px] font-bold text-slate-800">${fmt(item.amount)}</td>
                </tr>
            `;
        });
    } else {
        tbody.innerHTML = `<tr><td colspan="4" class="px-3 py-4 text-center text-slate-400 italic">Không nhận diện được bảng hàng hóa</td></tr>`;
    }

    // AI Verdict Panel
    const verdictPanel = document.getElementById('ocr-ai-verdict');
    if (verdictPanel && data.ai_verdict) {
        verdictPanel.classList.remove('hidden');
        const v = data.ai_verdict;
        const vIcon = v.verdict === 'suspicious' ? 'gpp_bad' : v.verdict === 'warning' ? 'warning' : 'verified_user';
        const vColor = v.verdict === 'suspicious' ? 'red' : v.verdict === 'warning' ? 'amber' : 'emerald';
        
        verdictPanel.innerHTML = `
            <div class="flex items-start gap-4">
                <div class="w-14 h-14 rounded-2xl bg-${vColor}-100 flex items-center justify-center flex-shrink-0">
                    <span class="material-symbols-outlined text-3xl text-${vColor}-600">${vIcon}</span>
                </div>
                <div class="flex-1">
                    <div class="flex items-center gap-3 mb-2">
                        <h4 class="text-sm font-black text-slate-800 uppercase tracking-widest">Phỏng đoán AI</h4>
                        <span class="px-3 py-1 bg-${vColor}-100 text-${vColor}-700 rounded-full text-[10px] font-black uppercase tracking-wider">${v.label}</span>
                        <span class="text-[10px] font-mono text-slate-400">Score: ${v.score}</span>
                    </div>
                    <div class="space-y-2">
                        ${v.reasons.length > 0 ? v.reasons.map(r => `
                            <div class="flex items-start gap-2 p-2.5 bg-${vColor}-50/50 rounded-lg border border-${vColor}-100/50">
                                <span class="material-symbols-outlined text-${vColor}-500 text-sm mt-0.5">arrow_right</span>
                                <p class="text-xs text-slate-700">${r}</p>
                            </div>
                        `).join('') : '<p class="text-xs text-slate-500 italic">Không phát hiện dấu hiệu bất thường.</p>'}
                    </div>
                </div>
            </div>
        `;
    }

}


// ═══════════════════════════════════════════
// NLP Batch Dashboard with ECharts
// ═══════════════════════════════════════════
// [REMOVED old renderNLPBatchDashboard]


// ═══════════════════════════════════════════
// Forensic Argumentation for Single Query
// ═══════════════════════════════════════════
function renderForensicArgumentation(data) {
    const panel = document.getElementById('forensic-argumentation-panel');
    if (!panel) return;
    panel.style.display = 'block';
    
    const score = data.risk_score || 0;
    const industry = data.industry || 'Không rõ';
    const redFlags = data.red_flags || [];
    const revenue = data.revenue || 0;
    const expenses = data.total_expenses || 0;
    const costRatio = revenue > 0 ? (expenses / revenue * 100).toFixed(1) : 0;
    const industryAvgMargin = data.peer_comparison?.industry_avg_margin || 9.0;
    const margin = revenue > 0 ? ((revenue - expenses) / revenue * 100).toFixed(1) : 0;
    
    // Generate argumentation points
    const args = [];
    
    if (score >= 60) {
        args.push({
            icon: 'trending_up', color: 'red',
            title: `Điểm rủi ro ${score.toFixed(1)} — Thuộc top ${score >= 80 ? '5%' : '15%'} doanh nghiệp nguy hiểm nhất`,
            desc: `Doanh nghiệp ngành ${industry} thường chỉ đạt điểm rủi ro trung bình 15-25, nhưng DN này đạt ${score.toFixed(1)}, cao gấp ${(score/20).toFixed(1)} lần.`
        });
    }
    
    if (parseFloat(costRatio) > 90) {
        args.push({
            icon: 'account_balance_wallet', color: 'orange',
            title: `Tỷ lệ chi phí/doanh thu đạt ${costRatio}% — Bất thường nghiêm trọng`,
            desc: `Trung bình ngành ${industry} có tỷ lệ chi phí/doanh thu ~${100 - industryAvgMargin}%. DN này vượt ngưỡng ${(parseFloat(costRatio) - (100 - industryAvgMargin)).toFixed(0)} điểm phần trăm — dấu hiệu khai khống chi phí đầu vào.`
        });
    }
    
    if (parseFloat(margin) < industryAvgMargin) {
        args.push({
            icon: 'insights', color: 'amber',
            title: `Biên lợi nhuận ${margin}% — Thấp hơn trung bình ngành (${industryAvgMargin}%)`,
            desc: `Khi biên lợi nhuận thấp bất thường so với cùng ngành, khả năng DN đang kê khai chi phí ảo để giảm thu nhập chịu thuế.`
        });
    }
    
    redFlags.forEach(f => {
        if (f.severity === 'high' || f.severity === 'critical') {
            args.push({
                icon: 'flag', color: 'red',
                title: f.title || f.description || 'Cảnh báo nghiêm trọng',
                desc: f.recommendation || f.description || ''
            });
        }
    });
    
    // Generate violation predictions
    const violations = [];
    if (parseFloat(costRatio) > 100) violations.push({ label: 'Khai khống chi phí đầu vào', confidence: 85, icon: 'receipt_long' });
    if (redFlags.some(f => (f.title || '').toLowerCase().includes('chi phí'))) violations.push({ label: 'Sử dụng hóa đơn khống', confidence: 72, icon: 'description' });
    if (redFlags.some(f => (f.title || '').toLowerCase().includes('doanh thu'))) violations.push({ label: 'Giấu doanh thu thực tế', confidence: 68, icon: 'visibility_off' });
    if (parseFloat(margin) < 0) violations.push({ label: 'Chuyển giá qua giao dịch liên kết', confidence: 55, icon: 'swap_horiz' });
    if (violations.length === 0 && score >= 40) violations.push({ label: 'Kê khai không đúng thực tế', confidence: 60, icon: 'edit_note' });
    
    panel.innerHTML = `
        <div class="flex items-center justify-between mb-5">
            <h4 class="text-sm font-black uppercase tracking-widest text-primary-container flex items-center gap-2">
                <span class="material-symbols-outlined text-[16px]">psychology</span>
                Biện chứng Lập luận AI
            </h4>
            <span class="px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider bg-indigo-100 text-indigo-700">${args.length} luận điểm</span>
        </div>
        
        <div class="space-y-3 mb-6">
            ${args.map(a => `
                <div class="flex gap-3 p-4 bg-${a.color}-50/50 rounded-xl border border-${a.color}-100/50 hover:shadow-sm transition-shadow">
                    <div class="w-8 h-8 rounded-lg bg-${a.color}-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span class="material-symbols-outlined text-${a.color}-600 text-sm">${a.icon}</span>
                    </div>
                    <div>
                        <p class="text-xs font-bold text-slate-800 mb-1">${a.title}</p>
                        <p class="text-[11px] text-slate-600 leading-relaxed">${a.desc}</p>
                    </div>
                </div>
            `).join('')}
        </div>
        
        <div class="border-t border-slate-100 pt-5">
            <h5 class="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-2">
                <span class="material-symbols-outlined text-[14px]">gavel</span>
                Phỏng đoán Hành vi Vi phạm
            </h5>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                ${violations.map(v => `
                    <div class="flex items-center gap-3 p-3 bg-white rounded-xl border border-slate-100 shadow-sm">
                        <div class="w-10 h-10 rounded-xl bg-red-50 flex items-center justify-center flex-shrink-0">
                            <span class="material-symbols-outlined text-red-500">${v.icon}</span>
                        </div>
                        <div class="flex-1 min-w-0">
                            <p class="text-xs font-bold text-slate-800 truncate">${v.label}</p>
                            <div class="flex items-center gap-2 mt-1">
                                <div class="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                    <div class="h-full bg-gradient-to-r from-red-400 to-red-600 rounded-full" style="width:${v.confidence}%"></div>
                                </div>
                                <span class="text-[9px] font-bold text-red-600">${v.confidence}%</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// ═══════════════════════════════════════════
// NLP SUB-TAB SYSTEM (NEW)
// ═══════════════════════════════════════════
window.__nlpBatchData = null;
window.__nlpTablePage = 1;
window.__nlpTablePageSize = 20;

function switchNLPSubTab(tab) {
    const batchPanel = document.getElementById('nlp-subtab-batch');
    const singlePanel = document.getElementById('nlp-subtab-single');
    const btnBatch = document.getElementById('nlp-subtab-btn-batch');
    const btnSingle = document.getElementById('nlp-subtab-btn-single');
    if (!batchPanel || !singlePanel) return;
    
    const activeClass = 'px-5 py-2 text-[11px] font-black rounded-lg bg-white text-blue-700 shadow-sm transition-all uppercase tracking-widest flex items-center gap-2';
    const inactiveClass = 'px-5 py-2 text-[11px] font-black rounded-lg text-slate-500 hover:text-slate-700 transition-all uppercase tracking-widest flex items-center gap-2';
    
    if (tab === 'batch') {
        batchPanel.classList.remove('hidden');
        singlePanel.classList.add('hidden');
        btnBatch.className = activeClass;
        btnSingle.className = inactiveClass;
    } else {
        batchPanel.classList.add('hidden');
        singlePanel.classList.remove('hidden');
        btnBatch.className = inactiveClass;
        btnSingle.className = activeClass;
    }
}

// ═══════════════════════════════════════════
// NLP BATCH: File Upload Handler
// ═══════════════════════════════════════════
function initNLPBatchUpload() {
    const fileInput = document.getElementById('nlp-file-input');
    if (!fileInput) return;
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const uploadZone = document.getElementById('nlp-batch-upload-zone');
        const progress = document.getElementById('nlp-batch-progress');
        const dashboard = document.getElementById('nlp-batch-dashboard');
        
        uploadZone.classList.add('hidden');
        progress.classList.remove('hidden');
        dashboard.classList.add('hidden');
        
        const progressText = document.getElementById('nlp-batch-progress-text');
        const progressBar = document.getElementById('nlp-batch-progress-bar');
        progressText.textContent = 'Đang tải file lên server...';
        progressBar.style.width = '30%';
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            progressText.textContent = 'Đang phân tích NLP toàn bộ dataset...';
            progressBar.style.width = '60%';
            
            const res = await fetch(`${API_BASE}/ml/redflag/batch_analyze`, {
                method: 'POST', body: formData
            });
            const data = await res.json();
            
            if (data.error) throw new Error(data.error);
            
            progressBar.style.width = '100%';
            progressText.textContent = `Hoàn tất! Đã quét ${data.total_analyzed.toLocaleString()} dòng`;
            
            window.__nlpBatchData = data;
            
            setTimeout(() => {
                progress.classList.add('hidden');
                dashboard.classList.remove('hidden');
                renderNLPBatchFullDashboard(data);
            }, 800);
            
        } catch (err) {
            progress.classList.add('hidden');
            uploadZone.classList.remove('hidden');
            showToast(err.message, 'error');
        }
    });
}

// ═══════════════════════════════════════════
// NLP BATCH: Full Dashboard with 8 Charts
// ═══════════════════════════════════════════
function renderNLPBatchFullDashboard(data) {
    // KPI Cards
    const kpiContainer = document.getElementById('nlp-kpi-cards');
    if (kpiContainer) {
        const flagPct = data.total_analyzed ? Math.round(data.total_flagged / data.total_analyzed * 100) : 0;
        const topInd = (data.summary?.by_industry || [])[0]?.industry || 'N/A';
        kpiContainer.innerHTML = `
            <div class="bg-surface-container-lowest p-5 rounded-2xl border border-outline-variant/10">
                <p class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Tổng Hóa đơn</p>
                <p class="text-3xl font-black text-slate-600">${data.total_analyzed.toLocaleString()}</p>
                <p class="text-[9px] text-slate-300 mt-1">Dòng CSV đã quét</p>
            </div>
            <div class="bg-red-50 p-5 rounded-2xl border border-red-100">
                <p class="text-[10px] font-black text-red-400 uppercase tracking-widest mb-2">Rủi ro phát hiện</p>
                <p class="text-3xl font-black text-red-600">${data.total_flagged.toLocaleString()}</p>
                <p class="text-[9px] text-red-300 mt-1">Hóa đơn bị gắn cờ</p>
            </div>
            <div class="bg-amber-50 p-5 rounded-2xl border border-amber-100">
                <p class="text-[10px] font-black text-amber-400 uppercase tracking-widest mb-2">Tỷ lệ rủi ro</p>
                <p class="text-3xl font-black text-amber-600">${flagPct}%</p>
                <p class="text-[9px] text-amber-300 mt-1">Trên tổng dataset</p>
            </div>
            <div class="bg-surface-container-lowest p-5 rounded-2xl border border-outline-variant/10">
                <p class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Điểm TB</p>
                <p class="text-3xl font-black text-primary-container">${(data.avg_risk_score * 100).toFixed(1)}</p>
                <p class="text-[9px] text-slate-300 mt-1">Trung bình toàn lô</p>
            </div>
            <div class="bg-indigo-50 p-5 rounded-2xl border border-indigo-100">
                <p class="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-2">Công ty</p>
                <p class="text-3xl font-black text-indigo-600">${(data.by_tax_code || []).length.toLocaleString()}</p>
                <p class="text-[9px] text-indigo-300 mt-1">MST duy nhất</p>
            </div>
        `;
    }

    // All charts
    setTimeout(() => { renderNLPBatchCharts(data); }, 200);
    
    // Company table
    renderNLPCompanyTable(data.by_tax_code || [], 1);
    initNLPTableControls(data.by_tax_code || []);
}

function renderNLPBatchCharts(data) {
    const levels = data.summary?.by_risk_level || {};
    const industries = data.summary?.by_industry || [];
    const scoreDist = data.summary?.score_distribution || [];
    const topKw = data.summary?.top_keywords || [];
    const heatmap = data.summary?.heatmap || {};
    const fraudByInd = data.summary?.fraud_type_by_industry || [];
    const companies = data.by_tax_code || [];

    // 1. Donut Chart
    const donutEl = document.getElementById('nlp-chart-donut');
    if (donutEl && typeof echarts !== 'undefined') {
        const c = echarts.init(donutEl);
        c.setOption({
            tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
            legend: { bottom: 0, textStyle: { fontSize: 10 } },
            color: ['#10b981', '#eab308', '#f97316', '#ef4444'],
            series: [{ type: 'pie', radius: ['40%', '70%'], center: ['50%', '45%'],
                label: { show: true, fontSize: 10, formatter: '{b}\n{d}%' },
                emphasis: { label: { fontSize: 14, fontWeight: 'bold' } },
                data: [
                    { value: levels.low || 0, name: 'An toàn' },
                    { value: levels.medium || 0, name: 'Trung bình' },
                    { value: levels.high || 0, name: 'Cao' },
                    { value: levels.critical || 0, name: 'Nguy hiểm' },
                ]
            }]
        });
    }

    // 2. Industry Bar
    const barEl = document.getElementById('nlp-chart-industry-bar');
    if (barEl && typeof echarts !== 'undefined') {
        const c = echarts.init(barEl);
        const top = industries.slice(0, 10);
        c.setOption({
            tooltip: { trigger: 'axis' },
            grid: { left: '3%', right: '10%', bottom: '3%', top: '5%', containLabel: true },
            xAxis: { type: 'value', axisLabel: { fontSize: 9, formatter: '{value}%' } },
            yAxis: { type: 'category', data: top.map(i => i.industry.substring(0,15)), axisLabel: { fontSize: 9 }, inverse: true },
            series: [{ type: 'bar', barWidth: 16, itemStyle: { borderRadius: [0,6,6,0] },
                data: top.map(i => ({ value: i.flag_rate, itemStyle: { color: i.flag_rate > 30 ? '#ef4444' : i.flag_rate > 15 ? '#f97316' : '#3b82f6' } })),
                label: { show: true, position: 'right', fontSize: 9, formatter: '{c}%' }
            }]
        });
    }

    // 3. Histogram
    const histEl = document.getElementById('nlp-chart-histogram');
    if (histEl && typeof echarts !== 'undefined') {
        const c = echarts.init(histEl);
        c.setOption({
            tooltip: { trigger: 'axis' },
            grid: { left: '3%', right: '5%', bottom: '3%', top: '8%', containLabel: true },
            xAxis: { type: 'category', data: scoreDist.map(d => d.range), axisLabel: { fontSize: 9 } },
            yAxis: { type: 'value', axisLabel: { fontSize: 9 } },
            series: [{ type: 'bar', barWidth: '60%',
                data: scoreDist.map((d,i) => ({ value: d.count, itemStyle: { color: ['#10b981','#eab308','#f97316','#ef4444','#dc2626'][i], borderRadius: [4,4,0,0] } })),
                label: { show: true, position: 'top', fontSize: 9 }
            }]
        });
    }

    // 4. Scatter Bubble (companies)
    const scatterEl = document.getElementById('nlp-chart-scatter');
    if (scatterEl && typeof echarts !== 'undefined') {
        const c = echarts.init(scatterEl);
        const scData = companies.slice(0, 200).map(co => [
            Math.round(co.avg_score * 100), co.invoices, co.flags.length + 1, co.tax_code, co.industry
        ]);
        c.setOption({
            tooltip: { formatter: p => `MST: ${p.data[3]}<br>Ngành: ${p.data[4]}<br>Điểm: ${p.data[0]}<br>HĐ: ${p.data[1]}<br>Flags: ${p.data[2]-1}` },
            grid: { left: '5%', right: '5%', bottom: '8%', top: '5%', containLabel: true },
            xAxis: { name: 'Điểm rủi ro', nameTextStyle: { fontSize: 9 }, axisLabel: { fontSize: 9 } },
            yAxis: { name: 'Số hóa đơn', nameTextStyle: { fontSize: 9 }, axisLabel: { fontSize: 9 } },
            series: [{ type: 'scatter', symbolSize: d => Math.max(8, d[2] * 6),
                data: scData,
                itemStyle: { color: p => p.data[0] >= 60 ? '#ef4444' : p.data[0] >= 30 ? '#f97316' : '#3b82f6', opacity: 0.7 }
            }]
        });
    }

    // 5. Treemap
    const treemapEl = document.getElementById('nlp-chart-treemap');
    if (treemapEl && typeof echarts !== 'undefined') {
        const c = echarts.init(treemapEl);
        c.setOption({
            tooltip: { formatter: p => `${p.name}<br>Hóa đơn: ${p.value}<br>Rủi ro: ${(p.data.avgScore*100).toFixed(0)}%` },
            series: [{ type: 'treemap', roam: false, width: '95%', height: '90%',
                label: { show: true, fontSize: 10, formatter: '{b}\n{c}' },
                data: industries.map(ind => ({
                    name: ind.industry, value: ind.total, avgScore: ind.avg_score,
                    itemStyle: { color: ind.avg_score >= 0.3 ? '#fecaca' : ind.avg_score >= 0.15 ? '#fef3c7' : '#d1fae5' }
                }))
            }]
        });
    }

    // 6. Stacked Bar
    const stackedEl = document.getElementById('nlp-chart-stacked');
    if (stackedEl && typeof echarts !== 'undefined') {
        const c = echarts.init(stackedEl);
        const topFraud = fraudByInd.sort((a,b) => (b.keyword_match+b.industry_mismatch) - (a.keyword_match+a.industry_mismatch)).slice(0,8);
        c.setOption({
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            legend: { bottom: 0, textStyle: { fontSize: 9 } },
            grid: { left: '3%', right: '5%', bottom: '12%', top: '5%', containLabel: true },
            yAxis: { type: 'category', data: topFraud.map(f => f.industry.substring(0,12)), axisLabel: { fontSize: 9 }, inverse: true },
            xAxis: { type: 'value', axisLabel: { fontSize: 9 } },
            series: [
                { name: 'Keyword Match', type: 'bar', stack: 'total', data: topFraud.map(f => f.keyword_match), itemStyle: { color: '#f97316' } },
                { name: 'Industry Mismatch', type: 'bar', stack: 'total', data: topFraud.map(f => f.industry_mismatch), itemStyle: { color: '#ef4444' } },
            ]
        });
    }

    // 7. Heatmap
    const heatEl = document.getElementById('nlp-chart-heatmap');
    if (heatEl && typeof echarts !== 'undefined' && heatmap.data) {
        const c = echarts.init(heatEl);
        const maxVal = Math.max(...heatmap.data.map(d => d[2]), 1);
        c.setOption({
            tooltip: { formatter: p => `${heatmap.industries[p.data[0]]}<br>${['An toàn','TB','Cao','Nguy hiểm'][p.data[1]]}: ${p.data[2]} hóa đơn` },
            grid: { left: '15%', right: '12%', bottom: '5%', top: '5%' },
            xAxis: { type: 'category', data: ['An toàn', 'TB', 'Cao', 'Nguy hiểm'], axisLabel: { fontSize: 9 }, splitArea: { show: true } },
            yAxis: { type: 'category', data: (heatmap.industries || []).map(i => i.substring(0,12)), axisLabel: { fontSize: 9 }, splitArea: { show: true } },
            visualMap: { min: 0, max: maxVal, calculable: true, orient: 'vertical', right: 0, top: '10%', itemHeight: 200, textStyle: { fontSize: 8 },
                inRange: { color: ['#f0fdf4', '#86efac', '#fbbf24', '#f97316', '#dc2626'] } },
            series: [{ type: 'heatmap', data: heatmap.data, label: { show: true, fontSize: 9 }, emphasis: { itemStyle: { shadowBlur: 10 } } }]
        });
    }

    // 8. Top Keywords (horizontal bar)
    const kwEl = document.getElementById('nlp-chart-keywords');
    if (kwEl && typeof echarts !== 'undefined') {
        const c = echarts.init(kwEl);
        c.setOption({
            tooltip: { trigger: 'axis' },
            grid: { left: '3%', right: '8%', bottom: '3%', top: '5%', containLabel: true },
            xAxis: { type: 'value', axisLabel: { fontSize: 9 } },
            yAxis: { type: 'category', data: topKw.map(k => k.keyword), axisLabel: { fontSize: 9 }, inverse: true },
            series: [{ type: 'bar', barWidth: 14, data: topKw.map(k => ({ value: k.count, itemStyle: { color: '#6366f1', borderRadius: [0,4,4,0] } })),
                label: { show: true, position: 'right', fontSize: 9 }
            }]
        });
    }
}

// ═══════════════════════════════════════════
// NLP BATCH: Company Table (CSDL style)
// ═══════════════════════════════════════════
function renderNLPCompanyTable(companies, page) {
    const tbody = document.getElementById('nlp-table-body');
    const info = document.getElementById('nlp-table-pagination-info');
    const controls = document.getElementById('nlp-table-pagination-controls');
    if (!tbody) return;
    
    const size = window.__nlpTablePageSize;
    const start = (page - 1) * size;
    const end = Math.min(start + size, companies.length);
    const pageData = companies.slice(start, end);
    const totalPages = Math.ceil(companies.length / size);
    
    const lvlColors = { critical: 'bg-red-100 text-red-700', high: 'bg-orange-100 text-orange-700', medium: 'bg-amber-100 text-amber-700', low: 'bg-emerald-100 text-emerald-700' };
    const lvlLabels = { critical: 'Nguy hiểm', high: 'Cao', medium: 'Trung bình', low: 'An toàn' };
    
    tbody.innerHTML = pageData.map((co, i) => `
        <tr class="hover:bg-blue-50/50 transition-colors">
            <td class="px-4 py-3 text-slate-400">${start + i + 1}</td>
            <td class="px-4 py-3 font-mono font-bold text-slate-700">${co.tax_code}</td>
            <td class="px-4 py-3 text-slate-600">${co.industry}</td>
            <td class="px-4 py-3 text-slate-600">${co.invoices}</td>
            <td class="px-4 py-3 font-black text-slate-800">${Math.round(co.avg_score * 100)}</td>
            <td class="px-4 py-3"><span class="px-2 py-0.5 rounded-full text-[9px] font-bold ${lvlColors[co.risk_level] || ''}">${lvlLabels[co.risk_level] || ''}</span></td>
            <td class="px-4 py-3 text-slate-500">${co.flags.join(', ') || '-'}</td>
            <td class="px-4 py-3 text-right">
                <button class="nlp-detail-btn text-[10px] font-bold text-blue-600 hover:text-blue-800 hover:underline" data-mst="${co.tax_code}" data-industry="${co.industry}" data-descs="${encodeURIComponent(JSON.stringify(co.descriptions))}">
                    🔍 Phân tích
                </button>
            </td>
        </tr>
    `).join('');
    
    if (info) info.textContent = `Hiển thị ${start+1} - ${end} / ${companies.length} công ty`;
    
    // Pagination buttons
    if (controls) {
        let btns = '';
        if (page > 1) btns += `<button onclick="renderNLPCompanyTable(window.__nlpBatchData.by_tax_code, ${page-1})" class="px-3 py-1 rounded text-xs bg-slate-100 hover:bg-slate-200">‹</button>`;
        for (let p = Math.max(1, page-2); p <= Math.min(totalPages, page+2); p++) {
            btns += `<button onclick="renderNLPCompanyTable(window.__nlpBatchData.by_tax_code, ${p})" class="px-3 py-1 rounded text-xs ${p===page ? 'bg-primary-container text-white' : 'bg-slate-100 hover:bg-slate-200'}">${p}</button>`;
        }
        if (page < totalPages) btns += `<button onclick="renderNLPCompanyTable(window.__nlpBatchData.by_tax_code, ${page+1})" class="px-3 py-1 rounded text-xs bg-slate-100 hover:bg-slate-200">›</button>`;
        controls.innerHTML = btns;
    }
    
    // Wire detail buttons
    document.querySelectorAll('.nlp-detail-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mst = btn.dataset.mst;
            const industry = btn.dataset.industry;
            const descs = JSON.parse(decodeURIComponent(btn.dataset.descs));
            
            switchNLPSubTab('single');
            document.getElementById('nlp-single-from-batch').classList.remove('hidden');
            document.getElementById('nlp-single-from-mst').textContent = mst;
            document.getElementById('nlp-mst-input').value = mst;
            document.getElementById('nlp-industry-select').value = industry;
            document.getElementById('nlp-desc-input').value = descs.join('\n');
            
            // Auto-trigger analysis
            setTimeout(() => { document.getElementById('btn-scan-nlp').click(); }, 300);
        });
    });
}

function initNLPTableControls(companies) {
    const searchInput = document.getElementById('nlp-table-search');
    const levelFilter = document.getElementById('nlp-table-level-filter');
    const summary = document.getElementById('nlp-table-summary');
    
    if (summary) summary.textContent = `Tổng cộng ${companies.length} công ty từ dataset`;
    
    const doFilter = () => {
        const q = (searchInput?.value || '').toLowerCase();
        const lvl = levelFilter?.value || '';
        let filtered = companies;
        if (q) filtered = filtered.filter(c => c.tax_code.includes(q) || c.industry.toLowerCase().includes(q));
        if (lvl) filtered = filtered.filter(c => c.risk_level === lvl);
        renderNLPCompanyTable(filtered, 1);
    };
    
    if (searchInput) searchInput.addEventListener('input', doFilter);
    if (levelFilter) levelFilter.addEventListener('change', doFilter);
}

// ═══════════════════════════════════════════
// NLP SINGLE: Dashboard with 4 Charts
// ═══════════════════════════════════════════
function renderNLPSingleDashboard(data) {
    const dashboard = document.getElementById('nlp-single-dashboard');
    if (!dashboard) return;
    dashboard.classList.remove('hidden');
    
    const score = Math.round((data.risk_score || 0) * 100);
    const flags = data.flags || [];
    const conf = Math.round((data.confidence || 0) * 100);
    
    // Parse items from description
    const descInput = document.getElementById('nlp-desc-input');
    const items = (descInput?.value || '').split('\n').filter(l => l.trim());
    const numItems = items.length || 1;
    const riskyItems = flags.length;
    const safeItems = Math.max(0, numItems - riskyItems);
    
    setTimeout(() => {
        // 1. Gauge Chart
        const gaugeEl = document.getElementById('nlp-single-gauge');
        if (gaugeEl && typeof echarts !== 'undefined') {
            const c = echarts.init(gaugeEl);
            c.setOption({
                series: [{
                    type: 'gauge', startAngle: 220, endAngle: -40,
                    min: 0, max: 100,
                    pointer: { show: true, length: '60%', width: 4 },
                    progress: { show: true, width: 12, roundCap: true,
                        itemStyle: { color: score >= 60 ? '#ef4444' : score >= 30 ? '#f97316' : '#10b981' }
                    },
                    axisLine: { lineStyle: { width: 12, color: [[0.3, '#10b981'], [0.6, '#f97316'], [1, '#ef4444']] } },
                    axisTick: { show: false },
                    splitLine: { show: false },
                    axisLabel: { fontSize: 9, distance: 18 },
                    title: { show: true, offsetCenter: [0, '70%'], fontSize: 11, fontWeight: 'bold',
                        color: score >= 60 ? '#ef4444' : score >= 30 ? '#f97316' : '#10b981' },
                    detail: { fontSize: 28, fontWeight: 900, offsetCenter: [0, '40%'], 
                        color: score >= 60 ? '#ef4444' : score >= 30 ? '#f97316' : '#10b981',
                        formatter: '{value}' },
                    data: [{ value: score, name: score >= 60 ? 'NGUY HIỂM' : score >= 30 ? 'CẢNH BÁO' : 'AN TOÀN' }]
                }]
            });
        }

        // 2. Radar Chart (5 axes)
        const radarEl = document.getElementById('nlp-single-radar');
        if (radarEl && typeof echarts !== 'undefined') {
            const c = echarts.init(radarEl);
            const hasMismatch = flags.some(f => f.type === 'industry_mismatch');
            c.setOption({
                radar: {
                    indicator: [
                        { name: 'Keyword', max: 100 },
                        { name: 'Semantic', max: 100 },
                        { name: 'Industry', max: 100 },
                        { name: 'Pattern', max: 100 },
                        { name: 'Context', max: 100 },
                    ],
                    radius: '65%', axisName: { fontSize: 9 }, splitNumber: 3,
                },
                series: [{ type: 'radar',
                    data: [{ value: [
                        Math.min(100, flags.filter(f=>f.type==='keyword_match').length * 30 + 10),
                        conf,
                        hasMismatch ? 85 : 10,
                        Math.min(100, flags.length * 25),
                        score
                    ],
                    areaStyle: { opacity: 0.25, color: score >= 60 ? '#ef4444' : '#3b82f6' },
                    lineStyle: { color: score >= 60 ? '#ef4444' : '#3b82f6', width: 2 },
                    itemStyle: { color: score >= 60 ? '#ef4444' : '#3b82f6' }
                    }]
                }]
            });
        }

        // 3. Pie (safe vs risky)
        const pieEl = document.getElementById('nlp-single-pie');
        if (pieEl && typeof echarts !== 'undefined') {
            const c = echarts.init(pieEl);
            c.setOption({
                tooltip: { trigger: 'item' },
                color: ['#10b981', '#ef4444'],
                series: [{ type: 'pie', radius: ['35%', '65%'],
                    label: { show: true, fontSize: 10, formatter: '{b}\n{d}%' },
                    data: [
                        { value: safeItems, name: 'An toàn' },
                        { value: riskyItems, name: 'Rủi ro' },
                    ]
                }]
            });
        }

        // 4. Per-item Bar
        const itemsBarEl = document.getElementById('nlp-single-items-bar');
        if (itemsBarEl && typeof echarts !== 'undefined') {
            const c = echarts.init(itemsBarEl);
            const kwSet = new Set((flags || []).filter(f => f.keyword).map(f => f.keyword.toLowerCase()));
            const itemScores = items.map(item => {
                const lower = item.toLowerCase();
                let s = 5;
                kwSet.forEach(kw => { if (lower.includes(kw)) s += 30; });
                return Math.min(100, s);
            });
            c.setOption({
                tooltip: { trigger: 'axis' },
                grid: { left: '3%', right: '5%', bottom: '3%', top: '5%', containLabel: true },
                xAxis: { type: 'category', data: items.map(it => it.substring(0,20) + (it.length > 20 ? '...' : '')), axisLabel: { fontSize: 8, rotate: 15 } },
                yAxis: { type: 'value', max: 100, axisLabel: { fontSize: 9 } },
                series: [{ type: 'bar', barWidth: '50%',
                    data: itemScores.map(v => ({
                        value: v,
                        itemStyle: { color: v >= 50 ? '#ef4444' : v >= 20 ? '#f97316' : '#10b981', borderRadius: [4,4,0,0] }
                    })),
                    label: { show: true, position: 'top', fontSize: 9 }
                }]
            });
        }
    }, 200);

    // Render flags list
    const flagsList = document.getElementById('nlp-flags-list');
    if (flagsList) {
        if (!flags.length) {
            flagsList.innerHTML = '<p class="text-xs text-slate-500 italic py-4 text-center">Không phát hiện dấu hiệu bất thường.</p>';
        } else {
            flagsList.innerHTML = flags.map(f => {
                const isMismatch = f.type === 'industry_mismatch';
                const icon = isMismatch ? 'category' : 'warning';
                const title = isMismatch ? (f.description || 'Industry Mismatch') : `Từ khóa nhạy cảm: "${f.keyword || f.type}"`;
                return `<div class="p-3 rounded-lg bg-red-50 border border-red-100/50 flex items-start gap-3">
                    <span class="material-symbols-outlined text-red-500 text-sm mt-0.5">${icon}</span>
                    <div>
                        <p class="text-[11px] font-bold text-red-700 mb-0.5 uppercase tracking-wider">${f.type}</p>
                        <p class="text-xs text-slate-600">${title}</p>
                    </div>
                </div>`;
            }).join('');
        }
    }
    
    // Confidence badge
    const confBadge = document.getElementById('nlp-confidence-badge');
    if (confBadge) {
        confBadge.classList.remove('hidden');
        confBadge.textContent = `Confidence: ${conf}% • Method: ${data.method || 'Ensemble'}`;
    }
}

// ═══════════════════════════════════════════
// NLP: Wire scan button for single mode
// ═══════════════════════════════════════════
function initNLPSingleScan() {
    const btn = document.getElementById('btn-scan-nlp');
    if (!btn) return;
    
    btn.addEventListener('click', async () => {
        const mst = document.getElementById('nlp-mst-input')?.value?.trim();
        const industry = document.getElementById('nlp-industry-select')?.value || '';
        const desc = document.getElementById('nlp-desc-input')?.value?.trim();
        
        if (!desc) {
            showToast('Vui lòng nhập mô tả hàng hóa', 'error');
            return;
        }
        
        btn.disabled = true;
        btn.innerHTML = '<span class="material-symbols-outlined text-[20px] animate-spin">autorenew</span> Đang phân tích...';
        
        try {
            const res = await secureFetch(`${API_BASE}/ml/redflag/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tax_code: mst || '0000000000',
                    description: desc,
                    industry: industry
                })
            });
            
            if (res.error) throw new Error(res.error);
            
            setTimeout(() => {
                btn.disabled = false;
                btn.innerHTML = '<span class="material-symbols-outlined text-[20px] group-hover:rotate-180 transition-transform duration-700">radar</span> Chạy Phân tích NLP';
                renderNLPSingleDashboard(res);
            }, 800);
            
        } catch (e) {
            showToast(e.message, 'error');
            btn.disabled = false;
            btn.innerHTML = '<span class="material-symbols-outlined text-[20px] group-hover:rotate-180 transition-transform duration-700">radar</span> Chạy Phân tích NLP';
        }
    });
    
    // "Tra cứu sâu" button → go to main single query tab
    const gotoBtn = document.getElementById('nlp-goto-single-main');
    if (gotoBtn) {
        gotoBtn.addEventListener('click', () => {
            const mst = document.getElementById('nlp-mst-input')?.value?.trim();
            if (typeof switchTab === 'function') switchTab('single');
            setTimeout(() => {
                const mainMst = document.getElementById('fraud-mst');
                if (mainMst && mst) { mainMst.value = mst; }
                if (typeof checkFraudRisk === 'function') checkFraudRisk();
            }, 300);
        });
    }
    
    // Paste sample button
    const pasteBtn = document.getElementById('nlp-paste-sample');
    if (pasteBtn) {
        pasteBtn.addEventListener('click', () => {
            document.getElementById('nlp-desc-input').value = 'Chi phí tư vấn quản lý rủi ro\nThiết kế website marketing\nHoa hồng môi giới bất động sản\nSắt thép Pomina\nXi măng Hà Tiên';
            document.getElementById('nlp-mst-input').value = '0312345678';
            document.getElementById('nlp-industry-select').value = 'Xây dựng';
        });
    }
}

// Initialize NLP system on page load
document.addEventListener('DOMContentLoaded', () => {
    initNLPBatchUpload();
    initNLPSingleScan();
});
