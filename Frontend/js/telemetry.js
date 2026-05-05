const API_BASE = window.API_BASE_URL || 'http://localhost:8000';

document.addEventListener('DOMContentLoaded', () => {
    const charts = {};
    const windowSelect = document.getElementById('timeWindow');
    const refreshBtn = document.getElementById('refreshBtn');

    Chart.defaults.font.family = "'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
    Chart.defaults.color = '#64748b';

    initCharts();
    fetchData();
    refreshBtn.addEventListener('click', fetchData);
    windowSelect.addEventListener('change', fetchData);
    setInterval(fetchData, 30000);

    async function fetchData() {
        refreshBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
        refreshBtn.disabled = true;
        const win = Number(windowSelect.value || 60);
        const driftWin = win === 0 ? 168 : (win > 60 ? 48 : 24);

        try {
            const [telemetryRes, driftRes, feedbackRes, activeRes] = await Promise.all([
                fetch(`${API_BASE}/api/tax-agent/telemetry/dashboard?window_minutes=${win}`),
                fetch(`${API_BASE}/api/tax-agent/telemetry/drift?window_hours=${driftWin}`),
                fetch(`${API_BASE}/api/tax-agent/feedback/stats?window_hours=${driftWin}`),
                fetch(`${API_BASE}/api/tax-agent/feedback/active-learning?limit=10`)
            ]);

            const telemetry = await telemetryRes.json();
            const drift = await driftRes.json();
            const feedback = await feedbackRes.json();
            const active = await activeRes.json();

            updateMetrics(telemetry, feedback);
            updateAlerts(drift, telemetry);
            updateCharts(telemetry);
            updateActiveLearning(active);
            updateDiagnostics(telemetry);
            updateDpoDebate(telemetry);
        } catch (e) {
            console.error('Failed to fetch telemetry data', e);
        } finally {
            refreshBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Refresh';
            refreshBtn.disabled = false;
        }
    }

    // Bind DPO Buttons
    document.getElementById('btn-dpo-dryrun')?.addEventListener('click', async () => {
        const btn = document.getElementById('btn-dpo-dryrun');
        const oldHtml = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running...';
        btn.disabled = true;
        try {
            await fetch(`${API_BASE}/api/tax-agent/dpo/dry-run`, { method: 'POST' });
            setTimeout(fetchData, 1000);
        } finally {
            btn.innerHTML = oldHtml;
            btn.disabled = false;
        }
    });

    document.getElementById('btn-dpo-train')?.addEventListener('click', async () => {
        const btn = document.getElementById('btn-dpo-train');
        
        // Simple client-side token prompt for demo (In real app, this comes from auth context)
        const token = prompt("Vui lòng nhập Admin Token (RBAC) để thực thi huấn luyện DPO:", "");
        if (!token) return;

        const oldHtml = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
        btn.disabled = true;
        try {
            const res = await fetch(`${API_BASE}/api/tax-agent/dpo/trigger?dry_run=false`, { 
                method: 'POST',
                headers: {
                    'X-Admin-Token': token
                }
            });
            if (res.status === 403) {
                alert("Lỗi: Token không hợp lệ. Bạn không có quyền (Senior AI Admin) để thực thi.");
            } else if (!res.ok) {
                alert("Lỗi server khi trigger DPO.");
            }
            setTimeout(fetchData, 1000);
        } finally {
            btn.innerHTML = oldHtml;
            btn.disabled = false;
        }
    });

    function initCharts() {
        charts.timeline = makeLine('timelineChart', 'Requests', '#0ea5e9', true);
        charts.intent = makeDoughnut('intentChart');
        charts.tool = new Chart(ctx('toolChart'), {
            type: 'bar',
            data: { labels: [], datasets: [
                { label: 'Success', data: [], backgroundColor: '#10b981', borderRadius: 4 },
                { label: 'Failed', data: [], backgroundColor: '#ef4444', borderRadius: 4 }
            ] },
            options: {
                responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                scales: { x: { beginAtZero: true, stacked: true }, y: { stacked: true, grid: { display: false } } },
                plugins: { legend: { position: 'bottom' } }
            }
        });
        charts.latency = new Chart(ctx('latencyChart'), {
            type: 'line',
            data: { labels: [], datasets: [
                { label: 'Avg', data: [], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,.08)', tension: .35 },
                { label: 'P95', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,.08)', tension: .35 }
            ] },
            options: chartLineOptions(true)
        });
        charts.confidence = new Chart(ctx('confidenceChart'), {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Turns', data: [], backgroundColor: '#14b8a6', borderRadius: 4 }] },
            options: chartBarOptions()
        });
        charts.contract = makeDoughnut('contractChart');
        charts.focus = new Chart(ctx('focusChart'), {
            type: 'bar',
            data: { labels: ['OK', 'Violation'], datasets: [{ data: [0, 0], backgroundColor: ['#10b981', '#ef4444'], borderRadius: 6 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
        });
    }

    function updateMetrics(telemetry, feedback) {
        setText('metric-total', telemetry.total_requests || 0);
        setText('metric-rpm', Number(telemetry.requests_per_minute || 0).toFixed(1));
        const latency = telemetry.latency?.avg_ms || 0;
        setText('metric-latency', latency > 0 ? `${Number(latency).toFixed(0)}ms` : '-');
        const sat = feedback.satisfaction_rate;
        setText('metric-satisfaction', sat !== null && sat !== undefined ? `${(sat * 100).toFixed(1)}%` : 'N/A');
        const satEl = document.getElementById('metric-satisfaction');
        if (sat >= 0.8) satEl.className = 'text-3xl font-bold text-emerald-500';
        else if (sat >= 0.5) satEl.className = 'text-3xl font-bold text-amber-500';
        else satEl.className = 'text-3xl font-bold text-red-500';
    }

    function updateDpoDebate(telemetry) {
        // DPO Status
        const dpo = telemetry.dpo_status || {};
        const badge = document.getElementById('dpo-status-badge');
        if (badge) {
            const status = dpo.current_status || 'unknown';
            badge.innerText = status.toUpperCase();
            badge.className = `px-2 py-1 text-xs font-semibold rounded-full ${
                status === 'training' ? 'bg-slate-100 text-slate-800 border border-slate-300 animate-pulse' :
                status === 'building_pairs' ? 'bg-slate-100 text-slate-800 border border-slate-300' :
                status === 'idle' ? 'bg-white text-slate-500 border border-slate-200' :
                'bg-slate-50 text-slate-500 border border-slate-200'
            }`;
        }
        setText('dpo-total-runs', dpo.total_runs || 0);
        const lastRun = dpo.last_run;
        if (lastRun && lastRun.timestamp) {
            const dt = new Date(lastRun.timestamp * 1000);
            setText('dpo-last-run', `${dt.toLocaleTimeString()} ${dt.toLocaleDateString()}`);
        } else {
            setText('dpo-last-run', 'N/A');
        }

        // Debate Metrics
        const debate = telemetry.debate_metrics || {};
        setText('debate-total', debate.total || 0);
        setText('debate-escalations', debate.escalations || 0);
        const adj = debate.adjudicator_triggered || 0;
        setText('debate-adjudicator', adj);
        
        const adjRate = debate.adjudicator_rate || 0;
        setText('adjudicator-rate-text', `${(adjRate * 100).toFixed(1)}%`);
        const adjBar = document.getElementById('adjudicator-rate-bar');
        if (adjBar) adjBar.style.width = `${Math.min(100, adjRate * 100)}%`;
    }

    function updateAlerts(drift, telemetry) {
        const container = document.getElementById('alertsContainer');
        container.innerHTML = '';

        const alerts = [];
        if (telemetry.empty_reason) {
            alerts.push({ level: 'info', metric: 'data_window', message: `Không có dữ liệu cho khung thời gian này: ${telemetry.empty_reason}` });
        }
        if (drift.drift_detected) {
            alerts.push({ level: 'warning', metric: 'intent_drift', message: `Phân phối Intent bị lệch: ${drift.divergence_score}` });
        }
        const focus = telemetry.focus_violations;
        if (focus && focus.count > 0) {
            alerts.push({ level: 'warning', metric: 'focus_guard', message: `${focus.count} lỗi điều hướng/focus (${(focus.rate * 100).toFixed(1)}%).` });
        }

        if (!alerts.length) {
            container.innerHTML = '<div class="text-sm text-slate-500 flex items-center gap-2 bg-slate-50 px-3 py-2 rounded-lg border border-slate-200"><i class="fa-solid fa-check"></i> Hệ thống ổn định. Không phát hiện độ lệch hoặc lỗi vi phạm.</div>';
            return;
        }

        alerts.forEach(alert => {
            const color = alert.level === 'info' ? 'text-slate-600 bg-slate-50 border-slate-200' : 'text-slate-800 bg-white border-slate-200 border-l-2 border-l-slate-400';
            container.innerHTML += `<div class="text-sm ${color} border px-3 py-2 rounded-lg flex items-start gap-2 shadow-sm">
                <i class="fa-solid fa-circle-exclamation mt-0.5 ${alert.level === 'info' ? 'text-slate-400' : 'text-slate-500'}"></i>
                <div><div class="font-bold">${escapeHtml(alert.metric)}</div><div class="text-xs mt-0.5 text-slate-500">${escapeHtml(alert.message)}</div></div>
            </div>`;
        });
    }

    function updateCharts(telemetry) {
        const timeline = telemetry.timeline || [];
        const labels = timeline.map(t => formatTime(t.timestamp));
        updateChart(charts.timeline, labels, [timeline.map(t => t.count || 0)]);
        updateChart(charts.latency, labels, [
            timeline.map(t => t.avg_latency_ms || 0),
            timeline.map(t => t.p95_latency_ms || 0)
        ]);

        const intents = entries(telemetry.intents || telemetry.intent_distribution).slice(0, 8);
        updateChart(charts.intent, intents.map(x => x[0]), [intents.map(x => x[1])]);

        const toolUsage = telemetry.tool_usage || legacyToolsToUsage(telemetry.tools || {});
        const topTools = toolUsage.slice(0, 8);
        charts.tool.data.labels = topTools.map(t => t.tool);
        charts.tool.data.datasets[0].data = topTools.map(t => t.success ?? t.total ?? 0);
        charts.tool.data.datasets[1].data = topTools.map(t => t.failed ?? 0);
        charts.tool.update();

        const hist = telemetry.confidence_histogram || [];
        updateChart(charts.confidence, hist.map(h => h.bucket), [hist.map(h => h.count)]);

        const contracts = entries(telemetry.answer_contract_distribution || {}).slice(0, 8);
        updateChart(charts.contract, contracts.map(x => x[0]), [contracts.map(x => x[1])]);

        const violations = telemetry.focus_violations || { count: 0 };
        const total = telemetry.total_requests || 0;
        charts.focus.data.datasets[0].data = [Math.max(0, total - (violations.count || 0)), violations.count || 0];
        charts.focus.update();
    }

    function updateActiveLearning(activeData) {
        const container = document.getElementById('activeLearningList');
        container.innerHTML = '';
        const candidates = activeData.candidates || [];
        if (!candidates.length) {
            container.innerHTML = '<div class="text-sm text-slate-500 italic">Không có ứng cử viên nào cần xem xét.</div>';
            return;
        }
        candidates.forEach(cand => {
            const isNeg = cand.feedback_type === 'negative';
            const badgeColor = isNeg ? 'bg-slate-100 border border-slate-200 text-slate-700' : 'bg-white border border-slate-200 text-slate-600';
            const label = isNeg ? 'Phản hồi tiêu cực' : 'Độ tin cậy thấp';
            container.innerHTML += `<div class="mb-3 pb-3 border-b border-slate-100 last:border-0 last:pb-0">
                <div class="flex items-center justify-between mb-1">
                    <span class="text-xs font-bold text-slate-700">${escapeHtml(cand.intent || 'N/A')}</span>
                    <span class="text-[10px] px-1.5 py-0.5 rounded font-bold ${badgeColor}">${label}</span>
                </div>
                <div class="text-xs text-slate-500 line-clamp-2">${escapeHtml(cand.message_preview || '...')}</div>
                <div class="text-[10px] text-slate-400 mt-1 flex justify-between">
                    <span>S: ${String(cand.session_id || '').substring(0, 8)} | Turn: ${cand.turn_id || '-'}</span>
                    <span>Conf: ${((cand.confidence || 0) * 100).toFixed(1)}%</span>
                </div>
            </div>`;
        });
    }

    function updateDiagnostics(telemetry) {
        const retrieval = telemetry.retrieval_quality || {};
        const faith = telemetry.legal_faithfulness || {};
        const route = telemetry.route_quality || {};
        const active = telemetry.active_learning_summary || {};
        const summary = document.getElementById('retrievalSummary');
        summary.innerHTML = [
            statCard('Retrieval queries', retrieval.total_queries || 0),
            statCard('Avg retrieval latency', `${retrieval.avg_latency_ms || 0}ms`),
            statCard('Unsupported legal claims', faith.unsupported_claims || 0),
            statCard('Avg focus score', `${((route.avg_focus_score || 1) * 100).toFixed(1)}%`),
            statCard('Low-confidence turns', active.low_confidence_turns || 0),
            statCard('Negative feedback', active.negative_or_correction_feedback || 0),
        ].join('');

        const failures = document.getElementById('failureList');
        const toolUsage = (telemetry.tool_usage || []).filter(t => (t.failed || 0) > 0).slice(0, 6);
        const focus = telemetry.focus_violations || {};
        if (!toolUsage.length && !(focus.count > 0)) {
            failures.innerHTML = '<div class="text-slate-500 italic">No failed tools or route errors in this window.</div>';
            return;
        }
        failures.innerHTML = '';
        if (focus.count > 0) {
            failures.innerHTML += `<div class="p-2 rounded-lg bg-white text-slate-700 border border-slate-200">Focus violations: ${focus.count} (${(focus.rate * 100).toFixed(1)}%)</div>`;
        }
        toolUsage.forEach(tool => {
            failures.innerHTML += `<div class="p-2 rounded-lg bg-slate-50 border border-slate-100 flex justify-between">
                <span>${escapeHtml(tool.tool)}</span><span class="font-bold text-red-600">${tool.failed}</span>
            </div>`;
        });
    }

    function makeLine(id, label, color, fill) {
        return new Chart(ctx(id), {
            type: 'line',
            data: { labels: [], datasets: [{ label, data: [], borderColor: color, backgroundColor: `${color}22`, fill, tension: .35 }] },
            options: chartLineOptions(false)
        });
    }

    function makeDoughnut(id) {
        return new Chart(ctx(id), {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], backgroundColor: ['#0ea5e9', '#3b82f6', '#8b5cf6', '#d946ef', '#f43f5e', '#f59e0b', '#10b981', '#64748b'] }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { boxWidth: 12 } } }, cutout: '68%' }
        });
    }

    function chartLineOptions(showLegend) {
        return {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: showLegend, position: 'bottom' } },
            scales: { y: { beginAtZero: true, grid: { color: '#f1f5f9' } }, x: { grid: { display: false } } }
        };
    }

    function chartBarOptions() {
        return {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, grid: { color: '#f1f5f9' } }, x: { grid: { display: false } } }
        };
    }

    function updateChart(chart, labels, datasetValues) {
        chart.data.labels = labels;
        datasetValues.forEach((values, index) => {
            if (chart.data.datasets[index]) chart.data.datasets[index].data = values;
        });
        chart.update();
    }

    function ctx(id) {
        return document.getElementById(id).getContext('2d');
    }

    function setText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    function entries(obj) {
        return Object.entries(obj || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
    }

    function legacyToolsToUsage(tools) {
        return Object.entries(tools || {})
            .filter(([key]) => !key.startsWith('total_'))
            .map(([tool, total]) => ({ tool, total: Number(total || 0), success: Number(total || 0), failed: 0 }))
            .sort((a, b) => b.total - a.total);
    }

    function formatTime(ts) {
        const d = new Date(Number(ts) * 1000);
        if (Number.isNaN(d.getTime())) return '';
        return `${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
    }

    function statCard(label, value) {
        return `<div class="rounded-xl bg-slate-50 border border-slate-100 p-3">
            <div class="text-xs uppercase tracking-wide text-slate-500 font-bold">${escapeHtml(label)}</div>
            <div class="text-xl font-bold text-slate-800 mt-1">${escapeHtml(String(value))}</div>
        </div>`;
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
