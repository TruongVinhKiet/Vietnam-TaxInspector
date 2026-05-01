document.addEventListener('DOMContentLoaded', () => {
    // Chart instances
    let timelineChart, intentChart, toolChart;

    // Elements
    const windowSelect = document.getElementById('timeWindow');
    const refreshBtn = document.getElementById('refreshBtn');

    // Init
    initCharts();
    fetchData();

    // Event Listeners
    refreshBtn.addEventListener('click', fetchData);
    windowSelect.addEventListener('change', fetchData);

    // Auto refresh every 30s
    setInterval(fetchData, 30000);

    async function fetchData() {
        refreshBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
        refreshBtn.disabled = true;

        const win = windowSelect.value;
        const driftWin = win > 60 ? 48 : 24; // Default drift windows

        try {
            const [telemetryRes, driftRes, feedbackRes, activeRes] = await Promise.all([
                fetch(`http://localhost:8000/api/tax-agent/telemetry/dashboard?window_minutes=${win}`),
                fetch(`http://localhost:8000/api/tax-agent/telemetry/drift?window_hours=${driftWin}`),
                fetch(`http://localhost:8000/api/tax-agent/feedback/stats?window_hours=${driftWin}`),
                fetch('http://localhost:8000/api/tax-agent/feedback/active-learning?limit=10')
            ]);

            const telemetry = await telemetryRes.json();
            const drift = await driftRes.json();
            const feedback = await feedbackRes.json();
            const active = await activeRes.json();

            updateMetrics(telemetry, feedback);
            updateAlerts(drift);
            updateCharts(telemetry);
            updateActiveLearning(active);

        } catch (e) {
            console.error('Failed to fetch telemetry data', e);
        } finally {
            refreshBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Refresh';
            refreshBtn.disabled = false;
        }
    }

    function updateMetrics(telemetry, feedback) {
        document.getElementById('metric-total').textContent = telemetry.total_requests || 0;
        document.getElementById('metric-rpm').textContent = (telemetry.requests_per_minute || 0).toFixed(1);
        
        const latency = telemetry.latency?.avg_ms || 0;
        document.getElementById('metric-latency').textContent = latency > 0 ? `${latency.toFixed(0)}ms` : '-';

        const sat = feedback.satisfaction_rate;
        document.getElementById('metric-satisfaction').textContent = sat !== null ? `${(sat * 100).toFixed(1)}%` : 'N/A';
        
        // Color code satisfaction
        const satEl = document.getElementById('metric-satisfaction');
        if (sat >= 0.8) satEl.className = 'text-3xl font-bold text-emerald-500';
        else if (sat >= 0.5) satEl.className = 'text-3xl font-bold text-amber-500';
        else satEl.className = 'text-3xl font-bold text-red-500';
    }

    function updateAlerts(drift) {
        const container = document.getElementById('alertsContainer');
        container.innerHTML = '';

        if (!drift.drift_detected || !drift.alerts || drift.alerts.length === 0) {
            container.innerHTML = '<div class="text-sm text-emerald-600 flex items-center gap-2 bg-emerald-50 px-3 py-2 rounded-lg border border-emerald-100"><i class="fa-solid fa-circle-check"></i> Hệ thống hoạt động ổn định, không phát hiện Drift.</div>';
            return;
        }

        drift.alerts.forEach(alert => {
            const isCritical = alert.level === 'critical';
            const colorClass = isCritical ? 'text-red-700 bg-red-50 border-red-200' : 'text-amber-700 bg-amber-50 border-amber-200';
            const icon = isCritical ? 'fa-triangle-exclamation text-red-500' : 'fa-circle-exclamation text-amber-500';
            
            container.innerHTML += `
                <div class="text-sm ${colorClass} border px-3 py-2 rounded-lg flex items-start gap-2">
                    <i class="fa-solid ${icon} mt-0.5"></i>
                    <div>
                        <div class="font-bold">${alert.metric}</div>
                        <div class="text-xs mt-0.5">${alert.message}</div>
                    </div>
                </div>
            `;
        });
    }

    function updateActiveLearning(activeData) {
        const container = document.getElementById('activeLearningList');
        container.innerHTML = '';

        if (!activeData.candidates || activeData.candidates.length === 0) {
            container.innerHTML = '<div class="text-sm text-slate-500 italic">Không có ứng viên nào cần review.</div>';
            return;
        }

        activeData.candidates.forEach(cand => {
            const isNeg = cand.feedback_type === 'negative';
            const badgeColor = isNeg ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700';
            const label = isNeg ? 'Feedback Kém' : 'Độ Tin Cậy Thấp';
            
            container.innerHTML += `
                <div class="mb-3 pb-3 border-b border-slate-100 last:border-0 last:pb-0">
                    <div class="flex items-center justify-between mb-1">
                        <span class="text-xs font-bold text-slate-700">${cand.intent || 'N/A'}</span>
                        <span class="text-[10px] px-1.5 py-0.5 rounded font-bold ${badgeColor}">${label}</span>
                    </div>
                    <div class="text-xs text-slate-500 line-clamp-2" title="${cand.message_preview || ''}">
                        "${cand.message_preview || '...'}"
                    </div>
                    <div class="text-[10px] text-slate-400 mt-1 flex justify-between">
                        <span>S: ${cand.session_id.substring(0,6)}... | Turn: ${cand.turn_id}</span>
                        <span>Conf: ${(cand.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        });
    }

    function initCharts() {
        Chart.defaults.font.family = "'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
        Chart.defaults.color = '#64748b';

        // Timeline
        const ctxTimeline = document.getElementById('timelineChart').getContext('2d');
        timelineChart = new Chart(ctxTimeline, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Requests', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14, 165, 233, 0.1)', fill: true, tension: 0.4 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
                    x: { grid: { display: false } }
                }
            }
        });

        // Intent
        const ctxIntent = document.getElementById('intentChart').getContext('2d');
        intentChart = new Chart(ctxIntent, {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], backgroundColor: ['#0ea5e9', '#3b82f6', '#8b5cf6', '#d946ef', '#f43f5e', '#f59e0b', '#10b981'] }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { boxWidth: 12 } } }, cutout: '70%' }
        });

        // Tools
        const ctxTool = document.getElementById('toolChart').getContext('2d');
        toolChart = new Chart(ctxTool, {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Usage', data: [], backgroundColor: '#6366f1', borderRadius: 4 }] },
            options: {
                responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: {
                    x: { beginAtZero: true, grid: { color: '#f1f5f9' } },
                    y: { grid: { display: false } }
                }
            }
        });
    }

    function updateCharts(telemetry) {
        // Timeline
        if (telemetry.timeline && telemetry.timeline.length > 0) {
            timelineChart.data.labels = telemetry.timeline.map(t => {
                const d = new Date(t.timestamp * 1000);
                return `${d.getHours()}:${d.getMinutes().toString().padStart(2, '0')}`;
            });
            timelineChart.data.datasets[0].data = telemetry.timeline.map(t => t.count);
            timelineChart.update();
        }

        // Intents
        if (telemetry.intents) {
            const intents = Object.entries(telemetry.intents).sort((a,b) => b[1] - a[1]);
            intentChart.data.labels = intents.map(i => i[0]);
            intentChart.data.datasets[0].data = intents.map(i => i[1]);
            intentChart.update();
        }

        // Tools
        if (telemetry.tools) {
            const tools = Object.entries(telemetry.tools).sort((a,b) => b[1] - a[1]).slice(0, 5);
            toolChart.data.labels = tools.map(t => t[0]);
            toolChart.data.datasets[0].data = tools.map(t => t[1]);
            toolChart.update();
        }
    }
});
