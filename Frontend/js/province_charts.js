// province_charts.js – Render time-series + static charts for a selected province.
// Uses the province-context API (time_series + province profile).

const ProvinceCharts = (() => {
    let charts = {};

    const COLORS = {
        primary: '#002147',
        slate: '#334155',
        emerald: '#059669',
        blue: '#2563eb',
        amber: '#d97706',
        sky: '#0284c7',
        red: '#dc2626',
        violet: '#7c3aed',
        rose: '#e11d48',
        teal: '#0d9488',
    };

    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: true, position: 'bottom', labels: { boxWidth: 8, font: { size: 10, family: 'Inter' } } },
            tooltip: { mode: 'index', intersect: false },
        },
        scales: {
            x: { grid: { display: false }, ticks: { font: { size: 10 } } },
            y: { grid: { color: 'rgba(148,163,184,0.15)' }, ticks: { font: { size: 10 } } },
        },
    };

    function destroyAll() {
        Object.values(charts).forEach(c => { if (c) c.destroy(); });
        charts = {};
    }

    function make(id, config) {
        const canvas = document.getElementById(id);
        if (!canvas || typeof Chart === 'undefined') return null;
        if (charts[id]) charts[id].destroy();
        charts[id] = new Chart(canvas, config);
        return charts[id];
    }

    // ── Time-series charts (need timeSeries array) ──

    function renderTimeSeries(timeSeries) {
        if (!timeSeries || !timeSeries.length) return;
        const years = timeSeries.map(r => String(r.year));

        // 1) GRDP Line
        make('province-grdp-line-chart', {
            type: 'line',
            data: {
                labels: years,
                datasets: [{
                    label: 'GRDP (tỷ VNĐ)',
                    data: timeSeries.map(r => Math.round(Number(r.grdp_billion_vnd_est || 0))),
                    borderColor: COLORS.primary, backgroundColor: 'rgba(0,33,71,0.06)',
                    borderWidth: 2, tension: 0.3, fill: true, pointRadius: 3, pointBackgroundColor: COLORS.primary,
                }],
            },
            options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => `${(v/1000).toFixed(0)}k` } } } },
        });

        // 2) Tax Revenue Bar
        make('province-tax-bar-chart', {
            type: 'bar',
            data: {
                labels: years,
                datasets: [{
                    label: 'Thu thuế (tỷ VNĐ)',
                    data: timeSeries.map(r => Math.round(Number(r.tax_revenue_est || 0))),
                    backgroundColor: COLORS.slate, borderRadius: 3, barThickness: 16,
                }],
            },
            options: chartDefaults,
        });

        // 3) Sector Area
        make('province-sector-area-chart', {
            type: 'line',
            data: {
                labels: years,
                datasets: [
                    { label: 'Nông nghiệp (%)', data: timeSeries.map(r => r.sector_agriculture_pct || 0), borderColor: COLORS.emerald, backgroundColor: 'rgba(5,150,105,0.12)', fill: true, tension: 0.3, borderWidth: 1.5 },
                    { label: 'Công nghiệp (%)', data: timeSeries.map(r => r.sector_industry_pct || 0), borderColor: COLORS.blue, backgroundColor: 'rgba(37,99,235,0.10)', fill: true, tension: 0.3, borderWidth: 1.5 },
                    { label: 'Dịch vụ (%)', data: timeSeries.map(r => r.sector_services_pct || 0), borderColor: COLORS.amber, backgroundColor: 'rgba(217,119,6,0.08)', fill: true, tension: 0.3, borderWidth: 1.5 },
                ],
            },
            options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => v + '%' } } } },
        });

        // 4) Trade Line
        make('province-trade-line-chart', {
            type: 'line',
            data: {
                labels: years,
                datasets: [
                    { label: 'Xuất khẩu (tỷ USD)', data: timeSeries.map(r => Number(r.export_billion_usd_est || 0)), borderColor: COLORS.sky, backgroundColor: 'rgba(2,132,199,0.06)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 },
                    { label: 'Nhập khẩu (tỷ USD)', data: timeSeries.map(r => Number(r.import_billion_usd_est || 0)), borderColor: COLORS.red, backgroundColor: 'rgba(220,38,38,0.04)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 },
                ],
            },
            options: chartDefaults,
        });

        // 5) GDP per Capita Line
        make('province-gdp-per-capita-chart', {
            type: 'line',
            data: {
                labels: years,
                datasets: [{
                    label: 'GDP/người (triệu VNĐ)',
                    data: timeSeries.map(r => {
                        const gdp = Number(r.grdp_billion_vnd_est || 0);
                        const pop = Number(r.population || 1);
                        return Math.round(gdp * 1000 / Math.max(pop, 1));
                    }),
                    borderColor: COLORS.violet, backgroundColor: 'rgba(124,58,237,0.06)',
                    borderWidth: 2, tension: 0.3, fill: true, pointRadius: 3,
                }],
            },
            options: chartDefaults,
        });
    }

    // ── Static charts (need province profile) ──

    function renderStaticCharts(province) {
        if (!province) return;

        // 6) Tax Breakdown Donut
        const taxB = province.tax_breakdown_billion_vnd || {};
        const taxLabels = ['GTGT (VAT)', 'TNDN (CIT)', 'TNCN (PIT)', 'TTĐB (SCT)', 'Khác'];
        const taxValues = [taxB.gtgt || 0, taxB.tndn || 0, taxB.tncn || 0, taxB.ttdb || 0, taxB.khac || 0];
        make('province-tax-donut-chart', {
            type: 'doughnut',
            data: {
                labels: taxLabels,
                datasets: [{ data: taxValues, backgroundColor: [COLORS.primary, COLORS.emerald, COLORS.sky, COLORS.amber, '#94a3b8'], borderWidth: 1, borderColor: '#fff' }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                cutout: '55%',
                plugins: {
                    legend: { position: 'right', labels: { boxWidth: 8, font: { size: 9 } } },
                    tooltip: { callbacks: { label: ctx => `${ctx.label}: ${Number(ctx.raw).toLocaleString('vi-VN')} tỷ` } },
                },
            },
        });

        // 7) Sector Composition Donut
        const sec = province.sector_composition_pct || {};
        make('province-sector-donut-chart', {
            type: 'doughnut',
            data: {
                labels: ['Nông nghiệp', 'Công nghiệp', 'Dịch vụ', 'Thuế SP'],
                datasets: [{ data: [sec.agriculture || 0, sec.industry || 0, sec.services || 0, sec.tax_product || 0], backgroundColor: [COLORS.emerald, COLORS.blue, COLORS.amber, '#94a3b8'], borderWidth: 1, borderColor: '#fff' }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                cutout: '55%',
                plugins: {
                    legend: { position: 'right', labels: { boxWidth: 8, font: { size: 9 } } },
                    tooltip: { callbacks: { label: ctx => `${ctx.label}: ${Number(ctx.raw).toFixed(1)}%` } },
                },
            },
        });
    }

    // ── Cross-province charts (need all provinces data) ──

    function renderCrossProvinceCharts(selectedCode) {
        const provinces = window.MacroMapData?.previewProvinces?.() || [];
        if (!provinces.length) return;

        // 8) Bubble Scatter: GDP vs Tax Revenue
        const bubbleData = provinces.map(p => ({
            x: Number(p.gdp_billion_vnd || 0),
            y: Number(p.tax_revenue_billion_vnd || 0),
            r: Math.max(3, Math.sqrt(Number(p.population || 0)) / 200),
        }));
        const bubbleColors = provinces.map(p =>
            String(p.province_code) === String(selectedCode) ? COLORS.red : 'rgba(0,33,71,0.35)'
        );
        make('province-gdp-tax-bubble-chart', {
            type: 'bubble',
            data: {
                datasets: [{
                    label: 'Tỉnh/Thành',
                    data: bubbleData,
                    backgroundColor: bubbleColors,
                    borderColor: bubbleColors.map(c => c === COLORS.red ? COLORS.red : 'rgba(0,33,71,0.5)'),
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const idx = ctx.dataIndex;
                                const p = provinces[idx];
                                return `${p?.province_name || ''}: GDP ${Number(p?.gdp_billion_vnd || 0).toLocaleString('vi-VN')} tỷ, Thuế ${Number(p?.tax_revenue_billion_vnd || 0).toLocaleString('vi-VN')} tỷ`;
                            },
                        },
                    },
                },
                scales: {
                    x: { title: { display: true, text: 'GRDP (tỷ VNĐ)', font: { size: 10 } }, grid: { color: 'rgba(148,163,184,0.1)' }, ticks: { font: { size: 9 }, callback: v => `${(v/1000).toFixed(0)}k` } },
                    y: { title: { display: true, text: 'Thu thuế (tỷ VNĐ)', font: { size: 10 } }, grid: { color: 'rgba(148,163,184,0.1)' }, ticks: { font: { size: 9 }, callback: v => `${(v/1000).toFixed(0)}k` } },
                },
            },
        });

        // 9) Top 10 Horizontal Bar
        const sorted = [...provinces].sort((a, b) => Number(b.gdp_billion_vnd || 0) - Number(a.gdp_billion_vnd || 0)).slice(0, 10);
        const barColors = sorted.map(p => String(p.province_code) === String(selectedCode) ? COLORS.primary : '#94a3b8');
        make('province-top10-bar-chart', {
            type: 'bar',
            data: {
                labels: sorted.map(p => p.province_name || p.province_code),
                datasets: [{
                    label: 'GRDP (tỷ VNĐ)',
                    data: sorted.map(p => Math.round(Number(p.gdp_billion_vnd || 0))),
                    backgroundColor: barColors,
                    borderRadius: 3,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${Number(ctx.parsed.x).toLocaleString('vi-VN')} tỷ VNĐ` } } },
                scales: {
                    x: { grid: { color: 'rgba(148,163,184,0.1)' }, ticks: { font: { size: 9 }, callback: v => `${(v/1000).toFixed(0)}k` } },
                    y: { grid: { display: false }, ticks: { font: { size: 9, weight: '700' } } },
                },
            },
        });
    }

    // ── Main entry ──

    function renderAll(data, selectedCode) {
        const section = document.getElementById('province-charts-section');
        if (!section) return;
        const ts = data.time_series || [];
        const prov = data.province || {};
        if (!ts.length && !prov.province_code) { section.classList.add('hidden'); return; }
        section.classList.remove('hidden');
        destroyAll();
        renderTimeSeries(ts);
        renderStaticCharts(prov);
        renderCrossProvinceCharts(selectedCode);
    }

    // Listen for province selection
    window.addEventListener('macro:province-selected', async (event) => {
        const code = event.detail?.provinceCode;
        if (!code) { destroyAll(); return; }
        try {
            const bv = window.MACRO_BOUNDARY_VERSION || 'vn_34_2025';
            const apiBase = window.API_BASE || 'http://localhost:8000/api';
            const resp = await fetch(`${apiBase}/simulation/province-context/${encodeURIComponent(code)}?boundary_version=${encodeURIComponent(bv)}`, { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const resolvedCode = data.province?.province_code || code;
            renderAll(data, resolvedCode);
        } catch (e) {
            console.warn('[ProvinceCharts] Failed to load:', e);
        }
    });

    return { renderAll, destroyAll };
})();
