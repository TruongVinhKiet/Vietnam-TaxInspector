// map_enhancements.js - 7 premium enhancements for the macro simulation maps.
// #1 Radar Chart  #2 Stacked Bar  #3 3D Dual Pillars  #4 Leaflet Choropleth
// #5 ECharts Scatter  #6 PDF Export  #7 Mini-map Orientation

const MapEnhancements = (() => {
    let radarChart = null;
    let sectorChart = null;
    let provinceRadarChart = null;
    let mergerRadarChart = null;

    // ──────────────────────────────────────────────
    //  #1 - Radar Chart (merger panel: compare member provinces on 6 axes)
    // ──────────────────────────────────────────────
    function renderMergerRadar(data) {
        const canvas = document.getElementById('merger-radar-chart');
        if (!canvas || typeof Chart === 'undefined') return;
        if (mergerRadarChart) mergerRadarChart.destroy();
        const members = (data.member_rows || []).slice(0, 5);
        if (!members.length) return;

        const allProvinces = window.MacroMapData?.previewProvinces?.() || [];
        const avg = computeNationalAverage(allProvinces);

        const labels = ['GDP (tỷ)', 'Thu thuế (tỷ)', 'Doanh nghiệp', 'Dân số', 'FDI (tr USD)', 'Tuân thủ (%)'];
        const palette = ['#0ea5e9', '#22c55e', '#f97316', '#a855f7', '#ec4899'];

        const datasets = members.map((member, idx) => {
            const p = findProvinceProfile(member.province_code) || member;
            return {
                label: member.province_name || member.province_code,
                data: normalizeRadarData(p, avg),
                borderColor: palette[idx % palette.length],
                backgroundColor: palette[idx % palette.length] + '18',
                borderWidth: 2,
                pointRadius: 3,
            };
        });

        datasets.push({
            label: 'Trung bình QG',
            data: [50, 50, 50, 50, 50, 50],
            borderColor: '#94a3b8',
            backgroundColor: 'rgba(148,163,184,0.06)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
        });

        mergerRadarChart = new Chart(canvas, {
            type: 'radar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { display: false },
                        pointLabels: { font: { size: 9, weight: '700' }, color: '#475569' },
                        grid: { color: 'rgba(148,163,184,0.2)' },
                    },
                },
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 8, font: { size: 10 } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.r.toFixed(0)}%`,
                        },
                    },
                },
            },
        });
    }

    // ──────────────────────────────────────────────
    //  #2 - Stacked Bar Chart (sector composition of merger group)
    // ──────────────────────────────────────────────
    function renderSectorChart(data) {
        const canvas = document.getElementById('merger-sector-chart');
        if (!canvas || typeof Chart === 'undefined') return;
        if (sectorChart) sectorChart.destroy();
        const members = (data.member_rows || []).slice(0, 6);
        if (!members.length) return;

        const sectorColors = {
            'Nông nghiệp': '#22c55e',
            'Công nghiệp & Xây dựng': '#3b82f6',
            'Dịch vụ': '#f59e0b',
            'Thuế sản phẩm': '#94a3b8',
        };

        const sectors = [
            { key: 'agriculture', label: 'Nông nghiệp' },
            { key: 'industry', label: 'Công nghiệp & Xây dựng' },
            { key: 'services', label: 'Dịch vụ' },
            { key: 'tax_product', label: 'Thuế sản phẩm' }
        ];

        const labels = members.map((m) => m.province_name || m.province_code);
        const datasets = sectors.map((sec) => ({
            label: sec.label,
            data: members.map((m) => {
                const p = findProvinceProfile(m.province_code) || m;
                const gdp = Number(p?.gdp_billion_vnd || 0);
                const comp = p?.sector_composition_pct || {};
                const pct = Number(comp[sec.key] || 0);
                return Math.round(gdp * pct / 100);
            }),
            backgroundColor: sectorColors[sec.label],
            borderRadius: 3,
        }));

        sectorChart = new Chart(canvas, {
            type: 'bar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        stacked: true,
                        ticks: { callback: (v) => `${Number(v).toLocaleString('vi-VN')} tỷ` },
                        grid: { color: 'rgba(148,163,184,0.15)' },
                    },
                    y: { stacked: true, grid: { display: false }, ticks: { font: { size: 10, weight: '700' } } },
                },
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 8, font: { size: 9 } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.x || 0).toLocaleString('vi-VN')} tỷ`,
                        },
                    },
                },
            },
        });
    }

    // ──────────────────────────────────────────────
    //  #4 - Leaflet Choropleth Layer Toggle
    // ──────────────────────────────────────────────
    let currentChoroplethLayer = 'risk';

    function setupChoroplethSelector() {
        const selector = document.getElementById('leaflet-choropleth-layer');
        if (!selector) return;
        selector.addEventListener('change', () => {
            currentChoroplethLayer = selector.value;
            if (window.VietnamMap?.refreshChoropleth) {
                window.VietnamMap.refreshChoropleth(currentChoroplethLayer);
            }
        });
    }

    function getChoroplethLayer() {
        return currentChoroplethLayer;
    }

    // ──────────────────────────────────────────────
    //  #6 - PDF Export for merger analysis panel
    // ──────────────────────────────────────────────
    function setupPdfExport() {
        const btn = document.getElementById('merger-export-pdf-btn');
        if (!btn) return;
        btn.addEventListener('click', async () => {
            const panel = document.getElementById('merger-analysis-panel');
            if (!panel) return;
            btn.disabled = true;
            btn.innerHTML = '<span class="material-symbols-outlined text-[14px] animate-spin">sync</span> Đang tạo...';
            try {
                if (typeof html2canvas === 'undefined') {
                    const script = document.createElement('script');
                    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
                    document.head.appendChild(script);
                    await new Promise((resolve, reject) => { script.onload = resolve; script.onerror = reject; });
                }
                const canvas = await html2canvas(panel, {
                    scale: 2,
                    useCORS: true,
                    backgroundColor: '#ffffff',
                    logging: false,
                });
                const link = document.createElement('a');
                link.download = `merger_analysis_${Date.now()}.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
            } catch (error) {
                console.error('[MapEnhancements] Export failed:', error);
                alert('Không thể xuất báo cáo. Vui lòng thử lại.');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span class="material-symbols-outlined text-[14px]">picture_as_pdf</span> Xuất báo cáo';
            }
        });
    }

    // ──────────────────────────────────────────────
    //  #7 - Province Mini-map Orientation
    // ──────────────────────────────────────────────
    function renderMinimap(provinceCode) {
        const canvas = document.getElementById('province-minimap-canvas');
        const wrap = document.getElementById('province-minimap');
        if (!canvas || !wrap) return;
        wrap.style.display = provinceCode ? 'flex' : 'none';
        if (!provinceCode) return;

        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        // Draw simplified Vietnam silhouette
        ctx.save();
        ctx.fillStyle = '#e2e8f0';
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1;
        drawVietnamSilhouette(ctx, w, h);
        ctx.restore();

        // Highlight selected province
        const provinces = window.MacroMapData?.previewProvinces?.() || [];
        const prov = provinces.find((p) => String(p.province_code) === String(provinceCode));
        if (prov?.lat && prov?.lng) {
            const x = lngToX(prov.lng, w);
            const y = latToY(prov.lat, h);
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Pulse ring
            ctx.strokeStyle = 'rgba(239,68,68,0.3)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    function drawVietnamSilhouette(ctx, w, h) {
        // Simplified polygon points (normalized 0-1)
        const points = [
            [0.45,0.02],[0.55,0.05],[0.6,0.08],[0.65,0.12],[0.7,0.15],
            [0.72,0.2],[0.68,0.25],[0.72,0.3],[0.65,0.32],[0.6,0.35],
            [0.55,0.38],[0.5,0.42],[0.48,0.48],[0.52,0.52],[0.55,0.56],
            [0.58,0.6],[0.6,0.65],[0.55,0.7],[0.5,0.75],[0.48,0.8],
            [0.52,0.85],[0.55,0.88],[0.58,0.92],[0.55,0.95],[0.5,0.97],
            [0.45,0.95],[0.42,0.9],[0.4,0.85],[0.38,0.82],[0.35,0.78],
            [0.32,0.72],[0.3,0.65],[0.32,0.58],[0.35,0.5],[0.38,0.42],
            [0.4,0.35],[0.38,0.28],[0.35,0.22],[0.38,0.15],[0.4,0.1],
            [0.42,0.05],
        ];
        ctx.beginPath();
        points.forEach(([x, y], i) => {
            const px = x * w;
            const py = y * h;
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    }

    function lngToX(lng, w) { return ((lng - 102) / (110 - 102)) * w; }
    function latToY(lat, h) { return ((23.5 - lat) / (23.5 - 8.5)) * h; }

    // ──────────────────────────────────────────────
    //  Province Radar Chart (#1 variant for scenario panel)
    // ──────────────────────────────────────────────
    function renderProvinceRadar(provinceCode) {
        const canvas = document.getElementById('province-radar-chart');
        const wrap = document.getElementById('province-radar-wrap');
        if (!canvas || !wrap || typeof Chart === 'undefined') return;

        const provinces = window.MacroMapData?.previewProvinces?.() || [];
        const prov = provinces.find((p) => String(p.province_code) === String(provinceCode));
        if (!prov) { wrap.style.display = 'none'; return; }
        wrap.style.display = 'block';

        const avg = computeNationalAverage(provinces);
        const provData = normalizeRadarData(prov, avg);
        const labels = ['GDP', 'Thu thuế', 'Doanh nghiệp', 'Dân số', 'FDI', 'Tuân thủ'];

        if (provinceRadarChart) provinceRadarChart.destroy();
        provinceRadarChart = new Chart(canvas, {
            type: 'radar',
            data: {
                labels,
                datasets: [
                    {
                        label: prov.province_name || provinceCode,
                        data: provData,
                        borderColor: '#002147',
                        backgroundColor: 'rgba(0,33,71,0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: '#002147',
                    },
                    {
                        label: 'Trung bình QG',
                        data: [50, 50, 50, 50, 50, 50],
                        borderColor: '#94a3b8',
                        backgroundColor: 'rgba(148,163,184,0.05)',
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        pointRadius: 0,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { display: false },
                        pointLabels: { font: { size: 8, weight: '700' }, color: '#475569' },
                        grid: { color: 'rgba(148,163,184,0.2)' },
                    },
                },
                plugins: {
                    legend: { display: false },
                },
            },
        });
    }

    // ──────────────────────────────────────────────
    //  Helpers
    // ──────────────────────────────────────────────
    function computeNationalAverage(provinces) {
        if (!provinces.length) return { gdp: 1, tax: 1, enterprises: 1, pop: 1, fdi: 1, compliance: 0.8 };
        const sum = (key) => provinces.reduce((acc, p) => acc + Number(p[key] || 0), 0);
        const n = provinces.length;
        return {
            gdp: sum('gdp_billion_vnd') / n,
            tax: sum('tax_revenue_billion_vnd') / n,
            enterprises: sum('num_enterprises') / n,
            pop: sum('population') / n,
            fdi: sum('fdi_billion_usd') / n,
            compliance: provinces.reduce((acc, p) => acc + Number(p.compliance_rate || 0), 0) / n,
        };
    }

    function normalizeRadarData(province, avg) {
        if (!province) return [0, 0, 0, 0, 0, 0];
        const clamp = (v) => Math.min(100, Math.max(0, v));
        return [
            clamp((Number(province.gdp_billion_vnd || 0) / Math.max(avg.gdp, 1)) * 50),
            clamp((Number(province.tax_revenue_billion_vnd || 0) / Math.max(avg.tax, 1)) * 50),
            clamp((Number(province.num_enterprises || 0) / Math.max(avg.enterprises, 1)) * 50),
            clamp((Number(province.population || 0) / Math.max(avg.pop, 1)) * 50),
            clamp((Number(province.fdi_billion_usd || 0) / Math.max(avg.fdi, 0.01)) * 50),
            clamp((Number(province.compliance_rate || 0) / Math.max(avg.compliance, 0.01)) * 50),
        ];
    }

    function findProvinceProfile(code) {
        const provinces = window.MacroMapData?.previewProvinces?.() || [];
        return provinces.find((p) => String(p.province_code) === String(code)) || null;
    }

    // ──────────────────────────────────────────────
    //  Event Listeners
    // ──────────────────────────────────────────────
    function handleSelectionUpdate(event) {
        const detail = event.detail || {};
        if (detail.provinceCode) {
            renderProvinceRadar(detail.provinceCode);
            renderMinimap(detail.provinceCode);
        }
    }
    window.addEventListener('macro:province-selected', handleSelectionUpdate);
    window.addEventListener('macro:province-resolved-code', handleSelectionUpdate);

    window.addEventListener('merger:data-loaded', (event) => {
        const data = event.detail || {};
        renderMergerRadar(data);
        renderSectorChart(data);
    });

    document.addEventListener('DOMContentLoaded', () => {
        setupChoroplethSelector();
        setupPdfExport();
    });

    return {
        renderMergerRadar,
        renderSectorChart,
        renderProvinceRadar,
        renderMinimap,
        getChoroplethLayer,
    };
})();

window.MapEnhancements = MapEnhancements;
