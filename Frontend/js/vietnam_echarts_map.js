// vietnam_echarts_map.js - deterministic ECharts heatmap for 34/63 boundary modes.

const VietnamEChartsMap = (() => {
    let chartInstance = null;
    let registeredMaps = new Set();
    let containerId = 'vietnam-echarts-map';
    let currentMapName = 'vietnam_macro_vn34';
    let currentTimelineYear = 2024;
    let selectedProvinceCode = null;

    async function init(id = 'vietnam-echarts-map') {
        containerId = id;
        const container = document.getElementById(containerId);
        if (!container || typeof echarts === 'undefined') return;
        if (!window.MacroMapData) {
            container.innerHTML = '<div class="flex h-full items-center justify-center text-sm text-slate-400">Không tải được dữ liệu bản đồ.</div>';
            return;
        }

        if (!chartInstance) {
            chartInstance = echarts.init(container, null, { renderer: 'canvas' });
            window.addEventListener('resize', onVisible);
        }

        chartInstance.showLoading({ text: 'Đang tải heatmap...', color: '#002147', fontSize: 13 });
        const mapState = await window.MacroMapData.loadState({ force: true });
        registerCurrentMap(mapState);
        chartInstance.hideLoading();
        renderChart();
    }

    function registerCurrentMap(mapState) {
        const boundaryVersion = window.MacroMapData.getBoundaryVersion();
        currentMapName = boundaryVersion === 'vn_63_legacy' ? 'vietnam_macro_vn63' : 'vietnam_macro_vn34';
        if (registeredMaps.has(currentMapName)) return;
        const geojson = JSON.parse(JSON.stringify(mapState.geojson || { type: 'FeatureCollection', features: [] }));
        (geojson.features || []).forEach((feature) => {
            feature.properties = feature.properties || {};
            feature.properties.name = feature.properties.province_code || feature.properties.name;
        });
        echarts.registerMap(currentMapName, geojson);
        registeredMaps.add(currentMapName);
    }

    function renderChart() {
        if (!chartInstance || !window.MacroMapData) return;
        const isLegacy = window.MacroMapData.getBoundaryVersion() === 'vn_63_legacy';
        const provinces = window.MacroMapData.previewProvinces();
        const mapData = provinces.map((province) => {
            const riskValue = Number(province.risk_score || 0);
            return {
                name: province.province_code,
                value: isLegacy ? legacyHeatValue(province, currentTimelineYear) : riskValue,
                riskValue,
                companyCount: Number(province.num_enterprises || 0),
                provinceName: province.province_name,
                provinceCode: province.province_code,
                riskLevel: province.risk_level,
                gdp: Number(province.gdp_billion_vnd || 0),
                taxRevenue: Number(province.tax_revenue_billion_vnd || 0),
            };
        });
        const maxValue = Math.max(isLegacy ? 1000000 : 80, ...mapData.map((item) => item.value));
        const years = [2019, 2020, 2021, 2022, 2023, 2024, 2025];

        chartInstance.setOption({
            title: {
                text: isLegacy ? `BẢN ĐỒ NHIỆT GRDP 63 TỈNH - ${currentTimelineYear}` : 'BẢN ĐỒ CẢNH BÁO RỦI RO THEO ĐỊA LÝ',
                left: 'center',
                top: 8,
                textStyle: { color: '#002147', fontSize: 15, fontWeight: 900, letterSpacing: 0 },
            },
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(0,33,71,0.94)',
                borderColor: '#465f88',
                textStyle: { color: '#fff', fontSize: 12 },
                formatter: (params) => {
                    const d = params.data;
                    if (!d) return `<b>${window.MacroMapData.escapeHtml(params.name)}</b><br/>Không có dữ liệu`;
                    const riskColor = d.riskValue >= 65 ? '#ff6b6b' : d.riskValue >= 35 ? '#ffd93d' : '#6bcb77';
                    return `<div style="min-width:220px">
                        <b style="font-size:13px">${window.MacroMapData.escapeHtml(d.provinceName || params.name)}</b>
                        <hr style="border-color:rgba(255,255,255,.2);margin:6px 0">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span>${isLegacy ? `GRDP ${currentTimelineYear}` : 'Điểm rủi ro'}</span>
                            <b style="color:${riskColor}">${isLegacy ? Number(d.value || 0).toLocaleString('vi-VN') + ' tỷ' : Number(d.riskValue).toFixed(1)}</b>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span>Doanh nghiệp</span><b>${Number(d.companyCount || 0).toLocaleString('vi-VN')}</b>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span>GDP/GRDP hiện tại</span><b>${Number(d.gdp || 0).toLocaleString('vi-VN')} tỷ</b>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span>Thu thuế</span><b>${Number(d.taxRevenue || 0).toLocaleString('vi-VN')} tỷ</b>
                        </div>
                    </div>`;
                },
            },
            visualMap: {
                min: 0,
                max: maxValue,
                left: 16,
                bottom: isLegacy ? 64 : 20,
                text: isLegacy ? ['GRDP cao', 'GRDP thấp'] : ['Rủi ro cao', 'Rủi ro thấp'],
                textStyle: { color: '#44474e', fontSize: 10, fontWeight: 700 },
                inRange: { color: isLegacy ? ['#eef2ff', '#bae6fd', '#7dd3fc', '#38bdf8', '#0369a1'] : ['#e8f5e9', '#fff9c4', '#ffe0b2', '#ffab91', '#ef5350', '#b71c1c'] },
                calculable: true,
                orient: 'vertical',
                itemWidth: 14,
                itemHeight: 120,
            },
            timeline: isLegacy ? {
                axisType: 'category',
                bottom: 8,
                left: 120,
                right: 60,
                data: years.map(String),
                currentIndex: Math.max(0, years.indexOf(currentTimelineYear)),
                label: { color: '#475569', fontSize: 10 },
                checkpointStyle: { color: '#002147', borderColor: '#002147' },
                controlStyle: { show: false },
            } : undefined,
            series: [{
                name: isLegacy ? 'GRDP tỉnh' : 'Rủi ro Thuế',
                type: 'map',
                map: currentMapName,
                nameProperty: 'province_code',
                roam: true,
                zoom: 1.08,
                center: [106.5, 16.2],
                scaleLimit: { min: 0.85, max: 6 },
                selectedMode: 'single',
                label: {
                    show: true,
                    fontSize: 7,
                    color: '#334155',
                    formatter: (params) => {
                        const item = mapData.find((row) => row.name === params.name);
                        const label = item?.provinceName || params.name;
                        return label.length > 9 ? label.substring(0, 8) + '…' : label;
                    },
                },
                emphasis: {
                    label: { show: true, fontSize: 12, fontWeight: 'bold', color: '#002147' },
                    itemStyle: { areaColor: '#aec7f6', shadowBlur: 18, shadowColor: 'rgba(0,33,71,0.35)', borderWidth: 2, borderColor: '#002147' },
                },
                select: {
                    label: { show: true, fontSize: 12, fontWeight: 'bold' },
                    itemStyle: { areaColor: '#d6e3ff' },
                },
                itemStyle: { borderColor: '#fff', borderWidth: 1, areaColor: '#f5f5f5' },
                data: mapData,
                animationDurationUpdate: 450,
                animationEasingUpdate: 'cubicOut',
            }],
        }, true);

        if (selectedProvinceCode && chartInstance) {
            chartInstance.dispatchAction({
                type: 'select',
                seriesIndex: 0,
                name: selectedProvinceCode
            });
        }

        chartInstance.off('click');
        chartInstance.on('click', (params) => {
            const d = params.data;
            if (d?.provinceCode && window.loadProvinceScenario) {
                selectedProvinceCode = d.provinceCode;
                window.loadProvinceScenario(d.provinceCode, d.provinceName || d.provinceCode);
                window.dispatchEvent(new CustomEvent('macro:province-selected', {
                    detail: { provinceCode: d.provinceCode, provinceName: d.provinceName || d.provinceCode },
                }));
            }
        });
        chartInstance.off('timelinechanged');
        chartInstance.on('timelinechanged', (params) => {
            currentTimelineYear = years[params.currentIndex] || 2024;
            renderChart();
        });
    }

    function legacyHeatValue(province, year) {
        const rows = province.time_series_preview || [];
        const row = rows.find((item) => Number(item.year) === Number(year));
        if (row?.grdp_billion_vnd_est != null) return Number(row.grdp_billion_vnd_est);
        return Number(province.gdp_billion_vnd || 0);
    }

    function applyMacroParams(params = {}) {
        if (!window.MacroMapData) return;
        window.MacroMapData.applyMacroParams(params);
        renderChart();
    }

    function simulateNationalRisk(params = {}) {
        applyMacroParams(params);
    }

    function applyProvinceImpacts(impacts = []) {
        if (!window.MacroMapData) return;
        window.MacroMapData.applyProvinceImpacts(impacts);
        renderChart();
    }

    function onVisible() {
        if (chartInstance) chartInstance.resize();
    }

    async function switchBoundary(boundaryVersion) {
        if (!window.MacroMapData) return;
        const normalized = window.MacroMapData.setBoundaryVersion(boundaryVersion);
        const mapState = await window.MacroMapData.loadState({ boundaryVersion: normalized, force: true });
        registerCurrentMap(mapState);
        renderChart();
        onVisible();
    }

    // #5 ECharts Scatter GDP vs Tax Revenue
    function renderScatterGdpTax() {
        if (!chartInstance || !window.MacroMapData) return;
        const provinces = window.MacroMapData.previewProvinces();
        const scatterData = provinces.map((p) => ({
            value: [Number(p.gdp_billion_vnd || 0), Number(p.tax_revenue_billion_vnd || 0)],
            symbolSize: Math.max(6, Math.min(30, Math.sqrt(Number(p.num_enterprises || 0)) / 4)),
            itemStyle: {
                color: Number(p.risk_score || 0) >= 65 ? '#ef4444'
                    : Number(p.risk_score || 0) >= 35 ? '#f59e0b' : '#22c55e',
                opacity: 0.85,
            },
            provinceName: p.province_name,
            provinceCode: p.province_code,
            riskScore: p.risk_score,
            enterprises: p.num_enterprises,
        }));

        // Linear regression
        const xs = scatterData.map((d) => d.value[0]);
        const ys = scatterData.map((d) => d.value[1]);
        const n = xs.length;
        const sumX = xs.reduce((a, b) => a + b, 0);
        const sumY = ys.reduce((a, b) => a + b, 0);
        const sumXY = xs.reduce((a, x, i) => a + x * ys[i], 0);
        const sumX2 = xs.reduce((a, x) => a + x * x, 0);
        const slope = (n * sumXY - sumX * sumY) / Math.max(n * sumX2 - sumX * sumX, 0.001);
        const intercept = (sumY - slope * sumX) / Math.max(n, 1);
        const minX = Math.min(...xs, 0);
        const maxX = Math.max(...xs);
        const regLine = [[minX, slope * minX + intercept], [maxX, slope * maxX + intercept]];

        chartInstance.setOption({
            title: {
                text: 'PHÂN TÁN GDP – THU THUẾ THEO TỈNH',
                left: 'center', top: 8,
                textStyle: { color: '#002147', fontSize: 15, fontWeight: 900 },
            },
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(0,33,71,0.94)',
                textStyle: { color: '#fff', fontSize: 12 },
                formatter: (params) => {
                    const d = params.data;
                    if (!d) return '';
                    return `<b>${d.provinceName || ''}</b><br/>GDP: ${Number(d.value[0]).toLocaleString('vi-VN')} tỷ<br/>Thu thuế: ${Number(d.value[1]).toLocaleString('vi-VN')} tỷ<br/>Doanh nghiệp: ${Number(d.enterprises || 0).toLocaleString('vi-VN')}<br/>Rủi ro: ${Number(d.riskScore || 0).toFixed(1)}`;
                },
            },
            grid: { left: 80, right: 40, top: 50, bottom: 50 },
            xAxis: {
                type: 'value', name: 'GDP (tỷ VND)',
                nameTextStyle: { color: '#475569', fontSize: 10, fontWeight: 700 },
                axisLabel: { formatter: (v) => (v / 1000).toFixed(0) + 'K' },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.15)' } },
            },
            yAxis: {
                type: 'value', name: 'Thu thuế (tỷ VND)',
                nameTextStyle: { color: '#475569', fontSize: 10, fontWeight: 700 },
                axisLabel: { formatter: (v) => (v / 1000).toFixed(0) + 'K' },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.15)' } },
            },
            visualMap: undefined,
            timeline: undefined,
            series: [
                { type: 'scatter', data: scatterData, animationDuration: 600 },
                {
                    type: 'line', data: regLine, symbol: 'none',
                    lineStyle: { color: '#94a3b8', width: 1.5, type: 'dashed' },
                    tooltip: { show: false },
                    silent: true,
                },
            ],
        }, true);

        chartInstance.off('click');
        chartInstance.on('click', (params) => {
            const d = params.data;
            if (d?.provinceCode && window.loadProvinceScenario) {
                selectedProvinceCode = d.provinceCode;
                window.loadProvinceScenario(d.provinceCode, d.provinceName || d.provinceCode);
                window.dispatchEvent(new CustomEvent('macro:province-selected', {
                    detail: { provinceCode: d.provinceCode, provinceName: d.provinceName },
                }));
            }
        });
    }

    window.addEventListener('macro:boundary-change', (event) => {
        switchBoundary(event.detail?.boundaryVersion || 'vn_34_2025');
    });

    window.addEventListener('macro:province-selected', (event) => {
        const code = event.detail?.provinceCode;
        if (code && String(code) !== String(selectedProvinceCode)) {
            selectedProvinceCode = code;
            if (chartInstance) {
                chartInstance.dispatchAction({
                    type: 'select',
                    seriesIndex: 0,
                    name: code
                });
            }
        }
    });

    window.addEventListener('macro:province-resolved-code', (event) => {
        const code = event.detail?.provinceCode;
        if (code && String(code) !== String(selectedProvinceCode)) {
            selectedProvinceCode = code;
            if (chartInstance) {
                chartInstance.dispatchAction({
                    type: 'select',
                    seriesIndex: 0,
                    name: code
                });
            }
        }
    });

    return { init, applyMacroParams, simulateNationalRisk, applyProvinceImpacts, switchBoundary, renderScatterGdpTax, onVisible };
})();

window.VietnamEChartsMap = VietnamEChartsMap;

