// vietnam_echarts_map.js - deterministic ECharts heatmap over vn_34_2025.

const VietnamEChartsMap = (() => {
    const MAP_NAME = 'vietnam_macro_vn34';
    let chartInstance = null;
    let geoRegistered = false;
    let containerId = 'vietnam-echarts-map';

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

        chartInstance.showLoading({ text: 'Đang tải heatmap 34 đơn vị...', color: '#002147', fontSize: 13 });
        const mapState = await window.MacroMapData.loadState();
        if (!geoRegistered) {
            const geojson = JSON.parse(JSON.stringify(mapState.geojson || { type: 'FeatureCollection', features: [] }));
            (geojson.features || []).forEach((feature) => {
                feature.properties = feature.properties || {};
                feature.properties.name = feature.properties.province_code || feature.properties.name;
            });
            echarts.registerMap(MAP_NAME, geojson);
            geoRegistered = true;
        }
        chartInstance.hideLoading();
        renderChart();
    }

    function renderChart() {
        if (!chartInstance || !window.MacroMapData) return;
        const provinces = window.MacroMapData.previewProvinces();
        const mapData = provinces.map((province) => ({
            name: province.province_code,
            value: Number(province.risk_score || 0),
            companyCount: Number(province.num_enterprises || 0),
            provinceName: province.province_name,
            provinceCode: province.province_code,
            riskLevel: province.risk_level,
            gdp: Number(province.gdp_billion_vnd || 0),
            taxRevenue: Number(province.tax_revenue_billion_vnd || 0),
        }));
        const maxRisk = Math.max(80, ...mapData.map((item) => item.value));

        chartInstance.setOption({
            title: {
                text: 'BẢN ĐỒ CẢNH BÁO RỦI RO THEO ĐỊA LÝ',
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
                    const riskColor = d.value >= 65 ? '#ff6b6b' : d.value >= 35 ? '#ffd93d' : '#6bcb77';
                    return `<div style="min-width:210px">
                        <b style="font-size:13px">${window.MacroMapData.escapeHtml(d.provinceName || params.name)}</b>
                        <hr style="border-color:rgba(255,255,255,.2);margin:6px 0">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span>Điểm rủi ro</span><b style="color:${riskColor}">${Number(d.value).toFixed(1)}</b>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span>Doanh nghiệp</span><b>${Number(d.companyCount || 0).toLocaleString('vi-VN')}</b>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span>GDP/GRDP</span><b>${Number(d.gdp || 0).toLocaleString('vi-VN')} tỷ</b>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span>Thu thuế</span><b>${Number(d.taxRevenue || 0).toLocaleString('vi-VN')} tỷ</b>
                        </div>
                    </div>`;
                },
            },
            visualMap: {
                min: 0,
                max: maxRisk,
                left: 16,
                bottom: 20,
                text: ['Rủi ro cao', 'Rủi ro thấp'],
                textStyle: { color: '#44474e', fontSize: 10, fontWeight: 700 },
                inRange: { color: ['#e8f5e9', '#fff9c4', '#ffe0b2', '#ffab91', '#ef5350', '#b71c1c'] },
                calculable: true,
                orient: 'vertical',
                itemWidth: 14,
                itemHeight: 120,
            },
            series: [{
                name: 'Rủi ro Thuế',
                type: 'map',
                map: MAP_NAME,
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
        });

        chartInstance.off('click');
        chartInstance.on('click', (params) => {
            const d = params.data;
            if (d?.provinceCode && window.loadProvinceScenario) {
                window.loadProvinceScenario(d.provinceCode, d.provinceName || d.provinceCode);
            }
        });
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

    return { init, applyMacroParams, simulateNationalRisk, applyProvinceImpacts, onVisible };
})();

window.VietnamEChartsMap = VietnamEChartsMap;
