/**
 * Vietnam ECharts Geographic Heatmap
 * Adapted from fraud.js for use in Simulation dashboard
 */
const VietnamEChartsMap = (function () {
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

    let chartInstance = null;
    let _vietnamGeoLoaded = false;
    let currentData = [];

    async function loadGeoJson() {
        if (_vietnamGeoLoaded) return true;
        try {
            const resp = await fetch('../json/vietnam.json');
            if (!resp.ok) throw new Error('Network response was not ok');
            const geoData = await resp.json();
            echarts.registerMap('vietnam', geoData);
            _vietnamGeoLoaded = true;
            return true;
        } catch (err) {
            console.error('[GeoMap] Failed to load vietnam.json:', err);
            return false;
        }
    }

    async function init(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (!chartInstance) {
            chartInstance = echarts.init(container);
            window.addEventListener('resize', () => chartInstance.resize());
        }

        const isLoaded = await loadGeoJson();
        if (!isLoaded) {
            chartInstance.setOption({
                title: {
                    text: 'Lỗi tải bản đồ GeoJSON', left: 'center', top: 'center',
                    textStyle: { color: '#dc2626', fontSize: 14 }
                }
            });
            return;
        }

        // Fetch simulation data from backend if not passed
        try {
            const boundaryVersion = window.MACRO_BOUNDARY_VERSION || 'vn_34_2025';
            const apiBase = window.API_BASE || 'http://localhost:8000/api';
            const resp = await fetch(`${apiBase}/simulation/provinces?boundary_version=${encodeURIComponent(boundaryVersion)}`);
            if (resp.ok) {
                const data = await resp.json();
                const provinces = Array.isArray(data.provinces) ? data.provinces : [];
                const stats = provinces.map(p => ({
                    province: p.province_name,
                    province_code: p.province_code || p.code,
                    avg_risk: p.risk_level === 'high' ? 80 : p.risk_level === 'medium' ? 40 : 10,
                    company_count: p.num_enterprises || 0
                }));
                currentData = stats;
                renderChart(stats);
            } else {
                renderChart([]);
            }
        } catch (e) {
            console.error('Lỗi khi fetch data cho bản đồ echarts', e);
            renderChart([]);
        }
    }

    function renderChart(provinceStats) {
        if (!chartInstance) return;

        if (!provinceStats || provinceStats.length === 0) {
            // Mock base data for 63 provinces with 0 risk just to show map
            provinceStats = Object.keys(PROVINCE_NAME_MAP).map(k => ({
                province: k, avg_risk: 0, company_count: 0
            }));
            // Provide a real looking fallback if we want
        }

        const mapData = provinceStats.map(p => {
            const geoName = PROVINCE_NAME_MAP[p.province] || p.province;
            return {
                name: geoName,
                value: Math.round(p.avg_risk * 100) / 100,
                companyCount: p.company_count,
                originalName: p.province,
                provinceCode: p.province_code,
            };
        });

        const maxRisk = Math.max(...mapData.map(d => d.value), 60);

        chartInstance.setOption({
            title: {
                text: 'BẢN ĐỒ CẢNH BÁO RỦI RO THEO ĐỊA LÝ',
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
                            <span>Điểm rủi ro (Mô phỏng)</span>
                            <b style="color:${riskColor}">${d.value.toFixed(1)}</b>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span>Số doanh nghiệp</span>
                            <b>${d.companyCount ? d.companyCount.toLocaleString() : '---'}</b>
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

        // Setup click handler
        chartInstance.off('click');
        chartInstance.on('click', (params) => {
            const d = params.data;
            if (d && window.loadProvinceScenario) {
                window.loadProvinceScenario(d.provinceCode, d.originalName || params.name);
            }
        });
    }

    // Exposed to let other scripts update the map (e.g. from macro params sync)
    function simulateNationalRisk(params) {
        if (!chartInstance) return;

        // Simple mock delta effect across provinces based on macro parameters
        const baseRisk = 25;
        const gdpEffect = -(params.gdp_delta_pct || 0) * 1.5;
        const compEffect = -(params.compliance_delta || 0) * 0.8;
        const totalDelta = gdpEffect + compEffect;

        const fakeStats = Object.keys(PROVINCE_NAME_MAP).map(k => {
            // Add some randomness per province
            const rand = Math.random() * 10 - 5;
            let risk = baseRisk + totalDelta + rand;
            if (k === 'TP.HCM' || k === 'Hà Nội') risk += 15; // Higher base for major cities
            if (risk < 0) risk = 0;
            if (risk > 100) risk = 100;

            return {
                province: k,
                avg_risk: risk,
                company_count: Math.floor(Math.random() * 50000) + 1000
            };
        });

        renderChart(fakeStats);
    }

    return { init, simulateNationalRisk };
})();

window.VietnamEChartsMap = VietnamEChartsMap;