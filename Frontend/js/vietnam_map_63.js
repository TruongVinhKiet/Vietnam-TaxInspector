// vietnam_map.js - Interactive Vietnam Province Map using ECharts.
// Renders a choropleth heatmap of 63 provinces with risk-level coloring,
// matching the visual style of the Geographic Risk Heatmap in fraud.js.

const VietnamMap = (() => {
    let chartInstance = null;
    let leafletMap = null;
    let leafletLayer = null;
    let selectedProvinceCode = null;
    let provincesData = [];
    let provincesByCode = new Map();
    let provincesByName = new Map();

    // Province name mapping: API names -> GeoJSON names (reuse from fraud.js pattern)
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

    let _geoRegistered = false;

    function getApiBase() {
        return window.API_BASE || 'http://localhost:8000/api';
    }

    function getBoundaryVersion() {
        return window.MACRO_BOUNDARY_VERSION || 'vn_34_2025';
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function normalizeName(value) {
        return String(value || '')
            .normalize('NFD')
            .replace(/[\u0300-\u036f]/g, '')
            .replace(/^(tinh|thanh pho|tp\.?|province|city)\s+/i, '')
            .replace(/\s+/g, ' ')
            .trim()
            .toLowerCase();
    }

    function rebuildProvinceIndexes() {
        provincesByCode = new Map();
        provincesByName = new Map();
        provincesData.forEach((province) => {
            const code = String(province.province_code || province.code || '').trim();
            if (code) provincesByCode.set(code, province);
            const name = normalizeName(province.province_name || province.name || '');
            if (name) provincesByName.set(name, province);
        });
    }

    function getRiskValue(riskLevel) {
        if (riskLevel === 'high') return 80;
        if (riskLevel === 'medium') return 45;
        if (riskLevel === 'low') return 15;
        return 5;
    }

    function getRiskColor(riskLevel) {
        const value = getRiskValue(riskLevel);
        if (value >= 75) return '#dc2626';
        if (value >= 55) return '#f97316';
        if (value >= 35) return '#facc15';
        if (value >= 15) return '#86efac';
        return '#dcfce7';
    }

    function resolveGeoName(provinceName) {
        return PROVINCE_NAME_MAP[provinceName] || provinceName;
    }

    function findProvinceByGeoName(geoName) {
        // Direct code match
        for (const [, prov] of provincesByCode) {
            const resolved = resolveGeoName(prov.province_name || prov.name || '');
            if (resolved === geoName) return prov;
        }
        // Fuzzy name match
        const normalizedGeo = normalizeName(geoName);
        if (provincesByName.has(normalizedGeo)) return provincesByName.get(normalizedGeo);
        return provincesData.find((p) => {
            const pName = normalizeName(p.province_name || p.name || '');
            return pName === normalizedGeo || pName.includes(normalizedGeo) || normalizedGeo.includes(pName);
        }) || null;
    }

    async function init(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (typeof L !== 'undefined') {
            await initLeaflet(containerId);
            return;
        }

        // Ensure ECharts is available
        if (typeof echarts === 'undefined') {
            container.innerHTML = `<div class="flex h-full min-h-[360px] items-center justify-center text-red-600 font-medium text-center px-6">
                Thư viện ECharts chưa được tải. Vui lòng kiểm tra kết nối mạng.
            </div>`;
            return;
        }

        // Dispose existing instance
        if (chartInstance) {
            chartInstance.dispose();
            chartInstance = null;
        }

        // Initialize ECharts
        chartInstance = echarts.init(container, null, { renderer: 'canvas' });

        // Show loading
        chartInstance.showLoading({ text: 'Đang tải bản đồ Việt Nam...', fontSize: 13, color: '#002147' });

        // Load provinces data from API
        try {
            const data = await fetchJsonSafe(`${getApiBase()}/simulation/provinces?boundary_version=${encodeURIComponent(getBoundaryVersion())}`, 12000);
            provincesData = Array.isArray(data.provinces) ? data.provinces : [];
            rebuildProvinceIndexes();
        } catch (error) {
            console.warn('[VietnamMap] Could not load provinces from API, using defaults:', error.message);
            provincesData = [];
        }

        // Register GeoJSON map if not already
        if (!_geoRegistered) {
            try {
                const geoResp = await fetch('../json/vietnam.json');
                if (!geoResp.ok) throw new Error(`HTTP ${geoResp.status}`);
                const geoData = await geoResp.json();
                echarts.registerMap('vietnam', geoData);
                _geoRegistered = true;
            } catch (err) {
                console.error('[VietnamMap] Failed to load vietnam.json:', err);
                chartInstance.hideLoading();
                chartInstance.setOption({
                    title: {
                        text: 'Lỗi tải bản đồ GeoJSON',
                        subtext: 'Vui lòng kiểm tra file json/vietnam.json',
                        left: 'center', top: 'center',
                        textStyle: { color: '#dc2626', fontSize: 14 }
                    }
                });
                return;
            }
        }

        chartInstance.hideLoading();
        renderMap();
        setupClickHandler();
        setupResize(container);
    }

    async function initLeaflet(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        if (leafletMap) {
            leafletMap.remove();
            leafletMap = null;
            leafletLayer = null;
        }
        if (chartInstance) {
            chartInstance.dispose();
            chartInstance = null;
        }

        try {
            const data = await fetchJsonSafe(`${getApiBase()}/simulation/provinces?boundary_version=${encodeURIComponent(getBoundaryVersion())}`, 12000);
            provincesData = Array.isArray(data.provinces) ? data.provinces : [];
            rebuildProvinceIndexes();
        } catch (error) {
            console.warn('[VietnamMap] Could not load provinces from API, using empty list:', error.message);
            provincesData = [];
        }

        leafletMap = L.map(containerId, {
            zoomControl: true,
            scrollWheelZoom: true,
            attributionControl: false,
            preferCanvas: true,
        }).setView([16.4, 106.2], 5.4);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 9,
            minZoom: 4,
            opacity: 0.28,
        }).addTo(leafletMap);

        const geojson = await fetchJsonSafe(`${getApiBase()}/simulation/geojson-vietnam?boundary_version=${encodeURIComponent(getBoundaryVersion())}`, 14000);
        leafletLayer = L.geoJSON(geojson, {
            style: leafletFeatureStyle,
            onEachFeature: bindLeafletFeature,
        }).addTo(leafletMap);

        try {
            leafletMap.fitBounds(leafletLayer.getBounds(), { padding: [16, 16] });
        } catch (_) {
            leafletMap.setView([16.4, 106.2], 5.2);
        }
        setTimeout(() => leafletMap && leafletMap.invalidateSize(), 100);
    }

    function getProvinceFromFeature(feature) {
        const props = feature?.properties || {};
        const code = String(props.province_code || props.code || props.gso_code || '').trim();
        if (code && provincesByCode.has(code)) return provincesByCode.get(code);
        const name = props.province_name || props.name || props.NAME_1 || props.Name || '';
        return findProvinceByGeoName(name) || {
            province_code: code,
            province_name: name,
            risk_level: 'unknown',
            gdp_billion_vnd: 0,
            tax_revenue_billion_vnd: 0,
            compliance_rate: 0,
        };
    }

    function leafletFeatureStyle(feature) {
        const province = getProvinceFromFeature(feature);
        const selected = selectedProvinceCode && String(province.province_code) === String(selectedProvinceCode);
        return {
            color: selected ? '#002147' : '#ffffff',
            weight: selected ? 2.5 : 1,
            fillColor: getRiskColor(province.risk_level),
            fillOpacity: selected ? 0.9 : 0.78,
            opacity: 1,
        };
    }

    function bindLeafletFeature(feature, layer) {
        const province = getProvinceFromFeature(feature);
        const name = province.province_name || feature?.properties?.name || 'Tỉnh/Thành phố';
        layer.bindTooltip(name, { sticky: true, direction: 'top', className: 'text-xs font-bold' });
        layer.on({
            mouseover: () => {
                layer.setStyle({ weight: 2.5, color: '#002147', fillOpacity: 0.92 });
                layer.bindPopup(leafletPopupHtml(province), { maxWidth: 280 }).openPopup();
            },
            mouseout: () => {
                if (leafletLayer) leafletLayer.resetStyle(layer);
            },
            click: () => {
                selectedProvinceCode = province.province_code;
                refreshLeafletStyles();
                if (province.province_code && window.loadProvinceScenario) {
                    window.loadProvinceScenario(province.province_code, name);
                }
            },
        });
    }

    function leafletPopupHtml(province) {
        const riskLabel = province.risk_level === 'high' ? 'Cao' : province.risk_level === 'medium' ? 'Trung bình' : 'Thấp';
        return `<div style="min-width:210px">
            <b style="color:#002147">${escapeHtml(province.province_name || 'Tỉnh/Thành phố')}</b>
            <div style="margin-top:6px;display:grid;gap:4px;font-size:12px">
                <span>GDP: <b>${Number(province.gdp_billion_vnd || 0).toLocaleString('vi-VN')} tỷ VND</b></span>
                <span>Thu thuế: <b>${Number(province.tax_revenue_billion_vnd || 0).toLocaleString('vi-VN')} tỷ VND</b></span>
                <span>Tuân thủ: <b>${(Number(province.compliance_rate || 0) * 100).toFixed(1)}%</b></span>
                <span>Rủi ro: <b>${riskLabel}</b></span>
            </div>
        </div>`;
    }

    function refreshLeafletStyles() {
        if (!leafletLayer) return;
        leafletLayer.setStyle(leafletFeatureStyle);
    }

    function renderMap() {
        if (!chartInstance) return;

        // Build map data from provinces
        const mapData = provincesData.map((p) => {
            const geoName = resolveGeoName(p.province_name || p.name || '');
            return {
                name: geoName,
                value: getRiskValue(p.risk_level),
                provinceCode: p.province_code || p.code,
                provinceName: p.province_name || p.name,
                riskLevel: p.risk_level || 'unknown',
                gdp: p.gdp_billion_vnd || 0,
                taxRevenue: p.tax_revenue_billion_vnd || 0,
                compliance: p.compliance_rate || 0,
            };
        });

        const maxRisk = Math.max(...mapData.map(d => d.value), 80);

        chartInstance.setOption({
            title: {
                text: 'BẢN ĐỒ KỊCH BẢN KINH TẾ VIỆT NAM',
                left: 'center',
                top: 8,
                textStyle: { color: '#002147', fontSize: 15, fontWeight: 900, letterSpacing: 2 },
            },
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(0,33,71,0.92)',
                borderColor: '#465f88',
                textStyle: { color: '#fff', fontSize: 12 },
                formatter: (params) => {
                    if (!params.data || params.data.value === undefined) {
                        return `<b>${escapeHtml(params.name)}</b><br/>Không có dữ liệu`;
                    }
                    const d = params.data;
                    const riskLabel = d.riskLevel === 'high' ? 'CAO' : d.riskLevel === 'medium' ? 'TRUNG BÌNH' : 'THẤP';
                    const riskColor = d.riskLevel === 'high' ? '#ff6b6b' : d.riskLevel === 'medium' ? '#ffd93d' : '#6bcb77';
                    return `<div style="min-width:200px">
                        <b style="font-size:13px">${escapeHtml(d.provinceName || params.name)}</b>
                        <hr style="border-color:rgba(255,255,255,.2);margin:6px 0">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span>GDP</span>
                            <b>${Number(d.gdp).toLocaleString('vi-VN')} tỷ VND</b>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span>Thu thuế</span>
                            <b>${Number(d.taxRevenue).toLocaleString('vi-VN')} tỷ VND</b>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span>Rủi ro</span>
                            <b style="color:${riskColor}">${riskLabel}</b>
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
                name: 'Kịch bản Kinh tế',
                type: 'map',
                map: 'vietnam',
                roam: true,
                zoom: 1.2,
                center: [106.5, 16.5],
                scaleLimit: { min: 0.8, max: 5 },
                selectedMode: 'single',
                label: {
                    show: true,
                    fontSize: 7,
                    color: '#333',
                    formatter: (p) => {
                        const short = p.name.replace(' city', '').replace('Thành phố ', '');
                        return short.length > 8 ? short.substring(0, 7) + '…' : short;
                    }
                },
                emphasis: {
                    label: { show: true, fontSize: 12, fontWeight: 'bold', color: '#002147' },
                    itemStyle: {
                        areaColor: '#aec7f6',
                        shadowBlur: 20,
                        shadowColor: 'rgba(0,33,71,0.4)',
                        borderWidth: 2,
                        borderColor: '#002147'
                    }
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

    function setupClickHandler() {
        if (!chartInstance) return;
        chartInstance.on('click', 'series', (params) => {
            if (!params.data) return;
            const d = params.data;
            selectedProvinceCode = d.provinceCode;

            if (d.provinceCode && window.loadProvinceScenario) {
                window.loadProvinceScenario(d.provinceCode, d.provinceName || params.name);
            }
        });
    }

    function setupResize(container) {
        const observer = new ResizeObserver(() => {
            if (chartInstance) chartInstance.resize();
            if (leafletMap) leafletMap.invalidateSize();
        });
        observer.observe(container);
        window.addEventListener('resize', () => {
            if (chartInstance) chartInstance.resize();
            if (leafletMap) leafletMap.invalidateSize();
        });
    }

    async function fetchJsonSafe(url, timeoutMs = 10000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, { signal: controller.signal });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } finally {
            clearTimeout(timer);
        }
    }

    function updateProvinceRisk(provinceCode, newRiskLevel) {
        const prov = provincesByCode.get(String(provinceCode));
        if (!prov) return;
        prov.risk_level = newRiskLevel;
        if (leafletLayer) refreshLeafletStyles();
        else if (chartInstance) renderMap();
    }

    function simulateNationalRisk(params) {
        if ((!chartInstance && !leafletLayer) || provincesData.length === 0) return;

        const gdp_delta = Number(params.gdp_delta_pct) || 0;
        const rawCompDelta = Number(params.compliance_delta) || 0;
        const comp_delta = Math.abs(rawCompDelta) > 1 ? rawCompDelta / 100.0 : rawCompDelta;
        const unemp_delta = Number(params.unemployment_delta) || 0;

        provincesData.forEach(p => {
            const base_unemp = Number(p.unemployment_rate || 2.5);
            const base_comp = Number(p.compliance_rate || 0.85);

            const projected_unemp = Math.max(0, base_unemp + unemp_delta);
            const effective_comp = Math.min(1.0, Math.max(0.3, base_comp + comp_delta));

            let risk_score = 0.0;
            if (projected_unemp > 5.0) risk_score += 0.3;
            if (effective_comp < 0.75) risk_score += 0.3;
            if (gdp_delta < -5.0) risk_score += 0.2;

            if (risk_score >= 0.5) p.risk_level = "high";
            else if (risk_score >= 0.2) p.risk_level = "medium";
            else p.risk_level = "low";
        });

        if (leafletLayer) refreshLeafletStyles();
        else if (chartInstance) renderMap();
        if (window.Vietnam3DMap && typeof window.Vietnam3DMap.applyMacroParams === 'function') {
            window.Vietnam3DMap.applyMacroParams(params || {});
        }
    }

    function applyMacroParams(params) {
        simulateNationalRisk(params || {});
    }

    return { init, updateProvinceRisk, simulateNationalRisk, applyMacroParams };
})();

document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('vietnam-map');
    if (container) {
        VietnamMap.init('vietnam-map');
    }
});