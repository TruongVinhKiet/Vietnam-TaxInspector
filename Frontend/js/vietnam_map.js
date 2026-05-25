// vietnam_map.js - Leaflet 2D vector choropleth for the macro digital twin.
// Uses MacroMapData so all renderers share vn_34_2025 GeoJSON + province state.

const VietnamMap = (() => {
    let leafletMap = null;
    let leafletLayer = null;
    let legacyBoundaryLayer = null;
    let selectedProvinceCode = null;
    let containerId = 'vietnam-map';

    async function init(id = 'vietnam-map') {
        containerId = id;
        const container = document.getElementById(containerId);
        if (!container) return;

        if (typeof L === 'undefined') {
            showStatus(container, 'Không tải được thư viện Leaflet.');
            return;
        }
        if (!window.MacroMapData) {
            showStatus(container, 'Không tải được bộ dữ liệu bản đồ vĩ mô.');
            return;
        }

        showStatus(container, 'Đang tải ranh giới hành chính 34 đơn vị...');
        try {
            const mapState = await window.MacroMapData.loadState({ force: true });
            renderLeaflet(container, mapState);
        } catch (error) {
            console.error('[VietnamMap] init failed', error);
            showStatus(container, `Không tải được bản đồ: ${error.message || error}`);
        }
    }

    function renderLeaflet(container, mapState) {
        if (leafletMap) {
            leafletMap.remove();
            leafletMap = null;
            leafletLayer = null;
        }
        container.innerHTML = '';

        leafletMap = L.map(containerId, {
            zoomControl: true,
            scrollWheelZoom: true,
            doubleClickZoom: true,
            touchZoom: true,
            attributionControl: false,
            preferCanvas: true,
            zoomSnap: 0.25,
            zoomDelta: 0.5,
            worldCopyJump: false,
        });

        // CartoDB Positron tiles for world map background with neighboring countries
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            subdomains: 'abcd',
            maxZoom: 20
        }).addTo(leafletMap);

        leafletMap.createPane('province-polygons');
        leafletMap.getPane('province-polygons').style.zIndex = 420;

        leafletLayer = L.geoJSON(mapState.geojson, {
            pane: 'province-polygons',
            style: leafletFeatureStyle,
            onEachFeature: bindLeafletFeature,
        }).addTo(leafletMap);

        setupLegacyOverlayControl();

        const bounds = leafletLayer.getBounds();
        if (bounds.isValid()) {
            leafletMap.fitBounds(bounds, { padding: [18, 18], animate: false });
            leafletMap.setMaxBounds(bounds.pad(0.35));
        } else {
            leafletMap.setView([16.4, 106.2], 5.25);
        }

        setTimeout(onVisible, 80);
        setTimeout(onVisible, 350);
        window.addEventListener('resize', onVisible);
    }

    function leafletFeatureStyle(feature) {
        const province = window.MacroMapData.provinceForFeature(feature);
        const selected = selectedProvinceCode && String(province.province_code) === String(selectedProvinceCode);
        const layer = window.MapEnhancements?.getChoroplethLayer?.() || 'risk';
        return {
            color: selected ? '#002147' : '#ffffff',
            weight: selected ? 2.6 : 1,
            fillColor: choroplethColor(province, layer),
            fillOpacity: selected ? 0.92 : 0.78,
            opacity: 1,
        };
    }

    function choroplethColor(province, layer) {
        if (layer === 'gdp') {
            const gdp = Number(province.gdp_billion_vnd || 0);
            if (gdp >= 500000) return '#1e3a5f';
            if (gdp >= 200000) return '#2563eb';
            if (gdp >= 100000) return '#60a5fa';
            if (gdp >= 50000) return '#93c5fd';
            return '#dbeafe';
        }
        if (layer === 'tax_efficiency') {
            const gdp = Number(province.gdp_billion_vnd || 1);
            const tax = Number(province.tax_revenue_billion_vnd || 0);
            const eff = (tax / Math.max(gdp, 1)) * 100;
            if (eff >= 20) return '#065f46';
            if (eff >= 15) return '#059669';
            if (eff >= 10) return '#34d399';
            if (eff >= 5) return '#a7f3d0';
            return '#ecfdf5';
        }
        if (layer === 'compliance') {
            const c = Number(province.compliance_rate || 0);
            if (c >= 0.88) return '#1e40af';
            if (c >= 0.84) return '#3b82f6';
            if (c >= 0.80) return '#93c5fd';
            if (c >= 0.76) return '#fbbf24';
            return '#f87171';
        }
        return window.MacroMapData.riskColor(province.risk_score);
    }

    function bindLeafletFeature(feature, layer) {
        const province = window.MacroMapData.provinceForFeature(feature);
        const name = province.province_name || feature?.properties?.name || 'Tỉnh/Thành phố';

        // Simple tooltip on hover (no popup to avoid jitter)
        layer.bindTooltip(name, { sticky: true, direction: 'top', className: 'text-xs font-bold' });

        layer.on({
            mouseover: () => {
                layer.setStyle({ weight: 2.6, color: '#002147', fillOpacity: 0.95 });
                if (layer.bringToFront) layer.bringToFront();
            },
            mouseout: () => {
                if (leafletLayer) leafletLayer.resetStyle(layer);
            },
            click: (e) => {
                // Select province and trigger province scenario panel
                selectProvince(province, name);

                // Show rich popup at click location
                const latlng = e.latlng;
                if (latlng && !isNaN(latlng.lat) && !isNaN(latlng.lng)) {
                    L.popup({ maxWidth: 340, className: 'province-popup-rich' })
                        .setLatLng(latlng)
                        .setContent(leafletPopupHtml(province))
                        .openOn(leafletMap);
                }
            },
        });
    }

    function selectProvince(province, name) {
        if (!province?.province_code) return;
        selectedProvinceCode = province.province_code;
        refreshLeafletStyles();
        if (window.loadProvinceScenario) {
            window.loadProvinceScenario(province.province_code, name || province.province_name);
        }
        window.dispatchEvent(new CustomEvent('macro:province-selected', {
            detail: { provinceCode: province.province_code, provinceName: name || province.province_name },
        }));
    }

    function leafletPopupHtml(province) {
        const esc = window.MacroMapData.escapeHtml;
        const riskScore = Number(province.risk_score || 0);
        const riskColor = riskScore >= 65 ? '#ef4444' : riskScore >= 35 ? '#f59e0b' : '#22c55e';
        const riskLabel = riskScore >= 65 ? 'Cao' : riskScore >= 35 ? 'Trung bình' : 'Thấp';
        const gdp = Number(province.gdp_billion_vnd || 0);
        const tax = Number(province.tax_revenue_billion_vnd || 0);
        const enterprises = Number(province.num_enterprises || 0);
        const compliance = Number(province.compliance_rate || 0);
        const population = Number(province.population || 0);
        const taxEfficiency = gdp > 0 ? ((tax / gdp) * 100).toFixed(1) : '—';
        const gdpPerCapita = population > 0 ? (gdp * 1e9 / population / 1e6).toFixed(1) : '—';

        return `<div style="min-width:260px;font-family:Inter,system-ui,sans-serif">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <b style="color:#002147;font-size:14px">${esc(province.province_name || 'Tỉnh/Thành phố')}</b>
                <span style="margin-left:auto;background:${riskColor};color:#fff;font-size:10px;font-weight:800;padding:2px 8px;border-radius:20px">${riskLabel}</span>
            </div>
            <hr style="border-color:rgba(0,33,71,0.12);margin:0 0 8px">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;font-size:12px;color:#334155">
                <div style="display:flex;justify-content:space-between"><span>Điểm rủi ro</span><b style="color:${riskColor}">${riskScore.toFixed(1)}</b></div>
                <div style="display:flex;justify-content:space-between"><span>Doanh nghiệp</span><b>${enterprises.toLocaleString('vi-VN')}</b></div>
                <div style="display:flex;justify-content:space-between"><span>GDP/GRDP</span><b>${gdp.toLocaleString('vi-VN')} tỷ</b></div>
                <div style="display:flex;justify-content:space-between"><span>Thu thuế</span><b>${tax.toLocaleString('vi-VN')} tỷ</b></div>
                <div style="display:flex;justify-content:space-between"><span>Tuân thủ</span><b>${(compliance * 100).toFixed(1)}%</b></div>
                <div style="display:flex;justify-content:space-between"><span>Hiệu suất thuế</span><b>${taxEfficiency}%</b></div>
                ${population > 0 ? `<div style="display:flex;justify-content:space-between"><span>Dân số</span><b>${population.toLocaleString('vi-VN')}</b></div>` : ''}
                ${gdpPerCapita !== '—' ? `<div style="display:flex;justify-content:space-between"><span>GDP/người</span><b>${gdpPerCapita} tr</b></div>` : ''}
            </div>
            <div style="margin-top:8px;font-size:10px;color:#94a3b8;text-align:center">Click để mở kịch bản chi tiết</div>
        </div>`;
    }

    function refreshLeafletStyles() {
        if (!leafletLayer) return;
        leafletLayer.setStyle(leafletFeatureStyle);
    }

    async function setupLegacyOverlayControl() {
        const toggle = document.getElementById('legacy-boundary-overlay-toggle');
        if (!toggle || !leafletMap) return;
        toggle.onchange = () => toggleLegacyBoundaryOverlay(toggle.checked);
        if (window.MACRO_BOUNDARY_VERSION === 'vn_63_legacy') {
            toggle.checked = false;
            await toggleLegacyBoundaryOverlay(false);
        } else if (toggle.checked) {
            await toggleLegacyBoundaryOverlay(true);
        }
    }

    async function toggleLegacyBoundaryOverlay(enabled) {
        if (!leafletMap) return;
        if (legacyBoundaryLayer) {
            leafletMap.removeLayer(legacyBoundaryLayer);
            legacyBoundaryLayer = null;
        }
        if (!enabled || window.MACRO_BOUNDARY_VERSION === 'vn_63_legacy') return;
        try {
            const apiBase = window.API_BASE || 'http://localhost:8000/api';
            const response = await fetch(`${apiBase}/simulation/geojson-vietnam?boundary_version=vn_63_legacy`, { cache: 'no-store' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const geojson = await response.json();
            legacyBoundaryLayer = L.geoJSON(geojson, {
                interactive: false,
                style: {
                    color: '#0f172a',
                    weight: 1.2,
                    opacity: 0.45,
                    fillOpacity: 0,
                    dashArray: '4 4',
                },
            }).addTo(leafletMap);
        } catch (error) {
            console.warn('[VietnamMap] Could not load legacy boundary overlay:', error);
        }
    }

    function updateProvinceRisk(provinceCode, newRiskLevel) {
        window.MacroMapData?.setProvinceRisk(provinceCode, newRiskLevel);
        refreshLeafletStyles();
    }

    function simulateNationalRisk(params = {}) {
        if (!window.MacroMapData) return;
        window.MacroMapData.applyMacroParams(params);
        refreshLeafletStyles();
        if (window.Vietnam3DMap?.applyMacroParams) window.Vietnam3DMap.applyMacroParams(params);
        if (window.VietnamEChartsMap?.applyMacroParams) window.VietnamEChartsMap.applyMacroParams(params);
    }

    function applyMacroParams(params = {}) {
        simulateNationalRisk(params);
    }

    function applyProvinceImpacts(impacts = []) {
        window.MacroMapData?.applyProvinceImpacts(impacts);
        refreshLeafletStyles();
        if (window.Vietnam3DMap?.applyProvinceImpacts) window.Vietnam3DMap.applyProvinceImpacts(impacts);
        if (window.VietnamEChartsMap?.applyProvinceImpacts) window.VietnamEChartsMap.applyProvinceImpacts(impacts);
    }

    function onVisible() {
        if (!leafletMap) return;
        leafletMap.invalidateSize(false);
        if (leafletLayer) {
            const bounds = leafletLayer.getBounds();
            if (bounds.isValid()) leafletMap.fitBounds(bounds, { padding: [18, 18], animate: false });
        }
    }

    async function switchBoundary(boundaryVersion) {
        if (!window.MacroMapData) return;
        selectedProvinceCode = null;
        const normalized = window.MacroMapData.setBoundaryVersion(boundaryVersion);
        const container = document.getElementById(containerId);
        if (!container) return;
        showStatus(container, normalized === 'vn_63_legacy' ? 'Đang tải bản đồ 63 tỉnh/thành...' : 'Đang tải bản đồ 34 đơn vị...');
        const mapState = await window.MacroMapData.loadState({ boundaryVersion: normalized, force: true });
        renderLeaflet(container, mapState);
    }

    function showStatus(container, message) {
        container.innerHTML = `<div class="flex h-full min-h-[420px] items-center justify-center text-sm font-semibold text-slate-500 text-center px-6">${window.MacroMapData?.escapeHtml ? window.MacroMapData.escapeHtml(message) : message}</div>`;
    }

    function refreshChoropleth(layer) {
        refreshLeafletStyles();
    }

    window.addEventListener('macro:boundary-change', (event) => {
        const boundaryVersion = event.detail?.boundaryVersion || 'vn_34_2025';
        switchBoundary(boundaryVersion);
    });

    window.addEventListener('macro:province-selected', (event) => {
        const code = event.detail?.provinceCode;
        if (code && String(code) !== String(selectedProvinceCode)) {
            selectedProvinceCode = code;
            refreshLeafletStyles();
        }
    });

    window.addEventListener('macro:province-resolved-code', (event) => {
        const code = event.detail?.provinceCode;
        if (code && String(code) !== String(selectedProvinceCode)) {
            selectedProvinceCode = code;
            refreshLeafletStyles();
        }
    });

    return { init, updateProvinceRisk, simulateNationalRisk, applyMacroParams, applyProvinceImpacts, switchBoundary, refreshChoropleth, onVisible };
})();

window.VietnamMap = VietnamMap;

document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('vietnam-map');
    if (container) VietnamMap.init('vietnam-map');
});
