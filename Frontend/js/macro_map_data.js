// macro_map_data.js - shared canonical data adapter for Vietnam macro maps.
// All map renderers must use this adapter so Leaflet/Three/ECharts stay on the
// same boundary version, province profiles, event coverage and scenario preview.

const MacroMapData = (() => {
    const DEFAULT_BOUNDARY_VERSION = 'vn_34_2025';
    let statePromise = null;
    let state = null;
    let boundaryVersion = window.MACRO_BOUNDARY_VERSION || DEFAULT_BOUNDARY_VERSION;
    let macroParams = { gdp_delta_pct: 0, compliance_delta: 0, unemployment_delta: 0, tax_rate_delta: 0 };

    function getApiBase() {
        return window.API_BASE || 'http://localhost:8000/api';
    }

    function getBoundaryVersion() {
        return window.MACRO_BOUNDARY_VERSION || boundaryVersion || DEFAULT_BOUNDARY_VERSION;
    }

    function setBoundaryVersion(nextBoundaryVersion) {
        const normalized = nextBoundaryVersion === 'vn_63_legacy' ? 'vn_63_legacy' : DEFAULT_BOUNDARY_VERSION;
        boundaryVersion = normalized;
        window.MACRO_BOUNDARY_VERSION = normalized;
        statePromise = null;
        state = null;
        return boundaryVersion;
    }

    async function loadState(options = {}) {
        const force = Boolean(options.force);
        boundaryVersion = options.boundaryVersion || getBoundaryVersion();
        if (statePromise && !force) return statePromise;

        statePromise = fetchJson(`${getApiBase()}/simulation/map-state?boundary_version=${encodeURIComponent(boundaryVersion)}&include_geojson=true`)
            .then((payload) => {
                state = normalizeState(payload);
                return state;
            })
            .catch(async (error) => {
                console.warn('[MacroMapData] map-state failed, falling back to legacy endpoints:', error);
                const [profilePayload, geojson] = await Promise.all([
                    fetchJson(`${getApiBase()}/simulation/provinces?boundary_version=${encodeURIComponent(boundaryVersion)}`),
                    fetchJson(`${getApiBase()}/simulation/geojson-vietnam?boundary_version=${encodeURIComponent(boundaryVersion)}`)
                ]);
                state = normalizeState({
                    boundary_version: boundaryVersion,
                    geojson,
                    geojson_metadata: geojson.metadata || {},
                    provinces: profilePayload.provinces || [],
                    data_quality: profilePayload.data_quality || {},
                    model_status: {},
                });
                return state;
            });
        return statePromise;
    }

    function normalizeState(payload) {
        const provinces = Array.isArray(payload.provinces) ? payload.provinces.map((province) => ({
            ...province,
            province_code: String(province.province_code || province.code || '').trim(),
            province_name: province.province_name || province.name || '',
            risk_score: numberOr(province.risk_score, riskScoreFromLevel(province.risk_level)),
        })) : [];
        const byCode = new Map();
        provinces.forEach((province) => {
            if (province.province_code) byCode.set(province.province_code, province);
        });

        const geojson = payload.geojson || { type: 'FeatureCollection', features: [] };
        const featureCodes = new Set();
        (geojson.features || []).forEach((feature) => {
            const props = feature.properties || {};
            const code = String(props.province_code || props.code || props.gso_code || '').trim();
            if (code) featureCodes.add(code);
        });

        return {
            boundaryVersion: payload.boundary_version || boundaryVersion,
            geojson,
            geojsonMetadata: payload.geojson_metadata || geojson.metadata || {},
            provinces,
            byCode,
            featureCodes,
            dataQuality: payload.data_quality || {},
            modelStatus: payload.model_status || {},
            generatedAt: payload.generated_at || null,
        };
    }

    async function fetchJson(url, timeoutMs = 15000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } finally {
            clearTimeout(timer);
        }
    }

    function provinceForFeature(feature) {
        const current = state;
        const props = feature?.properties || {};
        const code = String(props.province_code || props.code || props.gso_code || '').trim();
        if (code && current?.byCode?.has(code)) return previewProvince(current.byCode.get(code));
        const name = props.province_name || props.name || props.NAME_1 || props.Name || '';
        return {
            province_code: code,
            province_name: name,
            risk_level: 'unknown',
            risk_score: 5,
            gdp_billion_vnd: 0,
            tax_revenue_billion_vnd: 0,
            num_enterprises: 0,
            compliance_rate: 0,
        };
    }

    function applyMacroParams(params = {}) {
        macroParams = { ...macroParams, ...params };
        return previewProvinces();
    }

    function previewProvince(province, params = macroParams) {
        const baseRisk = numberOr(province.risk_score, riskScoreFromLevel(province.risk_level));
        const gdpDelta = numberOr(params.gdp_delta_pct, 0);
        const rawCompDelta = numberOr(params.compliance_delta, 0);
        const compDelta = Math.abs(rawCompDelta) > 1 ? rawCompDelta / 100.0 : rawCompDelta;
        const unempDelta = numberOr(params.unemployment_delta, 0);
        const rawTaxDelta = numberOr(params.tax_rate_delta, 0);
        const taxDelta = Math.abs(rawTaxDelta) > 1 ? rawTaxDelta / 100.0 : rawTaxDelta;

        let score = baseRisk;
        score += Math.max(0, -gdpDelta) * 2.4;
        score += Math.max(0, unempDelta) * 5.2;
        score += Math.max(0, -compDelta) * 85;
        score += Math.max(0, taxDelta) * 160;
        score -= Math.max(0, compDelta) * 40;
        score = clamp(score, 0, 100);

        return {
            ...province,
            risk_score: score,
            risk_level: riskLevelFromScore(score),
        };
    }

    function previewProvinces(params = macroParams) {
        if (!state) return [];
        return state.provinces.map((province) => previewProvince(province, params));
    }

    function setProvinceRisk(provinceCode, riskLevel) {
        if (!state?.byCode) return;
        const province = state.byCode.get(String(provinceCode));
        if (!province) return;
        province.risk_level = riskLevel;
        province.risk_score = riskScoreFromLevel(riskLevel);
    }

    function applyProvinceImpacts(impacts = []) {
        if (!state?.byCode || !Array.isArray(impacts)) return previewProvinces();
        impacts.forEach((impact) => {
            const code = String(impact.province_code || '').trim();
            const province = state.byCode.get(code);
            if (!province) return;
            if (impact.projected_risk) province.risk_level = impact.projected_risk;
            if (impact.risk_score != null) province.risk_score = numberOr(impact.risk_score, province.risk_score);
            else province.risk_score = riskScoreFromLevel(province.risk_level);
            province.delta_revenue_pct = numberOr(impact.delta_revenue_pct, province.delta_revenue_pct || 0);
            province.delta_gdp_pct = numberOr(impact.delta_gdp_pct, province.delta_gdp_pct || 0);
            province.projected_revenue = numberOr(impact.projected_revenue, province.projected_revenue || 0);
            province.projected_compliance = numberOr(impact.projected_compliance, province.projected_compliance || 0);
        });
        return previewProvinces();
    }

    function riskScoreFromLevel(level) {
        if (level === 'high') return 80;
        if (level === 'medium') return 45;
        if (level === 'low') return 15;
        return 5;
    }

    function riskLevelFromScore(score) {
        if (score >= 65) return 'high';
        if (score >= 35) return 'medium';
        return 'low';
    }

    function riskColor(score) {
        if (score >= 75) return '#dc2626';
        if (score >= 55) return '#f97316';
        if (score >= 35) return '#facc15';
        if (score >= 15) return '#86efac';
        return '#dcfce7';
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function numberOr(value, fallback) {
        const num = Number(value);
        return Number.isFinite(num) ? num : fallback;
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    return {
        loadState,
        getBoundaryVersion,
        setBoundaryVersion,
        provinceForFeature,
        previewProvince,
        previewProvinces,
        applyMacroParams,
        applyProvinceImpacts,
        setProvinceRisk,
        riskColor,
        riskScoreFromLevel,
        riskLevelFromScore,
        escapeHtml,
    };
})();

window.MacroMapData = MacroMapData;
