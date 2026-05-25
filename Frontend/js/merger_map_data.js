// merger_map_data.js - boundary mode + merger-analysis API adapter.

const MergerMapData = (() => {
    const MODE_TO_BOUNDARY = {
        '34': 'vn_34_2025',
        '63': 'vn_63_legacy',
    };
    let currentMode = (window.MACRO_BOUNDARY_VERSION === 'vn_63_legacy') ? '63' : '34';
    const analysisCache = new Map();

    function apiBase() {
        return window.API_BASE || 'http://localhost:8000/api';
    }

    function getMode() {
        return currentMode;
    }

    function getBoundaryVersion() {
        return MODE_TO_BOUNDARY[currentMode] || MODE_TO_BOUNDARY['34'];
    }

    async function setMode(mode) {
        currentMode = mode === '63' ? '63' : '34';
        const boundaryVersion = getBoundaryVersion();
        window.MACRO_BOUNDARY_VERSION = boundaryVersion;

        // Fade out maps before reloading
        ['vietnam-map', 'vietnam-3d-map', 'vietnam-echarts-map'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.add('map-fade-out');
        });

        window.MacroMapData?.setBoundaryVersion?.(boundaryVersion);
        updateModeUi();
        window.dispatchEvent(new CustomEvent('macro:boundary-change', {
            detail: { mode: currentMode, boundaryVersion },
        }));

        // Fade in after data reload (give map JS time to re-render)
        setTimeout(() => {
            ['vietnam-map', 'vietnam-3d-map', 'vietnam-echarts-map'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('map-fade-out');
            });
        }, 400);

        return boundaryVersion;
    }

    async function fetchJson(url, timeoutMs = 15000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
            const text = await response.text();
            const data = text ? JSON.parse(text) : {};
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            return data;
        } finally {
            clearTimeout(timer);
        }
    }

    async function getMergerAnalysis(provinceCode, options = {}) {
        const code = String(provinceCode || '').trim();
        if (!code) return null;
        const boundaryVersion = options.boundaryVersion || getBoundaryVersion();
        const cacheKey = `${boundaryVersion}:${code}`;
        if (!options.force && analysisCache.has(cacheKey)) return analysisCache.get(cacheKey);
        const payload = await fetchJson(`${apiBase()}/simulation/merger-analysis/${encodeURIComponent(code)}?boundary_version=${encodeURIComponent(boundaryVersion)}`);
        analysisCache.set(cacheKey, payload);
        return payload;
    }

    function updateModeUi() {
        const title = document.getElementById('province-map-title');
        if (title) {
            title.textContent = currentMode === '63'
                ? 'Bản đồ Kịch bản Kinh tế 63 Tỉnh/Thành phố trước sáp nhập'
                : 'Bản đồ Kịch bản Kinh tế 34 Đơn vị Hành chính 2025';
        }
        document.querySelectorAll('[data-boundary-mode]').forEach((btn) => {
            const active = btn.dataset.boundaryMode === currentMode;
            btn.classList.toggle('bg-primary-container', active);
            btn.classList.toggle('text-white', active);
            btn.classList.toggle('bg-white', !active);
            btn.classList.toggle('text-slate-600', !active);
            btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        const overlay = document.getElementById('legacy-boundary-overlay-wrap');
        if (overlay) overlay.classList.toggle('hidden', currentMode !== '34');
    }

    function formatNumber(value, digits = 0) {
        const num = Number(value || 0);
        return num.toLocaleString('vi-VN', {
            maximumFractionDigits: digits,
            minimumFractionDigits: digits,
        });
    }

    function formatPercent(value) {
        if (value == null || Number.isNaN(Number(value))) return '--';
        const num = Number(value);
        return `${num > 0 ? '+' : ''}${num.toFixed(2)}%`;
    }

    document.addEventListener('DOMContentLoaded', () => {
        updateModeUi();
        document.querySelectorAll('[data-boundary-mode]').forEach((btn) => {
            btn.addEventListener('click', () => setMode(btn.dataset.boundaryMode));
        });
    });

    return {
        getMode,
        getBoundaryVersion,
        setMode,
        getMergerAnalysis,
        formatNumber,
        formatPercent,
    };
})();

window.MergerMapData = MergerMapData;
