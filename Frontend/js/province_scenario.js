// province_scenario.js - Province Scenario Panel Controller.
let currentProvinceCode = null;
let currentProvinceName = '';
let scenarioAbortController = null;
let latestProvinceContext = null;

const PROVINCE_SCENARIO_CACHE_PREFIX = 'taxinspector:province-scenario:';

function _psGetApiBase() {
    return window.API_BASE || 'http://localhost:8000/api';
}

function byId(id) {
    return document.getElementById(id);
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

async function fetchJson(url, options = {}, timeoutMs = 15000) {
    const externalSignal = options.signal;
    const controller = new AbortController();
    const abortFromExternal = () => controller.abort();
    if (externalSignal) {
        if (externalSignal.aborted) controller.abort();
        externalSignal.addEventListener('abort', abortFromExternal, { once: true });
    }
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        const text = await response.text();
        let data = {};
        if (text) {
            try {
                data = JSON.parse(text);
            } catch (error) {
                throw new Error('Phản hồi JSON không hợp lệ từ máy chủ.');
            }
        }
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        return data;
    } finally {
        clearTimeout(timer);
        if (externalSignal) externalSignal.removeEventListener('abort', abortFromExternal);
    }
}

async function loadProvinceScenario(provinceCode, provinceName) {
    // Guard: skip if province code is missing or undefined
    if (!provinceCode || provinceCode === 'undefined' || provinceCode === 'null') {
        console.warn('[ProvinceScenario] Skipped: invalid province code', provinceCode);
        return;
    }

    currentProvinceCode = provinceCode;
    currentProvinceName = provinceName || provinceCode;

    const emptyState = byId('province-empty-state');
    const runBtn = byId('run-province-scenario-btn');
    const metrics = byId('province-metrics');
    const narrative = byId('province-narrative');

    if (emptyState) emptyState.style.display = 'none';
    if (metrics) metrics.style.display = 'none';
    if (narrative) narrative.style.display = 'none';

    if (runBtn) {
        runBtn.disabled = false;
        runBtn.innerHTML = `<span class="material-symbols-outlined text-sm align-middle mr-1">play_arrow</span>Chạy kịch bản cho ${escapeHtml(currentProvinceName)}`;
    }

    const events = await refreshProvinceEvents(provinceCode);
    renderProvinceEventCards(events);
    await refreshProvinceContext(provinceCode);
}

window.loadProvinceScenario = loadProvinceScenario;
window.getCurrentProvinceCode = () => currentProvinceCode;

document.addEventListener('DOMContentLoaded', async () => {
    await refreshProvinceEvents(null);
    setupSliders();
    setupRunButton();
    const horizon = byId('province-horizon-years');
    if (horizon) {
        horizon.addEventListener('change', () => {
            if (currentProvinceCode) refreshProvinceContext(currentProvinceCode);
        });
    }
});

async function refreshProvinceEvents(provinceCode) {
    const selector = byId('event-selector');
    if (!selector) return;

    const previous = selector.value;
    selector.innerHTML = '<option value="">-- Không áp dụng sự kiện --</option>';
    selector.disabled = true;

    try {
        const suffix = provinceCode ? `?province_code=${encodeURIComponent(provinceCode)}&limit=120` : '?limit=160';
        const data = await fetchJson(`${_psGetApiBase()}/simulation/economic-events${suffix}`, {}, 12000);
        const events = Array.isArray(data.events) ? data.events : [];
        events.forEach((event) => {
            const option = document.createElement('option');
            option.value = event.event_key || '';
            const score = event.relevance_score ? `, relevance ${Number(event.relevance_score).toFixed(2)}` : '';
            option.textContent = `${event.event_name_vi || event.event_name || event.event_key} (${event.event_type || 'macro'}${score})`;
            selector.appendChild(option);
        });
        if (previous && [...selector.options].some((option) => option.value === previous)) {
            selector.value = previous;
        }
        return events;
    } catch (error) {
        console.error('Failed to load economic events', error);
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'Không tải được danh sách sự kiện';
        selector.appendChild(option);
        return [];
    } finally {
        selector.disabled = false;
    }
}

async function refreshProvinceContext(provinceCode) {
    const container = byId('province-context-cards');
    if (!container) return;
    // Guard against undefined
    if (!provinceCode || provinceCode === 'undefined') {
        container.innerHTML = '';
        return;
    }
    const horizon = Number.parseInt(byId('province-horizon-years')?.value || '10', 10);
    try {
        const boundaryVersion = window.MACRO_BOUNDARY_VERSION || 'vn_34_2025';
        latestProvinceContext = await fetchJson(`${_psGetApiBase()}/simulation/province-context/${encodeURIComponent(provinceCode)}?horizon_years=${horizon}&boundary_version=${encodeURIComponent(boundaryVersion)}`, {}, 12000);
        const p = latestProvinceContext.province || {};
        const d = latestProvinceContext.demographics || {};
        const r = latestProvinceContext.economic_ratios || {};
        const sourceQuality = latestProvinceContext.source_quality || {};
        const selected = d.selected_projection || {};
        container.innerHTML = `
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-xs font-black uppercase tracking-wider text-slate-500">Dữ liệu thực tế & dự phóng</div>
                <div class="mt-2 grid grid-cols-2 gap-2 text-xs">
                    <div><span class="text-slate-500">Dân số hiện tại</span><b class="block text-primary-container">${Number(d.population_current || p.population || 0).toLocaleString('vi-VN')}</b></div>
                    <div><span class="text-slate-500">Dự phóng ${horizon} năm</span><b class="block text-primary-container">${Number(selected.population || 0).toLocaleString('vi-VN')}</b></div>
                    <div><span class="text-slate-500">Tỷ lệ sinh</span><b class="block">${Number(d.birth_rate_per_1000 || 0).toFixed(1)}/1000</b></div>
                    <div><span class="text-slate-500">Tỷ lệ tử</span><b class="block">${Number(d.death_rate_per_1000 || 0).toFixed(1)}/1000</b></div>
                    <div><span class="text-slate-500">GDP/người</span><b class="block">${Number(r.gdp_per_capita_million_vnd || 0).toLocaleString('vi-VN')} triệu</b></div>
                    <div><span class="text-slate-500">Thuế/người</span><b class="block">${Number(r.tax_revenue_per_capita_million_vnd || 0).toLocaleString('vi-VN')} triệu</b></div>
                </div>
                <div class="mt-3 rounded border border-slate-200 bg-white px-2 py-1 text-[10px] font-semibold text-slate-500">
                    Nguồn: ${escapeHtml(sourceQuality.observed_level || 'national_observed_province_estimated')} • ${escapeHtml(sourceQuality.method || 'estimated')}
                </div>
            </div>
        `;
    } catch (error) {
        container.innerHTML = `<div class="rounded-lg border border-red-100 bg-red-50 p-3 text-xs text-red-600">Không tải được context tỉnh: ${escapeHtml(error.message || error)}</div>`;
    }
}

function renderProvinceEventCards(events = []) {
    const container = byId('province-event-cards');
    if (!container) return;
    const top = events.slice(0, 5);
    if (!top.length) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = `
        <div class="text-xs font-black uppercase tracking-wider text-slate-500">Bài báo/Sự kiện liên quan</div>
        ${top.map((event) => `
            <button type="button" class="w-full rounded-lg border border-slate-200 bg-white p-3 text-left hover:border-sky-300 hover:shadow-sm transition" data-event-key="${escapeHtml(event.event_key || '')}">
                <div class="text-sm font-black text-primary-container">${escapeHtml(event.event_name_vi || event.event_name || event.event_key || 'Sự kiện')}</div>
                <div class="mt-1 text-xs text-slate-500">${escapeHtml(event.description_vi || event.description || '')}</div>
                <div class="mt-2 flex gap-2 text-[10px] font-bold uppercase text-slate-400">
                    <span>${escapeHtml(event.event_type || 'macro')}</span>
                    <span>${escapeHtml(event.start_date || '')}</span>
                    <span>score ${Number(event.relevance_score || 0).toFixed(2)}</span>
                </div>
            </button>
        `).join('')}
    `;
    container.querySelectorAll('[data-event-key]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const selector = byId('event-selector');
            if (selector) selector.value = btn.dataset.eventKey || '';
        });
    });
}

function setupSliders() {
    const setupSlider = (id, valId, suffix = '') => {
        const el = byId(id);
        const valEl = byId(valId);
        if (!el || !valEl) return;
        const render = () => {
            const val = Number.parseFloat(el.value || '0');
            valEl.textContent = `${val > 0 ? '+' : ''}${val}${suffix}`;
            applyMapScenarioPreview();
        };
        el.addEventListener('input', render);
        render();
    };
    setupSlider('gdp-delta-slider', 'gdp-delta-val', '%');
    setupSlider('tax-delta-slider', 'tax-delta-val', '%');
    setupSlider('compliance-delta-slider', 'compliance-delta-val', '%');
}

function applyMapScenarioPreview() {
    if (!window.VietnamMap || typeof window.VietnamMap.applyMacroParams !== 'function') return;
    window.VietnamMap.applyMacroParams({
        gdp_delta_pct: Number.parseFloat(byId('gdp-delta-slider')?.value || '0'),
        tax_rate_delta: Number.parseFloat(byId('tax-delta-slider')?.value || '0') / 100.0,
        compliance_delta: Number.parseFloat(byId('compliance-delta-slider')?.value || '0') / 100.0,
        unemployment_delta: 0
    });
}

function setupRunButton() {
    const runBtn = byId('run-province-scenario-btn');
    if (!runBtn) return;
    runBtn.addEventListener('click', async () => {
        if (!currentProvinceCode) return;

        if (scenarioAbortController) scenarioAbortController.abort();
        scenarioAbortController = new AbortController();

        const originalText = runBtn.innerHTML;
        setRunButtonLoading(runBtn, true);
        showScenarioStatus('Đang tính toán mô phỏng và dựng bản tin kinh tế...', 'loading');

        const payload = readScenarioPayload();
        const cacheKey = `${PROVINCE_SCENARIO_CACHE_PREFIX}${JSON.stringify(payload)}`;

        try {
            const data = await fetchJson(`${_psGetApiBase()}/simulation/province-scenario`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: scenarioAbortController.signal
            }, 22000);
            sessionStorage.setItem(cacheKey, JSON.stringify({ data, cachedAt: Date.now() }));
            await renderScenarioResult(data);
        } catch (error) {
            if (error.name === 'AbortError') {
                showScenarioStatus('Đã hủy yêu cầu mô phỏng đang chạy.', 'cancelled');
                return;
            }
            console.error('Scenario execution failed', error);
            const cached = getCachedScenario(cacheKey);
            if (cached) {
                cached.cache_status_note = 'Máy chủ phản hồi chậm. Đây là kết quả đã lưu gần nhất cho cùng kịch bản.';
                await renderScenarioResult(cached);
            } else {
                showScenarioStatus(`Không chạy được kịch bản: ${error.message || error}`, 'error');
            }
        } finally {
            scenarioAbortController = null;
            setRunButtonLoading(runBtn, false, originalText);
        }
    });
}

function readScenarioPayload() {
    const macroUnemployment = Number.parseFloat(byId('sim-unemployment')?.value || '2.3');
    const horizon = Number.parseInt(byId('province-horizon-years')?.value || '5', 10);
    return {
        province_code: currentProvinceCode,
        boundary_version: window.MACRO_BOUNDARY_VERSION || 'vn_34_2025',
        event_key: byId('event-selector')?.value || null,
        gdp_delta_pct: Number.parseFloat(byId('gdp-delta-slider')?.value || '0'),
        tax_rate_delta: Number.parseFloat(byId('tax-delta-slider')?.value || '0') / 100.0,
        compliance_delta: Number.parseFloat(byId('compliance-delta-slider')?.value || '0') / 100.0,
        unemployment_delta: macroUnemployment - 2.3,
        fdi_delta_pct: 0,
        projection_years: horizon,
        use_llm: true
    };
}

function getCachedScenario(cacheKey) {
    try {
        const raw = sessionStorage.getItem(cacheKey);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed.cachedAt || Date.now() - parsed.cachedAt > 30 * 60 * 1000) return null;
        return parsed.data || null;
    } catch (_) {
        return null;
    }
}

function setRunButtonLoading(button, loading, originalText = null) {
    button.disabled = loading;
    if (loading) {
        button.innerHTML = '<span class="material-symbols-outlined text-sm align-middle mr-1 animate-spin">refresh</span>Đang tính toán...';
    } else if (originalText) {
        button.innerHTML = originalText;
    }
}

async function renderScenarioResult(data) {
    if (Array.isArray(data.province_impacts) && data.province_impacts.length && window.VietnamMap?.applyProvinceImpacts) {
        window.VietnamMap.applyProvinceImpacts(data.province_impacts);
    }

    const metrics = byId('province-metrics');
    if (metrics) metrics.style.display = 'grid';

    setMetric('metric-gdp', formatSignedPercent(data.delta_gdp_pct), data.delta_gdp_pct >= 0);
    setMetric('metric-revenue', formatSignedPercent(data.delta_revenue_pct), data.delta_revenue_pct >= 0);
    setRiskMetric(data.projected_risk);
    const compliance = Number(data.projected_compliance || 0) * 100;
    setMetric('metric-compliance', `${compliance.toFixed(1)}%`, compliance >= 80);

    const narrativeBox = byId('province-narrative');
    const narrativeContent = byId('narrative-content');
    if (!narrativeBox || !narrativeContent) return;
    narrativeBox.style.display = 'block';

    const enrichedNarrative = buildNarrativeWithMeta(data);
    await typewriterContent(narrativeContent, enrichedNarrative);
}

function setMetric(id, text, positive) {
    const el = byId(id);
    if (!el) return;
    el.textContent = text;
    el.className = `text-lg font-black ${positive ? 'text-green-600' : 'text-red-600'}`;
}

function setRiskMetric(risk) {
    const el = byId('metric-risk');
    if (!el) return;
    const value = String(risk || 'unknown').toUpperCase();
    el.textContent = value;
    el.className = `text-lg font-black ${
        risk === 'high' ? 'text-red-600' :
            risk === 'medium' ? 'text-amber-600' : 'text-green-600'
    }`;
}

function formatSignedPercent(value) {
    const number = Number(value || 0);
    return `${number > 0 ? '+' : ''}${number.toFixed(1)}%`;
}

function buildNarrativeWithMeta(data) {
    const cacheNote = data.cache_status_note ? `**Lưu ý:** ${data.cache_status_note}\n\n` : '';
    const confidence = data.confidence_score != null
        ? `\n\n**Độ tin cậy mô phỏng:** ${(Number(data.confidence_score) * 100).toFixed(1)}%.`
        : '';
    const band = data.uncertainty_band_revenue || {};
    const uncertainty = band.low != null && band.high != null
        ? `\n**Dải bất định thu ngân sách:** ${Number(band.low).toLocaleString('vi-VN')} - ${Number(band.high).toLocaleString('vi-VN')} tỷ VND.`
        : '';
    const drivers = Array.isArray(data.impact_drivers) && data.impact_drivers.length
        ? `\n\n**Yếu tố kéo kịch bản:**\n${data.impact_drivers.slice(0, 5).map((driver) => `- ${driver.factor}: ${driver.delta_pct > 0 ? '+' : ''}${driver.delta_pct}% (${driver.direction})`).join('\n')}`
        : '';
    return `${cacheNote}${data.narrative_text || ''}${confidence}${uncertainty}${drivers}`;
}

function markdownToSafeHtml(markdown) {
    return escapeHtml(markdown)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/^- (.*)$/gm, '<div class="ml-3 my-1">• $1</div>')
        .replace(/\n/g, '<br/>');
}

async function typewriterContent(element, text) {
    const safeText = String(text || '');
    element.innerHTML = '';
    element.style.opacity = '1';
    const chunkSize = safeText.length > 1200 ? 24 : 8;
    for (let i = 0; i < safeText.length; i += chunkSize) {
        element.textContent = safeText.slice(0, i + chunkSize);
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    element.innerHTML = markdownToSafeHtml(safeText);
}

function showScenarioStatus(message, tone = 'info') {
    const narrativeBox = byId('province-narrative');
    const narrativeContent = byId('narrative-content');
    if (!narrativeBox || !narrativeContent) return;
    narrativeBox.style.display = 'block';
    const toneClass = tone === 'error'
        ? 'text-red-600'
        : tone === 'warn'
            ? 'text-amber-700'
            : tone === 'cancelled'
                ? 'text-slate-500'
                : 'text-slate-600';
    narrativeContent.innerHTML = `<div class="${toneClass} font-semibold">${escapeHtml(message)}</div>`;
}
