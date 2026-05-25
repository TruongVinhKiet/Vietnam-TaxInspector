// province_scenario.js - Province Scenario Panel Controller.
let currentProvinceCode = null;
let currentProvinceName = '';
let scenarioAbortController = null;
let latestProvinceContext = null;
let causalChartInstance = null;
let mcChartInstance = null;
let tornadoChartInstance = null;
let gaugeChartInstance = null;
let shapChartInstance = null;
let paretoChartInstance = null;
let bvarChartInstance = null;
let regimeChartInstance = null;
let chordChartInstance = null;

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
    window.dispatchEvent(new CustomEvent('macro:province-selected', {
        detail: { provinceCode: currentProvinceCode, provinceName: currentProvinceName },
    }));

    const emptyState = byId('province-empty-state');
    const runBtn = byId('run-province-scenario-btn');
    const metrics = byId('province-metrics');
    const narrative = byId('province-narrative');
    const causalContainer = byId('province-causal-container');

    const leftDetails = byId('province-details-left-section');
    const eventsCard = byId('province-events-sidebar-card');
    if (emptyState) emptyState.style.display = 'none';
    if (metrics) metrics.style.display = 'none';
    if (narrative) narrative.style.display = 'none';
    if (causalContainer) causalContainer.style.display = 'none';
    ['province-mc-container','province-tornado-container','province-gauge-container','province-spatial-container'].forEach(id => {
        const el = byId(id); if (el) el.style.display = 'none';
    });
    if (leftDetails) {
        leftDetails.style.display = 'block';
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 150);
    }
    if (eventsCard) eventsCard.style.display = 'block';
 
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
        
        if (p.province_code && String(p.province_code) !== String(currentProvinceCode)) {
            currentProvinceCode = p.province_code;
            currentProvinceName = p.province_name || currentProvinceName;
            window.dispatchEvent(new CustomEvent('macro:province-resolved-code', {
                detail: { provinceCode: p.province_code, provinceName: currentProvinceName }
            }));
        }

        const d = latestProvinceContext.demographics || {};
        const r = latestProvinceContext.economic_ratios || {};
        const sourceQuality = latestProvinceContext.source_quality || {};
        const selected = d.selected_projection || {};
        
        // Sector composition helper
        const sec = p.sector_composition_pct || { agriculture: 0, industry: 0, services: 0, tax_product: 0 };
        const taxB = p.tax_breakdown_billion_vnd || { tndn: 0, tncn: 0, gtgt: 0, ttdb: 0, khac: 0 };
        const totalTaxB = (taxB.tndn + taxB.tncn + taxB.gtgt + taxB.ttdb + taxB.khac) || 1;

        // Resilience Index
        const pci = p.pci_score_2024 || 60;
        const fdi = p.fdi_billion_usd || 0;
        const hhi = ((sec.agriculture / 100) ** 2) + ((sec.industry / 100) ** 2) + ((sec.services / 100) ** 2);
        
        const pciTerm = Math.min(1.0, pci / 80);
        const fdiTerm = Math.min(1.0, Math.log10(fdi + 1) / Math.log10(50));
        const hhiTerm = Math.max(0.0, 1.0 - hhi);
        
        const resilienceScore = (0.4 * pciTerm + 0.3 * fdiTerm + 0.3 * hhiTerm) * 100;

        container.innerHTML = `
            <!-- Basic demographics & ratios -->
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-xs font-black uppercase tracking-wider text-slate-500">Kinh tế & Dân số vĩ mô</div>
                <div class="mt-2 grid grid-cols-2 gap-2 text-xs">
                    <div><span class="text-slate-500">Dân số hiện tại</span><b class="block text-primary-container">${Number(d.population_current || p.population || 0).toLocaleString('vi-VN')}</b></div>
                    <div><span class="text-slate-500">Dự phóng ${horizon} năm</span><b class="block text-primary-container">${Number(selected.population || 0).toLocaleString('vi-VN')}</b></div>
                    <div><span class="text-slate-500">GDP/người</span><b class="block">${Number(r.gdp_per_capita_million_vnd || 0).toLocaleString('vi-VN')} triệu</b></div>
                    <div><span class="text-slate-500">Thuế/người</span><b class="block">${Number(r.tax_revenue_per_capita_million_vnd || 0).toLocaleString('vi-VN')} triệu</b></div>
                    <div><span class="text-slate-500">Sức chống chịu vĩ mô</span><b class="block text-emerald-600 font-bold">${resilienceScore.toFixed(1)}/100</b></div>
                    <div><span class="text-slate-500">Độ nhạy với cú sốc</span><b class="block text-amber-600 font-bold">${(100 - resilienceScore).toFixed(1)}/100</b></div>
                </div>
            </div>

            <!-- PCI & Foreign Trade -->
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-xs font-black uppercase tracking-wider text-slate-500">PCI & Ngoại thương 2024</div>
                <div class="mt-2 grid grid-cols-2 gap-2 text-xs">
                    <div><span class="text-slate-500">Điểm PCI 2024</span><b class="block text-emerald-600">${p.pci_score_2024 ? p.pci_score_2024.toFixed(2) : 'N/A'} điểm</b></div>
                    <div><span class="text-slate-500">FDI lũy kế</span><b class="block text-primary-container">${Number(p.fdi_billion_usd || 0).toFixed(2)} tỷ USD</b></div>
                    <div><span class="text-slate-500">Xuất khẩu</span><b class="block text-sky-600">${Number(p.export_billion_usd || 0).toLocaleString('vi-VN')} tỷ USD</b></div>
                    <div><span class="text-slate-500">Nhập khẩu</span><b class="block text-amber-600">${Number(p.import_billion_usd || 0).toLocaleString('vi-VN')} tỷ USD</b></div>
                </div>
            </div>

            <!-- Sector Composition Visual Bar -->
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-xs font-black uppercase tracking-wider text-slate-500 mb-2">Cơ cấu ngành kinh tế (GRDP)</div>
                <div class="flex h-3 w-full overflow-hidden rounded-full bg-slate-200">
                    <div style="width: ${sec.agriculture}%" class="bg-emerald-500" title="Nông nghiệp: ${sec.agriculture}%"></div>
                    <div style="width: ${sec.industry}%" class="bg-sky-500" title="Công nghiệp: ${sec.industry}%"></div>
                    <div style="width: ${sec.services}%" class="bg-amber-500" title="Dịch vụ: ${sec.services}%"></div>
                    <div style="width: ${sec.tax_product}%" class="bg-slate-400" title="Thuế sản phẩm: ${sec.tax_product}%"></div>
                </div>
                <div class="mt-2 grid grid-cols-4 gap-1 text-[9px] font-bold text-slate-600 text-center">
                    <div><span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1"></span>Nông nghiệp: ${sec.agriculture}%</div>
                    <div><span class="inline-block w-1.5 h-1.5 rounded-full bg-sky-500 mr-1"></span>CN-XD: ${sec.industry}%</div>
                    <div><span class="inline-block w-1.5 h-1.5 rounded-full bg-amber-500 mr-1"></span>Dịch vụ: ${sec.services}%</div>
                    <div><span class="inline-block w-1.5 h-1.5 rounded-full bg-slate-400 mr-1"></span>Thuế SP: ${sec.tax_product}%</div>
                </div>
            </div>

            <!-- Tax Breakdown -->
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-xs font-black uppercase tracking-wider text-slate-500 mb-2">Cơ cấu đóng góp sắc thuế</div>
                <div class="space-y-1.5 text-[11px]">
                    <div>
                        <div class="flex justify-between font-medium"><span>Thuế GTGT (VAT)</span><b>${Number(taxB.gtgt).toLocaleString('vi-VN')} tỷ (${Math.round(taxB.gtgt * 100 / totalTaxB)}%)</b></div>
                        <div class="h-1 w-full bg-slate-200 rounded-full overflow-hidden"><div class="h-full bg-primary-container" style="width: ${taxB.gtgt * 100 / totalTaxB}%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between font-medium"><span>Thuế TNDN (CIT)</span><b>${Number(taxB.tndn).toLocaleString('vi-VN')} tỷ (${Math.round(taxB.tndn * 100 / totalTaxB)}%)</b></div>
                        <div class="h-1 w-full bg-slate-200 rounded-full overflow-hidden"><div class="h-full bg-emerald-600" style="width: ${taxB.tndn * 100 / totalTaxB}%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between font-medium"><span>Thuế TNCN (PIT)</span><b>${Number(taxB.tncn).toLocaleString('vi-VN')} tỷ (${Math.round(taxB.tncn * 100 / totalTaxB)}%)</b></div>
                        <div class="h-1 w-full bg-slate-200 rounded-full overflow-hidden"><div class="h-full bg-sky-500" style="width: ${taxB.tncn * 100 / totalTaxB}%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between font-medium"><span>Thuế TTĐB (SCT)</span><b>${Number(taxB.ttdb).toLocaleString('vi-VN')} tỷ (${Math.round(taxB.ttdb * 100 / totalTaxB)}%)</b></div>
                        <div class="h-1 w-full bg-slate-200 rounded-full overflow-hidden"><div class="h-full bg-amber-500" style="width: ${taxB.ttdb * 100 / totalTaxB}%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between font-medium"><span>Thuế khác & Phí</span><b>${Number(taxB.khac).toLocaleString('vi-VN')} tỷ (${Math.round(taxB.khac * 100 / totalTaxB)}%)</b></div>
                        <div class="h-1 w-full bg-slate-200 rounded-full overflow-hidden"><div class="h-full bg-slate-400" style="width: ${taxB.khac * 100 / totalTaxB}%"></div></div>
                    </div>
                </div>
            </div>

            <!-- Data Source Quality Info -->
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3 text-[10px] font-semibold text-slate-500 md:col-span-2">
                Độ tin cậy dữ liệu: <span class="text-primary-container uppercase">${escapeHtml(sourceQuality.observed_level || 'estimated')}</span> • Phương pháp: <span class="text-primary-container">${escapeHtml(sourceQuality.method || 'direct')}</span>
            </div>
        `;
    } catch (error) {
        container.innerHTML = `<div class="rounded-lg border border-red-100 bg-red-50 p-3 text-xs text-red-600">Không tải được context tỉnh: ${escapeHtml(error.message || error)}</div>`;
    }
}

function renderProvinceEventCards(events = []) {
    const container = byId('province-event-cards');
    if (!container) return;
    const top = events.slice(0, 12);
    if (!top.length) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = top.map((event) => `
        <button type="button" class="w-full rounded-lg border border-slate-200 bg-white p-3 text-left hover:border-sky-300 hover:shadow-sm transition" data-event-key="${escapeHtml(event.event_key || '')}">
            <div class="text-sm font-black text-primary-container">${escapeHtml(event.event_name_vi || event.event_name || event.event_key || 'Sự kiện')}</div>
            <div class="mt-1 text-xs text-slate-500">${escapeHtml(event.description_vi || event.description || '')}</div>
            <div class="mt-2 flex gap-2 text-[10px] font-bold uppercase text-slate-400">
                <span>${escapeHtml(event.event_type || 'macro')}</span>
                <span>${escapeHtml(event.start_date || '')}</span>
                <span>score ${Number(event.relevance_score || 0).toFixed(2)}</span>
            </div>
        </button>
    `).join('');
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
    renderCausalChart(data.impact_drivers);
    renderMonteCarloChart(data.monte_carlo);
    renderTornadoChart(data.sensitivity_analysis);
    renderConfidenceGauge(data.confidence_score, data.monte_carlo);
    renderSpatialSpillover(data.spatial_analysis);
    renderShapChart(data.shap_analysis);
    renderParetoChart(data.pareto_analysis);
    renderBvarChart(data.bvar_analysis);
    renderRegimeChart(data.regime_analysis);
}

function renderCausalChart(drivers) {
    const container = byId('province-causal-container');
    const chartDiv = byId('province-causal-chart');
    if (!container || !chartDiv || typeof echarts === 'undefined') return;

    if (!drivers || !drivers.length) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    if (causalChartInstance) {
        try {
            causalChartInstance.dispose();
        } catch (_) {}
    }
    causalChartInstance = echarts.init(chartDiv);

    const sorted = [...drivers].reverse();
    const categories = sorted.map(d => d.factor);
    const values = sorted.map(d => d.delta_pct);

    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: function (params) {
                const p = params[0];
                return `${p.name}: <b>${p.value > 0 ? '+' : ''}${p.value.toFixed(2)}%</b>`;
            }
        },
        grid: {
            top: 10,
            bottom: 25,
            left: '3%',
            right: '8%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } },
            axisLabel: { fontSize: 9, color: '#64748b' }
        },
        yAxis: {
            type: 'category',
            data: categories,
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: {
                fontSize: 9,
                color: '#334155',
                width: 100,
                overflow: 'break'
            }
        },
        series: [
            {
                name: 'Tác động',
                type: 'bar',
                data: values,
                itemStyle: {
                    color: function (params) {
                        return params.value >= 0 ? '#10b981' : '#f43f5e';
                    },
                    borderRadius: 4
                },
                barWidth: 12
            }
        ]
    };

    causalChartInstance.setOption(option);
}

// ── Monte Carlo Distribution Histogram ──
function renderMonteCarloChart(mc) {
    const container = byId('province-mc-container');
    const chartDiv = byId('province-mc-chart');
    const badge = byId('mc-var-badge');
    const statsRow = byId('mc-stats-row');
    if (!container || !chartDiv || typeof echarts === 'undefined') return;

    if (!mc || mc.error || !mc.histogram) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    if (badge) badge.textContent = `VaR₅% = ${Number(mc.var_5pct).toLocaleString('vi-VN')} tỷ`;

    if (statsRow) {
        const p = mc.percentiles || {};
        statsRow.innerHTML = `
            <div class="rounded bg-white border border-slate-200 p-1.5 text-center">
                <span class="text-slate-400 block">P10</span>
                <b class="text-red-600">${Number(p.p10||0).toLocaleString('vi-VN')}</b>
            </div>
            <div class="rounded bg-white border border-slate-200 p-1.5 text-center">
                <span class="text-slate-400 block">Median</span>
                <b class="text-primary-container">${Number(p.p50||0).toLocaleString('vi-VN')}</b>
            </div>
            <div class="rounded bg-white border border-slate-200 p-1.5 text-center">
                <span class="text-slate-400 block">P90</span>
                <b class="text-emerald-600">${Number(p.p90||0).toLocaleString('vi-VN')}</b>
            </div>
            <div class="rounded bg-white border border-slate-200 p-1.5 text-center">
                <span class="text-slate-400 block">CVaR₅%</span>
                <b class="text-rose-600">${Number(mc.cvar_5pct||0).toLocaleString('vi-VN')}</b>
            </div>
        `;
    }

    if (mcChartInstance) { try { mcChartInstance.dispose(); } catch(_){} }
    mcChartInstance = echarts.init(chartDiv);

    const bins = mc.histogram;
    const xData = bins.map(b => ((b.bin_start + b.bin_end) / 2).toFixed(0));
    const yData = bins.map(b => b.count);
    const p10 = mc.percentiles?.p10 || 0;
    const p90 = mc.percentiles?.p90 || 0;

    mcChartInstance.setOption({
        tooltip: { trigger: 'axis', formatter: p => `Thu ngân sách: ${p[0].name} tỷ<br/>Tần suất: ${p[0].value} lần` },
        grid: { top: 20, bottom: 30, left: '3%', right: '3%', containLabel: true },
        xAxis: { type: 'category', data: xData, axisLabel: { fontSize: 8, color: '#64748b', rotate: 30 }, axisLine: { lineStyle: { color: '#e2e8f0' } } },
        yAxis: { type: 'value', axisLabel: { fontSize: 8, color: '#64748b' }, splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } } },
        series: [{
            type: 'bar',
            data: yData.map((v, i) => ({
                value: v,
                itemStyle: {
                    color: parseFloat(xData[i]) < p10 ? '#fda4af' : parseFloat(xData[i]) > p90 ? '#6ee7b7' : '#93c5fd',
                    borderRadius: [2, 2, 0, 0]
                }
            })),
            barWidth: '85%',
        }],
        markLine: { silent: true, data: [
            { xAxis: mc.percentiles?.p50?.toFixed(0), lineStyle: { color: '#002147', type: 'solid', width: 2 }, label: { formatter: 'Median', fontSize: 8 } }
        ] }
    });
}

// ── Tornado Sensitivity Chart ──
function renderTornadoChart(sensitivity) {
    const container = byId('province-tornado-container');
    const chartDiv = byId('province-tornado-chart');
    if (!container || !chartDiv || typeof echarts === 'undefined') return;

    if (!sensitivity || !Array.isArray(sensitivity) || !sensitivity.length) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    if (tornadoChartInstance) { try { tornadoChartInstance.dispose(); } catch(_){} }
    tornadoChartInstance = echarts.init(chartDiv);

    const sorted = [...sensitivity].reverse();
    const categories = sorted.map(s => s.factor);
    const baseRev = sorted[0]?.revenue_base || 0;
    const lowDeltas = sorted.map(s => Math.round(s.revenue_low - baseRev));
    const highDeltas = sorted.map(s => Math.round(s.revenue_high - baseRev));

    tornadoChartInstance.setOption({
        tooltip: {
            trigger: 'axis', axisPointer: { type: 'shadow' },
            formatter: params => {
                const idx = params[0].dataIndex;
                const s = sorted[idx];
                return `<b>${s.factor}</b><br/>Giảm: ${s.revenue_low.toLocaleString('vi-VN')} tỷ<br/>Tăng: ${s.revenue_high.toLocaleString('vi-VN')} tỷ<br/>Biên độ: ${s.spread.toLocaleString('vi-VN')} tỷ`;
            }
        },
        grid: { top: 10, bottom: 25, left: '3%', right: '5%', containLabel: true },
        xAxis: { type: 'value', axisLabel: { fontSize: 8, color: '#64748b' }, splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } } },
        yAxis: { type: 'category', data: categories, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { fontSize: 9, color: '#334155' } },
        series: [
            { name: 'Giảm', type: 'bar', stack: 'tornado', data: lowDeltas, itemStyle: { color: '#f43f5e', borderRadius: [4, 0, 0, 4] }, barWidth: 14 },
            { name: 'Tăng', type: 'bar', stack: 'tornado', data: highDeltas, itemStyle: { color: '#10b981', borderRadius: [0, 4, 4, 0] }, barWidth: 14 }
        ]
    });
}

// ── Confidence Gauge ──
function renderConfidenceGauge(confidence, mc) {
    const container = byId('province-gauge-container');
    const chartDiv = byId('province-gauge-chart');
    if (!container || !chartDiv || typeof echarts === 'undefined') return;

    const confPct = Math.round((confidence || 0) * 100);
    const cv = mc?.coefficient_of_variation || 0;
    const stabilityScore = Math.max(0, Math.min(100, Math.round((1 - Math.min(cv, 1)) * 100)));
    const overall = Math.round(confPct * 0.6 + stabilityScore * 0.4);

    container.style.display = 'block';
    if (gaugeChartInstance) { try { gaugeChartInstance.dispose(); } catch(_){} }
    gaugeChartInstance = echarts.init(chartDiv);

    gaugeChartInstance.setOption({
        series: [{
            type: 'gauge',
            startAngle: 200, endAngle: -20,
            min: 0, max: 100,
            splitNumber: 5,
            pointer: { length: '60%', width: 5, itemStyle: { color: '#002147' } },
            axisLine: {
                lineStyle: {
                    width: 18,
                    color: [[0.3, '#f43f5e'], [0.6, '#f59e0b'], [0.85, '#10b981'], [1, '#059669']]
                }
            },
            axisTick: { length: 4, lineStyle: { color: '#fff', width: 1 } },
            splitLine: { length: 10, lineStyle: { color: '#fff', width: 2 } },
            axisLabel: { fontSize: 8, color: '#64748b', distance: 22 },
            detail: {
                valueAnimation: true,
                formatter: val => `{score|${val}}{unit|/100}\n{label|Độ tin cậy tổng hợp}`,
                rich: {
                    score: { fontSize: 22, fontWeight: 'bold', color: '#002147' },
                    unit: { fontSize: 11, color: '#94a3b8', padding: [0, 0, 0, 2] },
                    label: { fontSize: 9, color: '#94a3b8', padding: [6, 0, 0, 0] }
                },
                offsetCenter: [0, '65%']
            },
            data: [{ value: overall }]
        }]
    });
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

let spatialChartInstance = null;

function renderSpatialSpillover(spatial) {
    const container = byId('province-spatial-container');
    const moranChartDiv = byId('province-moran-chart');
    const flowContainer = byId('province-spatial-flow');
    const moranBadge = byId('moran-index-val');
    
    if (!container || !moranChartDiv || !flowContainer || typeof echarts === 'undefined') return;

    if (!spatial || spatial.error || !spatial.spillover_effects || !spatial.spillover_effects.length) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    // 1. Update Moran's I Badge
    if (moranBadge) {
        moranBadge.textContent = `Moran's I = ${spatial.moran_i.toFixed(3)}`;
    }

    // 2. Render Moran Scatter Plot (ECharts Scatter Chart with Regression Line)
    if (spatialChartInstance) {
        try { spatialChartInstance.dispose(); } catch(_) {}
    }
    spatialChartInstance = echarts.init(moranChartDiv);

    const points = spatial.scatter_points || [];
    const seriesData = points.map(p => ({
        name: p.name,
        value: [p.x, p.y],
        itemStyle: {
            color: p.is_selected ? '#f43f5e' : '#3b82f6',
            borderColor: p.is_selected ? '#ffe4e6' : '#dbeafe',
            borderWidth: p.is_selected ? 4 : 2
        },
        symbolSize: p.is_selected ? 14 : 10
    }));

    // Simple linear regression line fit for plotting
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    const minX = Math.min(...xValues) - 0.2;
    const maxX = Math.max(...xValues) + 0.2;
    // Regression line: Y = Moran's I * X
    const regressionPoints = [
        [minX, minX * spatial.moran_i],
        [maxX, maxX * spatial.moran_i]
    ];

    const option = {
        grid: { top: 15, bottom: 25, left: '8%', right: '8%', containLabel: true },
        tooltip: {
            trigger: 'item',
            formatter: function (p) {
                if (p.seriesType === 'line') return 'Đường hồi quy Moran';
                return `<b>${p.data.name}</b><br/>Độ lệch cục bộ: ${p.data.value[0].toFixed(2)}<br/>Độ trễ Không gian: ${p.data.value[1].toFixed(2)}`;
            }
        },
        xAxis: {
            type: 'value',
            name: 'Độ lệch Cục bộ',
            nameLocation: 'middle',
            nameGap: 18,
            nameTextStyle: { fontSize: 8, color: '#94a3b8' },
            splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
            axisLabel: { fontSize: 8, color: '#64748b' }
        },
        yAxis: {
            type: 'value',
            name: 'Độ trễ Không gian (Lag)',
            nameLocation: 'middle',
            nameGap: 20,
            nameTextStyle: { fontSize: 8, color: '#94a3b8' },
            splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
            axisLabel: { fontSize: 8, color: '#64748b' }
        },
        series: [
            {
                type: 'scatter',
                data: seriesData,
                label: {
                    show: true,
                    position: 'top',
                    formatter: '{b}',
                    fontSize: 8,
                    color: '#475569'
                }
            },
            {
                type: 'line',
                data: regressionPoints,
                showSymbol: false,
                lineStyle: { type: 'dashed', color: '#10b981', width: 1.5 },
                silent: true
            }
        ]
    };
    spatialChartInstance.setOption(option);

    // 3. Render Spillover Flow List
    flowContainer.innerHTML = spatial.spillover_effects.map(effect => {
        const isPositive = effect.spillover_revenue_delta_pct >= 0;
        const colorClass = isPositive ? 'text-emerald-600 bg-emerald-50 border-emerald-100' : 'text-rose-600 bg-rose-50 border-rose-100';
        const icon = isPositive ? 'trending_up' : 'trending_down';
        
        return `
            <div class="flex items-center justify-between p-2 rounded-lg border border-slate-100 bg-white hover:shadow-sm transition-all">
                <div class="flex items-center gap-2">
                    <div class="w-1.5 h-7 rounded-full bg-slate-300"></div>
                    <div>
                        <span class="font-bold text-slate-700 text-xs">${effect.province_name}</span>
                        <span class="block text-[9px] text-slate-400 font-medium">Kênh: ${effect.transmission_channel}</span>
                    </div>
                </div>
                <div class="text-right">
                    <span class="inline-flex items-center gap-0.5 px-2 py-0.5 rounded text-[10px] font-extrabold border ${colorClass}">
                        <span class="material-symbols-outlined text-[10px]">${icon}</span>
                        ${isPositive ? '+' : ''}${effect.spillover_revenue_delta_pct.toFixed(2)}%
                    </span>
                    <span class="block text-[9px] text-slate-400 mt-0.5">Hệ số thích ứng PCI: ${effect.resilience_index}</span>
                </div>
            </div>
        `;
    }).join('');

    renderChordChart(spatial.chord_links);
}

window.addEventListener('macro:boundary-change', async (event) => {
    if (!currentProvinceCode) return;
    const newBoundary = event.detail?.boundaryVersion || 'vn_34_2025';
    let mappedCode = currentProvinceCode;
    let mappedName = currentProvinceName;

    try {
        if (newBoundary === 'vn_34_2025') {
            if (!String(currentProvinceCode).startsWith('VN34-')) {
                const data = await fetchJson(`${_psGetApiBase()}/simulation/provinces?boundary_version=vn_34_2025`, {}, 5000);
                const matched = (data.provinces || []).find(p => p.member_codes && p.member_codes.includes(String(currentProvinceCode)));
                if (matched) {
                    mappedCode = matched.province_code;
                    mappedName = matched.province_name;
                }
            }
        } else {
            if (String(currentProvinceCode).startsWith('VN34-')) {
                const data = await fetchJson(`${_psGetApiBase()}/simulation/provinces?boundary_version=vn_34_2025`, {}, 5000);
                const matched = (data.provinces || []).find(p => p.province_code === String(currentProvinceCode));
                if (matched && matched.member_codes && matched.member_codes.length) {
                    mappedCode = matched.member_codes[0];
                    const data63 = await fetchJson(`${_psGetApiBase()}/simulation/provinces?boundary_version=vn_63_legacy`, {}, 5000);
                    const legacyMatched = (data63.provinces || []).find(p => p.province_code === mappedCode);
                    if (legacyMatched) {
                        mappedName = legacyMatched.province_name;
                    }
                }
            }
        }
    } catch (e) {
        console.warn('[ProvinceScenario] Boundary mapping error:', e);
    }

    if (mappedCode) {
        loadProvinceScenario(mappedCode, mappedName);
    }
});


// ────────────────────────────────────────────────────────────
//  Advanced Macro Analytics Rendering Functions
// ────────────────────────────────────────────────────────────

function renderShapChart(shap) {
    const container = byId('province-shap-container');
    const chartDiv = byId('province-shap-chart');
    if (!container || !chartDiv || typeof echarts === 'undefined') return;

    if (!shap || shap.error || !shap.shap_values) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    if (shapChartInstance) {
        try { shapChartInstance.dispose(); } catch(_) {}
    }
    shapChartInstance = echarts.init(chartDiv);

    const target = 'delta_revenue_pct';
    const rawValues = shap.shap_values[target] || [];
    const labels = shap.feature_labels || [];
    
    const dataList = labels.map((label, idx) => ({
        label: label,
        value: rawValues[idx] || 0.0
    }));
    dataList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    
    const topData = dataList.slice(0, 8).reverse();
    
    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: function(params) {
                const item = params[0];
                const colorText = item.value >= 0 ? '#10b981' : '#ef4444';
                return `<div class="font-bold text-xs">${item.name}</div>
                        <div class="text-xs" style="color: ${colorText}">Đóng góp: ${item.value >= 0 ? '+' : ''}${item.value.toFixed(4)} %</div>`;
            }
        },
        grid: { left: '3%', right: '8%', bottom: '3%', top: '5%', containLabel: true },
        xAxis: {
            type: 'value',
            splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
            axisLabel: { fontSize: 10, color: '#64748b' }
        },
        yAxis: {
            type: 'category',
            data: topData.map(d => d.label),
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { fontSize: 10, color: '#475569', fontWeight: 'bold' }
        },
        series: [{
            name: 'SHAP Value',
            type: 'bar',
            data: topData.map(d => ({
                value: d.value,
                itemStyle: {
                    color: d.value >= 0 ? '#10b981' : '#ef4444',
                    borderRadius: 4
                }
            })),
            label: {
                show: true,
                position: 'right',
                formatter: function(p) {
                    return (p.value >= 0 ? '+' : '') + p.value.toFixed(3);
                },
                fontSize: 9,
                color: '#64748b'
            }
        }]
    };
    shapChartInstance.setOption(option);
}

function renderParetoChart(pareto) {
    const container = byId('province-pareto-container');
    const chartDiv = byId('province-pareto-chart');
    const statsRow = byId('pareto-stats-row');
    if (!container || !chartDiv || typeof echarts === 'undefined') return;

    if (!pareto || pareto.error || !pareto.all_points) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    if (paretoChartInstance) {
        try { paretoChartInstance.dispose(); } catch(_) {}
    }
    paretoChartInstance = echarts.init(chartDiv);

    const allPoints = pareto.all_points || [];
    const frontier = pareto.pareto_frontier || [];
    const optimal = pareto.optimal_point || {};

    const scatterData = allPoints.map(p => [p.risk_score, p.revenue_delta_pct, p]);
    const frontierData = frontier.map(p => [p.risk_score, p.revenue_delta_pct, p]);
    const optimalData = optimal ? [[optimal.risk_score, optimal.revenue_delta_pct, optimal]] : [];

    const option = {
        tooltip: {
            trigger: 'item',
            formatter: function(params) {
                const p = params.data[2];
                return `<div class="font-bold text-xs mb-1">Chính sách Thuế & Tuân thủ</div>
                        <div class="text-xs">Độ lệch Thuế suất: <b>${(p.tax_rate_delta * 100).toFixed(1)}%</b></div>
                        <div class="text-xs">Độ lệch Kiểm soát: <b>${(p.compliance_delta * 100).toFixed(1)}%</b></div>
                        <div class="text-xs mt-1 border-t border-slate-100 pt-1 font-semibold text-emerald-600">Tăng trưởng DT: ${p.revenue_delta_pct.toFixed(2)}%</div>
                        <div class="text-xs font-semibold text-rose-500">Chỉ số rủi ro: ${p.risk_score.toFixed(4)}</div>`;
            }
        },
        legend: {
            data: ['Phương án khảo sát', 'Đường biên Pareto', 'Điểm tối ưu đề xuất'],
            fontSize: 10,
            bottom: 0
        },
        grid: { left: '3%', right: '5%', bottom: '15%', top: '5%', containLabel: true },
        xAxis: {
            name: 'Rủi ro',
            type: 'value',
            min: 0,
            max: 1.0,
            splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
            axisLabel: { fontSize: 10, color: '#64748b' }
        },
        yAxis: {
            name: 'Doanh thu (%)',
            type: 'value',
            splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
            axisLabel: { fontSize: 10, color: '#64748b' }
        },
        series: [
            {
                name: 'Phương án khảo sát',
                type: 'scatter',
                data: scatterData,
                symbolSize: 6,
                itemStyle: { color: '#cbd5e1', opacity: 0.6 }
            },
            {
                name: 'Đường biên Pareto',
                type: 'line',
                data: frontierData,
                symbolSize: 8,
                itemStyle: { color: '#8b5cf6' },
                lineStyle: { width: 2, type: 'dashed' }
            },
            {
                name: 'Điểm tối ưu đề xuất',
                type: 'scatter',
                data: optimalData,
                symbolSize: 14,
                itemStyle: { color: '#10b981', borderColor: '#fff', borderWidth: 2 },
                label: {
                    show: true,
                    formatter: 'Nash / Compromise',
                    position: 'top',
                    fontSize: 9,
                    color: '#047857',
                    fontWeight: 'bold'
                }
            }
        ]
    };
    paretoChartInstance.setOption(option);

    if (statsRow && optimal) {
        statsRow.innerHTML = `
            <div>
                <span class="text-slate-400 block text-[9px] uppercase font-bold">Thuế suất Tối ưu</span>
                <span class="font-extrabold text-slate-800 text-[13px]">${(optimal.tax_rate_delta >= 0 ? '+' : '')}${(optimal.tax_rate_delta * 100).toFixed(2)}%</span>
            </div>
            <div>
                <span class="text-slate-400 block text-[9px] uppercase font-bold">Nỗ lực Tuân thủ</span>
                <span class="font-extrabold text-slate-800 text-[13px]">${(optimal.compliance_delta >= 0 ? '+' : '')}${(optimal.compliance_delta * 100).toFixed(2)}%</span>
            </div>
            <div>
                <span class="text-slate-400 block text-[9px] uppercase font-bold">Hiệu quả dự phóng</span>
                <span class="font-extrabold text-emerald-600 text-[13px]">+${optimal.revenue_delta_pct.toFixed(2)}% DT / ${optimal.risk_score.toFixed(3)} Rủi ro</span>
            </div>
        `;
    }
}

function renderBvarChart(bvar) {
    const container = byId('province-bvar-container');
    const chartDiv = byId('province-bvar-chart');
    if (!container || !chartDiv || typeof echarts === 'undefined') return;

    if (!bvar || bvar.error || !bvar.irf_data) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    if (bvarChartInstance) {
        try { bvarChartInstance.dispose(); } catch(_) {}
    }
    bvarChartInstance = echarts.init(chartDiv);

    const quarters = bvar.quarters || [];
    const irfData = bvar.irf_data;
    const lowerGdp = irfData.gdp_lower || [];
    const upperGdp = irfData.gdp_upper || [];
    
    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                let res = `<div class="font-bold text-xs mb-1">${params[0].axisValue} (Sau cú sốc GDP)</div>`;
                params.forEach(p => {
                    if (p.seriesName.includes('Band')) return;
                    res += `<div class="text-xs flex items-center gap-1.5">
                                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background-color:${p.color}"></span>
                                ${p.seriesName}: <b>${p.value >= 0 ? '+' : ''}${p.value.toFixed(3)}%</b>
                            </div>`;
                });
                return res;
            }
        },
        legend: {
            data: ['GRDP Phản ứng', 'Thu ngân sách Phản ứng', 'Tỷ lệ Thất nghiệp Phản ứng'],
            bottom: 0,
            fontSize: 10
        },
        grid: { left: '3%', right: '5%', bottom: '15%', top: '8%', containLabel: true },
        xAxis: {
            type: 'category',
            data: quarters,
            axisLabel: { fontSize: 10, color: '#64748b' }
        },
        yAxis: {
            type: 'value',
            name: 'Phản ứng (%)',
            splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
            axisLabel: { fontSize: 10, color: '#64748b' }
        },
        series: [
            {
                name: 'GRDP Phản ứng',
                type: 'line',
                data: irfData.gdp,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: { color: '#3b82f6' },
                lineStyle: { width: 3 },
                zIndex: 10
            },
            {
                name: 'GRDP 95% CI Band',
                type: 'line',
                data: upperGdp,
                lineStyle: { opacity: 0 },
                stack: 'gdp-stack',
                symbol: 'none'
            },
            {
                name: 'GRDP 95% CI Band Lower',
                type: 'line',
                data: lowerGdp.map((l, idx) => upperGdp[idx] - l),
                lineStyle: { opacity: 0 },
                stack: 'gdp-stack',
                symbol: 'none',
                areaStyle: {
                    color: '#3b82f6',
                    opacity: 0.15
                }
            },
            {
                name: 'Thu ngân sách Phản ứng',
                type: 'line',
                data: irfData.revenue,
                symbol: 'rect',
                symbolSize: 6,
                itemStyle: { color: '#10b981' },
                lineStyle: { width: 3 },
                zIndex: 9
            },
            {
                name: 'Tỷ lệ Thất nghiệp Phản ứng',
                type: 'line',
                data: irfData.unemployment,
                symbol: 'triangle',
                symbolSize: 6,
                itemStyle: { color: '#ef4444' },
                lineStyle: { width: 2, type: 'dotted' },
                zIndex: 8
            }
        ]
    };
    bvarChartInstance.setOption(option);
}

function renderRegimeChart(regime) {
    const container = byId('province-regime-container');
    const chartDiv = byId('province-regime-chart');
    const badge = byId('regime-status-badge');
    if (!container || !chartDiv || typeof echarts === 'undefined') return;

    if (!regime || regime.error || !regime.smoothed_probabilities) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    if (regimeChartInstance) {
        try { regimeChartInstance.dispose(); } catch(_) {}
    }
    regimeChartInstance = echarts.init(chartDiv);

    const probs = regime.smoothed_probabilities || [];
    const years = probs.map(p => p.year);
    const growthRates = probs.map(p => p.growth_rate);
    const growthProbs = probs.map(p => p.growth_probability);
    const recessionProbs = probs.map(p => p.recession_probability);

    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const year = params[0].axisValue;
                const gRate = params[0].value;
                const gProb = params[1].value;
                const rProb = params[2].value;
                return `<div class="font-bold text-xs mb-1">Năm ${year}</div>
                        <div class="text-xs">Tốc độ tăng trưởng: <b>${gRate.toFixed(2)}%</b></div>
                        <div class="text-xs text-blue-500 font-semibold mt-1">Xác suất Tăng trưởng: ${(gProb * 100).toFixed(1)}%</div>
                        <div class="text-xs text-rose-500 font-semibold">Xác suất Suy thoái: ${(rProb * 100).toFixed(1)}%</div>`;
            }
        },
        legend: {
            data: ['GRDP Tăng trưởng (%)', 'Xác suất Regime Tăng trưởng', 'Xác suất Regime Suy thoái'],
            bottom: 0,
            fontSize: 10
        },
        grid: { left: '3%', right: '5%', bottom: '15%', top: '8%', containLabel: true },
        xAxis: {
            type: 'category',
            data: years,
            axisLabel: { fontSize: 10, color: '#64748b' }
        },
        yAxis: [
            {
                type: 'value',
                name: 'Tăng trưởng (%)',
                splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
                axisLabel: { fontSize: 10, color: '#64748b' }
            },
            {
                type: 'value',
                name: 'Xác suất',
                min: 0,
                max: 1.0,
                splitLine: { show: false },
                axisLabel: { fontSize: 10, color: '#64748b' }
            }
        ],
        series: [
            {
                name: 'GRDP Tăng trưởng (%)',
                type: 'bar',
                data: growthRates,
                itemStyle: { color: '#cbd5e1' },
                barWidth: '40%'
            },
            {
                name: 'Xác suất Regime Tăng trưởng',
                type: 'line',
                yAxisIndex: 1,
                data: growthProbs,
                smooth: true,
                symbol: 'circle',
                itemStyle: { color: '#3b82f6' },
                lineStyle: { width: 3 },
                areaStyle: { color: '#3b82f6', opacity: 0.1 }
            },
            {
                name: 'Xác suất Regime Suy thoái',
                type: 'line',
                yAxisIndex: 1,
                data: recessionProbs,
                smooth: true,
                symbol: 'circle',
                itemStyle: { color: '#ef4444' },
                lineStyle: { width: 2, type: 'dashed' }
            }
        ]
    };
    regimeChartInstance.setOption(option);

    if (badge) {
        const curRegime = regime.current_regime || "Growth";
        if (curRegime.includes("Growth") || curRegime.toLowerCase() === 'growth') {
            badge.innerText = "Chu kỳ Tăng trưởng";
            badge.className = "text-[9px] font-black px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-100 uppercase tracking-wider";
        } else {
            badge.innerText = "Chu kỳ Suy thoái";
            badge.className = "text-[9px] font-black px-2.5 py-0.5 rounded-full bg-rose-50 text-rose-600 border border-rose-100 uppercase tracking-wider";
        }
    }
}

function renderChordChart(chordLinks) {
    const chartDiv = byId('province-chord-chart');
    if (!chartDiv || typeof echarts === 'undefined') return;

    if (!chordLinks || !chordLinks.length) {
        chartDiv.innerHTML = '<div class="text-slate-400 text-xs flex items-center justify-center h-full">Không có dữ liệu liên kết</div>';
        return;
    }

    if (chordChartInstance) {
        try { chordChartInstance.dispose(); } catch(_) {}
    }
    chordChartInstance = echarts.init(chartDiv);

    const nodesMap = {};
    const links = [];

    chordLinks.forEach(link => {
        nodesMap[link.source] = true;
        nodesMap[link.target] = true;
        links.push({
            source: link.source,
            target: link.target,
            value: link.value,
            lineStyle: {
                width: Math.max(1, Math.min(8, link.value / 25)),
                curveness: 0.3,
                opacity: 0.6
            }
        });
    });

    const nodes = Object.keys(nodesMap).map((name, idx) => ({
        name: name,
        id: name,
        symbolSize: idx === 0 ? 18 : 10,
        itemStyle: {
            color: idx === 0 ? '#3b82f6' : '#10b981'
        },
        label: {
            show: true,
            position: 'right',
            fontSize: 9,
            color: '#475569',
            fontWeight: 'bold'
        }
    }));

    const option = {
        tooltip: {
            trigger: 'item',
            formatter: function(params) {
                if (params.dataType === 'edge') {
                    return `<div class="text-xs">Luồng liên kết vĩ mô:<br/><b>${params.data.source}</b> → <b>${params.data.target}</b>: ${params.data.value.toFixed(1)}</div>`;
                }
                return `<div class="text-xs font-bold">${params.name}</div>`;
            }
        },
        series: [{
            type: 'graph',
            layout: 'circular',
            circular: {
                rotateLabel: true
            },
            data: nodes,
            links: links,
            roam: false,
            label: {
                position: 'right',
                formatter: '{b}'
            },
            lineStyle: {
                color: 'source',
                curveness: 0.3
            }
        }]
    };
    chordChartInstance.setOption(option);
}
