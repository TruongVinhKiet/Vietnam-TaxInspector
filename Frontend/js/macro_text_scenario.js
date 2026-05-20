// macro_text_scenario.js - Natural-language macro scenario workflow.
(() => {
    let latestScenarioPayload = null;

    function apiBase() {
        return window.API_BASE || 'http://localhost:8000/api';
    }

    function el(id) {
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

    document.addEventListener('DOMContentLoaded', () => {
        const interpretBtn = el('interpret-text-scenario-btn');
        const rememberBtn = el('remember-text-scenario-btn');
        if (interpretBtn) interpretBtn.addEventListener('click', interpretScenario);
        if (rememberBtn) rememberBtn.addEventListener('click', rememberScenario);
        refreshMacroRetrainStatus();
    });

    async function refreshMacroRetrainStatus() {
        const status = el('text-scenario-status');
        try {
            const response = await fetch(`${apiBase()}/simulation/macro-retrain/status`);
            const data = await response.json();
            if (!response.ok || !status || status.textContent) return;
            const modelState = data.status === 'ready' ? 'sẵn sàng' : 'chưa train';
            const counts = data.latest_report?.source_counts || {};
            status.textContent = `Model ghi nhớ cục bộ: ${modelState}. Sự kiện chuẩn: ${counts.canonical_events || 0}, memory đã duyệt: ${counts.approved_text_memories || 0}.`;
        } catch (_) {
            // Best-effort status only; interpretation still works through memory/LLM/rules.
        }
    }

    async function interpretScenario() {
        const text = el('scenario-text-input')?.value?.trim();
        const status = el('text-scenario-status');
        const cards = el('text-scenario-cards');
        const rememberBtn = el('remember-text-scenario-btn');
        if (!text) {
            if (status) status.textContent = 'Vui lòng nhập kịch bản cần phân tích.';
            return;
        }
        if (status) status.textContent = 'Đang phân tích kịch bản bằng bộ nhớ + LLM fallback...';
        if (cards) cards.innerHTML = '';
        if (rememberBtn) rememberBtn.disabled = true;

        const horizon = Number.parseInt(el('province-horizon-years')?.value || '5', 10);
        const provinceCode = window.getCurrentProvinceCode ? window.getCurrentProvinceCode() : null;
        try {
            const response = await fetch(`${apiBase()}/simulation/text-scenario/interpret`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, province_code: provinceCode, horizon_years: horizon })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            latestScenarioPayload = data;
            applyScenarioToSliders(data);
            renderScenarioCards(data);
            if (status) {
                const source = data.source === 'memory' ? `bộ nhớ (${Math.round((data.memory_similarity || 0) * 100)}%)` : (data.llm_provider || data.source || 'fallback');
                status.textContent = `Đã phân tích từ ${source}. Cần người dùng duyệt trước khi ghi nhớ.`;
            }
            if (rememberBtn) rememberBtn.disabled = false;
        } catch (error) {
            if (status) status.textContent = `Không phân tích được kịch bản: ${error.message || error}`;
        }
    }

    function applyScenarioToSliders(data) {
        const params = data.macro_parameters || {};
        setSlider('gdp-delta-slider', 'gdp-delta-val', Number(params.gdp_delta_pct || 0), '%');
        setSlider('tax-delta-slider', 'tax-delta-val', Number(params.tax_rate_delta || 0) * 100, '%');
        setSlider('compliance-delta-slider', 'compliance-delta-val', Number(params.compliance_delta || 0) * 100, '%');
        if (window.VietnamMap && typeof window.VietnamMap.applyMacroParams === 'function') {
            window.VietnamMap.applyMacroParams(params);
        }
    }

    function setSlider(id, valueId, value, suffix) {
        const slider = el(id);
        const label = el(valueId);
        if (!slider) return;
        const clamped = Math.max(Number(slider.min || -999), Math.min(Number(slider.max || 999), value));
        slider.value = String(clamped);
        if (label) label.textContent = `${clamped > 0 ? '+' : ''}${Number(clamped.toFixed(2))}${suffix}`;
    }

    function renderScenarioCards(data) {
        const cards = el('text-scenario-cards');
        if (!cards) return;
        const events = Array.isArray(data.candidate_events) ? data.candidate_events : [];
        const params = data.macro_parameters || {};
        cards.innerHTML = `
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-sm font-black text-primary-container">${escapeHtml(data.scenario_title || 'Kịch bản vĩ mô')}</div>
                <div class="mt-1 text-xs text-slate-500">${escapeHtml(data.reasoning_brief || '')}</div>
                <div class="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                    <span>GDP: <b>${signed(params.gdp_delta_pct)}%</b></span>
                    <span>FDI: <b>${signed(params.fdi_delta_pct)}%</b></span>
                    <span>Thất nghiệp: <b>${signed(params.unemployment_delta)}pp</b></span>
                    <span>Tuân thủ: <b>${signed(Number(params.compliance_delta || 0) * 100)}pp</b></span>
                </div>
                <div class="mt-2 text-[11px] text-slate-400">Confidence ${(Number(data.confidence || 0) * 100).toFixed(1)}% • ${escapeHtml(data.event_type || 'unknown')} • ${escapeHtml(data.severity || 'medium')}</div>
            </div>
            ${events.map((event) => `
                <details class="rounded-lg border border-slate-200 bg-white p-3">
                    <summary class="cursor-pointer text-sm font-black text-primary-container">${escapeHtml(event.headline || 'Bản tin kịch bản')}</summary>
                    <p class="mt-2 text-xs leading-relaxed text-slate-600">${escapeHtml(event.summary || '')}</p>
                    <div class="mt-2 text-[11px] font-bold text-slate-400">Xác suất ${Math.round(Number(event.probability || 0) * 100)}% • ${escapeHtml(event.impact_level || 'medium')}</div>
                </details>
            `).join('')}
        `;
    }

    async function rememberScenario() {
        if (!latestScenarioPayload) return;
        const ratingRaw = prompt('Chấm điểm độ chính xác kịch bản (0-5). Chỉ điểm >=4 mới được học sâu.', '5');
        if (ratingRaw === null) return;
        const rating = Math.max(0, Math.min(5, Number(ratingRaw || 0)));
        const approved = confirm('Duyệt kịch bản này để hệ thống ghi nhớ cho lần sau?');
        const status = el('text-scenario-status');
        try {
            const response = await fetch(`${apiBase()}/simulation/text-scenario/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scenario_text: latestScenarioPayload.scenario_text || el('scenario-text-input')?.value || '',
                    parsed_payload: latestScenarioPayload,
                    rating,
                    approved,
                    notes: '',
                    reviewer: 'ui_user'
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            if (status) status.textContent = `Đã lưu đánh giá: ${data.review_status}. Bộ nhớ đã duyệt: ${data.memory_status?.approved || 0}. Chạy retrain để đưa feedback này vào model cục bộ.`;
        } catch (error) {
            if (status) status.textContent = `Không lưu được feedback: ${error.message || error}`;
        }
    }

    function signed(value) {
        const number = Number(value || 0);
        return `${number > 0 ? '+' : ''}${Number(number.toFixed(2))}`;
    }
})();
