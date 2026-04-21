/**
 * simulation.js – Digital Twin Simulation Engine Frontend Logic
 * ==============================================================
 * Handles slider interactions, API calls, chart rendering, and
 * scenario comparison for the Tax Policy Simulation page.
 */

const SIM_API = `${API_BASE}/simulation`;

// ── State ──────────────────────────────────────────────────
let currentScenario = null;
let baselineData = null;
let chart = null;

// ── DOM Ready ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
    await checkAuth();
    initSliders();
    await loadPresets();
    await loadBaseline();
    bindEvents();
});

// ── Slider Init ────────────────────────────────────────────
function initSliders() {
    const sliders = document.querySelectorAll(".sim-slider");
    sliders.forEach(slider => {
        const display = document.getElementById(`${slider.id}-value`);
        if (display) {
            display.textContent = slider.value + (slider.dataset.unit || "");
        }
        slider.addEventListener("input", () => {
            if (display) display.textContent = slider.value + (slider.dataset.unit || "");
            debouncedRunSimulation();
        });
    });
}

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}
const debouncedRunSimulation = debounce(() => runSimulation("Điều chỉnh thủ công"), 400);

// ── Load Presets ───────────────────────────────────────────
async function loadPresets() {
    try {
        const res = await secureFetch(`${SIM_API}/presets`);
        if (!res.ok) return;
        const presets = await res.json();
        const container = document.getElementById("sim-presets");
        if (!container) return;

        container.innerHTML = presets.map(p => `
            <button type="button" class="sim-preset-btn group relative px-4 py-3 rounded-xl border border-slate-200 bg-white hover:border-sky-300 hover:shadow-lg transition-all text-left"
                    data-preset='${JSON.stringify(p.parameters)}' data-name="${p.name}">
                <div class="text-xs font-black text-primary-container tracking-tight">${p.name}</div>
                <div class="text-[10px] text-slate-500 mt-1 leading-relaxed">${p.description}</div>
                <span class="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span class="material-symbols-outlined text-sky-500 text-[16px]">play_arrow</span>
                </span>
            </button>
        `).join("");

        container.querySelectorAll(".sim-preset-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const params = JSON.parse(btn.dataset.preset);
                applyPresetToSliders(params);
                runSimulation(btn.dataset.name);
            });
        });
    } catch (e) {
        console.error("[Simulation] Failed to load presets:", e);
    }
}

function applyPresetToSliders(params) {
    const map = {
        "sim-vat-rate": "vat_rate",
        "sim-cit-rate": "cit_rate",
        "sim-audit-coverage": "audit_coverage_pct",
        "sim-penalty-multiplier": "penalty_multiplier",
        "sim-interest-rate": "interest_rate",
        "sim-gdp-growth": "economic_growth_pct",
    };
    for (const [sliderId, paramKey] of Object.entries(map)) {
        const slider = document.getElementById(sliderId);
        if (slider && params[paramKey] !== undefined) {
            slider.value = params[paramKey];
            const display = document.getElementById(`${sliderId}-value`);
            if (display) display.textContent = params[paramKey] + (slider.dataset.unit || "");
        }
    }
}

// ── Load Baseline ──────────────────────────────────────────
async function loadBaseline() {
    try {
        const res = await secureFetch(`${SIM_API}/baseline`);
        if (!res.ok) return;
        baselineData = await res.json();
        renderKPIs(baselineData, null);
    } catch (e) {
        console.error("[Simulation] Failed to load baseline:", e);
    }
}

// ── Gather Params from Sliders ─────────────────────────────
function gatherParams() {
    return {
        vat_rate: parseFloat(document.getElementById("sim-vat-rate")?.value || 10),
        cit_rate: parseFloat(document.getElementById("sim-cit-rate")?.value || 20),
        audit_coverage_pct: parseFloat(document.getElementById("sim-audit-coverage")?.value || 5),
        penalty_multiplier: parseFloat(document.getElementById("sim-penalty-multiplier")?.value || 1),
        interest_rate: parseFloat(document.getElementById("sim-interest-rate")?.value || 6),
        economic_growth_pct: parseFloat(document.getElementById("sim-gdp-growth")?.value || 6.5),
    };
}

// ── Run Simulation ─────────────────────────────────────────
async function runSimulation(name) {
    const params = gatherParams();
    const btn = document.getElementById("sim-run-btn");
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="material-symbols-outlined text-[16px] animate-spin">autorenew</span> Đang mô phỏng...';
    }

    try {
        const res = await secureFetch(`${SIM_API}/run-scenario?name=${encodeURIComponent(name || "Custom")}`, {
            method: "POST",
            body: JSON.stringify(params),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        currentScenario = await res.json();
        renderKPIs(baselineData, currentScenario);
        renderChart(currentScenario);
        renderIndustryTable(currentScenario);
        renderRiskDist(currentScenario);
    } catch (e) {
        console.error("[Simulation] Error:", e);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<span class="material-symbols-outlined text-[16px]">play_arrow</span> Chạy Mô phỏng';
        }
    }
}

// ── Render KPIs ────────────────────────────────────────────
function renderKPIs(baseline, scenario) {
    const kpi1 = document.querySelector('[data-kpi="sim-high-risk"]');
    const kpi2 = document.querySelector('[data-kpi="sim-loss"]');
    const kpi3 = document.querySelector('[data-kpi="sim-revenue"]');
    const kpi4 = document.querySelector('[data-kpi="sim-delinq-rate"]');

    if (baseline && !scenario) {
        if (kpi1) kpi1.innerHTML = `${baseline.baseline_high_risk_count.toLocaleString()}`;
        if (kpi2) kpi2.innerHTML = `${formatCurrencyCode(baseline.baseline_estimated_loss)} <span class="text-lg">VNĐ</span>`;
        if (kpi3) kpi3.innerHTML = `${formatCurrencyCode(baseline.baseline_total_revenue)} <span class="text-lg">VNĐ</span>`;
        if (kpi4) kpi4.innerHTML = `${baseline.baseline_delinquency_rate}%`;
    }

    if (scenario) {
        const deltaClass = (v) => v > 0 ? "text-rose-500" : v < 0 ? "text-emerald-500" : "text-slate-500";
        const deltaIcon = (v) => v > 0 ? "arrow_upward" : v < 0 ? "arrow_downward" : "remove";
        const deltaSign = (v) => v > 0 ? "+" : "";

        if (kpi1) kpi1.innerHTML = `${scenario.simulated_high_risk_count.toLocaleString()} <span class="text-sm ${deltaClass(scenario.delta_high_risk)}"><span class="material-symbols-outlined text-[14px]">${deltaIcon(scenario.delta_high_risk)}</span>${deltaSign(scenario.delta_high_risk)}${scenario.delta_high_risk}</span>`;
        if (kpi2) kpi2.innerHTML = `${formatCurrencyCode(scenario.simulated_estimated_loss)} <span class="text-sm ${deltaClass(scenario.delta_estimated_loss)}">${deltaSign(scenario.delta_estimated_loss)}${formatCurrencyCode(Math.abs(scenario.delta_estimated_loss))}</span>`;
        if (kpi3) kpi3.innerHTML = `${formatCurrencyCode(scenario.simulated_total_revenue)} <span class="text-sm ${deltaClass(-scenario.delta_revenue_pct)}">${deltaSign(scenario.delta_revenue_pct)}${scenario.delta_revenue_pct}%</span>`;
        if (kpi4) kpi4.innerHTML = `${scenario.simulated_delinquency_rate}% <span class="text-sm ${deltaClass(scenario.simulated_delinquency_rate - (baseline?.baseline_delinquency_rate || 0))}">${deltaSign(scenario.simulated_delinquency_rate - (baseline?.baseline_delinquency_rate || 0))}${(scenario.simulated_delinquency_rate - (baseline?.baseline_delinquency_rate || 0)).toFixed(2)}pp</span>`;
    }
}

// ── Render Chart (Canvas-based) ────────────────────────────
function renderChart(scenario) {
    const canvas = document.getElementById("sim-chart");
    if (!canvas || !scenario?.quarterly_projection?.length) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 300 * dpr;
    canvas.style.width = rect.width + "px";
    canvas.style.height = "300px";
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = 300;
    const pad = { top: 30, right: 30, bottom: 50, left: 70 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const data = scenario.quarterly_projection;
    const allVals = data.flatMap(d => [d.baseline_value, d.simulated_value]);
    const minV = Math.min(...allVals) * 0.95;
    const maxV = Math.max(...allVals) * 1.05;

    const xScale = (i) => pad.left + (i / (data.length - 1)) * plotW;
    const yScale = (v) => pad.top + plotH - ((v - minV) / (maxV - minV)) * plotH;

    // Background
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (i / 5) * plotH;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();
        const val = maxV - (i / 5) * (maxV - minV);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(val.toFixed(0) + " tỷ", pad.left - 8, y + 3);
    }

    // X labels
    ctx.textAlign = "center";
    data.forEach((d, i) => {
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px Inter, sans-serif";
        ctx.fillText(d.quarter, xScale(i), H - pad.bottom + 20);
    });

    // Baseline line
    ctx.beginPath();
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    data.forEach((d, i) => {
        if (i === 0) ctx.moveTo(xScale(i), yScale(d.baseline_value));
        else ctx.lineTo(xScale(i), yScale(d.baseline_value));
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Simulated line
    ctx.beginPath();
    const grad = ctx.createLinearGradient(pad.left, 0, W - pad.right, 0);
    grad.addColorStop(0, "#0284c7");
    grad.addColorStop(1, "#7c3aed");
    ctx.strokeStyle = grad;
    ctx.lineWidth = 3;
    data.forEach((d, i) => {
        if (i === 0) ctx.moveTo(xScale(i), yScale(d.simulated_value));
        else ctx.lineTo(xScale(i), yScale(d.simulated_value));
    });
    ctx.stroke();

    // Fill under simulated
    ctx.beginPath();
    data.forEach((d, i) => {
        if (i === 0) ctx.moveTo(xScale(i), yScale(d.simulated_value));
        else ctx.lineTo(xScale(i), yScale(d.simulated_value));
    });
    ctx.lineTo(xScale(data.length - 1), pad.top + plotH);
    ctx.lineTo(xScale(0), pad.top + plotH);
    ctx.closePath();
    const fillGrad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    fillGrad.addColorStop(0, "rgba(2, 132, 199, 0.15)");
    fillGrad.addColorStop(1, "rgba(2, 132, 199, 0.02)");
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Dots
    data.forEach((d, i) => {
        ctx.beginPath();
        ctx.arc(xScale(i), yScale(d.simulated_value), 4, 0, Math.PI * 2);
        ctx.fillStyle = "#0284c7";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // Legend
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "left";
    ctx.fillText("--- Baseline", pad.left, pad.top - 10);
    ctx.fillStyle = "#0284c7";
    ctx.fillText("━━ Mô phỏng", pad.left + 100, pad.top - 10);
}

// ── Industry Impact Table ──────────────────────────────────
function renderIndustryTable(scenario) {
    const tbody = document.getElementById("sim-industry-tbody");
    if (!tbody) return;

    tbody.innerHTML = scenario.industry_impacts.map(imp => {
        const color = imp.delta_pct > 0 ? "text-rose-600" : imp.delta_pct < 0 ? "text-emerald-600" : "text-slate-500";
        const sign = imp.delta_pct > 0 ? "+" : "";
        return `
            <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                <td class="px-4 py-3 text-xs font-bold text-primary-container">${imp.industry}</td>
                <td class="px-4 py-3 text-xs text-slate-600">${imp.company_count.toLocaleString()}</td>
                <td class="px-4 py-3 text-xs text-slate-600">${imp.baseline_delinquency_rate}%</td>
                <td class="px-4 py-3 text-xs font-bold ${color}">${imp.simulated_delinquency_rate}%</td>
                <td class="px-4 py-3 text-xs font-black ${color}">${sign}${imp.delta_pct}pp</td>
                <td class="px-4 py-3 text-xs ${imp.estimated_revenue_change >= 0 ? 'text-emerald-600' : 'text-rose-600'}">${imp.estimated_revenue_change >= 0 ? '+' : ''}${formatCurrencyCode(Math.abs(imp.estimated_revenue_change))}</td>
            </tr>`;
    }).join("");
}

// ── Risk Distribution (Donut Chart) ──────────────────────
function renderRiskDist(scenario) {
    const container = document.getElementById("sim-risk-dist");
    const canvas = document.getElementById("sim-risk-chart");
    if (!container || !canvas || !scenario?.risk_distribution) return;

    const dist = scenario.risk_distribution;
    const total = Object.values(dist).reduce((a, b) => a + b, 0);
    const colors = { low: "#10b981", medium: "#f59e0b", high: "#f97316", critical: "#ef4444" };
    const labels = { low: "Thấp", medium: "Trung bình", high: "Cao", critical: "Nghiêm trọng" };

    // Set up canvas
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = Math.min(250, rect.height) / 2;
    const radius = cy * 0.75;
    
    canvas.width = rect.width * dpr;
    canvas.height = 250 * dpr;
    canvas.style.width = rect.width + "px";
    canvas.style.height = "250px";
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, 250);

    // Draw Donut
    let startAngle = -Math.PI / 2;
    for (const [key, count] of Object.entries(dist)) {
        if (count === 0) continue;
        const sliceAngle = (count / total) * 2 * Math.PI;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
        ctx.lineWidth = radius * 0.4;
        ctx.strokeStyle = colors[key];
        ctx.stroke();
        
        // Add gap
        startAngle += sliceAngle;
    }
    
    // Draw center text
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "bold 24px Inter, sans-serif";
    ctx.fillStyle = "#002147";
    ctx.fillText(total.toLocaleString(), cx, cy - 8);
    ctx.font = "10px Inter, sans-serif";
    ctx.fillStyle = "#64748b";
    ctx.fillText("Tổng DN", cx, cy + 12);

    // Render legend
    container.innerHTML = Object.entries(dist).map(([key, count]) => {
        const pct = (count / Math.max(1, total) * 100).toFixed(1);
        return `
            <div class="flex items-center gap-3 bg-slate-50 rounded-lg p-2 transition-transform duration-300 hover:scale-105 fade-in">
                <div class="w-2 h-8 rounded-full" style="background:${colors[key]}"></div>
                <div class="flex flex-col">
                    <span class="text-[10px] font-bold text-slate-500 uppercase">${labels[key]}</span>
                    <span class="text-sm font-black text-slate-800">${count.toLocaleString()} <span class="text-xs font-normal text-slate-400">(${pct}%)</span></span>
                </div>
            </div>`;
    }).join("");
}

// ── Events ─────────────────────────────────────────────────
function bindEvents() {
    const runBtn = document.getElementById("sim-run-btn");
    if (runBtn) runBtn.addEventListener("click", () => runSimulation("Custom Scenario"));

    const resetBtn = document.getElementById("sim-reset-btn");
    if (resetBtn) resetBtn.addEventListener("click", () => {
        applyPresetToSliders({
            vat_rate: 10, cit_rate: 20, audit_coverage_pct: 5,
            penalty_multiplier: 1, interest_rate: 6, economic_growth_pct: 6.5,
        });
        if (baselineData) {
            renderKPIs(baselineData, null);
            // Re-run simulation with default values
            runSimulation("Baseline");
        }
    });

    window.addEventListener("resize", () => {
        if (currentScenario || baselineData) {
            renderChart(currentScenario);
            renderRiskDist(currentScenario);
        }
    });
}
